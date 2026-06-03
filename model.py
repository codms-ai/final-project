import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, X, Y):
        out, _ = self.attn(X, Y, Y)
        X = self.norm1(X + out)
        X = self.norm2(X + self.ffn(X))
        return X


class ISAB(nn.Module):
    """Induced Set Attention Block — Lee et al., ICML 2019."""
    def __init__(self, d_model, num_heads, m):
        super().__init__()
        self.I    = nn.Parameter(torch.randn(1, m, d_model))
        self.mab1 = MAB(d_model, num_heads)
        self.mab2 = MAB(d_model, num_heads)

    def forward(self, X):
        H = self.mab1(self.I.expand(X.size(0), -1, -1), X)
        return self.mab2(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention — Lee et al., ICML 2019."""
    def __init__(self, d_model, num_heads, k=1):
        super().__init__()
        self.S   = nn.Parameter(torch.randn(1, k, d_model))
        self.mab = MAB(d_model, num_heads)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)


class LocalizationModel(nn.Module):
    """
    Permutation-invariant Set Transformer encoder +
    differentiable unrolled Gauss-Newton refinement (K steps).

    refine=True: 2-round iterative refinement.
      Round 1 — static tokens → GN → p1
      Round 2 — tokens + residual(p1) → GN → p_hat
    The residual (d_hat_i - dist(p1, BS_i)) encodes how consistent the
    current estimate is with each BS, giving the encoder a per-BS
    "NLOS suspicion" signal without CIR access.
    """
    def __init__(self, in_features=6, d_model=128, num_heads=4,
                 num_inducing=16, num_isab=2, gn_steps=5, refine=True):
        super().__init__()
        self.gn_steps = gn_steps
        self.refine   = refine

        # Round-1 token encoder
        self.phi = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Round-2 token encoder: same dim but +1 residual feature
        if refine:
            self.phi_refine = nn.Sequential(
                nn.Linear(in_features + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        # Shared ISAB stack + heads (used in both rounds)
        self.isabs           = nn.ModuleList(
            [ISAB(d_model, num_heads, num_inducing) for _ in range(num_isab)]
        )
        self.confidence_head = nn.Linear(d_model, 1)
        self.bias_head       = nn.Linear(d_model, 1)

        self.pma      = PMA(d_model, num_heads, k=1)
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.register_buffer('eye2', torch.eye(2))

    # ------------------------------------------------------------------
    def _tokenize(self, d_hat, p_bs):
        B   = d_hat.size(0)
        pbs = p_bs.unsqueeze(0).expand(B, -1, -1)
        bs_x       = pbs[:, 0, :] / 60.0
        bs_y       = pbs[:, 1, :] / 30.0
        d_norm     = d_hat / 100.0
        d_med      = d_hat.median(dim=1, keepdim=True).values
        d_min      = d_hat.min(dim=1, keepdim=True).values
        d_diff_med = (d_hat - d_med) / 100.0
        d_rank     = d_hat.argsort(dim=1).argsort(dim=1).float() / 18.0
        d_diff_min = (d_hat - d_min) / 100.0
        return torch.stack([bs_x, bs_y, d_norm, d_diff_med, d_rank, d_diff_min], dim=-1)

    def _encode(self, tokens, phi):
        H = phi(tokens)
        for isab in self.isabs:
            H = isab(H)
        return H

    def _gauss_newton(self, p, p_bs_b, d_corr, weights):
        lam = F.softplus(self.log_lambda) + 1e-4
        for _ in range(self.gn_steps):
            diff  = p.unsqueeze(-1) - p_bs_b
            dist  = (diff.pow(2).sum(1) + 1e-6).sqrt()
            r     = dist - d_corr
            J     = (diff / dist.unsqueeze(1)).permute(0, 2, 1)
            W     = weights.unsqueeze(-1)
            JW    = J * W
            H_mat = JW.permute(0, 2, 1).bmm(J) + lam * self.eye2
            g     = (JW * r.unsqueeze(-1)).sum(1)
            delta = torch.linalg.solve(H_mat, g.unsqueeze(-1)).squeeze(-1)
            p     = p - delta
        return p

    # ------------------------------------------------------------------
    def forward(self, d_hat, p_bs, bs_mask=None):
        """
        Returns: p_hat (B,2), p0 (B,2), c_i (B,18), b_hat (B,18), p1 (B,2)|None
          p0    — Set Transformer direct position (deep supervision target)
          p1    — round-1 GN output (intermediate supervision target, if refine)
          p_hat — final output (round-2 GN if refine, else round-1 GN)
        """
        B      = d_hat.size(0)
        p_bs_b = p_bs.unsqueeze(0).expand(B, -1, -1)
        tokens = self._tokenize(d_hat, p_bs)          # (B,18,6)

        # ── Round 1 ──────────────────────────────────────────────────────
        H1    = self._encode(tokens, self.phi)
        c1    = torch.sigmoid(self.confidence_head(H1).squeeze(-1))
        b1    = F.softplus(self.bias_head(H1).squeeze(-1))
        if bs_mask is not None:
            c1 = c1 * (~bs_mask).float()
        p0    = self.pos_head(self.pma(H1).squeeze(1))
        p1    = self._gauss_newton(p0, p_bs_b, d_hat - b1, c1)

        if not self.refine:
            return p1, p0, c1, b1, None

        # ── Round 2: residual feedback ────────────────────────────────────
        with torch.no_grad() if not self.training else torch.enable_grad():
            diff1 = p1.unsqueeze(-1) - p_bs_b          # (B,2,18)
            dist1 = (diff1.pow(2).sum(1) + 1e-6).sqrt()# (B,18)
        resid   = (d_hat - dist1) / 100.0              # (B,18) positive = NLOS excess

        tokens2 = torch.cat([tokens, resid.unsqueeze(-1)], dim=-1)  # (B,18,7)
        H2      = self._encode(tokens2, self.phi_refine)
        c2      = torch.sigmoid(self.confidence_head(H2).squeeze(-1))
        b2      = F.softplus(self.bias_head(H2).squeeze(-1))
        if bs_mask is not None:
            c2 = c2 * (~bs_mask).float()
        p_hat   = self._gauss_newton(p1, p_bs_b, d_hat - b2, c2)

        return p_hat, p0, c2, b2, p1
