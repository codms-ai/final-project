import os
import glob
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Model Architecture (self-contained — no external imports required)
# =============================================================================

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
    def __init__(self, d_model, num_heads, m):
        super().__init__()
        self.I    = nn.Parameter(torch.randn(1, m, d_model))
        self.mab1 = MAB(d_model, num_heads)
        self.mab2 = MAB(d_model, num_heads)

    def forward(self, X):
        H = self.mab1(self.I.expand(X.size(0), -1, -1), X)
        return self.mab2(X, H)


class PMA(nn.Module):
    def __init__(self, d_model, num_heads, k=1):
        super().__init__()
        self.S   = nn.Parameter(torch.randn(1, k, d_model))
        self.mab = MAB(d_model, num_heads)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)


class LocalizationModel(nn.Module):
    def __init__(self, in_features=6, d_model=128, num_heads=4,
                 num_inducing=16, num_isab=2, gn_steps=5, refine=True):
        super().__init__()
        self.gn_steps = gn_steps
        self.refine   = refine

        self.phi = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        if refine:
            self.phi_refine = nn.Sequential(
                nn.Linear(in_features + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        self.isabs           = nn.ModuleList(
            [ISAB(d_model, num_heads, num_inducing) for _ in range(num_isab)]
        )
        self.confidence_head = nn.Linear(d_model, 1)
        self.bias_head       = nn.Linear(d_model, 1)
        self.pma             = PMA(d_model, num_heads, k=1)
        self.pos_head        = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2),
        )
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.register_buffer('eye2', torch.eye(2))

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

    def forward(self, d_hat, p_bs, bs_mask=None):
        B      = d_hat.size(0)
        p_bs_b = p_bs.unsqueeze(0).expand(B, -1, -1)
        tokens = self._tokenize(d_hat, p_bs)

        H1  = self._encode(tokens, self.phi)
        c1  = torch.sigmoid(self.confidence_head(H1).squeeze(-1))
        b1  = F.softplus(self.bias_head(H1).squeeze(-1))
        if bs_mask is not None:
            c1 = c1 * (~bs_mask).float()
        p0  = self.pos_head(self.pma(H1).squeeze(1))
        p1  = self._gauss_newton(p0, p_bs_b, d_hat - b1, c1)

        if not self.refine:
            return p1, p0, c1, b1, None

        with torch.no_grad() if not self.training else torch.enable_grad():
            diff1 = p1.unsqueeze(-1) - p_bs_b
            dist1 = (diff1.pow(2).sum(1) + 1e-6).sqrt()
        resid   = (d_hat - dist1) / 100.0
        tokens2 = torch.cat([tokens, resid.unsqueeze(-1)], dim=-1)
        H2      = self._encode(tokens2, self.phi_refine)
        c2      = torch.sigmoid(self.confidence_head(H2).squeeze(-1))
        b2      = F.softplus(self.bias_head(H2).squeeze(-1))
        if bs_mask is not None:
            c2 = c2 * (~bs_mask).float()
        p_hat   = self._gauss_newton(p1, p_bs_b, d_hat - b2, c2)
        return p_hat, p0, c2, b2, p1


# =============================================================================
# Inference
# =============================================================================

_cache: np.ndarray | None = None
_call_idx: int = 0


def _load_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = LocalizationModel(**ckpt['hyperparams']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def _batch_infer_single(model, d_hat_np, p_bs_np, device, batch_size=512):
    p_bs_t = torch.tensor(p_bs_np, dtype=torch.float32, device=device)
    d_t    = torch.tensor(d_hat_np.T, dtype=torch.float32, device=device)
    chunks = []
    with torch.no_grad():
        for i in range(0, d_t.size(0), batch_size):
            p_hat, *_ = model(d_t[i:i + batch_size], p_bs_t)
            chunks.append(p_hat.cpu().numpy())
    return np.concatenate(chunks, axis=0).T   # (2,N)


def _ensemble_infer(d_hat_np, p_bs_np, device):
    found = sorted(set(
        glob.glob('model_fold*.pt') +
        glob.glob('model_s*.pt') +
        (['model.pt'] if os.path.exists('model.pt') else [])
    ))
    model_paths = found if found else ['model.pt']
    print(f'Ensemble: {len(model_paths)} models')
    preds = []
    for path in model_paths:
        m    = _load_model(path, device)
        pred = _batch_infer_single(m, d_hat_np, p_bs_np, device)
        preds.append(pred)
        del m
    return np.mean(preds, axis=0)


# =============================================================================
# Grader interface
# =============================================================================

def your_algorithm(d_hat_u, p_bs):
    """
    d_hat_u : (18,)   RTT distances for one user
    p_bs    : (2,18)  base station coordinates
    return  : (2,)    estimated [x, y]
    """
    global _cache, _call_idx
    if _cache is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return _ensemble_infer(d_hat_u.reshape(18, 1), p_bs, device)[:, 0]
    result     = _cache[:, _call_idx]
    _call_idx += 1
    return result


def main():
    global _cache, _call_idx

    mat_path = 'DH_FR1.mat' if os.path.exists('DH_FR1.mat') else 'InF_DH_FR1.mat'
    data     = sio.loadmat(mat_path, squeeze_me=False)
    p_bs     = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat    = np.asarray(data['d_hat'], dtype=float)
    num_user = d_hat.shape[1]

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _cache    = _ensemble_infer(d_hat, p_bs, device)
    _call_idx = 0

    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)
    return p_hat


if __name__ == '__main__':
    t0    = time.time()
    p_hat = main()
    print(f'shape  : {p_hat.shape}  dtype: {p_hat.dtype}')
    print(f'x range: [{p_hat[0].min():.1f}, {p_hat[0].max():.1f}]')
    print(f'y range: [{p_hat[1].min():.1f}, {p_hat[1].max():.1f}]')
    print(f'elapsed: {time.time() - t0:.1f} s')

    mat_path = 'DH_FR1.mat' if os.path.exists('DH_FR1.mat') else 'InF_DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    if 'p' in data:
        p_gt = np.asarray(data['p'], dtype=float)
        rmse = np.sqrt(np.mean(np.sum((p_hat - p_gt) ** 2, axis=0)))
        print(f'RMSE   : {rmse:.3f} m  (N={p_hat.shape[1]})')
