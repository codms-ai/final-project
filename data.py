import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def load_mat(path):
    """Load .mat with robust variable-name handling (p_bs / BS_positions)."""
    data  = sio.loadmat(path, squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs',  data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p     = np.asarray(data['p'],     dtype=float)
    return p_bs, d_hat, p     # (2,18), (18,N), (2,N)


class LocalizationDataset(Dataset):
    def __init__(self, d_hat, p):
        self.d_hat = torch.tensor(d_hat.T, dtype=torch.float32)  # (N,18)
        self.p     = torch.tensor(p.T,     dtype=torch.float32)  # (N,2)

    def __len__(self):
        return len(self.d_hat)

    def __getitem__(self, idx):
        return self.d_hat[idx], self.p[idx]


def generate_synthetic(n, p_bs_t, device):
    """
    Synthesise n users with NLOS bias matched to InF statistics
    (overall mean ≈ 16 m, std ≈ 20 m, ~62 % NLOS per BS).

    Log-normal NLOS parameters derived from target mean/std:
      sigma = sqrt(ln(1 + (20/16)^2)) ≈ 0.9704
      mu    = ln(16) - sigma^2/2      ≈ 2.302
    """
    with torch.no_grad():
        pos = torch.stack([
            torch.empty(n, device=device).uniform_(-60, 60),
            torch.empty(n, device=device).uniform_(-30, 30),
        ], dim=1)                                                    # (n,2)

        diff   = pos.unsqueeze(-1) - p_bs_t.unsqueeze(0)            # (n,2,18)
        d_true = (diff.pow(2).sum(1) + 1e-6).sqrt()                 # (n,18)

        nlos   = torch.bernoulli(torch.full((n, 18), 0.62, device=device))
        b_nlos = torch.exp(2.302 + 0.9704 * torch.randn(n, 18, device=device)).clamp(0, 200)
        b_los  = torch.randn(n, 18, device=device)                  # N(0,1) ≈ LOS noise

        b      = nlos * b_nlos + (1 - nlos) * b_los
        d_hat  = (d_true + b).clamp(min=d_true * 0.5)
    return d_hat, pos
