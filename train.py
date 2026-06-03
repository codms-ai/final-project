import os
import sys
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

# =============================================================================
# Model Architecture (self-contained)
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
# Loss Functions (self-contained)
# =============================================================================

def huber_loss(pred, target, delta=3.0):
    err = pred - target
    return torch.where(err.abs() <= delta,
                       0.5 * err.pow(2),
                       delta * (err.abs() - 0.5 * delta)).mean()


def pinball_loss(p_hat, d_hat, p_bs, tau=0.15):
    B      = p_hat.size(0)
    diff   = p_hat.unsqueeze(-1) - p_bs.unsqueeze(0).expand(B, -1, -1)
    d_pred = (diff.pow(2).sum(1) + 1e-6).sqrt()
    u      = d_hat - d_pred
    return torch.where(u >= 0, tau * u, (tau - 1) * u).mean()


def ineq_loss(p_hat, d_hat, p_bs):
    B      = p_hat.size(0)
    diff   = p_hat.unsqueeze(-1) - p_bs.unsqueeze(0).expand(B, -1, -1)
    d_pred = (diff.pow(2).sum(1) + 1e-6).sqrt()
    return F.relu(d_pred - d_hat).pow(2).mean()


def gate_loss(c_i, target=0.5):
    return (c_i.mean() - target).pow(2)


def compute_loss(p_hat, p0, c_i, d_hat, p_bs, p_gt,
                 alpha=0.3, beta=0.1, gamma=0.05, delta_l=0.01):
    return (huber_loss(p_hat, p_gt)
            + alpha   * huber_loss(p0, p_gt)
            + beta    * pinball_loss(p_hat, d_hat, p_bs)
            + gamma   * ineq_loss(p_hat, d_hat, p_bs)
            + delta_l * gate_loss(c_i))


# =============================================================================
# Data (self-contained)
# =============================================================================

def load_mat(path):
    import scipy.io as sio
    data  = sio.loadmat(path, squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p     = np.asarray(data['p'],     dtype=float)
    return p_bs, d_hat, p


class LocalizationDataset(Dataset):
    def __init__(self, d_hat, p):
        self.d_hat = torch.tensor(d_hat.T, dtype=torch.float32)
        self.p     = torch.tensor(p.T,     dtype=torch.float32)

    def __len__(self):
        return len(self.d_hat)

    def __getitem__(self, idx):
        return self.d_hat[idx], self.p[idx]


def generate_synthetic(n, p_bs_t, device):
    with torch.no_grad():
        pos = torch.stack([
            torch.empty(n, device=device).uniform_(-60, 60),
            torch.empty(n, device=device).uniform_(-30, 30),
        ], dim=1)
        diff   = pos.unsqueeze(-1) - p_bs_t.unsqueeze(0)
        d_true = (diff.pow(2).sum(1) + 1e-6).sqrt()
        nlos   = torch.bernoulli(torch.full((n, 18), 0.62, device=device))
        b_nlos = torch.exp(2.302 + 0.9704 * torch.randn(n, 18, device=device)).clamp(0, 200)
        b_los  = torch.randn(n, 18, device=device)
        b      = nlos * b_nlos + (1 - nlos) * b_los
        d_hat  = (d_true + b).clamp(min=d_true * 0.5)
    return d_hat, pos


# =============================================================================
# Training Configuration
# =============================================================================

MAT_PATH  = 'InF_DH_FR1.mat'
SAVE_PATH = 'model.pt'

_CLI_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else None

HYPERPARAMS = dict(in_features=6, d_model=256, num_heads=8,
                   num_inducing=32, num_isab=3, gn_steps=8, refine=True)

CFG = dict(
    batch_size   = 64,
    n_epochs     = 600,
    lr           = 5e-4,
    weight_decay = 1e-4,
    aug_ratio    = 0.5,
    n_drop_min   = 2,
    n_drop_max   = 4,
    patience     = 50,
    seed         = 42,
    k_folds      = 5,
)


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_bs_mask(B, n_bs, n_drop, device):
    idx  = torch.rand(B, n_bs, device=device).argsort(dim=1)[:, :n_drop]
    mask = torch.zeros(B, n_bs, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def eval_rmse(model, loader, p_bs_t, device):
    sq_errors = []
    with torch.no_grad():
        for d_v, p_v in loader:
            p_hat, *_ = model(d_v.to(device, non_blocking=True), p_bs_t)
            sq_errors.append((p_hat.cpu() - p_v).pow(2).sum(1))
    return torch.cat(sq_errors).mean().sqrt().item()


def run_training(train_loader, val_loader, p_bs_t, device, tag=''):
    use_amp   = device.type == 'cuda'
    model     = LocalizationModel(**HYPERPARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['n_epochs'])
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_rmse, no_improve, best_epoch = float('inf'), 0, 0
    best_state = None

    for epoch in range(1, CFG['n_epochs'] + 1):
        model.train()
        total_loss = 0.0

        for d_real, p_real in train_loader:
            d_real = d_real.to(device, non_blocking=True)
            p_real = p_real.to(device, non_blocking=True)
            B      = d_real.size(0)
            n_syn  = int(B * CFG['aug_ratio'])

            if n_syn > 0:
                d_syn, p_syn = generate_synthetic(n_syn, p_bs_t, device)
                d_batch = torch.cat([d_real[:B - n_syn], d_syn])
                p_batch = torch.cat([p_real[:B - n_syn], p_syn])
            else:
                d_batch, p_batch = d_real, p_real

            n_drop  = random.randint(CFG['n_drop_min'], CFG['n_drop_max'])
            bs_mask = make_bs_mask(d_batch.size(0), 18, n_drop, device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(d_batch, p_bs_t, bs_mask=bs_mask)
                p_hat, p0, c_i, _, p1 = out
                loss = compute_loss(p_hat, p0, c_i, d_batch, p_bs_t, p_batch)
                if p1 is not None:
                    loss = loss + 0.15 * compute_loss(p1, p0, c_i, d_batch, p_bs_t, p_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_rmse = eval_rmse(model, val_loader, p_bs_t, device)
            improved = val_rmse < best_rmse
            print(f'{tag}Epoch {epoch:4d} | loss {avg_loss:.4f} | val RMSE {val_rmse:.3f} m'
                  + (' ← best' if improved else ''))
            if improved:
                best_rmse  = val_rmse
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= CFG['patience']:
                    print(f'{tag}Early stopping at epoch {epoch}.')
                    break
        else:
            print(f'{tag}Epoch {epoch:4d} | loss {avg_loss:.4f}')
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    return best_state, best_epoch, best_rmse


# =============================================================================
# Main Training Entry Point
# =============================================================================

def train():
    seed   = _CLI_SEED if _CLI_SEED is not None else CFG['seed']
    suffix = f'_s{seed}' if seed != CFG['seed'] else ''
    set_seed(seed)

    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f'{MAT_PATH} not found.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  seed={seed}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    p_bs, d_hat, p = load_mat(MAT_PATH)
    p_bs_t  = torch.tensor(p_bs, dtype=torch.float32, device=device)
    dataset = LocalizationDataset(d_hat, p)
    N       = len(dataset)
    print(f'Total samples: {N}')

    kf         = KFold(n_splits=CFG['k_folds'], shuffle=True, random_state=CFG['seed'])
    fold_rmses = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(N))):
        print(f'\n{"="*60}')
        print(f' Fold {fold+1}/{CFG["k_folds"]}  |  train={len(train_idx)}  val={len(val_idx)}')
        print(f'{"="*60}')

        pin          = device.type == 'cuda'
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=CFG['batch_size'], shuffle=True,
                                  drop_last=True, pin_memory=pin)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  batch_size=256, shuffle=False, pin_memory=pin)

        fold_state, best_epoch, best_rmse = run_training(
            train_loader, val_loader, p_bs_t, device, tag=f'[F{fold+1}] ')
        fold_rmses.append(best_rmse)
        best_epochs.append(best_epoch)
        torch.save({'state_dict': fold_state, 'hyperparams': HYPERPARAMS,
                    'val_rmse': best_rmse},
                   f'model_fold{fold+1}{suffix}.pt')
        print(f'Fold {fold+1} best → RMSE {best_rmse:.3f} m  (epoch {best_epoch})')

    mean_rmse = np.mean(fold_rmses)
    std_rmse  = np.std(fold_rmses)
    avg_epoch = int(np.mean(best_epochs) * 1.1)
    print(f'\n{"="*60}')
    print(f' CV Result: {mean_rmse:.3f} ± {std_rmse:.3f} m')
    print(f' Full retrain target: {avg_epoch} epochs')
    print(f'{"="*60}')

    full_cfg_backup = CFG['n_epochs']
    CFG['n_epochs'] = avg_epoch
    full_loader = DataLoader(dataset, batch_size=CFG['batch_size'],
                             shuffle=True, drop_last=True,
                             pin_memory=(device.type == 'cuda'))
    final_state, _, _ = run_training(full_loader, None, p_bs_t, device, tag='[Full] ')
    CFG['n_epochs'] = full_cfg_backup

    save_path = f'model{suffix}.pt' if suffix else SAVE_PATH
    torch.save({'state_dict': final_state,
                'cv_rmse_mean': mean_rmse, 'cv_rmse_std': std_rmse,
                'cv_fold_rmses': fold_rmses, 'hyperparams': HYPERPARAMS},
               save_path)
    print(f'\nSaved → {save_path}')
    print(f'CV RMSE: {mean_rmse:.3f} ± {std_rmse:.3f} m')


if __name__ == '__main__':
    train()
