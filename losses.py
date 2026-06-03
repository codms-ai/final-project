import torch
import torch.nn.functional as F


def huber_loss(pred, target, delta=3.0):
    err = pred - target
    return torch.where(
        err.abs() <= delta,
        0.5 * err.pow(2),
        delta * (err.abs() - 0.5 * delta),
    ).mean()


def pinball_loss(p_hat, d_hat, p_bs, tau=0.15):
    """
    Asymmetric loss encoding the NLOS prior d_hat >= d_true.
    Penalises d_pred > d_hat (u < 0) roughly 6× harder than d_pred < d_hat.
    """
    B     = p_hat.size(0)
    diff  = p_hat.unsqueeze(-1) - p_bs.unsqueeze(0).expand(B, -1, -1)  # (B,2,18)
    d_pred = (diff.pow(2).sum(1) + 1e-6).sqrt()                         # (B,18)
    u     = d_hat - d_pred
    return torch.where(u >= 0, tau * u, (tau - 1) * u).mean()


def ineq_loss(p_hat, d_hat, p_bs):
    """Penalise ||p_hat - p_bs_i|| > d_hat_i (violates d_hat >= d_true)."""
    B     = p_hat.size(0)
    diff  = p_hat.unsqueeze(-1) - p_bs.unsqueeze(0).expand(B, -1, -1)
    d_pred = (diff.pow(2).sum(1) + 1e-6).sqrt()
    return F.relu(d_pred - d_hat).pow(2).mean()


def gate_loss(c_i, target=0.5):
    """Prevent gating values from collapsing to zero."""
    return (c_i.mean() - target).pow(2)


def compute_loss(p_hat, p0, c_i, d_hat, p_bs, p_gt,
                 alpha=0.3, beta=0.1, gamma=0.05, delta_l=0.01):
    return (
        huber_loss(p_hat, p_gt)
        + alpha   * huber_loss(p0, p_gt)
        + beta    * pinball_loss(p_hat, d_hat, p_bs)
        + gamma   * ineq_loss(p_hat, d_hat, p_bs)
        + delta_l * gate_loss(c_i)
    )
