"""Training loss: KL (free bits) + node-vector reconstruction (MSE, cosine, norm) + FAPE + Kabsch RMSD."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TrainConfig


def _check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in loss term '{name}'")


def _masked_mean(tensor, mask, dim=None, eps=1e-6):
    mask_float = mask.float()
    if tensor.dim() > mask_float.dim() and tensor.shape[:2] == mask_float.shape:
        mask_float = mask_float.unsqueeze(-1)
    keep = dim is not None
    num_valid = mask_float.sum(dim=dim, keepdim=keep).clamp(min=eps)
    masked_sum = (tensor * mask_float).sum(dim=dim, keepdim=keep)
    return masked_sum / num_valid


def ankh_cos(pred, true, mask, eps=1e-8):
    cos_sim = F.cosine_similarity(pred, true, dim=-1, eps=eps)
    return _masked_mean(1.0 - cos_sim, mask)


def ankh_norm(pred, true, mask):
    norm_diff = torch.abs(torch.norm(true, p=2, dim=-1) - torch.norm(pred, p=2, dim=-1))
    return _masked_mean(norm_diff, mask)


def kabsch_rmsd(coords_pred, coords_true, mask, eps=1e-8):
    """RMSD after optimal superposition of the predicted onto the true C-alpha positions (batched, masked)."""
    mask_float = mask.float().unsqueeze(-1)
    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=eps)
    center_true = (coords_true * mask_float).sum(dim=1, keepdim=True) / num_valid
    center_pred = (coords_pred * mask_float).sum(dim=1, keepdim=True) / num_valid
    coords_true_c = (coords_true - center_true) * mask_float
    coords_pred_c = (coords_pred - center_pred) * mask_float

    H = torch.matmul(coords_pred_c.transpose(1, 2), coords_true_c)
    try:
        U, S, Vt = torch.linalg.svd(H)
    except torch.linalg.LinAlgError as e:  # pragma: no cover - numerical corner case
        print(f"[lenseek.loss] SVD failed in Kabsch ({e}); RMSD term set to 0 for this batch")
        return torch.tensor(0.0, device=coords_pred.device, dtype=coords_pred.dtype)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    S_diag = torch.diag_embed(torch.ones_like(S))
    S_diag[:, -1, -1] = d
    R = torch.matmul(Vt.transpose(1, 2), torch.matmul(S_diag, U.transpose(1, 2)))
    # R rotates column vectors (q = R p); the rows of coords_pred_c are points, so apply R^T.
    coords_pred_aligned = torch.matmul(coords_pred_c, R.transpose(1, 2))
    sq_diff = (coords_pred_aligned - coords_true_c).pow(2).sum(dim=-1)
    mse = sq_diff[mask].sum() / mask_float.sum().clamp(min=eps)
    return torch.sqrt(mse + eps)


def fape_loss(pred_T, pred_R, true_T, true_R, mask, eps=1e-8, clamp_dist=10.0):
    """Frame-aligned point error of C-alpha positions, clamped at ``clamp_dist`` Angstrom."""
    B, N, _ = pred_T.shape
    T_true_i = true_T.unsqueeze(2).expand(B, N, N, 3)
    T_true_j = true_T.unsqueeze(1).expand(B, N, N, 3)
    T_pred_i = pred_T.unsqueeze(2).expand(B, N, N, 3)
    T_pred_j = pred_T.unsqueeze(1).expand(B, N, N, 3)
    R_true_i_T = true_R.transpose(-1, -2).unsqueeze(2).expand(B, N, N, 3, 3)
    R_pred_i_T = pred_R.transpose(-1, -2).unsqueeze(2).expand(B, N, N, 3, 3)
    T_true_local = torch.matmul(R_true_i_T, (T_true_j - T_true_i).unsqueeze(-1)).squeeze(-1)
    T_pred_local = torch.matmul(R_pred_i_T, (T_pred_j - T_pred_i).unsqueeze(-1)).squeeze(-1)
    dist_error = torch.norm(T_true_local - T_pred_local, dim=-1, p=2)
    clamped = torch.clamp(dist_error, max=clamp_dist)
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    num_valid_j = mask.float().sum(dim=1, keepdim=True).clamp(min=eps)
    fape_per_i = (clamped * mask_2d.float()).sum(dim=2) / num_valid_j
    return _masked_mean(fape_per_i, mask)


def vae_loss(h_true, t_true, r_true, h_pred, t_pred, r_pred, mu, logvar, mask, kl_beta, cfg: TrainConfig):
    """All loss terms (dict) and their weighted sum ``total_loss``."""
    losses = {}
    for t, name in ((mu, "mu"), (logvar, "logvar"), (h_pred, "h_pred"), (t_pred, "t_pred"), (r_pred, "r_pred")):
        _check_nan(t, name)

    mu_c = torch.clamp(mu, min=-20, max=20)
    logvar_c = torch.clamp(logvar, min=-20, max=20)
    kl_per_node = (-0.5 * (1 + logvar_c - mu_c.pow(2) - logvar_c.exp())).sum(dim=-1)
    kl_per_node_clamped = torch.clamp(kl_per_node, min=cfg.free_bits)
    losses["kl_loss"] = _masked_mean(kl_per_node_clamped, mask)
    losses["kl_loss_raw"] = _masked_mean(kl_per_node, mask)
    _check_nan(losses["kl_loss"], "kl_loss")

    mse_per_res = F.mse_loss(h_pred, h_true, reduction="none").mean(dim=-1)
    losses["recon_ankh_mse"] = _masked_mean(mse_per_res, mask)
    losses["recon_ankh_cos"] = ankh_cos(h_pred, h_true, mask)
    losses["recon_ankh_norm"] = ankh_norm(h_pred, h_true, mask)
    losses["recon_fape"] = fape_loss(t_pred, r_pred, t_true, r_true, mask)
    losses["recon_rmsd"] = kabsch_rmsd(t_pred, t_true, mask)

    total = 0.0
    total += losses["kl_loss"] * kl_beta * cfg.w_kl
    total += losses["recon_ankh_mse"] * cfg.w_mse
    total += losses["recon_ankh_cos"] * cfg.w_cos
    total += losses["recon_ankh_norm"] * cfg.w_norm
    total += losses["recon_fape"] * cfg.w_fape
    total += losses["recon_rmsd"] * cfg.w_rmsd
    losses["total_loss"] = total
    _check_nan(losses["total_loss"], "total_loss")
    return losses
