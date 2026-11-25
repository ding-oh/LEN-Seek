"""Edge features and the C-alpha kNN graph of a binding site.

Edge (i, j) = [ RBF(d_ij) (32) | direction of j from i (3) | quaternion of R_i^T R_j (4) ]  -> 39 features.
All three groups are invariant to a rigid motion of the site when the direction is expressed in residue i's frame
(paper eq. 10). Residue frames R hold the frame axes as columns (lenseek.featurize.residue_frames).

kNN rule. Distances to padded residues and to the residue itself are +inf before the k nearest neighbours are taken.
A site with n <= k residues has only n - 1 finite neighbours per row, so the remaining picks are ties at +inf. On CUDA,
torch.topk fills ties in column order, so the diagonal (self) comes first and such sites get one self-edge per
residue; CPU torch.topk breaks the tie differently. Every model was trained on GPUs, so the released code ranks with a
stable sort (ties by column index) on every device, which reproduces the training graph exactly.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import RBF_COUNT, RBF_MAX, normalize_edge_drop

KNN_RULE = "stable-sort: ties by column index (= CUDA topk; self-edge when a site has <= k residues)"


def _rbf(D, D_min=0.0, D_max=RBF_MAX, D_count=RBF_COUNT, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view(1, 1, 1, -1)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D - D_mu) / D_sigma) ** 2)


def mat_to_quat(R, eps=1e-8):
    """Rotation matrices (..., 3, 3) -> unit quaternions (w, x, y, z)."""
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = R00 + R11 + R22
    q_w = torch.zeros_like(t)
    q_x = torch.zeros_like(t)
    q_y = torch.zeros_like(t)
    q_z = torch.zeros_like(t)

    m = t > 0
    if m.any():
        S = torch.sqrt(t[m] + 1.0) * 2
        q_w[m] = 0.25 * S
        q_x[m] = (R21[m] - R12[m]) / S
        q_y[m] = (R02[m] - R20[m]) / S
        q_z[m] = (R10[m] - R01[m]) / S
    m = (t <= 0) & (R00 > R11) & (R00 > R22)
    if m.any():
        S = torch.sqrt(1.0 + R00[m] - R11[m] - R22[m]) * 2
        q_w[m] = (R21[m] - R12[m]) / S
        q_x[m] = 0.25 * S
        q_y[m] = (R01[m] + R10[m]) / S
        q_z[m] = (R02[m] + R20[m]) / S
    m = (t <= 0) & (R11 > R00) & (R11 > R22)
    if m.any():
        S = torch.sqrt(1.0 + R11[m] - R00[m] - R22[m]) * 2
        q_w[m] = (R02[m] - R20[m]) / S
        q_x[m] = (R01[m] + R10[m]) / S
        q_y[m] = 0.25 * S
        q_z[m] = (R12[m] + R21[m]) / S
    m = (t <= 0) & (R22 > R00) & (R22 > R11)
    if m.any():
        S = torch.sqrt(1.0 + R22[m] - R00[m] - R11[m]) * 2
        q_w[m] = (R10[m] - R01[m]) / S
        q_x[m] = (R02[m] + R20[m]) / S
        q_y[m] = (R12[m] + R21[m]) / S
        q_z[m] = 0.25 * S
    q = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
    return F.normalize(q, dim=-1, eps=eps)


def knn_indices(D_masked: torch.Tensor, k: int) -> torch.Tensor:
    """Column indices of the k smallest entries of each row; equal values ranked by column index."""
    return torch.sort(D_masked, dim=-1, stable=True).indices[..., :k]


def get_edges(T: torch.Tensor, R: torch.Tensor, mask_2d: torch.Tensor, *, k: int = 15,
              edge_drop: Sequence[str] = (), fullgraph: bool = False, direction_frame: str = "local",
              eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Edge features (B, N, N, 39) and the encoder's edge mask (B, N, N).

    T: (B, N, 3) C-alpha positions (centred per site); R: (B, N, 3, 3) residue frames (axes as columns);
    mask_2d: (B, N, N) bool, valid residue pairs. Features outside the edge mask are zero.
    """
    edge_drop = normalize_edge_drop(edge_drop)
    B, N, _ = T.shape
    device = T.device

    D_ij = torch.cdist(T, T, p=2)
    D_ij_masked = D_ij.clone()
    D_ij_masked.masked_fill_(~mask_2d, float("inf"))
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    D_ij_masked[diag_mask] = float("inf")

    k_safe = min(k, N - 1)
    if k_safe <= 0:
        knn_mask = torch.zeros_like(mask_2d)
    else:
        top_k_indices = knn_indices(D_ij_masked, k_safe)
        knn_mask = torch.zeros_like(mask_2d)
        knn_mask.scatter_(dim=-1, index=top_k_indices, value=True)
    knn_mask = knn_mask | knn_mask.transpose(-1, -2)

    if fullgraph:
        final_edge_mask = mask_2d.clone()  # every valid residue pair, as in the decoder
    else:
        final_edge_mask = mask_2d & knn_mask
    final_edge_mask_3d = final_edge_mask.unsqueeze(-1)

    rbf_feats = _rbf(D_ij.unsqueeze(-1), device=device)

    T_i = T.unsqueeze(2).expand(-1, -1, N, -1)
    T_j = T.unsqueeze(1).expand(-1, N, -1, -1)
    D_vec = T_j - T_i
    if direction_frame == "local":
        # R_i^T (x_j - x_i): R holds the frame axes as columns, so component a is sum_c R[c, a] * v[c].
        D_vec = torch.einsum("bnca,bnmc->bnma", R, D_vec)
    elif direction_frame != "global":
        raise ValueError(f"direction_frame must be 'local' or 'global', got {direction_frame!r}")
    dir_feats = F.normalize(D_vec, dim=-1, eps=eps)

    R_i_T = R.transpose(-1, -2).unsqueeze(2).expand(-1, -1, N, -1, -1)
    R_j = R.unsqueeze(1).expand(-1, N, -1, -1, -1)
    quat_feats = mat_to_quat(torch.matmul(R_i_T, R_j))

    if "rbf" in edge_drop:
        rbf_feats = torch.zeros_like(rbf_feats)
    if "dir" in edge_drop:
        dir_feats = torch.zeros_like(dir_feats)
    if "quat" in edge_drop:
        quat_feats = torch.zeros_like(quat_feats)

    e_batch = torch.cat([rbf_feats, dir_feats, quat_feats], dim=-1)
    return e_batch * final_edge_mask_3d, final_edge_mask


def edges_for(cfg, T, R, mask_2d):
    """get_edges with the graph settings of a ModelConfig."""
    return get_edges(T, R, mask_2d, k=cfg.k_neighbors, edge_drop=cfg.edge_drop, fullgraph=cfg.fullgraph,
                     direction_frame=cfg.direction_frame)
