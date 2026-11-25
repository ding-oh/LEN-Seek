"""Encode binding sites into per-residue latent vectors (the posterior means mu).

The computation is the one that produced the released search database: each site is padded to 65 residues by
repeating its residues (wrap) with a validity mask, or truncated to its first 65 residues; C-alpha positions are
centred on the site's mean; the edge graph uses the stable kNN rule (see lenseek.geometry); the adaptor output is
encoded and mu is returned for the valid residues only.
"""
from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch

from .config import NUM_PTS, ModelConfig
from .featurize import SiteFeatures
from .geometry import edges_for
from .model import VAE

SiteInput = Union[SiteFeatures, Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _hTR(site: SiteInput):
    if isinstance(site, SiteFeatures):
        return site.h, site.T, site.R
    h, T, R = site
    return h, T, R


def pad_site(h: np.ndarray, T: np.ndarray, R: np.ndarray, num_pts: int = NUM_PTS):
    """Wrap-pad (or truncate) to ``num_pts`` rows; returns h, T, R and the validity mask."""
    n = min(len(h), num_pts)
    idx = np.arange(n)
    if n < num_pts:
        idx = np.pad(idx, (0, num_pts - n), mode="wrap")
    mask = np.zeros(num_pts, dtype=bool)
    mask[:n] = True
    return h[idx], T[idx], R[idx], mask


class Encoder:
    """Batched site encoder around a loaded :class:`~lenseek.model.VAE`.

    >>> enc = Encoder(model, cfg)              # model from lenseek.checkpoint.load_model
    >>> latents = enc.encode(sites)            # list of (n_i, 256) float32 arrays
    """

    def __init__(self, model: VAE, cfg: ModelConfig, device: Union[str, torch.device] = "cpu", batch_size: int = 32):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.no_grad()
    def encode_padded(self, sites: Sequence[SiteInput]) -> Tuple[np.ndarray, np.ndarray]:
        """mu (n_sites, 65, latent) and mask (n_sites, 65), padded exactly like the paper's database encoding."""
        mus, masks = [], []
        for s in range(0, len(sites), self.batch_size):
            chunk = [pad_site(*_hTR(x)) for x in sites[s:s + self.batch_size]]
            hs, Ts, Rs, Ms = zip(*chunk)
            h = torch.from_numpy(np.stack(hs)).float().to(self.device)
            if self.cfg.node_const:
                h = torch.zeros_like(h)
            T = torch.from_numpy(np.stack(Ts)).float().to(self.device)
            R = torch.from_numpy(np.stack(Rs)).float().to(self.device)
            M = torch.from_numpy(np.stack(Ms)).to(self.device)
            # centre the C-alpha positions of each site (training-time normalisation)
            T = T - (T * M[..., None]).sum(1, keepdim=True) / M.sum(1, keepdim=True)[..., None]
            e, knn = edges_for(self.cfg, T, R, M.unsqueeze(2) & M.unsqueeze(1))
            mu, _, _ = self.model.encode(self.model.feature_adaptor(h), e, M, knn)
            mus.append(mu.float().cpu().numpy())
            masks.append(np.stack(Ms))
        if not mus:
            return np.zeros((0, NUM_PTS, self.cfg.latent), np.float32), np.zeros((0, NUM_PTS), bool)
        return np.concatenate(mus), np.concatenate(masks)

    def encode(self, sites: Sequence[SiteInput], warn_truncation: bool = True) -> List[np.ndarray]:
        """Per-site latent point clouds: a list of (n_valid, latent) float32 arrays (n_valid = min(n, 65))."""
        if warn_truncation:
            big = [i for i, x in enumerate(sites) if len(_hTR(x)[0]) > NUM_PTS]
            if big:
                warnings.warn(f"{len(big)} site(s) have more than {NUM_PTS} residues; only the first {NUM_PTS} "
                              f"(in residue-key order) are encoded, as in the database", stacklevel=2)
        mu, mask = self.encode_padded(sites)
        return [np.ascontiguousarray(mu[i][mask[i]]) for i in range(len(mu))]
