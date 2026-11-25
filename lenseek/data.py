"""Training / database feature stores.

``FeatureStore`` (directory, written by ``lenseek-featurize``): per-site model inputs concatenated over sites,
memory-mapped for training on all 114,661 sites:

    h.npy (n_rows, node_dim) float32, T.npy (n_rows, 3) float32, R.npy (n_rows, 3, 3) float32,
    offsets.npy (n_sites + 1) int64, sites.tsv (index, site_id, n_res), meta.json

The training pickle (a pickled dataset object with the per-site arrays ``data`` and paths ``path``) holds the same
inputs. :func:`load_legacy_pickle` reads it without the module that defined that object, and
:func:`convert_legacy_pickle` turns it into a FeatureStore with identical arrays.

``SiteDataset`` samples a site as in the training of the released models: 65 residues drawn without replacement with
``np.random.choice`` when the site has at least 65, otherwise all residues wrap-padded to 65 with a mask.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NUM_PTS
from .featurize import SiteFeatures


class FeatureStore:
    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        mode = "r" if mmap else None
        self.h = np.load(os.path.join(path, "h.npy"), mmap_mode=mode)
        self.T = np.load(os.path.join(path, "T.npy"), mmap_mode=mode)
        self.R = np.load(os.path.join(path, "R.npy"), mmap_mode=mode)
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.site_ids: List[str] = []
        with open(os.path.join(path, "sites.tsv")) as f:
            col = f.readline().rstrip("\n").split("\t").index("site_id")
            for line in f:
                self.site_ids.append(line.rstrip("\n").split("\t")[col])

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def site(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return np.asarray(self.h[a:b]), np.asarray(self.T[a:b]), np.asarray(self.R[a:b])

    @staticmethod
    def write(path: str, sites: Sequence[SiteFeatures], meta: Optional[dict] = None) -> None:
        FeatureStore.write_arrays(path, [s.h for s in sites], [s.T for s in sites], [s.R for s in sites],
                                  [s.site_id for s in sites], meta)

    @staticmethod
    def write_arrays(path, hs, Ts, Rs, site_ids, meta=None) -> None:
        os.makedirs(path, exist_ok=True)
        sizes = [len(h) for h in hs]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        np.save(os.path.join(path, "h.npy"), np.concatenate(hs).astype(np.float32, copy=False))
        np.save(os.path.join(path, "T.npy"), np.concatenate(Ts).astype(np.float32, copy=False))
        np.save(os.path.join(path, "R.npy"), np.concatenate(Rs).astype(np.float32, copy=False))
        np.save(os.path.join(path, "offsets.npy"), offsets)
        with open(os.path.join(path, "sites.tsv"), "w") as f:
            f.write("index\tsite_id\tn_res\n")
            for i, (s, n) in enumerate(zip(site_ids, sizes)):
                f.write(f"{i}\t{s}\t{n}\n")
        m = dict(meta or {})
        m.update(format="lenseek-feature-store", version=1, n_sites=len(sizes), n_rows=int(offsets[-1]),
                 node_dim=int(hs[0].shape[1]) if hs else 0)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(m, f, indent=1)


# --------------------------------------------------------------------------- training pickle

class LegacySites:
    """Unpickled training pickle: ``.data`` = list of (node_feats, rots, sc_vecs, trans, res_nums, chain_ids),
    ``.path`` = list of pocket paths, ``.pocPts`` = 65."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.data)

    def site(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_feats, rots, _sc, trans, _res, _chain = self.data[i]
        return node_feats, trans, rots


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("utils.") or module in ("bsdb", "utils"):
            return LegacySites
        return super().find_class(module, name)


def load_legacy_pickle(path: str) -> LegacySites:
    """Read a training or database pickle without the module that defined its dataset class."""
    with open(path, "rb") as f:
        obj = _LegacyUnpickler(f).load()
    if not hasattr(obj, "data") or not hasattr(obj, "path"):
        raise ValueError(f"{path} is not a LEN-Seek v1.0 dataset pickle")
    return obj


def convert_legacy_pickle(pkl_path: str, out_dir: str, path_prefix: Optional[str] = None) -> FeatureStore:
    """Write a FeatureStore with the same h, T, R arrays (and site order) as a training pickle.

    Site ids are '<entry>/<pocket stem>' derived from the stored pocket paths; ``path_prefix`` is stripped from the
    stored paths for the 'source_path' column of sites.tsv when given."""
    ds = load_legacy_pickle(pkl_path)
    hs, Ts, Rs, ids = [], [], [], []
    for (node_feats, rots, _sc, trans, _res, _chain), p in zip(ds.data, ds.path):
        hs.append(np.asarray(node_feats, np.float32))
        Ts.append(np.asarray(trans, np.float32))
        Rs.append(np.asarray(rots, np.float32))
        ids.append(f"{os.path.basename(os.path.dirname(p))}/{os.path.splitext(os.path.basename(p))[0]}")
    FeatureStore.write_arrays(out_dir, hs, Ts, Rs, ids, meta={"source": os.path.basename(pkl_path),
                                                              "converted_from": "v1.0 pickle"})
    return FeatureStore(out_dir)


# --------------------------------------------------------------------------- training dataset

class SiteDataset(Dataset):
    """Training samples (h, T, mask, R), each padded or subsampled to 65 residues (see the module docstring)."""

    def __init__(self, source, num_pts: int = NUM_PTS):
        self.source = source           # FeatureStore or LegacySites
        self.num_pts = num_pts

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        real_idx = idx % len(self.source)
        node_feats, trans, rots = self.source.site(real_idx)
        num_coords = len(trans)
        if num_coords >= self.num_pts:
            poc_idx = np.random.choice(num_coords, self.num_pts, replace=False)
            mask_np = np.ones(self.num_pts, dtype=bool)
        else:
            poc_idx = np.arange(num_coords)
            poc_pad = self.num_pts - num_coords
            poc_idx = np.pad(poc_idx, (0, poc_pad), mode="wrap")
            mask_np = np.ones(self.num_pts, dtype=bool)
            mask_np[-poc_pad:] = False
        h = torch.from_numpy(np.asarray(node_feats[poc_idx])).float()
        t = torch.from_numpy(np.asarray(trans[poc_idx])).float()
        r = torch.from_numpy(np.asarray(rots[poc_idx])).float()
        mask = torch.from_numpy(mask_np)
        return h, t, mask, r


def open_training_data(path: str):
    """FeatureStore directory or training pickle -> object with ``site(i)`` and ``__len__``."""
    if os.path.isdir(path):
        return FeatureStore(path)
    return load_legacy_pickle(path)
