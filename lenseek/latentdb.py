"""Latent database: one point cloud of per-residue latent vectors per binding site.

Storage (ragged, float32, rows of all sites concatenated):

* directory ``<db>/``: ``mu.npy`` (n_rows, d), ``offsets.npy`` (n_sites + 1, int64), ``sites.tsv`` (index, site_id,
  n_res), ``meta.json``. ``mu.npy`` is memory-mapped, so a 3 GB database does not have to fit in memory.
* single file ``<name>.npz`` with the same arrays (``site_ids`` as a string array, ``meta`` as JSON) for small DBs.

Site i owns rows ``offsets[i]:offsets[i+1]``: the valid (unpadded) residues of the site, at most 65.

A *patch* (``.npz``: ``site_index``, ``offsets``, ``mu``, ``site_ids``, ``meta``) replaces the rows of some sites
without rewriting the database; :meth:`LatentDB.apply_patch` overlays it and :mod:`lenseek.search` honours the
overlay. The released standard edition is the paper edition plus such a patch (README, Usage:
Search).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

FORMAT = "lenseek-latent-db"
VERSION = 1


class LatentDB:
    def __init__(self, mu: np.ndarray, offsets: np.ndarray, site_ids: Sequence[str], meta: Optional[dict] = None):
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(mu) or np.any(np.diff(offsets) <= 0):
            raise ValueError("offsets must start at 0, end at len(mu) and be strictly increasing")
        if len(site_ids) != len(offsets) - 1:
            raise ValueError(f"{len(site_ids)} site ids for {len(offsets) - 1} sites")
        self.mu = mu
        self.offsets = offsets
        self.site_ids = list(site_ids)
        self.meta = dict(meta or {})
        self.overlay: Dict[int, np.ndarray] = {}
        self._index: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------ basic access
    def __len__(self) -> int:
        return len(self.site_ids)

    @property
    def dim(self) -> int:
        return int(self.mu.shape[1])

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def site(self, i: int) -> np.ndarray:
        """Latent rows of site i (overlay first)."""
        if i in self.overlay:
            return self.overlay[i]
        return np.asarray(self.mu[self.offsets[i]:self.offsets[i + 1]])

    def index_of(self, site_id: str) -> int:
        if self._index is None:
            self._index = {s: i for i, s in enumerate(self.site_ids)}
        return self._index[site_id]

    def __repr__(self) -> str:
        return (f"LatentDB({len(self)} sites, {len(self.mu)} rows, dim {self.dim}, model "
                f"{self.meta.get('model', '?')}, edition {self.meta.get('edition', '?')}, "
                f"{len(self.overlay)} patched)")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_latents(cls, latents: Sequence[np.ndarray], site_ids: Sequence[str], meta: Optional[dict] = None):
        sizes = [len(x) for x in latents]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        mu = np.concatenate([np.asarray(x, dtype=np.float32) for x in latents]) if latents else \
            np.zeros((0, 256), np.float32)
        return cls(mu, offsets, site_ids, meta)

    @classmethod
    def from_padded(cls, mu: np.ndarray, mask: np.ndarray, site_ids: Sequence[str], meta: Optional[dict] = None):
        """From the padded (n, 65, d) arrays of the paper's evaluation outputs (db_mu.npy / db_mask.npy)."""
        return cls.from_latents([np.asarray(mu[i])[np.asarray(mask[i])] for i in range(len(mu))], site_ids, meta)

    def subset(self, indices: Iterable[int], meta_update: Optional[dict] = None) -> "LatentDB":
        idx = list(indices)
        meta = dict(self.meta)
        meta.update(meta_update or {})
        return LatentDB.from_latents([self.site(i) for i in idx], [self.site_ids[i] for i in idx], meta)

    # ------------------------------------------------------------------ patches
    def apply_patch(self, patch_path: str, check_sizes: bool = True) -> None:
        """Overlay the rows of a patch file (e.g. the standard-edition patch of the released database)."""
        z = np.load(patch_path, allow_pickle=False)
        idx = z["site_index"].astype(np.int64)
        off = z["offsets"].astype(np.int64)
        mu = z["mu"].astype(np.float32)
        ids = [str(s) for s in z["site_ids"]]
        pmeta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        base = pmeta.get("base_sha256_mu")
        if base and self.meta.get("sha256_mu") and base != self.meta["sha256_mu"]:
            raise ValueError(f"patch {patch_path} was made for another database (mu sha256 {base[:12]}...)")
        for k, i in enumerate(idx):
            if self.site_ids[i] != ids[k]:
                raise ValueError(f"patch site {ids[k]} does not match database site {self.site_ids[i]} at {i}")
            rows = mu[off[k]:off[k + 1]]
            if check_sizes and len(rows) != self.sizes[i]:
                raise ValueError(f"patch rows for {ids[k]}: {len(rows)} != {self.sizes[i]}")
            self.overlay[int(i)] = rows
        self.meta.setdefault("patches", []).append({"file": os.path.basename(patch_path), **{
            k: v for k, v in pmeta.items() if k in ("name", "description", "n_sites")}})
        if pmeta.get("edition"):
            self.meta["edition"] = pmeta["edition"]

    @staticmethod
    def write_patch(path: str, base: "LatentDB", site_index: Sequence[int], latents: Sequence[np.ndarray],
                    meta: Optional[dict] = None) -> None:
        sizes = [len(x) for x in latents]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        m = dict(meta or {})
        m.setdefault("n_sites", len(site_index))
        if base.meta.get("sha256_mu"):
            m.setdefault("base_sha256_mu", base.meta["sha256_mu"])
        np.savez(path, site_index=np.asarray(site_index, np.int64), offsets=offsets,
                 mu=np.concatenate(latents).astype(np.float32),
                 site_ids=np.array([base.site_ids[i] for i in site_index]), meta=np.array(json.dumps(m)))

    # ------------------------------------------------------------------ I/O
    def save(self, path: str) -> None:
        """Write a directory (default) or, if ``path`` ends with .npz, a single file. The overlay is not written."""
        if self.overlay:
            raise ValueError("save() would drop the patch overlay; materialise it with subset() first")
        meta = dict(self.meta)
        meta.update(format=FORMAT, version=VERSION, n_sites=len(self), n_rows=int(len(self.mu)), dim=self.dim)
        if path.endswith(".npz"):
            np.savez_compressed(path, mu=np.asarray(self.mu, np.float32), offsets=self.offsets,
                                site_ids=np.array(self.site_ids), meta=np.array(json.dumps(meta)))
            return
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "mu.npy"), np.asarray(self.mu, np.float32))
        np.save(os.path.join(path, "offsets.npy"), self.offsets)
        with open(os.path.join(path, "sites.tsv"), "w") as f:
            f.write("index\tsite_id\tn_res\n")
            for i, (s, n) in enumerate(zip(self.site_ids, self.sizes)):
                f.write(f"{i}\t{s}\t{n}\n")
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    @classmethod
    def load(cls, path: str, mmap: bool = True) -> "LatentDB":
        if os.path.isdir(path):
            with open(os.path.join(path, "meta.json")) as f:
                meta = json.load(f)
            mu = np.load(os.path.join(path, "mu.npy"), mmap_mode="r" if mmap else None)
            offsets = np.load(os.path.join(path, "offsets.npy"))
            ids = []
            with open(os.path.join(path, "sites.tsv")) as f:
                header = f.readline().rstrip("\n").split("\t")
                col = header.index("site_id")
                for line in f:
                    ids.append(line.rstrip("\n").split("\t")[col])
        else:
            z = np.load(path, allow_pickle=False)
            mu, offsets = z["mu"], z["offsets"]
            ids = [str(s) for s in z["site_ids"]]
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if meta.get("format", FORMAT) != FORMAT:
            raise ValueError(f"{path} is not a LEN-Seek latent database")
        return cls(mu, offsets, ids, meta)
