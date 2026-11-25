"""Exhaustive Chamfer search of query sites against a latent database.

Distance between sites A and B (point clouds of per-residue latent vectors):

    CD(A, B) = mean_{a in A} min_{b in B} ||a - b||^2  +  mean_{b in B} min_{a in A} ||a - b||^2

computed from squared distances ``torch.cdist(...) ** 2`` (the matrix-multiplication form), minima per row and
column, and means over the valid residues. Smaller is more similar. ``chamfer_matrix`` works on the unpadded sites of
a latent database, ``chamfer_rows_cdist`` on padded encodings (the evaluation folders); the two agree up to float32
rounding.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .latentdb import LatentDB


def chamfer_pair(a: np.ndarray, b: np.ndarray) -> float:
    """CD of two point clouds (reference implementation, numpy float64)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(d.min(1).mean() + d.min(0).mean())


def _site_chunks(offsets: np.ndarray, max_rows: int):
    """[s0, s1) site ranges whose rows fit in max_rows (at least one site per chunk)."""
    n = len(offsets) - 1
    s0 = 0
    while s0 < n:
        limit = offsets[s0] + max_rows
        s1 = int(np.searchsorted(offsets, limit, side="right")) - 1
        s1 = max(s1, s0 + 1)
        s1 = min(s1, n)
        yield s0, s1
        s0 = s1


@torch.no_grad()
def chamfer_matrix(queries: Sequence[np.ndarray], db: LatentDB, *, threads: Optional[int] = None,
                   query_batch: int = 32, chunk_rows: int = 65536, progress: bool = False) -> np.ndarray:
    """(n_queries, n_sites) float32 Chamfer distances of every query against every database site."""
    if threads:
        torch.set_num_threads(int(threads))
    nq, ns = len(queries), len(db)
    out = np.empty((nq, ns), dtype=np.float32)
    t0 = time.time()
    for qs in range(0, nq, query_batch):
        qb = [np.asarray(q, np.float32) for q in queries[qs:qs + query_batch]]
        q_sizes = np.array([len(q) for q in qb], np.int64)
        q_off = np.concatenate([[0], np.cumsum(q_sizes)])
        Q = torch.from_numpy(np.concatenate(qb))
        for s0, s1 in _site_chunks(db.offsets, chunk_rows):
            r0, r1 = int(db.offsets[s0]), int(db.offsets[s1])
            X = torch.from_numpy(np.array(db.mu[r0:r1], dtype=np.float32))   # copy: the DB may be a read-only memmap
            D = (torch.cdist(Q.unsqueeze(0), X.unsqueeze(0)) ** 2)[0].numpy()     # (q rows, db rows)
            site_off = (db.offsets[s0:s1] - r0).astype(np.int64)
            site_n = np.diff(db.offsets[s0:s1 + 1]).astype(np.float32)
            a2b = np.add.reduceat(np.minimum.reduceat(D, site_off, axis=1), q_off[:-1], axis=0)
            a2b /= q_sizes[:, None].astype(np.float32)
            b2a = np.add.reduceat(np.minimum.reduceat(D, q_off[:-1], axis=0), site_off, axis=1)
            b2a /= site_n[None, :]
            out[qs:qs + len(qb), s0:s1] = a2b + b2a
        if db.overlay:
            for i, rows in db.overlay.items():
                X = torch.from_numpy(np.asarray(rows, np.float32))
                D = (torch.cdist(Q.unsqueeze(0), X.unsqueeze(0)) ** 2)[0].numpy()
                a2b = np.add.reduceat(D.min(1), q_off[:-1]) / q_sizes
                b2a = np.minimum.reduceat(D, q_off[:-1], axis=0).mean(1)
                out[qs:qs + len(qb), i] = a2b + b2a
        if progress:
            done = min(qs + query_batch, nq)
            print(f"[lenseek.search] {done}/{nq} queries, {time.time() - t0:.1f} s", flush=True)
    return out


def chamfer_rows_cdist(q: np.ndarray, q_mask: np.ndarray, db: np.ndarray, db_mask: np.ndarray, device: str = "cpu",
                       chunk: int = 2048, progress: bool = False) -> np.ndarray:
    """(n_q, n_db) float32 Chamfer distances of padded queries (q (n_q, P, d), q_mask (n_q, P) bool) to padded
    database sites (db (n_db, P, d), db_mask): per query, ``torch.cdist(...) ** 2`` against `chunk` database sites at a
    time, padded residues masked. This is the form of the evaluation folders (dist.npy of scripts/evaluate.py);
    chamfer_matrix computes the same distance on unpadded sites, with other float32 rounding. db may be a memmap."""
    dev = torch.device(device)

    @torch.no_grad()
    def chamfer(qm, qk, dm, dk):
        """One query against a DB chunk (squared L2, mean over each side's valid residues, both directions)."""
        d = torch.cdist(qm.expand(dm.shape[0], -1, -1), dm) ** 2
        qf, df = qk.float(), dk.float()
        a2b = (d + (1 - df).unsqueeze(1) * 1e9).min(2).values
        b2a = (d + (1 - qf).unsqueeze(2) * 1e9).min(1).values
        return (a2b * qf).sum(1) / qf.sum(1).clamp(min=1e-6) + (b2a * df).sum(1) / df.sum(1).clamp(min=1e-6)

    dist = np.empty((len(q), db.shape[0]), dtype=np.float32)
    qs = [(torch.from_numpy(q[i:i + 1]).to(dev), torch.from_numpy(q_mask[i:i + 1]).to(dev)) for i in range(len(q))]
    t0 = time.time()
    for s in range(0, db.shape[0], chunk):
        cm = torch.from_numpy(np.array(db[s:s + chunk])).to(dev)
        ck = torch.from_numpy(np.array(db_mask[s:s + chunk])).to(dev)
        for i, (qm, qk) in enumerate(qs):
            dist[i, s:s + cm.shape[0]] = chamfer(qm, qk, cm, ck).cpu().numpy()
        if progress and (s // chunk) % 10 == 9:
            print(f"[lenseek.search] {s + cm.shape[0]}/{db.shape[0]} database sites, {time.time() - t0:.0f} s",
                  flush=True)
    return dist


def topk(dist_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances; ties broken by database index (deterministic)."""
    k = min(k, len(dist_row))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    part = np.argpartition(dist_row, k - 1)[:k] if k < len(dist_row) else np.arange(len(dist_row))
    # widen to every entry equal to the k-th value so that index tie-breaking is exact
    kth = dist_row[part].max()
    cand = np.flatnonzero(dist_row <= kth)
    order = np.lexsort((cand, dist_row[cand]))
    return cand[order][:k].astype(np.int64)


@dataclass
class Hit:
    query: str
    rank: int
    site_index: int
    site_id: str
    distance: float


def search(queries: Sequence[np.ndarray], query_ids: Sequence[str], db: LatentDB, top_k: int = 100,
           **kwargs) -> List[Hit]:
    """Top-k hits of each query (rank 1 = closest)."""
    dist = chamfer_matrix(queries, db, **kwargs)
    hits: List[Hit] = []
    for qi, qid in enumerate(query_ids):
        for r, j in enumerate(topk(dist[qi], top_k), 1):
            hits.append(Hit(qid, r, int(j), db.site_ids[j], float(dist[qi, j])))
    return hits
