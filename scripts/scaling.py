"""Two-stage search on databases of increasing size: candidates, recall, time and memory (Supplementary Section S8 and
Table S15 B; analysis/table3_two_stage.py makes the tables and the ligand-similar fraction
of the two-stage top 10 from the candidate lists).

A feasibility measurement: the package itself searches by an exhaustive Chamfer scan. Stage 1: every query residue
retrieves its r = 256 nearest residues from an IVF-PQ index of all database residues (IVF{nlist},PQ32, nlist =
2^round(log2(2.5 sqrt(R))) for R residues, 16 lists probed), and each site hit this way gets an approximate
one-directional Chamfer score (`candidates`). Stage 2: the first K' = 1,000 (or 3,000) candidates are re-ranked by the
exact Chamfer distance, their residue vectors read in fp16 from a node-local disk. Recall is the fraction of the exact
top-k inside the K' candidates, which is the recall of the re-ranked list.

Databases: real (the search database), real150k (real + the sites removed by the ligand filters), chim3e5 / chim1e6 /
chim1e7 (real + chimeras that combine residues of two to four real sites from different 50 % sequence clusters, with
noise, up to 3x10^5, 10^6 and 10^7 sites), copy1e6 / copy1e7 (real + noisy copies of real sites). Synthetic sites
follow the real ones in site order. FAISS searches split the probed lists of a query among threads (parallel_mode 1),
and the recall at 10^7 sites uses one-thread stage-1 searches; --ref-cands counts the candidate lists of a run that
equal those of another run.

Subcommands, in run order (all write to $LENSEEK_OUT/scaling/, per database to <db>/):
  clusters  site -> 50 % sequence-cluster file of the real sites (input of the chimera generator)
  params    generator parameters: per-dimension std of real residue vectors and the noise scale sigma
  generate  fp16 residue store (node-local scratch) and exact top 1,000 of the queries -> gen.json, gt_top1000.npz
  index     IVF-PQ residue index (node-local scratch)                                 -> build_<tag>.json
  stage1    stage-1 candidates of the queries                                         -> cands_np16.npz, stage1.json
  recall    recall@1/10/100 at K' 1,000 and 3,000 per query                           -> recall_per_query.csv.gz
  time      per-query two-stage time, candidates read cold from the store            -> search_T<threads>.csv.gz/.json
  exact     per-query exhaustive-scan time, fp16 vectors in memory                    -> exact_T<threads>_<set>.csv/.json
  gpu       exhaustive scan on one GPU; a 10^6-site database streamed ten times stands in for 10^7 -> exact_gpu*.json
Inputs (environment): LENSEEK_SCALING_CACHE (CSR cache of scripts/single_vector.py prep: db_res.npy, db_off.npy,
q_res.npy, q_off.npy, gt_d.npy; default $LENSEEK_OUT/single_vector/cache), LENSEEK_SCALING_REAL_DIST (dist.npy of the
same model), LENSEEK_SCALING_ADDED (added_mu.npy, added_mask.npy, dist_added.npy of the sites removed by the ligand
filters, scripts/evaluate.py --added with the same checkpoint; default $LENSEEK_OUT/scaling/added),
LENSEEK_SCALING_CLUSTERS (site_cluster50.npy, default $LENSEEK_OUT/scaling/), LENSEEK_SCALING_LOCAL (node-local
scratch, default $TMPDIR/lenseek_scaling); clusters reads db_sites.tsv and
db50_cluster.tsv of scripts/site_annotation.py ($LENSEEK_OUT/cache/site_annot/) and $LENSEEK_EVAL_DIR/db_paths.txt.
A 10^7-site database needs about 150 GB of node-local disk and more memory than a workstation.

usage: python scripts/scaling.py clusters; python scripts/scaling.py params
       python scripts/scaling.py generate real; python scripts/scaling.py index real
       python scripts/scaling.py stage1 real; python scripts/scaling.py recall real
       python scripts/scaling.py time real --threads 1; python scripts/scaling.py exact real
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (SIZE_CLASSES, bucketize, candidates, chamfer_scan, log, pad_queries, peak_rss_gb,  # noqa: E402
                      rerank, set_threads, site_chain, status_gb, topk_stable)

OUT = os.path.join(os.environ.get("LENSEEK_OUT", "out"), "scaling")
ANNOT = os.path.join(os.environ.get("LENSEEK_OUT", "out"), "cache", "site_annot")
LOCAL = os.environ.get("LENSEEK_SCALING_LOCAL") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "lenseek_scaling")
CACHE = os.environ.get("LENSEEK_SCALING_CACHE") or os.path.join(os.environ.get("LENSEEK_OUT", "out"), "single_vector",
                                                                "cache")
REAL_DIST = os.environ.get("LENSEEK_SCALING_REAL_DIST", "")
ADDED = os.environ.get("LENSEEK_SCALING_ADDED") or os.path.join(OUT, "added")
CLUSTERS = os.environ.get("LENSEEK_SCALING_CLUSTERS") or os.path.join(OUT, "site_cluster50.npy")

N_REAL = 114_661
DBS = {"real": dict(kind="real", n_total=N_REAL), "real150k": dict(kind="real+added", n_total=N_REAL + 34_976),
       "chim3e5": dict(kind="chimera", n_total=300_000), "chim1e6": dict(kind="chimera", n_total=1_000_000),
       "chim1e7": dict(kind="chimera", n_total=10_000_000), "copy1e6": dict(kind="copy", n_total=1_000_000),
       "copy1e7": dict(kind="copy", n_total=10_000_000)}
DIM, Q = 256, 481
CH = 25_000                       # synthetic generation chunk (sites)
SEED_COPY, P_KEEP = 12345, 0.9    # noisy copy
SEED_CHIM = 20261105              # chimera
SEED_TRAIN = 0                    # IVF training sample
R_NN, NPROBE = 256, 16            # nearest database residues per query residue, lists probed
KP = (1000, 3000)                 # candidates re-ranked
KEEP = 3000                       # candidates stored per query


# ------------------------------------------------------------------------------------------------- helpers
def jdump(obj, path):
    """Write obj as JSON through a temporary file (numpy scalars as numbers)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------- files
def nlist_for(R):
    """nlist = 2^round(log2(2.5 sqrt(R)))."""
    return int(2 ** round(np.log2(2.5 * np.sqrt(R))))


def odir(db):
    d = os.path.join(OUT, db)
    os.makedirs(d, exist_ok=True)
    return d


def ldir(db):
    d = os.path.join(LOCAL, db)
    os.makedirs(d, exist_ok=True)
    return d


def host_info():
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    return dict(host=os.uname().nodename, cpu=cpu, job=os.environ.get("SLURM_JOB_ID", ""), torch=torch.__version__)


def load_real():
    return np.load(os.path.join(CACHE, "db_res.npy"), mmap_mode="r"), np.load(os.path.join(CACHE, "db_off.npy"))


def load_queries():
    return np.ascontiguousarray(np.load(os.path.join(CACHE, "q_res.npy"))), np.load(os.path.join(CACHE, "q_off.npy"))


def load_added():
    mu = np.load(os.path.join(ADDED, "added_mu.npy"), mmap_mode="r")
    mask = np.load(os.path.join(ADDED, "added_mask.npy"))
    n = mask.sum(1).astype(np.int64)
    assert all(mask[i, :n[i]].all() for i in range(len(n))), "added mask is not a prefix"
    return mu, n


def gen_params():
    p = json.load(open(os.path.join(OUT, "gen_params.json")))
    return float(p["sigma"]), np.load(os.path.join(OUT, "gen_std.npy"))


def save_lists(path, lists, **extra):
    np.savez(path, ids=np.concatenate(lists), off=np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int64), **extra)


def same_lists(lists, ref_path):
    """Number of queries whose candidate list equals the first len(list) entries of the reference list."""
    z = np.load(ref_path)
    ref = [z["ids"][z["off"][i]:z["off"][i + 1]][:KEEP].astype(np.int32) for i in range(len(z["off"]) - 1)]
    same = [i for i, x in enumerate(lists) if i < len(ref) and np.array_equal(x, ref[i])]
    return dict(ref_cands=ref_path, identical=len(same), queries=len(lists),
                differing=[i for i in range(len(lists)) if i not in set(same)][:200])


# ------------------------------------------------------------------------------------------------- generators
def gen_copy(cid, nsites, sigma, res, off, std):
    """Noisy copy: a uniformly drawn real site, each residue kept with p = 0.9 (all kept when fewer than min(n, 10)
    survive), noise sigma * std_d * N(0, 1) on every kept residue. Returns X (R, 256) float32, offsets, source ids."""
    n_real = np.diff(off)
    rng = np.random.default_rng([SEED_COPY, cid])
    src = rng.integers(0, N_REAL, nsites)
    ns = n_real[src]
    loc = np.zeros(nsites + 1, dtype=np.int64)
    np.cumsum(ns, out=loc[1:])
    idx = np.repeat(off[src] - loc[:-1], ns) + np.arange(loc[-1])
    keep = rng.random(loc[-1]) < P_KEEP
    cnt = np.add.reduceat(keep.astype(np.int64), loc[:-1])
    keep |= np.repeat(cnt < np.minimum(ns, 10), ns)
    cnt = np.add.reduceat(keep.astype(np.int64), loc[:-1])
    X = np.asarray(res[idx[keep]])
    g = torch.Generator().manual_seed(SEED_COPY * 1_000_003 + cid)
    noise = torch.randn(X.shape, generator=g).numpy()
    X += (sigma * std)[None, :] * noise
    o = np.zeros(nsites + 1, dtype=np.int64)
    np.cumsum(cnt, out=o[1:])
    return X, o, src


def gen_chimera(cid, nsites, sigma, res, off, std, clus):
    """Chimera. Per synthetic site:
      n = size of a uniformly drawn real site; m ~ U{2, 3, 4} sources; sources = uniformly drawn real sites from m
      different 50 % sequence clusters (draws whose cluster is already used are skipped); the n residues are split as
      evenly as possible (the first n mod m sources in draw order get one more), a source with fewer residues than its
      share gives all of them and the deficit goes to the other sources in draw order; residues of a source are drawn
      uniformly without replacement; every residue gets the noisy-copy noise sigma * std_d * N(0, 1).
    Returns X (R, 256) float32, offsets (nsites+1,), sources (nsites, 4) int32 (-1 = unused)."""
    n_real = np.diff(off)
    rng = np.random.default_rng([SEED_CHIM, cid])
    n = n_real[rng.integers(0, N_REAL, nsites)]
    m = rng.integers(2, 5, nsites)
    cand = rng.integers(0, N_REAL, (nsites, 12))
    cc = clus[cand]
    dup = np.zeros(cand.shape, bool)                       # same cluster as an earlier draw of the row
    for j in range(1, 12):
        dup[:, j] = (cc[:, :j] == cc[:, j:j + 1]).any(1)
    first = ~dup
    rank = np.cumsum(first, 1)                             # 1-based rank among the distinct-cluster draws
    take = first & (rank <= m[:, None])
    src = np.full((nsites, 4), -1, np.int64)
    short = np.flatnonzero(take.sum(1) < m)
    full = np.setdiff1d(np.arange(nsites), short)
    r_, c_ = np.nonzero(take[full])
    src[full[r_], rank[full][r_, c_] - 1] = cand[full][r_, c_]
    for i in short:                                        # rare: fewer than m distinct clusters in 12 draws
        seen, k, row, j = [], 0, cand[i], 0
        while k < m[i]:
            if j == len(row):
                row, j = rng.integers(0, N_REAL, 12), 0
            s_, j = row[j], j + 1
            if clus[s_] in seen:
                continue
            seen.append(clus[s_])
            src[i, k] = s_
            k += 1
    # allocation of residues to sources (vectorized; deficits go to the other sources in draw order)
    cap = np.where(src >= 0, n_real[np.maximum(src, 0)], 0)
    slot = np.arange(4)[None, :]
    a = np.where(slot < m[:, None], (n // m)[:, None] + (slot < (n % m)[:, None]), 0)
    deficit = np.maximum(a - cap, 0).sum(1)
    a = np.minimum(a, cap)
    for k in range(4):
        give = np.minimum(deficit, cap[:, k] - a[:, k])
        a[:, k] += give
        deficit -= give
    alloc = a
    # residues: per (site, source) pair, the alloc smallest random keys among the source's residues
    pi, pk = np.nonzero(alloc > 0)
    ps = src[pi, pk]
    pn = n_real[ps]
    pa = alloc[pi, pk]
    ploc = np.zeros(len(ps) + 1, np.int64)
    np.cumsum(pn, out=ploc[1:])
    pid = np.repeat(np.arange(len(ps)), pn)
    ridx = np.repeat(off[ps] - ploc[:-1], pn) + np.arange(ploc[-1])     # global real residue rows
    key = rng.random(ploc[-1])
    o_ = np.lexsort((key, pid))                                         # by pair, then random key
    rank = np.arange(ploc[-1]) - np.repeat(ploc[:-1], pn)              # rank within pair after sorting
    sel = o_[rank < np.repeat(pa, pn)]
    # sel is grouped by pair; pairs are ordered by (site, source slot) because np.nonzero is row-major
    rows = ridx[sel]
    X = np.asarray(res[rows])
    X += (sigma * std)[None, :] * rng.standard_normal(X.shape, dtype=np.float32)
    cnt = alloc.sum(1)
    o = np.zeros(nsites + 1, np.int64)
    np.cumsum(cnt, out=o[1:])
    assert o[-1] == len(X)
    return X.astype(np.float32, copy=False), o, src.astype(np.int32)


def pair_cd(a, b):
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    d = np.maximum(d, 0)
    return d.min(1).mean() + d.min(0).mean()


# ------------------------------------------------------------------------------------------------- fp16 store
class Store:
    """Residue vectors of every database site in fp16, CSR order (site i = rows off[i]:off[i+1]): raw file res.f16 on
    the node-local disk, offsets in memory."""

    def __init__(self, db):
        self.dir = ldir(db)
        self.path = os.path.join(self.dir, "res.f16")
        self.off = np.load(os.path.join(self.dir, "off.npy"))
        self.fd = os.open(self.path, os.O_RDONLY)
        self.R = int(self.off[-1])
        assert os.path.getsize(self.path) == self.R * DIM * 2, "store size mismatch"

    def drop_cache(self):
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def read_rows(self, s, e, out=None):
        """rows s:e as float16 (sequential read, no mmap)."""
        nb = (e - s) * DIM * 2
        buf = bytearray(nb) if out is None else out
        mv = memoryview(buf)[:nb]
        got = 0
        while got < nb:
            k = os.preadv(self.fd, [mv[got:]], s * DIM * 2 + got)
            if k <= 0:
                raise IOError("short read")
            got += k
        return np.frombuffer(buf, dtype=np.float16, count=(e - s) * DIM).reshape(e - s, DIM)


class StoreWriter:
    def __init__(self, db):
        self.dir = ldir(db)
        self.f = open(os.path.join(self.dir, "res.f16.part"), "wb", buffering=64 << 20)
        self.n = [np.zeros(0, np.int64)]

    def add(self, X, o):
        np.ascontiguousarray(X, dtype=np.float16).tofile(self.f)
        self.n.append(np.diff(o))

    def close(self):
        self.f.close()
        n = np.concatenate(self.n)
        off = np.zeros(len(n) + 1, np.int64)
        np.cumsum(n, out=off[1:])
        np.save(os.path.join(self.dir, "off.npy"), off)
        os.replace(os.path.join(self.dir, "res.f16.part"), os.path.join(self.dir, "res.f16"))
        return off


class Reranker:
    """Read candidate sites from the fp16 store (pread, optional thread pool) and re-rank them by exact Chamfer."""

    def __init__(self, store, threads=1, max_res=5000 * 65):
        self.st = store
        self.buf = bytearray(max_res * DIM * 2)
        self.mv = memoryview(self.buf)
        self.threads = threads
        self.pool = None
        if threads > 1:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(threads)

    def _read(self, starts, lens, dsts):
        fd, mv = self.st.fd, self.mv
        for s, l, d in zip(starts, lens, dsts):
            got = 0
            while got < l:
                got += os.preadv(fd, [mv[d + got:d + l]], s + got)

    def read(self, cand):
        o = np.sort(np.asarray(cand, np.int64))
        off = self.st.off
        n = off[o + 1] - off[o]
        loc = np.zeros(len(o) + 1, np.int64)
        np.cumsum(n, out=loc[1:])
        starts = (off[o] * DIM * 2).tolist()
        lens = (n * DIM * 2).tolist()
        dsts = (loc[:-1] * DIM * 2).tolist()
        if self.pool is None:
            self._read(starts, lens, dsts)
        else:
            T = self.threads
            parts = [(starts[k::T], lens[k::T], dsts[k::T]) for k in range(T)]
            list(self.pool.map(lambda p: self._read(*p), parts))
        return o, loc

    def rerank(self, q, o, loc, k=100):
        X = np.frombuffer(self.buf, dtype=np.float16, count=int(loc[-1]) * DIM).reshape(-1, DIM)
        return rerank(q, o, torch.from_numpy(X.astype(np.float32)), loc, k=k)


def load_index(path):
    import faiss
    index = faiss.read_index(path)
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    ivf.parallel_mode = 1
    ivf.nprobe = NPROBE
    return index


def fp16_blocks(db, pin=False):
    """Whole database in memory as fp16 size-class blocks (wrap-padded, the layout of bucketize), read from the store:
    (N, [(site ids, X (S, P, 256) fp16, mask, n)])."""
    st = Store(db)
    off = st.off
    N = len(off) - 1
    n = np.diff(off)
    cls = np.searchsorted(SIZE_CLASSES, n)
    blocks = []
    for ci in np.unique(cls):
        ids = np.flatnonzero(cls == ci)
        P = int(SIZE_CLASSES[ci])
        X = torch.empty((len(ids), P, DIM), dtype=torch.float16, pin_memory=pin)
        m = torch.from_numpy((np.arange(P)[None, :] < n[ids, None]).astype(np.float32))
        blocks.append((ids, X, m, torch.from_numpy(n[ids].astype(np.float32))))
    pos = np.empty(N, np.int64)
    for ids, *_r in blocks:
        pos[ids] = np.arange(len(ids))
    bof = np.searchsorted(np.unique(cls), cls)
    STEP = 1 << 20
    buf = bytearray(STEP * DIM * 2)
    s = 0
    while s < N:                                       # site ranges of <= STEP residues
        e = int(np.searchsorted(off, off[s] + STEP, side="right")) - 1
        e = max(min(e, N), s + 1)
        R0, R1 = int(off[s]), int(off[e])
        Xr = st.read_rows(R0, R1, buf if (R1 - R0) <= STEP else None)
        sl = np.arange(s, e)
        for b in np.unique(bof[sl]):
            ids_l = sl[bof[sl] == b]
            P = blocks[b][1].shape[1]
            j = np.arange(P)
            idx = (off[ids_l] - R0)[:, None] + (j[None, :] % n[ids_l, None])
            blocks[b][1][torch.from_numpy(pos[ids_l])] = torch.from_numpy(
                np.ascontiguousarray(Xr[idx.ravel()]).reshape(len(ids_l), P, DIM))
        s = e
    return N, blocks


def exact_scan_one(qres, qoff, i, N, blocks):
    """Exact Chamfer of query i against every site (chamfer_scan on the fp16 blocks) and its top 100, ties by index."""
    q = qres[qoff[i]:qoff[i + 1]]
    out = chamfer_scan(q, np.array([0, len(q)]), blocks, np.empty((1, N), dtype=np.float32), qbatch=1)[0]
    return out, topk_stable(out, 100)


# ------------------------------------------------------------------------------------------------- subcommands
def cmd_clusters(a):
    """Cluster of a site = the representative of its main chain (the chain with the most pocket residues, ties
    alphabetical) in mmseqs easy-cluster --min-seq-id 0.5 -c 0.8 --cov-mode 0 of the database chains. A main chain absent
    from the FASTA (shorter than 10 residues) falls back to the site's next chain by pocket residues; with no chain in
    the FASTA the site is its own singleton cluster."""
    S = pd.read_csv(a.db_sites, sep="\t", dtype=str, keep_default_na=False)
    keys = ["/".join(p.split("/")[-2:]) for p in open(a.db_paths).read().split()]
    assert list(S.key) == keys, "db_sites.tsv order != db_paths.txt order"
    C = pd.read_csv(a.clusters50, sep="\t", header=None, names=["rep", "mem"])
    rep = dict(zip(C.mem, C.rep))
    reps = {r: i for i, r in enumerate(sorted(set(C.rep)))}
    out = np.empty(len(S), np.int32)
    nxt = len(reps)
    for i, r in enumerate(S.itertuples()):
        ch = site_chain(r.key, r.main_chain, r.chains, rep)
        if ch is None:
            cid, nxt = nxt, nxt + 1
        else:
            cid = reps[rep[ch]]
        out[i] = cid
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, out)
    log(f"sites {len(out)}, clusters used {len(np.unique(out))}")


def cmd_params(a):
    """std_d = per-dimension std of 200,000 real residue vectors; sigma such that the median exact Chamfer distance
    between a noisy copy and its source equals the median exact top-1 distance of the queries, by 25 bisection steps
    on a 2,000-site chunk (chunk id 10^6, never part of a database)."""
    set_threads(a.threads)
    res, off = load_real()
    res = np.ascontiguousarray(res)
    gd = np.load(os.path.join(CACHE, "gt_d.npy"))
    samp = np.sort(np.random.default_rng(1).choice(int(off[-1]), 200_000, replace=False))
    std = np.asarray(res[samp]).std(0).astype(np.float32)
    target = float(np.median(gd[:, 0]))
    lo, hi = 1e-3, 3.0
    for _ in range(25):
        s = np.sqrt(lo * hi)
        X, o, src = gen_copy(10 ** 6, 2000, s, res, off, std)
        med = np.median([pair_cd(X[o[i]:o[i + 1]], np.asarray(res[off[src[i]]:off[src[i] + 1]])) for i in range(2000)])
        lo, hi = (s, hi) if med < target else (lo, s)
    sigma = float(np.sqrt(lo * hi))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "gen_std.npy"), std)
    jdump(dict(sigma=sigma, achieved_median_cd=float(med), target_median_top1_cd=target),
          os.path.join(OUT, "gen_params.json"))
    log(f"sigma {sigma:.6f} (median CD {med:.3f}, target {target:.3f})")


def cmd_generate(a):
    """Store = fp16 of the real latents (+ the filtered sites, or synthetic chunks of 25,000 sites). Exact distances:
    real part from LENSEEK_SCALING_REAL_DIST, filtered sites from dist_added.npy, synthetic chunks by the CPU kernel
    in fp32 before the fp16 write. Ground truth = exact top 1,000 per query, ties by index."""
    cfg = DBS[a.db]
    set_threads(a.threads)
    od = odir(a.db)
    info = dict(db=a.db, kind=cfg["kind"], store_only=a.store_only, threads=a.threads, **host_info())
    t_all = time.perf_counter()
    res, off = load_real()
    res = np.ascontiguousarray(res)               # 2.8 GB in memory: the generators gather from it
    qres, qoff = load_queries()
    N = cfg["n_total"]
    W = StoreWriter(a.db)
    gt = not a.store_only
    D = np.empty((Q, N), np.float32) if gt else None
    tw = tg = tgt = 0.0
    t = time.perf_counter()
    for s in range(0, N_REAL, CH):
        e = min(N_REAL, s + CH)
        W.add(res[off[s]:off[e]], off[s:e + 1] - off[s])
    tw += time.perf_counter() - t
    if gt:
        D[:, :N_REAL] = np.load(REAL_DIST, mmap_mode="r")
    if cfg["kind"] == "real+added":
        mu, n_add = load_added()
        t = time.perf_counter()
        for s in range(0, len(n_add), 5000):
            e = min(len(n_add), s + 5000)
            blk = np.asarray(mu[s:e])
            X = np.concatenate([blk[i, :n_add[s + i]] for i in range(e - s)])
            o = np.zeros(e - s + 1, np.int64)
            np.cumsum(n_add[s:e], out=o[1:])
            W.add(X, o)
        tw += time.perf_counter() - t
        if gt:
            D[:, N_REAL:N] = np.load(os.path.join(ADDED, "dist_added.npy"), mmap_mode="r")
    elif cfg["kind"] in ("chimera", "copy"):
        sigma, std = gen_params()
        clus = np.load(CLUSTERS) if cfg["kind"] == "chimera" else None
        n_syn = N - N_REAL
        nch = (n_syn + CH - 1) // CH
        info.update(sigma=sigma, n_syn=n_syn, chunks=nch)
        for k in range(nch):
            ns = min(CH, n_syn - k * CH)
            t = time.perf_counter()
            if cfg["kind"] == "chimera":
                X, o, _ = gen_chimera(k, ns, sigma, res, off, std, clus)
            else:
                X, o, _ = gen_copy(k, ns, sigma, res, off, std)
            tg += time.perf_counter() - t
            s0 = N_REAL + k * CH
            if gt:
                t = time.perf_counter()
                chamfer_scan(qres, qoff, bucketize(X, o), D[:, s0:s0 + ns])
                tgt += time.perf_counter() - t
            t = time.perf_counter()
            W.add(X, o)
            tw += time.perf_counter() - t
            if k % 20 == 0 or k == nch - 1:
                log(f"{a.db} chunk {k + 1}/{nch}: gen {tg:.0f}s GT {tgt:.0f}s write {tw:.0f}s, peak {peak_rss_gb():.1f} GB")
            del X, o
    offw = W.close()
    info.update(N=int(len(offw) - 1), R=int(offw[-1]), store_bytes=int(offw[-1]) * DIM * 2,
                store_bytes_per_site=float(offw[-1]) * DIM * 2 / (len(offw) - 1), nlist=nlist_for(int(offw[-1])),
                gen_s=tg, gt_s=tgt, store_write_s=tw)
    if gt:
        K = min(1000, N)
        gi = np.empty((Q, K), np.int32)
        gd = np.empty((Q, K), np.float32)
        for i in range(Q):
            idx = topk_stable(D[i], K)
            gi[i], gd[i] = idx, D[i, idx]
        np.savez(os.path.join(od, "gt_top1000.npz"), idx=gi, d=gd)
    info.update(total_s=time.perf_counter() - t_all, peak_rss_gb=peak_rss_gb())
    jdump(info, os.path.join(od, "gen_storeonly.json" if a.store_only else "gen.json"))
    log(info)


def cmd_index(a):
    """IVF{nlist},PQ32 (flat L2 coarse quantizer, residual PQ, 8-bit codes), trained on 64 x nlist residues drawn
    uniformly without replacement (default_rng(0)) with FAISS k-means defaults; every residue added with its site id.
    Vectors = the fp16 store upcast to fp32. build_s = train_s + add_s (store reads excluded). The reported build time
    and memory are those of a 16-thread build (--tag time, written to another --index); searches use the index of the
    accuracy build (--tag acc)."""
    import faiss
    set_threads(a.threads)
    st = Store(a.db)
    R, N = st.R, len(st.off) - 1
    nlist = nlist_for(R)
    ntr = 64 * nlist
    info = dict(db=a.db, tag=a.tag, threads=a.threads, R=R, N=N, nlist=nlist, n_train=ntr, factory=f"IVF{nlist},PQ32",
                **host_info())
    STEP = 1 << 20
    buf = bytearray(STEP * DIM * 2)
    t = time.perf_counter()
    samp = np.sort(np.random.default_rng(SEED_TRAIN).choice(R, ntr, replace=False))
    xt = np.empty((ntr, DIM), np.float32)
    p = 0
    for s in range(0, R, STEP):
        e = min(R, s + STEP)
        j = samp[(samp >= s) & (samp < e)]
        if len(j):
            X = st.read_rows(s, e, buf)
            xt[p:p + len(j)] = X[j - s]
            p += len(j)
    assert p == ntr
    read_s = time.perf_counter() - t
    index = faiss.index_factory(DIM, f"IVF{nlist},PQ32")
    t = time.perf_counter()
    index.train(xt)
    train_s = time.perf_counter() - t
    del xt
    add_s = 0.0
    for s in range(0, R, STEP):
        e = min(R, s + STEP)
        t = time.perf_counter()
        X = st.read_rows(s, e, buf).astype(np.float32)
        sid = np.searchsorted(st.off, np.arange(s, e), side="right") - 1
        read_s += time.perf_counter() - t
        t = time.perf_counter()
        index.add_with_ids(X, sid.astype(np.int64))
        add_s += time.perf_counter() - t
    assert index.ntotal == R
    peak = peak_rss_gb()
    faiss.write_index(index, a.index)
    nb = os.path.getsize(a.index)
    info.update(read_s=read_s, train_s=train_s, add_s=add_s, build_s=train_s + add_s, index_bytes=nb,
                index_bytes_per_site=nb / N, index_bytes_per_residue=nb / R, peak_rss_gb=peak,
                store_bytes_per_site=R * DIM * 2 / N)
    jdump(info, os.path.join(odir(a.db), f"build_{a.tag}.json"))
    log(info)


def cmd_stage1(a):
    """Stage-1 candidates (first 3,000) of every query."""
    set_threads(a.threads)
    index = load_index(a.index)
    qres, qoff = load_queries()
    N = int(np.load(os.path.join(ldir(a.db), "off.npy")).shape[0] - 1)
    info = dict(db=a.db, threads=a.threads, N=N, parallel_mode=1, **host_info())
    lists, us = [], []
    t0 = time.perf_counter()
    for i in range(Q):
        D, I = index.search(qres[qoff[i]:qoff[i + 1]], R_NN)
        U = candidates(D, I, N)
        us.append(len(U))
        lists.append(U[:KEEP].astype(np.int32))
    info.update(search_s=time.perf_counter() - t0, union_median=float(np.median(us)), peak_rss_gb=peak_rss_gb())
    save_lists(os.path.join(odir(a.db), "cands_np16.npz"), lists, union_size=np.array(us, np.int64))
    if a.ref_cands:
        info["check"] = same_lists(lists, a.ref_cands)
    jdump(info, os.path.join(odir(a.db), "stage1.json"))
    log({k: v for k, v in info.items() if k in ("db", "search_s", "union_median", "check")})


def cmd_recall(a):
    """recall@k = |exact top-k inside the first K' candidates| / k; top{k}_n_syn = synthetic sites in the exact top-k."""
    od = odir(a.db)
    gi = np.load(os.path.join(od, "gt_top1000.npz"))["idx"]
    c = np.load(os.path.join(od, "cands_np16.npz"))
    ids, off, us = c["ids"], c["off"], c["union_size"]
    syn = DBS[a.db]["kind"] in ("chimera", "copy")
    rows = []
    for i in range(Q):
        U = ids[off[i]:off[i + 1]].astype(np.int64)
        for kp in KP:
            sC = set(U[:kp].tolist())
            r = dict(q=i, nprobe=NPROBE, Kp=kp, Kp_eff=len(U[:kp]), union=int(us[i]))
            for k in (1, 10, 100):
                gk = gi[i, :k]
                r[f"recall@{k}"] = float(np.mean([x in sC for x in gk.tolist()]))
                if syn:
                    r[f"top{k}_n_syn"] = int((gk >= N_REAL).sum())
            rows.append(r)
    pq = pd.DataFrame(rows)
    pq.to_csv(os.path.join(od, "recall_per_query.csv.gz"), index=False)
    log(a.db, pq.groupby("Kp")[["recall@1", "recall@10", "recall@100"]].mean().round(4).to_dict("index"))


def cmd_time(a):
    """One query at a time (all queries after 5 untimed warm-up queries), FAISS and torch threads = --threads,
    candidate reads with a pool of --threads threads (1: sequential). Per query and K':
      t_s1    FAISS search (r = 256, nprobe 16) + candidate scoring
      t_read  pread of the candidate sites from the fp16 store, its page cache dropped before every read (cold)
      t_rr    fp16 -> fp32, exact Chamfer of the candidates, sort, top 100
      total = t_s1 + t_read + t_rr"""
    set_threads(a.threads)
    t = time.perf_counter()
    index = load_index(a.index)
    load_s = time.perf_counter() - t
    st = Store(a.db)
    os.posix_fadvise(st.fd, 0, 0, os.POSIX_FADV_RANDOM)
    st.drop_cache()
    N = len(st.off) - 1
    rr = Reranker(st, threads=a.threads)
    qres, qoff = load_queries()
    info = dict(db=a.db, threads=a.threads, N=N, R=st.R, parallel_mode=1, index_load_s=load_s,
                rss_after_load_gb=status_gb("VmRSS"), **host_info())

    def one(i, lists):
        q = qres[qoff[i]:qoff[i + 1]]
        t0 = time.perf_counter()
        D, I = index.search(q, R_NN)
        U = candidates(D, I, N)
        t1 = time.perf_counter()
        if lists is not None:
            lists.append(U[:KEEP].astype(np.int32))
        out = []
        for kp in KP:
            Cc = U[:kp]
            st.drop_cache()
            t2 = time.perf_counter()
            o, loc = rr.read(Cc)
            t3 = time.perf_counter()
            rr.rerank(q, o, loc)
            t4 = time.perf_counter()
            out.append(dict(q=i, nprobe=NPROBE, Kp=kp, Kp_eff=len(Cc), t_s1=t1 - t0, t_read=t3 - t2, t_rr=t4 - t3,
                            total=(t1 - t0) + (t4 - t2), n_res=int(loc[-1])))
        return out

    for i in range(5):
        one(i, None)
    rows, lists = [], []
    t = time.perf_counter()
    for i in range(Q):
        rows += one(i, lists)
    info.update(wall_s=time.perf_counter() - t, peak_rss_gb=peak_rss_gb())
    pd.DataFrame(rows).to_csv(os.path.join(odir(a.db), f"search_T{a.threads}.csv.gz"), index=False)
    if a.ref_cands:
        info["check"] = same_lists(lists, a.ref_cands)
    jdump(info, os.path.join(odir(a.db), f"search_T{a.threads}.json"))
    log({k: v for k, v in info.items() if k in ("db", "threads", "wall_s", "check")})


def cmd_exact(a):
    """Exhaustive scan, one query at a time after 2 warm-up queries; the time covers scoring every site and the top-100
    selection. Queries: all, or sub48 = default_rng(0).permutation(481)[:48]. top100_overlap: first 3 timed queries
    against the exact ground truth."""
    set_threads(a.threads)
    qres, qoff = load_queries()
    t = time.perf_counter()
    N, blocks = fp16_blocks(a.db)
    info = dict(db=a.db, threads=a.threads, query_set=a.queries, layout="fp16", N=N, layout_s=time.perf_counter() - t,
                rss_after_layout_gb=status_gb("VmRSS"), **host_info())
    sel = np.arange(Q) if a.queries == "all" else np.random.default_rng(0).permutation(Q)[:48]
    gp = os.path.join(odir(a.db), "gt_top1000.npz")
    g = np.load(gp)["idx"] if os.path.exists(gp) else None
    for i in sel[:2]:
        exact_scan_one(qres, qoff, int(i), N, blocks)
    rows, checks = [], []
    for k, i in enumerate(sel):
        t0 = time.perf_counter()
        d, top = exact_scan_one(qres, qoff, int(i), N, blocks)
        rows.append(dict(q=int(i), t=time.perf_counter() - t0, nq=int(qoff[i + 1] - qoff[i])))
        if k < 3 and g is not None:
            checks.append(dict(q=int(i), top100_overlap=len(set(top.tolist()) & set(g[i, :100].tolist())) / 100))
    tag = f"exact_T{a.threads}_{a.queries}"
    pd.DataFrame(rows).to_csv(os.path.join(odir(a.db), tag + ".csv"), index=False)
    ts = np.array([r["t"] for r in rows])
    info.update(n_queries=len(rows), median_s=float(np.median(ts)), p95_s=float(np.percentile(ts, 95)),
                mean_s=float(ts.mean()), checks=checks, peak_rss_gb=peak_rss_gb())
    jdump(info, os.path.join(odir(a.db), tag + ".json"))
    log(info)


def cmd_gpu(a):
    """Databases (the fp16 store of generate) held in GPU memory as size-class blocks; per query (all after 2 warm-up
    queries): fp16 -> fp32 per step, dense Chamfer kernel, top 100 (torch.topk), synchronised before the clock stops.
    --stream-db is also kept in pinned host memory and streamed to the GPU ten times per query (host -> device copy +
    the same kernel): the exhaustive-scan cost depends only on the residue count, and 10^7 sites (~150 GB of fp16
    blocks) do not fit in 24 GB."""
    dev = torch.device(a.device)
    CUDA = dev.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    set_threads(a.threads)
    qres, qoff = load_queries()
    info0 = dict(gpu=torch.cuda.get_device_name(0) if CUDA else "cpu", **host_info())

    @torch.no_grad()
    def scan(i, N, blocks, stream_passes=0, max_cols=1 << 20):
        qX, qm, qn = (t.to(dev) for t in pad_queries(qres, qoff, np.array([i])))
        qf = qX.reshape(-1, DIM)
        q2 = (qf * qf).sum(1)
        out = torch.empty(N, device=dev)
        for _ in range(max(1, stream_passes)):
            for ids, X, m, n in blocks:
                P = X.shape[1]
                step = max(1, max_cols // P)
                for s in range(0, X.shape[0], step):
                    x = X[s:s + step]
                    mm, nn = m[s:s + step], n[s:s + step]
                    if stream_passes:
                        x, mm, nn = x.to(dev, non_blocking=True), mm.to(dev, non_blocking=True), nn.to(dev, non_blocking=True)
                    x = x.float()
                    sb = x.shape[0]
                    xf = x.reshape(sb * P, DIM)
                    D = torch.addmm((xf * xf).sum(1).unsqueeze(0), qf, xf.T, beta=1.0, alpha=-2.0)
                    D.add_(q2.unsqueeze(1)).clamp_(min=0.0)
                    D = D.view(1, qX.shape[1], sb, P)
                    a2b = (D.amin(3) * qm[:, :, None]).sum(1) / qn[:, None]
                    b2a = (D.amin(1) * mm[None]).sum(2) / nn[None]
                    out[ids[s:s + sb].to(dev, non_blocking=True)] = (a2b + b2a)[0]
        top = torch.topk(out, 100, largest=False).indices
        if CUDA:
            torch.cuda.synchronize()
        return top.cpu().numpy()

    def timed(db, N, blocks, tag, stream=0):
        gp = os.path.join(odir(db), "gt_top1000.npz")
        g = np.load(gp)["idx"] if os.path.exists(gp) else None
        for i in range(2):
            scan(i, N, blocks, stream)
        rows, checks = [], []
        for i in range(Q):
            t0 = time.perf_counter()
            top = scan(i, N, blocks, stream)
            rows.append(dict(q=i, t=time.perf_counter() - t0))
            if i < 3 and g is not None and not stream:
                checks.append(dict(q=i, top100_overlap=len(set(top.tolist()) & set(g[i, :100].tolist())) / 100))
        ts = np.array([r["t"] for r in rows])
        info = dict(db=db, mode=tag, N=N if not stream else N * stream, N_db=N, stream_passes=stream, n_queries=len(rows),
                    median_s=float(np.median(ts)), p95_s=float(np.percentile(ts, 95)), mean_s=float(ts.mean()),
                    checks=checks, gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if CUDA else 0,
                    peak_rss_gb=peak_rss_gb(), **info0)
        pd.DataFrame(rows).to_csv(os.path.join(odir(db), f"exact_gpu{tag}.csv"), index=False)
        jdump(info, os.path.join(odir(db), f"exact_gpu{tag}.json"))
        log(info)

    for db in a.dbs:
        N, blocks = fp16_blocks(db, pin=(db == a.stream_db) and CUDA)
        blocks = [(torch.from_numpy(ids), X, m, n) for ids, X, m, n in blocks]
        gb = [(ids.to(dev), X.to(dev), m.to(dev), n.to(dev)) for ids, X, m, n in blocks]
        timed(db, N, gb, "")
        del gb
        if CUDA:
            torch.cuda.empty_cache()
        if db == a.stream_db:
            timed(db, N, blocks, "_stream10x", stream=10)
        del blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("clusters", help=cmd_clusters.__doc__.split("\n")[0])
    p.add_argument("--db-sites", default=os.path.join(ANNOT, "db_sites.tsv"),
                   help="site annotation of the database sites (default $LENSEEK_OUT/cache/site_annot/db_sites.tsv)")
    p.add_argument("--clusters50", default=os.path.join(ANNOT, "db50_cluster.tsv"),
                   help="MMseqs2 50 %% clusters of the database chains, rep and member "
                        "(default $LENSEEK_OUT/cache/site_annot/db50_cluster.tsv)")
    p.add_argument("--db-paths", default=os.path.join(os.environ.get("LENSEEK_EVAL_DIR", ""), "db_paths.txt"),
                   help="database site order (default $LENSEEK_EVAL_DIR/db_paths.txt)")
    p.add_argument("--out", default=CLUSTERS, help="output .npy (default $LENSEEK_SCALING_CLUSTERS)")
    p = sp.add_parser("params", help=cmd_params.__doc__.split("\n")[0])
    p.add_argument("--threads", type=int, default=4)
    for name, fn in (("generate", cmd_generate), ("index", cmd_index), ("stage1", cmd_stage1), ("recall", cmd_recall),
                     ("time", cmd_time), ("exact", cmd_exact)):
        p = sp.add_parser(name, help=fn.__doc__.split("\n")[0])
        p.add_argument("db", choices=list(DBS))
        if name != "recall":
            p.add_argument("--threads", type=int, default=16)
        if name in ("index", "stage1", "time"):
            p.add_argument("--index", help="index file (default <node-local scratch>/<db>/index.faiss)")
        if name in ("stage1", "time"):
            p.add_argument("--ref-cands", help="cands_np16.npz of another run (e.g. one-thread): count identical lists")
    sp.choices["generate"].add_argument("--store-only", action="store_true", help="store only, no ground truth")
    sp.choices["index"].add_argument("--tag", default="acc", help="label of build_<tag>.json (acc, time)")
    sp.choices["exact"].add_argument("--queries", choices=["all", "sub48"], default="all")
    p = sp.add_parser("gpu", help=cmd_gpu.__doc__.split("\n")[0])
    p.add_argument("--dbs", nargs="+", default=["real", "real150k", "chim3e5", "chim1e6"])
    p.add_argument("--stream-db", default="chim1e6")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    if getattr(a, "index", 0) is None:
        a.index = os.path.join(ldir(a.db), "index.faiss")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
