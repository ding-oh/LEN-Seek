"""Single-vector baselines and the residue two-stage retrieval on the search database (Supplementary Section S8 and
Table S15 A; analysis/table3_two_stage.py makes the table).

Can one vector per site, searched with a standard approximate nearest-neighbour library (FAISS), stand in for the
exact Chamfer ranking? The reference is the exact Chamfer top-k of every query over the search database; recall@k is
the fraction of it that a method returns (after an exact Chamfer re-rank of K' candidates, the fraction inside the K'
candidates). Per-query times are taken one query at a time on one thread.

Subcommands, in run order (outputs in $LENSEEK_OUT/single_vector/):
  prep       valid residues of every site, stored contiguously (CSR: res[off[i]:off[i+1]] = site i), and the exact
             top 1,000 of every query (ties by index)                     -> cache/, data_stats.tsv
  exact      exhaustive Chamfer scan on the CPU: all queries against dist.npy, and the one-thread time per query on 5
             queries (sites grouped into size classes, wrap-padded)       -> exact_scan.tsv
  pooling    one pooled vector per site (mean, max, mean||max; Euclidean or cosine; exact search): recall of the pooled
             top-k and of the pooled top-K' after an exact re-rank        -> pooling.tsv
  faiss      the pooled-vector rows of Table S15 A: exact search over mean and over mean||max vectors (FAISS Flat), and
             the mean vectors in HNSW32 (efSearch 1,000) and IVF1024,PQ32 (nprobe 64) as the first stage of an exact
             re-rank of K' = 1,000          -> faiss_pooled_single.tsv, faiss_pooled_2stage.tsv, rerank_latency.tsv
  residue    two-stage search with an IVF4096,PQ32 index of all residue vectors (r = 256 neighbours per query residue,
             16 lists probed; candidates as in scripts/scaling.py), exact re-rank of K' = 1,000 / 3,000
                                                                          -> twostage_residue.tsv
  fde        fixed-dimensional encoding (FDE) of the MUVERA type adapted to the squared-L2 Chamfer distance, exact
             inner-product search over the full FDE (R = 4 repetitions, k = 7 SimHash bits, d_proj = 16) -> fde.tsv
  fde-faiss  the same FDE balanced, centred and reduced by PCA to 1,024 dimensions, in FAISS indexes -> fde_faiss.tsv
Input: $LENSEEK_EVAL_DIR with db_mu.npy (N, 65, 256), db_mask.npy, q_mu.npy, q_mask.npy and dist.npy (exact Chamfer,
queries x database sites) of one model (scripts/evaluate.py). scripts/scaling.py reads cache/.

usage: python scripts/single_vector.py prep; ... exact; ... pooling; ... faiss; ... residue; ... fde; ... fde-faiss
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (bucketize, candidates, chamfer_scan, die, log, peak_rss_gb, rerank, set_threads,  # noqa: E402
                      topk_stable)

OUT = os.path.join(os.environ.get("LENSEEK_OUT", "out"), "single_vector")
CACHE = os.path.join(OUT, "cache")
KS = (1, 10, 100)
NTHREADS = 8


# ------------------------------------------------------------------------------------------------- helpers
def eval_dir():
    d = os.environ.get("LENSEEK_EVAL_DIR")
    if not d or not os.path.isdir(d):
        raise SystemExit("LENSEEK_EVAL_DIR must point to one model's evaluation outputs "
                         "(db_mu.npy, db_mask.npy, q_mu.npy, q_mask.npy, dist.npy)")
    return d


def check_inputs(cmd):
    """Stop with one '[error]' line at the first input of a subcommand that does not exist: the evaluation outputs of
    one model for prep and exact, the cache of prep for the other subcommands."""
    if cmd in ("prep", "exact"):
        src = eval_dir()
        for f in ("db_mu.npy", "db_mask.npy", "q_mu.npy", "q_mask.npy", "dist.npy") if cmd == "prep" else ("dist.npy",):
            if not os.path.exists(f"{src}/{f}"):
                die(f"$LENSEEK_EVAL_DIR: {src}/{f} does not exist (written by scripts/evaluate.py)")
    if cmd != "prep":
        for f in ("db_res.npy", "db_off.npy", "q_res.npy", "q_off.npy") + (() if cmd == "exact" else ("gt_idx.npy",)):
            if not os.path.exists(f"{CACHE}/{f}"):
                die(f"{CACHE}/{f} does not exist (written by `python scripts/single_vector.py prep` below "
                    "$LENSEEK_OUT)")


def load_csr(which="db"):
    return np.load(f"{CACHE}/{which}_res.npy", mmap_mode="r"), np.load(f"{CACHE}/{which}_off.npy")


def load_norms():
    """Squared L2 norm of every database residue (computed once, stored next to the cache)."""
    p = f"{CACHE}/db_norm.npy"
    if not os.path.exists(p):
        res, _ = load_csr()
        nr = np.empty(res.shape[0], dtype=np.float32)
        for s in range(0, res.shape[0], 262144):
            b = np.asarray(res[s:s + 262144])
            nr[s:s + len(b)] = np.einsum("ij,ij->i", b, b)
        np.save(p, nr)
    return np.load(p)


def rerank_csr(q, res, off, norms, cand):
    """rerank of candidate sites gathered from the CSR cache (all candidates returned)."""
    cand = np.asarray(cand, dtype=np.int64)
    n = off[cand + 1] - off[cand]
    loc = np.zeros(len(cand) + 1, dtype=np.int64)
    np.cumsum(n, out=loc[1:])
    idx = np.repeat(off[cand] - loc[:-1], n) + np.arange(loc[-1])
    return rerank(q, cand, torch.from_numpy(np.asarray(res[idx])), loc, dn=torch.from_numpy(norms[idx]))


def pooled(res, off, kind):
    """One vector per site: mean, max or meanmax (concatenated) of its residue vectors; '_n' = L2-normalised (each half
    for meanmax, then / sqrt 2)."""
    base = kind.replace("_n", "")
    mean = np.add.reduceat(res, off[:-1], axis=0) / np.diff(off)[:, None]
    if base == "mean":
        v = mean
    else:
        mx = np.maximum.reduceat(res, off[:-1], axis=0)
        v = mx if base == "max" else np.concatenate([mean, mx], 1)
    v = v.astype(np.float32)
    if kind.endswith("_n"):
        if base == "meanmax":
            h = v.shape[1] // 2
            v[:, :h] /= np.linalg.norm(v[:, :h], axis=1, keepdims=True) + 1e-12
            v[:, h:] /= np.linalg.norm(v[:, h:], axis=1, keepdims=True) + 1e-12
            v /= np.sqrt(2.0)
        else:
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(v)


def contain(I, gi, kp, k):
    """Mean fraction of the exact top-k inside the first kp entries of each row of I."""
    return float(np.mean([len(set(np.asarray(I[i][:kp]).tolist()) & set(gi[i, :k].tolist())) / k for i in range(len(gi))]))


def timed_search(index, Y, K):
    """One query at a time on one thread: (results (Q, K), median ms)."""
    set_threads(1)
    ts, Is = [], []
    for i in range(len(Y)):
        t = time.perf_counter()
        _, I = index.search(Y[i:i + 1], K)
        ts.append(time.perf_counter() - t)
        Is.append(I[0])
    set_threads(NTHREADS)
    return np.stack(Is), 1e3 * np.median(ts)


def nbytes(index):
    import faiss
    return int(faiss.serialize_index(index).nbytes)


def write_tsv(name, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{OUT}/{name}", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in (r.get(c, "") for c in cols)) + "\n")


# ------------------------------------------------------------------------------------------------- subcommands
def cmd_prep(a):
    src = eval_dir()
    os.makedirs(CACHE, exist_ok=True)

    def csr(prefix):
        mask = np.load(f"{src}/{prefix}_mask.npy")
        n = mask.sum(1).astype(np.int64)
        assert all(mask[i, :n[i]].all() for i in range(len(n))), "mask is not a prefix"
        off = np.zeros(len(n) + 1, dtype=np.int64)
        np.cumsum(n, out=off[1:])
        out = np.lib.format.open_memmap(f"{CACHE}/{prefix}_res.npy", mode="w+", dtype=np.float32, shape=(int(off[-1]), 256))
        for s in range(0, len(n), 4096):
            mu = np.load(f"{src}/{prefix}_mu.npy", mmap_mode="r")    # fresh map per chunk keeps the RSS bounded
            out[off[s]:off[min(s + 4096, len(n))]] = np.asarray(mu[s:s + 4096])[mask[s:s + 4096]]
            del mu
        out.flush()
        np.save(f"{CACHE}/{prefix}_off.npy", off)
        return n

    n_db, n_q = csr("db"), csr("q")
    dist = np.load(f"{src}/dist.npy", mmap_mode="r")
    Q, N = dist.shape
    gi = np.empty((Q, 1000), dtype=np.int32)
    gd = np.empty((Q, 1000), dtype=np.float32)
    for i in range(Q):
        row = np.asarray(dist[i])
        gi[i] = topk_stable(row, 1000)
        gd[i] = row[gi[i]]
    np.save(f"{CACHE}/gt_idx.npy", gi)
    np.save(f"{CACHE}/gt_d.npy", gd)
    write_tsv("data_stats.tsv", [dict(item=k, value=v) for k, v in (
        ("db_sites", N), ("db_residues", int(n_db.sum())), ("db_res_per_site_mean", float(n_db.mean())),
        ("db_res_per_site_median", float(np.median(n_db))), ("queries", Q), ("q_res_per_site_mean", float(n_q.mean())),
        ("gt_top1_cd_median", float(np.median(gd[:, 0]))), ("gt_top10_cd_median", float(np.median(gd[:, 9]))),
        ("gt_top100_cd_median", float(np.median(gd[:, 99]))), ("gt_top1000_cd_median", float(np.median(gd[:, 999]))))])


def cmd_exact(a):
    res, off = load_csr()
    qres, qoff = load_csr("q")
    qres = np.ascontiguousarray(qres)
    dist = np.load(f"{eval_dir()}/dist.npy", mmap_mode="r")
    Q, N = dist.shape
    t = time.perf_counter()
    B = [b for s in range(0, N, 50000) for b in bucketize(res, off[s:min(N, s + 50000) + 1], base=s)]
    t_layout = time.perf_counter() - t
    del res
    t = time.perf_counter()
    D = chamfer_scan(qres, qoff, B, np.empty((Q, N), dtype=np.float32), qbatch=64)
    tb = time.perf_counter() - t
    ref = np.asarray(dist)
    err = np.abs(D - ref)
    agree = {k: np.mean([len(set(topk_stable(D[i], k)) & set(topk_stable(ref[i], k))) / k for i in range(Q)]) for k in KS}
    rows = [dict(mode="dense_layout_build", queries="", threads=NTHREADS, total_s=t_layout, s_per_query=""),
            dict(mode="dense_batch64", queries=Q, threads=NTHREADS, total_s=tb, s_per_query=tb / Q,
                 **{"max_abs_diff_vs_dist.npy": float(err.max()),
                    "max_rel_diff": float((err / np.maximum(np.abs(ref), 1e-6)).max())},
                 **{f"top{k}_agree": agree[k] for k in KS})]
    del D, ref, err
    sel = np.random.default_rng(0).choice(Q, 30, replace=False)[:5]
    set_threads(1)
    ts = []
    for i in sel:
        q = qres[qoff[i]:qoff[i + 1]]
        t = time.perf_counter()
        chamfer_scan(q, np.array([0, len(q)]), B, np.empty((1, N), dtype=np.float32), qbatch=1)
        ts.append(time.perf_counter() - t)
    set_threads(NTHREADS)
    rows.append(dict(mode="dense_single_1thread", queries=len(ts), threads=1, total_s=float(np.sum(ts)),
                     s_per_query=float(np.median(ts))))
    write_tsv("exact_scan.tsv", rows)
    log(rows[1:], f"peak RSS {peak_rss_gb():.2f} GB")


def cmd_pooling(a):
    """recall@k = |pooled top-k & exact top-k| / k; cand{k}@{K'} = fraction of the exact top-k inside the pooled top-K'
    (= recall@k after the exact re-rank of K'). Cosine = Euclidean distance between normalised vectors."""
    res, off = load_csr()
    qres, qoff = load_csr("q")
    gi = np.load(f"{CACHE}/gt_idx.npy")
    Q, N = len(qoff) - 1, len(off) - 1
    KP = (10, 100, 300, 1000, 3000, 10000)
    cache, rows = {}, []
    for kind, metric in (("mean", "l2"), ("mean", "cos"), ("max", "l2"), ("max", "cos"), ("meanmax", "l2"),
                         ("meanmax", "cos"), ("meanmax_n", "l2")):
        if kind not in cache:
            cache[kind] = (pooled(np.asarray(res), off, kind), pooled(np.asarray(qres), qoff, kind))
        Xt, Yt = (torch.from_numpy(v) for v in cache[kind])
        if metric == "cos":
            Xt = Xt / Xt.norm(dim=1, keepdim=True).clamp(min=1e-12)
            Yt = Yt / Yt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        D = torch.cdist(Yt, Xt).pow(2).numpy()
        top = np.stack([np.lexsort((np.arange(N), D[i]))[:max(KP)] for i in range(Q)])
        r = dict(pooling=kind, metric=metric, dim=Xt.shape[1])
        r.update({f"recall@{k}": float(np.mean([len(set(top[i, :k]) & set(gi[i, :k])) / k for i in range(Q)])) for k in KS})
        r.update({f"cand{k}@{kp}": contain(top, gi, kp, k) for k in KS for kp in KP if kp >= k})
        rows.append(r)
        log(kind, metric, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    write_tsv("pooling.tsv", rows)


def cmd_faiss(a):
    """faiss_pooled_single.tsv: build time and size of every index, and the top 10 / 100 of the Flat indexes (time per
    query). Two stage: the index returns K' candidates; total time = stage-1 time + the median time of an exact
    re-rank of K' candidates (gather, Chamfer, sort; 100 queries, candidates from the exact pooled-mean top K')."""
    import faiss
    res, off = load_csr()
    qres, qoff = load_csr("q")
    gi = np.load(f"{CACHE}/gt_idx.npy")
    Q, N = len(qoff) - 1, len(off) - 1
    kp = 1000
    flat = faiss.IndexFlatL2(256)
    flat.add(pooled(np.asarray(res), off, "mean"))
    _, Iflat = flat.search(pooled(np.asarray(qres), qoff, "mean"), kp)
    norms = load_norms()
    qsel = np.random.default_rng(0).choice(Q, 100, replace=False)
    set_threads(1)
    tt = []
    for i in qsel:
        q = np.array(qres[qoff[i]:qoff[i + 1]])
        t = time.perf_counter()
        rerank_csr(q, res, off, norms, Iflat[i, :kp])
        tt.append(time.perf_counter() - t)
    rr_ms = 1e3 * np.median(tt)
    write_tsv("rerank_latency.tsv", [{"K'": kp, "total_ms_median": rr_ms, "total_ms_mean": 1e3 * np.mean(tt),
                                      "threads": 1}])
    set_threads(NTHREADS)
    single, two = [], []
    for var, name, param in (("mean_l2", "Flat", None), ("meanmax_l2", "Flat", None),
                             ("mean_l2", "HNSW32", 1000), ("mean_l2", "IVF1024,PQ32", 64)):
        kind = var.rsplit("_", 1)[0]
        X, Y = pooled(np.asarray(res), off, kind), pooled(np.asarray(qres), qoff, kind)
        d = X.shape[1]
        t = time.perf_counter()
        idx = faiss.index_factory(d, name)
        idx.train(X)
        idx.add(X)
        tb = time.perf_counter() - t
        mb = nbytes(idx)
        row = dict(vector=var, dim=d, index=name, param="", build_s=tb, index_MB=mb / 1e6, bytes_per_site=mb / N)
        if name == "Flat":
            I10, med10 = timed_search(idx, Y, 10)
            I, med = timed_search(idx, Y, 100)
            row.update(lat_ms_median_k10=med10, lat_ms_median_k100=med,
                       **{"recall@1": contain(I10, gi, 1, 1), "recall@10": contain(I10, gi, 10, 10),
                          "recall@100": contain(I, gi, 100, 100)})
        else:
            if name.startswith("HNSW"):
                idx.hnsw.efSearch = param
                ps = f"efSearch={param}"
            else:
                idx.nprobe = param
                ps = f"nprobe={param}"
            I, med = timed_search(idx, Y, kp)
            two.append({"vector": var, "index": name, "param": ps, "K'": kp, "index_MB": mb / 1e6,
                        "stage1_ms_median": med, "rerank_ms_median": rr_ms, "total_ms_median": med + rr_ms,
                        **{f"recall@{k}": contain(I, gi, kp, k) for k in KS}})
        single.append(row)
        log(var, name, f"build {tb:.1f}s, {mb / 1e6:.1f} MB")
        del idx
    write_tsv("faiss_pooled_single.tsv", single)
    write_tsv("faiss_pooled_2stage.tsv", two)


def cmd_residue(a):
    """Index: IVF4096,PQ32 trained on 262,144 residues (default_rng(0)), every residue added. Per query site, its
    residues are searched as one batch; the hit residues' sites are ordered by `candidates`; the first K' are
    re-ranked by exact Chamfer. Recall on all queries; end-to-end time (search, scoring, gather, Chamfer, sort) on one
    thread over 100 queries."""
    import faiss
    res, off = load_csr()
    qres, qoff = load_csr("q")
    qres = np.ascontiguousarray(qres)
    gi = np.load(f"{CACHE}/gt_idx.npy")
    Q, N, R = len(qoff) - 1, len(off) - 1, int(off[-1])
    site_of = np.repeat(np.arange(N, dtype=np.int64), np.diff(off))
    norms = load_norms()
    qsel = set(np.random.default_rng(0).choice(Q, 100, replace=False).tolist())
    set_threads(NTHREADS)
    t = time.perf_counter()
    idx = faiss.index_factory(256, "IVF4096,PQ32")
    idx.train(np.ascontiguousarray(res[np.sort(np.random.default_rng(0).choice(R, 262144, replace=False))]))
    for s in range(0, R, 262144):
        idx.add(np.ascontiguousarray(res[s:s + 262144]))
    tb = time.perf_counter() - t
    mb = nbytes(idx)
    r_nn, idx.nprobe = 256, 16
    KPS = (1000, 3000)
    cont = {kp: {k: [] for k in KS} for kp in KPS}
    size_u, t_s, t_tot = [], [], {kp: [] for kp in KPS}
    set_threads(1)
    for i in range(Q):
        q = qres[qoff[i]:qoff[i + 1]]
        t0 = time.perf_counter()
        D, I = idx.search(q, r_nn)
        U = candidates(D, np.where(I >= 0, site_of[np.maximum(I, 0)], -1), N)
        t1 = time.perf_counter()
        t_s.append(t1 - t0)
        size_u.append(len(U))
        for kp in KPS:
            sC = set(U[:kp].tolist())
            for k in KS:
                cont[kp][k].append(len(sC & set(gi[i, :k].tolist())) / k)
            if i in qsel:
                t2 = time.perf_counter()
                rerank_csr(q, res, off, norms, U[:kp])
                t_tot[kp].append((t1 - t0) + time.perf_counter() - t2)
    set_threads(NTHREADS)
    rows = [{"index": "IVF4096,PQ32", "param": f"r={r_nn},nprobe=16", "K'": kp,
             "K'_effective_mean": float(np.mean([min(u, kp) for u in size_u])),
             "union_size_median": float(np.median(size_u)),
             "index_MB": mb / 1e6, "bytes_per_site": mb / N, "build_s": tb, "stage1_ms_median": 1e3 * np.median(t_s),
             "total_ms_median": 1e3 * np.median(t_tot[kp]), "total_ms_mean": 1e3 * np.mean(t_tot[kp]),
             **{f"recall@{k}": float(np.mean(cont[kp][k])) for k in KS}} for kp in KPS]
    write_tsv("twostage_residue.tsv", rows)
    log(rows, f"peak RSS {peak_rss_gb():.2f} GB")


def fde_encode(X, o, is_query, params, rng_seed=0):
    """FDE of the sites of a CSR run, one vector per site, such that <query FDE, site FDE> = const(query) - Chamfer
    in expectation. min_j |q - p_j|^2 = |q|^2 - max_j (2<q, p_j> - |p_j|^2), so each Chamfer direction is a sum of
    maximum inner products between augmented vectors; FDE approximates sum_i max_j <x_i, y_j> by <sum of x per SimHash
    bucket, mean of y per bucket>, over R random SimHash partitions (B = 2^k buckets; an empty bucket on the mean side
    takes the residue whose code is nearest in Hamming distance), the 256-d vectors projected to d_proj dimensions (the
    |p|^2 terms stay exact). Both directions and the site constant mean |p|^2 are packed into one vector."""
    R, k, dp = params
    B = 2 ** k
    rng = np.random.default_rng(rng_seed)
    S = len(o) - 1
    n = np.diff(o)
    site = np.repeat(np.arange(S), n)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    sq = (Xt * Xt).sum(1)
    blk = B * (dp + 1)
    out = np.empty((S, 2 * R * blk + 1), dtype=np.float32)
    ham = torch.tensor([[bin(a ^ b).count("1") for b in range(B)] for a in range(B)], dtype=torch.int16)
    maxn = int(n.max())
    pad_idx = torch.from_numpy(o[:-1, None] + np.minimum(np.arange(maxn)[None, :], n[:, None] - 1))  # (S, maxn)
    pad_ok = torch.from_numpy(np.arange(maxn)[None, :] < n[:, None])
    nn = torch.from_numpy(n.astype(np.float32))[:, None]
    for t in range(R):
        G = rng.standard_normal((X.shape[1], k)).astype(np.float32)
        P = (rng.standard_normal((X.shape[1], dp)).astype(np.float32) / np.sqrt(dp)) if dp else None
        codes = torch.from_numpy((((X @ G) > 0).astype(np.int64) @ (1 << np.arange(k))))
        Xp = Xt @ torch.from_numpy(P.astype(np.float32)) if dp else Xt
        key = torch.from_numpy(site) * B + codes
        cnt = torch.zeros(S * B).index_add_(0, key, torch.ones(len(key))).view(S, B)
        sum_p = torch.zeros(S * B, Xp.shape[1]).index_add_(0, key, Xp).view(S, B, -1)
        sum_sq = torch.zeros(S * B).index_add_(0, key, sq).view(S, B)
        mean_p = sum_p / cnt.clamp(min=1)[..., None]
        mean_sq = sum_sq / cnt.clamp(min=1)
        empty = cnt == 0
        if bool(empty.any()):
            hd = ham[codes[pad_idx]]                      # (S, maxn, B)
            hd = torch.where(pad_ok[..., None], hd, torch.full_like(hd, 99))
            best = pad_idx.gather(1, hd.argmin(1))        # (S, B) residue index (first minimum)
            si, bi = torch.nonzero(empty, as_tuple=True)
            mean_p[si, bi] = Xp[best[si, bi]]
            mean_sq[si, bi] = sq[best[si, bi]]
        if is_query:   # a2b: sum role with [2x, 1]; b2a: mean role with [x, -|x|^2]
            a2b = torch.cat([2 * sum_p / nn[..., None], (cnt / nn)[..., None]], -1)
            b2a = torch.cat([mean_p, -mean_sq[..., None]], -1)
        else:          # a2b: mean role with [y, -|y|^2]; b2a: sum role with [2y, 1]
            a2b = torch.cat([mean_p, -mean_sq[..., None]], -1)
            b2a = torch.cat([2 * sum_p / nn[..., None], (cnt / nn)[..., None]], -1)
        out[:, 2 * t * blk:(2 * t + 1) * blk] = a2b.reshape(S, -1).numpy()
        out[:, (2 * t + 1) * blk:(2 * t + 2) * blk] = b2a.reshape(S, -1).numpy()
    if is_query:
        out[:, :-1] /= R
        out[:, -1] = -1
    else:
        out[:, -1] = np.add.reduceat(sq.numpy(), o[:-1]) / n
    return out


def cmd_fde(a):
    """Exact inner-product search over the full FDE of R repetitions, k SimHash bits and projection dimension
    d_proj."""
    res, off = load_csr()
    qres, qoff = load_csr("q")
    qres = np.ascontiguousarray(qres)
    gi = np.load(f"{CACHE}/gt_idx.npy")
    Q, N = len(qoff) - 1, len(off) - 1
    KP = (10, 100, 300, 1000, 3000, 10000)
    params = (a.R, a.k, a.d_proj)
    t = time.perf_counter()
    Yq = torch.from_numpy(fde_encode(qres, qoff, True, params))
    S = np.empty((Q, N), dtype=np.float32)
    for s in range(0, N, 10000):
        e = min(N, s + 10000)
        Xd = fde_encode(np.asarray(res[off[s]:off[e]]), off[s:e + 1] - off[s], False, params)
        S[:, s:e] = (Yq @ torch.from_numpy(Xd).T).numpy()
    tb = time.perf_counter() - t
    top = torch.topk(torch.from_numpy(S), max(KP), dim=1).indices.numpy()
    r = dict(R=params[0], k_simhash=params[1], d_proj=params[2], dim=Xd.shape[1], bytes_per_site_fp16=Xd.shape[1] * 2,
             encode_and_score_s=tb, **{f"recall@{k}": contain(top, gi, k, k) for k in KS},
             **{f"cand10@{kp}": contain(top, gi, kp, 10) for kp in KP},
             **{f"cand100@{kp}": contain(top, gi, kp, 100) for kp in KP[1:]})
    log(r)
    write_tsv("fde.tsv", [r])


def cmd_fde_faiss(a):
    """The raw FDE has very unequal coordinate scales, so before the projection each coordinate i is rescaled by
    s_i = sqrt(rms_query_i / rms_site_i) on the site side and 1 / s_i on the query side (inner products unchanged), and
    the mean site vector is subtracted (every score of a query shifts by the same constant); both statistics come from
    20,000 sites (default_rng(5)). Projection: the top --dim principal axes of these sites (torch.svd_lowrank,
    q = dim + 64, 3 iterations, torch seed 0). Indexes on the projected vectors (inner product): Flat, HNSW32,
    IVF1024,PQ(d/8), IVF1024,Flat, and HNSW32 in L2 after the MIPS-to-L2 augmentation x' = [x, sqrt(M^2 - |x|^2)],
    q' = [q, 0]. K' candidates, recall = fraction of the exact top-k inside them."""
    import faiss
    res, off = load_csr()
    qres, qoff = load_csr("q")
    qres = np.ascontiguousarray(qres)
    gi = np.load(f"{CACHE}/gt_idx.npy")
    N = len(off) - 1
    params, DFIN = (a.R, a.k, a.d_proj), [a.dim]
    t0 = time.perf_counter()
    Yq = fde_encode(qres, qoff, True, params)
    D = Yq.shape[1]
    samp = np.sort(np.random.default_rng(5).choice(N, 20000, replace=False))
    blocks = []
    for s in range(0, len(samp), 2000):
        sites = samp[s:s + 2000]
        n = off[sites + 1] - off[sites]
        o = np.zeros(len(sites) + 1, dtype=np.int64)
        np.cumsum(n, out=o[1:])
        blocks.append(fde_encode(np.asarray(res[np.repeat(off[sites] - o[:-1], n) + np.arange(o[-1])]), o, False, params))
    Xs = np.concatenate(blocks)
    rx = np.sqrt((Xs.astype(np.float64) ** 2).mean(0)) + 1e-12
    ry = np.sqrt((Yq.astype(np.float64) ** 2).mean(0)) + 1e-12
    scale = np.sqrt(ry / rx).astype(np.float32)
    mu = (Xs * scale).mean(0)
    torch.manual_seed(0)
    _, _, V = torch.svd_lowrank(torch.from_numpy(Xs * scale - mu), q=max(DFIN) + 64, niter=3)
    P = {d: np.ascontiguousarray(V[:, :d].numpy().astype(np.float32)) for d in DFIN}
    del Xs, V, blocks
    Yf = {d: np.ascontiguousarray((Yq / scale) @ P[d]) for d in DFIN}
    Xf = {d: np.empty((N, d), dtype=np.float32) for d in DFIN}
    for s in range(0, N, 5000):
        e = min(N, s + 5000)
        Xb = fde_encode(np.asarray(res[off[s]:off[e]]), off[s:e + 1] - off[s], False, params) * scale - mu
        for d in DFIN:
            Xf[d][s:e] = Xb @ P[d]
    t_enc = time.perf_counter() - t0
    log(f"FDE {params} dim {D}, PCA to {DFIN}; encode {t_enc:.0f}s")
    fde = f"R={a.R},k={a.k},d_proj={a.d_proj}"
    names = {"Flat": "Flat", "HNSW32": "HNSW32", "IVFPQ": "IVF1024,PQ{}", "IVFFlat": "IVF1024,Flat",
             "HNSW32-L2aug": "HNSW32 (L2, MIPS-augmented)"}
    rows = []
    for d in DFIN:
        X0, Y0 = Xf[d], Yf[d]
        nx = (X0.astype(np.float64) ** 2).sum(1)
        aug = (np.ascontiguousarray(np.concatenate([X0, np.sqrt(nx.max() - nx)[:, None].astype(np.float32)], 1)),
               np.ascontiguousarray(np.concatenate([Y0, np.zeros((len(Y0), 1), np.float32)], 1)))
        for key in ("Flat", "HNSW32", "IVFPQ", "IVFFlat", "HNSW32-L2aug"):
            name = names[key].format(d // 8)
            fac = "HNSW32" if key.startswith("HNSW") else name
            metric, (X, Y) = (faiss.METRIC_L2, aug) if key == "HNSW32-L2aug" else (faiss.METRIC_INNER_PRODUCT, (X0, Y0))
            t = time.perf_counter()
            idx = faiss.index_factory(X.shape[1], fac, metric)
            if not idx.is_trained:
                idx.train(X[np.random.default_rng(0).choice(N, 60000, replace=False)])
            idx.add(X)
            tb = time.perf_counter() - t
            nb = nbytes(idx)
            for p in [16, 64, 256] if key.startswith("IVF") else [None]:
                for kp in (10, 100, 300, 1000, 3000):
                    if key.startswith("HNSW"):
                        idx.hnsw.efSearch = max(kp, 64)
                        ps = f"efSearch={max(kp, 64)}"
                    else:
                        ps = "" if p is None else f"nprobe={p}"
                        if p is not None:
                            idx.nprobe = p
                    I, ms = timed_search(idx, Y, kp)
                    rows.append({"fde": fde, "projection": "pca", "dim_fde": D, "dim_index": d, "index": name, "param": ps,
                                 "K'": kp, "stage1_ms_median": ms, "index_MB": nb / 1e6, "bytes_per_site": nb / N,
                                 "build_s": tb, "encode_s": t_enc,
                                 **{f"recall@{k}": contain(I, gi, kp, k) if kp >= k else "" for k in KS}})
            log(fde, d, name, {r["param"]: round(r["recall@10"], 3) for r in rows[-5:] if r["K'"] == 1000})
            del idx
    write_tsv("fde_faiss.tsv", rows)


def main():
    global NTHREADS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "exact", "pooling", "faiss", "residue", "fde", "fde-faiss"):
        p = sp.add_parser(name)
        p.add_argument("--threads", type=int, default=NTHREADS, help="threads of the builds and batch work (default 8)")
        if name.startswith("fde"):
            p.add_argument("--R", type=int, default=4, help="SimHash repetitions (default 4)")
            p.add_argument("--k", type=int, default=7, help="SimHash bits per repetition (default 7)")
            p.add_argument("--d-proj", type=int, default=16, help="projection dimension (default 16)")
        if name == "fde-faiss":
            p.add_argument("--dim", type=int, default=1024, help="PCA dimension of the indexed vectors (default 1024)")
    a = ap.parse_args()
    NTHREADS = a.threads
    check_inputs(a.cmd)
    os.makedirs(OUT, exist_ok=True)
    set_threads(NTHREADS)
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
