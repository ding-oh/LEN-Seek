#!/usr/bin/env python
"""Supplementary Table S15 (single-vector and two-stage retrieval), the two-stage values of the main text, of
Supplementary Section S8 and of Table S16, and the ligand-similar fraction of the two-stage search (Section S8), from
the outputs of scripts/scaling.py and scripts/single_vector.py.

Writes to $LENSEEK_OUT/table3_two_stage/:
  recall_ci.csv        recall@1/10/100 of the two-stage search per database, nprobe and K' (means over the queries)
                       with 95 % intervals from 10,000 bootstrap resamples of 50 % sequence clusters, query-weighted and
                       entry-weighted (pockets of a PDB entry averaged first)
  synthetic_share.csv  share of synthetic sites in the exact top-1/10/100 of the synthetic databases
  time_twostage.csv    median and 95th-percentile per-query time of the two-stage search and of its parts
  time_exact.csv       median and 95th-percentile per-query time of the exhaustive scan
  fold_change.csv      growth of these times from the real database (same threads and setting)
  fold_exact_T1_sub48_paired.csv  one-thread exhaustive scan timed on the 48-query subset: growth against the
                       real-database median over the same 48 queries
  memory_build.csv     index and store sizes, index build time and peak memory
  build_cost.csv       the measured rows and an extrapolation from the 10^7 chimera database to 2x10^8 sites: index and
                       store bytes per site x 2x10^8; train time x (nlist ratio)^2 (64 x nlist training vectors); add time
                       x (residues x nlist) ratio; encoding without the Ankh embedding = core-seconds per site of the
                       encoding run of the filtered sites (encode_time.json of scripts/evaluate.py --added) x 2x10^8
  gpu_exact.csv        exhaustive scan on one GPU
  table3.csv           real database, 10^6 and 10^7 chimeras; nprobe 16, K' = 1,000
  tableS9_A.csv        Table S15 A: single-vector and residue indexes on the search database (scripts/single_vector.py)
  tableS9_B.csv        Table S15 B: every database, exhaustive scan and two-stage search at K' = 1,000 and 3,000
  ligand_ratio.csv, ligand_<db>_per_query.csv   real and real150k: per primary query with ligand-similar and
                       dissimilar database sites, the ligand-similar fraction of the top 10 of the exhaustive scan and of
                       the two-stage search (exact re-rank of the first 1,000 candidates), sites without a ligand
                       fingerprint dropped from both lists (Labels below); entry-weighted means and the ratio
                       two-stage / exhaustive with a 95 % paired bootstrap interval over 50 % sequence clusters (10,000
                       resamples, seed 0), non-inferior when the lower limit is >= 0.95
Inputs: see `python analysis/run.py -h`; read here: --acc DIR (per database: recall_per_query.csv.gz, build_acc.json,
gen.json, exact_gpu*.json, cands_np16.npz; default $LENSEEK_OUT/scaling), --time DIR (per database:
search_T{1,16}.csv.gz/.json, exact_T*.json, build_time.json; default = --acc), --ann DIR (scripts/single_vector.py
TSVs; default $LENSEEK_OUT/single_vector) and LENSEEK_SCALING_ADDED (the sites removed by the ligand filters,
scripts/evaluate.py --added: encode_time.json; dist_added.npy, added_keys.txt for real150k; default
$LENSEEK_OUT/scaling/added).

usage: python analysis/table3_two_stage.py [--acc ACC_DIR] [--time TIME_DIR] [--ann ANN_DIR]
"""
import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT_ROOT, REFERENCE_DIR, boot_weights, bulk_dice, cluster_of, cluster_sums, die,  # noqa: E402
                    fingerprints, ligand_code, load_cache, primary_excluded, protein_of, site_key, summary)

OUT = os.path.join(OUT_ROOT, "table3_two_stage")
N_REAL = 114_661
DBS = {"real": dict(kind="real", n_total=N_REAL), "real150k": dict(kind="real+added", n_total=N_REAL + 34_976),
       "chim3e5": dict(kind="chimera", n_total=300_000), "chim1e6": dict(kind="chimera", n_total=1_000_000),
       "chim1e7": dict(kind="chimera", n_total=10_000_000), "copy1e6": dict(kind="copy", n_total=1_000_000),
       "copy1e7": dict(kind="copy", n_total=10_000_000)}
ADDED = os.environ.get("LENSEEK_SCALING_ADDED") or os.path.join(OUT_ROOT, "scaling", "added")
EVAL = os.environ.get("LENSEEK_EVAL_DIR", "")


def paired_ratio(df, a, b, W, cl, entry=True):
    """Ratio of the means of columns a and b on the same rows (both finite) with a 95 % paired cluster-bootstrap
    interval."""
    d = df[np.isfinite(df[a].astype(float)) & np.isfinite(df[b].astype(float))]
    sa, na, ma, cnt = cluster_sums(d, a, cl, entry)
    sb, nb, mb, _ = cluster_sums(d, b, cl, entry)
    with np.errstate(invalid="ignore", divide="ignore"):
        rat = ((W @ sa) / (W @ na)) / ((W @ sb) / (W @ nb))
    return dict(a=a, b=b, mean_a=ma, mean_b=mb, ratio=ma / mb if mb else np.nan,
                ratio_lo=float(np.nanpercentile(rat, 2.5)), ratio_hi=float(np.nanpercentile(rat, 97.5)),
                n_queries=int(len(d)), n_entries=int(d.groupby(["cluster", "protein"]).ngroups),
                n_clusters=int(d.cluster.nunique()))


def query_meta(eval_dir):
    """q, query_key, protein (PDB id) and 50 % sequence cluster of every query."""
    keys = [site_key(p) for p in open(os.path.join(eval_dir, "q_paths.txt")).read().split()]
    return pd.DataFrame(dict(q=np.arange(len(keys)), query_key=keys, protein=[protein_of(k) for k in keys],
                             cluster=[cluster_of(k) for k in keys]))


class Labels:
    """Per primary query: database sites whose ligand has a Morgan fingerprint (radius 2, 2,048 bits, SMILES from the
    CCD snapshot) are positive when the Dice similarity to the query ligand is >= 0.46. Ligand codes of the
    database sites from db_ligand_codes.tsv of asset reference_data, of the queries and filtered sites from their
    ligand files."""

    def __init__(self, with_added=False):
        self.q_keys = [site_key(p) for p in open(os.path.join(EVAL, "q_paths.txt")).read().split()]
        db_keys = [site_key(p) for p in open(os.path.join(EVAL, "db_paths.txt")).read().split()]
        excl = primary_excluded()
        self.sel = np.array([i for i, k in enumerate(self.q_keys) if protein_of(k) not in excl], dtype=int)
        fz = pd.read_csv(os.path.join(REFERENCE_DIR, "db_ligand_codes.tsv"), sep="\t", dtype=str, keep_default_na=False)
        codes = list(fz.set_index("key").code.reindex(db_keys, fill_value=""))
        if with_added:
            codes += [ligand_code(os.environ["LENSEEK_BSDB_ROOT"], k) or ""
                      for k in open(os.path.join(ADDED, "added_keys.txt")).read().split()]
        self.N = len(codes)
        self.qc = {self.q_keys[i]: ligand_code(os.environ["LENSEEK_QUERY_ROOT"], self.q_keys[i]) for i in self.sel}
        self.fps = fingerprints({c for c in codes if c} | {c for c in self.qc.values() if c}, load_cache())
        self.uniq = sorted(c for c in self.fps if self.fps[c] is not None)
        col = {c: j for j, c in enumerate(self.uniq)}
        self.db_col = np.array([col.get(c, -1) for c in codes])
        self.has = self.db_col >= 0

    def lab(self, qi):
        """bool labels (N,) for query qi (False without fingerprint; see self.has), or None without a query
        fingerprint."""
        code = self.qc.get(self.q_keys[qi])
        if not code or self.fps.get(code) is None:
            return None
        lab = np.zeros(self.N, bool)
        lab[self.has] = np.array(bulk_dice(self.fps[code], [self.fps[u] for u in self.uniq]))[self.db_col[self.has]] >= 0.46
        return lab


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def recall_tables(acc):
    meta = query_meta(EVAL)
    cl, W = boot_weights(meta.cluster)
    rows, share = [], []
    for db in DBS:
        f = f"{acc}/{db}/recall_per_query.csv.gz"
        if not os.path.exists(f):
            continue
        pq = pd.read_csv(f).merge(meta, on="q")
        for (npb, kp), d in pq.groupby(["nprobe", "Kp"]):
            assert len(d) == 481
            r = dict(db=db, N=DBS[db]["n_total"], nprobe=npb, Kp=kp, Kp_eff_mean=float(d.Kp_eff.mean()))
            for k in (1, 10, 100):
                q, _ = summary(d, f"recall@{k}", W, cl, entry=False)
                e, _ = summary(d, f"recall@{k}", W, cl, entry=True)
                r.update({f"R{k}": q["mean"], f"R{k}_lo": q["lo"], f"R{k}_hi": q["hi"],
                          f"R{k}_entry": e["mean"], f"R{k}_entry_lo": e["lo"], f"R{k}_entry_hi": e["hi"]})
            rows.append(r)
            if "top10_n_syn" in d:
                share.append(dict(db=db, nprobe=npb, Kp=kp, **{f"top{k}_share_syn": d[f"top{k}_n_syn"].sum() / (k * len(d))
                                                               for k in (1, 10, 100)}))
    return pd.DataFrame(rows), pd.DataFrame(share)


def time_tables(tdir):
    rows, ex = [], []
    for db in DBS:
        for T in (1, 16):
            f = f"{tdir}/{db}/search_T{T}.csv.gz"
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            if "warm" in d:                                    # cold reads only
                d = d[d.warm == 0]
            for (npb, kp), g in d.groupby(["nprobe", "Kp"]):
                rows.append(dict(db=db, N=DBS[db]["n_total"], threads=T, nprobe=npb, Kp=kp, n_queries=len(g),
                                 total_med_ms=1e3 * g.total.median(), total_p95_ms=1e3 * g.total.quantile(0.95),
                                 s1_med_ms=1e3 * g.t_s1.median(), read_med_ms=1e3 * g.t_read.median(),
                                 rr_med_ms=1e3 * g.t_rr.median(), read_p95_ms=1e3 * g.t_read.quantile(0.95),
                                 Kp_eff_mean=g.Kp_eff.mean(), n_res_mean=g.n_res.mean()))
        for f in sorted(os.listdir(f"{tdir}/{db}")) if os.path.isdir(f"{tdir}/{db}") else []:
            if f.startswith("exact_T") and f.endswith(".json"):
                j = jl(f"{tdir}/{db}/{f}")
                ex.append(dict(db=db, N=j["N"], threads=j["threads"], query_set=j["query_set"], layout=j["layout"],
                               n_queries=j["n_queries"], median_s=j["median_s"], p95_s=j["p95_s"], mean_s=j["mean_s"],
                               layout_s=j["layout_s"], rss_after_layout_gb=j["rss_after_layout_gb"],
                               top100_overlap_min=min(c["top100_overlap"] for c in j["checks"]) if j["checks"] else np.nan))
    tw = pd.DataFrame(rows, columns=list(rows[0]) if rows else ["db", "threads", "nprobe", "Kp"])
    te = pd.DataFrame(ex, columns=list(ex[0]) if ex else ["db", "threads", "layout"])
    fc = []
    base = tw[tw.db == "real"].set_index(["threads", "nprobe", "Kp"])
    for _, r in tw.iterrows():
        k = (r.threads, r.nprobe, r.Kp)
        if k in base.index:
            b = base.loc[k]
            fc.append(dict(db=r.db, N=r.N, sites_fold=r.N / N_REAL, threads=r.threads, nprobe=r.nprobe, Kp=r.Kp,
                           fold_median=r.total_med_ms / b.total_med_ms, fold_p95=r.total_p95_ms / b.total_p95_ms,
                           fold_s1=r.s1_med_ms / b.s1_med_ms))
    eb = te[(te.db == "real") & (te.layout == "fp16")].set_index("threads")
    for _, r in te[te.layout == "fp16"].iterrows():
        if r.threads in eb.index:
            fc.append(dict(db=r.db, N=r.N, sites_fold=r.N / N_REAL, threads=r.threads, nprobe="exact", Kp="exact",
                           fold_median=r.median_s / eb.loc[r.threads].median_s, fold_p95=r.p95_s / eb.loc[r.threads].p95_s,
                           query_set=r.query_set))
    paired = []
    rd = lambda db, n: {int(x["q"]): float(x["t"]) for x in csv.DictReader(open(f"{tdir}/{db}/{n}"))}
    for db in DBS:
        if os.path.exists(f"{tdir}/{db}/exact_T1_sub48.csv"):
            r1, s = rd("real", "exact_T1_all.csv"), rd(db, "exact_T1_sub48.csv")
            m_db, m_same = statistics.median(s.values()), statistics.median(r1[q] for q in s)
            m_all = statistics.median(r1.values())
            paired.append(dict(db=db, threads=1, query_set="sub48", n=len(s), median_s=round(m_db, 4),
                               real_median_same48_s=round(m_same, 4), fold_paired=round(m_db / m_same, 3),
                               real_median_all481_s=round(m_all, 4), fold_vs_all481=round(m_db / m_all, 3)))
    return tw, te, pd.DataFrame(fc), pd.DataFrame(paired)


def build_tables(acc, tdir):
    mb = []
    for db in DBS:
        a, t = jl(f"{acc}/{db}/build_acc.json"), jl(f"{tdir}/{db}/build_time.json")
        if not a:
            continue
        g = jl(f"{acc}/{db}/gen.json")
        s1, s16 = jl(f"{tdir}/{db}/search_T1.json"), jl(f"{tdir}/{db}/search_T16.json")
        mb.append(dict(db=db, N=a["N"], R=a["R"], nlist=a["nlist"], n_train=a["n_train"], index_bytes=a["index_bytes"],
                       index_B_per_site=a["index_bytes_per_site"], store_B_per_site=a["store_bytes_per_site"],
                       store_GB=a["R"] * 512 / 1e9, build16_train_s=t.get("train_s"), build16_add_s=t.get("add_s"),
                       build16_s=t.get("build_s"), build16_peak_rss_gb=t.get("peak_rss_gb"),
                       build32_s=a["build_s"], build32_peak_rss_gb=a["peak_rss_gb"],
                       search_T1_peak_rss_gb=s1.get("peak_rss_gb"), search_T16_peak_rss_gb=s16.get("peak_rss_gb"),
                       gt_s_acc=g.get("gt_s"), gen_s_acc=g.get("gen_s")))
    mb = pd.DataFrame(mb)
    rows = [dict(db=r.db, N=r.N, status="measured", nlist=r.nlist, index_GB=r.index_bytes / 1e9,
                 index_B_per_site=r.index_B_per_site, store_GB=r.store_GB, store_B_per_site=r.store_B_per_site,
                 build16_train_s=r.build16_train_s, build16_add_s=r.build16_add_s, build16_s=r.build16_s,
                 build16_peak_rss_gb=r.build16_peak_rss_gb) for r in mb.itertuples()]
    b = mb[mb.db == "chim1e7"]
    if len(b) and os.path.exists(os.path.join(ADDED, "encode_time.json")):
        b = b.iloc[0]
        N2 = 2e8
        R2 = b.R / b.N * N2
        nl2 = int(2 ** round(np.log2(2.5 * np.sqrt(R2))))
        tr, ad = (nl2 / b.nlist) ** 2, (R2 * nl2) / (b.R * b.nlist)
        enc = json.load(open(os.path.join(ADDED, "encode_time.json")))
        enc_core_s = enc["encode_seconds"] * enc["threads"] / enc["added_sites"]
        rows.append(dict(db="2e8 (extrapolated from chim1e7)", N=N2, status="EXTRAPOLATED", nlist=nl2,
                         index_GB=b.index_B_per_site * N2 / 1e9, index_B_per_site=b.index_B_per_site,
                         store_GB=b.store_B_per_site * N2 / 1e9, store_B_per_site=b.store_B_per_site,
                         build16_train_s=b.build16_train_s * tr, build16_add_s=b.build16_add_s * ad,
                         build16_s=b.build16_train_s * tr + b.build16_add_s * ad,
                         encode_only_core_s_per_site=enc_core_s, encode_only_core_h_total=enc_core_s * N2 / 3600))
    return mb, pd.DataFrame(rows)


def gpu_table(acc):
    rows = []
    for db in DBS:
        for tag in ("", "_stream10x"):
            j = jl(f"{acc}/{db}/exact_gpu{tag}.json")
            if j:
                rows.append({k: j[k] for k in ("db", "mode", "N", "N_db", "stream_passes", "n_queries", "median_s",
                                               "p95_s", "gpu")})
    return pd.DataFrame(rows)


def manuscript_tables(rc, tw, te, mb, ann):
    pick = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    exact = lambda db, T: pick(te[te.layout == "fp16"], db=db, threads=T).median_s
    t3, b = [], []
    for db in DBS:
        if not all((df.db == db).any() for df in (rc, mb, tw[tw.threads == 1], tw[tw.threads == 16],
                                                  te[te.threads == 1], te[te.threads == 16])):
            continue
        m = pick(mb, db=db)
        b.append(dict(db=db, sites=DBS[db]["n_total"], method="Exact scan", Kp="all", ms_1=1e3 * exact(db, 1),
                      ms_16=1e3 * exact(db, 16)))
        for kp in (1000, 3000):
            r = pick(rc, db=db, nprobe=16, Kp=kp)
            b.append(dict(db=db, sites=DBS[db]["n_total"], method="Two-stage", Kp=kp, Kp_eff_mean=r.Kp_eff_mean, R1=r.R1,
                          R10=r.R10, R10_lo=r.R10_lo, R10_hi=r.R10_hi, R100=r.R100,
                          ms_1=pick(tw, db=db, threads=1, nprobe=16, Kp=kp).total_med_ms,
                          ms_16=pick(tw, db=db, threads=16, nprobe=16, Kp=kp).total_med_ms, B_per_site=m.index_B_per_site))
            if kp == 1000 and db in ("real", "chim1e6", "chim1e7"):
                t3.append(dict(db=db, sites=DBS[db]["n_total"], exact_s_1=exact(db, 1), exact_s_16=exact(db, 16),
                               twostage_ms_1=b[-1]["ms_1"], twostage_ms_16=b[-1]["ms_16"], R1=r.R1, R10=r.R10,
                               R10_lo=r.R10_lo, R10_hi=r.R10_hi, R100=r.R100, index_GB=m.index_bytes / 1e9,
                               store_GB=m.store_GB))
    A = []
    if os.path.exists(f"{ann}/pooling.tsv"):
        tsv = lambda n: pd.read_csv(f"{ann}/{n}", sep="\t", keep_default_na=False)
        ex, po, fs, f2, tr = (tsv(n) for n in ("exact_scan.tsv", "pooling.tsv", "faiss_pooled_single.tsv",
                                               "faiss_pooled_2stage.tsv", "twostage_residue.tsv"))
        A.append(dict(method="Exhaustive exact Chamfer scan", Kp="all", R1=1.0, R10=1.0, R100=1.0,
                      ms=1e3 * float(pick(ex, mode="dense_single_1thread").s_per_query)))
        for name, pool in (("Mean pooling, exact search", "mean"), ("Mean and max pooling, exact search", "meanmax")):
            p, f = pick(po, pooling=pool, metric="l2"), pick(fs, vector=f"{pool}_l2", index="Flat")
            A.append(dict(method=name, Kp="", R1=p["recall@1"], R10=p["recall@10"], R100=p["recall@100"],
                          ms=f.lat_ms_median_k10, B_per_site=f.bytes_per_site))
        for name, idx, par in (("Mean pooling, HNSW, re-ranking", "HNSW32", "efSearch=1000"),
                               ("Mean pooling, IVF-PQ, re-ranking", "IVF1024,PQ32", "nprobe=64")):
            r = pick(f2, vector="mean_l2", index=idx, param=par, **{"K'": 1000})
            A.append(dict(method=name, Kp=1000, R1=r["recall@1"], R10=r["recall@10"], R100=r["recall@100"],
                          ms=r.total_ms_median, B_per_site=pick(fs, vector="mean_l2", index=idx).bytes_per_site))
        tr["K'"] = tr["K'"].astype(str)
        for kp in ("1000", "3000"):
            r = pick(tr, index="IVF4096,PQ32", param="r=256,nprobe=16", **{"K'": kp})
            A.append(dict(method="Residue IVF-PQ, re-ranking", Kp=int(kp), Kp_eff_mean=r["K'_effective_mean"],
                          R1=r["recall@1"], R10=r["recall@10"], R100=r["recall@100"], ms=r.total_ms_median,
                          B_per_site=r.bytes_per_site))
    return pd.DataFrame(t3), pd.DataFrame(A), pd.DataFrame(b)


def ligand_ratio(acc, db):
    """Ligand-similar fraction of the top 10, two-stage (K' 1,000) against the exhaustive scan, for database db."""
    need = ["LENSEEK_QUERY_ROOT"] + (["LENSEEK_BSDB_ROOT"] if db == "real150k" else [])
    if not EVAL or any(not os.environ.get(v) for v in need):
        die(f"ligand fraction of {db}: set LENSEEK_EVAL_DIR and " + ", ".join(need))
    if db == "real150k" and not os.path.exists(os.path.join(ADDED, "dist_added.npy")):
        die(f"ligand fraction of real150k: {ADDED}/dist_added.npy missing (scripts/evaluate.py --added; "
            "LENSEEK_SCALING_ADDED)")
    L = Labels(with_added=(db == "real150k"))
    dist = np.load(os.path.join(EVAL, "dist.npy"), mmap_mode="r")
    dadd = np.load(os.path.join(ADDED, "dist_added.npy"), mmap_mode="r") if db == "real150k" else None
    c = np.load(f"{acc}/{db}/cands_np16.npz")
    ids, off = c["ids"], c["off"]
    rows = []
    for qi in L.sel:
        lab = L.lab(qi)
        r = dict(q=int(qi), query_key=L.q_keys[qi], has_query_fp=lab is not None)
        if lab is not None:
            d = np.asarray(dist[qi], np.float64)
            if dadd is not None:
                d = np.concatenate([d, np.asarray(dadd[qi], np.float64)])
            assert len(d) == L.N
            npos = int(lab[L.has].sum())
            r.update(n_db=int(L.has.sum()), n_pos=npos)
            if 0 < npos < r["n_db"]:
                idx = np.flatnonzero(L.has)
                cand = ids[off[qi]:off[qi + 1]][:1000].astype(np.int64)
                for name, o in (("exact", idx[np.lexsort((idx, d[idx]))]), ("ts1000", cand[np.lexsort((cand, d[cand]))])):
                    o = o[L.has[o]]
                    r[f"{name}_top10"] = float(lab[o][:10].mean()) if len(o) >= 10 else np.nan
        rows.append(r)
    pq = pd.DataFrame(rows)
    use = pq[pq.has_query_fp & (pq.n_pos.fillna(0) > 0) & (pq.n_pos < pq.n_db)].copy()
    use = use.merge(query_meta(EVAL)[["q", "protein", "cluster"]], on="q")
    cl, W = boot_weights(use.cluster)
    p = paired_ratio(use, "ts1000_top10", "exact_top10", W, cl, entry=True)
    pq.to_csv(os.path.join(OUT, f"ligand_{db}_per_query.csv"), index=False)
    return dict(db=db, n_queries=p["n_queries"], n_entries=p["n_entries"], n_clusters=p["n_clusters"],
                exact_top10=p["mean_b"], twostage_K1000_top10=p["mean_a"], ratio=p["ratio"], ratio_lo95=p["ratio_lo"],
                ratio_hi95=p["ratio_hi"], criterion="ratio_lo95 >= 0.95", passed=bool(p["ratio_lo"] >= 0.95))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acc", default=os.path.join(OUT_ROOT, "scaling"))
    ap.add_argument("--time", help="timing outputs (default --acc)")
    ap.add_argument("--ann", default=os.path.join(OUT_ROOT, "single_vector"))
    a = ap.parse_args()
    tdir = a.time or a.acc
    os.makedirs(OUT, exist_ok=True)
    rc, share = recall_tables(a.acc)
    tw, te, fc, paired = time_tables(tdir)
    mb, cost = build_tables(a.acc, tdir)
    t3, sA, sB = manuscript_tables(rc, tw, te, mb, a.ann)
    for name, df in (("recall_ci", rc), ("synthetic_share", share), ("time_twostage", tw), ("time_exact", te),
                     ("fold_change", fc), ("fold_exact_T1_sub48_paired", paired), ("memory_build", mb),
                     ("build_cost", cost), ("gpu_exact", gpu_table(a.acc)), ("table3", t3), ("tableS9_A", sA),
                     ("tableS9_B", sB)):
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    lig = [ligand_ratio(a.acc, db) for db in ("real", "real150k") if os.path.exists(f"{a.acc}/{db}/cands_np16.npz")]
    if lig:
        pd.DataFrame(lig).to_csv(os.path.join(OUT, "ligand_ratio.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
