#!/usr/bin/env python
"""Sequence-identity and random baselines and further comparison methods against ProBiS and the ligand labels over the
whole database (Supplementary Table S13 B and its caption, Table S8 B).

Baselines (higher score = more similar) for every evaluation query q and database site s (db_paths.txt order):
  seqid      the sequence-identity baseline: the largest MMseqs2 fident between the main chains of q and s over hits
             with query or target coverage >= 0.5 ($LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8; no hit = 0)
  random     uniform(0, 1) from default_rng(SeedSequence([0, q_idx]))
Ties: a strict order (top-1, RE@1%) sorts by (score desc, key asc), key = default_rng(SeedSequence([1, q_idx]))
.random(n_db), one key per query for every baseline; ROC-AUC uses mid-ranks. Main chain = the chain with the most
binding-site residues (site annotation tables).

Further comparison methods (--method NAME=DIR, compare) rank the database by the distance matrix DIR/dist.npy (smaller
= more similar, ties by database index): the Folddisco runs of scripts/folddisco.py assemble.

Produces the ProBiS columns of Supplementary Table S13 B (ProBiS ROC-AUC, top-1 MCC and P@1 of LEN-Seek, DeeplyTough,
Folddisco and the two baselines, from `compare`) and its ligand columns of the two baselines (ligand ROC-AUC and
RE@1%, from `ligand`), the paired differences of LEN-Seek from Folddisco in the caption of Table S13 with their
detection, the ProBiS ROC-AUC of the Folddisco runs (caption of Table S13), the share of the database the Folddisco
pre-filter returns (`compare`, Supplementary Section S6) and Supplementary Table S8 B (ROC-AUC over the whole database
with the G-LoSA and the ProBiS labels).

compare: per primary query, the database sites whose ProBiS pair
did not fail and that every method can score; queries without a positive or a negative there are dropped for every
method. (1) per-query ProBiS ROC-AUC, entry-weighted; (2) pooled out-of-fold top-1 MCC: GroupKFold(5) over the
primary queries by 50 % sequence cluster, each method's threshold on its own d (LEN-Seek: Chamfer distance, grid step
0.5; DeeplyTough: descriptor distance, step 0.0001; baselines: d = 1 - score, step 0.001; further methods: step 0.0001)
as the first maximum on the training folds; (3) P@1. Differences LEN-Seek minus comparator, paired sequence-cluster
bootstrap (10,000 resamples, seed 0; fold thresholds fixed). Detected: |delta| > 2 SD over the three LEN-Seek seeds
and the interval excludes 0. (1) is repeated with the G-LoSA labels (positive at GA >= 0.59, pairs without a G-LoSA
score left out) on the database sites every method can score, over the queries with a G-LoSA positive and negative
there: entry-weighted ROC-AUC of every method (same bootstrap; no detection rule). For a further method with
DIR/per_query.tsv (n_returned per query): the database sites it returns per evaluation query, as a share of the
database.
ligand: positives are database sites whose ligand has Morgan Dice >= 0.46 to the query ligand (CCD snapshot;
sites without a fingerprint left out); ROC-AUC, RE@1% (not below 5 positives) and the ROC-AUC on the database sites
< 30 % identical and in another Pfam clan; LEN-Seek, DeeplyTough and the further methods of the grid (its
<name>_raw columns) come from the whole-database ligand grid of fig4_ligand.py; entry-weighted means with 10,000
cluster-bootstrap resamples (seed 0), also of the differences. LEN-Seek minus each further method in ROC-AUC and RE@1%
is detected as in compare, with the SD of the entry-weighted means of the LEN-Seek seeds (the grid and --seed-grid)
over the same queries.

Subcommands and outputs (in $LENSEEK_OUT/baselines/):
  compare   compare/: per_query.csv, excluded_queries.tsv, summary_entry_weighted.csv, top1_pairs.csv, mcc_oof.csv,
            top1_paired.csv, detected.csv, summary_glosa.csv (G-LoSA ROC-AUC), returned.csv (further methods)
  ligand    ligand/: per_query.csv, summary.csv, detected.csv
Inputs: see `python analysis/run.py -h`; read here: the other seeds of the original model (--seed), the further methods
(--method) and, for ligand, the per-query files of fig4_ligand.py grid (--grid, default
$LENSEEK_OUT/fig4/main_s0/ligand_grid_per_query.csv; --seed-grid of the other seeds) and the identity strata of
scripts/site_annotation.py identity-strata ($LENSEEK_OUT/site_annotation/protein_strata.tsv).
usage: python analysis/baselines.py compare --seed main_s1=$LENSEEK_EVAL_ROOT/out_main_s1 --seed main_s2=... \\
           --method folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter
       python analysis/baselines.py ligand --seed-grid main_s1=$LENSEEK_OUT/fig4/main_s1/ligand_grid_per_query.csv \\
           --seed-grid main_s2=...
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold, PredefinedSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BASELINE_OUT, BASELINES, BL_STEP, DT_STEP, FOLDS, G, LS_STEP, OUT_ROOT, Baselines,  # noqa: E402
                    EvalOutputs, auc_from_ranks, boot_weights, bulk_dice, cluster_of, die, enrich, entry_weighted,
                    fingerprints, identity_strata, load_cache, make_grid, mcc, mw_auc, probis_query, require_path,
                    run_cv, site_key, site_tables, tie_key)

RE_MIN_POS = 5
N_BOOT, BOOT_SEED = 10000, 0
MAIN = "lenseek_main_s0"
ID_GE30 = ["30-40", "40-50", "50-80", "80-95", ">=95"]


# ---------------------------------------------------------------- ProBiS comparison
def grid_step(method):
    """Threshold grid step on d: LEN-Seek 0.5, baselines 0.001, DeeplyTough and the further methods 0.0001."""
    return LS_STEP if method.startswith("lenseek_") else BL_STEP if method in BASELINES else DT_STEP


def detect(delta, lo, hi, seed_vals):
    """Detection rule: (1) |delta| > 2 SD over the seeds, (2) the interval excludes 0; one seed: (2) only."""
    v = [x for x in seed_vals if np.isfinite(x)]
    sd = float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan
    c2_pos, c2_neg = lo > 0, hi < 0
    if np.isfinite(sd):
        c1_pos, c1_neg = delta > 2 * sd, -delta > 2 * sd
    else:
        c1_pos = c1_neg = None
    pos = c2_pos and (c1_pos is None or c1_pos)
    neg = c2_neg and (c1_neg is None or c1_neg)
    verdict = "LEN-Seek > {} detected" if pos else ("{} > LEN-Seek detected" if neg else "not detected")
    return dict(sd_seed=sd, n_seeds=len(v), two_sd=2 * sd if np.isfinite(sd) else np.nan,
                exceeds_2sd=(None if c1_pos is None else bool(c1_pos or c1_neg)),
                interval_excludes_0=bool(c2_pos or c2_neg), verdict=verdict)


def cmd_compare(a):
    od = os.path.join(a.out, "compare")
    ev = EvalOutputs(a.eval_dir, "primary")
    bl = Baselines(ev)
    scan = require_path(a.scan_dir, "--scan-dir", "LENSEEK_SCAN_DIR")
    st = np.load(os.path.join(scan, "all_status.npy"), mmap_mode="r")
    raw = np.load(os.path.join(scan, "all_raw.npy"), mmap_mode="r")
    gscan = require_path(a.glosa_scan_dir, "--glosa-scan-dir", "LENSEEK_GLOSA_SCAN_DIR")
    gst = np.load(os.path.join(gscan, "all_status.npy"), mmap_mode="r")
    ga = np.load(os.path.join(gscan, "all_ga.npy"), mmap_mode="r")
    pairs = require_path(a.pairs_dir, "--pairs-dir", "LENSEEK_PAIRS_DIR")
    sq = [site_key(p) for p in open(os.path.join(pairs, "all_query_sites.txt")).read().split()]
    sd = [site_key(p) for p in open(os.path.join(pairs, "all_db_sites.txt")).read().split()]
    if st.shape != (len(sq), len(sd)) or {raw.shape, gst.shape, ga.shape} != {st.shape} or sq != ev.q_keys or \
            sd != ev.db_keys:
        die("scan arrays or site lists do not follow the evaluation order")
    ls = {MAIN: ev.dist}
    for s in a.seed:
        name, d = s.split("=", 1)
        e = EvalOutputs(d, "primary")
        if e.q_keys != ev.q_keys or e.db_keys != ev.db_keys:
            die(f"{d}: query/DB order differs from main_s0")
        ls[f"lenseek_{name}"] = e.dist
    dt_dir = require_path(a.dt_dir, "--dt-dir", "LENSEEK_DT_DIR")
    if [site_key(p) for p in open(os.path.join(dt_dir, "db_paths.txt")).read().split()] != ev.db_keys:
        die("DeeplyTough DB order differs")
    extra, returned = {}, []
    for s in a.method:
        name, d = s.split("=", 1)
        if name in ls or name in ["deeplytough"] + BASELINES or name in extra:
            die(f"--method {name}: the name is taken")
        if [site_key(p) for p in open(os.path.join(d, "db_paths.txt")).read().split()] != ev.db_keys:
            die(f"{d}: DB order differs")
        extra[name] = np.load(os.path.join(d, "dist.npy"), mmap_mode="r")
        if os.path.exists(os.path.join(d, "per_query.tsv")):
            n = pd.read_csv(os.path.join(d, "per_query.tsv"), sep="\t").n_returned
            returned.append(dict(method=name, n_queries=len(n), n_db=bl.n_db, returned_mean=float(n.mean()),
                                 returned_frac_mean=float(n.mean() / bl.n_db), returned_median=float(n.median()),
                                 returned_min=int(n.min()), returned_max=int(n.max())))
    G.update(status=st, raw=raw, gst=gst, ga=ga, q_keys=ev.q_keys, n_db=bl.n_db, dbidx=np.arange(bl.n_db), ls=ls,
             bl=bl, dt=np.load(os.path.join(dt_dir, "dist.npy"), mmap_mode="r"), extra=extra)
    with Pool(a.workers) as p:
        res = p.map(probis_query, [int(q) for q in ev.sel], chunksize=2)
    gq = pd.DataFrame([dict(query_key=r["query_key"], protein=r["protein"], **r.pop("glosa", {})) for r, _ in res])
    pq = pd.DataFrame([r for r, _ in res])
    if (pq.n_pending > 0).any():
        die(f"{int((pq.n_pending > 0).sum())} primary queries have pending pairs; nothing written")
    os.makedirs(od, exist_ok=True)
    if returned:
        rt = pd.DataFrame(returned)
        rt.to_csv(os.path.join(od, "returned.csv"), index=False)
        print("database sites returned per evaluation query\n" + rt.to_string(index=False))
    pq[["q_idx", "query_key", "excluded"]][pq.excluded != ""].to_csv(os.path.join(od, "excluded_queries.tsv"),
                                                                    sep="\t", index=False)
    pq["cluster"] = pq.query_key.map(cluster_of)
    use = pq[pq.excluded == ""].copy()
    methods = list(ls) + ["deeplytough"] + BASELINES + list(extra)
    # (1) ProBiS ROC-AUC, entry-weighted, with paired differences
    cmp_pairs = [(MAIN, b) for b in BASELINES] + [(MAIN, "deeplytough")] + [(MAIN, x) for x in extra]
    for x, yb in cmp_pairs:
        use[f"diff__{x}__minus__{yb}__auc"] = use[f"{x}__auc"] - use[f"{yb}__auc"]
    use.to_csv(os.path.join(od, "per_query.csv"), index=False)
    cols = [f"{m}__auc" for m in methods] + [c for c in use.columns if c.startswith("diff__")]
    summ = pd.DataFrame([dict(stratum="all", column=c, **v) for c, v in
                         entry_weighted(use, cols, n_boot=a.n_boot, seed=BOOT_SEED).items()])
    summ.to_csv(os.path.join(od, "summary_entry_weighted.csv"), index=False)
    # (1) with the G-LoSA labels, over the queries with a G-LoSA positive and negative
    gq = gq[(gq.n_pos > 0) & (gq.n_pos < gq.n_db)].assign(cluster=gq.query_key.map(cluster_of))
    gres = entry_weighted(gq, [f"{m}__auc" for m in methods], n_boot=a.n_boot, seed=BOOT_SEED)
    gsum = pd.DataFrame([dict(column=c, **v) for c, v in gres.items()])
    gsum.to_csv(os.path.join(od, "summary_glosa.csv"), index=False)
    # (2) out-of-fold top-1 MCC on the same folds for every method and (3) P@1, on the same queries and top-1 pairs
    groups = np.array([cluster_of(ev.q_keys[i]) for i in ev.sel])
    fold = np.full(len(ev.sel), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=FOLDS).split(ev.sel, groups=groups)):
        fold[te] = f
    use["fold"] = use.q_idx.map(dict(zip(ev.sel.tolist(), fold.tolist())))
    top1 = {r["q_idx"]: t for r, t in res if t is not None}
    T, mcc_rows, ind, p1i = [], [], {}, {}
    for m in methods:
        d1 = np.array([top1[q][m][0] for q in use.q_idx])
        y1 = np.array([top1[q][m][1] for q in use.q_idx])
        grid = make_grid(d1, grid_step(m))
        t1 = pd.DataFrame({"q_idx": use.q_idx.to_numpy(), "cd": d1, "truth": y1.astype(float)})
        folds, _, pred = run_cv(t1, t1, grid, PredefinedSplit(use.fold.to_numpy()), None)
        t = y1.astype(bool)
        ind[m] = np.stack([pred & t, pred & ~t, ~pred & t, ~pred & ~t], 1).astype(float)
        p1i[m] = np.stack([t, np.ones(len(t), bool)], 1).astype(float)
        tp, fp, fn, tn = ind[m].sum(0)
        mcc_rows.append(dict(method=m, oof_top1_mcc=mcc(tp, fp, fn, tn), TP=int(tp), FP=int(fp), FN=int(fn),
                             TN=int(tn), p_at_1=float(t.mean()), top1_positive=int(t.sum()), n_queries=len(use),
                             grid_step=float(grid[1] - grid[0]) if len(grid) > 1 else np.nan, grid_max=float(grid[-1]),
                             fold_thresholds=";".join(f"{f['threshold']:g}" for f in folds)))
        for q, dv, yv in zip(use.q_idx, d1, y1):
            T.append(dict(q_idx=q, method=m, top1_d=dv, top1_label=yv, top1_db_idx=top1[q][m][2]))
    pd.DataFrame(T).to_csv(os.path.join(od, "top1_pairs.csv"), index=False)
    cl, W = boot_weights(use.cluster, a.n_boot, BOOT_SEED)     # paired cluster bootstrap, thresholds fixed
    ci_ = use.cluster.map({c: i for i, c in enumerate(cl)}).to_numpy()
    boot_mcc, boot_p1 = {}, {}
    for m in methods:
        c4 = np.zeros((len(cl), 4))
        np.add.at(c4, ci_, ind[m])
        boot_mcc[m] = mcc(*(W @ c4).T)
        c2 = np.zeros((len(cl), 2))
        np.add.at(c2, ci_, p1i[m])
        b2 = W @ c2
        boot_p1[m] = b2[:, 0] / b2[:, 1]
    for r in mcc_rows:
        for k, b in [("ci", boot_mcc[r["method"]]), ("p_at_1_ci", boot_p1[r["method"]])]:
            r[f"{k}_lo"], r[f"{k}_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    M = pd.DataFrame(mcc_rows)
    M.to_csv(os.path.join(od, "mcc_oof.csv"), index=False)
    M = M.set_index("method")
    mp = []
    for x, yb in cmp_pairs:
        for metric, col, boot in [("top1_mcc", "oof_top1_mcc", boot_mcc), ("p_at_1", "p_at_1", boot_p1)]:
            bd = boot[x] - boot[yb]
            mp.append(dict(pair=f"{x} - {yb}", metric=metric, delta=float(M.loc[x, col] - M.loc[yb, col]),
                           ci_lo=float(np.percentile(bd, 2.5)), ci_hi=float(np.percentile(bd, 97.5)),
                           n_queries=len(use), n_clusters=len(cl)))
    mp = pd.DataFrame(mp)
    mp.to_csv(os.path.join(od, "top1_paired.csv"), index=False)
    MP, S, seeds = mp.set_index(["pair", "metric"]), summ.set_index("column"), list(ls)
    det = []
    for comp, cname in [("deeplytough", "DeeplyTough")] + [(x, x) for x in extra]:
        for metric, key in [("(1) ProBiS ROC-AUC, entry-weighted", "auc"), ("(2) top-1 MCC, OOF", "top1_mcc"),
                            ("(3) P@1", "p_at_1")]:
            if key == "auc":
                vals = {m: float(S.loc[f"{m}__auc", "entry_mean"]) for m in seeds + [comp]}
                c = f"diff__{MAIN}__minus__{comp}__auc"
                delta, lo, hi = S.loc[c, "entry_mean"], S.loc[c, "entry_ci_lo"], S.loc[c, "entry_ci_hi"]
            else:
                col = "oof_top1_mcc" if key == "top1_mcc" else "p_at_1"
                vals = {m: float(M.loc[m, col]) for m in seeds + [comp]}
                row = MP.loc[(f"{MAIN} - {comp}", key)]
                delta, lo, hi = row.delta, row.ci_lo, row.ci_hi
            d = detect(float(delta), float(lo), float(hi), [vals[m] for m in seeds])
            d["verdict"] = d["verdict"].format(cname)
            det.append(dict(comparison=f"LEN-Seek main_s0 vs {cname}", metric=metric, lenseek_main_s0=vals[MAIN],
                            comparator=vals[comp], delta_main_s0_minus_comparator=float(delta), ci_lo=float(lo),
                            ci_hi=float(hi), seed_values=";".join(f"{m}={vals[m]:.4f}" for m in seeds), **d))
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(od, "detected.csv"), index=False)
    print(f"{len(use)} of {len(ev.sel)} primary queries ({use.protein.nunique()} entries, {len(cl)} clusters)")
    print(det[["comparison", "metric", "lenseek_main_s0", "comparator", "delta_main_s0_minus_comparator", "ci_lo",
               "ci_hi", "verdict"]].to_string(index=False))
    show = gsum[["column", "entry_mean", "entry_ci_lo", "entry_ci_hi"]]
    print(f"G-LoSA labels, {len(gq)} queries\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------- ligand metrics
def ligand_query(qi):
    g = G
    bl = g["bl"]
    code = g["q_code"][qi]
    r = dict(q_idx=int(qi), query_key=bl.ev.q_keys[qi], has_query_fp=bool(code in g["dice"]))
    if not r["has_query_fp"]:
        return r
    has = g["has"]
    dice = np.full(bl.n_db, -1.0)
    dice[has] = g["dice"][code][g["db_col"][has]]
    lab = dice >= 0.46
    unrel, q_pfam = bl.H.unrelated(qi, bl.H.identity(qi))
    tie = tie_key(qi, bl.n_db)
    S = has
    r.update(n_db=int(S.sum()), n_pos=int(lab[S].sum()), query_has_pfam=int(q_pfam))
    d = np.asarray(bl.ev.dist[qi], dtype=np.float64)
    r["chk_lenseek__auc"] = auc_from_ranks(rankdata(-d[S]), lab[S])   # must equal the grid's LEN-Seek value
    for name, s in bl.ranking(qi).items():
        r[f"{name}__auc"] = auc_from_ranks(rankdata(s[S]), lab[S])
        re1, _, _ = enrich(s[S], lab[S], tie=tie[S])
        r[f"{name}__re1"] = re1 if r["n_pos"] >= RE_MIN_POS else np.nan
        SM = S & unrel
        r[f"{name}__auc_unrel"] = mw_auc(s[SM], lab[SM]) if q_pfam else np.nan
    return r


def cmd_ligand(a):
    od = os.path.join(a.out, "ligand")
    ev = EvalOutputs(a.eval_dir, "primary")
    bl = Baselines(ev)
    qs, ds = site_tables()
    assert list(ds.key) == ev.db_keys and list(qs.key) == ev.q_keys
    cache = load_cache()
    db_code, q_code = ds.ligand.to_numpy(dtype=object), qs.ligand.to_numpy(dtype=object)
    ids = {c for c in list(db_code) + list(q_code) if c}
    fps = fingerprints(ids, cache)
    uniq = sorted(i for i in ids if fps.get(i) is not None)
    col = {c: j for j, c in enumerate(uniq)}
    db_col = np.array([col.get(c, -1) for c in db_code])
    dice = {c: np.array(bulk_dice(fps[c], [fps[u] for u in uniq]), dtype=np.float64)
            for c in set(q_code[ev.sel]) if c and fps.get(c) is not None}
    G.update(bl=bl, q_code=q_code, db_col=db_col, has=db_col >= 0, dice=dice)
    with Pool(a.workers) as p:
        pq = pd.DataFrame(p.map(ligand_query, list(ev.sel), chunksize=4))
    grid = pd.read_csv(require_path(a.grid, "--grid", "LENSEEK_OUT"))
    keep = {"q_idx": "q_idx", "protein": "protein", "seqid__auc": "grid_seqid__auc", "n_pos": "grid_n_pos",
            "n_db": "grid_n_db", "lenseek_raw__auc": "lenseek__auc", "lenseek_raw__re1": "lenseek__re1",
            "dt_raw__auc": "deeplytough__auc"}
    extra = sorted({c.split("_raw__")[0] for c in grid.columns if "_raw__" in c} - {"lenseek", "dt"})
    keep.update({f"{x}_raw__{m}": f"{x}__{m}" for x in extra for m in ("auc", "re1")})
    pq = pq.merge(grid[list(keep)].rename(columns=keep), on="q_idx", how="left", validate="1:1")
    ok = pq.has_query_fp
    for c, ref in [("chk_lenseek__auc", "lenseek__auc"), ("seqid__auc", "grid_seqid__auc"), ("n_pos", "grid_n_pos"),
                   ("n_db", "grid_n_db")]:
        x, y = pq.loc[ok, c].astype(float), pq.loc[ok, ref].astype(float)
        if not ((x - y).abs()[~(x.isna() & y.isna())].fillna(np.inf) <= 1e-12).all():
            die(f"{c} differs from the ligand grid {ref}: the labels are not those of the grid")
    pq = pq.drop(columns=["grid_seqid__auc", "grid_n_pos", "grid_n_db"])
    pq["id_stratum"] = pq.protein.map(identity_strata()).fillna("missing")
    pq["cluster50"] = pq.query_key.map(cluster_of)
    for x in extra:
        for m in ["auc", "re1"]:
            pq[f"diff_lenseek_minus_{x}__{m}"] = pq[f"lenseek__{m}"] - pq[f"{x}__{m}"]
    os.makedirs(od, exist_ok=True)
    pq.to_csv(os.path.join(od, "per_query.csv"), index=False)
    u = pq[pq.has_query_fp & (pq.n_pos > 0) & (pq.n_pos < pq.n_db)].assign(cluster=lambda d: d.cluster50)
    cols = ["lenseek__auc", "lenseek__re1", "deeplytough__auc"] + \
           [f"{x}__{m}" for x in extra for m in ("auc", "re1")] + \
           [f"{b}__{x}" for b in BASELINES for x in ("auc", "re1", "auc_unrel")] + \
           [c for c in u.columns if c.startswith("diff_")]
    rows = []
    for stratum, d in [("all", u), ("query identity <30", u[u.id_stratum == "<30"]),
                       ("query identity >=30", u[u.id_stratum.isin(ID_GE30)])]:
        rows += [dict(stratum=stratum, column=c, **v) for c, v in
                 entry_weighted(d, cols, n_boot=a.n_boot, seed=BOOT_SEED).items()]
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(od, "summary.csv"), index=False)
    show = summ[summ.stratum == "all"][["column", "n_queries", "entry_mean", "entry_ci_lo", "entry_ci_hi"]]
    print(f"{len(u)} usable primary queries\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # LEN-Seek minus each comparator, detection with the SD of the LEN-Seek seeds over the same queries
    S = summ[summ.stratum == "all"].set_index("column")
    seeds = {MAIN: {m: S.loc[f"lenseek__{m}", "entry_mean"] for m in ("auc", "re1")}}
    for s in a.seed_grid:
        name, f = s.split("=", 1)
        x = pd.read_csv(f).set_index("q_idx").reindex(u.q_idx)
        if (x.n_pos.to_numpy() != u.n_pos.to_numpy()).any():
            die(f"{f}: the ligand labels differ from those of --grid")
        x = pd.DataFrame({"protein": u.protein.to_numpy(), "cluster": u.cluster.to_numpy(),
                          "auc": x.lenseek_raw__auc.to_numpy(), "re1": x.lenseek_raw__re1.to_numpy()})
        seeds[f"lenseek_{name}"] = {m: v["entry_mean"] for m, v in entry_weighted(x, ["auc", "re1"], n_boot=1).items()}
    det = []
    for comp, cname in [(x, x) for x in extra]:
        for m, metric in [("auc", "ligand ROC-AUC, entry-weighted"), ("re1", "ligand RE@1%, entry-weighted")]:
            c = S.loc[f"diff_lenseek_minus_{comp}__{m}"]
            d = detect(float(c.entry_mean), float(c.entry_ci_lo), float(c.entry_ci_hi), [v[m] for v in seeds.values()])
            d["verdict"] = d["verdict"].format(cname)
            det.append(dict(comparison=f"LEN-Seek main_s0 vs {cname}", metric=metric, lenseek_main_s0=seeds[MAIN][m],
                            comparator=float(S.loc[f"{comp}__{m}", "entry_mean"]),
                            delta_main_s0_minus_comparator=float(c.entry_mean), ci_lo=float(c.entry_ci_lo),
                            ci_hi=float(c.entry_ci_hi), n_queries=int(c.n_queries),
                            seed_values=";".join(f"{k}={v[m]:.4f}" for k, v in seeds.items()), **d))
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(od, "detected.csv"), index=False)
    print(det[["comparison", "metric", "lenseek_main_s0", "comparator", "delta_main_s0_minus_comparator", "ci_lo",
               "ci_hi", "two_sd", "verdict"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("compare", cmd_compare), ("ligand", cmd_ligand)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"), help="LEN-Seek ($LENSEEK_EVAL_DIR)")
        p.add_argument("--out", default=BASELINE_OUT,
                       help="writes compare/ or ligand/ (default $LENSEEK_OUT/baselines)")
        if name == "compare":
            p.add_argument("--pairs-dir", default=os.environ.get("LENSEEK_PAIRS_DIR"),
                           help="scan site lists all_query_sites.txt, all_db_sites.txt ($LENSEEK_PAIRS_DIR)")
            p.add_argument("--scan-dir", default=os.environ.get("LENSEEK_SCAN_DIR"))
            p.add_argument("--glosa-scan-dir", default=os.environ.get("LENSEEK_GLOSA_SCAN_DIR"))
            p.add_argument("--dt-dir", default=os.environ.get("LENSEEK_DT_DIR"))
            p.add_argument("--seed", action="append", default=[], help="further LEN-Seek seed: main_s1=<eval dir>")
            p.add_argument("--method", action="append", default=[],
                           help="further comparison method: name=<dir with dist.npy, db_paths.txt (and per_query.tsv "
                                "with n_returned)>, e.g. folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter")
        if name == "ligand":
            p.add_argument("--grid", default=os.path.join(OUT_ROOT, "fig4", "main_s0", "ligand_grid_per_query.csv"),
                           help="LEN-Seek's per-query whole-database ligand values (fig4_ligand.py grid)")
            p.add_argument("--seed-grid", action="append", default=[],
                           help="further LEN-Seek seed: main_s1=<its ligand_grid_per_query.csv of fig4_ligand.py grid>")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
