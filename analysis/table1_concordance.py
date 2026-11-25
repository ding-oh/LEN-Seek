#!/usr/bin/env python
"""Concordance of LEN-Seek and the comparison methods with ProBiS and G-LoSA on the evaluation data.

Produces Table 1 of the main text (LEN-Seek top-k concordance at the fixed threshold), Supplementary Table S7 A (the
same at out-of-fold thresholds) with the MCC rows of the comparison methods (Supplementary Table S13 A), the fixed
thresholds of every method (highest top-1 MCC over all primary queries), Supplementary Table S8 A (G-LoSA labels),
Table S9 (search results classified as similar), the 50 shuffled splits of Supplementary Section S2 and the
evaluation-data rows of Table S10 B, the LEN-Seek ProBiS PR-AUC over the whole database (`fullscan`) and the positive
labels among unrelated pairs of the database-wide scans (`unrelated`, the caption of Table S8). The ligand columns of
Supplementary Table S13 B come from fig4_ligand.py and its ProBiS ROC-AUC / top-1 MCC / P@1 columns from the
database-wide comparison of baselines.py.

Every method ranks the whole database itself: LEN-Seek by Chamfer distance (ties by database index), DeeplyTough by
descriptor distance over the sites with a descriptor (a distance matrix .npy), and any other method from its own
top-100 list (TSV with query_idx, rank, db_idx and dist, or score with d = 1 - score, written at full precision; for
Folddisco the top100_pairs.tsv of scripts/folddisco.py assemble, d = -score). A further comparison method is added by
passing its top-100 list with --method.
Labels are the database-wide scans: ProBiS positive at Z >= 2 (Z = round((raw - 1.958) / 2.21468, 2); no alignment
negative, failed comparisons excluded), G-LoSA positive at GA >= 0.59. Queries are the primary queries.

Fixed threshold: the threshold with the highest top-1 MCC over all primary queries (first maximum on the method's grid
of common.make_grid); Table 1 classifies the LEN-Seek top-k results at it. Out-of-fold thresholds: GroupKFold(5) over
the queries by 50 % sequence cluster; on the four training folds the threshold with the highest top-1 MCC (first
maximum on the grid, step given with --method: LEN-Seek 0.5, DeeplyTough and Folddisco 0.0001, the sequence-identity
and random baselines 0.001 on d) classifies the held-out fold. Counts are pooled over queries for rank <= k, with 95 %
intervals from 1,000 cluster bootstrap resamples (thresholds fixed). The same fold thresholds give Table S8 A, whose
paired LEN-Seek - --paired method MCC difference uses 10,000 resamples. The LEN-Seek split is repeated 50 times with
shuffled folds (Section S2). Stratum rows use the per-query top-1 outcomes with 10,000 resamples within each stratum.
The whole-database PR-AUC interval uses 1,000 resamples.

Subcommands and outputs ($LENSEEK_OUT/table1/, default ./out/table1/):
  table1    table1_fixed.csv (Table 1), table1A.csv (Table S7 A and S13 A), tableS10A_glosa.csv (Table S8 A),
            thresholds.csv, threshold_cv.csv, threshold_cv_repeats.csv, tableS12B_strata.csv (Table S10 B),
            paired.csv (LEN-Seek minus --paired method, G-LoSA top-1 and top-100 MCC), top1_oof.csv (LEN-Seek top-1 of
            every query with its fold threshold; read by ablations.py)
  counts    tableS11_counts.csv, tableS11_overlap.csv (Table S9): pairs classified as similar among the top-100
            results at the fixed thresholds of thresholds.csv (run table1 first), and the overlaps with the LEN-Seek row
  fullscan  fullscan_summary.csv: database-wide ProBiS ROC-AUC and PR-AUC of LEN-Seek (entry-weighted)
  unrelated unrelated_probis.tsv, unrelated_glosa.tsv: ProBiS Z >= 2 and G-LoSA GA >= 0.59 among the unrelated pairs
            (no chain pair of the two entries >= 30 % identical, different ligands with Dice < 0.46, both ligands >= 6
            heavy atoms; $LENSEEK_OUT/labels/null/null_set.npz of scripts/labels.py null), among all pairs and among
            the others

Inputs: see `python analysis/run.py -h`; read here: the comparison methods (--method NAME GRID_STEP FILE) and the
identity, structure and ligand strata of Table S10 B ($LENSEEK_OUT/site_annotation/protein_strata.tsv and
query_strata.csv of scripts/site_annotation.py identity-strata and strata).

usage:
  python analysis/table1_concordance.py table1 --paired Folddisco \\
      --method DeeplyTough 0.0001 $LENSEEK_DT_DIR/dist.npy \\
      --method "Sequence-identity baseline" 0.001 seqid_top100.tsv --method "Random ranking" 0.001 random_top100.tsv \\
      --method Folddisco 0.0001 $LENSEEK_FOLDDISCO_DIR/prefilter/top100_pairs.tsv
  python analysis/table1_concordance.py counts --method DeeplyTough 0.0001 $LENSEEK_DT_DIR/dist.npy \\
      --method Folddisco 0.0001 $LENSEEK_FOLDDISCO_DIR/prefilter/top100_pairs.tsv
  python analysis/table1_concordance.py fullscan
  python analysis/table1_concordance.py unrelated
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LENSEEK, OUT_ROOT, PROTEIN_STRATA, QUERY_STRATA, EvalOutputs, ScanLabels,  # noqa: E402
                    boot_weights, bootstrap_proteins, cluster_of, concordance, die, entry_weighted, identity_strata,
                    labelled_lists,
                    ligand_tables, mcc, mw_auc, oof_tables, protein_of, require_path, run_cv,
                    structure_ligand_strata)

KS_REPEAT = (1, 10, 100)
Z_POS, GA_POS = 2.0, 0.59         # positive labels: ProBiS Z-score, G-LoSA GA-score
SIZE = [(10, 20, "10-19"), (20, 30, "20-29"), (30, 66, ">=30")]


def pooled_mcc(m, y, pred, k):
    sel = (m["rank"] <= k).to_numpy() & np.isfinite(y)
    t, p = y[sel].astype(bool), pred[sel]
    return mcc((t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum())


def repeats(m, top1, groups, grid, n_repeats):
    """OOF thresholds and MCCs over shuffled GroupKFold(5) splits (random_state 0 .. n_repeats - 1)."""
    y = m.truth.to_numpy()
    reps = []
    for r in range(n_repeats):
        f_r, _, pred_r = run_cv(top1, m, grid, GroupKFold(n_splits=5, shuffle=True, random_state=r), groups)
        th = [f["threshold"] for f in f_r]
        reps.append(dict(repeat=r, thresholds=";".join(f"{v:g}" for v in th), median_threshold=float(np.median(th)),
                         **{f"oof_mcc_top{k}": pooled_mcc(m, y, pred_r, k) for k in KS_REPEAT}))
    reps = pd.DataFrame(reps)
    q = lambda s: f"{s.min():g} / {s.quantile(.025):g} / {s.median():g} / {s.quantile(.975):g} / {s.max():g}"
    summ = dict(n_repeats=n_repeats,
                all_fold_thresholds_min_p2_5_median_p97_5_max=q(pd.Series(
                    [float(v) for s in reps.thresholds for v in s.split(";")])),
                median_threshold_min_p2_5_median_p97_5_max=q(reps.median_threshold),
                **{f"oof_mcc_top{k}_min_median_max": f"{reps[f'oof_mcc_top{k}'].min():.4f} / "
                   f"{reps[f'oof_mcc_top{k}'].median():.4f} / {reps[f'oof_mcc_top{k}'].max():.4f}"
                   for k in KS_REPEAT})
    return reps, summ


def p1_stat(S):
    TP, FP, FN, TN = (np.asarray(S[k], float) for k in ("TP", "FP", "FN", "TN"))
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = (TP + FN) / (TP + FP + FN + TN)
    return dict(MCC=mcc(TP, FP, FN, TN), P1=p1)


def summarise_top1(d, stype, stratum, n_boot=10000):
    """Top-1 MCC and P@1 of one query set (d: one row per query, truth, pred, cluster) with a cluster bootstrap."""
    t, p = d.truth.to_numpy(bool), d.pred.to_numpy(bool)
    c = pd.DataFrame({"cluster": d.cluster.to_numpy(), "TP": (t & p).astype(int), "FP": (~t & p).astype(int),
                      "FN": (t & ~p).astype(int), "TN": (~t & ~p).astype(int)})
    tot = c.sum(numeric_only=True)
    v = p1_stat({k: float(tot[k]) for k in ("TP", "FP", "FN", "TN")})
    ci = bootstrap_proteins(c, p1_stat, n=n_boot, seed=0)
    r = dict(stratum_type=stype, stratum=stratum, n_queries=len(d), n_clusters=int(d.cluster.nunique()),
             mcc=float(v["MCC"]), mcc_ci_lo=ci["MCC"][0], mcc_ci_hi=ci["MCC"][1], p1=float(v["P1"]),
             p1_ci_lo=ci["P1"][0], p1_ci_hi=ci["P1"][1], TP=int(tot.TP), FP=int(tot.FP), FN=int(tot.FN),
             TN=int(tot.TN))
    r["mcc_defined"] = bool((r["TP"] + r["FP"]) * (r["TP"] + r["FN"]) * (r["TN"] + r["FP"]) * (r["TN"] + r["FN"]) > 0)
    if not r["mcc_defined"]:
        r["mcc"] = r["mcc_ci_lo"] = r["mcc_ci_hi"] = np.nan
    return r


def strata_rows(ev, top1, a):
    """Supplementary Table S10 B (evaluation data): one method's per-query top-1 outcomes by identity, structure,
    ligand and site-size stratum."""
    t = top1[np.isfinite(top1.truth)].copy()
    t["truth"], t["pred"] = t.truth.astype(bool), t.pred.astype(bool)
    ident = t.protein.map(identity_strata(a.protein_strata))
    defs = [("sequence identity", "all", np.ones(len(t), bool)), ("sequence identity", "<30%", ident.eq("<30")),
            ("sequence identity", "30-80%", ident.isin(["30-40", "40-50", "50-80"]))]
    qs = structure_ligand_strata(a.query_strata).set_index("q_idx")
    for col, stype, levels in [("structure_stratum", "max Foldseek TM-score to training",
                                ["TM < 0.5", "TM 0.5-0.8", "TM >= 0.8"]),
                               ("ligand_stratum", "query ligand", ["same_ccd", "new_ccd_similar", "new_ccd_novel"])]:
        v = t.q_idx.map(qs[col])
        defs += [(stype, lev, v.eq(lev)) for lev in levels]
    size = ev.sizes("q")[t.q_idx.to_numpy()]
    defs += [("site size", name, (size >= lo) & (size < hi)) for lo, hi, name in SIZE]
    return [summarise_top1(t[np.asarray(sel)], stype, name) for stype, name, sel in defs if np.asarray(sel).any()]


def paired(frames, a_name, b_name, specs, n_boot=10000, seed=0):
    """MCC of method a minus method b over the same queries, paired cluster bootstrap (same resample matrix)."""
    out = []
    meta = frames[a_name].drop_duplicates("q_idx").set_index("q_idx").cluster
    cl, W = boot_weights(meta, n_boot, seed)
    pos = {c: i for i, c in enumerate(cl)}
    for col, k in specs:
        vals, boots = {}, {}
        for name in (a_name, b_name):
            m = frames[name]
            s = m[(m["rank"] <= k) & np.isfinite(m[col])]
            t, p = s[col].astype(bool).to_numpy(), s.pred.to_numpy(bool)
            c = pd.DataFrame({"q_idx": s.q_idx.to_numpy(), "TP": t & p, "FP": ~t & p, "FN": t & ~p,
                              "TN": ~t & ~p}).groupby("q_idx").sum().astype(int).reindex(meta.index)
            c4 = np.zeros((len(cl), 4))
            np.add.at(c4, meta.map(pos).to_numpy(), c[["TP", "FP", "FN", "TN"]].to_numpy(float))
            vals[name] = mcc(*c4.sum(0))
            boots[name] = mcc(*(W @ c4).T)
        bd = boots[a_name] - boots[b_name]
        out.append(dict(labels="ProBiS" if col == "truth" else "G-LoSA", k=k, method_a=a_name, method_b=b_name,
                        mcc_a=vals[a_name], mcc_b=vals[b_name], delta=vals[a_name] - vals[b_name],
                        ci_lo=float(np.percentile(bd, 2.5)), ci_hi=float(np.percentile(bd, 97.5)),
                        n_queries=len(meta), n_clusters=len(cl), n_boot=n_boot))
    return pd.DataFrame(out)


def cmd_table1(a):
    ev = EvalOutputs(a.eval_dir)
    od = os.path.join(a.out, "")
    os.makedirs(od, exist_ok=True)
    lists, _ = labelled_lists(ev, a)
    t1, s10, thr, strata, fixed = [], [], [], [], None
    for name, (m, step) in lists.items():
        t_p, t_g, th, (top1, groups, grid) = oof_tables(ev, m, step)
        if name == LENSEEK:          # Table 1: the top-k results classified at the fixed threshold
            mf = m.assign(pred=(m.cd <= th["fixed_threshold"]).to_numpy())
            fixed = concordance(mf, mf.truth.to_numpy()).assign(method=name, threshold=th["fixed_threshold"])
        t1.append(t_p.assign(method=name, fold_thresholds=th["fold_thresholds"]))
        s10.append(t_g.assign(method=name, fold_thresholds=th["fold_thresholds"]))
        thr.append(dict(method=name, **th))
        if name == LENSEEK:
            m[m["rank"] == 1].drop_duplicates("q_idx")[["q_idx", "query_key", "protein", "cluster", "cd", "truth", "fold",
                                                        "fold_threshold", "pred"]].to_csv(
                os.path.join(od, "top1_oof.csv"), index=False)
        print(f"{name}: fold thresholds {th['fold_thresholds']}; fixed threshold {th['fixed_threshold']:g}; "
              f"top-1 MCC {t_p.MCC.iloc[0]:.4f} [{t_p.MCC_ci_lo.iloc[0]:.4f}, {t_p.MCC_ci_hi.iloc[0]:.4f}]; "
              f"G-LoSA top-1 MCC {t_g.MCC.iloc[0]:.4f}")
        if name == LENSEEK and a.repeats:
            reps, summ = repeats(m, top1, groups, grid, a.repeats)
            reps.to_csv(os.path.join(od, "threshold_cv_repeats.csv"), index=False)
            pd.DataFrame([dict(fold_thresholds=th["fold_thresholds"], median_fold_threshold=th["median_fold_threshold"],
                               n_queries=th["n_queries"], n_entries=int(top1.protein.nunique()),
                               n_clusters=int(len(set(groups))), **summ)]).to_csv(
                os.path.join(od, "threshold_cv.csv"), index=False)
            print(f"{a.repeats} shuffled splits: thresholds {summ['all_fold_thresholds_min_p2_5_median_p97_5_max']}; "
                  f"OOF top-1 MCC {summ['oof_mcc_top1_min_median_max']}")
        if name in a.strata_methods.split(","):
            strata += [dict(method=name, **r) for r in strata_rows(ev, m[m["rank"] == 1].copy(), a)]
    first = ["method", "k", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "MCC", "MCC_ci_lo", "MCC_ci_hi"]
    if fixed is not None:
        fixed[first + [c for c in fixed.columns if c not in first]].to_csv(os.path.join(od, "table1_fixed.csv"),
                                                                           index=False)
        print(f"Table 1 (LEN-Seek, fixed threshold {fixed.threshold.iloc[0]:g})")
        print(fixed[first[1:10]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for tab, f in [(t1, "table1A.csv"), (s10, "tableS10A_glosa.csv")]:
        t = pd.concat(tab, ignore_index=True)
        t[first + [c for c in t.columns if c not in first]].to_csv(os.path.join(od, f), index=False)
    pd.DataFrame(thr).to_csv(os.path.join(od, "thresholds.csv"), index=False)
    pd.DataFrame(strata).to_csv(os.path.join(od, "tableS12B_strata.csv"), index=False)
    if a.paired:
        frames = {n: m for n, (m, _) in lists.items()}
        if a.paired not in frames:
            die(f"--paired {a.paired} is not a --method label")
        p = paired(frames, LENSEEK, a.paired, [("truth_glosa", 1), ("truth_glosa", 100)])
        p.to_csv(os.path.join(od, "paired.csv"), index=False)
        print(p.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.concat(t1)[["method", "k", "TP", "FP", "FN", "TN", "MCC"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("->", od)


def count_row(name, df, qcode, dcode, dice):
    d = df[["q_idx", "db_idx"]].copy()
    d["q_lig"] = [qcode[i] for i in d.q_idx]
    d["t_lig"] = [dcode[i] for i in d.db_idx]
    d["dice"] = [dice(x, y) for x, y in zip(d.q_lig, d.t_lig)]
    ok = d.dice.notna()
    return dict(method=name, n_queries=int(d.q_idx.nunique()), n_pairs=int(len(d)),
                n_db_sites=int(d.db_idx.nunique()), n_ligands=int(d.loc[d.t_lig != "", "t_lig"].nunique()),
                n_links=int(d[["q_lig", "db_idx"]].drop_duplicates().shape[0]), n_pairs_with_fp=int(ok.sum()),
                n_dice_ge_046=int((d.dice[ok] >= 0.46).sum()),
                pct_dice_ge_046=float(100 * (d.dice[ok] >= 0.46).mean()) if ok.any() else np.nan), d


def own_top(ev, lab, which, k=100):
    """ProBiS (Z) or G-LoSA (GA) own top-k of the scan: status-1 sites by score (descending, ties by database index),
    first k, kept at Z >= 2 / GA >= 0.59."""
    rec = []
    for qi in ev.sel:
        st, z, gst, ga = lab.row(qi)
        s, v, cut = (st, z, 2.0) if which == "probis" else (gst, ga, 0.59)
        idx = np.nonzero(s == 1)[0]
        o = idx[np.lexsort((idx, -v[idx]))][:k]
        rec.append(pd.DataFrame({"q_idx": qi, "db_idx": o[v[o] >= cut]}))
    return pd.concat(rec, ignore_index=True)


def volume_matched(dt, n_target):
    v = np.sort(dt.cd.to_numpy())
    return float((v[n_target - 1] + v[n_target]) / 2) if 0 < n_target < len(v) else float(v[-1])


def cmd_counts(a):
    ev = EvalOutputs(a.eval_dir)
    thr = pd.read_csv(require_path(os.path.join(a.out, "thresholds.csv"), "thresholds.csv", "table1 output"))
    thr = thr.set_index("method")
    lists, lab = labelled_lists(ev, a)
    qcode, dcode, dice = ligand_tables()
    ls = lists[LENSEEK][0]
    t_fix, t_med = thr.loc[LENSEEK, "fixed_threshold"], thr.loc[LENSEEK, "median_fold_threshold"]
    vm = a.volume_match if a.volume_match in lists else None
    rows, frames = [], {}

    def add(key, name, df):
        c, d = count_row(name, df, qcode, dcode, dice)
        rows.append(c)
        frames[key] = d

    add("lenseek", f"LEN-Seek, d <= {t_fix:g}", ls[ls.cd <= t_fix])
    add("probis_pool", "ProBiS, Z >= 2 (same pool)", ls[ls.truth == 1])
    add("probis_own", "ProBiS, Z >= 2 (own top-100 of the scan)", own_top(ev, lab, "probis"))
    add("glosa_pool", "G-LoSA, GA >= 0.59 (same pool)", ls[ls.truth_glosa == 1])
    add("glosa_own", "G-LoSA, GA >= 0.59 (own top-100 of the scan)", own_top(ev, lab, "glosa"))
    if vm:
        t_vm = volume_matched(lists[vm][0], int((ls.cd <= t_fix).sum()))
        add("vm", f"{vm}, d <= {t_vm:.4f} (same number of pairs)", lists[vm][0][lists[vm][0].cd <= t_vm])
    for name, (m, _) in list(lists.items())[1:]:
        t = thr.loc[name, "fixed_threshold"]
        add(name, f"{name}, d <= {t:g} (fixed threshold, own top-100)", m[m.cd <= t])
    add("pool", "Pool: LEN-Seek top-100", ls[np.isfinite(ls.truth)])
    add("lenseek_med", f"LEN-Seek, d <= {t_med:g} (median fold threshold)", ls[ls.cd <= t_med])
    if vm:
        t_vm2 = volume_matched(lists[vm][0], int((ls.cd <= t_med).sum()))
        add("vm_med", f"{vm}, d <= {t_vm2:.4f} (same number of pairs as d <= {t_med:g})",
            lists[vm][0][lists[vm][0].cd <= t_vm2])
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tableS11_counts.csv"), index=False)
    b, ov = frames["lenseek"], []
    for key in [k for k in ("probis_own", "vm") if k in frames]:
        d = frames[key]
        for what, ka, kb in [("pairs", set(zip(d.q_idx, d.db_idx)), set(zip(b.q_idx, b.db_idx))),
                             ("DB sites", set(d.db_idx), set(b.db_idx)),
                             ("ligands", set(d.t_lig) - {""}, set(b.t_lig) - {""}),
                             ("links", set(zip(d.q_lig, d.db_idx)), set(zip(b.q_lig, b.db_idx)))]:
            ov.append(dict(row=key, what=what, n_row=len(ka), n_lenseek=len(kb), both=len(ka & kb),
                           jaccard=len(ka & kb) / max(1, len(ka | kb))))
    pd.DataFrame(ov).to_csv(os.path.join(a.out, "tableS11_overlap.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(ov).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_fullscan(a):
    ev = EvalOutputs(a.eval_dir)
    lab = ScanLabels(ev, a.scan_dir, None, a.pairs_dir)
    rows = []
    for qi in ev.sel:
        st, z, _, _ = lab.row(qi)
        y = np.full(len(st), np.nan)
        y[st == 1] = (z[st == 1] >= 2.0).astype(float)
        y[st == 2] = 0.0
        ok = np.isfinite(y)
        yb = np.where(ok, y, 0).astype(bool)
        key = ev.q_keys[qi]
        r = dict(q_idx=int(qi), query_key=key, protein=protein_of(key), cluster=cluster_of(key), n_db=int(ok.sum()),
                 n_pos=int(yb[ok].sum()))
        if 0 < r["n_pos"] < r["n_db"]:
            sc = -np.asarray(ev.dist[qi], dtype=np.float64)[ok]
            r["lenseek_auc"] = mw_auc(sc, yb[ok])
            r["lenseek_pr_auc"] = average_precision_score(yb[ok], sc)
        rows.append(r)
    pq = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    pq.to_csv(os.path.join(a.out, "fullscan_per_query.csv"), index=False)
    usable = pq[(pq.n_pos > 0) & (pq.n_pos < pq.n_db)]
    res = entry_weighted(usable, ["lenseek_auc", "lenseek_pr_auc"], n_boot=1000)
    out = pd.DataFrame([dict(metric=c, **v) for c, v in res.items()])
    out.to_csv(os.path.join(a.out, "fullscan_summary.csv"), index=False)
    print(out[["metric", "entry_mean", "entry_ci_lo", "entry_ci_hi", "n_entries", "n_queries"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


def cmd_unrelated(a):
    """Positive labels among the unrelated pairs of scripts/labels.py null. ProBiS: Z rounded to 2 decimals, no
    alignment = negative, denominator = computed pairs (status 1 or 2). G-LoSA: denominator = pairs with a GA (status
    1). One line per set (null, all, non_null): quantiles of the score, the rate at Z >= 2 / GA >= 0.59 with its count,
    the score thresholds for 1 % and 0.1 %, per-query rate quantiles."""
    nz = np.load(require_path(a.null, "--null", "LENSEEK_OUT"), allow_pickle=True)
    Q, N = nz["shape"]
    null = np.unpackbits(nz["null"], axis=1)[:, :N].astype(bool)
    os.makedirs(a.out, exist_ok=True)
    if a.scan:
        st = np.load(f"{a.scan}/all_status.npy")
        z = np.round(((np.load(f"{a.scan}/all_raw.npy").astype(np.float64) - 1.958) / 2.21468), 2)
        assert null.shape == st.shape
        done = (st == 1) | (st == 2)
        lines = []
        for name, m in (("null", null), ("all", np.ones_like(null)), ("non_null", ~null)):
            m = m & done
            n = int(m.sum())
            aligned = m & (st == 1)
            za = z[aligned]
            pos = m & (st == 1) & (z >= Z_POS)
            fp_all = pos.sum() / max(n, 1)
            fp_aln = pos.sum() / max(int(aligned.sum()), 1)
            qs = np.quantile(za, [0.5, 0.9, 0.95, 0.99, 0.999]) if len(za) else [np.nan] * 5
            zz = np.where(aligned, z, -np.inf)[m]
            thr = [np.quantile(zz, 1 - p) for p in (0.01, 0.001)] if n else [np.nan, np.nan]
            perq = np.array([(pos[i].sum() / max(int(m[i].sum()), 1)) for i in range(Q) if m[i].any()])
            pq = np.quantile(perq, [0.25, 0.5, 0.75, 0.95]) if len(perq) else [np.nan] * 4
            lines.append(f"{name}\tpairs={n}\taligned={int(aligned.sum())} ({aligned.sum() / max(n, 1):.4f})\t"
                         f"Z_q50/90/95/99/99.9 (aligned)={'/'.join(f'{x:.2f}' for x in qs)}\t"
                         f"FP@Z>=2 all={fp_all:.5f} aligned={fp_aln:.5f} (n={int(pos.sum())})\t"
                         f"Z for FP 1%={thr[0]:.2f} 0.1%={thr[1]:.2f}\tper-query FP q25/50/75/95="
                         f"{'/'.join(f'{x:.4f}' for x in pq)}\tqueries={len(perq)}")
        open(os.path.join(a.out, "unrelated_probis.tsv"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    if a.glosa_scan:
        ga, st = np.load(f"{a.glosa_scan}/all_ga.npy"), np.load(f"{a.glosa_scan}/all_status.npy")
        assert null.shape == st.shape
        ok = st == 1
        lines = []
        for name, m in (("null", null), ("all", np.ones_like(null)), ("non_null", ~null)):
            n_all, n_ok = int(m.sum()), int((m & ok).sum())
            g = ga[m & ok]
            pos = int((g >= GA_POS).sum())
            qs = np.quantile(g, [0.5, 0.9, 0.95, 0.99, 0.999]) if len(g) else [np.nan] * 5
            thr = [np.quantile(g, 1 - p) for p in (0.01, 0.001)] if len(g) else [np.nan, np.nan]
            perq = np.array([((ga[i][m[i] & ok[i]] >= GA_POS).sum() / max(int((m[i] & ok[i]).sum()), 1))
                             for i in range(Q) if (m[i] & ok[i]).any()])
            pq = np.quantile(perq, [0.25, 0.5, 0.75, 0.95]) if len(perq) else [np.nan] * 4
            other = {c: int((m & (st == c)).sum()) for c in range(8) if c != 1 and (m & (st == c)).any()}
            lines.append(f"{name}\tpairs={n_all}\twith_GA={n_ok}\tother_status={other}\tGA_q50/90/95/99/99.9="
                         + "/".join(f"{x:.4f}" for x in qs) + f"\tFP@GA>=0.59={pos / max(n_ok, 1):.5f} (n={pos})\t"
                         f"GA for FP 1%={thr[0]:.4f} 0.1%={thr[1]:.4f}\tper-query FP q25/50/75/95="
                         + "/".join(f"{x:.4f}" for x in pq))
        open(os.path.join(a.out, "unrelated_glosa.tsv"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    env = os.environ.get
    for name, fn in [("table1", cmd_table1), ("counts", cmd_counts), ("fullscan", cmd_fullscan)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--eval-dir", default=env("LENSEEK_EVAL_DIR"))
        p.add_argument("--out", default=os.path.join(OUT_ROOT, "table1"))
        if name in ("table1", "counts", "fullscan"):
            p.add_argument("--scan-dir", default=env("LENSEEK_SCAN_DIR"), help="ProBiS scan (all_status, all_raw)")
            p.add_argument("--pairs-dir", default=env("LENSEEK_PAIRS_DIR"), help="site lists of the scans")
        if name in ("table1", "counts"):
            p.add_argument("--glosa-scan-dir", default=env("LENSEEK_GLOSA_SCAN_DIR"))
            p.add_argument("--method", nargs=3, action="append", default=[], metavar=("LABEL", "STEP", "PATH"),
                           help="comparison method: label, threshold grid step, distance .npy or top-100 TSV")
        if name == "table1":
            p.add_argument("--repeats", type=int, default=50, help="shuffled LEN-Seek splits (0 = none)")
            p.add_argument("--paired", default=None, help="method label for the paired difference from LEN-Seek")
            p.add_argument("--protein-strata", default=PROTEIN_STRATA,
                           help="identity strata (scripts/site_annotation.py identity-strata)")
            p.add_argument("--query-strata", default=QUERY_STRATA,
                           help="structure and ligand strata (scripts/site_annotation.py strata)")
            p.add_argument("--strata-methods", default="LEN-Seek,DeeplyTough,Folddisco",
                           help="comma-separated method labels of Supplementary Table S10 B")
        if name == "counts":
            p.add_argument("--volume-match", default="DeeplyTough",
                           help="method cut at the same number of pairs as LEN-Seek")
    p = sub.add_parser("unrelated")
    p.set_defaults(fn=cmd_unrelated)
    p.add_argument("--null", default=os.path.join(OUT_ROOT, "labels", "null", "null_set.npz"),
                   help="unrelated pairs (scripts/labels.py null)")
    p.add_argument("--scan", default=env("LENSEEK_SCAN_DIR"), help="ProBiS scan (all_status, all_raw)")
    p.add_argument("--glosa-scan", default=env("LENSEEK_GLOSA_SCAN_DIR"), help="G-LoSA scan (all_status, all_ga)")
    p.add_argument("--out", default=os.path.join(OUT_ROOT, "table1"))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
