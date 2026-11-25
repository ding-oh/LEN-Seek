#!/usr/bin/env python
"""Agreement of a model's Chamfer distance with the ProBiS and G-LoSA scores of its top-100 results: main-text
Figure 4 and Supplementary Figures S4-S6 and Tables S4 and S5.

  corr     Figure 4 (LEN-Seek) and Supplementary Figure S6 (Non-adapted Model): Chamfer distance against the ProBiS
           Z-score (A) and the G-LoSA GA-score (B) for the top-100 results of the primary queries, with Pearson and
           Spearman correlations
  mcc      Supplementary Figures S4 (LEN-Seek) and S5 (Non-adapted Model): MCC against ProBiS as a function of the
           Chamfer threshold, top-1 ... top-100
  table    Supplementary Tables S4 and S5: top-k concordance of the Non-adapted Model at the Chamfer threshold
           --threshold (48.5, the maximum of its top-1 MCC in Figure S5), computed as Table 1 (table1_concordance.py),
           with the database-wide ProBiS labels

Pairs: each query's top-100 database sites by distance (ties by database index), with the ProBiS Z-score and the
G-LoSA GA-score of the scored pair table (corr, mcc) or of the database-wide scans (table); panel A shows the pairs
with a ProBiS alignment, panel B every pair with a GA-score. Correlation intervals: 1,000 bootstrap resamples of 50 %
sequence clusters (seed 0). MCC curves: ProBiS positive at Z >= 2 (no alignment negative, failures excluded),
threshold grid 0, 1, ..., 100 plus --threshold; the maximum of each curve is its first maximum. Table intervals: 1,000
resamples of the clusters.

Outputs ($LENSEEK_OUT/score_agreement/, default ./out/score_agreement/; NAME = fig3/figS4 for LEN-Seek, figS6/figS5
for the Non-adapted Model):
  corr     NAME_corr.png/.pdf, NAME_corr_points.csv.gz, NAME_corr_stats.csv
  mcc      NAME_mcc_threshold.png/.pdf, NAME_mcc_curves.csv, NAME_mcc_max.csv
  table    tableS4_S5.csv

Inputs: see `python analysis/run.py -h`; read here: the model (--eval-dir, and --pairs with its scored top-100 pair
table: probis_z, probis_status, ga_all, ga_all_status; the Non-adapted Model through these two options); `table`
reads the database-wide scans instead of --pairs.

usage:
  python analysis/score_agreement.py corr --name fig3
  python analysis/score_agreement.py mcc --name figS4 --threshold 42.5
  python analysis/score_agreement.py corr --name figS6 --eval-dir <non-adapted eval dir> --pairs <its pairs>
  python analysis/score_agreement.py mcc --name figS5 --threshold 48.5 --eval-dir ... --pairs ...
  python analysis/score_agreement.py table --threshold 48.5 --eval-dir <non-adapted eval dir>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as st  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT_ROOT, EvalOutputs, ScanLabels, apply_style, classify_probis, cluster_of,  # noqa: E402
                    concordance, die, mcc, probis_truth, require_path, savefig, site_key, topk_rows)


PT_PER_IN = 72.27
TEXTWIDTH_MAIN, TEXTWIDTH_SUPP = 526.37598, 469.75502
# print width (pt) and the aspect ratio (height / width) of the manuscript figures
PLACEMENT = {"fig3": (TEXTWIDTH_MAIN, 824 / 2014), "figS6": (0.9 * TEXTWIDTH_SUPP, 778 / 1993),
             "figS4": (0.8 * TEXTWIDTH_SUPP, 1077 / 1762), "figS5": (0.8 * TEXTWIDTH_SUPP, 1077 / 1788)}
FS_CORR = {"fig3": dict(title=7.9, xlabel=7.2, ylabel=6.8, ticks=3.9, legend=4.3, panel=7.5),
           "figS6": dict(title=6.4, xlabel=5.8, ylabel=5.5, ticks=3.2, legend=3.5, panel=6.1)}
FS_MCC = dict(title=6.4, xlabel=6.4, ylabel=6.4, ticks=5.3, legend=5.3)
COLORS = ["red", "green", "blue", "purple", "black", "gray"]
KS = [1, 5, 10, 20, 50, 100]


def load_pair_scores(tsv):
    """Scored top-100 pairs reduced to the ProBiS Z / G-LoSA GA columns (query_path, target_path, probis_z,
    probis_status, ga_all, ga_all_status) and their classes, keyed by site keys; stops on pending statuses."""
    require_path(tsv, "--pairs", "LENSEEK_PAIRS")
    p = pd.read_csv(tsv, sep="\t", low_memory=False, usecols=["query_path", "target_path", "probis_z",
                                                               "probis_status", "ga_all", "ga_all_status"],
                    dtype={"probis_status": str, "ga_all_status": str})
    for c in ("probis_status", "ga_all_status"):
        if p[c].fillna("").eq("pending").any():
            die(f"{tsv}: rows with {c} = pending (scoring not finished)")
    p["query_key"] = p.query_path.map(site_key)
    p["target_key"] = p.target_path.map(site_key)
    p["probis_z"] = pd.to_numeric(p.probis_z, errors="coerce")
    p["ga"] = pd.to_numeric(p.ga_all, errors="coerce")
    p["probis_class"] = classify_probis(p.probis_z, p.probis_status).to_numpy()
    p["glosa_class"] = np.where(np.isfinite(p.ga.to_numpy(dtype=float)) & p.ga_all_status.fillna("").eq("ok"),
                                "ok", "failure")
    return p.drop_duplicates(["query_key", "target_key"])


def attach_pair_scores(rows, pairs):
    """Join the columns of load_pair_scores onto top-k rows; every top-k row must be scored."""
    m = rows.merge(pairs[["query_key", "target_key", "probis_z", "probis_class", "ga", "glosa_class"]],
                   on=["query_key", "target_key"], how="left")
    if m.probis_class.isna().any():
        die(f"{int(m.probis_class.isna().sum())} of {len(m)} top-k rows have no row in the pair TSV")
    return m


def figsize(fig_id):
    """(width, height) in inches at the manuscript placement."""
    w, aspect = PLACEMENT[fig_id]
    return w / PT_PER_IN, w * aspect / PT_PER_IN


def scored_rows(a):
    ev = EvalOutputs(a.eval_dir)
    return attach_pair_scores(topk_rows(ev, 100), load_pair_scores(a.pairs))


def boot_corr(x, y, keys, n=1000, seed=0):
    """Cluster bootstrap 95 % CIs of Pearson and Spearman (resampling 50 % sequence clusters)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(np.array([cluster_of(k) for k in keys]), return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(u))]
    pe, sp = [], []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        pe.append(np.corrcoef(x[idx], y[idx])[0, 1])
        sp.append(st.spearmanr(x[idx], y[idx])[0])
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return q(pe), q(sp)


def cmd_corr(a):
    m = scored_rows(a)
    od = a.out
    os.makedirs(od, exist_ok=True)
    m.to_csv(os.path.join(od, f"{a.name}_corr_points.csv.gz"), index=False)
    z, g = m[m.probis_class.eq("ok")], m[m.glosa_class.eq("ok")]
    res = {}
    for key, d, col in [("probis", z, "probis_z"), ("glosa", g, "ga")]:
        x, yv = d.cd.to_numpy(), d[col].to_numpy(dtype=float)
        res[key] = (st.pearsonr(x, yv)[0], st.spearmanr(x, yv)[0])
        res[key + "_ci"] = boot_corr(x, yv, d.query_key.to_numpy(), n=a.n_boot)
    fs = FS_CORR[a.name]
    apply_style(base=fs["ticks"], lw=0.5)
    sns.set_theme(style="whitegrid", rc={"axes.linewidth": 0.5, "grid.linewidth": 0.5,
                                         "xtick.labelsize": fs["ticks"], "ytick.labelsize": fs["ticks"]})
    fig = plt.figure(figsize=figsize(a.name))
    axA = fig.add_axes([0.058, 0.13, 0.423, 0.71])
    axB = fig.add_axes([0.535, 0.13, 0.433, 0.71])
    for ax, d, col, color, ylab, title, (p, s), fmt, loc, lab in [
            (axA, z, "probis_z", "darkgreen", "Z-Score", "ProBiS Z-Score – LEN-Seek Chamfer distance",
             res["probis"], "{} = {:.3f}", "right", "A"),
            (axB, g, "ga", "royalblue", "GA-Score", "G-LoSA GA-Score – LEN-Seek Chamfer distance",
             res["glosa"], "{}: {:.3f}", "left", "B")]:
        ax.scatter(d.cd, d[col], s=2.0, alpha=0.3, color=color, edgecolors="none", rasterized=True)
        b = np.polyfit(d.cd, d[col], 1)
        xx = np.array([d.cd.min(), d.cd.max()])
        ax.plot(xx, np.polyval(b, xx), color="crimson", linewidth=0.75)
        ax.set_xlabel("Chamfer Distance", fontsize=fs["xlabel"], labelpad=2)
        ax.set_ylabel(ylab, fontsize=fs["ylabel"], labelpad=2)
        ax.tick_params(labelsize=fs["ticks"], pad=1.5)
        ax.text(0.99 if loc == "right" else 0.01, 0.985, fmt.format("Pearson", p) + "\n" + fmt.format("Spearman", s),
                transform=ax.transAxes, va="top", ha=loc, fontsize=fs["legend"], linespacing=1.0,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5))
        fig.text(ax.get_position().x0 + 0.075, 0.9, title, fontsize=fs["title"], va="baseline")
        fig.text(ax.get_position().x0 - 0.025, 0.9, lab, fontsize=fs["panel"], fontweight="bold", va="baseline")
    savefig(fig, os.path.join(od, f"{a.name}_corr"))
    sns.reset_orig()
    r = dict(topk=100, n_queries=int(m.q_idx.nunique()), n_proteins=int(m.protein.nunique()), n_rows=len(m),
             n_probis=len(z), n_glosa=len(g), probis_pearson=res["probis"][0], probis_spearman=res["probis"][1],
             probis_pearson_ci=res["probis_ci"][0], probis_spearman_ci=res["probis_ci"][1],
             glosa_pearson=res["glosa"][0], glosa_spearman=res["glosa"][1],
             glosa_pearson_ci=res["glosa_ci"][0], glosa_spearman_ci=res["glosa_ci"][1])
    pd.DataFrame([r]).to_csv(os.path.join(od, f"{a.name}_corr_stats.csv"), index=False)
    print(pd.DataFrame([r]).T.to_string(header=False))


def cmd_mcc(a):
    m = scored_rows(a)
    y = probis_truth(m)
    thr = np.arange(0.0, 100.0 + 1.0, 1.0)
    if a.threshold is not None:
        thr = np.union1d(np.round(thr, 9), [a.threshold])
    cd, rank = m.cd.to_numpy(), m["rank"].to_numpy()
    curves, best = {"threshold": thr}, []
    for k in KS:
        sel = (rank <= k) & np.isfinite(y)
        t, d = y[sel].astype(bool), cd[sel]
        p = d[None, :] <= thr[:, None]
        curve = mcc((p & t).sum(1), (p & ~t).sum(1), (~p & t).sum(1), (~p & ~t).sum(1))
        curves[f"top{k}"] = curve
        i = int(np.argmax(curve))
        best.append(dict(k=k, max_mcc=curve[i], threshold=thr[i], n_pairs=int(sel.sum()),
                         n_queries=int(m.q_idx[sel].nunique())))
    od = a.out
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(curves).to_csv(os.path.join(od, f"{a.name}_mcc_curves.csv"), index=False)
    best = pd.DataFrame(best)
    best.to_csv(os.path.join(od, f"{a.name}_mcc_max.csv"), index=False)
    print(best.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fs = FS_MCC
    apply_style(base=fs["ticks"], lw=0.6)
    fig = plt.figure(figsize=figsize(a.name))
    ax = fig.add_axes([0.075, 0.13, 0.905, 0.79])
    for i, k in enumerate(KS):
        ax.plot(thr, curves[f"top{k}"], color=COLORS[i], linewidth=1.1,
                label=f"Top-{k} (Max MCC: {best.max_mcc.iloc[i]:.4f})")
    title = "Estimated MCC With ProBiS Results by Chamfer Distance Threshold" + (" (Ablation)" if a.name == "figS5" else "")
    ax.set_title(title, fontsize=fs["title"], pad=6)
    ax.set_xlabel("Chamfer Distance Threshold", fontsize=fs["xlabel"], labelpad=8)
    ax.set_ylabel("Estimated MCC", fontsize=fs["ylabel"])
    ax.legend(fontsize=fs["legend"], loc="upper right" if a.name != "figS5" else "upper left")
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(od, f"{a.name}_mcc_threshold"))


def cmd_table(a):
    ev = EvalOutputs(a.eval_dir)
    m = ScanLabels(ev, a.scan_dir, None, a.pairs_dir).label(topk_rows(ev, 100))
    m["pred"] = (m.cd <= a.threshold).to_numpy()
    t = concordance(m, m.truth.to_numpy()).assign(threshold=a.threshold)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "tableS4_S5.csv"), index=False)
    print(f"Chamfer distance <= {a.threshold:g}")
    print(t[["k", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "MCC"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    env = os.environ.get
    for name, fn in [("corr", cmd_corr), ("mcc", cmd_mcc), ("table", cmd_table)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--eval-dir", default=env("LENSEEK_EVAL_DIR"))
        p.add_argument("--out", default=os.path.join(OUT_ROOT, "score_agreement"))
        if name == "table":
            p.add_argument("--scan-dir", default=env("LENSEEK_SCAN_DIR"))
            p.add_argument("--pairs-dir", default=env("LENSEEK_PAIRS_DIR"))
            p.add_argument("--threshold", type=float, required=True, help="Chamfer distance classified as similar (<=)")
            continue
        p.add_argument("--pairs", default=env("LENSEEK_PAIRS"), help="scored top-100 pair table of the model")
        if name == "corr":
            p.add_argument("--name", default="fig3", choices=["fig3", "figS6"])
            p.add_argument("--n-boot", type=int, default=1000, help="resamples of the intervals (default 1,000)")
        else:
            p.add_argument("--name", default="figS4", choices=["figS4", "figS5"])
            p.add_argument("--threshold", type=float, default=None, help="extra grid point (the fixed threshold)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
