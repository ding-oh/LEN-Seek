#!/usr/bin/env python
"""Node features and geometry in trained models: Supplementary Table S15 (Section S7).

A 2 x 2 of node features (Ankh, one-hot residue types) by geometry (with, without) over four models that reconstruct
their input node features (common.FACTORIAL, data/runs.yaml); without geometry the residue graph is fully connected,
the geometric edge features are removed and the FAPE and RMSD losses are off:
  Ankh, geometry      abl_nonadapted            Ankh, no geometry      abl_ankh_fullgraph_na
  one-hot, geometry   abl_onehot_na             one-hot, no geometry   abl_onehot_fullgraph_na
Each model and seed is read from the per-run outputs of the ablations result (analysis/run.py factorial makes those that
are missing), every model on the same queries:
  AU              active latent units (ablations.py check); a model without any is not compared
  ligand          whole-database ROC-AUC and RE@1% of fig4_ligand.py grid on the queries with a ligand-similar and
                  another database site (RE@1%: at least 5 positives), all and by identity to the training proteins
                  (< 30 %, >= 30 %)
  common pairs    ProBiS ROC-AUC, P@1 and out-of-fold top-1 MCC on the common pair set of baselines.py compare
                  (Supplementary Table S13 B: the database sites whose ProBiS pair did not fail and that every method
                  scores), with its folds; the four models are ranked as LEN-Seek (ties by database index, threshold
                  grid step 0.5)
  Table 1         out-of-fold top-1 MCC and P@1 of table1_concordance.py table1 (database-wide ProBiS labels, each
                  model with its own fold thresholds)
Values: entry-weighted means (ligand, ProBiS ROC-AUC) and pooled top-1 outcomes (MCC, P@1) with 95 % intervals from
10,000 resamples of the 50 % sequence clusters (seed 0), one resampling per query set for every model. Differences,
over the same queries and resamples: Ankh - one-hot with and without geometry, with - without geometry for Ankh and
for one-hot, and the interaction (Ankh, geometry - Ankh, no geometry) - (one-hot, geometry - one-hot, no geometry).
Detected: |difference| > 2 x the sample s.d. over main_s0/s1/s2 of the metric on the same queries, and the interval
excludes 0. Table S15 prints the ligand metrics, the ProBiS ROC-AUC and P@1 on the common pair set and the top-1 MCC
of Table 1; the CSV files also hold the top-1 MCC on the common pair set and P@1 of Table 1.

Inputs: see `python analysis/run.py -h`; read here: the evaluation folders of main_s0 (--eval-dir), the four models
(--model) and the other seeds (--seed), the comparison methods of the common pair set (--method: the Folddisco runs),
the per-run outputs $LENSEEK_OUT/ablations/<run>/degeneracy.json, $LENSEEK_OUT/fig4/<run>/ligand_grid_per_query.csv
and $LENSEEK_OUT/table1/<run>/top1_oof.csv, and the identity strata ($LENSEEK_OUT/site_annotation/protein_strata.tsv).
Outputs: $LENSEEK_OUT/factorial/ (default ./out/factorial/): cells.csv (values), contrasts.csv (differences and
interaction), per_query_common_pairs.csv (ProBiS ROC-AUC and top-1 of each model on the common pair set).

usage: python analysis/factorial.py --model abl_nonadapted=$LENSEEK_EVAL_ROOT/out_abl_nonadapted ... \\
           --seed main_s1=$LENSEEK_EVAL_ROOT/out_main_s1 --seed main_s2=... \\
           --method folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter ...
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, PredefinedSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FACTORIAL, FOLDS, FRAME_STRATA, G, LS_STEP, OUT_ROOT, PATHS, Baselines,  # noqa: E402
                    EvalOutputs, boot_weights, cluster_of, deg_state, die, entry_weighted, grid_file, identity_strata,
                    make_grid, mcc, oof_top1, probis_query, require_path, run_cv, site_key, t1_dir, usable)

SEEDS = ["main_s0", "main_s1", "main_s2"]
MODELS = [run for run, *_ in FACTORIAL]
LABEL = {run: f"{feat}, {geom}" for run, feat, geom in FACTORIAL}
AN, AF, ON, OF = MODELS
CONTRASTS = [  # name, terms (model, sign) in the order they are summed
    ("Ankh - one-hot, with geometry", [(AN, 1), (ON, -1)]),
    ("Ankh - one-hot, without geometry", [(AF, 1), (OF, -1)]),
    ("geometry, Ankh (with - without)", [(AN, 1), (AF, -1)]),
    ("geometry, one-hot (with - without)", [(ON, 1), (OF, -1)]),
    ("interaction", [(AN, 1), (AF, -1), (ON, -1), (OF, 1)]),
]
STRATA = FRAME_STRATA
CELLS, CONTR = [], []


def two_sd(vals):
    return float(2 * np.std(vals, ddof=1))


def pctl(b):
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def combine(values, terms):
    """sum(sign * value) over the terms of a contrast, in their order."""
    return sum(s * values[run] for run, s in terms)


def cell(metric, qset, stratum, run, value, lo, hi, nq, ne, nc):
    feat, geom = LABEL[run].split(", ")
    CELLS.append(dict(metric=metric, query_set=qset, stratum=stratum, run=run, node_features=feat, geometry=geom,
                      value=value, ci_lo=lo, ci_hi=hi, n_queries=nq, n_entries=ne, n_clusters=nc))


def contrast(name, metric, qset, stratum, delta, lo, hi, nq, ne, nc, tsd):
    det = bool(abs(delta) > tsd and (lo > 0 or hi < 0))
    CONTR.append(dict(contrast=name, metric=metric, query_set=qset, stratum=stratum, delta=float(delta),
                      ci_lo=float(lo), ci_hi=float(hi), n_queries=nq, n_entries=ne, n_clusters=nc, two_sd_seed=tsd,
                      detected=det))


def pooled(ind, idx, W):
    """Pooled top-1 outcomes (rows TP, FP, FN, TN) and their cluster resamples: MCC, resampled MCC, P@1 (share of
    queries whose top-1 is similar by the labels), resampled P@1."""
    c4 = np.zeros((W.shape[1], 4))
    np.add.at(c4, idx, ind)
    hits = np.stack([ind[:, 0] + ind[:, 2], np.ones(len(ind))], 1)
    c2 = np.zeros((W.shape[1], 2))
    np.add.at(c2, idx, hits)
    b2 = W @ c2
    return float(mcc(*ind.sum(0))), mcc(*(W @ c4).T), float(hits[:, 0].mean()), b2[:, 0] / b2[:, 1]


def outcomes(truth, pred):
    t, p = np.asarray(truth, bool), np.asarray(pred, bool)
    return np.stack([t & p, ~t & p, t & ~p, ~t & ~p], 1).astype(float)


def top1_cells(qset, pts, nq, ne, nc):
    """Cells and contrasts of the top-1 MCC and P@1 from pooled() of every model and seed."""
    for metric, i in [("top-1 MCC", 0), ("P@1", 2)]:
        tsd = two_sd([pts[s][i] for s in SEEDS])
        for run in MODELS:
            cell(metric, qset, "all", run, pts[run][i], *pctl(pts[run][i + 1]), nq, ne, nc)
        for name, terms in CONTRASTS:
            v = combine({r: pts[r][i] for r in MODELS}, terms)
            contrast(name, metric, qset, "all", v, *pctl(combine({r: pts[r][i + 1] for r in MODELS}, terms)), nq, ne,
                     nc, tsd)


# ------------------------------------------------------------------ AU
def latent_units():
    """AU of every model from its degeneracy check; stops when a model has none or was not checked."""
    for run in MODELS:
        state, j = deg_state(run)
        if state != "ok":
            die(f"{run}: {state} ({os.path.join(PATHS['out'], run, 'degeneracy.json')}); the 2 x 2 is read "
                "only when every model has active latent units (ablations.py check)")
        CELLS.append(dict(metric="AU", query_set="2,000 database sites", stratum="all", run=run,
                          node_features=LABEL[run].split(", ")[0], geometry=LABEL[run].split(", ")[1], value=j["au"]))


# ------------------------------------------------------------------ ligand ROC-AUC and RE@1%
def ligand(n_boot):
    """Ligand ROC-AUC and RE@1% per identity stratum on the queries every model scores (the labels of main_s0)."""
    grids = {}
    for run in SEEDS + MODELS:
        f = grid_file(run)
        if not os.path.exists(f):
            die(f"{f} missing: run fig4_ligand.py grid --eval-dir <out_{run}>")
        grids[run] = pd.read_csv(f).set_index("query_key")
    u = usable(grids["main_s0"].reset_index())
    L = pd.DataFrame({"query_key": u.query_key.to_numpy(), "protein": u.protein.to_numpy()})
    L["cluster"] = L.query_key.map(cluster_of)
    strata = identity_strata()
    L["id_stratum"] = L.protein.map(strata).fillna("missing")
    for run in SEEDS + MODELS:
        g = grids[run].reindex(L.query_key)
        if not ((g.n_db.to_numpy() == u.n_db.to_numpy()) & (g.n_pos.to_numpy() == u.n_pos.to_numpy())).all():
            die(f"{run}: ligand labels of its grid differ from those of main_s0")
        L[f"{run}__auc"], L[f"{run}__re1"] = g.lenseek_raw__auc.to_numpy(), g.lenseek_raw__re1.to_numpy()
    for m in ["auc", "re1"]:
        for name, terms in CONTRASTS:
            L[f"{name}__{m}"] = combine({r: L[f"{r}__{m}"] for r in MODELS}, terms)
    for m, metric in [("auc", "ligand ROC-AUC"), ("re1", "ligand RE@1%")]:
        for st, allowed in STRATA:
            d0 = L if allowed is None else L[L.id_stratum.isin(allowed)]
            tsd = two_sd([entry_weighted(d0.rename(columns={f"{s}__{m}": "x"}), ["x"], n_boot=1)["x"]["entry_mean"]
                          for s in SEEDS])
            d = d0[np.isfinite(d0[[f"{r}__{m}" for r in MODELS]].to_numpy(float)).all(1)].copy()
            res = entry_weighted(d, [f"{r}__{m}" for r in MODELS] + [f"{n}__{m}" for n, _ in CONTRASTS],
                                 n_boot=n_boot, seed=0)
            for run in MODELS:
                r = res[f"{run}__{m}"]
                cell(metric, "ligand", st, run, r["entry_mean"], r["entry_ci_lo"], r["entry_ci_hi"], r["n_queries"],
                     r["n_entries"], r["n_clusters"])
            for name, _ in CONTRASTS:
                r = res[f"{name}__{m}"]
                contrast(name, metric, "ligand", st, r["entry_mean"], r["entry_ci_lo"], r["entry_ci_hi"],
                         r["n_queries"], r["n_entries"], r["n_clusters"], tsd)


# ------------------------------------------------------------------ the common pair set of Table S13 B
def common_pairs(a, n_boot):
    """ProBiS ROC-AUC, top-1 MCC and P@1 of the models on the common pair set (the per-query comparison of
    baselines.py compare with the models and seeds as LEN-Seek models)."""
    ev = EvalOutputs(a.eval_dir, "primary")
    ls = {"main_s0": ev.dist}
    for spec in a.seed + a.model:
        name, d = spec.split("=", 1)
        e = EvalOutputs(d, "primary")
        if e.q_keys != ev.q_keys or e.db_keys != ev.db_keys:
            die(f"{d}: query/DB order differs from main_s0")
        if not np.isfinite(e.dist[ev.sel]).all():
            die(f"{d}: distances that are not finite would change the common pair set")
        ls[name] = e.dist
    missing = [r for r in SEEDS + MODELS if r not in ls]
    if missing:
        die(f"no evaluation folder of {', '.join(missing)} (--seed / --model)")
    scan = require_path(a.scan_dir, "--scan-dir", "LENSEEK_SCAN_DIR")
    gscan = require_path(a.glosa_scan_dir, "--glosa-scan-dir", "LENSEEK_GLOSA_SCAN_DIR")
    pairs = require_path(a.pairs_dir, "--pairs-dir", "LENSEEK_PAIRS_DIR")
    sq = [site_key(p) for p in open(os.path.join(pairs, "all_query_sites.txt")).read().split()]
    sd = [site_key(p) for p in open(os.path.join(pairs, "all_db_sites.txt")).read().split()]
    if sq != ev.q_keys or sd != ev.db_keys:
        die("scan site lists do not follow the evaluation order")
    dt_dir = require_path(a.dt_dir, "--dt-dir", "LENSEEK_DT_DIR")
    if [site_key(p) for p in open(os.path.join(dt_dir, "db_paths.txt")).read().split()] != ev.db_keys:
        die("DeeplyTough DB order differs")
    extra = {}
    for spec in a.method:
        name, d = spec.split("=", 1)
        if [site_key(p) for p in open(os.path.join(d, "db_paths.txt")).read().split()] != ev.db_keys:
            die(f"{d}: DB order differs")
        extra[name] = np.load(os.path.join(d, "dist.npy"), mmap_mode="r")
    bl = Baselines(ev)
    G.update(status=np.load(os.path.join(scan, "all_status.npy"), mmap_mode="r"),
             raw=np.load(os.path.join(scan, "all_raw.npy"), mmap_mode="r"),
             gst=np.load(os.path.join(gscan, "all_status.npy"), mmap_mode="r"),
             ga=np.load(os.path.join(gscan, "all_ga.npy"), mmap_mode="r"), q_keys=ev.q_keys, n_db=bl.n_db,
             dbidx=np.arange(bl.n_db), ls=ls, bl=bl, dt=np.load(os.path.join(dt_dir, "dist.npy"), mmap_mode="r"),
             extra=extra)
    with Pool(a.workers) as p:
        res = p.map(probis_query, [int(q) for q in ev.sel], chunksize=2)
    pq = pd.DataFrame([r for r, _ in res])
    if (pq.n_pending > 0).any():
        die(f"{int((pq.n_pending > 0).sum())} primary queries have pending ProBiS pairs")
    use = pq[pq.excluded == ""].copy()
    use["cluster"] = use.query_key.map(cluster_of)
    for name, terms in CONTRASTS:
        use[f"{name}__auc"] = combine({r: use[f"{r}__auc"] for r in MODELS}, terms)
    r = entry_weighted(use, [f"{x}__auc" for x in SEEDS + MODELS] + [f"{n}__auc" for n, _ in CONTRASTS],
                       n_boot=n_boot, seed=0)
    tsd = two_sd([r[f"{s}__auc"]["entry_mean"] for s in SEEDS])
    for run in MODELS:
        x = r[f"{run}__auc"]
        cell("ProBiS ROC-AUC", "common pairs", "all", run, x["entry_mean"], x["entry_ci_lo"], x["entry_ci_hi"],
             x["n_queries"], x["n_entries"], x["n_clusters"])
    for name, _ in CONTRASTS:
        x = r[f"{name}__auc"]
        contrast(name, "ProBiS ROC-AUC", "common pairs", "all", x["entry_mean"], x["entry_ci_lo"], x["entry_ci_hi"],
                 x["n_queries"], x["n_entries"], x["n_clusters"], tsd)
    # top-1 on the same queries: the folds of baselines.py compare (GroupKFold over the primary queries by cluster)
    groups = np.array([cluster_of(ev.q_keys[i]) for i in ev.sel])
    fold = np.full(len(ev.sel), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=FOLDS).split(ev.sel, groups=groups)):
        fold[te] = f
    use["fold"] = use.q_idx.map(dict(zip(ev.sel.tolist(), fold.tolist())))
    top1 = {q["q_idx"]: t for q, t in res if t is not None}
    cl, W = boot_weights(use.cluster, n_boot, 0)
    idx = use.cluster.map({c: i for i, c in enumerate(cl)}).to_numpy()
    pts = {}
    for run in SEEDS + MODELS:
        d1 = np.array([top1[q][run][0] for q in use.q_idx])
        y1 = np.array([top1[q][run][1] for q in use.q_idx])
        t1 = pd.DataFrame({"q_idx": use.q_idx.to_numpy(), "cd": d1, "truth": y1.astype(float)})
        _, _, pred = run_cv(t1, t1, make_grid(d1, LS_STEP), PredefinedSplit(use.fold.to_numpy()), None)
        pts[run] = pooled(outcomes(y1, pred), idx, W)
        use[f"{run}__top1_cd"], use[f"{run}__top1_label"] = d1, y1
        use[f"{run}__top1_db_idx"] = [top1[q][run][2] for q in use.q_idx]
        use[f"{run}__oof_pred"] = pred
    top1_cells("common pairs", pts, len(use), int(use.protein.nunique()), len(cl))
    keep = ["q_idx", "query_key", "protein", "cluster", "fold", "n_db", "n_pos"] + \
        [f"{run}__{c}" for run in MODELS for c in ("auc", "top1_cd", "top1_label", "top1_db_idx", "oof_pred")]
    use[keep].to_csv(os.path.join(a.out, "per_query_common_pairs.csv"), index=False)


# ------------------------------------------------------------------ out-of-fold top-1 in the format of Table 1
def table1_top1(n_boot):
    """Top-1 MCC and P@1 from the out-of-fold top-1 outcomes of each model's Table 1, on the queries of main_s0."""
    t = {}
    for run in SEEDS + MODELS:
        t[run] = oof_top1(run)
        if t[run] is None:
            die(f"no out-of-fold Table 1 of {run} in {t1_dir(run)} (table1_concordance.py table1)")
    base = t["main_s0"].reset_index(drop=True)
    for run in t:
        t[run] = t[run].set_index("query_key").reindex(base.query_key).reset_index()
        if not ((t[run].cluster == base.cluster) & (t[run].fold == base.fold)).all():
            die(f"{run}: queries, clusters or folds differ from main_s0")
    cl, W = boot_weights(base.cluster, n_boot, 0)
    idx = base.cluster.map({c: i for i, c in enumerate(cl)}).to_numpy()
    pts = {run: pooled(outcomes(t[run].truth, t[run].pred), idx, W) for run in t}
    top1_cells("Table 1", pts, len(base), int(base.protein.nunique()), len(cl))


# ------------------------------------------------------------------ the table
def fmt(v, lo, hi, nd, sign=False):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return f"{s} [{lo:.{nd}f}, {hi:.{nd}f}]"


ROWS = [("Ligand ROC-AUC", "ligand ROC-AUC", "ligand", "all", 3),
        ("  <30 % identity", "ligand ROC-AUC", "ligand", "<30", 3),
        ("  >=30 % identity", "ligand ROC-AUC", "ligand", ">=30", 3),
        ("Ligand RE@1%", "ligand RE@1%", "ligand", "all", 2),
        ("ProBiS ROC-AUC", "ProBiS ROC-AUC", "common pairs", "all", 3),
        ("Top-1 MCC", "top-1 MCC", "Table 1", "all", 3),
        ("P@1", "P@1", "common pairs", "all", 3)]


def show(cells, con):
    """The rows of Supplementary Table S15 (A. values, B. differences and interaction)."""
    c = cells.set_index(["metric", "query_set", "stratum", "run"])
    k = con.set_index(["metric", "query_set", "stratum", "contrast"])
    w = 24
    print("A. Values\n" + "".ljust(18) + "".join(LABEL[r].ljust(w) for r in MODELS))
    print("AU".ljust(18) + "".join(str(int(c.loc[("AU", "2,000 database sites", "all", r), "value"])).ljust(w)
                                   for r in MODELS))
    for label, m, q, st, nd in ROWS:
        print(label.ljust(18) + "".join(fmt(*c.loc[(m, q, st, r), ["value", "ci_lo", "ci_hi"]], nd).ljust(w)
                                        for r in MODELS))
    print("B. Differences (+ detected)\n" + "".ljust(18) +
          "".join(h.ljust(w + 4) for h in ("Ankh - one-hot, geom.", "Ankh - one-hot, no geom.", "geometry, Ankh",
                                           "geometry, one-hot", "interaction")))
    for label, m, q, st, nd in ROWS:
        cols = []
        for name, _ in CONTRASTS:
            x = k.loc[(m, q, st, name)]
            cols.append((fmt(x.delta, x.ci_lo, x.ci_hi, nd, sign=True) + (" +" if x.detected else "")).ljust(w + 4))
        print(label.ljust(18) + "".join(cols))
    print("twice the seed s.d.: " + ", ".join(f"{label.strip()} {k.loc[(m, q, st, 'interaction'), 'two_sd_seed']:.4f}"
                                              for label, m, q, st, _ in ROWS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"), help="main_s0 ($LENSEEK_EVAL_DIR)")
    ap.add_argument("--model", action="append", default=[], help="run=<eval dir> of each model of the 2 x 2")
    ap.add_argument("--seed", action="append", default=[], help="main_s1=<eval dir>, main_s2=<eval dir>")
    ap.add_argument("--method", action="append", default=[],
                    help="further method of the common pair set: name=<dir with dist.npy, db_paths.txt>, e.g. "
                         "folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter")
    ap.add_argument("--pairs-dir", default=os.environ.get("LENSEEK_PAIRS_DIR"))
    ap.add_argument("--scan-dir", default=os.environ.get("LENSEEK_SCAN_DIR"))
    ap.add_argument("--glosa-scan-dir", default=os.environ.get("LENSEEK_GLOSA_SCAN_DIR"))
    ap.add_argument("--dt-dir", default=os.environ.get("LENSEEK_DT_DIR"))
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "factorial"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    latent_units()
    ligand(a.n_boot)
    common_pairs(a, a.n_boot)
    table1_top1(a.n_boot)
    cells, con = pd.DataFrame(CELLS), pd.DataFrame(CONTR)
    cells["value"] = pd.Series([int(v) if m == "AU" else v for m, v in zip(cells.metric, cells.value)], dtype=object)
    cells.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    con.to_csv(os.path.join(a.out, "contrasts.csv"), index=False)
    show(cells, con)
    print("->", a.out)


if __name__ == "__main__":
    main()
