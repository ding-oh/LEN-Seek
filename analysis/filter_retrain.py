#!/usr/bin/env python
"""Retraining without the ligand filters (Supplementary Section S1, 'Model trained without the ligand filters', and the
row 'Trained without the ligand filters' of Supplementary Table S14).

Rule: F = primary queries whose ligand the training-data filters remove (filtered = 1 in
data/query_sets/query_ligands.tsv, lenseek.exclusion.valid_ligand), R = the other primary queries. If F does worse
than R on the whole-database ligand ROC-AUC, the model is trained once more without the filters
(scripts/filter_free_data.py) and the change of the difference is read. Steps (subcommands):
  compare   D = entry-weighted mean(F) - mean(R) of a per-query AUC column (column lenseek_raw__auc of the per-query
            file of fig4_ligand.py grid); queries without a positive or a negative are dropped; resampling units (PDB
            entry, sequence cluster >= 50 %) drawn with replacement, 10,000 times, seed 0, the queries of the drawn
            units split into F and R (replicates with an empty group discarded); lower_in_F when the upper 95 % bound
            of D < 0.
  sets      set (b): primary queries whose protein gains a training protein with fident >= 0.8 at query coverage >= 0.8
            from the entries that only the filter-free data contains (did_exclude_ge80.tsv), from the identity search
            of scripts/filter_free_data.py identity.
  did       DiD = D(retrained) - D(main_s0) on set (a) = primary set and set (b) = primary set minus did_exclude_ge80;
            (1) DiD > 2 x SD_seed(D) (sample s.d. of D over main_s0/s1/s2) and (2) lower 95 % bound of DiD > 0, paired
            cluster bootstrap (the same draw for both models; 10,000, seed 0); both in (a) and (b) -> 'detected', in
            one -> 'detected_a_only' / 'detected_b_only', in neither -> 'not_detected' (the filters alone do not explain
            the difference). Read on the cluster50 rows.

Inputs: see `python analysis/run.py -h`; read here: the per-query files of fig4_ligand.py grid (one per model) and
$LENSEEK_OUT/filter_free_data/identity/eval_vs_new.m8 (scripts/filter_free_data.py identity).
Outputs: $LENSEEK_OUT/filter_retrain/ (default ./out/filter_retrain/).

usage: python analysis/filter_retrain.py compare --per-query <per-query csv of the model> [--tag main_s0]
       python analysis/filter_retrain.py sets
       python analysis/filter_retrain.py did --retrain <csv> --base <csv> --seed <s0> --seed <s1> --seed <s2>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLUSTER_FILE, OUT_ROOT, QUERY_SETS, boot_weights, die, primary_excluded  # noqa: E402

QUERIES = os.path.join(QUERY_SETS, "query_ligands.tsv")
OUT = os.path.join(OUT_ROOT, "filter_retrain")
EVAL = os.environ.get("LENSEEK_EVAL_DIR", "")
COL = "lenseek_raw__auc"
COLS = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits".split(",")


# ------------------------------------------------------------------ compare: D = mean(F) - mean(R)
def prepare(pq, col, qset, queries=QUERIES):
    lst = pd.read_csv(queries, sep="\t", comment="#", dtype=str, keep_default_na=False)
    cl = pd.read_csv(CLUSTER_FILE, sep="\t", comment="#", dtype=str)
    d = pq[["query_key", col, "n_pos", "n_db"]].copy()
    d["entry"] = d.query_key.str.split("/").str[0].str.lower()
    if qset == "primary":
        d = d[~d.entry.isin(primary_excluded())]
    d["usable"] = (d.n_pos > 0) & (d.n_pos < d.n_db) & np.isfinite(d[col].astype(float))
    d["group"] = np.where(d.query_key.isin(set(lst[lst.filtered == "1"].query_key)), "F", "R")
    d["cluster50"] = d.query_key.map(dict(zip(cl.query_key, cl.cluster)))
    assert d.cluster50.notna().all()
    return d


def unit_sums(d, col, unit, units):
    """per unit: sum of entry means and number of entry means, for F and R (an entry split over clusters gives one
    mean per cluster)."""
    keys = ["entry", "group"] if unit == "entry" else [unit, "entry", "group"]
    em = d[d.usable].groupby(keys)[col].mean().reset_index()
    out = {}
    for g in ("F", "R"):
        e = em[em.group == g]
        out[g] = (e.groupby(unit)[col].sum().reindex(units).fillna(0).to_numpy(),
                  e.groupby(unit)[col].size().reindex(units).fillna(0).to_numpy())
    return out


def compare(d, col, unit, n_boot=10000, seed=0):
    u = d[d.usable]
    units = sorted(u[unit].unique())
    T = unit_sums(d, col, unit, units)
    (sF, nF), (sR, nR) = T["F"], T["R"]
    res = dict(unit=unit, n_units=len(units), n_queries_F=int((u.group == "F").sum()),
               n_queries_R=int((u.group == "R").sum()),
               n_entries_F=int(u[u.group == "F"].entry.nunique()), n_entries_R=int(u[u.group == "R"].entry.nunique()),
               n_clusters_F=int(u[u.group == "F"].cluster50.nunique()),
               n_clusters_R=int(u[u.group == "R"].cluster50.nunique()))
    if nF.sum() == 0 or nR.sum() == 0:
        res.update(D=np.nan, mean_F=np.nan, mean_R=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_boot=n_boot,
                   n_discarded=n_boot, lower_in_F=False)
        return res
    mF, mR = sF.sum() / nF.sum(), sR.sum() / nR.sum()
    _, W = boot_weights(units, n_boot, seed)
    dF, dR = W @ nF, W @ nR
    ok = (dF > 0) & (dR > 0)
    bF = (W[ok] @ sF) / dF[ok]
    bR = (W[ok] @ sR) / dR[ok]
    D = bF - bR
    lo, hi = np.percentile(D, [2.5, 97.5])
    res.update(mean_F=float(mF), mean_R=float(mR), D=float(mF - mR), ci_lo=float(lo), ci_hi=float(hi),
               mean_F_ci_lo=float(np.percentile(bF, 2.5)), mean_F_ci_hi=float(np.percentile(bF, 97.5)),
               mean_R_ci_lo=float(np.percentile(bR, 2.5)), mean_R_ci_hi=float(np.percentile(bR, 97.5)),
               n_boot=n_boot, n_discarded=int((~ok).sum()), p_boot_D_ge_0=float((D >= 0).mean()),
               lower_in_F=bool(hi < 0))
    return res


def cmd_compare(a):
    d = prepare(pd.read_csv(a.per_query), a.col, a.set, a.queries)
    t = pd.DataFrame([dict(metric=a.metric_name, column=a.col, query_set=a.set, stratum="all",
                           role="rule" if a.set == "primary" else "report",
                           **compare(d, a.col, unit, a.n_boot, a.rng_seed))
                      for unit in ["entry", "cluster50"]])
    out = a.out or os.path.join(OUT, "compare", a.tag)
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"filter_{a.metric_name}_{a.set}")
    t.to_csv(base + ".csv", index=False)
    rule = t[t.role == "rule"]
    dec = dict(metric=a.metric_name, per_query=os.path.abspath(a.per_query), query_set=a.set, n_queries_set=len(d),
               n_usable=int(d.usable.sum()), n_without_positive=int((~d.usable).sum()),
               rule_rows=rule[["unit", "D", "ci_lo", "ci_hi", "n_discarded", "lower_in_F"]].to_dict("records"),
               lower_in_F=bool(rule.lower_in_F.any()) if len(rule) else None)
    json.dump(dec, open(base + ".json", "w"), indent=1)
    print(t[["unit", "n_queries_F", "n_queries_R", "mean_F", "mean_R", "D", "ci_lo", "ci_hi", "lower_in_F"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


# ------------------------------------------------------------------ set (b)
def query_table(q_paths):
    qk = ["/".join(l.strip().split("/")[-2:]) for l in open(q_paths) if l.strip()]
    return pd.DataFrame({"query_key": qk, "protein": [k.split("/")[0].lower() for k in qk]})


def cmd_sets(a):
    H = pd.read_csv(a.m8, sep="\t", header=None, names=COLS)
    H["protein"] = H["query"].str.split("_").str[0].str.lower()
    H["entry"] = H["target"].str.rsplit("_", n=1).str[0]
    Q = query_table(a.q_paths)
    lst = pd.read_csv(a.queries, sep="\t", comment="#", dtype=str, keep_default_na=False)
    Q = Q[~Q.protein.isin(primary_excluded())].copy()
    Q["group"] = Q.query_key.isin(set(lst[lst.filtered == "1"].query_key)).map({True: "F", False: "R"})
    h80 = H[(H.qcov >= 0.8) & (H.fident >= 0.8)]
    rows = []
    for _, q in Q[Q.protein.isin(set(h80.protein))].iterrows():
        h = h80[h80.protein == q.protein].sort_values(["fident", "qcov"], ascending=False)
        b = h.iloc[0]
        rows.append(dict(query_key=q.query_key, protein=q.protein, group=q.group, best_fident=b.fident,
                         best_qcov=b.qcov, best_tcov=b.tcov, best_eval_chain=b["query"], best_train_chain=b.target,
                         n_train_chains_ge80=len(h), train_entries_ge80=",".join(sorted(set(h.entry)))))
    E = pd.DataFrame(rows).sort_values("query_key")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "did_exclude_ge80.tsv"), "w") as f:
        f.write("# set (b): primary-set queries that gain a training protein with fident >= 0.8 at query coverage >= 0.8 "
                "from the entries added by the filter-free data\n")
        E.to_csv(f, sep="\t", index=False)
    print(E[["query_key", "group", "best_fident", "best_train_chain"]].to_string(index=False))
    print(f"set (b) leaves out {len(E)} queries: {E.group.value_counts().to_dict()}")


# ------------------------------------------------------------------ did: the readout of the retrained model
def paired(dr, db, col, unit, n_boot, seed):
    units = sorted(dr[dr.usable][unit].unique())
    assert units == sorted(db[db.usable][unit].unique())
    Tr, Tb = unit_sums(dr, col, unit, units), unit_sums(db, col, unit, units)
    assert np.array_equal(Tr["F"][1], Tb["F"][1]) and np.array_equal(Tr["R"][1], Tb["R"][1])
    _, W = boot_weights(units, n_boot, seed)
    nF, nR = Tr["F"][1], Tr["R"][1]
    dF, dR = W @ nF, W @ nR
    ok = (dF > 0) & (dR > 0)

    def boot(T):
        return (W[ok] @ T["F"][0]) / dF[ok] - (W[ok] @ T["R"][0]) / dR[ok]
    did = boot(Tr) - boot(Tb)
    pr = Tr["F"][0].sum() / nF.sum() - Tr["R"][0].sum() / nR.sum()
    pb = Tb["F"][0].sum() / nF.sum() - Tb["R"][0].sum() / nR.sum()
    lo, hi = np.percentile(did, [2.5, 97.5])
    return dict(unit=unit, n_units=len(units), D_retrain=float(pr), D_base=float(pb), DiD=float(pr - pb),
                ci_lo=float(lo), ci_hi=float(hi), n_boot=n_boot, n_discarded=int((~ok).sum()),
                p_boot_DiD_le_0=float((did <= 0).mean()))


def evaluate(dr, db, seeds, col, n_boot, rng_seed, keep=None):
    """Rule rows (cluster50, then entry) for the queries in keep (None = the primary set); D(main_s0), D(retrained)
    and SD_seed(D) all on that query set and in the row's unit."""
    if keep is not None:
        dr, db = dr[dr.query_key.isin(keep)], db[db.query_key.isin(keep)]
        seeds = {k: v[v.query_key.isin(keep)] for k, v in seeds.items()}
    rows = []
    for unit in ["cluster50", "entry"]:
        p = paired(dr, db, col, unit, n_boot, rng_seed)
        pr, pb = compare(dr, col, unit), compare(db, col, unit)
        u = dr.usable
        p.update(n_queries=int(len(dr)), n_usable=int(u.sum()), n_F=int((u & (dr.group == "F")).sum()),
                 n_R=int((u & (dr.group == "R")).sum()),
                 n_entries_F=int(dr[u & (dr.group == "F")].entry.nunique()),
                 n_entries_R=int(dr[u & (dr.group == "R")].entry.nunique()),
                 n_clusters_F=int(dr[u & (dr.group == "F")].cluster50.nunique()),
                 n_clusters_R=int(dr[u & (dr.group == "R")].cluster50.nunique()),
                 D_retrain_ci=[pr["ci_lo"], pr["ci_hi"]], D_base_ci=[pb["ci_lo"], pb["ci_hi"]],
                 mean_F_retrain=pr["mean_F"], mean_R_retrain=pr["mean_R"], mean_F_base=pb["mean_F"],
                 mean_R_base=pb["mean_R"])
        Ds = [compare(v, col, unit)["D"] for v in seeds.values()]
        p.update(n_seeds=len(Ds), D_seeds=Ds, SD_seed_ddof1=float(np.std(Ds, ddof=1)),
                 SD_seed_ddof0=float(np.std(Ds, ddof=0)))
        p["exceeds_2sd"] = bool(p["DiD"] > 2 * p["SD_seed_ddof1"])
        p["interval_excludes_0"] = bool(p["ci_lo"] > 0)
        rows.append(p)
    return rows


VERDICTS = {
    "detected": "the filters lowered performance on these queries (both conditions in set a and in set b)",
    "detected_a_only": "detected in set a only: not concluded that the filters lowered performance; both values "
                       "reported",
    "detected_b_only": "detected in set b only: not concluded that the filters lowered performance; both values "
                       "reported",
    "not_detected": "the filters alone do not explain the difference (neither set meets both conditions)",
}


def verdict(rule_a, rule_b):
    """Joint verdict on the cluster50 rule rows of set a and set b."""
    a_ok = rule_a["exceeds_2sd"] and rule_a["interval_excludes_0"]
    b_ok = rule_b["exceeds_2sd"] and rule_b["interval_excludes_0"]
    code = {(True, True): "detected", (True, False): "detected_a_only", (False, True): "detected_b_only",
            (False, False): "not_detected"}[(a_ok, b_ok)]
    return dict(verdict_code=code, verdict=VERDICTS[code])


def cmd_did(a):
    if len(a.seed) != 3:
        die(f"did needs the per-query files of the three seeds of the original model (--seed x 3, got {len(a.seed)})")
    load = lambda p: prepare(pd.read_csv(p), a.col, "primary", a.queries)  # noqa: E731
    dr, db = load(a.retrain), load(a.base)
    m = dr.merge(db, on="query_key", suffixes=("_r", "_b"))
    assert len(m) == len(dr) == len(db) and (m.usable_r == m.usable_b).all() and (m.n_pos_r == m.n_pos_b).all() and \
        (m.group_r == m.group_b).all(), "queries, usable queries or labels differ between the models"
    seeds = {os.path.abspath(s): load(s) for s in a.seed}
    X = pd.read_csv(os.path.join(a.sets, "did_exclude_ge80.tsv"), sep="\t", comment="#")
    allq = set(dr.query_key)
    assert set(X.query_key) <= allq, "set (b) list"
    keep_b = sorted(allq - set(X.query_key))
    rows = []
    for set_name, keep in [("a_primary", None), ("b_primary_minus_ge80", keep_b)]:
        rows += [dict(set=set_name, role="rule", **p) for p in evaluate(dr, db, seeds, a.col, a.n_boot, a.rng_seed, keep)]
    rule = {r["set"]: r for r in rows if r["unit"] == "cluster50"}
    res = dict(retrain=os.path.abspath(a.retrain), base=os.path.abspath(a.base), seed_files=list(seeds),
               n_seeds=len(seeds), excluded_queries=X[["query_key", "group"]].to_dict("records"),
               **verdict(rule["a_primary"], rule["b_primary_minus_ge80"]), rule_unit="cluster50", rows=rows)
    out = a.out or os.path.join(OUT, "did")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "did.csv"), index=False)
    json.dump(res, open(os.path.join(out, "did.json"), "w"), indent=1, default=str)
    print(pd.DataFrame(rows)[["set", "unit", "n_F", "n_R", "D_retrain", "D_base", "DiD", "ci_lo", "ci_hi", "n_seeds",
                              "SD_seed_ddof1", "exceeds_2sd", "interval_excludes_0"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("verdict:", res["verdict_code"], "|", res["verdict"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare", help="D = mean(F) - mean(R), and whether F is lower")
    p.add_argument("--per-query", required=True)
    p.add_argument("--metric-name", default="ligand")
    p.add_argument("--col", default=COL)
    p.add_argument("--set", default="primary", choices=["primary", "full"])
    p.add_argument("--queries", default=QUERIES)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--tag", default="main_s0", help="model of the per-query file (output subfolder)")
    p.add_argument("--out", default=None, help="default $LENSEEK_OUT/filter_retrain/compare/<tag>")
    p = sub.add_parser("sets", help="set (b) of the readout")
    p.add_argument("--m8", default=os.path.join(OUT_ROOT, "filter_free_data", "identity", "eval_vs_new.m8"),
                   help="identity search of scripts/filter_free_data.py identity")
    p.add_argument("--q-paths", default=os.path.join(EVAL, "q_paths.txt"))
    p.add_argument("--queries", default=QUERIES)
    p.add_argument("--out", default=os.path.join(OUT, "sets"))
    p = sub.add_parser("did", help="difference in differences and the verdict")
    p.add_argument("--retrain", required=True)
    p.add_argument("--base", required=True, help="per-query file of main_s0")
    p.add_argument("--seed", action="append", default=[], help="per-query file of main_s0 / main_s1 / main_s2")
    p.add_argument("--sets", default=os.path.join(OUT, "sets"))
    p.add_argument("--queries", default=QUERIES)
    p.add_argument("--col", default=COL)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default $LENSEEK_OUT/filter_retrain/did")
    a = ap.parse_args()
    {"compare": cmd_compare, "sets": cmd_sets, "did": cmd_did}[a.cmd](a)


if __name__ == "__main__":
    main()
