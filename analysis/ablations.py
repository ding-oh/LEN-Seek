#!/usr/bin/env python
"""Variants of LEN-Seek: Supplementary Table S14 and Section S7.

Every variant (common.RUNS: run, node features, training sites, group, row label) is evaluated like the original
model: its eval dir out_<run>/ is written by scripts/evaluate.py, in the database order of the original model
(--db-order; the language-model variants lack the sites without a chain embedding, which get NaN distances). Steps,
in order (each a subcommand):
  check      active latent units: variance of each latent dimension of mu over the valid residues of 2,000 database
             sites (default_rng(0), without replacement), AU = dimensions with variance > 0.01. AU = 0 is a latent
             collapse: the run is reported but no comparison containing it is read. -> ablations/<run>/degeneracy.json
  common-sites  the sites common to the language-model variants: database sites with a finite distance in every
             non-degenerate model of Ankh (original model), ESM2, ESM3, ESM C and no node features; per-query ligand
             ROC-AUC of these models and of the seeds re-ranked on them (Dice >= 0.46, CCD snapshot);
             --make-restricted writes common_sites_eval/out_<run>__common_sites/ (the other columns set to +inf) for
             the out-of-fold concordance on these sites.
  table      paired contrasts original model - variant (ligand ROC-AUC, entry-weighted; out-of-fold top-1 MCC with
             each model's own fold thresholds), paired sequence-cluster bootstrap (10,000 resamples, seed 0), 2 x the
             seed s.d. over main_s0/s1/s2, strata all and < 30 % identity to the training proteins; the identity strata
             of the ligand metrics, and the ligand ROC-AUC ranked among the database sites of an entry list for the
             runs whose grid has it (fig4_ligand.py grid --db-entries; strict30_s0 with the entries excluded from its
             training: Supplementary Section S5); the rows of Supplementary Table S14 and the seed rows of Table S7 B.
             Detected = |delta| > 2 x seed s.d. and the interval excludes zero. -> variants.csv, contrasts.csv,
             frame.csv, contrast_missing/
Between check and table every run gets, as the original model, its whole-database ligand metrics from
fig4_ligand.py grid --eval-dir out_<run> ($LENSEEK_OUT/fig4/<run>/ligand_grid_{per_query,summary}.csv) and its
out-of-fold concordance in the format of Table 1 from table1_concordance.py table1 --eval-dir out_<run> --out
$LENSEEK_OUT/table1/<run> --repeats 0 (table1A.csv, top1_oof.csv; labels from the database-wide ProBiS scan), the
latter also for every $LENSEEK_OUT/ablations/common_sites_eval/out_<run>__common_sites/ after common-sites
--make-restricted (analysis/run.py ablations runs these steps in this order).

Inputs: see `python analysis/run.py -h`; read here: the evaluation folders $LENSEEK_EVAL_ROOT/out_<run>/, the
per-run outputs of fig4_ligand.py and table1_concordance.py below $LENSEEK_OUT and the identity strata of
scripts/site_annotation.py identity-strata ($LENSEEK_OUT/site_annotation/protein_strata.tsv).
Outputs: $LENSEEK_OUT/ablations/ (default ./out/ablations/).

usage: python analysis/ablations.py check --eval-dir out_<run> --tag <run>
       python analysis/ablations.py common-sites --make-restricted
       python analysis/ablations.py table
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LENSEEK, OUT_ROOT, PROTEIN_STRATA, RUNS, EvalOutputs, auc_from_ranks,  # noqa: E402
                    bootstrap_proteins, bulk_dice, die, entry_weighted, fingerprints, identity_strata, load_cache,
                    mcc, protein_of, site_tables)

REF, SEEDS = "main_s0", ["main_s0", "main_s1", "main_s2"]
COMMON_SITE_RUNS = ["main_s0", "plm_esm2", "plm_esm3", "plm_esmc", "plm_none"]
N_BOOT = 10000
COLLAPSE = "latent collapse; values reflect ties and rounding, not information"
PATHS = dict(eval_root=os.environ.get("LENSEEK_EVAL_ROOT") or os.path.join(OUT_ROOT, "eval"),
             fig=os.path.join(OUT_ROOT, "fig4"), t1=os.path.join(OUT_ROOT, "table1"),
             out=os.path.join(OUT_ROOT, "ablations"), common=os.path.join(OUT_ROOT, "ablations", "common_sites_eval"),
             strata=PROTEIN_STRATA)


# ------------------------------------------------------------------ paths and run state
def eval_dir(run):
    """out_<run>/ of the eval root; out_<run>__common_sites/ of the common-sites root."""
    if run.endswith("__common_sites"):
        return os.path.join(PATHS["common"], f"out_{run}")
    return os.path.join(PATHS["eval_root"], f"out_{run}")


def grid_file(run, name="ligand_grid_per_query.csv"):
    """A whole-database ligand-metrics file of the run (fig4_ligand.py grid)."""
    return os.path.join(PATHS["fig"], run, name)


def t1_dir(run):
    """The run's out-of-fold Table 1 folder (table1_concordance.py table1 --out)."""
    return os.path.join(PATHS["t1"], run)


def t1_row(run):
    """LEN-Seek top-1 row (k = 1) of the run's table1A.csv, or None."""
    f = os.path.join(t1_dir(run), "table1A.csv")
    if not os.path.exists(f):
        return None
    t = pd.read_csv(f)
    t = t[(t.method == LENSEEK) & (t.k == 1)]
    return t.iloc[0] if len(t) else None


def deg_state(run):
    """('ok' | 'degenerate' | 'unchecked', degeneracy.json or None); an unchecked run is treated as degenerate."""
    f = os.path.join(PATHS["out"], run, "degeneracy.json")
    if not os.path.exists(f):
        return "unchecked", None
    try:
        with open(f) as fh:
            j = json.load(fh)
        au = int(j["au"])
    except (OSError, ValueError, KeyError, TypeError):
        return "unchecked", None
    return ("degenerate" if au == 0 else "ok"), j


_ID = {}


def id_stratum(protein):
    """Identity stratum of an evaluation protein to the training proteins (either coverage >= 0.5)."""
    if not _ID:
        _ID.update(identity_strata(PATHS["strata"]))
    return _ID.get(protein, "missing")


def usable(t):
    """Queries with a ligand fingerprint, at least one ligand-similar and one other database site."""
    return t[t.has_query_fp & (t.n_pos > 0) & (t.n_pos < t.n_db)]


# ------------------------------------------------------------------ check (active latent units)
def latent_spread(eval_dir, n_sites=2000, seed=0, au_thr=0.01):
    sub = os.path.join(eval_dir, "db_subset")
    if os.path.exists(os.path.join(eval_dir, "db_mu.npy")):
        mu_f, mk_f = os.path.join(eval_dir, "db_mu.npy"), os.path.join(eval_dir, "db_mask.npy")
    elif os.path.exists(os.path.join(sub, "db_mu.npy")):
        mu_f, mk_f = os.path.join(sub, "db_mu.npy"), os.path.join(sub, "db_mask.npy")
    else:
        die(f"no db_mu.npy in {eval_dir} (or db_subset/)")
    mu = np.load(mu_f, mmap_mode="r")
    mask = np.load(mk_f, mmap_mode="r")
    assert mu.shape[:2] == mask.shape, (mu.shape, mask.shape)
    n = mu.shape[0]
    order = np.sort(np.random.default_rng(seed).choice(n, n_sites, replace=False))   # the sample, read in file order
    m = np.asarray(mask[order], dtype=bool)
    x = np.asarray(mu[order], dtype=np.float32)[m].astype(np.float64)   # (valid residues, latent)
    var = x.var(axis=0)
    return dict(db_rows=int(n), latent_dim=int(mu.shape[2]), n_sites=n_sites,
                sample_idx_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
                n_valid_residues=int(len(x)), au=int((var > au_thr).sum()), total_var=float(var.sum()),
                n_dims_var_gt_1e3=int((var > 1e-3).sum()), max_dim_var=float(var.max()),
                median_dim_sd=float(np.median(np.sqrt(var))), mean_residue_norm=float(np.linalg.norm(x, axis=1).mean()))


def cmd_check(a):
    lat = latent_spread(os.path.abspath(a.eval_dir))
    res = dict(run=a.tag, latent=lat, au=lat["au"], total_var=lat["total_var"], degenerate=lat["au"] == 0,
               label=COLLAPSE if lat["au"] == 0 else "")
    od = os.path.join(PATHS["out"], a.tag)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "degeneracy.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"[check] {a.tag}: AU {res['au']}, total var {res['total_var']:.4g}" + (f"  ** {COLLAPSE} **" if res["degenerate"] else ""))


# ------------------------------------------------------------------ sites common to the language-model variants
def evaluated(runs):
    return [r for r in runs if os.path.exists(os.path.join(eval_dir(r), "dist.npy"))]


def cmd_common_sites(a):
    out = PATHS["out"]
    os.makedirs(out, exist_ok=True)
    runs_all = evaluated(COMMON_SITE_RUNS)
    state = {r: deg_state(r)[0] for r in runs_all}
    runs = [r for r in runs_all if state[r] == "ok"]
    info = dict(runs_evaluated=runs, runs_left_out={r: s for r, s in state.items() if s != "ok"})
    if REF not in runs:
        die("main_s0 is not usable (not evaluated, degenerate or unchecked): no common sites")
    seeds = evaluated(SEEDS[1:])
    models = runs + seeds
    evs = {r: EvalOutputs(eval_dir(r), "primary") for r in models}
    keys = evs[REF].db_keys
    assert all(e.db_keys == keys for e in evs.values()), "DB order differs between runs"
    fin = np.ones(len(keys), bool)
    for r in runs:   # NaN columns (sites a language-model variant lacks, evaluate.py --db-order): the same in every row
        fin &= np.isfinite(np.asarray(evs[r].dist[evs[r].sel[0]]))
    qs, ds = site_tables()
    cache = load_cache()
    db_code, q_code = ds.ligand.to_numpy(dtype=object), qs.ligand.to_numpy(dtype=object)
    ids = {c for c in list(db_code) + list(q_code) if c}
    fps = fingerprints(ids, cache)
    uniq = sorted(i for i in ids if fps.get(i) is not None)
    col = {c: j for j, c in enumerate(uniq)}
    db_col = np.array([col.get(c, -1) for c in db_code])
    keep = (db_col >= 0) & fin
    rows = []
    for qi in evs[REF].sel:
        code = q_code[qi]
        if not code or fps.get(code) is None:
            continue
        dq = np.array(bulk_dice(fps[code], [fps[u] for u in uniq]), dtype=np.float64)
        L = dq[db_col[keep]] >= 0.46
        if not 0 < L.sum() < len(L):
            continue
        for r in models:
            d = np.asarray(evs[r].dist[qi], dtype=np.float64)[keep]
            rows.append(dict(model=r, q_idx=int(qi), query_key=evs[r].q_keys[qi],
                             protein=protein_of(evs[r].q_keys[qi]), n_db=int(keep.sum()), n_pos=int(L.sum()),
                             auc=auc_from_ranks(rankdata(-d), L)))
    pd.DataFrame(rows).to_csv(os.path.join(out, "common_sites_per_query.csv"), index=False)
    excl = [k for k, f in zip(keys, fin) if not f]
    txt = ("\n".join(excl) + ("\n" if excl else "")).encode()
    with open(os.path.join(out, "common_sites_excluded.txt"), "wb") as fh:
        fh.write(txt)
    basis = hashlib.sha256(txt).hexdigest()
    info.update(seeds_evaluated=seeds, n_db=len(keys), n_common_sites=int(fin.sum()),
                n_common_sites_with_fp=int(keep.sum()), n_queries=int(len({r["query_key"] for r in rows})),
                excluded_sites_sha256=basis, restricted_dirs=[])
    if a.make_restricted and (~fin).any():
        for r in models:
            src, dst = eval_dir(r), eval_dir(r + "__common_sites")
            os.makedirs(dst, exist_ok=True)
            for f in ["q_paths.txt", "db_paths.txt", "q_mask.npy", "db_mask.npy"]:
                if not os.path.lexists(os.path.join(dst, f)):
                    os.symlink(os.path.join(os.path.abspath(src), f), os.path.join(dst, f))
            meta = dict(source=os.path.abspath(src), excluded_sites=int((~fin).sum()),
                        rule="non-common DB columns set to +inf", excluded_sites_sha256=basis)
            rf, df = os.path.join(dst, "RESTRICTED.json"), os.path.join(dst, "dist.npy")
            old = json.load(open(rf)) if os.path.exists(rf) else None
            if old != meta or not os.path.exists(df):
                d = np.array(np.load(os.path.join(src, "dist.npy"), mmap_mode="r"), dtype=np.float32)
                d[:, ~fin] = np.inf
                np.save(df, d)
                del d
                json.dump(meta, open(rf, "w"), indent=1)
            info["restricted_dirs"].append(dst)
    json.dump(info, open(os.path.join(out, "common_sites.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "restricted_dirs"}, indent=1))


# ------------------------------------------------------------------ contrasts
BASIS = {"common": "sites common to the language-model variants", "full": "full DB"}
NAMED = {  # name: (terms, group, role)
    "D_N": ([(REF, 1), ("plm_none", -1)], "language_model", "PLM node features as a whole"),
    "D_E2": ([(REF, 1), ("plm_esm2", -1)], "language_model", "Ankh vs ESM2"),
    "D_E3": ([(REF, 1), ("plm_esm3", -1)], "language_model", "Ankh vs ESM3"),
    "D_EC": ([(REF, 1), ("plm_esmc", -1)], "language_model", "Ankh vs ESM C"),
    "D_G": ([(REF, 1), ("abl_ankh_fullgraph", -1)], "structure", "geometry removed, full graph"),
    "D_G_na": ([(REF, 1), ("abl_ankh_fullgraph_na", -1)], "structure",
               "geometry removed, full graph, raw Ankh reconstruction target"),
    "D_abl_nonadapted": ([(REF, 1), ("abl_nonadapted", -1)], "structure",
                         "raw Ankh reconstruction target in the original model (shown beside D_G_na)"),
}
NOTE = {"D_G_na": "replacement for D_G, used only if abl_ankh_fullgraph collapsed"}


def lig_values(run, basis="full"):
    """Per-query ligand AUC of the usable queries: 'full' whole database, 'common' the sites common to the
    language-model variants."""
    if basis == "common":
        f = os.path.join(PATHS["out"], "common_sites_per_query.csv")
        if not os.path.exists(f):
            return None
        t = pd.read_csv(f)
        t = t[t.model == run]
        return t[["query_key", "protein", "auc"]].rename(columns={"auc": "v"}).dropna() if len(t) else None
    f = grid_file(run)
    if not os.path.exists(f):
        return None
    t = usable(pd.read_csv(f))
    return t[["query_key", "protein", "lenseek_raw__auc"]].rename(columns={"lenseek_raw__auc": "v"}).dropna()


def mcc_values(run, basis="full"):
    """Out-of-fold top-1 outcomes. 'common' reads <run>__common_sites when the common sites are fewer than the
    database; its Table 1 is refused unless out_<run>__common_sites/RESTRICTED.json carries the current sites."""
    if basis == "common":
        lc = os.path.join(PATHS["out"], "common_sites.json")
        if not os.path.exists(lc):
            return None
        info = json.load(open(lc))
        if info["n_common_sites"] < info["n_db"]:
            rf = os.path.join(eval_dir(run + "__common_sites"), "RESTRICTED.json")
            got = json.load(open(rf)).get("excluded_sites_sha256") if os.path.exists(rf) else None
            if got != info["excluded_sites_sha256"]:
                return f"REFUSED: {run}__common_sites is not on the current common sites (run common-sites " \
                       f"--make-restricted and its Table 1 again)"
            return oof_top1(run + "__common_sites")
    return oof_top1(run)


def oof_top1(run):
    """Per-query out-of-fold top-1 outcome (query_key, protein, cluster, fold, truth, pred) from top1_oof.csv of
    table1_concordance.py table1; its top-1 MCC must equal the LEN-Seek k = 1 row of table1A.csv."""
    f = os.path.join(t1_dir(run), "top1_oof.csv")
    r1 = t1_row(run)
    if not os.path.exists(f) or r1 is None:
        return None
    top1 = pd.read_csv(f, dtype={"query_key": str, "protein": str, "cluster": str})
    ok = np.isfinite(top1.truth)
    m = top1_mcc(top1.truth[ok].astype(bool).to_numpy(), top1.pred[ok].astype(bool).to_numpy())
    if abs(m - r1["MCC"]) > 1e-9:
        die(f"{run}: top-1 MCC of top1_oof.csv {m} != table1A.csv {r1['MCC']}")
    return top1[ok][["query_key", "protein", "cluster", "fold", "truth", "pred"]]


def _get(metric, run, basis):
    return lig_values(run, basis) if metric == "lig_auc" else mcc_values(run, basis)


def _mcc(S, i=""):
    return mcc(*(S[f"{k}{i}"] for k in ["TP", "FP", "FN", "TN"]))


def top1_mcc(t, p):
    """MCC of boolean truth t and prediction p."""
    return mcc((t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum())


def contrast(metric, terms, stratum="all", basis="full", universe=None):
    """terms: [(run, sign)]; universe: runs whose scored queries all count (default the runs of terms).
    Returns dict(value, ci_lo, ci_hi, n_queries, n_entries, n_clusters, missing), dict(refused=...) or None."""
    tabs = []
    for run, _ in terms:
        t = _get(metric, run, basis)
        if t is None:
            return None
        if isinstance(t, str):
            return dict(refused=t)
        tabs.append(t)
    keyset = [set(t.query_key) for t in tabs]
    for r in (universe or []):
        if r not in [x for x, _ in terms]:
            u = _get(metric, r, basis)
            if isinstance(u, str):
                return dict(refused=u)
            if u is not None:
                keyset.append(set(u.query_key))
    both, union = set.intersection(*keyset), set.union(*keyset)
    if stratum == "<30":
        both = {k for k in both if id_stratum(protein_of(k)) == "<30"}
        union = {k for k in union if id_stratum(protein_of(k)) == "<30"}
    missing = sorted(union - both)
    if not both:
        return None
    if metric == "lig_auc":
        d = None
        for (run, sg), t in zip(terms, tabs):
            t = t[t.query_key.isin(both)].set_index("query_key")
            if d is None:
                d = pd.DataFrame({"query_key": t.index, "protein": t.protein.to_numpy(), "diff": 0.0}).set_index("query_key")
            d["diff"] += sg * t.v.reindex(d.index)
        r = entry_weighted(d.reset_index(), ["diff"], n_boot=N_BOOT)["diff"]
        return dict(value=r["entry_mean"], ci_lo=r["entry_ci_lo"], ci_hi=r["entry_ci_hi"], query_mean=r["query_mean"],
                    n_queries=r["n_queries"], n_entries=r["n_entries"], n_clusters=r["n_clusters"], missing=missing)
    d = None
    for i, ((run, sg), t) in enumerate(zip(terms, tabs)):
        t = t[t.query_key.isin(both)].set_index("query_key")
        if d is None:
            d = pd.DataFrame({"query_key": t.index, "protein": t.protein.to_numpy(), "cluster": t.cluster.to_numpy(),
                              "fold": t.fold.to_numpy()}).set_index("query_key")
        elif not (t.fold.reindex(d.index).to_numpy() == d.fold.to_numpy()).all():
            die(f"{run}: cross-validation folds differ from {terms[0][0]} (the rule requires the same folds)")
        tt, pp = t.truth.reindex(d.index).astype(bool).to_numpy(), t.pred.reindex(d.index).astype(bool).to_numpy()
        d[f"TP{i}"], d[f"FP{i}"] = (tt & pp).astype(int), (~tt & pp).astype(int)
        d[f"FN{i}"], d[f"TN{i}"] = (tt & ~pp).astype(int), (~tt & ~pp).astype(int)
    signs = [sg for _, sg in terms]

    def stat(S):
        v = 0.0
        for i, sg in enumerate(signs):
            v = v + sg * np.asarray(_mcc(S, i))
        return {"diff": v}
    d = d.drop(columns="fold")
    tot = d.sum(numeric_only=True)
    val = float(stat({c: float(tot[c]) for c in tot.index})["diff"])
    lo, hi = bootstrap_proteins(d.reset_index(), stat, n=N_BOOT, seed=0)["diff"]
    return dict(value=val, ci_lo=lo, ci_hi=hi, query_mean=np.nan, n_queries=int(len(d)),
                n_entries=int(d.protein.nunique()), n_clusters=int(d.cluster.nunique()), missing=missing)


def seed_sd(metric, stratum="all", basis="full"):
    """Sample s.d. (n - 1) of M over the evaluated seeds of the original model, each on the queries all seeds scored."""
    tabs, refused = {}, []
    for s in SEEDS:
        t = _get(metric, s, basis)
        if isinstance(t, str):
            refused.append(t)
        elif t is not None:
            tabs[s] = t
    if refused or len(tabs) < 2:
        return dict(n_seeds=len(tabs), seeds=",".join(tabs), sd=np.nan, two_sd=np.nan, values="",
                    refused="; ".join(refused))
    both = set.intersection(*[set(t.query_key) for t in tabs.values()])
    if stratum == "<30":
        both = {k for k in both if id_stratum(protein_of(k)) == "<30"}
    vals = []
    for t in tabs.values():
        t = t[t.query_key.isin(both)]
        if metric == "lig_auc":
            vals.append(entry_weighted(t.rename(columns={"v": "x"}), ["x"], n_boot=1)["x"]["entry_mean"])
        else:
            vals.append(top1_mcc(t.truth.astype(bool).to_numpy(), t.pred.astype(bool).to_numpy()))
    sd = float(np.std(vals, ddof=1))
    return dict(n_seeds=len(tabs), seeds=",".join(tabs), sd=sd, two_sd=2 * sd, values=";".join(f"{v:.4f}" for v in vals),
                refused="")


def build_contrasts(runs):
    """Every contrast x metric x basis x stratum -> contrasts.csv and the lists of queries left out."""
    out = PATHS["out"]
    miss_dir = os.path.join(out, "contrast_missing")
    os.makedirs(miss_dir, exist_ok=True)
    st = {r: deg_state(r)[0] for r in set(runs) | set(SEEDS)}
    common_ok = [r for r in COMMON_SITE_RUNS if r in runs and st[r] == "ok"]
    named = {r for terms, _, _ in NAMED.values() for r, _ in terms}
    specs = [(n, terms, g, role) for n, (terms, g, role) in NAMED.items() if all(r in runs for r, _ in terms)]
    specs += [(f"D_{r}", [(REF, 1), (r, -1)], "auxiliary", "report only") for r in runs
              if r not in named and r not in SEEDS]
    sd_cache, rows = {}, []
    for name, terms, group, role in specs:
        models = " ".join(f'{"+" if s > 0 else "-"}{r}' for r, s in terms)
        universe = common_ok if group == "language_model" else None
        for metric in ["lig_auc", "mcc_top1"]:
            for basis in (["common", "full"] if group == "language_model" else ["full"]):
                for stratum in ["all", "<30"]:
                    if (metric, stratum, basis) not in sd_cache:
                        sd_cache[metric, stratum, basis] = seed_sd(metric, stratum, basis)
                    sd = sd_cache[metric, stratum, basis]
                    c = contrast(metric, terms, stratum, basis, universe)
                    row = dict(contrast=name, group=group, role=role, models=models, metric=metric,
                               basis=BASIS[basis], stratum=stratum)
                    notes = []
                    if c is None:
                        row["value"] = np.nan
                        notes.append("inputs missing")
                    elif "refused" in c:
                        row["value"] = np.nan
                        notes.append(c["refused"])
                    else:
                        mf = os.path.join(miss_dir, f'{name}__{metric}__{basis}__{stratum.replace("<", "lt")}.txt')
                        with open(mf, "w") as fh:
                            fh.write("\n".join(c["missing"]) + ("\n" if c["missing"] else ""))
                        row.update({k: c[k] for k in ["value", "ci_lo", "ci_hi", "query_mean", "n_queries",
                                                      "n_entries", "n_clusters"]}, n_missing=len(c["missing"]))
                    bad = [r for r, _ in terms if st[r] != "ok"]
                    if bad:
                        notes.append(f"{', '.join(f'{r} {st[r]}' for r in bad)}: not interpreted")
                    if name in NOTE:
                        notes.append(NOTE[name])
                    seed_note = sd.get("refused") or ""
                    bad_s = [s for s in (sd["seeds"].split(",") if sd["seeds"] else []) if st[s] != "ok"]
                    if bad_s:
                        seed_note = "; ".join(x for x in (seed_note, f"seed {', '.join(bad_s)} not ok") if x)
                    row.update(note="; ".join(notes), seed_n=sd["n_seeds"], seeds=sd["seeds"],
                               seed_values=sd["values"], seed_sd=sd["sd"], seed_2sd=sd["two_sd"], seed_note=seed_note)
                    rows.append(row)
    cols = ["contrast", "group", "role", "models", "metric", "basis", "stratum", "value", "ci_lo", "ci_hi",
            "query_mean", "n_queries", "n_entries", "n_clusters", "n_missing", "note", "seed_n", "seeds",
            "seed_values", "seed_sd", "seed_2sd", "seed_note"]
    t = pd.DataFrame(rows, columns=cols)
    t.to_csv(os.path.join(out, "contrasts.csv"), index=False, float_format="%.6f")
    return t


# ------------------------------------------------------------------ identity strata of the ligand metrics
FRAME_METRICS = [("raw_auc", "lenseek_raw__auc"), ("re1", "lenseek_raw__re1"),
                 ("raw_auc_db_entries", "lenseek_raw__auc_db_entries")]
FRAME_STRATA = [("all", None), ("<30", {"<30"}), (">=30", {"30-40", "40-50", "50-80", "80-95", ">=95"})]


def frame_values(run, col):
    f = grid_file(run)
    if not os.path.exists(f):
        return None
    t = pd.read_csv(f)
    if col not in t:
        return None
    return usable(t)[["query_key", "protein", col]].rename(columns={col: "v"}).dropna()


def in_stratum(keys, allowed):
    return keys if allowed is None else {k for k in keys if id_stratum(protein_of(k)) in allowed}


def build_frame(runs):
    """M(main_s0) - M(run) per identity stratum on the queries both have (entry-weighted, paired cluster bootstrap),
    with 2 x the seed s.d. of the metric in that stratum -> frame.csv."""
    rows, sd_cache = [], {}
    for run in runs:
        state = deg_state(run)[0]
        flags = dict(degenerate=None if state == "unchecked" else state == "degenerate",
                     label="degeneracy not checked" if state == "unchecked" else (COLLAPSE if state == "degenerate" else ""))
        for mname, col in FRAME_METRICS:
            t, ref = frame_values(run, col), frame_values(REF, col)
            if run == REF or t is None or ref is None:
                continue
            for sname, allowed in FRAME_STRATA:
                both = in_stratum(set(ref.query_key) & set(t.query_key), allowed)
                if not both:
                    continue
                d = ref[ref.query_key.isin(both)].set_index("query_key")[["protein", "v"]]
                d["diff"] = d.v - t.set_index("query_key").v.reindex(d.index)
                r = entry_weighted(d.reset_index(), ["diff"], n_boot=N_BOOT)["diff"]
                if (col, sname) not in sd_cache:
                    tabs = {s: x for s in SEEDS if (x := frame_values(s, col)) is not None}
                    sd_cache[col, sname] = dict(seed_n=len(tabs), seeds=",".join(tabs), seed_values="", seed_sd=np.nan,
                                                seed_2sd=np.nan, seed_note="")
                    if len(tabs) >= 2:
                        sb = in_stratum(set.intersection(*[set(x.query_key) for x in tabs.values()]), allowed)
                        vals = [entry_weighted(x[x.query_key.isin(sb)].rename(columns={"v": "x"}), ["x"],
                                               n_boot=1)["x"]["entry_mean"] for x in tabs.values()]
                        sd = float(np.std(vals, ddof=1))
                        sd_cache[col, sname].update(seed_values=";".join(f"{v:.4f}" for v in vals), seed_sd=sd,
                                                    seed_2sd=2 * sd)
                rows.append(dict(run=run, metric=mname, column=col, stratum=sname, quantity="delta_vs_main_s0",
                                 reference=REF, value=r["entry_mean"], ci_lo=r["entry_ci_lo"], ci_hi=r["entry_ci_hi"],
                                 query_mean=r["query_mean"], n_queries=r["n_queries"], n_entries=r["n_entries"],
                                 n_clusters=r["n_clusters"], **sd_cache[col, sname], **flags))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(PATHS["out"], "frame.csv"), index=False, float_format="%.6f")
    return t


def db_entries_lines(frame):
    """The ligand ROC-AUC among the database sites of the grids' --db-entries list: each run, the original model, and
    the original model minus the run (frame.csv)."""
    if not len(frame):
        return
    for _, x in frame[(frame.metric == "raw_auc_db_entries") & (frame.stratum == "all")].iterrows():
        lv = []
        for run in (x.run, REF):
            s = pd.read_csv(grid_file(run, "ligand_grid_summary.csv"))
            lv.append(s[(s.stratum == "all") & (s.column == x.column)].entry_mean.iloc[0])
        print(f"ligand ROC-AUC among the database sites of the listed entries: {x.run} {lv[0]:.3f}, {REF} {lv[1]:.3f}; "
              f"{REF} - {x.run} {x.value:+.3f} [{x.ci_lo:.3f}, {x.ci_hi:.3f}] ({x.n_queries} queries)")


# ------------------------------------------------------------------ Supplementary Table S14
def levels(run):
    r = {}
    f = grid_file(run, "ligand_grid_summary.csv")
    if os.path.exists(f):
        s = pd.read_csv(f)
        s = s[s.stratum == "all"].set_index("column")
        g = lambda c, k="entry_mean": s.loc[c, k] if c in s.index else np.nan  # noqa: E731
        r.update(lig_auc=g("lenseek_raw__auc"), lig_auc_ci_lo=g("lenseek_raw__auc", "entry_ci_lo"),
                 lig_auc_ci_hi=g("lenseek_raw__auc", "entry_ci_hi"), lig_n_queries=g("lenseek_raw__auc", "n_queries"),
                 re1=g("lenseek_raw__re1"), re1_n_queries=g("lenseek_raw__re1", "n_queries"))
    r1 = t1_row(run)
    if r1 is not None:
        r.update(mcc_top1=r1["MCC"], mcc_top1_ci_lo=r1["MCC_ci_lo"], mcc_top1_ci_hi=r1["MCC_ci_hi"])
    return r


def pick(con, name, metric):
    """The contrast row of the table: the common sites when finite, else the full database (stratum all)."""
    s = con[(con.contrast == name) & (con.metric == metric) & (con.stratum == "all")]
    for b in (BASIS["common"], BASIS["full"]):
        x = s[s.basis == b]
        if len(x) and np.isfinite(x.value.iloc[0]):
            return x.iloc[0]
    return None


def delta_cols(prefix, r, ok):
    if r is None:
        return {}
    det = bool(abs(r.value) > r.seed_2sd and (r.ci_lo > 0 or r.ci_hi < 0)) if ok and np.isfinite(r.seed_2sd) else None
    return {f"{prefix}": r.value, f"{prefix}_ci_lo": r.ci_lo, f"{prefix}_ci_hi": r.ci_hi, f"{prefix}_basis": r.basis,
            f"{prefix}_2sd_seed": r.seed_2sd, f"{prefix}_detected": det, f"{prefix}_n_queries": r.n_queries}


def cmd_table(a):
    os.makedirs(PATHS["out"], exist_ok=True)
    reg = {r[0]: r for r in RUNS}
    runs = a.runs.split(",") if a.runs else [r for r in reg if os.path.exists(grid_file(r))]
    if REF not in runs:
        die(f"no ligand metrics of {REF} in {PATHS['fig']}")
    con, frame = build_contrasts(runs), build_frame(runs)
    run_contrast = {terms[1][0]: n for n, (terms, _, _) in NAMED.items() if terms[0][0] == REF}
    rows = []
    for run in runs:
        state, dg = deg_state(run)
        _, _, sites, group, label = reg.get(run, (run, "", pd.NA, "", run))
        row = dict(run=run, label=label, group=group, train_sites=sites, au=dg["au"] if dg else pd.NA,
                   total_var=dg["total_var"] if dg else np.nan, latent_state=state,
                   note=COLLAPSE if state == "degenerate" else ("degeneracy not checked" if state == "unchecked" else ""))
        row.update(levels(run))
        if run == REF:
            row.update(delta_lig_auc_basis="reference", delta_mcc_top1_basis="reference")
        elif run in SEEDS:
            f = frame[(frame.run == run) & (frame.metric == "raw_auc") & (frame.stratum == "all")]
            if len(f):
                x = f.iloc[0].copy()
                x["basis"] = "full DB (seed)"
                row.update(delta_cols("delta_lig_auc", x, False))
            row.update(delta_mcc_top1_basis="seed")
        else:
            name = run_contrast.get(run, f"D_{run}")
            row.update(delta_cols("delta_lig_auc", pick(con, name, "lig_auc"), state == "ok"))
            row.update(delta_cols("delta_mcc_top1", pick(con, name, "mcc_top1"), state == "ok"))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(PATHS["out"], "variants.csv"), index=False, float_format="%.6f")
    print(f"variants.csv: {len(t)} runs; contrasts.csv {len(con)} rows; frame.csv {len(frame)} rows")
    db_entries_lines(frame)
    for (m, b), g in con[con.stratum == "all"].groupby(["metric", "basis"]):
        v = g.seed_2sd.dropna()
        if len(v):
            print(f"2 x seed s.d., {m}, {b}: {v.iloc[0]:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="active latent units (latent collapse)")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("common-sites", help="the sites common to the language-model variants")
    p.add_argument("--make-restricted", action="store_true")
    p = sub.add_parser("table", help="contrasts, identity strata and the rows of Supplementary Table S14")
    p.add_argument("--runs", default=None, help="comma list (default: every run with ligand metrics)")
    a = ap.parse_args()
    {"check": cmd_check, "common-sites": cmd_common_sites, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
