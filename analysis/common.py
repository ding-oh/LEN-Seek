"""Paths, input readers, statistics and plotting helpers shared by the analysis scripts.

The scripts in analysis/ insert their own directory on sys.path and import what they need from here
(`from common import ...`); die, require_path, site_key, read_tsv, tie_key and strict_order are those of
scripts/pipeline.py, so that the result scripts join the producers' files by the same site key and break ties the same
way. Inputs: see `python analysis/run.py -h`; read here: LENSEEK_OUT (outputs, default ./out; the site annotations
in $LENSEEK_OUT/cache/; the identity, structure and ligand strata of the evaluation queries in
$LENSEEK_OUT/site_annotation/), LENSEEK_REFERENCE_DIR (asset reference_data; default: where lenseek-fetch unpacks it)
and LENSEEK_BASELINE_DIR (default $LENSEEK_OUT/baselines); the query sets and sequence clusters are read from
data/query_sets/.
"""
import collections
import gzip
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))   # for this import only
import pipeline  # noqa: E402
sys.path.remove(os.path.join(REPO, "scripts"))
die, require_path, site_key, read_tsv = pipeline.die, pipeline.require_path, pipeline.site_key, pipeline.read_tsv
tie_key, strict_order = pipeline.tie_key, pipeline.strict_order
OUT_ROOT = os.path.abspath(os.environ.get("LENSEEK_OUT") or "out")
CACHE = os.path.join(OUT_ROOT, "cache")
ANNOT = os.path.join(CACHE, "site_annot")
QUERY_SETS = os.path.join(REPO, "data", "query_sets")
PRIMARY_EXCLUDE_FILE = os.path.join(QUERY_SETS, "primary_exclude_pdb.txt")
CLUSTER_FILE = os.path.join(QUERY_SETS, "seq_clusters_50.tsv")
PROTEIN_STRATA = os.path.join(OUT_ROOT, "site_annotation", "protein_strata.tsv")   # site_annotation.py identity-strata
QUERY_STRATA = os.path.join(OUT_ROOT, "site_annotation", "query_strata.csv")       # site_annotation.py strata
# The reference tables of the asset reference_data, where `lenseek-fetch reference_data` unpacks them.
REFERENCE_DIR = os.environ.get("LENSEEK_REFERENCE_DIR") or os.path.join(
    os.environ.get("LENSEEK_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "lenseek"),
    "assets", "reference_data", "reference_data")
SNAP = os.path.join(REFERENCE_DIR, "ccd_smiles.tsv")
SNAP_META = os.path.join(REFERENCE_DIR, "ccd_snapshot.json")
PFAM_FILES = ("Pfam-A.clans.tsv.gz", "pdb_chain_pfam.tsv.gz")
PFAM_DIRS = (os.path.join(REFERENCE_DIR, "pfam"), os.path.join(CACHE, "pfam"))   # the copy the results used, a download
STRAT_DIR = os.environ.get("LENSEEK_STRAT_DIR")
BASELINE_DIR = os.environ.get("LENSEEK_BASELINE_DIR") or os.path.join(OUT_ROOT, "baselines")   # input top100/
BASELINE_OUT = os.path.join(OUT_ROOT, "baselines")                                       # outputs compare/, ligand/
M8 = os.path.join(STRAT_DIR, "eval_vs_train_uncapped.m8") if STRAT_DIR else None
PROBIS_NOALIGN = {"nosig_empty", "nosig_filtered"}
PAIR_REQUIRED = ["query_path", "target_path", "probis_z", "probis_align", "probis_status"]
UNRELATED_IDENT = 0.30
KS = [1, 5, 10, 20, 50, 100]
LENSEEK = "LEN-Seek"
_CLUSTERS = {}


# ---------------------------------------------------------------- sites, entries and clusters
def protein_of(key):
    """PDB id (lower case) of a site key."""
    return key.split("/")[0].lower()


def primary_excluded(path=PRIMARY_EXCLUDE_FILE):
    """PDB ids (lower case) left out of the primary query set (data/query_sets/primary_exclude_pdb.txt)."""
    return {l.split()[0].lower() for l in open(path) if l.strip() and not l.startswith("#")}


def identity_strata(path=PROTEIN_STRATA):
    """PDB id (lower case) -> sequence-identity stratum of each evaluation protein to the training proteins ('<30',
    '30-40', ..., '>=95'; best hit with query or target coverage >= 0.5), from scripts/site_annotation.py
    identity-strata."""
    if not os.path.exists(path):
        die(f"{path} missing: run scripts/site_annotation.py identity-strata")
    s = pd.read_csv(path, sep="\t", dtype=str)
    return dict(zip(s.pdb.str.lower(), s.stratum_either50))


def structure_ligand_strata(path=QUERY_STRATA):
    """Structure and ligand stratum of each evaluation query (q_idx, structure_stratum, ligand_stratum), from
    scripts/site_annotation.py strata."""
    if not os.path.exists(path):
        die(f"{path} missing: run scripts/site_annotation.py strata")
    return pd.read_csv(path, usecols=["q_idx", "structure_stratum", "ligand_stratum"])


def cluster_of(key, pct=50):
    """Sequence cluster of a query site (chains linked at >= pct % identity over >= 50 % of the shorter chain; a site
    belongs to the cluster of its main chain), by query key '<pdb>/ligand_pocket_...pdb' or by a PDB id whose sites all
    share one cluster. Read from data/query_sets/seq_clusters_<pct>.tsv."""
    if pct not in _CLUSTERS:
        f = os.path.join(QUERY_SETS, f"seq_clusters_{pct}.tsv")
        if not os.path.exists(f):
            die(f"{f} missing: run scripts/site_annotation.py clusters first")
        c = pd.read_csv(f, sep="\t", comment="#", dtype=str)
        by_pdb = c.groupby(c.pdb.str.lower()).cluster.agg(lambda v: v.iloc[0] if v.nunique() == 1 else None)
        _CLUSTERS[pct] = (dict(zip(c.query_key, c.cluster)), by_pdb.to_dict())
    by_key, by_pdb = _CLUSTERS[pct]
    if "/" in key:
        if key not in by_key:
            die(f"query {key} has no sequence cluster in seq_clusters_{pct}.tsv")
        return by_key[key]
    v = by_pdb.get(key.lower(), "missing")
    if v is None or v == "missing":
        die(f"entry {key}: " + ("sites in different clusters, group by query key" if v is None else
                                f"no sequence cluster in seq_clusters_{pct}.tsv"))
    return v


def chain_name(key, chain):
    """'<pdb>_<chain>' of a site's chain, as in the MMseqs2 and Pfam tables."""
    return f"{key.split('/')[0]}_{chain}"


# ---------------------------------------------------------------- evaluation outputs and ranked lists
class EvalOutputs:
    """Evaluation outputs of one model (q_paths.txt, db_paths.txt, dist.npy; q_mask.npy / db_mask.npy for the site
    sizes); sel = the primary queries (query_set='primary') or all queries ('full')."""

    def __init__(self, eval_dir, query_set="primary"):
        require_path(eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
        self.dir = os.path.abspath(eval_dir)
        miss = [f for f in ("q_paths.txt", "db_paths.txt", "dist.npy") if not os.path.exists(os.path.join(eval_dir, f))]
        if miss:
            die(f"{eval_dir} is missing {miss} (evaluation outputs incomplete?)")
        self.q_paths = open(os.path.join(eval_dir, "q_paths.txt")).read().split()
        self.db_paths = open(os.path.join(eval_dir, "db_paths.txt")).read().split()
        self.q_keys = [site_key(p) for p in self.q_paths]
        self.db_keys = [site_key(p) for p in self.db_paths]
        self.dist = np.load(os.path.join(eval_dir, "dist.npy"), mmap_mode="r")
        assert self.dist.shape == (len(self.q_paths), len(self.db_paths)), self.dist.shape
        self.query_set = query_set
        if query_set == "full":
            excl = set()
        elif query_set == "primary":
            excl = primary_excluded()
        else:
            die(f"unknown query set {query_set}")
        self.sel = np.array([i for i, k in enumerate(self.q_keys) if protein_of(k) not in excl], dtype=int)
        self.excluded_proteins = sorted(excl)

    @property
    def tag(self):
        """Model name: the eval dir name without 'out_'."""
        b = os.path.basename(os.path.normpath(self.dir))
        return b[4:] if b.startswith("out_") else b

    def sizes(self, which="q"):
        """Residues per query ('q') or database ('db') site."""
        return np.asarray(np.load(os.path.join(self.dir, f"{which}_mask.npy"), mmap_mode="r").sum(1))


def annotate_rows(df, ev):
    """Add query_key, protein, cluster and target_key to a (q_idx, db_idx) frame."""
    df["query_key"] = [ev.q_keys[i] for i in df.q_idx]
    df["protein"] = df.query_key.map(protein_of)
    df["cluster"] = df.query_key.map(cluster_of)
    df["target_key"] = [ev.db_keys[i] for i in df.db_idx]
    return df


def topk_counts(rows, ks):
    """Each query has exactly k rows with rank <= k."""
    nq = rows.q_idx.nunique()
    for k in ks:
        c = rows[rows["rank"] <= k].groupby("q_idx").size()
        if len(c) != nq or (c != k).any():
            die(f"top-{k}: {int((c != k).sum())} queries do not have exactly {k} rows")


def topk_rows(ev, kmax=100):
    """Top-kmax database sites of every selected query, sorted by (Chamfer distance, database index): exactly kmax rows
    per query, with query_key, protein, cluster and target_key."""
    out = []
    for qi in ev.sel:
        d = np.asarray(ev.dist[qi])
        cand = np.nonzero(d <= np.partition(d, kmax)[kmax])[0]
        o = np.lexsort((cand, d[cand]))[:kmax]
        out.append(pd.DataFrame({"q_idx": qi, "db_idx": cand[o], "cd": d[cand[o]].astype(np.float64),
                                 "rank": np.arange(1, len(o) + 1)}))
    df = pd.concat(out, ignore_index=True)
    topk_counts(df, [kmax])
    return annotate_rows(df, ev)


def method_list(ev, path, k=100):
    """Own top-k list of a method: a distance matrix .npy (sites without a descriptor NaN; ties by database index) or a
    top-k TSV (query_idx, rank, db_idx and dist, or score with d = 1 - score)."""
    if path.endswith(".npy"):
        D = np.load(path, mmap_mode="r")
        rec = []
        for qi in ev.sel:
            d = np.asarray(D[qi], dtype=np.float64)
            idx = np.nonzero(np.isfinite(d))[0]
            o = idx[np.lexsort((idx, d[idx]))][:k]
            rec.append(pd.DataFrame({"q_idx": qi, "db_idx": o, "cd": d[o], "rank": np.arange(1, len(o) + 1)}))
        m = pd.concat(rec, ignore_index=True)
    else:
        t = pd.read_csv(path, sep="\t")
        t = t[t.query_idx.isin(ev.sel) & (t["rank"] <= k)]
        d = t.dist.astype(float) if "dist" in t else 1.0 - t.score.astype(float)
        m = pd.DataFrame({"q_idx": t.query_idx.to_numpy(), "db_idx": t.db_idx.to_numpy(), "cd": d.to_numpy(),
                          "rank": t["rank"].to_numpy()})
        order = {q: i for i, q in enumerate(ev.sel)}
        m = m.assign(_o=m.q_idx.map(order)).sort_values(["_o", "rank"]).drop(columns="_o").reset_index(drop=True)
    if not ((m.groupby("q_idx").size() == k).all() and m.q_idx.nunique() == len(ev.sel)):
        die(f"{path}: not exactly {k} rows for each of the {len(ev.sel)} queries")
    return annotate_rows(m, ev)


# ---------------------------------------------------------------- ProBiS and G-LoSA labels
def classify_probis(z, status):
    """'ok' (Z present) / 'no_alignment' (nosig_*: ProBiS ran, nothing significant) / 'failure'."""
    st = status.fillna("").astype(str)
    zf = np.isfinite(z.to_numpy(dtype=float))
    cls = np.where(st.eq("ok") & zf, "ok", np.where(st.isin(PROBIS_NOALIGN), "no_alignment", "failure"))
    return pd.Series(cls, index=z.index)


def load_pairs(tsv, ga_col="ga_all"):
    """The scored pair table; stops on missing columns, blank or pending statuses and conflicting duplicates."""
    require_path(tsv, "--pairs", "LENSEEK_PAIRS")
    p = pd.read_csv(tsv, sep="\t", low_memory=False, dtype={"probis_status": str, ga_col + "_status": str})
    miss = [c for c in PAIR_REQUIRED + [ga_col, ga_col + "_status"] if c not in p.columns]
    if miss:
        die(f"{tsv} lacks columns {miss} (expected the schema of the ProBiS / G-LoSA scorer output)")
    st_cols = [c for c in p.columns if c.endswith("_status")]
    if all(p[c].fillna("").astype(str).str.strip().eq("").all() for c in st_cols):
        die(f'{tsv}: every status column ({", ".join(st_cols)}) is blank')
    for c in ["probis_status", ga_col + "_status"]:
        pend = int(p[c].fillna("").eq("pending").sum())
        if pend:
            die(f"{tsv}: {pend} rows still have {c} = pending (scoring not finished)")
    p["query_key"] = p.query_path.map(site_key)
    p["target_key"] = p.target_path.map(site_key)
    for c in ["probis_z", "probis_align", ga_col]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["ga"] = p[ga_col]
    p["ga_status"] = p[ga_col + "_status"].fillna("")
    p["probis_class"] = classify_probis(p.probis_z, p.probis_status)
    p["glosa_class"] = np.where(np.isfinite(p.ga.to_numpy(dtype=float)) & p.ga_status.eq("ok"), "ok", "failure")
    dup = p.duplicated(["query_key", "target_key"], keep=False)
    if dup.any():
        if p[dup].groupby(["query_key", "target_key"])[["probis_z", "ga"]].nunique(dropna=False).max().max() > 1:
            die(f"{tsv}: {int(dup.sum())} rows are duplicated (query, target) pairs with different scores")
        p = p.drop_duplicates(["query_key", "target_key"])
    return p


def attach_pairs(rows, pairs):
    """Join pair scores onto top-k rows; every top-k row must be in the TSV."""
    cols = ["query_key", "target_key", "probis_z", "probis_align", "probis_status", "probis_class", "ga",
            "ga_status", "glosa_class"]
    m = rows.merge(pairs[cols], on=["query_key", "target_key"], how="left")
    if m.probis_class.isna().any():
        die(f"{int(m.probis_class.isna().sum())} of {len(m)} top-k rows have no row in the pair TSV")
    return m


def probis_truth(m, thr=2.0):
    """1 at Z >= thr, 0 below or without an alignment, NaN for failed comparisons."""
    y = (m.probis_z.to_numpy(dtype=float) >= thr).astype(float)
    cls = m.probis_class.to_numpy()
    y[cls == "no_alignment"] = 0.0
    y[cls == "failure"] = np.nan
    return y


class ScanLabels:
    """Database-wide ProBiS (and G-LoSA) labels of every query-site pair, rows matched to the eval dir by site key."""

    def __init__(self, ev, probis_dir, glosa_dir, pairs_dir):
        require_path(probis_dir, "--scan-dir", "LENSEEK_SCAN_DIR")
        require_path(pairs_dir, "--pairs-dir", "LENSEEK_PAIRS_DIR")
        sq = [site_key(p) for p in open(os.path.join(pairs_dir, "all_query_sites.txt")).read().split()]
        sd = [site_key(p) for p in open(os.path.join(pairs_dir, "all_db_sites.txt")).read().split()]
        if sd != ev.db_keys:
            die("scan database order differs from db_paths.txt of the eval dir")
        self.qpos = {k: i for i, k in enumerate(sq)}
        self.ev = ev
        self.pst = np.load(os.path.join(probis_dir, "all_status.npy"), mmap_mode="r")
        self.raw = np.load(os.path.join(probis_dir, "all_raw.npy"), mmap_mode="r")
        self.gst = self.ga = None
        if glosa_dir:
            self.gst = np.load(os.path.join(glosa_dir, "all_status.npy"), mmap_mode="r")
            self.ga = np.load(os.path.join(glosa_dir, "all_ga.npy"), mmap_mode="r")

    def row(self, qi):
        i = self.qpos[self.ev.q_keys[qi]]
        st = np.asarray(self.pst[i])
        if (st == 0).any():
            die(f"query {self.ev.q_keys[qi]}: pending ProBiS pairs in the scan")
        z = np.round((np.asarray(self.raw[i], dtype=np.float64) - 1.958) / 2.21468, 2)
        if self.gst is None:
            return st, z, None, None
        return st, z, np.asarray(self.gst[i]), np.asarray(self.ga[i], dtype=np.float64)

    def label(self, m):
        """Add pst, z, gst, ga, truth (ProBiS) and truth_glosa columns to a (q_idx, db_idx) frame."""
        cols = {c: np.full(len(m), np.nan) for c in ("pst", "z", "gst", "ga")}
        for qi, idx in m.groupby("q_idx").indices.items():
            st, z, gst, ga = self.row(qi)
            t = m.db_idx.to_numpy()[idx]
            cols["pst"][idx], cols["z"][idx] = st[t], z[t]
            if gst is not None:
                cols["gst"][idx], cols["ga"][idx] = gst[t], ga[t]
        for c, v in cols.items():
            m[c] = v
        m["truth"] = np.where(m.pst == 1, (m.z >= 2.0).astype(float), np.where(m.pst == 2, 0.0, np.nan))
        m["truth_glosa"] = np.where(m.gst == 1, (m.ga >= 0.59).astype(float), np.nan)
        return m


# ---------------------------------------------------------------- out-of-fold thresholds and top-k concordance
def scores(c):
    """Precision, recall, F1 and MCC (0 when undefined); works on arrays too."""
    TP, FP, FN, TN = (np.asarray(c[k], dtype=float) for k in ("TP", "FP", "FN", "TN"))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rec = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = np.where(den > 0, (TP * TN - FP * FN) / np.where(den > 0, den, 1), 0.0)
    out = dict(Precision=prec, Recall=rec, F1=f1, MCC=mcc)
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}


def mcc(tp, fp, fn, tn):
    """MCC of counts (arrays too), 0 when undefined."""
    return scores(dict(TP=tp, FP=fp, FN=fn, TN=tn))["MCC"]


def per_query_counts(m, y, pred):
    """Per-query TP/FP/FN/TN, rows and scored rows (y NaN = not scored)."""
    ok = np.isfinite(y)
    t = np.where(ok, y, 0).astype(bool)
    d = pd.DataFrame({"q_idx": m.q_idx.to_numpy(), "protein": m.protein.to_numpy(), "cluster": m.cluster.to_numpy(),
                      "TP": ok & t & pred, "FP": ok & ~t & pred, "FN": ok & t & ~pred, "TN": ok & ~t & ~pred,
                      "rows": 1, "scored": ok})
    return d.groupby(["q_idx", "protein", "cluster"], sort=False).sum().reset_index()


def make_grid(d1, step):
    """Threshold grid of a method on the multiples of step, each the double nearest to k * step (k / (1 / step)), from
    the top-1 values d1: 0, 0.5, ... up to ceil(max) for LEN-Seek (step 0.5); 0, step, ... up to the first multiple of
    step >= max for a distance; for a negated score (d < 0: Folddisco, d = -score of 4 decimals with a tie-break term
    below half a step, sites not returned at 1000) the distinct lattice values of d1 and one step below the smallest
    (nothing called similar). In the matching and entries runs of scripts/folddisco.py the tie-break term is added (d
    just above its lattice value), so a threshold equal to a site's own score does not call it similar; this touches
    only the top-1 MCC of these two runs, which is not reported."""
    d1 = np.asarray(d1, dtype=np.float64)
    if step == 0.5:
        return np.arange(0, np.ceil(np.max(d1)) + step, step)
    n = round(1 / step)
    if np.min(d1) < 0:
        k = np.unique(np.round(d1 * n))
        return np.r_[k[0] - 1, k] / n
    return np.arange(0, int(np.ceil(np.max(d1) / step)) + 1) / n


def mcc_curve(cd, y, grid):
    """MCC of 'similar = cd <= t' against labels y at every threshold t of grid."""
    t = y.astype(bool)
    p = cd[None, :] <= grid[:, None]
    return mcc((p & t).sum(1), (p & ~t).sum(1), (~p & t).sum(1), (~p & ~t).sum(1))


def run_cv(top1, m, grid, splitter, groups):
    """Fold thresholds (training folds only; first maximum of the top-1 MCC on grid over the labelled rows of top1:
    q_idx, cd, truth) and the out-of-fold prediction of every row of m."""
    thr_of, folds = {}, []
    queries = top1.q_idx.to_numpy()
    for f, (tr, te) in enumerate(splitter.split(queries, groups=groups)):
        t = top1.iloc[tr]
        ok = np.isfinite(t.truth.to_numpy())
        i = int(np.argmax(mcc_curve(t.cd.to_numpy()[ok], t.truth.to_numpy()[ok], grid)))
        folds.append(dict(fold=f, threshold=float(grid[i]), n_test_queries=len(te)))
        for q in queries[te]:
            thr_of[q] = (f, float(grid[i]))
    pred = m.cd.to_numpy() <= m.q_idx.map(lambda q: thr_of[q][1]).to_numpy()
    return folds, thr_of, pred


def concordance(m, y, ks=KS, n_boot=1000):
    """Pooled top-k counts and scores at the fold thresholds of m.pred, cluster-bootstrap CIs, entry-weighted MCC."""
    out = []
    for k in ks:
        sel = (m["rank"] <= k).to_numpy()
        mk = m[sel]
        pq = per_query_counts(mk, y[sel], mk.pred.to_numpy())
        tot = pq.sum(numeric_only=True)
        S = {c: float(tot[c]) for c in ["TP", "FP", "FN", "TN", "rows", "scored"]}
        sc = scores(S)
        ci = bootstrap_proteins(pq.drop(columns=["q_idx"]), scores, n=n_boot)
        w = 1.0 / pq.groupby("protein").q_idx.transform("count")
        ew = scores({c: float((pq[c] * w).sum()) for c in ["TP", "FP", "FN", "TN"]})
        den = (S["TP"] + S["FP"]) * (S["TP"] + S["FN"]) * (S["TN"] + S["FP"]) * (S["TN"] + S["FN"])
        r = dict(k=k, TP=int(S["TP"]), FP=int(S["FP"]), FN=int(S["FN"]), TN=int(S["TN"]), **sc,
                 MCC_defined=bool(den > 0), MCC_entry_weighted=ew["MCC"], n_queries=int(pq.q_idx.nunique()),
                 n_entries=int(pq.protein.nunique()), n_clusters=int(pq.cluster.nunique()), n_rows=int(S["rows"]),
                 n_scored=int(S["scored"]))
        for name, (lo, hi) in ci.items():
            r[f"{name}_ci_lo"], r[f"{name}_ci_hi"] = lo, hi
        if not r["MCC_defined"]:
            r["MCC"] = r["MCC_ci_lo"] = r["MCC_ci_hi"] = np.nan
        out.append(r)
    return pd.DataFrame(out)


def oof_tables(ev, m, step, glosa=True, n_boot=1000):
    """Out-of-fold ProBiS (and G-LoSA) tables of one method's labelled top-k frame; also returns its thresholds."""
    top1 = m[m["rank"] == 1].drop_duplicates("q_idx")
    groups = top1.cluster.to_numpy()
    grid = make_grid(top1.cd.to_numpy(), step)
    folds, thr_of, pred = run_cv(top1, m, grid, GroupKFold(n_splits=5), groups)
    m["fold"] = m.q_idx.map(lambda q: thr_of[q][0])
    m["fold_threshold"] = m.q_idx.map(lambda q: thr_of[q][1])
    m["pred"] = pred
    ok = np.isfinite(top1.truth.to_numpy())
    curve = mcc_curve(top1.cd.to_numpy()[ok], top1.truth.to_numpy()[ok], grid)
    i = int(np.argmax(curve))
    thr = dict(grid_step=step, fold_thresholds=";".join(f'{f["threshold"]:g}' for f in folds),
               fold_sizes=";".join(str(f["n_test_queries"]) for f in folds),
               median_fold_threshold=float(np.median([f["threshold"] for f in folds])),
               fixed_threshold=float(grid[i]), fixed_threshold_top1_mcc=float(curve[i]), n_queries=int(len(top1)))
    t_p = concordance(m, m.truth.to_numpy(), n_boot=n_boot)
    t_g = concordance(m, m.truth_glosa.to_numpy(), n_boot=n_boot) if glosa else None
    return t_p, t_g, thr, (top1, groups, grid)


def labelled_lists(ev, a, need_glosa=True):
    """{label: (labelled own top-100 frame, grid step)}, LEN-Seek first."""
    lab = ScanLabels(ev, a.scan_dir, a.glosa_scan_dir if need_glosa else None, a.pairs_dir)
    out = {LENSEEK: (lab.label(topk_rows(ev, 100).copy()), 0.5)}
    for name, step, path in a.method:
        out[name] = (lab.label(method_list(ev, require_path(path, "--method", name))), float(step))
    return out, lab


# ---------------------------------------------------------------- cluster bootstrap
def boot_weights(clusters, n_boot=10000, seed=0):
    """Cluster bootstrap: the sorted distinct clusters and the (n_boot, n_clusters) resample counts (clusters drawn with
    replacement, default_rng(seed))."""
    cl = sorted(set(clusters))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(cl), size=(n_boot, len(cl)))
    W = np.zeros((n_boot, len(cl)))
    np.add.at(W, (np.repeat(np.arange(n_boot), len(cl)), pick.ravel()), 1)
    return cl, W


def bootstrap_proteins(df, stat, n=1000, seed=0, group="cluster"):
    """Cluster bootstrap: resample clusters with replacement, sum their rows, apply stat(sum_df) -> dict of CIs."""
    if group == "cluster" and "cluster" not in df.columns:
        df = df.assign(cluster=(df.query_key if "query_key" in df.columns else df.protein).map(cluster_of))
    g = df.groupby(group).sum(numeric_only=True)
    cols = list(g.columns)
    _, w = boot_weights(g.index, n, seed)
    S = w @ g.to_numpy(dtype=float)             # (n, n_cols) resampled sums
    res = stat({c: S[:, j] for j, c in enumerate(cols)})
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in res.items()}


def entry_weighted(df, cols, n_boot=1000, seed=0):
    """Entry-weighted mean (mean within each PDB entry, then across entries) with a cluster-bootstrap 95 % CI, beside
    the query-weighted and cluster-weighted means. df needs 'protein' (+ 'query_key' or 'cluster')."""
    out = {}
    d = df if "cluster" in df.columns else \
        df.assign(cluster=(df.query_key if "query_key" in df.columns else df.protein).map(cluster_of))
    cl, W = boot_weights(d.cluster, n_boot, seed)
    for c in cols:
        v = d[np.isfinite(d[c].astype(float))]
        if not len(v):
            continue
        ent = v.groupby(["cluster", "protein"])[c].mean().reset_index()
        e_sum = ent.groupby("cluster")[c].sum().reindex(cl).fillna(0).to_numpy()
        e_n = ent.groupby("cluster")[c].size().reindex(cl).fillna(0).to_numpy()
        q_sum = v.groupby("cluster")[c].sum().reindex(cl).fillna(0).to_numpy()
        q_n = v.groupby("cluster")[c].size().reindex(cl).fillna(0).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            be = (W @ e_sum) / (W @ e_n)
            bq = (W @ q_sum) / (W @ q_n)
            cm = e_sum / e_n                                   # cluster means of entry means (NaN: no value)
            ok = np.isfinite(cm)
            bc = (W @ np.where(ok, cm, 0.0)) / (W @ ok.astype(float))
        out[c] = dict(entry_mean=float(ent[c].mean()), entry_ci_lo=float(np.nanpercentile(be, 2.5)),
                      entry_ci_hi=float(np.nanpercentile(be, 97.5)), n_entries=int(len(ent)),
                      query_mean=float(v[c].mean()), query_ci_lo=float(np.nanpercentile(bq, 2.5)),
                      query_ci_hi=float(np.nanpercentile(bq, 97.5)), n_queries=int(len(v)),
                      cluster_mean=float(cm[ok].mean()), cluster_ci_lo=float(np.nanpercentile(bc, 2.5)),
                      cluster_ci_hi=float(np.nanpercentile(bc, 97.5)), n_clusters=int(ok.sum()),
                      median=float(v[c].median()))
    return out


def cluster_sums(df, col, cl, entry):
    """Per cluster of cl: sum and count of the finite values of col (entry-weighted: the pockets of an entry averaged
    first), the mean and the number of values."""
    v = df[np.isfinite(df[col].astype(float))]
    if entry:
        v = v.groupby(["cluster", "protein"])[col].mean().reset_index()
    s = v.groupby("cluster")[col].sum().reindex(cl).fillna(0).to_numpy()
    n = v.groupby("cluster")[col].size().reindex(cl).fillna(0).to_numpy()
    return s, n, float(v[col].mean()), len(v)


def summary(df, col, W, cl, entry=True):
    """Mean of col (entry-weighted: pockets of an entry averaged first) with a 95 % cluster-bootstrap interval from the
    resample counts W over the clusters cl (boot_weights)."""
    s, n, m, cnt = cluster_sums(df, col, cl, entry)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W @ s) / (W @ n)
    return dict(mean=m, lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5)), n=cnt), b


# ---------------------------------------------------------------- ranking metrics
def auc_from_ranks(r, lab):
    """ROC-AUC from precomputed ranks r of the scores; NaN without positives or negatives."""
    npos = int(lab.sum())
    nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return float((r[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def mw_auc(score, lab):
    """Mann-Whitney ROC-AUC with mid-ranks (score higher = more similar); NaN without positives or negatives."""
    return auc_from_ranks(rankdata(score), lab)


def enrich(score, lab, frac=0.01, tie=None):
    """RE@frac (TPR at FPR frac / frac), EF@frac, EF_max@frac; ties broken by tie (ascending; default the database
    index)."""
    n = len(score)
    npos = int(lab.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan, np.nan, np.nan
    ls = lab[strict_order(score, np.arange(n) if tie is None else tie)]
    fp_cum = np.cumsum(~ls)
    stop = np.searchsorted(fp_cum, int(np.floor(frac * nneg)) + 1)
    re1 = ls[:stop].sum() / npos / frac
    top = max(1, int(np.ceil(frac * n)))
    p = npos / n
    return float(re1), float(ls[:top].mean() / p), float(min(npos, top) / top / p)


# ---------------------------------------------------------------- ligands and homology
def ligand_code(root, key, first_only=True):
    """First HETATM residue name of the site's ligand file (or the set of all names with first_only=False)."""
    pdb, pocket = key.split("/")
    f = os.path.join(root, pdb, pocket.replace("ligand_pocket_", "ligand_"))
    if not os.path.exists(f):
        return None
    names = []
    with open(f) as fh:
        for line in fh:
            if line.startswith("HETATM"):
                n = line[17:20].strip()
                if first_only:
                    return n
                if n not in names:
                    names.append(n)
    return names or None


def load_cache():
    """comp_id -> SMILES from the CCD snapshot (empty string = no SMILES in the CCD)."""
    if not os.path.exists(SNAP):
        die(f"CCD snapshot {SNAP} is missing (release asset reference_data; set LENSEEK_REFERENCE_DIR)")
    c = pd.read_csv(SNAP, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(c.comp_id, c.smiles))


def fingerprints(ids, cache):
    """CCD code -> Morgan fingerprint (radius 2, 2,048 bits) of its SMILES in cache, None without one."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}
    for i in ids:
        smi = cache.get(i, "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        fps[i] = gen.GetFingerprint(mol) if mol is not None else None
    return fps


def bulk_dice(a, bs):
    """Dice similarities of fingerprint a to every fingerprint of bs."""
    from rdkit import DataStructs
    return DataStructs.BulkDiceSimilarity(a, bs)


def site_tables():
    """Query and database site annotations (scripts/site_annotation.py annotate): query_sites.tsv, db_sites.tsv."""
    for f in ("query_sites.tsv", "db_sites.tsv"):
        if not os.path.exists(os.path.join(ANNOT, f)):
            die(f"{os.path.join(ANNOT, f)} missing: run scripts/site_annotation.py annotate first")
    kw = dict(sep="\t", dtype={"main_chain": str, "ligand": str, "chains": str}, keep_default_na=False)
    return pd.read_csv(os.path.join(ANNOT, "query_sites.tsv"), **kw), pd.read_csv(os.path.join(ANNOT, "db_sites.tsv"),
                                                                                  **kw)


def pfam_dir():
    """Folder of the SIFTS chain-to-Pfam table and the Pfam clan table: $LENSEEK_REFERENCE_DIR/pfam/ (the copies the
    results used; SIFTS changes weekly), else $LENSEEK_OUT/cache/pfam/ (examples/analysis.sh pfam)."""
    for d in PFAM_DIRS:
        if all(os.path.exists(os.path.join(d, f)) for f in PFAM_FILES):
            return d
    die(f'{" and ".join(PFAM_FILES)} are in neither {" nor ".join(PFAM_DIRS)} (release asset reference_data, or '
        f"examples/analysis.sh pfam)")


def pfam_clans():
    """Chain -> set of Pfam clans (a family without a clan counts as its own clan)."""
    d = pfam_dir()
    fam2clan = {}
    with gzip.open(os.path.join(d, "Pfam-A.clans.tsv.gz"), "rt") as f:
        for l in f:
            p = l.rstrip("\n").split("\t")
            fam2clan[p[0]] = p[1] if p[1] else f"fam:{p[0]}"
    clans = collections.defaultdict(set)
    with gzip.open(os.path.join(d, "pdb_chain_pfam.tsv.gz"), "rt") as f:
        for l in f:
            if l.startswith("#") or l.startswith("PDB\t"):
                continue
            p = l.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            clans[f"{p[0].lower()}_{p[1]}"].add(fam2clan.get(p[3], f"fam:{p[3]}"))
    return clans


def identity_hits(m8=None, min_cov=0.5):
    """query chain -> {train chain: max fident over hits with qcov or tcov >= min_cov} of an MMseqs2 search (default
    $LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8, the evaluation chains)."""
    out = collections.defaultdict(dict)
    if m8 is None and not STRAT_DIR:
        die("$LENSEEK_STRAT_DIR is not set (directory with eval_vs_train_uncapped.m8)")
    m8 = m8 or M8
    if not os.path.exists(m8):
        die(f"{m8} missing" + (" ($LENSEEK_STRAT_DIR)" if m8 == M8 else ""))
    with open(m8) as f:
        for l in f:
            q, t, fid, aln, ql, tl, qc, tc = l.split("\t")[:8]
            if float(qc) >= min_cov or float(tc) >= min_cov:
                v = float(fid)
                if v > out[q].get(t, 0.0):
                    out[q][t] = v
    return out


class Homology:
    """Main-chain identity of each query to every database site and the homology-unrelated restriction. Queries
    outside the site annotation (the binding sites from later PDB entries) give their m8 query names (q_chains) and
    search (m8); they get the identity only."""
    def __init__(self, q_keys, db_keys, q_chains=None, m8=None):
        qs, ds = site_tables()
        dm = dict(zip(ds.key, ds.main_chain))
        if q_chains is None:
            qm = dict(zip(qs.key, qs.main_chain))
            q_chains = [chain_name(k, qm[k]) for k in q_keys]
        self.q_chain = list(q_chains)
        db_chain = [chain_name(k, dm[k]) for k in db_keys]
        self.chains, self.db_ci = np.unique(np.array(db_chain), return_inverse=True)
        self.hits = identity_hits(m8)
        if m8 is not None:
            return
        clans = pfam_clans()
        self.q_clans = [clans.get(c, set()) for c in self.q_chain]
        ch_clans = [clans.get(c, set()) for c in self.chains]
        self.ch_has_pfam = np.array([bool(c) for c in ch_clans])
        inv = collections.defaultdict(list)
        for i, cs in enumerate(ch_clans):
            for c in cs:
                inv[c].append(i)
        self.inv = {c: np.array(v) for c, v in inv.items()}
        self.db_has_pfam = self.ch_has_pfam[self.db_ci]

    def identity(self, qi):
        v = np.zeros(len(self.chains))
        h = self.hits.get(self.q_chain[qi], {})
        if h:
            idx = np.searchsorted(self.chains, list(h))
            ok = (idx < len(self.chains))
            ok[ok] &= self.chains[idx[ok]] == np.array(list(h))[ok]
            v[idx[ok]] = np.array(list(h.values()))[ok]
        return v[self.db_ci]

    def unrelated(self, qi, ident=None):
        """DB sites < 30 % identical to query qi and in a different Pfam clan (both annotated)."""
        ident = self.identity(qi) if ident is None else ident
        if not self.q_clans[qi]:
            return np.zeros(len(self.db_ci), bool), False
        shared = np.zeros(len(self.chains), bool)
        for c in self.q_clans[qi]:
            if c in self.inv:
                shared[self.inv[c]] = True
        diff = self.ch_has_pfam & ~shared
        return (ident < UNRELATED_IDENT) & diff[self.db_ci], True


def ligand_tables():
    """Ligand codes of the query and database sites (site annotation) and a Dice lookup on their Morgan fingerprints
    (radius 2, 2,048 bits) from the CCD snapshot."""
    from rdkit import DataStructs
    qs, ds = site_tables()
    qcode, dcode = list(qs.ligand), list(ds.ligand)
    fps = fingerprints({c for c in qcode + dcode if c}, load_cache())
    memo = {}

    def dice(x, y):
        if not x or not y or fps.get(x) is None or fps.get(y) is None:
            return np.nan
        if (x, y) not in memo:
            memo[(x, y)] = DataStructs.DiceSimilarity(fps[x], fps[y])
        return memo[(x, y)]
    return qcode, dcode, dice


# ---------------------------------------------------------------- training runs of the variant table
RUNS = [  # run, node features, training sites, group, row label of Supplementary Table S14
    ("main_s0", "ankh", 114661, "seed", "Original model (seed 0)"),
    ("main_s1", "ankh", 114661, "seed", "Original model, seed 1"),
    ("main_s2", "ankh", 114661, "seed", "Original model, seed 2"),
    ("strict30_s0", "ankh", 101977, "auxiliary", "Trained without entries >=30% identical to an evaluation protein"),
    ("strict50_s0", "ankh", 111255, "auxiliary", "Trained without entries >=50% identical to an evaluation protein"),
    ("plm_esm2", "esm2", 113812, "language_model", "ESM2 node features"),
    ("plm_esm3", "esm3", 113080, "language_model", "ESM3 node features"),
    ("plm_esmc", "esmc", 113812, "language_model", "ESM C node features"),
    ("plm_none", "none", 114661, "language_model", "No node features (structure only)"),
    ("abl_onehot", "onehot", 113830, "auxiliary", "One-hot residue type"),
    ("abl_ankh_fullgraph", "ankh", 114661, "structure", "Ankh only, fully connected"),
    ("abl_ankh_fullgraph_na", "ankh", 114661, "structure", "Ankh only, fully connected, raw Ankh target"),
    ("abl_ankh_only", "ankh", 114661, "auxiliary", "No geometric edges, 15-NN graph"),
    ("abl_nodir", "ankh", 114661, "auxiliary", "No direction channel"),
    ("abl_norbf", "ankh", 114661, "auxiliary", "No RBF channel"),
    ("abl_noquat", "ankh", 114661, "auxiliary", "No quaternion channel"),
    ("abl_noFAPE", "ankh", 114661, "auxiliary", "No FAPE loss"),
    ("abl_noRMSD", "ankh", 114661, "auxiliary", "No RMSD loss"),
    ("abl_noCosNorm", "ankh", 114661, "auxiliary", "No cosine and norm losses"),
    ("abl_beta0", "ankh", 114661, "auxiliary", "No KL term (beta = 0)"),
    ("abl_nonadapted", "ankh", 114661, "structure", "Non-adapted Model (raw 768-d Ankh target)"),
    ("abl_stopgrad", "ankh", 114661, "auxiliary", "Adapted target detached"),
    ("filter_free_s0", "ankh", 149637, "auxiliary", "Trained without the ligand filters"),
]


# ---------------------------------------------------------------- files and figures
def apply_style(base=6.0, lw=0.8):
    """rcParams for print-size figures: a font of N pt prints at N pt."""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": base, "ytick.labelsize": base, "legend.fontsize": base,
        "axes.linewidth": lw, "xtick.major.width": lw, "ytick.major.width": lw,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42})


def savefig(fig, path_noext, dpi=300):
    """Write fig as <path_noext>.png (dpi) and .pdf, then close it."""
    import matplotlib.pyplot as plt
    fig.savefig(path_noext + ".png", dpi=dpi)
    fig.savefig(path_noext + ".pdf")
    plt.close(fig)
    print(f"  wrote {path_noext}.png/.pdf")
