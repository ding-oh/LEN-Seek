#!/usr/bin/env python
"""The false-negative and false-positive cases of main-text Figures 6 and 7.

Candidates are the top-100 pairs of the primary queries that ProBiS (database-wide labels, Z >= 2.0 = similar) and
LEN-Seek disagree on, with LEN-Seek 'similar' = Chamfer distance <= t, t = the median of the five out-of-fold
thresholds (threshold_cv.csv of table1_concordance.py). False negatives: Z >= 2 and CD > t; false positives: no
alignment or Z < 2 and CD <= t. Pairs are grouped by ligand: identical CCD code, Dice >= 0.7 of 2048-bit Morgan
fingerprints (radius 2, CCD SMILES), or below. A pair is eligible when both ligands have >= 6 heavy atoms, the
query ligand is not on the training exclusion lists, neither site is co-bound, both main chains have a sequence, and
its class holds at every fold threshold. Eligible pairs are ranked by main-chain sequence identity (MMseqs2, coverage
>= 0.5 of either chain, no hit = below 25 %), then by Z (descending, FN) or CD (ascending, FP). The cases shown in
Figures 6 and 7 are listed in CASES below; their values are the identity, the TM-score of the main chains normalised
by the query chain (Foldseek TM-align mode), CD, Z and Dice. The structure renderings of the two figures are images
made outside this code. The false negatives and false positives of the Discussion and their G-LoSA GA-scores are
counted at the fixed threshold of Table 1 (thresholds.csv of table1_concordance.py).

Outputs ($LENSEEK_OUT/cases/):
  selection.csv            the cases shown in Figures 6 and 7 (Fig 6 A-C, Fig 7 A-C) with their values
  candidates.csv           every FN / FP pair of the top 100 at t, per figure and ligand group in the order above, with
                           the reasons that make a pair ineligible
  counts_by_scenario.csv   FN / FP counts per scenario at k = 1..100 at the fixed threshold of Table 1, and the pairs
                           among them with GA >= 0.59 (main-text Discussion)
  structs/                 main-chain PDB files given to TM-align

Inputs: see `python analysis/run.py -h`; read here: FOLDSEEK (Foldseek executable, default foldseek), the out-of-fold
and fixed thresholds (--threshold-cv, --thresholds: threshold_cv.csv and thresholds.csv of table1_concordance.py
table1), the database chains and the MMseqs2 search of the evaluation chains against them (db_chains.fasta of
scripts/site_annotation.py annotate, eval_vs_db_sens.m8 of db-search).

usage: python analysis/cases.py
"""
import argparse
import collections
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LENSEEK, OUT_ROOT, QUERY_SETS, EvalOutputs, attach_pairs, chain_name, die,  # noqa: E402
                    fingerprints, load_cache, load_pairs, probis_truth, topk_rows)
from lenseek.exclusion import REMOVE_HET_NAMES  # noqa: E402

FOLDSEEK = os.environ.get("FOLDSEEK", "foldseek")
K_VALUES = [1, 5, 10, 20, 50, 100]
K_SELECT = 100
DICE_SIMILAR = 0.7
MIN_HEAVY = 6
MIN_COV = 0.5
SCENARIOS = ["identical", "similar", "dissimilar"]
PANEL = dict(zip(SCENARIOS, "ABC"))
FIG = {"FN": "Fig 6", "FP": "Fig 7"}
CASES = [  # the cases shown in Figures 6 and 7: figure, panel, query site, database site
    ("Fig 6", "A", "7yw2/ligand_pocket_1_A.pdb", "6ede/ligand_pocket_1_A.pdb"),
    ("Fig 6", "B", "8iyb/ligand_pocket_1_A.pdb", "3pfb/ligand_pocket_2_B.pdb"),
    ("Fig 6", "C", "8szu/ligand_pocket_1_AB.pdb", "5bwz/ligand_pocket_1_AB.pdb"),
    ("Fig 7", "A", "8iaa/ligand_pocket_1_A.pdb", "6g80/ligand_pocket_3_C.pdb"),
    ("Fig 7", "B", "8kih/ligand_pocket_1_A.pdb", "6wkh/ligand_pocket_1_A.pdb"),
    ("Fig 7", "C", "8erb/ligand_pocket_13_N.pdb", "1asc/ligand_pocket_1_A.pdb"),
]
COFACTOR_SMARTS = {
    "nucleoside": "[#7;R]-[#6;r5]1-[#8;r5]-[#6;r5]-[#6;r5]-[#6;r5]-1",
    "flavin": "[#6]1~[#6]~[#6]2~[#7]~[#6]3~[#6](~[#8])~[#7]~[#6](~[#8])~[#7]~[#6]~3~[#7]~[#6]~2~[#6]~[#6]~1",
    "pyridoxal": "[CH3]c1[n,n+]cc(C[O]P(~O)(~O)~O)c(*)c1[OX2,OX1-]",
    "pterin": "[#7]~[#6]1~[#7]~[#6]~2~[#7]~[#6]~[#6]~[#7]~[#6]~2~[#6](~[#8])~[#7]~1",
    "thiamine": "[#6]1~[#7]~[#6](~[#7])~[#6](~[CH2]~[#7]2~[#6]~[#16]~[#6]~[#6]~2)~[#6]~[#7]~1",
    "biotin": "[#8]=[#6]1-[#7]-[#6]2-[#6]-[#16]-[#6]-[#6]-2-[#7]-1",
    "tetrapyrrole": "[#6]1~[#6]~[#6]2~[#7]~[#6]~1~[#6]~[#6]1~[#6]~[#6]~[#6](~[#7]~1)~[#6]~[#6]1~[#6]~[#6]~[#6]"
                    "(~[#7]~1)~[#6]~[#6]1~[#6]~[#6]~[#6](~[#7]~1)~[#6]~2",
}
COFACTOR_LIST = set("HEM HEC HEA HEB HEO HAS HDD HNI DHE SRM COH ZNH CLA BCL BCB CHL PHO F43 B12 COB CNC PQQ TPQ UQ1 "
                    "UQ2 UQ5 UQ9 U10 MQ7 MQ8 MQ9 PL9 SF4 FES F3S CLF LPA LPB TPP TDP FAD FMN NAD NAP NDP NAI COA SAM "
                    "SAH PLP PMP BTN FOL THF MGD MTE PCD".split())


class LigandInfo:
    """code -> fingerprint (common.fingerprints), heavy-atom count, cofactor family."""

    def __init__(self, codes):
        cache = load_cache()
        pats = {k: Chem.MolFromSmarts(v) for k, v in COFACTOR_SMARTS.items()}
        codes = sorted({c for c in codes if isinstance(c, str) and c})
        self.fp, self.heavy, self.cof = fingerprints(codes, cache), {}, {}
        for c in codes:
            smi = cache.get(c, "")
            mol = Chem.MolFromSmiles(smi) if smi else None
            self.heavy[c] = mol.GetNumHeavyAtoms() if mol is not None else np.nan
            fam = [k for k, p in pats.items() if mol is not None and mol.HasSubstructMatch(p)]
            if c in COFACTOR_LIST:
                fam.append("list")
            self.cof[c] = ",".join(fam)

    def dice(self, a, b):
        fa, fb = self.fp.get(a), self.fp.get(b)
        if fa is None or fb is None:
            return np.nan
        return DataStructs.DiceSimilarity(fa, fb)


def read_fasta_names(path):
    return {l[1:].strip().split()[0] for l in open(path) if l.startswith(">")}


def identity_table(pairs_needed, m8_files):
    """(query chain, target chain) -> max fident over hits with qcov or tcov >= 0.5 in either search."""
    need = collections.defaultdict(set)
    for q, t in pairs_needed:
        need[q].add(t)
    best, src = {}, {}
    for path, tag in m8_files:
        with open(path) as f:
            for l in f:
                p = l.split("\t")
                q, t = p[0], p[1]
                if q not in need or t not in need[q]:
                    continue
                if float(p[6]) >= MIN_COV or float(p[7]) >= MIN_COV:
                    v = float(p[2])
                    if v > best.get((q, t), -1):
                        best[(q, t)] = v
                        src[(q, t)] = tag
    return best, src


def scenario(qlig, tlig, d):
    if not isinstance(qlig, str) or not isinstance(tlig, str) or not qlig or not tlig or not np.isfinite(d):
        return "undetermined"
    if qlig == tlig:
        return "identical"
    return "similar" if d >= DICE_SIMILAR else "dissimilar"


def extract_chain(src, chain, dst):
    """ATOM records of one chain (first model, first altloc) of protein.pdb -> dst."""
    if os.path.exists(dst):
        return dst
    out = []
    with open(src) as f:
        for l in f:
            if l.startswith("ENDMDL"):
                break
            if l.startswith("ATOM") and l[21] == chain and l[16] in " A":
                out.append(l[:16] + " " + l[17:])
    if not out:
        die(f"no ATOM records for chain {chain} in {src}")
    with open(dst, "w") as f:
        f.writelines(out)
        f.write("END\n")
    return dst


def tmalign(qpdb, tpdb):
    """Foldseek TM-align mode, one pair: dict(qtm, ttm, alntm, rmsd, fident_aln, alnlen, u, t)."""
    with tempfile.TemporaryDirectory(dir=os.path.dirname(qpdb)) as tmp:
        res = os.path.join(tmp, "r.m8")
        cmd = [FOLDSEEK, "easy-search", qpdb, tpdb, res, os.path.join(tmp, "tmp"), "--exhaustive-search", "1",
               "--alignment-type", "1", "--threads", "2", "-e", "inf", "-v", "1", "--format-output",
               "query,target,fident,alnlen,qtmscore,ttmscore,alntmscore,rmsd,u,t,prob"]
        r = subprocess.run(["nice", "-n", "19"] + cmd, capture_output=True, text=True)
        if r.returncode != 0:
            die(f"foldseek failed: {r.stderr[-500:]}")
        lines = [l.rstrip("\n").split("\t") for l in open(res) if l.strip()]
    if not lines:
        return dict(qtm=np.nan, ttm=np.nan, alntm=np.nan, rmsd=np.nan, fident_aln=np.nan, alnlen=np.nan, u="", t="")
    p = lines[0]
    return dict(fident_aln=float(p[2]), alnlen=int(p[3]), qtm=float(p[4]), ttm=float(p[5]), alntm=float(p[6]),
                rmsd=float(p[7]), u=p[8], t=p[9])


def display(seq_identity, tm_qtm, cd, probis_class, probis_z, dice):
    return dict(disp_identity="< 25 %" if not np.isfinite(seq_identity) else f"{100 * seq_identity:.0f} %",
                disp_tm=f"{tm_qtm:.2f}" if np.isfinite(tm_qtm) else "n.a.", disp_cd=f"{cd:.2f}",
                disp_z="no alignment" if probis_class == "no_alignment" else f"{probis_z:.2f}",
                disp_dice=f"{dice:.2f}")


def candidate_pairs(ev, pairs, thr):
    """Top-100 rows of the primary queries with their ProBiS labels, the FN / FP kind at t and the confusion cell."""
    m = attach_pairs(topk_rows(ev, K_SELECT), load_pairs(pairs))
    m["truth"] = probis_truth(m)
    return classify(m, thr)


def classify(m, thr):
    """The prediction at Chamfer distance <= thr, the FN / FP kind and the confusion cell of every row of m."""
    m["pred"] = m.cd <= thr
    m["kind"] = np.where((m.truth == 1) & ~m.pred, "FN", np.where((m.truth == 0) & m.pred, "FP", ""))
    m.loc[~np.isfinite(m.truth), "kind"] = ""
    m["cell"] = np.where(~np.isfinite(m.truth), "", np.where(m.truth == 1, np.where(m.pred, "TP", "FN"),
                                                             np.where(m.pred, "FP", "TN")))
    return m


def annotate_sites(m, annot_dir):
    """Ligand (first HETATM), main chain, co-bound flag and size of both sites of every pair; returns the tables."""
    kw = dict(sep="\t", dtype={"main_chain": str, "ligand": str, "chains": str}, keep_default_na=False)
    qs = pd.read_csv(os.path.join(annot_dir, "query_sites.tsv"), **kw).set_index("key")
    ds = pd.read_csv(os.path.join(annot_dir, "db_sites.tsv"), **kw).set_index("key")
    for side, tab, col in [("q", qs, "query_key"), ("t", ds, "target_key")]:
        miss = ~m[col].isin(tab.index)
        if miss.any():
            die(f"{int(miss.sum())} {side} sites lack site annotations")
        m[f"{side}_ligand"] = m[col].map(tab.ligand).replace("", np.nan)
        m[f"{side}_main_chain"] = m[col].map(tab.main_chain)
        m[f"{side}_ambiguous"] = m[col].map(tab.ambiguous).astype(int)
        m[f"{side}_n_res"] = m[col].map(tab.n_res).astype(int)
    return qs, ds


def add_scenarios(d, li):
    """Dice, ligand scenario and cofactor flags of every pair."""
    d["dice"] = [li.dice(q, t) for q, t in zip(d.q_ligand, d.t_ligand)]
    d["scenario"] = [scenario(q, t, x) for q, t, x in zip(d.q_ligand, d.t_ligand, d.dice)]
    d["q_cofactor"] = d.q_ligand.map(li.cof).fillna("")
    d["t_cofactor"] = d.t_ligand.map(li.cof).fillna("")
    d["cofactor_pair"] = d.q_cofactor.ne("") | d.t_cofactor.ne("")


def scenario_counts(m, thr):
    """Every FN / FP at t per scenario and k (no eligibility filter)."""
    crow = []
    for k in K_VALUES:
        s = m[(m["rank"] <= k) & m.kind.ne("")]
        for kind in ["FN", "FP"]:
            sk = s[s.kind == kind]
            groups = [(sc, sk if sc == "ALL" else sk[sk.scenario == sc]) for sc in SCENARIOS + ["undetermined", "ALL"]]
            if kind == "FP":
                groups.append(("identical+similar (numerous cases)", sk[sk.scenario.isin(["identical", "similar"])]))
            for sc, ss in groups:
                crow.append(dict(k=k, kind=kind, scenario=sc, n_pairs=len(ss), n_queries=ss.q_idx.nunique(),
                                 n_entries=ss.protein.nunique(),
                                 frac_of_kind=len(ss) / len(sk) if len(sk) else np.nan, n_kind=len(sk),
                                 n_cofactor_pairs=int(ss.cofactor_pair.sum()),
                                 n_ga_ge_0_59=int((ss.ga >= 0.59).sum()),
                                 n_ga_scored=int(ss.glosa_class.eq("ok").sum())))
    counts = pd.DataFrame(crow)
    counts.insert(0, "threshold", thr)
    return counts


def eligible(m, li, a, fn_bound, fp_bound):
    """The FN / FP candidates with the reasons that make a pair ineligible and the main-chain identity."""
    c = m[m.kind.ne("")].copy()
    q_chains = read_fasta_names(os.path.join(QUERY_SETS, "eval_chains.fasta"))
    t_chains = read_fasta_names(os.path.join(a.annot_dir, "db_chains.fasta"))
    c["q_chain_id"] = [chain_name(k, ch) for k, ch in zip(c.query_key, c.q_main_chain)]
    c["t_chain_id"] = [chain_name(k, ch) for k, ch in zip(c.target_key, c.t_main_chain)]
    c["fold_stable"] = np.where(c.kind == "FN", c.cd > fn_bound, c.cd <= fp_bound)
    reasons = []
    for r in c.itertuples():
        why = []
        if r.scenario == "undetermined":
            why.append("undetermined_ligand")
        elif min(li.heavy.get(r.q_ligand, 0), li.heavy.get(r.t_ligand, 0)) < MIN_HEAVY:
            why.append(f"heavy_atoms<{MIN_HEAVY}")
        if isinstance(r.q_ligand, str) and r.q_ligand in REMOVE_HET_NAMES:
            why.append("query_ligand_in_training_exclusion")
        if r.q_ambiguous or r.t_ambiguous:
            why.append("co_bound_site")
        if r.q_chain_id not in q_chains or r.t_chain_id not in t_chains:
            why.append("main_chain_sequence_missing")
        if not r.fold_stable:
            why.append("unstable_across_fold_thresholds")
        reasons.append(";".join(why))
    c["ineligible"] = reasons
    ident, isrc = identity_table(set(zip(c.q_chain_id, c.t_chain_id)),
                                 [(os.path.join(a.strat_dir, "eval_vs_train_uncapped.m8"), "uncapped"),
                                  (os.path.join(a.annot_dir, "eval_vs_db_sens.m8"), "sens")])
    c["seq_identity"] = [ident.get((q, t), np.nan) for q, t in zip(c.q_chain_id, c.t_chain_id)]
    c["seq_identity_src"] = [isrc.get((q, t), "no_hit") for q, t in zip(c.q_chain_id, c.t_chain_id)]
    c["ident_sort"] = c.seq_identity.fillna(0.0)
    c["noalign"] = c.probis_class.eq("no_alignment")
    c["z_sort"] = c.probis_z.fillna(-np.inf)
    return c


def ranked_candidates(c):
    """Candidates of each figure and scenario ordered by identity, then Z (FN) or CD (FP)."""
    ranked = []
    for kind in ["FN", "FP"]:
        for sc in SCENARIOS:
            s = c[(c.kind == kind) & (c.scenario == sc)]
            if kind == "FN":
                s = s.sort_values(["ident_sort", "z_sort", "cd", "q_idx", "db_idx"],
                                  ascending=[True, False, False, True, True], kind="mergesort")
            else:
                s = s.assign(_na=~s.noalign).sort_values(["ident_sort", "cd", "_na", "z_sort", "q_idx", "db_idx"],
                                                         ascending=[True, True, True, True, True, True],
                                                         kind="mergesort").drop(columns="_na")
            s = s.assign(order_all=np.arange(1, len(s) + 1))
            el = s.ineligible.eq("")
            s["order_eligible"] = np.where(el, el.cumsum(), 0)
            ranked.append(s)
    return pd.concat(ranked, ignore_index=True)


def fixed_cases(c):
    """The rows of the candidates c for the cases shown in Figures 6 and 7 (CASES); panels A, B and C are the
    identical, similar and dissimilar ligand groups."""
    rows = []
    for fig, panel, qkey, tkey in CASES:
        r = c[(c.query_key == qkey) & (c.target_key == tkey)]
        if len(r) != 1 or FIG[r.kind.iloc[0]] != fig or PANEL.get(r.scenario.iloc[0]) != panel:
            die(f"{fig} {panel}: {qkey} - {tkey} is not a candidate of this panel in the top-{K_SELECT} at t")
        rows.append(dict(r.iloc[0], figure=fig, panel=panel))
    return pd.DataFrame(rows)


def tm_pair(a, sdir, qkey, qch, tkey, tch):
    """TM-align of the main chains of a pair (chain files kept in sdir)."""
    qp = os.path.join(a.query_root, qkey.split("/")[0], "protein.pdb")
    tp = os.path.join(a.db_root, tkey.split("/")[0], "protein.pdb")
    qf = extract_chain(qp, qch, os.path.join(sdir, f"q_{qkey.split('/')[0]}_{qch}.pdb"))
    tf = extract_chain(tp, tch, os.path.join(sdir, f"t_{tkey.split('/')[0]}_{tch}.pdb"))
    return tmalign(qf, tf)


SELECTION_COLUMNS = [
    "figure", "panel", "kind", "scenario", "query_key", "q_main_chain", "q_ligand", "q_cofactor", "target_key",
    "t_main_chain", "t_ligand", "t_cofactor", "cofactor_pair", "rank", "cd", "probis_z", "probis_status",
    "probis_class", "ga", "dice", "seq_identity", "seq_identity_src", "tm_qtm", "tm_ttm", "tm_alntm", "tm_rmsd",
    "tm_fident_aln", "tm_alnlen", "q_n_res", "t_n_res", "disp_identity", "disp_tm", "disp_cd", "disp_z", "disp_dice",
    "tm_u", "tm_t", "q_idx", "db_idx"]
CANDIDATE_COLUMNS = [
    "kind", "scenario", "order_all", "order_eligible", "ineligible", "query_key", "q_ligand", "target_key", "t_ligand",
    "rank", "cd", "fold_stable", "probis_z", "probis_class", "ga", "dice", "seq_identity", "seq_identity_src",
    "q_cofactor", "t_cofactor", "cofactor_pair", "q_ambiguous", "t_ambiguous", "q_idx", "db_idx"]


def panel_values(sel, a, sdir, thr):
    """TM-align of the panel pairs and the values of each panel."""
    tm_cols = ["qtm", "ttm", "alntm", "rmsd", "fident_aln", "alnlen", "u", "t"]
    for col in tm_cols:
        sel[f"tm_{col}"] = np.nan if col not in ("u", "t") else ""
    for i, r in sel.iterrows():
        res = tm_pair(a, sdir, r.query_key, r.q_main_chain, r.target_key, r.t_main_chain)
        for col in tm_cols:
            sel.at[i, f"tm_{col}"] = res[col]
    disp = [display(r.seq_identity, r.tm_qtm, r.cd, r.probis_class, r.probis_z, r.dice) for r in sel.itertuples()]
    sel = pd.concat([sel, pd.DataFrame(disp)], axis=1)
    sel["threshold"] = thr
    return sel[[k for k in SELECTION_COLUMNS if k in sel.columns] + ["threshold"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"))
    ap.add_argument("--pairs", default=os.environ.get("LENSEEK_PAIRS"))
    ap.add_argument("--threshold-cv", default=os.path.join(OUT_ROOT, "table1", "threshold_cv.csv"),
                    help="threshold_cv.csv of table1_concordance.py (median_fold_threshold, fold_thresholds)")
    ap.add_argument("--thresholds", default=os.path.join(OUT_ROOT, "table1", "thresholds.csv"),
                    help="thresholds.csv of table1_concordance.py (fixed_threshold of LEN-Seek, the threshold of "
                         "Table 1)")
    ap.add_argument("--annot-dir", default=os.path.join(OUT_ROOT, "cache", "site_annot"))
    ap.add_argument("--strat-dir", default=os.environ.get("LENSEEK_STRAT_DIR"))
    ap.add_argument("--query-root", default=os.environ.get("LENSEEK_QUERY_ROOT"))
    ap.add_argument("--db-root", default=os.environ.get("LENSEEK_BSDB_ROOT"))
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "cases"))
    a = ap.parse_args()
    for flag, v in [("--threshold-cv", a.threshold_cv), ("--thresholds", a.thresholds), ("--annot-dir", a.annot_dir),
                    ("--strat-dir", a.strat_dir), ("--query-root", a.query_root), ("--db-root", a.db_root)]:
        if not v or not os.path.exists(v):
            die(f"{flag} {v} does not exist")
    ev = EvalOutputs(a.eval_dir)
    out = a.out
    os.makedirs(out, exist_ok=True)
    t = pd.read_csv(a.threshold_cv)
    thr = float(t.median_fold_threshold.iloc[0])
    folds = [float(x) for x in str(t.fold_thresholds.iloc[0]).split(";")]
    fn_bound, fp_bound = max(folds), min(folds)
    t_fix = float(pd.read_csv(a.thresholds).set_index("method").fixed_threshold[LENSEEK])
    print(f"[threshold] t = {thr:g}, fold thresholds {folds} (FN needs CD > {fn_bound:g}, FP needs CD <= {fp_bound:g});"
          f" counts at the fixed threshold {t_fix:g}")

    m = candidate_pairs(ev, a.pairs, thr)
    annotate_sites(m, a.annot_dir)
    li = LigandInfo(set(m.q_ligand.dropna()) | set(m.t_ligand.dropna()))
    add_scenarios(m, li)
    scenario_counts(classify(m.copy(), t_fix), t_fix).to_csv(os.path.join(out, "counts_by_scenario.csv"), index=False)

    c = eligible(m, li, a, fn_bound, fp_bound)
    ranked_candidates(c)[CANDIDATE_COLUMNS].to_csv(os.path.join(out, "candidates.csv"), index=False)
    sel = fixed_cases(c)
    sdir = os.path.join(out, "structs")
    os.makedirs(sdir, exist_ok=True)
    sel = panel_values(sel, a, sdir, thr)
    sel.to_csv(os.path.join(out, "selection.csv"), index=False)
    print(f"FN {int((m.kind == 'FN').sum())}, FP {int((m.kind == 'FP').sum())} in the top-{K_SELECT}; outputs in {out}")


if __name__ == "__main__":
    main()
