#!/usr/bin/env python
"""Top-100 lists of the sequence-identity and random baselines (Supplementary Table S13 A; analysis/table1_concordance.py
reads them, analysis/baselines.py compares the baselines with ProBiS and the ligand labels over the whole database).

Baselines (higher score = more similar) for every evaluation query q and database site s (db_paths.txt order):
  seqid      the sequence-identity baseline: the largest MMseqs2 fident between the main chains of q and s over hits
             with query or target coverage >= 0.5 ($LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8; no hit = 0); main
             chain = the chain with the most binding-site residues (site annotation tables)
  random     uniform(0, 1) from default_rng(SeedSequence([0, q_idx]))
Ties: the top-k sorts by (score desc, key asc), key = default_rng(SeedSequence([1, q_idx])).random(n_db), one key per
query for every baseline.

Subcommand and output ($LENSEEK_OUT/baselines/, default ./out/baselines/):
  top100    top100/{seqid,random}_top100.tsv: exactly 100 rows per query over all database sites, query_idx,
            query_path, rank, db_idx, target_path (paths from the scan site lists), score (full precision), n_tied
            (database sites with exactly this score), primary
Inputs: LENSEEK_EVAL_DIR (q_paths.txt, db_paths.txt), LENSEEK_PAIRS_DIR (scan site lists all_query_sites.txt,
all_db_sites.txt), LENSEEK_STRAT_DIR, $LENSEEK_OUT/cache/site_annot/ (scripts/site_annotation.py annotate),
data/query_sets/primary_exclude_pdb.txt.

usage: python scripts/baseline_features.py top100
"""
import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import REPO, require_path, site_key, strict_order, tie_key  # noqa: E402

OUT_ROOT = os.path.abspath(os.environ.get("LENSEEK_OUT") or "out")
OUT = os.path.join(OUT_ROOT, "baselines")
ANNOT = os.path.join(OUT_ROOT, "cache", "site_annot")
PRIMARY_EXCLUDE_FILE = os.path.join(REPO, "data", "query_sets", "primary_exclude_pdb.txt")
BASELINES = ["seqid", "random"]
RANDOM_SEED = 0
K = 100


# ---------------------------------------------------------------- helpers
def eval_sites(eval_dir):
    """(query keys, database keys) of an evaluation folder."""
    require_path(eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
    return tuple([site_key(p) for p in open(os.path.join(eval_dir, f)).read().split()]
                 for f in ("q_paths.txt", "db_paths.txt"))


class SeqIdentity:
    """Main-chain identity of each query to every database site (the seqid baseline)."""

    def __init__(self, q_keys, db_keys, strat_dir):
        kw = dict(sep="\t", dtype={"main_chain": str, "ligand": str, "chains": str}, keep_default_na=False)
        qs, ds = (pd.read_csv(require_path(os.path.join(ANNOT, f), f, "LENSEEK_OUT (scripts/site_annotation.py)"), **kw)
                  for f in ("query_sites.tsv", "db_sites.tsv"))
        qm, dm = dict(zip(qs.key, qs.main_chain)), dict(zip(ds.key, ds.main_chain))
        self.q_chain = [f"{k.split('/')[0]}_{qm[k]}" for k in q_keys]
        db_chain = [f"{k.split('/')[0]}_{dm[k]}" for k in db_keys]
        self.chains, self.db_ci = np.unique(np.array(db_chain), return_inverse=True)
        m8 = require_path(os.path.join(strat_dir or "", "eval_vs_train_uncapped.m8"), "--strat-dir", "LENSEEK_STRAT_DIR")
        self.hits = collections.defaultdict(dict)   # eval chain -> {train chain: max fident, qcov or tcov >= 0.5}
        with open(m8) as f:
            for l in f:
                q, t, fid, aln, ql, tl, qc, tc = l.split("\t")[:8]
                if float(qc) >= 0.5 or float(tc) >= 0.5:
                    v = float(fid)
                    if v > self.hits[q].get(t, 0.0):
                        self.hits[q][t] = v

    def __call__(self, qi):
        v = np.zeros(len(self.chains))
        h = self.hits.get(self.q_chain[qi], {})
        if h:
            idx = np.searchsorted(self.chains, list(h))
            ok = (idx < len(self.chains))
            ok[ok] &= self.chains[idx[ok]] == np.array(list(h))[ok]
            v[idx[ok]] = np.array(list(h.values()))[ok]
        return v[self.db_ci]


# ---------------------------------------------------------------- top-100 lists
def cmd_top100(a):
    q_keys, db_keys = eval_sites(a.eval_dir)
    n_db = len(db_keys)
    seqid = SeqIdentity(q_keys, db_keys, a.strat_dir)
    pairs = require_path(a.pairs_dir, "--pairs-dir", "LENSEEK_PAIRS_DIR")
    qp = open(os.path.join(pairs, "all_query_sites.txt")).read().split()
    dp = open(os.path.join(pairs, "all_db_sites.txt")).read().split()
    assert [site_key(p) for p in qp] == q_keys and [site_key(p) for p in dp] == db_keys
    excl = {l.split()[0].lower() for l in open(PRIMARY_EXCLUDE_FILE) if l.strip() and not l.startswith("#")}
    prim = {i for i, k in enumerate(q_keys) if k.split("/")[0].lower() not in excl}
    rows = {n: [] for n in BASELINES}
    for qi in range(len(q_keys)):
        tie = tie_key(qi, n_db)
        scores = dict(seqid=seqid(qi),
                      random=np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, int(qi)])).random(n_db))
        for n, s in scores.items():
            o = strict_order(s, tie)[:K]
            vals, cnt = np.unique(s, return_counts=True)
            nt = cnt[np.searchsorted(vals, s[o])]
            rows[n].append(pd.DataFrame({"query_idx": qi, "query_path": qp[qi], "rank": np.arange(1, K + 1),
                                         "db_idx": o, "target_path": [dp[j] for j in o], "score": s[o],
                                         "n_tied": nt, "primary": int(qi in prim)}))
    od = os.path.join(a.out, "top100")
    os.makedirs(od, exist_ok=True)
    for n, parts in rows.items():
        df = pd.concat(parts, ignore_index=True)
        assert (df.groupby("query_idx").size() == K).all()
        f = os.path.join(od, f"{n}_top{K}.tsv")
        df.to_csv(f, sep="\t", index=False, float_format="%.17g")   # full precision (thresholds on a 0.001 grid)
        print(f"{f}: {len(df)} rows; top-1 inside a tie for {df[df['rank'] == 1].n_tied.gt(1).mean():.1%} of the queries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("top100")
    p.set_defaults(fn=cmd_top100)
    p.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"), help="LEN-Seek ($LENSEEK_EVAL_DIR)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--pairs-dir", default=os.environ.get("LENSEEK_PAIRS_DIR"),
                   help="scan site lists all_query_sites.txt, all_db_sites.txt ($LENSEEK_PAIRS_DIR)")
    p.add_argument("--strat-dir", default=os.environ.get("LENSEEK_STRAT_DIR"),
                   help="eval_vs_train_uncapped.m8 ($LENSEEK_STRAT_DIR)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
