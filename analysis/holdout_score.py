#!/usr/bin/env python
"""Binding sites from later PDB entries: thresholds and metrics. Produces Supplementary Table S11 (panels A-C) and the
rows of the later PDB entries in Supplementary Table S10 B.

The sites are built and scored once by scripts/holdout.py (score); their pairs are labelled by scripts/labels.py.
Subcommands, in order:
  thresholds  thresholds fixed on the evaluation data before the later PDB entries are scored. Per method, the
              threshold with the highest top-1 MCC against ProBiS over the primary queries (first maximum on the grid;
              a pair is predicted similar when d <= t; ProBiS Z >= 2 positive, no alignment negative, failures
              excluded): LEN-Seek Chamfer distance, grid 0.5, labels of its scored top-100 table, top-1 ties by DB
              index; DeeplyTough descriptor distance, grid 0.0001, queries without a descriptor left out, labels of the
              full ProBiS scan (Z = (raw - 1.958) / 2.21468, two decimals).
  ligand      per-query ligand ROC-AUC and RE@1% of the scoring run: LEN-Seek (Chamfer distance), the model trained
              without homologous entries, DeeplyTough, the sequence identity and a random ranking
              (SeedSequence([0, q_idx]); ties of the two baselines by the key SeedSequence([1, q_idx])); for the sites
              below 30 % identity the ligand ROC-AUC of LEN-Seek and of the model trained without homologous entries.
              Ligand labels: Dice >= 0.46 of Morgan r2 2048-bit fingerprints of the CCD SMILES; database sites without
              a fingerprint are left out; ROC-AUC with mid-ranks; RE@1% = TPR at 1 % FPR / 0.01 over a strict order
              (distances ascending, ties by database index; scores descending, ties by the key), queries with >= 5
              positives. A DeeplyTough query without a descriptor is a failure, not a negative. Sequence identity =
              largest MMseqs2 fident (query or target coverage >= 0.5) between the main chains, from --seqid-m8:
              easy-search of seqs/holdout_query.fasta against the training chains, -s 7.5 --max-seqs 200000 -e 10
              --min-seq-id 0.25
  mcc         top-k MCC (k = 1 ... 100) and P@1 against the ProBiS labels of the union of the top-100 lists, at each
              method's fixed threshold, 95 % percentile intervals from 10,000 resamples of the 50 % sequence clusters
              (seed 0); the paired difference LEN-Seek - DeeplyTough on the queries labelled for both; the top-1 rows
              by identity stratum (the stratum's clusters resampled)
  table       Supplementary Table S11 and the top-1 values of the later PDB entries from the per-query outputs of
              `ligand` and from `mcc`: entry-weighted means (the pockets of one PDB entry averaged first) with 95 %
              intervals from 10,000 resamples of the 50 % sequence clusters (seed 0); release-weighted means with the
              manifest's sampling weights

Inputs: see `python analysis/run.py -h`; read here only: LENSEEK_HOLDOUT_RUN (the scoring run, default
$LENSEEK_OUT/holdout/run), LENSEEK_HOLDOUT_LABELS (its ProBiS labels, scripts/labels.py tables) and
LENSEEK_HOLDOUT_THRESHOLDS (default the thresholds/ output), with the manifest of the sites LENSEEK_HOLDOUT_MANIFEST.
`ligand` takes the database ligand codes from asset reference_data.
Outputs: $LENSEEK_OUT/holdout_score/ (default ./out/holdout_score/): thresholds/; descriptive_per_query.csv,
below30_<model>_per_query.csv (ligand); mcc_holdout.json, mcc_strata.csv (mcc); table_S14A.csv, table_S14B.csv,
table_S14C.csv (Supplementary Table S11 A-C), table1B_holdout.csv (top-1 MCC, P@1 and ligand values per method)
(table).

usage: python analysis/holdout_score.py {thresholds,ligand,mcc,table} [options]   (-h per subcommand)
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT_ROOT, PRIMARY_EXCLUDE_FILE, PROBIS_NOALIGN, REFERENCE_DIR, Homology,  # noqa: E402
                    boot_weights, bulk_dice, die, enrich, fingerprints, load_cache, make_grid, mcc, mw_auc,
                    primary_excluded, read_tsv, require_path, site_key, summary, tie_key)

OUT = os.path.join(OUT_ROOT, "holdout_score")
ENV = os.environ.get
KS = (1, 5, 10, 20, 50, 100)
N_BOOT = 10000
THRESHOLD_FILES = {"lenseek_main_s0": "holdout_threshold.json", "deeplytough": "dt_threshold.json"}
DICE_CUT, RE_MIN_POS, RANDOM_SEED = 0.46, 5, 0
MODELS = ("main_s0", "strict30_s0")


def write_csv(path, rows):
    """List of dicts -> CSV (columns of the first row)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pair_labels(path, status=None):
    """{(query key, target key): ProBiS label} of a scored pair table: 1 at Z >= 2, 0 below or without an alignment,
    NaN for failed comparisons; counts the statuses into status (a dict) when given."""
    lab = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        st = r["probis_status"]
        if status is not None:
            if st == "pending":
                sys.exit("pending rows: labels not finished")
            status[st] = status.get(st, 0) + 1
        try:
            z = float(r["probis_z"])
        except ValueError:
            z = float("nan")
        lab[(site_key(r["query_path"]), site_key(r["target_path"]))] = \
            (1.0 if z >= 2.0 else 0.0) if st == "ok" and math.isfinite(z) else \
            (0.0 if st in PROBIS_NOALIGN else float("nan"))
    return lab


# ---------------------------------------------------------------- thresholds (evaluation data)
def cmd_thresholds(a):
    inputs = (("--eval-dir", "LENSEEK_EVAL_DIR", a.eval_dir, ("q_paths.txt", "db_paths.txt", "dist.npy")),
              ("--pairs", "LENSEEK_PAIRS", a.pairs, ()),
              ("--scan", "LENSEEK_SCAN_DIR", a.scan, ("all_status.npy", "all_raw.npy")),
              ("--dt-dist", "LENSEEK_DT_DIR", a.dt_dist, ()))
    for flag, var, path, files in inputs:   # every input before the output folder
        require_path(path, flag, var)
        for f in files:
            if not os.path.exists(os.path.join(path, f)):
                die(f"{os.path.join(path, f)} does not exist (set {flag} or ${var})")
    os.makedirs(a.out_dir, exist_ok=True)
    q_paths = open(os.path.join(a.eval_dir, "q_paths.txt")).read().split()
    db_paths = open(os.path.join(a.eval_dir, "db_paths.txt")).read().split()
    D = np.load(os.path.join(a.eval_dir, "dist.npy"), mmap_mode="r")
    excl = primary_excluded(a.exclude)
    sel = [i for i, p in enumerate(q_paths) if site_key(p).split("/")[0].lower() not in excl]
    lab = pair_labels(a.pairs)
    st = np.load(os.path.join(a.scan, "all_status.npy"), mmap_mode="r")
    raw = np.load(os.path.join(a.scan, "all_raw.npy"), mmap_mode="r")

    def scan_label(i, j):   # status 1 ok, 2 no alignment, other codes failures
        c = int(st[i, j])
        z = round((float(raw[i, j]) - 1.958) / 2.21468, 2) if c == 1 else float("nan")
        return (1.0 if z >= 2.0 else 0.0) if c == 1 else (0.0 if c == 2 else float("nan"))

    top = {"lenseek_main_s0": [], "deeplytough": []}
    DT = np.load(a.dt_dist, mmap_mode="r")
    for i in sel:
        row = np.asarray(D[i], dtype=np.float64)
        j = int(np.lexsort((np.arange(len(row)), row))[0])
        top["lenseek_main_s0"].append((row[j], lab[(site_key(q_paths[i]), site_key(db_paths[j]))]))
        v = np.asarray(DT[i], dtype=np.float64)
        idx = np.nonzero(np.isfinite(v))[0]
        if len(idx):
            j = int(idx[np.lexsort((idx, v[idx]))][0])
            top["deeplytough"].append((float(v[j]), scan_label(i, j)))
    for m, step in (("lenseek_main_s0", 0.5), ("deeplytough", 0.0001)):
        d = np.array([t[0] for t in top[m]])
        y = np.array([t[1] for t in top[m]])
        grid = make_grid(d, step)
        ok = np.isfinite(y)
        t_ok, d_ok = y[ok].astype(bool), d[ok]
        p = d_ok[None, :] <= grid[:, None]
        tp, fp = (p & t_ok).sum(1), (p & ~t_ok).sum(1)
        fn, tn = (~p & t_ok).sum(1), (~p & ~t_ok).sum(1)
        m_ = mcc(tp, fp, fn, tn)
        b = int(np.argmax(m_))
        res = dict(method=m, mcc_threshold=float(grid[b]), rule="pair predicted similar when d <= threshold",
                   selection_rule=f"max top-1 MCC over the primary queries, grid {step:g}, first maximum",
                   in_sample_top1_mcc=float(m_[b]),
                   in_sample_top1_counts=dict(TP=int(tp[b]), FP=int(fp[b]), FN=int(fn[b]), TN=int(tn[b])),
                   n_top1=len(top[m]), n_top1_labelled=int(ok.sum()), query_set=f"primary ({len(sel)} pockets)")
        json.dump(res, open(os.path.join(a.out_dir, THRESHOLD_FILES[m]), "w"), indent=1)
        with open(os.path.join(a.out_dir, THRESHOLD_FILES[m].replace("threshold.json", "mcc_curve.csv")), "w") as f:
            f.write("threshold,mcc,tp,fp,fn,tn\n")
            for gi, g_ in enumerate(grid):
                f.write(f"{g_:g},{m_[gi]:.6f},{tp[gi]},{fp[gi]},{fn[gi]},{tn[gi]}\n")
        print(f"{m}: threshold {grid[b]:g} (grid {step:g}), top-1 MCC {m_[b]:.4f}, TP/FP/FN/TN "
              f"{int(tp[b])}/{int(fp[b])}/{int(fn[b])}/{int(tn[b])}, {len(top[m])} top-1 pairs, {int(ok.sum())} labelled")


# ---------------------------------------------------------------- ligand (per-query ligand metrics of the run)
def ligand_labels(q_codes, db_codes, cut=DICE_CUT):
    """has (n_db bool: the database site has a fingerprint), {query ligand: n_db bool positives}."""
    fps = fingerprints(sorted({c for c in list(db_codes) + list(q_codes) if c}), load_cache())
    uniq = sorted(c for c, f in fps.items() if f is not None)
    col = {c: j for j, c in enumerate(uniq)}
    db_col = np.array([col.get(c, -1) for c in db_codes])
    has = db_col >= 0
    out = {}
    for c in set(q_codes):
        if c and fps.get(c) is not None:
            dice = np.array(bulk_dice(fps[c], [fps[u] for u in uniq]))
            lab = np.zeros(len(db_codes), bool)
            lab[has] = dice[db_col[has]] >= cut
            out[c] = lab
    return has, out


def random_row(q_idx, n_db):
    return np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, int(q_idx)])).random(n_db)


def descriptive(qrows, rows_of, labs, has, path):
    """Ligand ROC-AUC and RE@1% per query and method -> path. rows_of: {method: fn(q_idx) -> (values, 'dist' | 'score',
    tie key) or None for a failure}. Queries need a fingerprint and a positive and a negative among the DB sites with a
    fingerprint."""
    per = []
    fail = {m: 0 for m in rows_of}
    excl = {"no_query_fingerprint": 0, "no_positive": 0, "no_negative": 0}
    for r in qrows:
        lab_all = labs.get(r["ligand"])
        if lab_all is None:
            excl["no_query_fingerprint"] += 1
            continue
        npos_all = int(lab_all[has].sum())
        if npos_all == 0 or npos_all == int(has.sum()):
            excl["no_positive" if npos_all == 0 else "no_negative"] += 1
            continue
        rec = dict(key=r["key"], entry=r["entry"], cluster=r["cluster"], n_pos=npos_all)
        for m, fn in rows_of.items():
            got = fn(r["q_idx"])
            if got is None:            # e.g. DeeplyTough without a descriptor: a failure, not 'no_positive'
                fail[m] += 1
                continue
            v, kind, tie = got
            ok = has & np.isfinite(v)
            lab = lab_all[ok]
            if lab.sum() == 0 or lab.all():
                continue
            score = -v[ok] if kind == "dist" else v[ok]
            rec[f"{m}__auc"] = mw_auc(score, lab)
            rec[f"{m}__re1"] = enrich(score, lab, tie=None if tie is None else tie[ok])[0] \
                if lab.sum() >= RE_MIN_POS else float("nan")
        per.append(rec)
    cols = ["key", "entry", "cluster", "n_pos"] + [f"{m}__{x}" for m in rows_of for x in ("auc", "re1")]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for p in per:
            f.write(",".join(str(p.get(c, "")) for c in cols) + "\n")
    print(f"  ligand metrics: {len(per)} queries, failures {fail}, excluded {excl}")


def cmd_ligand(a):
    R = a.run
    qrows = [dict(r, q_idx=int(r["q_idx"])) for r in read_tsv(os.path.join(R, "queries.tsv"))]
    db_keys = [site_key(p) for p in open(a.db_paths).read().split()]
    n_db = len(db_keys)
    codes = read_tsv(os.path.join(REFERENCE_DIR, "db_ligand_codes.tsv"))
    if [int(r["idx"]) for r in codes] != list(range(n_db)):
        die("db_ligand_codes.tsv is not in the database order")
    has, labs = ligand_labels([r["ligand"] for r in qrows], [r["code"] for r in codes])
    D = {m: np.load(os.path.join(R, f"model_{m}", "dist.npy"), mmap_mode="r") for m in MODELS}
    Ddt = np.load(os.path.join(R, "dist_deeplytough.npy"), mmap_mode="r")
    os.makedirs(a.out_dir, exist_ok=True)

    # ligand ROC-AUC and RE@1% of every method
    rows_of = {}
    for m in MODELS:
        rows_of[f"{m}_raw"] = (lambda i, _D=D[m]: (np.asarray(_D[i], np.float64), "dist", None))

    def _dt(i):
        v = np.asarray(Ddt[i], np.float64)
        if not np.isfinite(v).any():
            return None                 # no descriptor: failure
        return v, "dist", None
    rows_of["deeplytough_raw"] = _dt
    primary = {r["site_id"] + ".pdb": r["primary_chain"] for r in read_tsv(a.manifest) if r.get("featurised") == "1"}
    smap = {r["chain_id"]: r["seq_id"] for r in read_tsv(a.seq_map)}
    chains = [f"{r['key'].split('/')[0]}_{primary[r['key']]}" for r in qrows]
    H = Homology([r["key"] for r in qrows], db_keys, q_chains=[smap.get(c, c) for c in chains], m8=a.seqid_m8)
    for name, fn in (("random", lambda i: random_row(i, n_db)), ("seqid", H.identity)):
        rows_of[f"desc_{name}"] = (lambda i, _f=fn: (_f(i), "score", tie_key(i, n_db)))
    descriptive(qrows, rows_of, labs, has, os.path.join(a.out_dir, "descriptive_per_query.csv"))

    # sites below 30 % identity: ligand ROC-AUC of both models
    sel = [r for r in qrows if r["stratum"] == "lt30"]
    for m in MODELS:
        per = []
        for r in sel:
            lab_all = labs.get(r["ligand"])
            if lab_all is None:
                continue
            dd = np.asarray(D[m][r["q_idx"]], dtype=np.float64)
            ok = has & np.isfinite(dd)
            lab = lab_all[ok]
            npos = int(lab.sum())
            if npos == 0 or npos == len(lab):
                continue
            per.append(dict(key=r["key"], entry=r["entry"], cluster=r["cluster"], n_pos=npos,
                            auc_raw=mw_auc(-dd[ok], lab)))
        with open(os.path.join(a.out_dir, f"below30_{m}_per_query.csv"), "w") as f:
            f.write("key,entry,cluster,n_pos,auc_raw\n")
            for p in per:
                f.write(f"{p['key']},{p['entry']},{p['cluster']},{p['n_pos']},{p['auc_raw']}\n")


# ---------------------------------------------------------------- mcc
def cmd_mcc(a):
    R = a.run
    qs = read_tsv(os.path.join(R, "queries.tsv"))
    LD = os.path.join(R, "model_main_s0")
    qp = open(os.path.join(LD, "q_paths.txt")).read().split()
    dbk = [site_key(p) for p in open(a.db_paths).read().split()]
    status = {}
    lab = pair_labels(a.pairs, status)
    thr = {m: float(json.load(open(os.path.join(a.thresholds, f)))["mcc_threshold"]) for m, f in THRESHOLD_FILES.items()}
    mats = {"lenseek_main_s0": np.load(os.path.join(LD, "dist.npy"), mmap_mode="r"),
            "deeplytough": np.load(os.path.join(R, "dist_deeplytough.npy"), mmap_mode="r")}
    rows, fail_q = {m: {} for m in mats}, {m: [] for m in mats}    # top-k (label, d) per method and query
    for q in qs:
        i = int(q["q_idx"])
        assert site_key(qp[i]) == q["key"], (qp[i], q["key"])
        for m, M in mats.items():
            v = np.asarray(M[i], dtype=np.float64)
            idx = np.nonzero(np.isfinite(v))[0]  # distance asc over finite values, ties by DB index
            if len(idx) == 0:
                fail_q[m].append(q["key"])      # DeeplyTough without a descriptor
                continue
            o = idx[np.lexsort((idx, v[idx]))][:max(KS)]
            d = v[o]
            out = []
            for j, dj in zip(o, d):
                y = lab.get((q["key"], dbk[j]))
                if y is None:
                    sys.exit(f'{m}: top-{max(KS)} pair ({q["key"]}, {dbk[j]}) has no row in {a.pairs}')
                out.append((y, dj))
            rows[m][q["key"]] = out
    clus = {q["key"]: q["cluster"] for q in qs}
    cl, W = boot_weights(clus.values(), N_BOOT)
    ci = {c: i for i, c in enumerate(cl)}

    def cell(y, d, t):          # column of TP, FP, FN, TN
        return (0 if y else 1) if d <= t else (2 if y else 3)

    def table(m):
        out = []
        for k in KS:
            C = np.zeros((len(cl), 4))
            for qk, lst in rows[m].items():
                for y, d in lst[:k]:
                    if math.isfinite(y):
                        C[ci[clus[qk]], cell(y, d, thr[m])] += 1
            tot = C.sum(0)
            b = mcc(*(W @ C).T)
            out.append(dict(k=k, TP=int(tot[0]), FP=int(tot[1]), FN=int(tot[2]), TN=int(tot[3]), MCC=float(mcc(*tot)),
                            MCC_ci_lo=float(np.percentile(b, 2.5)), MCC_ci_hi=float(np.percentile(b, 97.5))))
        return out

    def compare(x):
        """LEN-Seek - x on top-1 MCC and P@1, queries with both top-1 labels."""
        L, X = rows["lenseek_main_s0"], rows[x]
        cL, cX = np.zeros((len(cl), 4)), np.zeros((len(cl), 4))
        pL, pX, n = np.zeros(len(cl)), np.zeros(len(cl)), np.zeros(len(cl))
        for qk in L:
            if qk not in X:
                continue
            (yl, dl), (yx, dx) = L[qk][0], X[qk][0]
            if not (math.isfinite(yl) and math.isfinite(yx)):
                continue
            c = ci[clus[qk]]
            cL[c, cell(yl, dl, thr["lenseek_main_s0"])] += 1
            cX[c, cell(yx, dx, thr[x])] += 1
            pL[c] += yl
            pX[c] += yx
            n[c] += 1
        res = dict(n_queries=int(n.sum()), mcc_lenseek=float(mcc(*cL.sum(0))), mcc_other=float(mcc(*cX.sum(0))),
                   p1_lenseek=float(pL.sum() / n.sum()), p1_other=float(pX.sum() / n.sum()))
        dm = mcc(*(W @ cL).T) - mcc(*(W @ cX).T)
        dp = (W @ pL - W @ pX) / (W @ n)
        for name, dv, point in (("mcc", dm, res["mcc_lenseek"] - res["mcc_other"]),
                                ("p1", dp, res["p1_lenseek"] - res["p1_other"])):
            res[f"{name}_diff"] = float(point)
            res[f"{name}_ci_lo"], res[f"{name}_ci_hi"] = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
        up = res["mcc_ci_lo"] > 0 and res["p1_ci_lo"] > 0
        down = res["mcc_ci_hi"] < 0 or res["p1_ci_hi"] < 0
        res["detection"] = f"{x} > LEN-Seek" if down else ("LEN-Seek > " + x if up else "not detected")
        return res

    res = dict(pair_status=status, thresholds=thr, n_queries=len(qs), n_clusters=len(cl), n_boot=N_BOOT,
               failed_queries={m: v for m, v in fail_q.items() if v},
               rule="LEN-Seek favoured only when both top-1 MCC and P@1 differences are detected; the other method "
                    "favoured when either is detected",
               tables={m: dict(threshold=thr[m], rows=table(m)) for m in mats},
               differences={x: compare(x) for x in ("deeplytough",)})
    strata = []                 # top-1 by identity stratum; unscored sites keep their clusters in the resampling
    for m in mats:
        for sname in ("all", "lt30", "30to80"):
            allq = [q for q in qs if sname in ("all", q["stratum"])]
            sel = [q for q in allq if q["key"] in rows[m] and math.isfinite(rows[m][q["key"]][0][0])]
            scl, SW = boot_weights([q["cluster"] for q in allq], N_BOOT)
            sci = {c: i for i, c in enumerate(scl)}
            C = np.zeros((len(scl), 4))
            for q in sel:
                y, d = rows[m][q["key"]][0]
                C[sci[q["cluster"]], cell(y, d, thr[m])] += 1
            tot, B = C.sum(0), SW @ C
            bm, bp = mcc(*B.T), (B[:, 0] + B[:, 2]) / B.sum(1)
            strata.append(dict(method=m, stratum=sname, n_queries=len(sel), n_clusters=len({q["cluster"] for q in sel}),
                               mcc=float(mcc(*tot)),
                               mcc_ci_lo=np.percentile(bm, 2.5), mcc_ci_hi=np.percentile(bm, 97.5),
                               p1=(tot[0] + tot[2]) / tot.sum(), p1_ci_lo=np.percentile(bp, 2.5),
                               p1_ci_hi=np.percentile(bp, 97.5), TP=int(tot[0]), FP=int(tot[1]), FN=int(tot[2]),
                               TN=int(tot[3])))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    write_csv(os.path.join(os.path.dirname(os.path.abspath(a.out)), "mcc_strata.csv"), strata)
    for m, t in res["tables"].items():
        r1 = t["rows"][0]
        print(f"{m} (t = {t['threshold']:g}): top-1 TP/FP/FN/TN {r1['TP']}/{r1['FP']}/{r1['FN']}/{r1['TN']} "
              f"MCC {r1['MCC']:.4f} [{r1['MCC_ci_lo']:.4f}, {r1['MCC_ci_hi']:.4f}]")
    for x, d in res["differences"].items():
        print(f"LEN-Seek - {x}: {d['n_queries']} queries; MCC {d['mcc_diff']:+.4f} [{d['mcc_ci_lo']:+.4f}, "
              f"{d['mcc_ci_hi']:+.4f}]; P@1 {d['p1_diff']:+.4f} [{d['p1_ci_lo']:+.4f}, {d['p1_ci_hi']:+.4f}] -> "
              f"{d['detection']}")


# ---------------------------------------------------------------- table
PANEL_A = [("LEN-Seek", "main_s0_raw"), ("Model trained without homologous entries", "strict30_s0_raw"),
           ("DeeplyTough", "deeplytough_raw"), ("Sequence-identity baseline", "desc_seqid"),
           ("Random ranking", "desc_random")]


def cmd_table(a):
    rd = lambda name: list(csv.DictReader(open(os.path.join(a.ligand_dir, name))))  # noqa: E731
    per = rd("descriptive_per_query.csv")
    weight = {r["site_id"] + ".pdb": float(r["sampling_weight"]) for r in read_tsv(a.manifest) if r["sampling_weight"]}

    def summ(rows, col, value=None):
        """Entry-weighted mean and cluster-bootstrap 95 % interval of a per-query value (missing / NaN left out)."""
        value = value or (lambda r: float(r[col]) if r[col] not in ("", None) else float("nan"))
        vs = [v for v in ((value(r), r["entry"], r["cluster"], r["key"]) for r in rows) if np.isfinite(v[0])]
        df = pd.DataFrame({"v": [v[0] for v in vs], "protein": [v[1] for v in vs], "cluster": [v[2] for v in vs]})
        cl, W = boot_weights(df.cluster, n_boot=N_BOOT, seed=0)
        s, _ = summary(df, "v", W, cl, entry=True)
        return dict(n=len(vs), mean=s["mean"], lo=s["lo"], hi=s["hi"], vals=vs)

    A = {}
    for label, m in PANEL_A:
        au, r1 = summ(per, f"{m}__auc"), summ(per, f"{m}__re1")
        w = np.array([weight[v[3]] for v in au["vals"]])
        A[m] = dict(method=label, sites=au["n"], auc=au["mean"], auc_lo=au["lo"], auc_hi=au["hi"], re1_sites=r1["n"],
                    re1=r1["mean"], re1_lo=r1["lo"], re1_hi=r1["hi"],
                    auc_release_weighted=float((np.array([v[0] for v in au["vals"]]) * w).sum() / w.sum()))
    B = []
    for label, m in (("LEN-Seek", "main_s0"), ("Model trained without homologous entries", "strict30_s0")):
        r = summ(rd(f"below30_{m}_per_query.csv"), "auc_raw")
        B.append(dict(model=label, sites=r["n"], auc=r["mean"], auc_lo=r["lo"], auc_hi=r["hi"]))
    mj = json.load(open(a.mcc))
    tab = {m: {r["k"]: r for r in mj["tables"][m]["rows"]} for m in THRESHOLD_FILES}
    C = [dict(k=k, **{f"{m}{s}": tab[m][k][f"MCC{s}"] for m in THRESHOLD_FILES for s in ("", "_ci_lo", "_ci_hi")})
         for k in KS]
    dif = mj["differences"]
    r1 = tab["lenseek_main_s0"][1]                  # P@1 of LEN-Seek: queries whose top-1 result is positive
    p1_lenseek = (r1["TP"] + r1["FN"]) / (r1["TP"] + r1["FP"] + r1["FN"] + r1["TN"])
    T1 = [dict(method=A[m]["method"], top1_mcc=tab[k][1]["MCC"] if k else "", p1=p1, ligand_sites=A[m]["sites"],
               ligand_auc=A[m]["auc"], re1_sites=A[m]["re1_sites"], re1=A[m]["re1"])
          for m, k, p1 in (("main_s0_raw", "lenseek_main_s0", p1_lenseek),
                           ("deeplytough_raw", "deeplytough", dif["deeplytough"]["p1_other"]),
                           ("desc_seqid", None, ""), ("desc_random", None, ""))]
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("table_S14A", list(A.values())), ("table_S14B", B), ("table_S14C", C),
                       ("table1B_holdout", T1)):
        write_csv(os.path.join(a.out_dir, name + ".csv"), rows)
        print(f"== {name}")
        for r in rows:
            print("  " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = ENV("LENSEEK_HOLDOUT_RUN") or os.path.join(OUT_ROOT, "holdout", "run")
    thresholds = ENV("LENSEEK_HOLDOUT_THRESHOLDS") or os.path.join(OUT, "thresholds")
    manifest = ENV("LENSEEK_HOLDOUT_MANIFEST") or os.path.join(OUT_ROOT, "holdout", "manifest.tsv")
    db_paths = os.path.join(ENV("LENSEEK_EVAL_DIR") or ".", "db_paths.txt")
    p = sub.add_parser("thresholds", help="thresholds on the evaluation data")
    p.add_argument("--pairs", default=ENV("LENSEEK_PAIRS"), help="scored top-100 pairs of LEN-Seek")
    p.add_argument("--eval-dir", default=ENV("LENSEEK_EVAL_DIR"))
    p.add_argument("--scan", default=ENV("LENSEEK_SCAN_DIR"), help="full scan: all_status.npy, all_raw.npy")
    p.add_argument("--dt-dist", default=os.path.join(ENV("LENSEEK_DT_DIR") or ".", "dist.npy"))
    p.add_argument("--exclude", default=PRIMARY_EXCLUDE_FILE)
    p.add_argument("--out-dir", default=os.path.join(OUT, "thresholds"))
    p = sub.add_parser("ligand", help="per-query ligand ROC-AUC and RE@1%% of the scoring run")
    p.add_argument("--run", default=run, help="scoring run (default $LENSEEK_HOLDOUT_RUN)")
    p.add_argument("--db-paths", default=db_paths,
                   help="db_paths.txt of the LEN-Seek database encoding (default $LENSEEK_EVAL_DIR/db_paths.txt)")
    p.add_argument("--manifest", default=manifest, help="manifest.tsv of the sites (primary chains)")
    p.add_argument("--seqid-m8", default=os.path.join(OUT_ROOT, "holdout", "mm", "holdout_vs_train_uncapped.m8"),
                   help="MMseqs2 search of the sequences of the sites against the training chains")
    p.add_argument("--seq-map", default=os.path.join(OUT_ROOT, "holdout", "seqs", "seq_map.tsv"),
                   help="chain -> sequence id of the m8 queries")
    p.add_argument("--out-dir", default=OUT)
    p = sub.add_parser("mcc", help="top-k MCC, P@1, paired differences")
    p.add_argument("--run", default=run, help="scoring run (default $LENSEEK_HOLDOUT_RUN)")
    p.add_argument("--pairs", default=ENV("LENSEEK_HOLDOUT_LABELS") or
                   os.path.join(OUT_ROOT, "holdout", "labels", "tables", "pairs_union_for_labels.scored.tsv"),
                   help="ProBiS labels of the union of the top-100 lists (default $LENSEEK_HOLDOUT_LABELS)")
    p.add_argument("--db-paths", default=db_paths,
                   help="db_paths.txt of the LEN-Seek database encoding (default $LENSEEK_EVAL_DIR/db_paths.txt)")
    p.add_argument("--thresholds", default=thresholds, help="default $LENSEEK_HOLDOUT_THRESHOLDS")
    p.add_argument("--out", default=os.path.join(OUT, "mcc_holdout.json"))
    p = sub.add_parser("table", help="Supplementary Table S11 and the top-1 values of the later PDB entries")
    p.add_argument("--ligand-dir", default=OUT, help="per-query outputs of ligand")
    p.add_argument("--mcc", default=os.path.join(OUT, "mcc_holdout.json"))
    p.add_argument("--manifest", default=manifest,
                   help="manifest.tsv of the sites with the sampling weights (default $LENSEEK_HOLDOUT_MANIFEST)")
    p.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    dict(thresholds=cmd_thresholds, ligand=cmd_ligand, mcc=cmd_mcc, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
