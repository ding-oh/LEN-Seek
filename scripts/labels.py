#!/usr/bin/env python3
"""ProBiS and G-LoSA labels of query-database site pairs.

Computes the structural labels behind every ProBiS and G-LoSA result: the database-wide ProBiS and G-LoSA scans of the
evaluation queries against every database site (Table 1, Figures 4 and 5, Supplementary Sections S2 and S4 and the
tables built on them), the label tables of each method's top-100 list, the failed comparisons of Supplementary Section
S2, and the unrelated pairs whose positive labels the caption of Supplementary Table S8 counts
(analysis/table1_concordance.py unrelated).

Labels. A pair is ProBiS-positive when Z >= 2.0, Z = (raw alignment score - 1.958) / 2.21468 rounded to two decimals,
and G-LoSA-positive when its GA-score is >= 0.59. A pair without a ProBiS alignment is negative; failed comparisons
(missing input, timeout, tool failure) are kept apart and never counted as negatives.

Protocol. ProBiS 2.4.7 compares the protein.pdb files of the two BsitePDB entries restricted to the binding-site
residues: `probis -compare -noprune -f1 <q> -c1 <chains> -motif1 "[:A and (48,50) or ...]" -f2 <t> -c2 ... -motif2 ...
-nosql p.nosql`, a fresh working directory per pair and no -super; the raw score is that of cluster 0 of the .nosql row
(what `probis -results -z_score -1000` reports). In the scan, a comparison that ends in a timeout (300 s) or a failure
is run once more with a 1,800 s limit; listed pairs (score) have 900 s. G-LoSA 2.2: `AssignChemicalFeatures` on each
pocket file, then `glosa -s1 <query> -s1cf ... -s2 <target> -s2cf ... -o 0`, score "GA-score" (query = s1). Input
handling, on staged copies only: chains with negative residue numbers are shifted by a constant in the ProBiS copy of
protein.pdb and insertion-code residues are left out of the motif; inner TER lines are removed from the G-LoSA copies
and the residues renumbered 1..N in file order; coordinates written one column left of the PDB standard are moved to
the standard columns (normalize_line).

Subcommands (outputs below OUT = $LENSEEK_OUT/labels, default ./out/labels, where no path is given):
  build-glosa TARBALL       compile G-LoSA 2.2, AssignChemicalFeatures and BatchCF.java (next to this file) into
                            $GLOSA_DIR
  sites EVAL_DIR            OUT/sites/all_query_sites.txt, all_db_sites.txt (row order of q_paths.txt / db_paths.txt)
  pairs EVAL_DIR OUT_TSV    top-k pair list of one model: the k smallest distances, ties to the lower database index
  score                     ProBiS and G-LoSA scores of listed pairs into an append-only store (OUT/store/part_*.tsv)
  scan-probis, scan-glosa   one database chunk of a database-wide scan (array task, 115 chunks)
  assemble, assemble-glosa  chunk files -> OUT/scan/{all_status.npy, all_raw.npy}, OUT/scan_glosa/{all_status.npy,
                            all_ga.npy}
  tables LIST ...           label tables <list>.scored.tsv: ProBiS from the scan, G-LoSA from the score store
  census                    per site: negative residue numbers, residue merges seen by G-LoSA, ligand code and size
  smiles                    SMILES of the census ligand codes from the RCSB REST API (network)
  null M8                   unrelated pairs: no chain pair of the two entries >= 30 % identical (MMseqs2 hits M8),
                            different ligands with Morgan Dice < 0.46, both ligands >= 6 heavy atoms, all files present
Usage (examples/labels.sh runs the array steps under SLURM):
  python scripts/labels.py sites $LENSEEK_EVAL_DIR && python scripts/labels.py census
  python scripts/labels.py scan-probis --chunk K && python scripts/labels.py assemble          # K = 0..114
  python scripts/labels.py scan-glosa --chunk K && python scripts/labels.py assemble-glosa
  python scripts/labels.py pairs $LENSEEK_EVAL_DIR out/labels/pairs/main_s0_top100.tsv
  python scripts/labels.py score --tools glosa --pairs out/labels/pairs/*_top100.tsv
  python scripts/labels.py tables out/labels/pairs/*_top100.tsv
  python scripts/labels.py smiles && python scripts/labels.py null $LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8
Tools and inputs: PROBIS_BIN (ProBiS 2.4.7), GLOSA_DIR (from build-glosa; GLOSA_BIN overrides the glosa binary),
JAVA_BIN (the published runs used Eclipse Temurin 17.0.21+9), LENSEEK_TMP (node-local scratch; its path must not
contain 'pdb'). LENSEEK_QUERY_ROOT and LENSEEK_BSDB_ROOT hold the structures in the BsitePDB layout
<root>/<pdbid>/{protein.pdb, ligand_pocket_<n>_<chains>.pdb, ligand_<n>_<chains>.pdb}. null uses RDKit Morgan
fingerprints (radius 2, 2,048 bits) of the SMILES written by `smiles` (the CCD table
$LENSEEK_REFERENCE_DIR/ccd_smiles.tsv holds other SMILES strings of the same components and gives a different set); a
ligand code whose SMILES RDKit cannot parse counts as without SMILES, so the set of unrelated pairs depends slightly on
the RDKit version (organometallic SMILES). ProBiS limits its clique search by CPU time, so a scan and the pair scores
belong on one CPU type.
Scan status codes: 0 pending, 1 alignment (G-LoSA: GA), 2 no alignment (G-LoSA: GA -nan), 3 missing input, 4 timeout,
5 tool failure, 6 wrapper error (G-LoSA: 6 no chemical features, 7 wrapper error).
"""
import argparse
import csv
import fcntl
import glob
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import (GLOSA, GLOSA_CF_CP, GLOSA_DIR, JAVA, JAVA_OPTS, PROBIS, ST_CF, ST_FAIL, ST_MISSING,  # noqa: E402
                      ST_OK, ST_TIMEOUT, cf_of, cpu_model, glosa_compare, pocket_chains, pocket_protein,
                      pocket_residues, probis_motif, read_list, run, scratch_base, stage_name, stage_protein,
                      strip_inner_ter, under)

OUT = os.path.join(os.environ.get("LENSEEK_OUT") or "out", "labels")
PAIR_TIMEOUT, CKPT_MIN = 900, 15   # seconds per listed pair (score); minutes between scan checkpoints
ST_PENDING, ST_NOALN, ST_ERR = 0, 2, 6   # ProBiS scan codes besides the common ones (no alignment, wrapper error)
ST_GERR = 7                              # G-LoSA scan: wrapper error
SCAN_STATUS = {1: "ok", 2: "nosig_empty", 3: "missing_input", 4: "timeout", 5: "fail_compare", 6: "fail_wrapper",
               0: "pending"}
GST_NAMES = {1: "ok", 2: "nan_score", 3: "missing_input", 4: "timeout", 5: "fail_glosa", 6: "fail_cf",
             7: "fail_wrapper", 0: "pending"}
G = {}   # per-worker globals of the pools
NAN = np.nan


# ------------------------------------------------------------------------------------------------- inputs and tools
def root(var):
    """Absolute path of the directory named by environment variable `var` (exit with a clear message otherwise)."""
    v = os.environ.get(var)
    if not v:
        sys.exit(f"[error] environment variable {var} is not set")
    if not os.path.isdir(v):
        sys.exit(f"[error] {var}={v!r} is not a directory")
    return os.path.abspath(v)


def _exe(cmd):
    return os.path.isfile(cmd) and os.access(cmd, os.X_OK) if os.sep in cmd else shutil.which(cmd) is not None


def check_tools(probis=False, glosa=False, batch_cf=False):
    """Exit with a clear message if a required external tool is not configured."""
    err = []
    if probis and not _exe(PROBIS):
        err.append(f"ProBiS executable not found: PROBIS_BIN={PROBIS!r} (set PROBIS_BIN or put `probis` on PATH)")
    if glosa or batch_cf:
        if not GLOSA_DIR or not os.path.isdir(GLOSA_DIR):
            err.append(f"G-LoSA directory not found: GLOSA_DIR={GLOSA_DIR!r} (set GLOSA_DIR, see build-glosa)")
        else:
            if not _exe(GLOSA):
                err.append(f"glosa executable not found: {GLOSA!r} (set GLOSA_BIN or build GLOSA_DIR/glosa)")
            if not os.path.isfile(os.path.join(GLOSA_CF_CP, "AssignChemicalFeatures.class")):
                err.append(f"AssignChemicalFeatures.class not found in {GLOSA_CF_CP!r} (GLOSA_DIR/cf)")
            if batch_cf and not os.path.isfile(os.path.join(GLOSA_DIR, "cfbatch", "BatchCF.class")):
                err.append(f"BatchCF.class not found in {os.path.join(GLOSA_DIR, 'cfbatch')!r} (GLOSA_DIR/cfbatch)")
        if not _exe(JAVA):
            err.append(f"java not found: JAVA_BIN={JAVA!r} (set JAVA_BIN or put `java` on PATH)")
    if err:
        sys.exit("[error] " + "\n[error] ".join(err))


# ------------------------------------------------------------------------------------------------- ProBiS comparison
def stage_site(path, tag, dest_dir, cache_dir):
    """Staged protein.pdb (chain shift for negative numbers, standard columns; cached per protein file and shift, so an
    entry with several sites is read once) and the motif of one site: {file, name, chains, motif}, None when a file
    is missing."""
    prot = pocket_protein(path)            # protein.pdb of the same BsitePDB entry
    if not (os.path.isfile(path) and os.path.isfile(prot)):
        return None
    motif, shifts, _ = probis_motif(path, prot)
    chains, _ = pocket_chains(path)
    name = stage_name(path, tag)                  # <pdbid>_<tag>_<stem>.pdb, 'srf' masked
    key = hashlib.md5(f"{os.path.abspath(prot)}|{sorted(shifts.items())}|norm".encode()).hexdigest()
    cached = os.path.join(cache_dir, key + ".pdb")
    if not os.path.exists(cached):
        tmp = f"{cached}.{os.getpid()}.tmp"
        stage_protein(prot, tmp, shifts)
        os.replace(tmp, cached)
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, name)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(cached, dst)
    return {"file": dst, "name": name, "chains": chains, "motif": motif}


def nosql_clusters(nosql):
    """[(cluster_id, raw_score_text, raw_score)] of the last row of a pair .nosql; [] without a cluster, None without
    a row (the comparison did not run)."""
    if not os.path.isfile(nosql):
        return None
    with open(nosql) as f:
        rows = [ln.rstrip("\n") for ln in f if ln.strip("\t\n ")]
    if not rows:
        return None
    fields = rows[-1].split("\t")
    if len(fields) < 4 or not fields[2].strip():
        return []
    body = fields[2].strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    out = []
    for c in body.split("),("):
        p = c.split(",")
        out.append((int(p[0]), p[8], float(p[8])))
    return out


def probis_compare(q, t, w, timeout):
    """One ProBiS comparison of two staged sites (stage_site) in the working directory w: (status, raw, nclus, sec),
    raw = alignment score of cluster 0 (float64, status 1)."""
    if q is None or t is None:
        return ST_MISSING, np.nan, 0, 0.0
    qlink = os.path.join(w, q["name"])
    tname = t["name"] if t["name"] != q["name"] else "x" + t["name"]
    tlink = os.path.join(w, tname)
    nos = os.path.join(w, "p.nosql")
    try:
        for link, f in ((qlink, q["file"]), (tlink, t["file"])):
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(f, link)
        if os.path.exists(nos):
            os.unlink(nos)
        cmd = [PROBIS, "-compare", "-noprune", "-f1", q["name"], "-c1", q["chains"], "-motif1", q["motif"],
               "-f2", tname, "-c2", t["chains"], "-motif2", t["motif"], "-nosql", "p.nosql"]
        t0 = time.perf_counter()
        rc, out, err, _, to = run(cmd, w, timeout)
        sec = time.perf_counter() - t0
        if to:
            return ST_TIMEOUT, np.nan, 0, sec
        cl = nosql_clusters(nos)
        if rc != 0 or cl is None:
            return ST_FAIL, np.nan, 0, sec
        if not cl:
            return ST_NOALN, np.nan, 0, sec
        c0 = min(cl, key=lambda x: x[0])
        return ST_OK, c0[2], min(len(cl), 255), sec
    except Exception:  # noqa: BLE001
        return ST_ERR, np.nan, 0, 0.0
    finally:
        for link in (qlink, tlink):
            if os.path.lexists(link):
                os.unlink(link)


# ------------------------------------------------------------------------------------------------- G-LoSA comparison
def stage_pocket(src, dest_dir):
    """Staged copy of one pocket (strip_inner_ter) in dest_dir, named by the md5 of its path; None if the file is
    missing."""
    if not os.path.isfile(src):
        return None
    dst = os.path.join(dest_dir, hashlib.md5(os.path.abspath(src).encode()).hexdigest() + ".pdb")
    if not os.path.exists(dst):
        tmp = f"{dst}.{os.getpid()}.tmp"
        strip_inner_ter(src, tmp)
        os.replace(tmp, dst)
    return dst


def features(files, workdir):
    """Chemical features (<stem>-cf.pdb) of the staged files that lack them, in one JVM (BatchCF)."""
    need = [f for f in files
            if f and not (os.path.exists(f[:-4] + "-cf.pdb") and os.path.getsize(f[:-4] + "-cf.pdb") > 0)]
    if not need:
        return 0.0
    lst = os.path.join(workdir, f"cf_list_{os.getpid()}.txt")
    with open(lst, "w") as f:
        f.write("\n".join(need) + "\n")
    cp = GLOSA_CF_CP + ":" + os.path.join(GLOSA_DIR, "cfbatch")
    t0 = time.time()
    rc, out, err, _, to = run([JAVA, *JAVA_OPTS, "-cp", cp, "BatchCF", lst], workdir, 7200)
    if rc != 0 or to:
        print(f"BatchCF rc={rc} timeout={to}: {err.strip()[-300:]}", flush=True)
    return time.time() - t0


# ------------------------------------------------------------------------------------------------- site and pair lists
def cmd_sites(a):
    qroot, dbroot = root("LENSEEK_QUERY_ROOT"), root("LENSEEK_BSDB_ROOT")
    q = [under(p, qroot) for p in read_list(os.path.join(a.eval_dir, "q_paths.txt"))]
    db = [under(p, dbroot) for p in read_list(os.path.join(a.eval_dir, "db_paths.txt"))]
    os.makedirs(a.out, exist_ok=True)
    for name, lst in (("all_query_sites.txt", q), ("all_db_sites.txt", db), ("all_sites.txt", q + db)):
        with open(os.path.join(a.out, name), "w") as f:
            f.write("\n".join(lst) + "\n")
    missing = sum(not os.path.isfile(p) for p in q + db)
    print(f"{len(q)} query sites, {len(db)} database sites -> {a.out}; pocket files not found: {missing}")


def cmd_pairs(a):
    """Columns query_idx, query_path, rank, db_idx, target_path, dist; non-finite distances rank last."""
    qroot, dbroot = root("LENSEEK_QUERY_ROOT"), root("LENSEEK_BSDB_ROOT")
    q = [ln.strip() for ln in open(os.path.join(a.eval_dir, "q_paths.txt")) if ln.strip()]
    db = [ln.strip() for ln in open(os.path.join(a.eval_dir, "db_paths.txt")) if ln.strip()]
    d = np.load(os.path.join(a.eval_dir, "dist.npy"), mmap_mode="r")
    assert d.shape == (len(q), len(db)), (d.shape, len(q), len(db))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_tsv)), exist_ok=True)
    n_nonfinite = 0
    with open(a.out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query_idx", "query_path", "rank", "db_idx", "target_path", "dist"])
        for i in range(len(q)):
            row = np.asarray(d[i], dtype=np.float64)
            n_nonfinite += int((~np.isfinite(row)).sum())
            row = np.where(np.isfinite(row), row, np.inf)
            order = np.argsort(row, kind="stable")[: a.topk]  # ties -> lower DB index first
            qp = under(q[i], qroot)
            for r, j in enumerate(order, 1):
                w.writerow([i, qp, r, int(j), under(db[j], dbroot), f"{row[j]:.6f}"])
    print(f"{a.out_tsv}: {len(q)} queries x top{a.topk}; non-finite distances: {n_nonfinite}")


# ------------------------------------------------------------------------------------------------- pair scores
FINAL = {"ok", "nosig_empty", "missing_input", "nan_score"}
STORE_COLS = ["query_path", "target_path", "tool", "score1", "score2", "status", "sec", "extra",
              "host", "cpu", "ts", "toolcfg"]
CPU = cpu_model()
HOST = socket.gethostname()
TOOLS = {"probis": "probis2.4.7 -compare -noprune; protein.pdb + site motif, negative chains shifted, standard columns",
         "glosa": "glosa2.2 -o 0; pocket files without inner TER, renumbered, standard columns; q=s1; jre17.0.21"}


def read_pairs(files):
    keys = set()
    for fn in files:
        with open(fn) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                keys.add((r["query_path"], r["target_path"]))
    return sorted(keys)


def load_store(store):
    done = set()
    for fn in glob.glob(os.path.join(store, "part_*.tsv")):
        with open(fn) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("status") in FINAL:
                    done.add((r["query_path"], r["target_path"], r["tool"]))
    return done


def fmt(x):
    return "" if x is None else str(x)


def probis_pair(q, t):
    """ProBiS score of one listed pair: both sites staged into a fresh directory (protein copies cached per task),
    then probis_compare. (status, raw, nclus, sec)."""
    w = tempfile.mkdtemp(prefix="pgP_", dir=scratch_base())
    try:
        qs = stage_site(q, "q", os.path.join(w, "q"), G["prot"])
        ts = stage_site(t, "t", os.path.join(w, "t"), G["prot"])
        return probis_compare(qs, ts, w, PAIR_TIMEOUT)
    except Exception:  # noqa: BLE001
        return ST_ERR, np.nan, 0, 0.0
    finally:
        shutil.rmtree(w, ignore_errors=True)


def glosa_pair(q, t):
    """G-LoSA score of one listed pair: both pockets staged into a fresh directory, chemical features by
    AssignChemicalFeatures per file (cached per pocket for the task), then glosa_compare. (status, ga, sec)."""
    if not (os.path.isfile(q) and os.path.isfile(t)):
        return ST_MISSING, np.nan, None
    w = tempfile.mkdtemp(prefix="pgG_", dir=scratch_base())
    try:
        for src, dst in ((q, "s1.pdb"), (t, "s2.pdb")):
            strip_inner_ter(src, os.path.join(w, dst))
        t0 = time.time()
        for s, src in (("s1", q), ("s2", t)):
            cf = os.path.join(w, f"{s}-cf.pdb")
            cached = os.path.join(G["cf"], hashlib.md5(os.path.abspath(src).encode()).hexdigest() + "-cf.txt")
            if os.path.isfile(cached) and os.path.getsize(cached) > 0:
                shutil.copyfile(cached, cf)
                continue
            rc, out, err, _, to = run([JAVA, *JAVA_OPTS, "-cp", GLOSA_CF_CP, "AssignChemicalFeatures", f"{s}.pdb"],
                                      w, PAIR_TIMEOUT)
            if to:
                return ST_TIMEOUT, np.nan, round(time.time() - t0, 3)
            if rc != 0 or not os.path.isfile(cf) or os.path.getsize(cf) == 0:
                return ST_CF, np.nan, round(time.time() - t0, 3)
            tmp = f"{cached}.{os.getpid()}.tmp"
            shutil.copyfile(cf, tmp)
            os.replace(tmp, cached)
        st, ga, _ = glosa_compare(os.path.join(w, "s1.pdb"), os.path.join(w, "s2.pdb"), w,
                                  max(60, PAIR_TIMEOUT - (time.time() - t0)))
        return st, ga, round(time.time() - t0, 3)
    except Exception:  # noqa: BLE001
        return ST_GERR, np.nan, None
    finally:
        shutil.rmtree(w, ignore_errors=True)


def score_pair(job):
    """Store rows of one pair: probis (score1 Z, score2 alignment score, both rounded to 2 decimals) and glosa
    (score1 GA)."""
    (q, t), tools = job
    out = []
    if "probis" in tools:
        st, raw, ncl, sec = probis_pair(q, t)
        ok = st == ST_OK
        out.append([q, t, "probis", fmt(round((raw - 1.958) / 2.21468, 2) if ok else None),
                    fmt(round(raw, 2) if ok else None), SCAN_STATUS[st], fmt(round(sec, 3)),
                    json.dumps({"nclus": int(ncl)})])
    if "glosa" in tools:
        st, ga, sec = glosa_pair(q, t)
        out.append([q, t, "glosa", fmt(ga if st == ST_OK else None), "", GST_NAMES[st], fmt(sec), "{}"])
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return [row + [HOST, CPU, ts, TOOLS[row[2]]] for row in out]


def score_init(cache):
    """Per-task caches: staged protein copies (each protein.pdb read once) and G-LoSA feature files."""
    G["cf"], G["prot"] = cache, os.path.join(cache, "prot")
    os.makedirs(G["prot"], exist_ok=True)


def cmd_score(a):
    """Queries (sorted) are dealt round-robin to --ntasks tasks; a (pair, tool) already in the store with a final
    status is skipped, failures are run again."""
    tools_all = a.tools.split(",")
    unknown = [t for t in tools_all if t not in TOOLS]
    if unknown:
        sys.exit(f"[error] unknown tool(s) {unknown}; known: {sorted(TOOLS)}")
    check_tools(probis="probis" in tools_all, glosa="glosa" in tools_all)
    os.makedirs(a.store, exist_ok=True)

    allkeys = read_pairs(a.pairs)
    queries = sorted({k[0] for k in allkeys})
    mine = {qp for i, qp in enumerate(queries) if i % a.ntasks == a.task}  # round-robin by query
    keys = [k for k in allkeys if k[0] in mine]
    done = load_store(a.store)
    jobs = []
    for k in keys:
        need = tuple(tl for tl in tools_all if (k[0], k[1], tl) not in done)
        if need:
            jobs.append((k, need))
    jid = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    part = os.path.join(a.store, f"part_{jid}_{a.task:04d}.tsv")
    print(f"task {a.task}/{a.ntasks}: {len(keys)} pairs, {len(jobs)} to run, workers={a.workers}, "
          f"host={HOST}, cpu={CPU}, out={part}", flush=True)
    if not jobs:
        print("TASK_DONE (nothing to do)")
        return
    cache = tempfile.mkdtemp(prefix="pgcf_", dir=scratch_base())
    new = not os.path.exists(part)
    t0 = time.time()
    n = 0
    with open(part, "a", newline="") as f, Pool(a.workers, initializer=score_init, initargs=(cache,)) as pool:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new:
            w.writerow(STORE_COLS)
        for rows in pool.imap_unordered(score_pair, jobs, chunksize=1):
            w.writerows(rows)
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} pairs, {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    print(f"TASK_DONE {n} pairs in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------- database-wide scans
def chunk_of(a):
    """Database indices [K*C, (K+1)*C) of chunk K = --chunk, C = ceil(N_db / nchunks); the site lists."""
    queries, dbs = read_list(a.queries), read_list(a.db)
    C = math.ceil(len(dbs) / a.nchunks)
    return queries, dbs, np.arange(a.chunk * C, min(len(dbs), (a.chunk + 1) * C))


def scan_init(db_sites, timeout, retry_timeout, scratch):
    G["db"], G["timeout"], G["retry_timeout"] = db_sites, timeout, retry_timeout
    G["w"] = tempfile.mkdtemp(prefix="w_", dir=scratch)


def scan_query(arg):
    """All database sites of the chunk against one query; a comparison that ends in a timeout or a failure is run
    once more with the longer limit. Returns (qi, z, raw, status, nclus, sec)."""
    qi, q = arg
    C = len(G["db"])
    z = np.full(C, np.nan, np.float32)
    raw = np.full(C, np.nan, np.float32)
    st = np.zeros(C, np.uint8)
    ncl = np.zeros(C, np.uint8)
    sec = np.zeros(C, np.float32)
    for j, t in enumerate(G["db"]):
        r = probis_compare(q, t, G["w"], G["timeout"])
        if r[0] in (ST_TIMEOUT, ST_FAIL, ST_ERR):
            r = probis_compare(q, t, G["w"], G["retry_timeout"])
        st[j], raw[j], ncl[j], sec[j] = r
        if r[0] == ST_OK:
            z[j] = (r[1] - 1.958) / 2.21468
    return qi, z, raw, st, ncl, sec


def scan_save(path, z, raw, st, ncl, sec, idx, done):
    tmp = path + ".tmp.npz"
    np.savez(tmp, z=z.astype(np.float16), raw=raw, status=st, nclus=ncl, sec=sec.astype(np.float16),
             db_idx=idx, done=done)
    os.replace(tmp, path)


def cmd_scan_probis(a):
    """Chunk K against every query -> OUT/chunk_KKKK.npz (z, raw, status, nclus, sec, db_idx, done), checkpointed
    every CKPT_MIN minutes; a resubmitted task resumes."""
    check_tools(probis=True)
    queries, dbs, idx = chunk_of(a)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"chunk_{a.chunk:04d}.npz")
    Q, Cc = len(queries), len(idx)
    if os.path.exists(out):
        d = np.load(out)
        z, raw, st, ncl = d["z"].astype(np.float32), d["raw"], d["status"], d["nclus"]
        sec, done = d["sec"].astype(np.float32), d["done"]
        assert z.shape == (Q, Cc) and (d["db_idx"] == idx).all()
    else:
        z = np.full((Q, Cc), np.nan, np.float32)
        raw = np.full((Q, Cc), np.nan, np.float32)
        st = np.zeros((Q, Cc), np.uint8)
        ncl = np.zeros((Q, Cc), np.uint8)
        sec = np.zeros((Q, Cc), np.float32)
        done = np.zeros(Q, bool)
    if done.all():
        print(f"chunk {a.chunk}: already complete")
        return
    base = scratch_base()
    scratch = tempfile.mkdtemp(prefix=f"pgscan_{a.chunk}_", dir=base)
    t0 = time.time()
    try:
        qcache = os.path.join(base, "pgscan_qcache_" + os.environ.get("USER", "u"))   # shared by tasks on this node
        os.makedirs(qcache, exist_ok=True)
        tcache = os.path.join(scratch, "tcache")
        os.makedirs(tcache, exist_ok=True)
        qs = [stage_site(p, "q", os.path.join(scratch, "q", str(i)), qcache) for i, p in enumerate(queries)]
        ts = [stage_site(dbs[i], "t", os.path.join(scratch, "t", str(i)), tcache) for i in idx]
        print(f"chunk {a.chunk}/{a.nchunks}: DB [{idx[0]},{idx[-1] + 1}) x {Q} queries; staged in "
              f"{time.time() - t0:.0f}s (missing: {sum(x is None for x in qs)} queries, {sum(x is None for x in ts)} "
              f"DB sites); {int(done.sum())} queries already done; workers {a.workers}; host {os.uname().nodename} "
              f"cpu {cpu_model()}; scratch {scratch}", flush=True)
        todo = [(i, qs[i]) for i in range(Q) if not done[i]]
        last = time.time()
        with Pool(a.workers, initializer=scan_init, initargs=(ts, a.timeout, a.retry_timeout, scratch)) as pool:
            for qi, zz, rr, ss, nn, tt in pool.imap_unordered(scan_query, todo):
                z[qi], raw[qi], st[qi], ncl[qi], sec[qi], done[qi] = zz, rr, ss, nn, tt, True
                if time.time() - last > CKPT_MIN * 60:
                    scan_save(out, z, raw, st, ncl, sec, idx, done)
                    last = time.time()
                    print(f"  {int(done.sum())}/{Q} queries, {time.time() - t0:.0f}s", flush=True)
        scan_save(out, z, raw, st, ncl, sec, idx, done)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"chunk {a.chunk} done in {time.time() - t0:.0f}s: ok {int((st == ST_OK).sum())}, noaln "
          f"{int((st == ST_NOALN).sum())}, missing {int((st == ST_MISSING).sum())}, timeout "
          f"{int((st == ST_TIMEOUT).sum())}, fail {int((st == ST_FAIL).sum())}, err {int((st == ST_ERR).sum())} "
          f"of {st.size}; mean s/pair {float(sec[st == ST_OK].mean()) if (st == ST_OK).any() else float('nan'):.3f}",
          flush=True)
    print("SCAN_CHUNK_DONE", flush=True)


def gscan_init(db_staged, timeout, scratch):
    G["db"], G["timeout"] = db_staged, timeout
    G["w"] = tempfile.mkdtemp(prefix="gw_", dir=scratch)


def gscan_query(arg):
    qi, q = arg
    C = len(G["db"])
    ga = np.full(C, np.nan, np.float32)
    st = np.zeros(C, np.uint8)
    sec = np.zeros(C, np.float32)
    for j, t in enumerate(G["db"]):
        try:
            st[j], ga[j], sec[j] = glosa_compare(q, t, G["w"], G["timeout"])
        except Exception:  # noqa: BLE001
            st[j] = ST_GERR
    return qi, ga, st, sec


def stage_queries(queries, base):
    """Node-level query cache: staged copies + features, built once under a file lock."""
    qdir = os.path.join(base, "pgglosa_qcache_" + os.environ.get("USER", "u"))
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        staged = [stage_pocket(p, qdir) for p in queries]
        features(staged, qdir)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return staged


def gscan_save(path, ga, st, sec, idx, done):
    tmp = path + ".tmp.npz"
    np.savez(tmp, ga=ga, status=st, sec=sec.astype(np.float16), db_idx=idx, done=done)
    os.replace(tmp, path)


def cmd_scan_glosa(a):
    """Same chunking as scan-probis; OUT/chunk_KKKK.npz holds ga (float32), status, sec, db_idx, done. Pockets are
    staged and featurized once (queries once per node), then glosa runs per pair in a private working directory."""
    check_tools(glosa=True, batch_cf=True)
    base = scratch_base()
    scratch = tempfile.mkdtemp(prefix="pgglosa_", dir=base)
    assert "pdb" not in scratch and "pdb" not in base, scratch
    t0 = time.time()
    try:
        queries, dbs, idx = chunk_of(a)
        os.makedirs(a.out, exist_ok=True)
        out = os.path.join(a.out, f"chunk_{a.chunk:04d}.npz")
        Q, Cc = len(queries), len(idx)
        if os.path.exists(out):
            d = np.load(out)
            ga, st, sec, done = d["ga"], d["status"], d["sec"].astype(np.float32), d["done"]
            assert ga.shape == (Q, Cc) and (d["db_idx"] == idx).all()
        else:
            ga = np.full((Q, Cc), np.nan, np.float32)
            st = np.zeros((Q, Cc), np.uint8)
            sec = np.zeros((Q, Cc), np.float32)
            done = np.zeros(Q, bool)
        if done.all():
            print(f"chunk {a.chunk}: already complete")
            return
        qs = stage_queries(queries, base)
        tdir = os.path.join(scratch, "t")
        os.makedirs(tdir)
        ts = [stage_pocket(dbs[i], tdir) for i in idx]
        tf = features(ts, tdir)
        print(f"chunk {a.chunk}/{a.nchunks}: DB [{idx[0]},{idx[-1] + 1}) x {Q} queries; staged + features in "
              f"{time.time() - t0:.0f}s (DB features {tf:.0f}s; missing {sum(x is None for x in qs)} queries, "
              f"{sum(x is None for x in ts)} DB; no features {sum(cf_of(x) is None for x in qs if x)} q, "
              f"{sum(cf_of(x) is None for x in ts if x)} DB); {int(done.sum())} queries done; workers {a.workers}; "
              f"host {os.uname().nodename} cpu {cpu_model()}", flush=True)
        todo = [(i, qs[i]) for i in range(Q) if not done[i]]
        last = time.time()
        with Pool(a.workers, initializer=gscan_init, initargs=(ts, a.timeout, scratch)) as pool:
            for qi, g, s, sc in pool.imap_unordered(gscan_query, todo):
                ga[qi], st[qi], sec[qi], done[qi] = g, s, sc, True
                if time.time() - last > CKPT_MIN * 60:
                    gscan_save(out, ga, st, sec, idx, done)
                    last = time.time()
                    print(f"  {int(done.sum())}/{Q} queries, {time.time() - t0:.0f}s", flush=True)
        gscan_save(out, ga, st, sec, idx, done)
        cnt = {GST_NAMES[c]: int((st == c).sum()) for c in range(8) if (st == c).any()}
        print(f"chunk {a.chunk} done in {time.time() - t0:.0f}s: {cnt}; mean s/pair "
              f"{float(sec[st == ST_OK].mean()) if (st == ST_OK).any() else float('nan'):.4f}", flush=True)
        print("GLOSA_CHUNK_DONE", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def cmd_assemble(a):
    """Chunk files of a scan -> OUT/all_status.npy (uint8) and all_raw.npy (ProBiS) or all_ga.npy (G-LoSA), float32,
    NaN unless status 1."""
    key = "raw" if a.cmd == "assemble" else "ga"
    Q, N = len(read_list(a.queries)), len(read_list(a.db))
    val = np.full((Q, N), np.nan, np.float32)
    st = np.zeros((Q, N), np.uint8)
    incomplete = []
    for k in range(a.nchunks):
        fn = os.path.join(a.chunks, f"chunk_{k:04d}.npz")
        if not os.path.exists(fn):
            incomplete.append((k, "missing"))
            continue
        d = np.load(fn)
        i = d["db_idx"]
        val[:, i], st[:, i] = d[key], d["status"]
        if not d["done"].all():
            incomplete.append((k, f"{int((~d['done']).sum())} queries pending"))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "all_status.npy"), st)
    np.save(os.path.join(a.out, f"all_{key}.npy"), val)
    print(f"{Q} x {N} = {Q * N} pairs; status counts 0..7: {[int((st == c).sum()) for c in range(8)]}; "
          f"incomplete chunks: {incomplete or 'none'}")


# ------------------------------------------------------------------------------------------------- label tables
def load_records(stores):
    """{(query_path, target_path, tool): record} of every part_*.tsv of the stores; per key the last record by time,
    final statuses before failures."""
    fs = [f for s in stores for f in glob.glob(os.path.join(s, "part_*.tsv"))]
    if not fs:
        return {}
    d = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in fs])
    d = d[d.tool.isin(set(TOOLS)) & (d.status != "")]
    d["fin"] = d.status.isin(FINAL)
    d = d.sort_values(["fin", "ts"]).drop_duplicates(["query_path", "target_path", "tool"], keep="last")
    return {(q, t, tl): r for q, t, tl, r in zip(d.query_path, d.target_path, d.tool, d.to_dict("records"))}


def num(x):
    try:
        return float(x) if x not in ("", None, "None") else NAN
    except ValueError:
        return NAN


def cmd_tables(a):
    """Every pair-list row with probis_z, probis_align, probis_status (the scan: Z and raw score rounded to 2 decimals
    for status 1, the status name otherwise; for a pair outside the scan the `probis` pair score) and ga_all,
    ga_all_status (the `glosa` pair score). 'pending' = not computed."""
    S = load_records(a.store)
    SC = None
    if a.scan:
        qs, dbs = read_list(a.query_sites), read_list(a.db_sites)
        SC = {"qi": {p: i for i, p in enumerate(qs)}, "di": {p: i for i, p in enumerate(dbs)},
              "raw": np.load(f"{a.scan}/all_raw.npy", mmap_mode="r"),
              "st": np.load(f"{a.scan}/all_status.npy", mmap_mode="r")}
    os.makedirs(a.out, exist_ok=True)
    for lst in a.lists:
        d = pd.read_csv(lst, sep="\t")
        rows = []
        for q, t in zip(d.query_path, d.target_path):
            r = {}
            p = S.get((q, t, "probis"))
            if SC is not None and q in SC["qi"] and t in SC["di"]:
                st, raw = int(SC["st"][SC["qi"][q], SC["di"][t]]), float(SC["raw"][SC["qi"][q], SC["di"][t]])
                p = None
                r.update(probis_z=round((raw - 1.958) / 2.21468, 2) if st == 1 else NAN,
                         probis_align=round(raw, 2) if st == 1 else NAN, probis_status=SCAN_STATUS.get(st, "pending"))
            if p:
                r.update(probis_z=num(p["score1"]), probis_align=num(p["score2"]), probis_status=p["status"])
            elif "probis_status" not in r:
                r.update(probis_z=NAN, probis_align=NAN, probis_status="pending")
            g = S.get((q, t, "glosa"))
            r.update(ga_all=num(g["score1"]) if g else NAN, ga_all_status=g["status"] if g else "pending")
            rows.append(r)
        x = pd.DataFrame(rows, columns=["probis_z", "probis_align", "probis_status", "ga_all", "ga_all_status"])
        out = pd.concat([d.reset_index(drop=True), x], axis=1)
        fn = os.path.join(a.out, os.path.basename(lst).replace(".tsv", ".scored.tsv"))
        out.to_csv(fn, sep="\t", index=False, na_rep="NaN")
        print(f"{fn}: {len(out)} rows; probis {out.probis_status.value_counts().to_dict()}; ga_all "
              f"{out.ga_all_status.value_counts().to_dict()}")


# ------------------------------------------------------------------------------------------------- census
def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def java_merges(lines):
    """Residue boundaries AssignChemicalFeatures misses: consecutive (non-hydrogen) ATOM lines whose residue key
    (chain, number, icode) changes while the number field stays the same."""
    merges, prev = 0, None
    for ln in lines:
        if not (ln.startswith("ATOM") and len(ln) > 26) or ln[12:16].strip().startswith("H"):
            continue
        key = (ln[21], ln[22:26], ln[26])
        if prev is not None and key != prev and int(ln[22:26]) == int(prev[1]):
            merges += 1
        prev = key
    return merges


def ligand_info(pocket):
    """(ligand file, residue name of its first HETATM line, heavy atoms) of a pocket."""
    d, b = os.path.dirname(pocket), os.path.basename(pocket)
    cand = os.path.join(d, b.replace("ligand_pocket_", "ligand_"))
    if not os.path.isfile(cand):
        num = b.split("_")[2] if b.count("_") >= 2 else ""
        alts = sorted(glob.glob(os.path.join(d, f"ligand_{num}_*.pdb")))
        alts = [a for a in alts if "ligand_pocket_" not in a]
        cand = alts[0] if alts else ""
    if not cand:
        return "", "", 0
    code, heavy = "", 0
    try:
        for ln in read_lines(cand):
            if ln.startswith("HETATM"):
                if not code:
                    code = ln[17:20].strip()
                el = ln[76:78].strip() if len(ln) > 77 else ""
                name = ln[12:16].strip()
                if el != "H" and not (not el and name.startswith("H")):
                    heavy += 1
    except OSError:
        pass
    return os.path.basename(cand), code, heavy


CENSUS_COLS = ["site", "pocket_ok", "protein_ok", "n_neg_res", "glosa_merge_all", "lig_file", "lig_code", "lig_heavy"]


def census_row(pocket):
    """One census row; a pocket file that cannot be parsed gives a row with the site only (not usable)."""
    row = {"site": pocket, "pocket_ok": 0}
    if not os.path.isfile(pocket):
        return row
    try:
        row["pocket_ok"] = 1
        res = pocket_residues(pocket)
        row["n_neg_res"] = sum(1 for r in res if r[2] == " " and r[1] < 0)
        row["glosa_merge_all"] = java_merges([ln for ln in read_lines(pocket) if ln.startswith("ATOM")])
        row["lig_file"], row["lig_code"], row["lig_heavy"] = ligand_info(pocket)
        row["protein_ok"] = int(os.path.isfile(os.path.join(os.path.dirname(pocket), "protein.pdb")))
    except Exception:  # noqa: BLE001
        row = {"site": pocket}
    return row


def cmd_census(a):
    sites = read_list(a.sites)
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        f.write("\t".join(CENSUS_COLS) + "\n")
        rows = pool.imap(census_row, sites, chunksize=200)
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in CENSUS_COLS) + "\n")
    c = pd.read_csv(a.out, sep="\t", dtype=str, keep_default_na=False)
    neg, mer = (pd.to_numeric(c[k], errors="coerce").fillna(0) for k in ("n_neg_res", "glosa_merge_all"))
    print(f"{len(c)} sites -> {a.out}; pocket files missing {int((c.pocket_ok != '1').sum())}; sites with negative "
          f"residue numbers {int((neg > 0).sum())}; G-LoSA residue merges {int(mer.sum())} in "
          f"{int((mer > 0).sum())} files")


# ------------------------------------------------------------------------------------------------- unrelated pairs
def fetch_smiles_code(code):
    import requests
    for _ in range(3):
        try:
            r = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=10)
            if r.status_code == 404:
                return code, "", "missing"
            r.raise_for_status()
            s = r.json().get("rcsb_chem_comp_descriptor", {}).get("smiles") or ""
            return code, s, "rcsb" if s else "missing"
        except Exception:  # noqa: BLE001
            continue
    return code, "", "missing"


def cmd_smiles(a):
    """SMILES of the census ligand codes: rcsb_chem_comp_descriptor.smiles of
    https://data.rcsb.org/rest/v1/core/chemcomp/<code> (network; 8 threads, 3 tries), reusing the entries of --cache
    (comp_id, smiles). Writes comp_id, smiles, source (cache | rcsb | missing)."""
    from concurrent.futures import ThreadPoolExecutor
    codes = sorted(set(pd.read_csv(a.census, sep="\t", dtype={"lig_code": str}, keep_default_na=False).lig_code) - {""})
    cache = {}
    if a.cache:
        c = pd.read_csv(a.cache, sep="\t", dtype=str, keep_default_na=False)
        cache = dict(zip(c.comp_id.str.upper(), c.smiles))
    rows = [(c, cache[c.upper()], "cache") for c in codes if c.upper() in cache and cache[c.upper()]]
    todo = [c for c in codes if not (c.upper() in cache and cache[c.upper()])]
    print(f"{len(codes)} codes; {len(rows)} from the cache; fetching {len(todo)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        rows += list(ex.map(fetch_smiles_code, todo))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows, columns=["comp_id", "smiles", "source"]).sort_values("comp_id").to_csv(
        a.out, sep="\t", index=False)


def cmd_null(a):
    """OUT/null/null_set.npz: null = packbits of the (Q, N) mask along axis 1, shape, per_query, and the pairs removed
    by each rule in order. No scan score is read."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    ID_T, DICE_T, HEAVY_T = 0.30, 0.46, 6
    queries, dbs = read_list(a.query_sites), read_list(a.db_sites)
    c = pd.read_csv(a.census, sep="\t", dtype={"lig_code": str, "chains": str}, keep_default_na=False)
    c = c.set_index("site")

    def info(paths):
        sub = c.reindex(paths)
        ok = (sub.pocket_ok.astype(str) == "1") & (sub.protein_ok.astype(str) == "1")
        code = sub.lig_code.fillna("").astype(str).str.upper().values
        heavy = pd.to_numeric(sub.lig_heavy, errors="coerce").fillna(0).values
        pdb = np.array([p.split("/")[-2].lower() for p in paths])
        return ok.values, code, heavy, pdb

    q_ok, q_code, q_heavy, q_pdb = info(queries)
    d_ok, d_code, d_heavy, d_pdb = info(dbs)
    smi = pd.read_csv(a.smiles, sep="\t", dtype=str, keep_default_na=False)
    smi = dict(zip(smi.iloc[:, 0].str.upper(), smi.iloc[:, 1]))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}
    for code in set(q_code) | set(d_code):
        s = smi.get(code, "")
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            fps[code] = gen.GetFingerprint(m)
    codes = sorted(fps)
    cidx = {k: i for i, k in enumerate(codes)}
    dice = {q: np.array(DataStructs.BulkDiceSimilarity(fps[q], [fps[b] for b in codes]), dtype=np.float32)
            for q in sorted({x for x in q_code if x in fps})}
    m8 = pd.read_csv(a.m8, sep="\t", header=None, usecols=[0, 1, 2], names=["q", "t", "fident"])
    m8 = m8[m8.fident >= ID_T]
    hom = set(zip(m8.q.str.split("_").str[0].str.lower(), m8.t.str.split("_").str[0].str.lower()))
    Q, N = len(queries), len(dbs)
    null = np.zeros((Q, N), dtype=bool)
    reasons = {"missing_file": 0, "homolog_ge30": 0, "same_code": 0, "no_smiles": 0, "small_ligand": 0,
               "dice_ge_0.46": 0}
    hom_cache = {}
    d_has_fp = np.array([x in fps for x in d_code])
    d_cidx = np.array([cidx.get(x, -1) for x in d_code])
    for i in range(Q):
        m = np.ones(N, dtype=bool)
        base = ~(d_ok & q_ok[i])
        reasons["missing_file"] += int(base.sum())
        m &= ~base
        if q_pdb[i] not in hom_cache:
            hom_cache[q_pdb[i]] = np.array([(q_pdb[i], p) in hom for p in d_pdb])
        homo = hom_cache[q_pdb[i]]
        reasons["homolog_ge30"] += int((m & homo).sum())
        m &= ~homo
        same = d_code == q_code[i]
        reasons["same_code"] += int((m & same).sum())
        m &= ~same
        nofp = ~d_has_fp if q_code[i] in fps else np.ones(N, dtype=bool)
        reasons["no_smiles"] += int((m & nofp).sum())
        m &= ~nofp
        small = (d_heavy < HEAVY_T) | (q_heavy[i] < HEAVY_T)
        reasons["small_ligand"] += int((m & small).sum())
        m &= ~small
        if q_code[i] in fps:
            dsim = np.where(d_cidx >= 0, dice[q_code[i]][np.maximum(d_cidx, 0)], 1.0)
            hi = dsim >= DICE_T
            reasons["dice_ge_0.46"] += int((m & hi).sum())
            m &= ~hi
        null[i] = m
    per_q = null.sum(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, null=np.packbits(null, axis=1), shape=np.array([Q, N]), per_query=per_q,
                        reasons=np.array(list(reasons.items()), dtype=object))
    print(f"null pairs: {int(per_q.sum())} of {Q * N} ({per_q.sum() / (Q * N):.3%}); removed in order: {reasons}; "
          f"ligand codes with a fingerprint: {len(codes)}")


# ------------------------------------------------------------------------------------------------- G-LoSA build
def cmd_build_glosa(a):
    """G-LoSA v2.2 from https://compbio.lehigh.edu/GLoSA/glosa_v2.2.tar.gz (free for academic use; sha256 8a69adca...):
    GLOSA_DIR/glosa (g++ -O3; published runs g++ 9.4.0), GLOSA_DIR/cf/AssignChemicalFeatures.class and
    GLOSA_DIR/cfbatch/BatchCF.class (javac --release 8)."""
    if not GLOSA_DIR:
        sys.exit("[error] set GLOSA_DIR to the install directory")
    expected = "8a69adca75b172dd622dd84724f5cedcf91c21a19efb0e6ba298e9b496cafe3d"
    got = hashlib.sha256(open(a.tarball, "rb").read()).hexdigest()
    if got != expected:
        print(f"warning: sha256 {got} differs from the published build ({expected})", file=sys.stderr)
    src = tempfile.mkdtemp()
    try:
        subprocess.run(["tar", "-xzf", a.tarball, "-C", src], check=True)
        s = os.path.join(src, "glosa_v2.2")
        for d in ("cf", "cfbatch"):
            os.makedirs(os.path.join(GLOSA_DIR, d), exist_ok=True)
        subprocess.run(["g++", "-O3", "-o", os.path.join(GLOSA_DIR, "glosa"), "glosa.cpp"], cwd=s, check=True)
        subprocess.run(["javac", "--release", "8", "-d", GLOSA_CF_CP, "AssignChemicalFeatures.java"], cwd=s, check=True)
        subprocess.run(["javac", "--release", "8", "-cp", GLOSA_CF_CP, "-d", os.path.join(GLOSA_DIR, "cfbatch"),
                        os.path.join(HERE, "BatchCF.java")], check=True)
    finally:
        shutil.rmtree(src, ignore_errors=True)
    print(f"G-LoSA built in {GLOSA_DIR}")


# ------------------------------------------------------------------------------------------------- command line
def main():
    sites = os.path.join(OUT, "sites")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, help_):
        x = sub.add_parser(name, help=help_)
        x.set_defaults(fn=fn)
        return x

    def site_args(x, q="--query-sites", d="--db-sites"):
        x.add_argument(q, default=os.path.join(sites, "all_query_sites.txt"))
        x.add_argument(d, default=os.path.join(sites, "all_db_sites.txt"))

    x = p("build-glosa", cmd_build_glosa, "compile G-LoSA 2.2 and its Java helpers into $GLOSA_DIR")
    x.add_argument("tarball")
    x = p("sites", cmd_sites, "site lists of the scans")
    x.add_argument("eval_dir", help="directory with q_paths.txt and db_paths.txt (e.g. $LENSEEK_EVAL_DIR)")
    x.add_argument("--out", default=sites)
    x = p("pairs", cmd_pairs, "top-k pair list of one model run")
    x.add_argument("eval_dir", help="directory with dist.npy, q_paths.txt, db_paths.txt")
    x.add_argument("out_tsv")
    x.add_argument("--topk", type=int, default=100)
    x = p("score", cmd_score, "ProBiS / G-LoSA scores of listed pairs into an append-only store")
    x.add_argument("--pairs", nargs="+", required=True, help="pair lists (query_path, target_path columns)")
    x.add_argument("--store", default=os.path.join(OUT, "store"))
    x.add_argument("--tools", default="probis,glosa")
    x.add_argument("--ntasks", type=int, default=1)
    x.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    x.add_argument("--workers", type=int, default=workers)
    for name, fn, out, tmo in (("scan-probis", cmd_scan_probis, "scan/chunks", 300),
                               ("scan-glosa", cmd_scan_glosa, "scan_glosa/chunks", 600)):
        x = p(name, fn, "one database chunk of the database-wide scan")
        site_args(x, "--queries", "--db")
        x.add_argument("--out", default=os.path.join(OUT, out))
        x.add_argument("--nchunks", type=int, default=115)
        x.add_argument("--chunk", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
        x.add_argument("--workers", type=int, default=workers)
        x.add_argument("--timeout", type=int, default=tmo, help="seconds per comparison")
        if name == "scan-probis":
            x.add_argument("--retry-timeout", type=int, default=1800, help="seconds for the second run of a timed-out "
                           "or failed comparison")
    for name, d in (("assemble", "scan"), ("assemble-glosa", "scan_glosa")):
        x = p(name, cmd_assemble, "chunk files -> full arrays")
        x.add_argument("--chunks", default=os.path.join(OUT, d, "chunks"))
        x.add_argument("--nchunks", type=int, default=115)
        site_args(x, "--queries", "--db")
        x.add_argument("--out", default=os.path.join(OUT, d))
    x = p("tables", cmd_tables, "label tables of pair lists")
    x.add_argument("lists", nargs="+")
    x.add_argument("--store", action="append", default=None, help="score store directory (repeatable)")
    x.add_argument("--scan", default=os.environ.get("LENSEEK_SCAN_DIR") or os.path.join(OUT, "scan"))
    site_args(x)
    x.add_argument("--out", default=os.path.join(OUT, "tables"))
    x = p("census", cmd_census, "per-site counts")
    x.add_argument("--sites", default=os.path.join(sites, "all_sites.txt"))
    x.add_argument("--out", default=os.path.join(OUT, "census.tsv"))
    x.add_argument("--workers", type=int, default=workers)
    x = p("smiles", cmd_smiles, "SMILES of the census ligand codes (RCSB, network)")
    x.add_argument("--census", default=os.path.join(OUT, "census.tsv"))
    x.add_argument("--cache", default=None, help="comp_id <tab> smiles table whose entries are reused")
    x.add_argument("--out", default=os.path.join(OUT, "null", "ccd_smiles.tsv"))
    x = p("null", cmd_null, "unrelated (null) pairs")
    x.add_argument("m8", help="MMseqs2 hits of the query chains against the database chains (ids <pdb>_<chain>)")
    x.add_argument("--census", default=os.path.join(OUT, "census.tsv"))
    x.add_argument("--smiles", default=os.path.join(OUT, "null", "ccd_smiles.tsv"), help="comp_id <tab> smiles")
    site_args(x)
    x.add_argument("--out", default=os.path.join(OUT, "null", "null_set.npz"))
    a = ap.parse_args()
    if a.cmd == "tables" and not a.store:
        a.store = [os.path.join(OUT, "store")]
    a.fn(a)


if __name__ == "__main__":
    main()
