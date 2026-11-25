#!/usr/bin/env python
"""Folddisco on the evaluation data (Supplementary Section S6): input files, index, search, and the distance matrix and
top-100 list that the analysis scripts read as for DeeplyTough.

Folddisco 3.0.0 (steineggerlab/folddisco, commit 3e193e3) runs with its default index and query parameters. Its inputs
are the files of the site lists, rewritten as for DeeplyTough (normalize_pdb of scripts/deeplytough.py: standard PDB
columns, same atoms and coordinates, altLoc blanked) with the residues in (chain, residue number, insertion code) order
and the atoms of a residue in file order, first MODEL only. Folddisco reads fixed columns, starts a new residue only
when the residue number changes and reads ATOM records of the first model.

Runs: one Folddisco process per query; every database site gets a ranking key d (smaller = more similar).
  prefilter  index of the database pocket files, one structure per site; the query pocket file searched with all its
             residues, pre-filter only (--skip-match --per-structure). Score idf (IDF-weighted coverage with a length
             penalty), d = -(idf + 1e-7 node_count): ties go to more matched query residues, then to the lower
             database index. Supplementary Tables S8, S9, S10, S13 and S16.
  matching   the same index and queries with residue matching after the pre-filter: d = -structure_score + 1e-7
             min(min_rmsd, 999).
  blosum62   as prefilter, with every residue Folddisco reads from the query file listed (-q) and --aa-subst blosum62.
  entries    index of the protein.pdb files of the database entries; the query entry searched with the residues of the
             query site file (-q), one output row per match (no --top limit). A match goes to the database site of its
             entry whose pocket file holds most of its matched residues (chain and number; ties: lower database index);
             a site's score is match_score of its first match in Folddisco's order (match_score desc, rmsd asc),
             d = -match_score + 1e-7 min(rmsd, 999); matches that share no residue with a site are not used.
matching, blosum62 and entries are the sensitivity runs of Supplementary Section S6. The printed scores have 4
decimals, so the 1e-7 terms only order sites of equal score. A database site that a query does not return gets
d = 1000 (the last rank, shared); a query that fails keeps d = 1000 for every site.

Subcommands (OUT = --out, default $LENSEEK_OUT/folddisco; WORK = --work, default OUT/work, best on a local disk):
  prepare   WORK/q_pockets/<q:03d>.pdb, WORK/db_pockets/<i:06d>.pdb (row order of the site lists) and
            WORK/manifest_{q,db}_pockets.tsv (per file: the residues Folddisco reads and every residue of the file, as
            chain_number lists, or the error); with --entries also WORK/q_entries/<pdb>.pdb, WORK/db_entries/<pdb>.pdb
            (protein.pdb of every entry) and their manifests
  index     WORK/idx/pockets (with --entries WORK/idx/entries), built in WORK
  search    OUT/runs/<run>/q<q:03d>.tsv.gz and q<q:03d>.log (command, Folddisco's log, exit status, wall and CPU
            seconds); queries with an output are skipped
  assemble  OUT/<run>/dist.npy (float64, queries x database sites in site-list order), q_paths.txt, db_paths.txt,
            per_query.tsv (status, output rows, database sites returned) and top100_pairs.tsv (the 100 smallest d of
            every query, ties to the lower database index, in the pair-list format of scripts/labels.py pairs with d at
            full precision)
Inputs: FOLDDISCO (the folddisco executable; default `folddisco` on PATH) and the site lists of scripts/labels.py sites
(--queries, --db; protein.pdb next to each site file).

usage:
  python scripts/folddisco.py prepare && python scripts/folddisco.py index
  python scripts/folddisco.py search && python scripts/folddisco.py assemble
  python scripts/folddisco.py search --run matching && python scripts/folddisco.py assemble --run matching
  python scripts/folddisco.py prepare --entries && python scripts/folddisco.py index --entries
  python scripts/folddisco.py search --run entries --part 0/3          (parts 0/3, 1/3, 2/3 on three nodes)
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deeplytough import normalize_pdb  # noqa: E402
from pipeline import die, pocket_protein, read_list, site_key  # noqa: E402

OUT_ROOT = os.environ.get("LENSEEK_OUT") or "out"
OUT = os.path.join(OUT_ROOT, "folddisco")
SITES = os.path.join(OUT_ROOT, "labels", "sites")
RUNS = ("prefilter", "matching", "blosum62", "entries")
COLUMNS = {"prefilter": "tid,idf,node_count", "blosum62": "tid,idf,node_count",
           "matching": "tid,structure_score,min_rmsd", "entries": "tid,match_score,rmsd,matching_residues"}
NOT_RETURNED = 1000.0
TOP_ALL = 1000000000   # entries: --top counts printed match rows
K = 100
G = {}


def folddisco_bin():
    """The folddisco executable: $FOLDDISCO, else `folddisco` on PATH."""
    b = os.environ.get("FOLDDISCO") or shutil.which("folddisco")
    if not b or not os.access(b, os.X_OK):
        die(f"Folddisco executable not found: FOLDDISCO={os.environ.get('FOLDDISCO')!r} (set FOLDDISCO or put "
            "`folddisco` on PATH)")
    return b


def entry_of(path):
    return site_key(path).split("/")[0]


# ---------------------------------------------------------------- prepare
def residue_sorted(lines):
    """Coordinate records grouped by residue (chain, number, insertion code) in sorted order; atoms keep their order.
    TER/END/MODEL records are dropped. Only the first MODEL is kept."""
    groups, order = {}, []
    nmodel = 0
    for L in lines:
        if L.startswith("MODEL"):
            nmodel += 1
            if nmodel > 1:
                break
        if not L.startswith(("ATOM  ", "HETATM")):
            continue
        k = (L[21], int(L[22:26]), L[26])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(L)
    keys = sorted(order)
    return [L for k in keys for L in groups[k]], keys


def folddisco_residues(lines):
    """The residues Folddisco reads from a rewritten file, [(chain, number)]: ATOM records only; a residue ends when the
    residue number changes; it is kept when it has N and CA; the last atom of the file is never part of a kept
    residue; negative residue numbers are not read."""
    atoms = []
    for L in lines:
        if not L.startswith("ATOM  "):
            continue
        try:
            float(L[30:38]); float(L[38:46]); float(L[46:54]); int(L[6:11]); s = int(L[22:26])  # noqa: E702
        except ValueError:
            continue
        if s < 0:
            continue
        atoms.append((s, L[12:16], L[21]))
    kept, prev, n, ca, prev_chain = [], None, False, False, None
    for i, (s, name, ch) in enumerate(atoms):
        if prev != s or i == len(atoms) - 1:
            if n and ca:
                kept.append((prev_chain, prev))
            n = ca = False
            prev, prev_chain = s, ch
        if name == " CA ":
            ca = True
        elif name == " N  ":
            n = True
    return kept


def res_string(res):
    return ",".join(f"{c}_{r}" for c, r in res)


def rewrite(task):
    """One input file -> its rewritten copy and manifest row."""
    kind, key, src, dst = task
    try:
        normalize_pdb(src, dst)
        rec, keys = residue_sorted(open(dst).read().splitlines())
        with open(dst, "w") as f:
            f.write("\n".join(rec) + "\nEND\n")
        pocket = kind.endswith("pockets")
        return dict(key=key, path=src, ok=1, residues=res_string(folddisco_residues(rec)) if pocket else "",
                    residues_all=res_string([(c, r) for c, r, _ in keys]) if pocket else "", error="")
    except Exception as e:  # noqa: BLE001  (recorded in the manifest)
        if os.path.exists(dst):
            os.remove(dst)
        return dict(key=key, path=src, ok=0, residues="", residues_all="", error=f"{type(e).__name__}: {e}"[:300])


def cmd_prepare(a):
    q, db = read_list(a.queries), read_list(a.db)
    kinds = [("q_pockets", [(f"{i:03d}", p) for i, p in enumerate(q)]),
             ("db_pockets", [(f"{i:06d}", p) for i, p in enumerate(db)])]
    if a.entries:
        for kind, paths in (("q_entries", q), ("db_entries", db)):
            first = {}
            for p in paths:
                first.setdefault(entry_of(p), pocket_protein(p))
            kinds.append((kind, list(first.items())))
    with Pool(a.workers) as pool:
        for kind, items in kinds:
            os.makedirs(os.path.join(a.work, kind), exist_ok=True)
            rows = pool.map(rewrite, [(kind, key, src, os.path.join(a.work, kind, f"{key}.pdb")) for key, src in items],
                            chunksize=64)
            pd.DataFrame(rows).to_csv(os.path.join(a.work, f"manifest_{kind}.tsv"), sep="\t", index=False)
            bad = [r for r in rows if not r["ok"]]
            print(f"{kind}: {len(rows)} files, {len(bad)} failed {[(r['path'], r['error']) for r in bad[:5]]}",
                  flush=True)


def manifest(work, kind):
    f = os.path.join(work, f"manifest_{kind}.tsv")
    if not os.path.exists(f):
        die(f"{f} missing: run `prepare{' --entries' if 'entries' in kind else ''}` first")
    return pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- index and search
def timed(cmd, cwd, log):
    """Run cmd in cwd, stdout and stderr to log (after the command line): exit status, wall seconds, CPU seconds (user +
    system of the process) and peak resident memory in kB, from os.wait4."""
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=f)
        _, status, ru = os.wait4(p.pid, 0)
        wall = time.perf_counter() - t0
    p.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
    return p.returncode, wall, ru.ru_utime + ru.ru_stime, ru.ru_maxrss


def build_index(work, kind, threads, prefix):
    """folddisco index of WORK/db_<kind> (relative target paths) into prefix; log prefix.log."""
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    rc, wall, cpu, rss = timed([folddisco_bin(), "index", "-p", f"db_{kind}", "-i", os.path.abspath(prefix), "-t",
                                str(threads), "-v"], work, os.path.abspath(prefix) + ".log")
    if rc:
        die(f"folddisco index exited with {rc} (see {prefix}.log)")
    return dict(kind=kind, n_files=len(os.listdir(os.path.join(work, f"db_{kind}"))), threads=threads, rc=rc,
                wall_s=round(wall, 3), cpu_s=round(cpu, 3), max_rss_kb=rss)


def query_cmd(run, q, row, work, index, out, threads, n_db):
    """folddisco query command line of query q (row: its manifest_q_pockets row) in one run."""
    if run == "entries":
        pdb, res = os.path.join(work, "q_entries", entry_of(row["path"]) + ".pdb"), row["residues_all"]
        extra = ["--top", str(TOP_ALL)]
    else:
        pdb, res = os.path.join(work, "q_pockets", f"{q:03d}.pdb"), row["residues"] if run == "blosum62" else ""
        extra = (["--aa-subst", "blosum62"] if run == "blosum62" else []) + \
                (["--skip-match"] if run != "matching" else []) + ["--per-structure", "--top", str(n_db)]
    return [folddisco_bin(), "query", "-i", index, "-p", pdb] + (["-q", res] if res else []) + \
        ["-o", out, "-t", str(threads)] + extra + ["--header", "--format-output", COLUMNS[run], "-v"]


def query_one(run, q, row, work, index, tsv, threads, n_db):
    """One query: Folddisco writes tsv (timed), which is then compressed to tsv.gz; the status line ends the log."""
    log = tsv[:-4] + ".log"
    rc, wall, cpu, _ = timed(query_cmd(run, q, row, work, index, tsv, threads, n_db), work, log)
    n = -1
    if rc == 0 and os.path.exists(tsv):
        with open(tsv, "rb") as fi, gzip.open(tsv + ".gz.part", "wb", compresslevel=6) as fo:
            n = sum(1 for _ in fi)
            fi.seek(0)
            shutil.copyfileobj(fi, fo)
        os.replace(tsv + ".gz.part", tsv + ".gz")
    if os.path.exists(tsv):
        os.remove(tsv)
    r = dict(q_idx=q, rc=rc, wall_s=round(wall, 4), cpu_s=round(cpu, 4), n_lines=n)
    with open(log, "a") as f:
        f.write("STATUS " + " ".join(f"{k}={v}" for k, v in r.items()) + "\n")
    return r


def cmd_index(a):
    kind = "entries" if a.entries else "pockets"
    r = build_index(a.work, kind, a.threads, os.path.join(a.work, "idx", kind))
    print(f"index {kind}: {r['n_files']} files, {r['wall_s']:.1f} s wall, {r['cpu_s']:.0f} s CPU, peak "
          f"{r['max_rss_kb'] / 1e6:.1f} GB")


def cmd_search(a):
    nq, n_db = len(read_list(a.queries)), len(read_list(a.db))
    man = manifest(a.work, "q_pockets")
    if len(man) != nq or not (man.ok == "1").all():
        die("manifest_q_pockets.tsv does not list every query as rewritten")
    index = os.path.join(a.work, "idx", "entries" if a.run == "entries" else "pockets")
    od = os.path.abspath(os.path.join(a.out, "runs", a.run))
    os.makedirs(od, exist_ok=True)
    part, nparts = (int(x) for x in a.part.split("/"))
    todo = [q for q in range(nq) if q % nparts == part and not os.path.exists(os.path.join(od, f"q{q:03d}.tsv.gz"))]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda q: query_one(a.run, q, man.iloc[q], os.path.abspath(a.work), os.path.abspath(index),
                                              os.path.join(od, f"q{q:03d}.tsv"), a.threads, n_db), todo))
    bad = [r["q_idx"] for r in res if r["rc"] != 0]
    print(f"{a.run}: {len(res)} queries run, non-zero exit {bad}, {time.time() - t0:.0f} s -> {od}")


# ---------------------------------------------------------------- assemble
def read_out(path):
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep="\t", dtype={"tid": str}, keep_default_na=False, na_values=[""])


def parse_res(s):
    """matching_residues of one match: 'A57,_,B102' (or 'AA_250,...') -> {'A_57', 'B_102'}."""
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or tok == "_":
            continue
        if "_" in tok:
            c, r = tok.rsplit("_", 1)
        else:
            i = len(tok)
            while i > 0 and tok[i - 1].isdigit():
                i -= 1
            c, r = tok[:i], tok[i:]
        out.add(f"{c}_{int(r)}")
    return out


def site_index(work, db):
    """entry -> ({chain_number: [database index, ...]}, number of sites), from the residues of each pocket file."""
    out = {}
    for i, (p, res) in enumerate(zip(db, manifest(work, "db_pockets").residues_all)):
        m, n = out.setdefault(entry_of(p), ({}, 0))
        for x in (set(res.split(",")) if res else set()):
            m.setdefault(x, []).append(i)
        out[entry_of(p)] = (m, n + 1)
    return out


def one_query(q):
    """(q, database indices, d, per-query row) of one output file."""
    run, f = G["run"], os.path.join(G["rd"], f"q{q:03d}.tsv.gz")
    r = dict(q_idx=q, status="ok", n_rows=0, n_returned=0)
    if not os.path.exists(f):
        r["status"] = "no output"
        return q, None, None, r
    t = read_out(f)
    r["n_rows"] = len(t)
    ids = t.tid.str.split("/").str[-1].str.replace(".pdb", "", regex=False)
    if run == "entries":
        best, left = {}, {}
        for e, mr, ms, rm in zip(ids.to_numpy(), t.matching_residues.astype(str).to_numpy(), t.match_score.to_numpy(),
                                 t.rmsd.to_numpy()):
            m, ns = G["sites"].get(e, ({}, 0))
            if left.get(e, ns) == 0:
                continue                         # every site of this entry has its first match already
            cnt = {}
            for x in parse_res(mr):
                for i in m.get(x, ()):
                    cnt[i] = cnt.get(i, 0) + 1
            if not cnt:
                continue
            i = min(cnt, key=lambda k: (-cnt[k], k))       # most matched residues, then lower database index
            if i not in best:
                best[i] = (float(ms), float(rm))
                left[e] = left.get(e, ns) - 1
        idx = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
        sc = np.array([v[0] for v in best.values()], dtype=np.float64)
        rmv = np.array([v[1] for v in best.values()], dtype=np.float64)
        d = -sc + 1e-7 * np.minimum(rmv, 999)
    else:
        idx = ids.astype(np.int64).to_numpy()
        if len(np.unique(idx)) != len(idx):
            raise RuntimeError(f"{run} q{q}: a site appears twice")
        if run == "matching":
            d = -t.structure_score.to_numpy(np.float64) + 1e-7 * np.minimum(t.min_rmsd.to_numpy(np.float64), 999)
        else:
            d = -(t.idf.to_numpy(np.float64) + 1e-7 * t.node_count.to_numpy(np.float64))
    if len(idx) and (idx.min() < 0 or idx.max() >= G["n_db"]):
        raise RuntimeError(f"{run} q{q}: site index out of range")
    if (d >= NOT_RETURNED).any():
        raise RuntimeError(f"{run} q{q}: a score below the not-returned value")
    r["n_returned"] = int(len(idx))
    return q, idx, d, r


def cmd_assemble(a):
    q, db = read_list(a.queries), read_list(a.db)
    rd, od = os.path.join(a.out, "runs", a.run), os.path.join(a.out, a.run)
    os.makedirs(od, exist_ok=True)
    G.update(run=a.run, rd=rd, n_db=len(db), sites=site_index(a.work, db) if a.run == "entries" else None)
    D = np.full((len(q), len(db)), NOT_RETURNED, dtype=np.float64)
    rows = []
    with Pool(a.workers) as p:
        for qi, idx, d, r in p.imap(one_query, range(len(q))):
            if idx is not None:
                D[qi, idx] = d
            rows.append(r)
    np.save(os.path.join(od, "dist.npy"), D)
    pq = pd.DataFrame(rows)
    pq.to_csv(os.path.join(od, "per_query.tsv"), sep="\t", index=False)
    for name, lst in (("q_paths.txt", q), ("db_paths.txt", db)):
        with open(os.path.join(od, name), "w") as f:
            f.write("\n".join(lst) + "\n")
    ar = np.arange(len(db))
    with open(os.path.join(od, "top100_pairs.tsv"), "w") as f:
        f.write("query_idx\tquery_path\trank\tdb_idx\ttarget_path\tdist\n")
        for i in range(len(q)):
            for r, j in enumerate(np.lexsort((ar, D[i]))[:K], 1):
                f.write(f"{i}\t{q[i]}\t{r}\t{j}\t{db[j]}\t{float(D[i, j])!r}\n")
    ret = D < NOT_RETURNED
    print(f"{a.run}: queries with output {int((pq.status == 'ok').sum())}/{len(q)}; database sites returned per query "
          f"mean {ret.sum(1).mean():.1f} ({100 * ret.sum(1).mean() / len(db):.1f} %), never returned "
          f"{int((~ret.any(0)).sum())} -> {od}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, h in [("prepare", cmd_prepare, "rewritten input files and their manifests"),
                        ("index", cmd_index, "Folddisco index of the database pocket (or entry) files"),
                        ("search", cmd_search, "one Folddisco process per query"),
                        ("assemble", cmd_assemble, "distance matrix and top-100 list of one run")]:
        p = sub.add_parser(name, help=h)
        p.set_defaults(fn=fn)
        p.add_argument("--queries", default=os.path.join(SITES, "all_query_sites.txt"),
                       help="query site files, one per line (scripts/labels.py sites)")
        p.add_argument("--db", default=os.path.join(SITES, "all_db_sites.txt"), help="database site files")
        p.add_argument("--out", default=OUT)
        p.add_argument("--work", default=None, help="rewritten files and indexes (default OUT/work)")
        if name in ("prepare", "index"):
            p.add_argument("--entries", action="store_true", help="also the protein.pdb files (run entries)")
        if name in ("search", "assemble"):
            p.add_argument("--run", default="prefilter", choices=RUNS)
        if name in ("index", "search"):
            p.add_argument("--threads", type=int, default=16,
                           help="Folddisco threads (the output does not depend on it)")
        if name == "search":
            p.add_argument("--workers", type=int, default=1, help="queries run at the same time")
            p.add_argument("--part", default="0/1", help="I/N: the queries q with q %% N == I")
        if name in ("prepare", "assemble"):
            p.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.work = a.work or os.path.join(a.out, "work")
    a.fn(a)


if __name__ == "__main__":
    main()
