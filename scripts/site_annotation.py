"""Inputs derived from the search database and the evaluation queries: query sets, identity, structure and ligand
strata, sequence clusters, per-site annotations, MMseqs2 clusters and searches of the database chains, and the CCD
snapshot.

Subcommands, in the order they depend on each other:
  query-sets  primary query set (Supplementary Section S1): the evaluation proteins whose best MMseqs2 hit against
              the training chains has fident >= 0.80 at query coverage >= 0.80 are listed in primary_exclude_pdb.txt;
              the primary queries are the evaluation queries except the pockets of these proteins, the full set is
              all of them.
              Input $LENSEEK_STRAT_DIR/eval_vs_train.m8 (query = <pdb>_<chain>).
  identity-strata  sequence-identity stratum of every evaluation protein (the proteins of the query structures;
              Supplementary Tables S10 and S14): its best hit in the MMseqs2 search of its chains against the
              training chains among the hits with query or target coverage >= 0.5 (highest fident, then bit score,
              then target name); strata < 30 %, 30-40, 40-50, 50-80, 80-95, >= 95 %; a protein without such a hit is
              < 30 % (the search keeps hits >= 25 %). Input $LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8.
  strata      structure and ligand strata of every evaluation query (Supplementary Table S10), with the two functions
              of the strata of the later PDB entries (scripts/holdout.py strata): maximum Foldseek query TM-score of the pocket chains
              (chains with a standard-residue ATOM record in the pocket file; protein.pdb in standard columns) to the
              chains of the database entries, < 0.5, 0.5-0.8, >= 0.8; query ligand (first HETATM residue of the ligand
              file) with the same CCD as a database site / a new CCD with Morgan (r 2, 2,048 bits) Dice >= 0.46 to a
              database ligand / below (CCD snapshot and database ligand codes of asset reference_data).
  annotate    per-site annotation of every query and database site (Supplementary Section S2):
                pocket residues  ATOM/HETATM lines of ligand_pocket_<n>_<chains>.pdb, residue = (chain, resSeq, resName);
                                 the files have no insertion-code column
                main chain       the chain with the most pocket residues (ties alphabetical): it defines the site's
                                 sequence cluster and sequence identity
                ligand           first HETATM residue name of ligand_<n>_<chains>.pdb
                shared site      'ambiguous' when another ligand of the same entry with a different CCD code shares
                                 >= 50 % of the residues of the smaller pocket or has a heavy atom within 4.0 A of a
                                 heavy atom of the site's ligand (every ligand file of the entry counts)
                chain sequences  protein.pdb ATOM records, one letter per (chain, resSeq+iCode), non-standard -> X,
                                 chains of >= 10 residues -> db_chains.fasta (database entries)
  clusters    sequence clusters of the evaluation queries, the unit of the cross-validation folds and of the
              bootstrap (Supplementary Section S2): chain sequences of every query protein -> eval_chains.fasta;
              all-vs-all MMseqs2 search (-e 1e-3); chains linked at fident >= 0.5 (0.3 for the sensitivity) over
              >= 50 % of the shorter chain, clusters = connected components; a pocket belongs to the cluster of
              its main chain (the next chain by pocket residues when the main chain has < 10 residues).
              The search shipped in data/query_sets/ is reused unless --rerun-search.
  db-search   MMseqs2 easy-cluster of the database chains at 80 %, 50 % and 30 % identity (-c 0.8 --cov-mode 0; the
              80 % and 30 % clusters for the database composition and the site weights, the 50 % clusters for the
              chimeras of scripts/scaling.py), and the sensitive search of the evaluation chains against the database
              chains (-s 7.5, -e 1e-3), for the sequence-identity strata.
  ccd         CCD snapshot for the ligand fingerprints (network): downloads components.cif.gz once and keeps one
              SMILES per component (SMILES / OpenEye > SMILES / CACTVS > SMILES_CANONICAL / OpenEye >
              SMILES_CANONICAL / CACTVS > SMILES / ACDLabs). The snapshot of the paper is the one of asset
              reference_data ($LENSEEK_REFERENCE_DIR/ccd_smiles.tsv, ccd_snapshot.json); a new download is newer.

Inputs: $LENSEEK_EVAL_DIR (q_paths.txt, db_paths.txt), $LENSEEK_QUERY_ROOT and $LENSEEK_BSDB_ROOT (structures,
BsitePDB layout), $LENSEEK_STRAT_DIR, $LENSEEK_REFERENCE_DIR (asset reference_data: ccd_smiles.tsv,
db_ligand_codes.tsv), MMseqs2 ($MMSEQS, else mmseqs on PATH; the paper used 18.8cc5c), Foldseek ($FOLDSEEK, else
foldseek; the paper used 9.427df8a).
Outputs: $LENSEEK_OUT/site_annotation/query_sets/ (primary_exclude_pdb.txt, eval_chains.fasta, eval_vs_eval.m8,
seq_clusters_50.tsv, seq_clusters_30.tsv), $LENSEEK_OUT/site_annotation/protein_strata.tsv (pdb, maxid_either50,
stratum_either50, besthit_either50), $LENSEEK_OUT/site_annotation/query_strata.csv (q_idx, query_key, max_tm_train,
structure_stratum, ligand_ccd, max_dice_train, ligand_stratum; the work files in site_annotation/strata/),
$LENSEEK_OUT/cache/site_annot/ (query_sites.tsv, db_sites.tsv, db_chains.fasta, db80_cluster.tsv, db50_cluster.tsv,
db30_cluster.tsv, eval_vs_db_sens.m8, meta.json), $LENSEEK_OUT/cache/ccd/ (ccd_smiles.tsv, ccd_snapshot.json). The
shipped data/query_sets/ files were made with query-sets and clusters.

    python scripts/site_annotation.py query-sets
    python scripts/site_annotation.py identity-strata
    python scripts/site_annotation.py strata [--train-db <Foldseek database of the database entries>]
    python scripts/site_annotation.py annotate
    python scripts/site_annotation.py clusters
    python scripts/site_annotation.py db-search
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import AA, REPO, chain_seqs, die, require_path, single_linkage, site_chain, site_key  # noqa: E402
from lenseek.assets import local_path  # noqa: E402

QUERY_SETS = os.path.join(REPO, "data", "query_sets")
OUT_ROOT = os.path.abspath(os.environ.get("LENSEEK_OUT") or "out")
CACHE = os.path.join(OUT_ROOT, "cache")
ANNOT = os.path.join(CACHE, "site_annot")
OUT = os.path.join(OUT_ROOT, "site_annotation")
MMSEQS = os.environ.get("MMSEQS") or "mmseqs"
FOLDSEEK = os.environ.get("FOLDSEEK") or "foldseek"
REFERENCE_DIR = os.environ.get("LENSEEK_REFERENCE_DIR") or str(local_path("reference_data") or "")
M8_COLS = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"
OVERLAP, CONTACT = 0.5, 4.0                 # shared-site rule
CLUSTER_LEVELS, CLUSTER_MIN_COV = (50, 30), 0.50
DB_LEVELS = (80, 50, 30)                    # database chain clusters
POCKET_RE = re.compile(r"ligand_pocket_(\d+)_([^.]+)\.pdb$")
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
CCD_PRIORITY = [("SMILES", "OpenEye OEToolkits"), ("SMILES", "CACTVS"), ("SMILES_CANONICAL", "OpenEye OEToolkits"),
                ("SMILES_CANONICAL", "CACTVS"), ("SMILES", "ACDLabs")]
STANDARD = set(AA) - {"MSE", "SEC", "PYL"}  # the 20 standard residues
TM_LABEL = {"lt0.5": "TM < 0.5", "0.5to0.8": "TM 0.5-0.8", "ge0.8": "TM >= 0.8"}


# ---------------------------------------------------------------- helpers
def mmseqs_version(binary):
    return subprocess.run([binary, "version"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- query sets
def cmd_query_sets(a):
    m8 = require_path(a.m8, "--m8", "LENSEEK_STRAT_DIR")
    best = collections.defaultdict(float)
    for line in open(m8):
        f = line.rstrip("\n").split("\t")
        if float(f[6]) >= 0.80:
            pdb = f[0].split("_")[0].lower()
            best[pdb] = max(best[pdb], float(f[2]))
    hit = sorted(p for p, v in best.items() if v >= 0.80)
    out = os.path.join(a.out, "primary_exclude_pdb.txt")
    os.makedirs(a.out, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# evaluation proteins with max fident >= 0.8 at qcov >= 0.8 vs training "
                 f"(source {os.path.basename(m8)}); excluded from the primary query set\n")
        for p in hit:
            fh.write(f"{p}\t# {best[p]:.3f}\n")
    print(f"{len(hit)} proteins -> {out}: {hit}")


# ---------------------------------------------------------------- identity, structure and ligand strata
def identity_stratum(m):
    if m is None or m < 0.3:
        return "<30"
    return "30-40" if m < 0.4 else "40-50" if m < 0.5 else "50-80" if m < 0.8 else "80-95" if m < 0.95 else ">=95"


def cmd_identity_strata(a):
    m8 = require_path(a.m8, "--m8", "LENSEEK_STRAT_DIR")
    root = require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT")
    best = {}
    for line in open(m8):
        q, t, fid, _, _, _, qc, tc, _, bits = line.rstrip("\n").split("\t")
        if float(qc) >= 0.5 or float(tc) >= 0.5:
            pdb, k = q.split("_")[0], (float(fid), float(bits), t)
            if pdb not in best or k > best[pdb][0]:
                best[pdb] = (k, q)
    proteins = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "protein_strata.tsv")
    with open(out, "w") as f:
        f.write("pdb\tmaxid_either50\tstratum_either50\tbesthit_either50\n")
        for p in proteins:
            if p in best:
                (fid, _, t), q = best[p]
                f.write(f"{p}\t{fid:.3f}\t{identity_stratum(fid)}\t{q}>{t}\n")
            else:
                f.write(f"{p}\t\t<30\t\n")
    n = collections.Counter(identity_stratum(best[p][0][0] if p in best else None) for p in proteins)
    print(f"{len(proteins)} proteins -> {out}: {dict(sorted(n.items()))}")


def cmd_strata(a):
    import holdout   # the strata functions of the later PDB entries (Foldseek search, ligand novelty)
    ev = require_path(a.eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
    root = require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT")
    ref = require_path(a.reference_dir, "--reference-dir", "LENSEEK_REFERENCE_DIR")
    sd, keys = os.path.join(OUT, "strata"), [site_key(p) for p in open(os.path.join(ev, "q_paths.txt")).read().split()]
    os.makedirs(sd, exist_ok=True)
    man = os.path.join(sd, "sites.tsv")
    with open(man, "w") as f:
        f.write("site_id\tpdbid\tpocket_chains\tligand_ccd\n")
        for i, k in enumerate(keys):
            pdb, name = k.split("/")
            chains = []
            for l in open(os.path.join(root, k)):
                if l.startswith("ATOM") and l[17:20] in STANDARD and l[21] not in chains:
                    chains.append(l[21])
            m = POCKET_RE.search(name)
            code = ligand(os.path.join(root, pdb, f"ligand_{m.group(1)}_{m.group(2)}.pdb"))[0] or ""
            f.write(f"{i}\t{pdb}\t{''.join(chains)}\t{code}\n")
    std = os.path.join(sd, "query_std")
    for pdb in sorted({k.split("/")[0] for k in keys}):    # standard columns: a blank insertion code at column 27
        os.makedirs(os.path.join(std, pdb), exist_ok=True)
        with open(os.path.join(root, pdb, "protein.pdb")) as f, open(os.path.join(std, pdb, "protein.pdb"), "w") as g:
            for l in f:
                if l.startswith("ATOM"):
                    l = l.rstrip("\n")
                    g.write(l[:26] + " " + l[26:] + "\n")
                elif l.startswith(("TER", "END")):
                    g.write(l)
    tdb = a.train_db
    if not tdb:
        ids = sorted({site_key(p).split("/")[0] for p in open(os.path.join(ev, "db_paths.txt")).read().split()})
        tdb = holdout.train_foldseek_db(ids, require_path(a.bsdb_root, "--bsdb-root", "LENSEEK_BSDB_ROOT"), sd,
                                        a.foldseek, a.nproc)
    holdout.strata_structure(argparse.Namespace(man=man, root=std, tdb=tdb, work=os.path.join(sd, "foldseek"),
                                                out=os.path.join(sd, "structure.tsv"), threads=str(a.threads),
                                                foldseek=a.foldseek))
    codes = os.path.join(sd, "db_ligand_codes.tsv")       # the database CCDs in the layout strata_ligand reads
    db = pd.read_csv(os.path.join(ref, "db_ligand_codes.tsv"), sep="\t", dtype=str, keep_default_na=False)
    db.assign(set="db").rename(columns={"code": "ccd"})[["set", "idx", "ccd"]].to_csv(codes, sep="\t", index=False)
    holdout.strata_ligand(argparse.Namespace(man=man, codes_path=codes, cache_path=os.path.join(ref, "ccd_smiles.tsv"),
                                             out=os.path.join(sd, "ligand.tsv"),
                                             fetched_path=os.path.join(sd, "fetched_smiles.tsv")))
    s, g = (pd.read_csv(os.path.join(sd, f), sep="\t", dtype=str, keep_default_na=False)
            for f in ("structure.tsv", "ligand.tsv"))
    t = pd.DataFrame(dict(q_idx=range(len(keys)), query_key=keys, max_tm_train=s.max_tm_train,
                          structure_stratum=s.structure_stratum.map(TM_LABEL), ligand_ccd=g.ligand_ccd,
                          max_dice_train=g.max_dice_train, ligand_stratum=g.ligand_stratum))
    t.to_csv(os.path.join(OUT, "query_strata.csv"), index=False)
    print(t.structure_stratum.value_counts().to_dict(), t.ligand_stratum.value_counts().to_dict(), "->",
          os.path.join(OUT, "query_strata.csv"))


# ---------------------------------------------------------------- site annotation
def residues(f):
    res = collections.OrderedDict()
    for l in open(f, errors="replace"):
        if l.startswith(("ATOM", "HETATM")):
            res[(l[21], l[22:26].strip(), l[17:20].strip())] = None
    return list(res)


def ligand(f):
    """(CCD code of the first HETATM, heavy-atom coordinates) of a ligand file."""
    code, xyz = None, []
    if not os.path.exists(f):
        return None, np.zeros((0, 3))
    for l in open(f, errors="replace"):
        if l.startswith("HETATM"):
            if code is None:
                code = l[17:20].strip()
            el = l[76:78].strip() or l[12:16].strip()[:1]
            if el.upper() == "H":
                continue
            try:
                xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            except ValueError:
                pass
    return code, np.array(xyz).reshape(-1, 3)


def entry(args):
    root, pdb, want_seq = args
    d = os.path.join(root, pdb)
    sites = {}
    for f in sorted(glob.glob(os.path.join(d, "ligand_pocket_*.pdb"))):
        m = POCKET_RE.search(os.path.basename(f))
        if not m:
            continue
        res = residues(f)
        code, xyz = ligand(os.path.join(d, f"ligand_{m.group(1)}_{m.group(2)}.pdb"))
        cnt = collections.Counter(r[0] for r in res)
        main = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cnt else ""
        sites[os.path.basename(f)] = dict(res=set(res), code=code, xyz=xyz, cnt=cnt, main=main)
    rows = {}
    names = list(sites)
    for a in names:
        A = sites[a]
        amb, why, partner, best = False, "", "", 0.0
        for b in names:
            if b == a:
                continue
            B = sites[b]
            if not A["code"] or not B["code"] or A["code"] == B["code"]:
                continue
            ov = len(A["res"] & B["res"]) / max(1, min(len(A["res"]), len(B["res"])))
            contact = False
            if len(A["xyz"]) and len(B["xyz"]):
                dd = ((A["xyz"][:, None, :] - B["xyz"][None, :, :]) ** 2).sum(-1)
                contact = bool(dd.min() <= CONTACT ** 2)
            if ov >= OVERLAP or contact:
                r = ("overlap" if ov >= OVERLAP else "") + ("+" if ov >= OVERLAP and contact else "") + \
                    ("contact" if contact else "")
                if not amb or ov > best:
                    amb, why, partner, best = True, r, f"{b}:{B['code']}", ov
        rows[a] = dict(main_chain=A["main"], n_res=len(A["res"]), n_res_main=A["cnt"].get(A["main"], 0),
                       chains=",".join(f"{c}:{n}" for c, n in sorted(A["cnt"].items())), ligand=A["code"] or "",
                       ambiguous=int(amb), amb_reason=why, amb_partner=partner, amb_overlap=round(best, 3))
    seqs = chain_seqs(os.path.join(d, "protein.pdb")) if want_seq and os.path.exists(os.path.join(d, "protein.pdb")) \
        else {}
    return pdb, rows, seqs


def annotate(root, keys, want_seq, workers, fasta=None):
    pdbs = sorted({k.split("/")[0] for k in keys})
    out, n_seq = {}, 0
    fh = open(fasta, "w") if fasta else None
    t0 = time.time()
    with Pool(workers) as p:
        for i, (pdb, rows, seqs) in enumerate(p.imap_unordered(entry, [(root, x, want_seq) for x in pdbs],
                                                               chunksize=16)):
            for name, r in rows.items():
                out[f"{pdb}/{name}"] = r
            for c, s in seqs.items():
                fh.write(f">{pdb}_{c}\n{s}\n")
                n_seq += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(pdbs)} entries, {time.time() - t0:.0f} s", flush=True)
    if fh:
        fh.close()
    missing = [k for k in keys if k not in out]
    if missing:
        die(f"{len(missing)} sites have no pocket file under {root}, e.g. {missing[:3]}")
    return out, n_seq


def write_sites(path, keys, ann):
    cols = ["main_chain", "n_res", "n_res_main", "chains", "ligand", "ambiguous", "amb_reason", "amb_partner",
            "amb_overlap"]
    with open(path, "w") as f:
        f.write("idx\tkey\t" + "\t".join(cols) + "\n")
        for i, k in enumerate(keys):
            r = ann[k]
            f.write(f"{i}\t{k}\t" + "\t".join(str(r[c]) for c in cols) + "\n")


def cmd_annotate(a):
    require_path(a.eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
    require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT")
    require_path(a.db_root, "--db-root", "LENSEEK_BSDB_ROOT")
    os.makedirs(ANNOT, exist_ok=True)
    q_keys, db_keys = ([site_key(p) for p in open(os.path.join(a.eval_dir, f)).read().split()]
                       for f in ("q_paths.txt", "db_paths.txt"))
    meta = dict(rules=dict(main_chain="most pocket residues, ties alphabetical", overlap_frac=OVERLAP,
                           contact_A=CONTACT, residue_key="(chain, resSeq cols 23-26, resName); no iCode column"),
                eval_dir=os.path.abspath(a.eval_dir), query_root=os.path.abspath(a.query_root),
                db_root=os.path.abspath(a.db_root), date=time.strftime("%Y-%m-%d %H:%M:%S"))
    qa, _ = annotate(a.query_root, q_keys, False, a.workers)
    write_sites(os.path.join(ANNOT, "query_sites.tsv"), q_keys, qa)
    meta["query"] = dict(n_sites=len(q_keys), n_ambiguous=int(sum(qa[k]["ambiguous"] for k in q_keys)))
    print("queries:", meta["query"], flush=True)
    da, n_seq = annotate(a.db_root, db_keys, True, a.workers, os.path.join(ANNOT, "db_chains.fasta"))
    write_sites(os.path.join(ANNOT, "db_sites.tsv"), db_keys, da)
    meta["db"] = dict(n_sites=len(db_keys), n_ambiguous=int(sum(da[k]["ambiguous"] for k in db_keys)),
                      n_chains_fasta=n_seq)
    print("db:", meta["db"], flush=True)
    json.dump(meta, open(os.path.join(ANNOT, "meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- sequence clusters of the queries
def cmd_clusters(a):
    require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT")
    qs = a.out
    os.makedirs(qs, exist_ok=True)
    fasta = os.path.join(qs, "eval_chains.fasta")
    root = a.query_root
    pdbs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "protein.pdb")))
    with open(fasta, "w") as f:
        for p in pdbs:
            for c, s in chain_seqs(os.path.join(root, p, "protein.pdb")).items():
                f.write(f">{p}_{c}\n{s}\n")
    m8 = os.path.join(qs, "eval_vs_eval.m8")
    shipped = os.path.join(QUERY_SETS, "eval_vs_eval.m8")
    if a.rerun_search:
        ver = mmseqs_version(a.mmseqs)
        cmd = [a.mmseqs, "easy-search", fasta, fasta, m8, os.path.join(qs, "mmseqs_tmp"), "--format-output",
               M8_COLS, "-e", "1e-3", "--threads", "4"]
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.rmtree(os.path.join(qs, "mmseqs_tmp"), ignore_errors=True)
        rel = ["mmseqs"] + [os.path.relpath(c, qs) if os.path.isabs(c) else c for c in cmd[1:]]
        open(os.path.join(qs, "eval_vs_eval.cmd"), "w").write(f"# MMseqs2 {ver} (run in query_sets/)\n"
                                                              + " ".join(rel) + "\n")
    elif not os.path.exists(m8):
        if not os.path.exists(shipped):
            die(f"{shipped} missing: run with --rerun-search")
        m8 = shipped
    print(f"MMseqs2 search: {m8}")
    seqs = {}
    for l in open(fasta):
        if l.startswith(">"):
            name = l[1:].strip()
        else:
            seqs[name] = l.strip()
    hits = [l.split("\t")[:6] for l in open(m8)]
    qsites_f = os.path.join(ANNOT, "query_sites.tsv")
    if not os.path.exists(qsites_f):
        die(f"{qsites_f} missing: run the annotate step first")
    qsites = pd.read_csv(qsites_f, sep="\t", dtype=str, keep_default_na=False)
    for pct in CLUSTER_LEVELS:
        root_of = single_linkage(seqs, hits, pct, CLUSTER_MIN_COV)
        rows = []
        for _, r in qsites.iterrows():
            pdb = r.key.split("/")[0]
            chain = site_chain(r.key, r.main_chain, r.chains, seqs) or f"{pdb}_{r.main_chain}"
            rows.append(dict(query_key=r.key, pdb=pdb, main_chain=r.main_chain, cluster_chain=chain,
                             root=root_of.get(chain, chain)))
        df = pd.DataFrame(rows)
        size = df.groupby("root").agg(n_q=("query_key", "size"), n_e=("pdb", "nunique"))
        roots = sorted(size.index, key=lambda r: (-size.n_q[r], r))
        name = {r: f"c{i:03d}" for i, r in enumerate(roots)}
        df["cluster"] = df.root.map(name)
        df["cluster_n_queries"] = df.root.map(size.n_q)
        df["cluster_n_entries"] = df.root.map(size.n_e)
        out = os.path.join(qs, f"seq_clusters_{pct}.tsv")
        with open(out, "w") as f:
            f.write(f"# chains linked when they align with fident >= {pct / 100} over >= {CLUSTER_MIN_COV} of the "
                    f"shorter chain (eval_vs_eval.m8, command in eval_vs_eval.cmd); single linkage; pocket -> cluster "
                    f"of its main chain (most pocket residues)\n")
            df.drop(columns="root").to_csv(f, sep="\t", index=False)
        split = df.groupby("pdb").cluster.nunique()
        print(f"{pct}%: {len(df)} query pockets -> {df.cluster.nunique()} clusters "
              f"({(size.n_q > 1).sum()} with > 1 pocket); entries split over > 1 cluster: "
              f"{list(split[split > 1].index)}; fallback main chains: "
              f"{int((df.cluster_chain != df.pdb + '_' + df.main_chain).sum())}")


# ---------------------------------------------------------------- database chains: clusters and search
def cmd_db_search(a):
    db = os.path.join(ANNOT, "db_chains.fasta")
    if not os.path.exists(db):
        die(f"{db} missing: run the annotate step first")
    if not os.path.exists(a.eval_fasta):
        die(f"--eval-fasta {a.eval_fasta} does not exist (written by the clusters step)")
    ver = mmseqs_version(a.mmseqs)
    steps = [(f"db{p}", [a.mmseqs, "easy-cluster", "db_chains.fasta", f"db{p}", f"tmp{p}", "--min-seq-id", f"{p / 100}",
                         "-c", "0.8", "--cov-mode", "0", "--threads", "4"], [f"tmp{p}", f"db{p}_all_seqs.fasta"])
             for p in DB_LEVELS]
    steps.append(("eval_vs_db_sens", [a.mmseqs, "easy-search", os.path.abspath(a.eval_fasta), "db_chains.fasta",
                                      "eval_vs_db_sens.m8", "tmp_sens", "-s", "7.5", "-e", "1e-3", "--max-seqs", "100000",
                                      "--threads", "3", "--format-output", M8_COLS], ["tmp_sens"]))
    for name, cmd, tmp in steps:
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, capture_output=True, cwd=ANNOT)
        for t in tmp:
            p = os.path.join(ANNOT, t)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        open(os.path.join(ANNOT, f"{name}.cmd"), "w").write(" ".join(cmd) + f"\n{ver}\n")


# ---------------------------------------------------------------- CCD snapshot
def cmd_ccd(a):
    import gemmi
    d = os.path.join(CACHE, "ccd")
    os.makedirs(d, exist_ok=True)
    gz = os.path.join(d, "components.cif.gz")
    if not os.path.exists(gz):
        subprocess.run(["curl", "-sS", "-o", gz, "-D", os.path.join(d, "components.headers.txt"), CCD_URL], check=True)
    hdr = open(os.path.join(d, "components.headers.txt")).read() if os.path.exists(
        os.path.join(d, "components.headers.txt")) else ""
    last_mod = next((l.split(":", 1)[1].strip() for l in hdr.splitlines() if l.lower().startswith("last-modified")),
                    "unknown")
    md5 = hashlib.md5(open(gz, "rb").read()).hexdigest()
    doc = gemmi.cif.read(gz)
    rows, n_none = [], 0
    for block in doc:
        t = block.find("_pdbx_chem_comp_descriptor.", ["type", "program", "descriptor"])
        got = {}
        for r in t:
            got.setdefault((gemmi.cif.as_string(r[0]), gemmi.cif.as_string(r[1])), gemmi.cif.as_string(r[2]))
        smi, src = "", ""
        for key in CCD_PRIORITY:
            if got.get(key):
                smi, src = got[key], "/".join(key)
                break
        n_none += not smi
        rows.append((block.name, smi, src))
    df = pd.DataFrame(rows, columns=["comp_id", "smiles", "source"])
    df.to_csv(os.path.join(d, "ccd_smiles.tsv"), sep="\t", index=False)
    meta = dict(url=CCD_URL, server_last_modified=last_mod,
                downloaded=datetime.datetime.fromtimestamp(os.path.getmtime(gz), datetime.timezone.utc).isoformat(),
                md5=md5, n_components=len(df), n_without_smiles=n_none,
                selection="first available of " + " > ".join("/".join(k) for k in CCD_PRIORITY),
                gemmi=gemmi.__version__)
    json.dump(meta, open(os.path.join(d, "ccd_snapshot.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    qroot = dict(default=os.environ.get("LENSEEK_QUERY_ROOT"), help="query structures (default $LENSEEK_QUERY_ROOT)")
    mm = dict(default=MMSEQS, help="MMseqs2 binary (default $MMSEQS, else mmseqs)")
    strat = os.environ.get("LENSEEK_STRAT_DIR")
    p = sub.add_parser("query-sets", help="primary_exclude_pdb.txt")
    p.add_argument("--m8", default=os.path.join(strat, "eval_vs_train.m8") if strat else None,
                   help="MMseqs2 search of the evaluation vs the training chains "
                        "(default $LENSEEK_STRAT_DIR/eval_vs_train.m8)")
    p.add_argument("--out", default=os.path.join(OUT, "query_sets"))
    ev = dict(default=os.environ.get("LENSEEK_EVAL_DIR"),
              help="evaluation outputs with q_paths.txt, db_paths.txt (default $LENSEEK_EVAL_DIR)")
    p = sub.add_parser("identity-strata", help="protein_strata.tsv")
    p.add_argument("--m8", default=os.path.join(strat, "eval_vs_train_uncapped.m8") if strat else None,
                   help="MMseqs2 search of the evaluation vs the training chains "
                        "(default $LENSEEK_STRAT_DIR/eval_vs_train_uncapped.m8)")
    p.add_argument("--query-root", **qroot)
    p = sub.add_parser("strata", help="query_strata.csv")
    p.add_argument("--eval-dir", **ev)
    p.add_argument("--query-root", **qroot)
    p.add_argument("--reference-dir", default=REFERENCE_DIR,
                   help="ccd_smiles.tsv, db_ligand_codes.tsv (default $LENSEEK_REFERENCE_DIR, else the asset)")
    p.add_argument("--train-db", default=None, help="Foldseek database of the chains of the database entries "
                                                    "(default: built from --bsdb-root, as scripts/holdout.py strata)")
    p.add_argument("--bsdb-root", default=os.environ.get("LENSEEK_BSDB_ROOT"),
                   help="database structures (default $LENSEEK_BSDB_ROOT)")
    p.add_argument("--foldseek", default=FOLDSEEK, help="Foldseek binary (default $FOLDSEEK, else foldseek)")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nproc", type=int, default=8)
    p = sub.add_parser("annotate", help="query_sites.tsv, db_sites.tsv, db_chains.fasta")
    p.add_argument("--eval-dir", **ev)
    p.add_argument("--query-root", **qroot)
    p.add_argument("--db-root", default=os.environ.get("LENSEEK_BSDB_ROOT"),
                   help="database structures (default $LENSEEK_BSDB_ROOT)")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("clusters", help="eval_chains.fasta, seq_clusters_{50,30}.tsv")
    p.add_argument("--query-root", **qroot)
    p.add_argument("--mmseqs", **mm)
    p.add_argument("--rerun-search", action="store_true",
                   help="run the MMseqs2 search instead of reusing data/query_sets/eval_vs_eval.m8")
    p.add_argument("--out", default=os.path.join(OUT, "query_sets"))
    p = sub.add_parser("db-search", help="db{80,50,30}_cluster.tsv, eval_vs_db_sens.m8")
    p.add_argument("--eval-fasta", default=os.path.join(QUERY_SETS, "eval_chains.fasta"))
    p.add_argument("--mmseqs", **mm)
    sub.add_parser("ccd", help="CCD snapshot (network)")
    a = ap.parse_args()
    {"query-sets": cmd_query_sets, "identity-strata": cmd_identity_strata, "strata": cmd_strata,
     "annotate": cmd_annotate, "clusters": cmd_clusters, "db-search": cmd_db_search, "ccd": cmd_ccd}[a.cmd](a)


if __name__ == "__main__":
    main()
