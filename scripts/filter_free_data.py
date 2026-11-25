#!/usr/bin/env python
"""Training data without the ligand filters (Supplementary Section S1, 'Model trained without the ligand filters', and
the row 'Trained without the ligand filters' of Supplementary Table S14): the added sites, the merged feature store, and
the identity of the evaluation proteins to the entries the added sites bring in. analysis/filter_retrain.py reads the
retrained model's ligand metrics and the identity search.

Subcommands:
  sites     the site path of the training data with the ligand filter recorded but not applied, for every pocket file of
            BSDB_231103 outside the training set, in the order of that path: no_protein, no_embedding, min_pts
            (fewer than 10 standard residues), protein_parse_fail, no_ligand, structure_parse, featurize_failed,
            exception, success. Sites that pass with valid_ligand False are added; structures are read with the
            legacy reader of the training data (lenseek.io.pdb.parse_pdb_legacy), so they get the same inputs. Node
            features are rows of the per-chain embedding files (--embeddings, lenseek-embed).
  build     one feature store of the training sites and the added sites in site-key order (= the sorted pocket-path
            order of the training data), with a reload check of every array; train on it with examples/train.sh.
  identity  sequence identity of the evaluation proteins to the entries that only the filter-free data contains (no
            site in the search database): chain sequences with gemmi (standard amino acids, >= 10 residues), MMseqs2
            easy-search -s 7.5 --max-seqs 200000 -e 10 --min-seq-id 0.25.

Inputs: $LENSEEK_BSDB_ROOT (BsitePDB BSDB_231103), $LENSEEK_EVAL_DIR (q_paths.txt, db_paths.txt), data/query_sets/
(primary_exclude_pdb.txt, eval_chains.fasta), $MMSEQS (default mmseqs).
Outputs: $LENSEEK_OUT/filter_free_data/ (sites/, features/, identity/).

usage: python scripts/filter_free_data.py sites --embeddings EMB
       python scripts/filter_free_data.py build --base BASE
       python scripts/filter_free_data.py identity
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import REPO, entry_chain_seqs, site_key  # noqa: E402

QUERY_SETS = os.path.join(REPO, "data", "query_sets")
PRIMARY_EXCLUDE = os.path.join(QUERY_SETS, "primary_exclude_pdb.txt")
OUT = os.path.join(os.path.abspath(os.environ.get("LENSEEK_OUT") or "out"), "filter_free_data")
EVAL = os.environ.get("LENSEEK_EVAL_DIR", "")
COLS = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits".split(",")
PATTERN = "ligand_pocket*.pdb"
_W = {}


def primary_excluded():
    return {l.split()[0].lower() for l in open(PRIMARY_EXCLUDE) if l.strip() and not l.startswith("#")}


# ------------------------------------------------------------------ sites: the site path with the ligand filter off
def reason(lig):
    """First failing rule of valid_ligand: the exclusion lists, then '^\\d', then len < 2."""
    from lenseek.exclusion import CRYSTALLIZATION_AIDS_SET, GLYCANS_SET, LIGAND_EXCLUSION_SET
    if lig in CRYSTALLIZATION_AIDS_SET or lig in LIGAND_EXCLUSION_SET or lig in GLYCANS_SET:
        sub = [n for n, s in (("crystallization_aids", CRYSTALLIZATION_AIDS_SET), ("Ligand_exclusion", LIGAND_EXCLUSION_SET),
                              ("glycans", GLYCANS_SET)) if lig in s]
        return "exclusion_list", "+".join(sub)
    if re.search(r"^\d", lig):
        return "digit_start", "digit_start"
    if len(lig) < 2:
        return "len_lt_2", "len_lt_2"
    return "valid", "valid"


def has_chain_ranges(protein_path):
    """At least one ATOM record with an integer residue number (the chain-range test of the legacy reader)."""
    try:
        with open(protein_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        int(line[22:26].strip())
                        return True
                    except ValueError:
                        continue
    except Exception:
        return False
    return False


def ligand_name(pocket_path):
    lig = pocket_path.replace("ligand_pocket_", "ligand_")
    if not os.path.exists(lig):
        return None
    try:
        with open(lig) as f:
            for line in f:
                if line.startswith("HETATM"):
                    return line[17:20].strip()
    except Exception:
        return None
    return None


FAILURES = [("missing embedding file", "missing_embedding"), ("not in the protein sequence map", "chain_not_in_seq_map"),
            ("length mismatch", "length_mismatch"), ("of the pocket is not in the protein sequence map",
                                                     "residue_not_in_seq_map"),
            ("out of bounds", "index_out_of_bounds"), ("not found in the protein structure", "residue_not_in_structure"),
            ("missing backbone atom", "missing_backbone_atom")]


def failure_code(msg):
    """Short code of a featurize_site_legacy failure (the last matching phrase wins)."""
    code = "other"
    for phrase, c in FAILURES:
        if phrase in msg:
            code = c
    return code


def _init(root, emb):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    _W.update(root=root, emb=emb, cache=(None, None))


def _structure(protein):
    from lenseek.io.pdb import parse_pdb_legacy
    path, st = _W["cache"]
    if path != protein:
        st = parse_pdb_legacy(protein)
        _W["cache"] = (protein, st)
    return st


def site_record(key):
    """Status of one pocket file along the site path of the training data, the ligand filter recorded only."""
    from lenseek.config import MIN_RESIDUES
    from lenseek.exclusion import valid_ligand
    from lenseek.featurize import featurize_site_legacy, legacy_resdefs
    root, emb = _W["root"], _W["emb"]
    pocket = os.path.join(root, key)
    entry = key.split("/")[0]
    rec = dict(key=key, entry=entry, status="", detail="", ligand="", valid_ligand="", n_res=0)
    try:
        protein = os.path.join(os.path.dirname(pocket), "protein.pdb")
        if not os.path.exists(protein):
            rec["status"] = "no_protein"
            return rec
        if not glob.glob(os.path.join(emb, f"{entry}_chain_*.npy")):
            rec["status"] = "no_embedding"
            return rec
        resdefs = legacy_resdefs(pocket)
        rec["n_res"] = len(resdefs)
        if len(resdefs) < MIN_RESIDUES:
            rec["status"] = "min_pts"
            return rec
        if not has_chain_ranges(protein):
            rec["status"] = "protein_parse_fail"
            return rec
        lig = ligand_name(pocket)
        if lig is None:
            rec["status"] = "no_ligand"
            return rec
        rec["ligand"] = lig
        rec["valid_ligand"] = int(valid_ligand(lig))
        st = _structure(protein)
        if st is None:
            rec["status"] = "structure_parse"
            return rec
        try:
            sf = featurize_site_legacy(pocket, protein, emb, structure=st)
        except Exception as e:   # every featurization error is one status
            rec["status"] = "featurize_failed"
            rec["detail"] = failure_code(str(e))
            return rec
        rec["status"] = "success"
        rec["arrays"] = (sf.h, sf.T, sf.R)
    except Exception as e:
        rec["status"] = "exception"
        rec["detail"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def open_store(path):
    """(site ids, n_res, site(i) -> (h, T, R), meta) of a feature store directory or a training pickle."""
    from lenseek.data import FeatureStore, load_legacy_pickle
    if os.path.isdir(path):
        fs = FeatureStore(path)
        return list(fs.site_ids), np.diff(fs.offsets), fs.site, dict(fs.meta)
    ds = load_legacy_pickle(path)
    ids = [f"{os.path.basename(os.path.dirname(p))}/{os.path.splitext(os.path.basename(p))[0]}" for p in ds.path]
    return ids, np.array([len(d[0]) for d in ds.data]), ds.site, dict(source=os.path.basename(path))


def write_store(path, sites, meta):
    """sites: list of (site_id, n_res, getter() -> (h, T, R), source). Streams the arrays into the .npy files."""
    os.makedirs(path)
    sizes = np.array([s[1] for s in sites], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    width = int(np.asarray(sites[0][2]()[0]).shape[1])
    fmt = np.lib.format
    H = fmt.open_memmap(os.path.join(path, "h.npy"), "w+", np.float32, (int(offsets[-1]), width))
    T = fmt.open_memmap(os.path.join(path, "T.npy"), "w+", np.float32, (int(offsets[-1]), 3))
    R = fmt.open_memmap(os.path.join(path, "R.npy"), "w+", np.float32, (int(offsets[-1]), 3, 3))
    for i, (_, n, get, _) in enumerate(sites):
        h, t, r = get()
        a, b = offsets[i], offsets[i + 1]
        if len(h) != n or len(t) != n or len(r) != n:
            raise SystemExit(f"site {sites[i][0]}: {len(h)} rows, expected {n}")
        H[a:b], T[a:b], R[a:b] = h, t, r
    for x in (H, T, R):
        x.flush()
    del H, T, R
    np.save(os.path.join(path, "offsets.npy"), offsets)
    with open(os.path.join(path, "sites.tsv"), "w") as f:
        f.write("index\tsite_id\tn_res\tsource\n")
        for i, (s, n, _, src) in enumerate(sites):
            f.write(f"{i}\t{s}\t{n}\t{src}\n")
    m = dict(meta)
    m.update(format="lenseek-feature-store", version=1, n_sites=len(sites), n_rows=int(offsets[-1]), node_dim=width)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(m, f, indent=1)


def cmd_sites(a):
    t0 = time.time()
    root = os.path.abspath(a.root or os.environ.get("LENSEEK_BSDB_ROOT") or "")
    out = a.out or os.path.join(OUT, "sites")
    if not os.path.isdir(root):
        sys.exit("[error] --root not given and $LENSEEK_BSDB_ROOT is not a directory")
    if os.path.exists(os.path.join(out, "added")):
        sys.exit(f"[error] {out}/added exists: not overwritten")
    db_keys = [site_key(p) for p in open(a.db_paths).read().split()]
    db_set = set(db_keys)
    if len(db_set) != len(db_keys):
        sys.exit("[error] duplicate sites in --db-paths")
    if a.entries:
        ents = sorted({l.split()[0] for l in open(a.entries) if l.strip() and not l.startswith("#")})
        pockets = sorted(p for e in ents for p in glob.glob(os.path.join(root, e, PATTERN)))
    else:
        pockets = sorted(glob.glob(os.path.join(root, "*", PATTERN)))
    keys = [site_key(p) for p in pockets]
    cand = [k for k in keys if k not in db_set]
    print(f"pocket files {len(keys)}; training sites {len(db_keys)}; candidates {len(cand)}", flush=True)
    os.makedirs(out, exist_ok=True)
    with Pool(a.workers, initializer=_init, initargs=(root, os.path.abspath(a.embeddings))) as p:
        recs = list(p.imap(site_record, cand, chunksize=16))
    cols = ["key", "entry", "status", "detail", "ligand", "valid_ligand", "n_res"]
    with open(os.path.join(out, "site_status.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in recs:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    db_entries = {k.split("/")[0].lower() for k in db_keys}
    added = [r for r in recs if r["status"] == "success" and r["valid_ligand"] == 0]
    if a.queries:
        q_entries = {site_key(p).split("/")[0].lower() for p in open(a.queries).read().split()}
        leak = sorted({r["entry"].lower() for r in added} & q_entries)
        if leak:
            sys.exit(f"[error] added sites from evaluation-query entries: {leak}")
    flow = lambda lig: "digit_start" if re.search(r"^\d", lig) else reason(lig)[0]  # noqa: E731
    with open(os.path.join(out, "added_sites.tsv"), "w") as f:
        f.write("key\tentry\tligand\treason\treason_detail\treason_flow\tn_res\tentry_in_db\n")
        for r in added:
            rsn, sub = reason(r["ligand"])
            f.write(f"{r['key']}\t{r['entry']}\t{r['ligand']}\t{rsn}\t{sub}\t{flow(r['ligand'])}\t{r['n_res']}\t"
                    f"{int(r['entry'].lower() in db_entries)}\n")
    if added:
        sites = [(os.path.splitext(r["key"])[0], len(r["arrays"][0]), (lambda r=r: r["arrays"]), "added")
                 for r in added]
        write_store(os.path.join(out, "added"), sites, dict(root=root, parser="legacy",
                                                            filters="training-data site path, ligand filter off"))
    cnt = Counter((r["status"], r["valid_ligand"]) for r in recs)
    summ = dict(root=root, embeddings=os.path.abspath(a.embeddings), n_pocket_files=len(keys),
                n_training_sites=len(db_keys), n_candidates=len(cand),
                counts=[dict(status=s, valid_ligand=v, n=c) for (s, v), c in sorted(cnt.items(), key=str)],
                n_added=len(added), added_entries=len({r["entry"] for r in added}),
                added_from_entries_without_training_site=sum(r["entry"].lower() not in db_entries for r in added),
                added_by_reason=dict(Counter(reason(r["ligand"])[0] for r in added)),
                added_by_reason_flow=dict(Counter(flow(r["ligand"]) for r in added)),
                featurize_failed_detail=dict(Counter(r["detail"] for r in recs if r["status"] == "featurize_failed")),
                seconds=round(time.time() - t0))
    json.dump(summ, open(os.path.join(out, "sites_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "counts"}, indent=1))


def cmd_build(a):
    from lenseek.data import FeatureStore
    out = a.out or os.path.join(OUT, "features")
    if os.path.exists(out):
        sys.exit(f"[error] {out} exists: not overwritten")
    ids, n_base, get_base, meta = open_store(a.base)
    add_dir = os.path.join(a.added or os.path.join(OUT, "sites"), "added")
    add = FeatureStore(add_dir)
    add_ids, n_add = list(add.site_ids), np.diff(add.offsets)
    w_base, w_add = int(np.asarray(get_base(0)[0]).shape[1]), int(add.h.shape[1])
    if w_base != w_add:
        sys.exit(f"[error] node-feature width {w_base} of the training sites != {w_add} of the added sites")
    clash = set(ids) & set(add_ids)
    if clash:
        sys.exit(f"[error] {len(clash)} added sites are already training sites, e.g. {sorted(clash)[:3]}")
    items = [(s + ".pdb", s, int(n), (lambda i=i: get_base(i)), "base") for i, (s, n) in enumerate(zip(ids, n_base))]
    items += [(s + ".pdb", s, int(n), (lambda i=i: add.site(i)), "added") for i, (s, n) in enumerate(zip(add_ids, n_add))]
    items.sort(key=lambda x: x[0])   # pocket-file key order = the sorted path order of the training data
    write_store(out, [(s, n, g, src) for _, s, n, g, src in items],
                dict(base=os.path.abspath(a.base), added=os.path.abspath(add_dir), base_meta=meta,
                     n_base=len(ids), n_added=len(add_ids)))
    m = FeatureStore(out)   # reload check: order, counts, every array, finite values
    src = [l.rstrip("\n").split("\t")[3] for l in open(os.path.join(out, "sites.tsv"))][1:]
    keys = [s + ".pdb" for s in m.site_ids]
    assert len(m) == len(ids) + len(add_ids) and keys == sorted(keys) and len(set(keys)) == len(keys)
    bpos = {s: i for i, s in enumerate(ids)}
    apos = {s: i for i, s in enumerate(add_ids)}
    n_equal = 0
    for i, s in enumerate(m.site_ids):
        h, t, r = m.site(i)
        h0, t0, r0 = get_base(bpos[s]) if src[i] == "base" else add.site(apos[s])
        n_equal += bool(np.array_equal(h, h0) and np.array_equal(t, t0) and np.array_equal(r, r0))
    finite = all(bool(np.isfinite(x[a_:a_ + 200000]).all()) for x in (m.h, m.T, m.R)
                 for a_ in range(0, len(x), 200000))
    summ = dict(out=os.path.abspath(out), n_sites=len(m), n_base=len(ids), n_added=len(add_ids),
                n_rows=int(m.offsets[-1]), node_dim=int(m.h.shape[1]), sites_identical_to_source=n_equal,
                order_sorted=True, all_finite=finite)
    json.dump(summ, open(os.path.join(out, "build_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    if n_equal != len(m) or not finite:
        sys.exit("[error] reload check failed")


# ------------------------------------------------------------------ identity of the evaluation proteins to the added entries
def query_table(q_paths):
    qk = ["/".join(l.strip().split("/")[-2:]) for l in open(q_paths) if l.strip()]
    return pd.DataFrame({"query_key": qk, "protein": [k.split("/")[0].lower() for k in qk]})


def cmd_identity(a):
    if not a.root or not os.path.isdir(a.root):
        sys.exit("[error] --root not given and $LENSEEK_BSDB_ROOT is not a directory")
    T = pd.read_csv(a.added, sep="\t", dtype={"entry": str})
    new = sorted(T[T.entry_in_db == 0].entry.unique())
    os.makedirs(a.out, exist_ok=True)
    fa = os.path.join(a.out, "new_entries.fasta")
    n = 0
    with open(fa, "w") as f:
        for e in new:
            for name, s in entry_chain_seqs(a.root, e):
                f.write(f">{name}\n{s}\n")
                n += 1
    print(f"{len(new)} entries without a training site, {n} chains -> {fa}")
    m8, tmp = os.path.join(a.out, "eval_vs_new.m8"), os.path.join(a.out, "tmp")
    cmd = [os.environ.get("MMSEQS", "mmseqs"), "easy-search", a.eval_fasta, fa, m8, tmp, "--threads", str(a.threads),
           "-s", "7.5", "--max-seqs", "200000", "-e", "10", "--min-seq-id", "0.25", "--format-output", ",".join(COLS)]
    with open(os.path.join(a.out, "mmseqs.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    shutil.rmtree(tmp, ignore_errors=True)
    H = pd.read_csv(m8, sep="\t", header=None, names=COLS)
    H["protein"] = H["query"].str.split("_").str[0].str.lower()
    a_ = H[H.qcov >= 0.8].groupby("protein").fident.max().rename("max_fident_qcov08")
    b_ = H[(H.qcov >= 0.5) | (H.tcov >= 0.5)].groupby("protein").fident.max().rename("max_fident_cov05")
    P = pd.concat([a_, b_], axis=1).reset_index()
    P.to_csv(os.path.join(a.out, "eval_protein_vs_new_entries.tsv"), sep="\t", index=False)
    Q = query_table(a.queries).merge(P, on="protein", how="left")
    Qp = Q[~Q.protein.isin(primary_excluded())]
    res = dict(new_entries=len(new), new_entry_chains=n, sites_from_new_entries=int((T.entry_in_db == 0).sum()),
               hits=int(len(H)), primary_queries=int(len(Qp)),
               primary_queries_ge80_qcov08=int((Qp.max_fident_qcov08 >= 0.8).sum()),
               primary_queries_ge30_cov05=int((Qp.max_fident_cov05 >= 0.3).sum()),
               proteins_ge80_qcov08=sorted(P[P.max_fident_qcov08 >= 0.8].protein.tolist()))
    json.dump(res, open(os.path.join(a.out, "added_identity.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sites", help="site path with the ligand filter off; added sites and their features")
    p.add_argument("--root", default=None, help="BsitePDB BSDB_231103 (default $LENSEEK_BSDB_ROOT)")
    p.add_argument("--embeddings", required=True, help="directory of <entry>_chain_<C>.npy embeddings")
    p.add_argument("--db-paths", default=os.path.join(EVAL, "db_paths.txt"), help="training sites (db_paths.txt)")
    p.add_argument("--queries", default=os.path.join(EVAL, "q_paths.txt"),
                   help="q_paths.txt: stop if an added site comes from a query entry")
    p.add_argument("--entries", default=None, help="only the pocket files of these entries")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None, help="default $LENSEEK_OUT/filter_free_data/sites")
    p = sub.add_parser("build", help="merge the training sites and the added sites into one feature store")
    p.add_argument("--base", required=True, help="feature store or training pickle of the training sites")
    p.add_argument("--added", default=None, help="output folder of the sites step")
    p.add_argument("--out", default=None, help="new feature store (default $LENSEEK_OUT/filter_free_data/features)")
    p = sub.add_parser("identity", help="identity of the evaluation proteins to the entries the data adds")
    p.add_argument("--added", default=os.path.join(OUT, "sites", "added_sites.tsv"))
    p.add_argument("--out", default=os.path.join(OUT, "identity"))
    p.add_argument("--root", default=os.environ.get("LENSEEK_BSDB_ROOT"))
    p.add_argument("--queries", default=os.path.join(EVAL, "q_paths.txt"))
    p.add_argument("--eval-fasta", default=os.path.join(QUERY_SETS, "eval_chains.fasta"))
    p.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"sites": cmd_sites, "build": cmd_build, "identity": cmd_identity}[a.cmd](a)


if __name__ == "__main__":
    main()
