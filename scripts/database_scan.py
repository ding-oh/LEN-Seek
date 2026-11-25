#!/usr/bin/env python
"""Per-site checks of the BsitePDB files: the site path of the training data (Supplementary Figure S7 and Section S1),
the coordinate columns, and the latent-database patch of the standard database edition. analysis/database_derivation.py
turns the site checks into the reported counts.

sites   every binding site <root>/<entry>/ligand_pocket_<n>_<chains>.pdb of the training period (entries also found in
        --eval-period-root are skipped) and of --query-root, with the checks of the training-data code: CCD code of the
        ligand (first HETATM residue of ligand_<n>_<chains>.pdb) and the flags of lenseek.exclusion (exclusion lists,
        leading digit, one character, valid_ligand), standard residues (RESNAME + chain + %04d of the ATOM records), and
        featurization: protein.pdb has no readable ATOM record, a binding-site residue is missing from the sequence map
        of protein.pdb (the residue order of the per-chain Ankh embeddings), or its N, CA or C atom is missing in the
        fixed-column reader of the training code. A failure whose N, CA and C atoms are found when the coordinates are
        read as by the strict reader (lenseek.io.pdb.atom_xyz) has its own feat_detail value.
coords  BsitePDB writes x, y, z one column left of the PDB standard, so the fixed-column reader of the training code
        (lenseek.io.pdb.parse_pdb_legacy) and the strict reader can read different coordinates for 8-character values
        (<= -100 A). Every structure with an 8-character coordinate is read both ways (parse_pdb_legacy with
        strict_coordinates False and True), and every atom of the binding-site residues (RESNAME + chain + %04d keys of
        the ATOM records of each binding-site file) is compared: database sites, sites outside the database, queries.
patch   re-encodes with LEN-Seek (lenseek.encode) the database sites whose N, CA or C coordinates differ between the two
        readings: residue keys and Ankh rows of the training data, N/CA/C as read by the strict reader. patch_coords.npz
        holds the arrays (site_index, offsets, mu, site_ids) of release asset latentdb_main_s0_standard, the patch of
        the standard database edition of lenseek-search.

Inputs: --root ($LENSEEK_BSDB_ROOT, every entry directory), --eval-period-root ($LENSEEK_EVAL_PERIOD_ROOT),
--query-root ($LENSEEK_QUERY_ROOT), --eval-dir ($LENSEEK_EVAL_DIR: db_paths.txt, q_paths.txt), and for patch --train
($LENSEEK_TRAIN_PKL, the training features in db_paths.txt order) and --ckpt ($LENSEEK_CKPT, main_s0).
Outputs ($LENSEEK_OUT/database_scan/): scan_db.tsv, scan_queries.tsv, eval_period.json; coords_scan.tsv,
coords_counts.tsv; patch_coords.npz (latent-DB patch: site_index, offsets, mu, site_ids, meta).

    python scripts/database_scan.py sites --workers 8
    python scripts/database_scan.py coords --workers 8
    python scripts/database_scan.py patch
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import die, require_path, site_key  # noqa: E402
from lenseek.data import open_training_data  # noqa: E402
from lenseek.exclusion import CRYSTALLIZATION_AIDS_SET, GLYCANS_SET, LIGAND_EXCLUSION_SET, valid_ligand  # noqa: E402
from lenseek.featurize import legacy_resdefs, residue_frames  # noqa: E402
from lenseek.io.pdb import THREE_TO_ONE, VALID_RES, parse_pdb_legacy  # noqa: E402

OUT = os.path.join(os.path.abspath(os.environ.get("LENSEEK_OUT") or "out"), "database_scan")
EIGHT = re.compile(rb"^ATOM.{22}.*?(?:-\d{3}|\d{4})\.\d{3}", re.M)    # an 8-character coordinate
BB = ("N", "CA", "C")


# ---------------------------------------------------------------- helpers
def eval_keys(eval_dir):
    require_path(eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
    return ([site_key(p) for p in open(os.path.join(eval_dir, "db_paths.txt")).read().split()],
            [site_key(p) for p in open(os.path.join(eval_dir, "q_paths.txt")).read().split()])


def write_counts(name, rows):
    df = pd.DataFrame(rows, columns=["count", "value"])
    df.to_csv(os.path.join(OUT, name), sep="\t", index=False)
    print(df.to_string(index=False))


def entry_dirs(root):
    return sorted(e.name for e in os.scandir(root) if e.is_dir())


# ---------------------------------------------------------------- backbone check of the two readings (sites, coords)
def readers(prot):
    """protein.pdb read by the fixed-column reader of the training code, with its own coordinates and with those of
    the strict reader ({} when it cannot be read)."""
    return parse_pdb_legacy(prot) or {}, parse_pdb_legacy(prot, strict_coordinates=True) or {}


def backbone_atoms(fixed, strict, key):
    """Atoms of the binding-site residue key (RESNAME + chain + %04d) in the two readings of readers(); None for a
    residue that a reading does not have. ValueError for a key without a residue number."""
    chain, num = key[3], int(key[4:])
    a, b = fixed.get(chain, {}).get(num), strict.get(chain, {}).get(num)
    return (None if a is None else a["atoms"]), (None if b is None else b["atoms"])


# ---------------------------------------------------------------- sites: the checks of the training-data code
def seq_map(lines):
    """{chain: {RESNAME + chain + %04d: index}}: standard ATOM residues per chain, first name per (chain, number),
    sorted by number (the residue order of the per-chain Ankh embeddings)."""
    per_chain, seen = defaultdict(list), set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        name, chain = line[17:20].strip(), line[21].strip()
        if name not in THREE_TO_ONE or not chain:
            continue
        try:
            num = int(line[22:26].strip())
        except ValueError:
            continue
        if (chain, num) not in seen:
            seen.add((chain, num))
            per_chain[chain].append((num, name))
    return {c: {f"{n}{c}{num:04d}": i for i, (num, n) in enumerate(sorted(per_chain[c], key=lambda x: x[0]))}
            for c in sorted(per_chain)}


def readable_atoms(lines):
    """True when protein.pdb has an ATOM record with a readable residue number."""
    for line in lines:
        if line.startswith("ATOM"):
            try:
                int(line[22:26].strip())
                return True
            except ValueError:
                continue
    return False


def first_hetatm(path):
    """Residue name of the first HETATM record, or None (no record or no file)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    return line[17:20].strip()
    except Exception:
        pass
    return None


def ligand_flags(name):
    if name is None:
        return dict(lig_none=1, exclusion_list="", lig_digit=0, lig_len1=0, valid_ligand=0)
    lists = [n for n, s in [("Crystallization_aids", CRYSTALLIZATION_AIDS_SET),
                            ("Ligand_exclusion", LIGAND_EXCLUSION_SET), ("glycans", GLYCANS_SET)] if name in s]
    return dict(lig_none=0, exclusion_list=lists[0] if lists else "", lig_digit=int(bool(re.search(r"^\d", name))),
                lig_len1=int(len(name) < 2), valid_ligand=int(valid_ligand(name)))


def scan_entry(args):
    """One row per binding site of an entry directory."""
    root, entry = args
    d = os.path.join(root, entry)
    pockets = sorted(g for g in os.listdir(d) if g.startswith("ligand_pocket") and g.endswith(".pdb"))
    prot = os.path.join(d, "protein.pdb")
    has_prot = os.path.exists(prot)
    plines = open(prot).readlines() if has_prot else []
    smap = seq_map(plines)
    sites = []
    for pk in pockets:
        r = dict(key=f"{entry}/{pk}", entry=entry)
        try:
            lines = open(os.path.join(d, pk)).readlines()
            try:
                res = {f"{l[17:20].strip()}{l[21].strip()}{int(l[22:26].strip()):04d}" for l in lines
                       if l.startswith("ATOM") and l[17:20].strip() in VALID_RES}
            except Exception:
                res = set()
        except Exception:
            res = set()
        r["n_std_res"] = len(res)
        name = first_hetatm(os.path.join(d, pk.replace("ligand_pocket_", "ligand_")))
        r["ligand"] = name if name is not None else ""
        r.update(ligand_flags(name))
        sites.append((r, sorted(res)))
    s_fixed, s_strict = readers(prot) if has_prot else ({}, {})
    protein_fail = not (has_prot and readable_atoms(plines))
    rows = []
    for r, vr in sites:
        reason = ""
        if not has_prot:
            reason = "no_protein"
        elif not vr:
            reason = "no_residues"
        elif not smap:
            reason = "seqmap_empty"
        else:
            chs = sorted({rd[3] for rd in vr if len(rd) > 3})
            reason = next(("chain_not_in_seqmap" for c in chs if c not in smap), "")
            if not reason and any(smap[rd[3]].get(rd) is None for rd in vr):
                reason = "residue_not_in_seqmap"
        st_reason, bb_st = "", 0
        if has_prot and vr:
            for rd in vr:
                try:
                    a, b = backbone_atoms(s_fixed, s_strict, rd)
                except Exception:
                    st_reason = st_reason or "bad_resdef"
                    continue
                if a is None:
                    st_reason = st_reason or "residue_not_in_struct"
                elif any(x not in a for x in BB):
                    st_reason = st_reason or "missing_backbone"
                bb_st += 3 if b is None else sum(x not in b for x in BB)
        feat_ok = has_prot and bool(vr) and not reason and not st_reason
        r["feat_fail"] = int(not feat_ok or protein_fail)
        r["feat_detail"] = ("protein.pdb unreadable" if protein_fail else
                            "residue not in the sequence map" if reason else
                            "backbone lost by the PDB reader" if st_reason == "missing_backbone" and bb_st == 0 and vr
                            else "backbone atoms absent" if st_reason == "missing_backbone" else "")
        rows.append(r)
    return rows


def scan_root(root, entries, out_tsv, workers):
    cols = None
    n = 0
    with Pool(workers) as pool, open(out_tsv, "w") as f:
        for rows in pool.imap(scan_entry, [(root, e) for e in entries], chunksize=16):
            for r in rows:
                if cols is None:
                    cols = list(r)
                    f.write("\t".join(cols) + "\n")
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
                n += 1
    print(f"  {out_tsv}: {n} sites of {len(entries)} entries", flush=True)


def cmd_sites(a):
    root = os.path.abspath(require_path(a.root, "--root", "LENSEEK_BSDB_ROOT"))
    ev_root = require_path(a.eval_period_root, "--eval-period-root", "LENSEEK_EVAL_PERIOD_ROOT")
    q_root = os.path.abspath(require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT"))
    os.makedirs(OUT, exist_ok=True)
    ev_entries = entry_dirs(ev_root)
    n_ev = sum(sum(g.startswith("ligand_pocket") for g in os.listdir(os.path.join(ev_root, e))) for e in ev_entries)
    skip = set(ev_entries)
    entries = [e for e in entry_dirs(root) if e not in skip]
    json.dump(dict(root=root, n_entries_skipped=len(set(entry_dirs(root)) & skip),
                   eval_period_root=os.path.abspath(ev_root), eval_period_sites=n_ev,
                   eval_period_entries=len(ev_entries), query_root=q_root),
              open(os.path.join(OUT, "eval_period.json"), "w"), indent=1)
    scan_root(root, entries, os.path.join(OUT, "scan_db.tsv"), a.workers)
    scan_root(q_root, entry_dirs(q_root), os.path.join(OUT, "scan_queries.tsv"), a.workers)


# ---------------------------------------------------------------- coords: coordinate columns
def eight_char(d):
    try:
        with open(os.path.join(d, "protein.pdb"), "rb") as f:
            return d if EIGHT.search(f.read()) else None
    except OSError:
        return None


def coords_entry(args):
    """Per binding-site file of one structure: residues whose N/CA/C or other atoms differ between the readers."""
    d, pockets = args
    sf, ss = readers(os.path.join(d, "protein.pdb"))
    rows = []
    for tag, pocket in pockets:
        bb_res = bb_diff = bb_only_strict = sc = 0
        fixed_ok = strict_ok = True
        keys = legacy_resdefs(pocket)
        for k in keys:
            af, ast = backbone_atoms(sf, ss, k)
            af, ast = af or {}, ast or {}
            changed = False
            for name in set(af) | set(ast):
                a, b = af.get(name), ast.get(name)
                if a is not None and b is not None and np.array_equal(a, b):
                    continue
                if name in BB:
                    changed = True
                    bb_only_strict += a is None
                    bb_diff += a is not None
                else:
                    sc += 1
            bb_res += changed
            fixed_ok &= all(n in af for n in BB)
            strict_ok &= all(n in ast for n in BB)
        rows.append(dict(set=tag, key=site_key(pocket), n_res=len(keys), bb_res_changed=bb_res,
                         bb_atoms_changed=bb_diff, bb_atoms_only_strict=bb_only_strict, sc_atoms_changed=sc,
                         featurizable_fixed=int(fixed_ok), featurizable_strict=int(strict_ok)))
    return rows


def cmd_coords(a):
    root = os.path.abspath(require_path(a.root, "--root", "LENSEEK_BSDB_ROOT"))
    q_root = os.path.abspath(require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT"))
    db_keys, q_keys = eval_keys(a.eval_dir)
    db_keys, q_keys = set(db_keys), set(q_keys)
    dirs = sorted(glob.glob(os.path.join(root, "*"))) + sorted({os.path.join(q_root, k.split("/")[0]) for k in q_keys})
    with Pool(a.workers) as pool:
        hit = [d for d in pool.imap(eight_char, dirs, chunksize=64) if d]
    print(f"  {len(dirs)} structures, {len(hit)} with an 8-character coordinate", flush=True)
    jobs = []
    for d in hit:
        in_db_root = d.startswith(root + os.sep)
        pockets = []
        for p in sorted(glob.glob(os.path.join(d, "ligand_pocket*.pdb"))):
            k = site_key(p)
            if in_db_root:
                pockets.append(("db" if k in db_keys else "not_in_db", p))
            elif k in q_keys:
                pockets.append(("query", p))
        if pockets:
            jobs.append((d, pockets))
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.workers) as pool:
        S = pd.DataFrame([r for rows in pool.imap(coords_entry, jobs, chunksize=4) for r in rows])
    S.to_csv(os.path.join(OUT, "coords_scan.tsv"), sep="\t", index=False)
    db, nd, q = S[S.set == "db"], S[S.set == "not_in_db"], S[S.set == "query"]
    qa = q[(q.bb_res_changed + q.sc_atoms_changed) > 0]
    n_bb = int((db.bb_res_changed > 0).sum())
    write_counts("coords_counts.tsv", [
        ("database sites with N, CA or C coordinates that differ between the readings", n_bb),
        ("  share of the database sites", f"{n_bb / len(db_keys):.4f}"),
        ("database sites in which only side-chain coordinates differ",
         int(((db.bb_res_changed == 0) & (db.sc_atoms_changed > 0)).sum())),
        ("queries with an atom whose coordinates differ", len(qa)),
        ("  N, CA or C atoms", int(qa.bb_atoms_changed.sum() + qa.bb_atoms_only_strict.sum())),
        ("  side-chain atoms", int(qa.sc_atoms_changed.sum())),
        ("sites outside the database whose N, CA and C only the strict reading finds (featurization failed)",
         int(((nd.featurizable_fixed == 0) & (nd.featurizable_strict == 1)).sum()))])


# ---------------------------------------------------------------- patch: re-encoded database sites
def frames_of(blocks):
    n = len(blocks)
    xyz = {x: np.empty((n, 3), np.float32) for x in BB}
    for j, b in enumerate(blocks):
        for x in BB:
            xyz[x][j] = b["atoms"][x]
    return residue_frames(xyz["N"], xyz["CA"], xyz["C"])


def cmd_patch(a):
    from lenseek.checkpoint import load_model
    from lenseek.encode import Encoder
    root = require_path(a.root, "--root", "LENSEEK_BSDB_ROOT")
    db_keys, _ = eval_keys(a.eval_dir)
    index = {k: i for i, k in enumerate(db_keys)}
    if not os.path.exists(os.path.join(OUT, "coords_scan.tsv")):
        die(f"{os.path.join(OUT, 'coords_scan.tsv')} missing: run coords first")
    cs = pd.read_csv(os.path.join(OUT, "coords_scan.tsv"), sep="\t", dtype={"key": str})
    sel = sorted(index[k] for k in cs[(cs.set == "db") & (cs.bb_res_changed > 0)].key)
    ds = open_training_data(require_path(a.train, "--train", "LENSEEK_TRAIN_PKL"))
    ids = [site_key(p) for p in ds.path] if hasattr(ds, "path") else [i + ".pdb" for i in ds.site_ids]
    if ids != db_keys:
        die(f"{a.train}: sites are not those of db_paths.txt in the same order")
    torch.set_num_threads(a.threads)
    items = []
    for i in sel:
        h, _, _ = ds.site(i)
        pocket = os.path.join(root, db_keys[i])
        st = parse_pdb_legacy(os.path.join(os.path.dirname(pocket), "protein.pdb"), strict_coordinates=True)
        T, R = frames_of([{"atoms": st[k[3]][int(k[4:])]["atoms"]} for k in legacy_resdefs(pocket)])
        items.append((h, T, R))
    model, cfg, info = load_model(require_path(a.ckpt, "--ckpt", "LENSEEK_CKPT"))
    lat = Encoder(model, cfg).encode(items, warn_truncation=False)
    np.savez(os.path.join(OUT, "patch_coords.npz"), site_index=np.asarray(sel, np.int64),
             offsets=np.concatenate([[0], np.cumsum([len(x) for x in lat])]).astype(np.int64),
             mu=np.concatenate(lat).astype(np.float32), site_ids=np.array([db_keys[i][:-4] for i in sel]),
             meta=np.array(json.dumps(dict(name="coords", n_sites=len(sel), checkpoint_sha256=info.sha256))))
    print(f"  {os.path.join(OUT, 'patch_coords.npz')}: {len(sel)} re-encoded database sites", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func, h in [("sites", cmd_sites, "per-site checks of the training-period and the query binding sites"),
                          ("coords", cmd_coords, "binding sites whose atoms the two readings place differently"),
                          ("patch", cmd_patch, "re-encode the database sites whose N, CA or C coordinates differ")]:
        p = sub.add_parser(name, help=h)
        p.add_argument("--root", default=os.environ.get("LENSEEK_BSDB_ROOT"),
                       help="BsitePDB root (default $LENSEEK_BSDB_ROOT)")
        p.add_argument("--query-root", default=os.environ.get("LENSEEK_QUERY_ROOT"),
                       help="evaluation entries kept by the identity filter (default $LENSEEK_QUERY_ROOT)")
        p.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"),
                       help="evaluation outputs of main_s0 (default $LENSEEK_EVAL_DIR)")
        p.add_argument("--workers", type=int, default=4)
        p.set_defaults(func=func)
        if name == "sites":
            p.add_argument("--eval-period-root", default=os.environ.get("LENSEEK_EVAL_PERIOD_ROOT"),
                           help="entries of the evaluation period (default $LENSEEK_EVAL_PERIOD_ROOT)")
        if name == "patch":
            p.add_argument("--threads", type=int, default=8, help="torch threads")
            p.add_argument("--train", default=os.environ.get("LENSEEK_TRAIN_PKL"),
                           help="training features (default $LENSEEK_TRAIN_PKL)")
            p.add_argument("--ckpt", default=os.environ.get("LENSEEK_CKPT"),
                           help="main_s0 checkpoint (default $LENSEEK_CKPT)")
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
