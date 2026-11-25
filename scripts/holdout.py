#!/usr/bin/env python
"""Binding sites from later PDB entries: construction from the PDB entries released after the database was compiled,
the check of the build rules against BsitePDB, and the one scoring run (Supplementary Section S5; inputs of
Supplementary Tables S10 B and S11).

The set holds the ligand binding sites of X-ray PDB entries released from 2023-11-07 to 2026-10-31, built from RCSB
mmCIF files with rules that follow the BsitePDB definitions (Supplementary Table S1). Subcommands, in order; all
outputs go below --out (default $LENSEEK_OUT/holdout/):
  entries   RCSB Search API: entry IDs released in the window -> meta/ids_window.txt; RCSB Data API entry metadata
            -> meta/meta_window.jsonl (network); the entries to build -> meta/entries_to_build.txt: X-ray diffraction
            as the only method, resolution <= 3.0 A (at two decimals), at least one protein entity and one-character
            identifiers of every polymer chain (--validation: the method and the resolution only, the entries of
            validate)
  fetch     <id>.cif.gz from files.rcsb.org -> raw/cif/ (network)
  build     every candidate site -> build/{all_sites,candidates,entries,chains}.tsv
              entry   1-character chain ids, residue numbers -999..9999, not superseding a BsitePDB or training entry
              atoms   first model; one alternate location per position (blank plus 'A', else the lowest label)
              ligand  one HETATM residue, not water, >= 1 carbon, 140 <= RDKit MolWt < 800 (MolFromPDBBlock of the
                      residue's lines), minimum heavy-atom distance to ATOM-record heavy atoms >= 1.6 A (not covalent)
                      and <= 4.0 A (in contact), CCD not in the exclusion list E (data/exclusion_ccd.txt: 346 CCDs
                      seen as a non-polymer in >= 5 BsitePDB entries that never form a BsitePDB site)
              pocket  ATOM-record residues with any atom (hydrogens included) closer than r_vdW(ligand atom) + 4.0 A to
                      any ligand atom; r_vdW = RDKit periodic-table radius (lenseek.pocket)
  identity  unique sequences of the pocket and primary chains of the sites with 10-65 standard residues; training-chain
            FASTA from the BsitePDB protein.pdb files; MMseqs2 easy-search against the training chains (-s 7.5
            --max-seqs 2000 -e 10) and 80 % clusters of the primary chains -> seqs/, mm/
  select    identity of a site = maximum fident over its pocket chains with max(qcov, tcov) >= 0.5 (strata < 30 %,
            30-80 %, >= 80 %); one site per (80 % cluster of the primary chain, CCD): best resolution, earliest release,
            PDB id, ligand number; at most 2,000 sites, 667 / 667 / 666 per stratum, permuted with default_rng(20261102)
            and taken round-robin over the clusters; sampling weight = stratum size / sites taken -> sel/manifest.tsv
  write     the selected sites in strict PDB v3.3 columns with insertion codes (five-character CCDs as UNL with
            REMARK 999 LIGAND_CCD) -> BSDB_HO/<pdbid>/{protein.pdb, ligand_<n>_<CH>.pdb, ligand_pocket_<n>_<CH>.pdb}
  embed     Ankh-base per-chain embeddings (lenseek.ankh) of the chain sequences over the 20 standard residues,
            residues keyed by (chain, resSeq, iCode) in (resSeq, first appearance) order -> ankh_holdout/
  features  node features and residue frames per site (node order RES + chain + %04d + iCode; a malformed line, a
            duplicate atom or two residue names at one key fail loudly) -> query_cache_holdout.pkl; manifest.tsv =
            sel/manifest.tsv with the columns featurised, skip_reason, n_nodes (the scored sites: featurised = 1)
            embed and features read the files themselves: lenseek-embed sorts insertion codes alphabetically, these
            steps keep the file order within a residue number, the order the embeddings of these sites were
            computed in
  clusters  50 % (and 30 %) sequence clusters of the sites: all-vs-all MMseqs2 (-e 1e-3), single linkage at fident
            >= 0.5 over >= 50 % of the shorter chain, a site follows its primary chain -> clusters/seq_clusters_*.tsv
  strata    Foldseek (3Di+AA, -e 10) maximum query TM-score of the pocket chains to the training chains (< 0.5,
            0.5-0.8, >= 0.8); ligand novelty: same CCD in training / new CCD with Morgan r2 2048-bit Dice >= 0.46 to a
            training ligand / below -> strata/{structure,ligand}.tsv (scripts/site_annotation.py strata: the same
            two steps on the evaluation queries)
  counts    entries built (status ok), valid sites (entries), sites after de-duplication (80 % clusters, identity
            strata), sampled sites (entries, clusters, strata), featurised sites per identity stratum, and the
            structure and ligand strata of the sampled sites -> counts.tsv (Supplementary Section S5, Counts)
  score     the one scoring run, into a new folder (--run, default OUT/run; --prepare-only stops after queries.tsv):
              queries.tsv             q_idx, key, entry, 50 % cluster, ligand CCD (from the manifest), stratum
              model_<name>/           query encodings and Chamfer distances to every database site of the original
                                      model and of the model trained without homologous entries (scripts/evaluate.py;
                                      the database is not re-encoded)
              dist_deeplytough.npy    DeeplyTough descriptor distance (rows of queries without a descriptor NaN)
              pairs_*.tsv             top-100 of the original model and of DeeplyTough and their union, the pairs that
                                      scripts/labels.py scores with ProBiS
            analysis/holdout_score.py computes the ligand metrics and the MCC of the run from these files.
A rebuild at a later date can differ from the set that was scored, because RCSB entries are revised.
The build rules are checked on entries that BsitePDB covers:
  validate  build_entry with the exclusion list E on the entries of --ids (mmCIF files in --cifdir); each ligand
            decision is matched to an unused BsitePDB ligand of the entry (--bsdb-root) with the same residue number,
            the same name (UNL for a five-character CCD) and a first atom within 0.0015 A. Recall = BsitePDB sites
            matched by an accepted site / all BsitePDB sites; Jaccard index of the (chain, number) binding-site
            residues of each recovered site -> validate/sites.tsv, validate/summary.tsv, on the X-ray entries with
            resolution <= 3.0 A released in the evaluation period (2023-07-21 to 2023-11-06; entries --validation on
            that window). The ligand decisions depend on the RDKit version.

Inputs: data/exclusion_ccd.txt, the PDB IDs of BsitePDB and of the training list, $LENSEEK_BSDB_ROOT (training
protein.pdb files), MMseqs2 ($MMSEQS, else mmseqs), Foldseek ($FOLDSEEK, else foldseek), Ankh-base weights
($LENSEEK_ANKH_DIR, else the pinned revision), database ligand CCDs and a CCD SMILES table (strata),
$LENSEEK_EVAL_PERIOD_ROOT (validate: BsitePDB entries of the evaluation period); score: see `score -h`.
usage: python scripts/holdout.py {entries,fetch,build,identity,select,write,embed,features,clusters,strata,counts,
                                  validate,score} [options]
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import gemmi
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (AA, REPO, chain_seqs, die, entry_chain_seqs, log, read_tsv, single_linkage,  # noqa: E402
                      site_key)
from lenseek.pocket import POCKET_ADD, WATER, read_atoms, rvdw, select_altloc, write_entry_strict  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = os.path.join(os.path.abspath(os.environ.get("LENSEEK_OUT") or "out"), "holdout")
MMSEQS = os.environ.get("MMSEQS", "mmseqs")


# ---------------------------------------------------------------- RCSB (network)
def rcsb_search_window(_args):
    """PDB entry IDs whose initial release date falls in [FROM, TO] (RCSB Search API v2)."""
    import requests
    frm, to, out = _args.frm, _args.to, _args.out
    q = {
        "query": {"type": "terminal", "service": "text",
                  "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "range",
                                 "value": {"from": frm, "to": to, "include_lower": True, "include_upper": True}}},
        "return_type": "entry",
        "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]},
    }
    for attempt in range(5):
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=300)
        if r.status_code == 200:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    res = r.json()
    ids = sorted(x["identifier"].lower() for x in res.get("result_set", []))
    with open(out, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{frm}..{to}: total_count={res.get('total_count')} written={len(ids)}")


def rcsb_entry_meta(_args):
    """Entry metadata (RCSB Data API, GraphQL), one raw JSON object per line; entries already in the output are skipped."""
    import requests
    ids_path, out_path, batch = _args.ids_path, _args.out_path, _args.batch
    QUERY = """query($ids: [String!]!) { entries(entry_ids: $ids) {
  rcsb_id
  rcsb_accession_info { initial_release_date deposit_date revision_date major_revision minor_revision }
  exptl { method }
  pdbx_database_status { pdb_format_compatible status_code }
  rcsb_entry_info { resolution_combined experimental_method polymer_entity_count_protein polymer_entity_count_nucleic_acid
                    nonpolymer_entity_count deposited_atom_count deposited_model_count polymer_composition
                    selected_polymer_entity_types nonpolymer_bound_components deposited_polymer_monomer_count }
  nonpolymer_entities { rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids asym_ids nonpolymer_comp_id }
                        nonpolymer_comp { chem_comp { id type formula formula_weight } } }
  polymer_entities { entity_poly { type rcsb_entity_polymer_type pdbx_strand_id } }
} }"""
    ids = [l.strip().upper() for l in open(ids_path) if l.strip()]
    done = set()
    try:
        for l in open(out_path):
            done.add(json.loads(l)["rcsb_id"].upper())
    except FileNotFoundError:
        pass
    todo = [i for i in ids if i not in done]
    with open(out_path, "a") as fo:
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            for attempt in range(6):
                try:
                    r = requests.post("https://data.rcsb.org/graphql", json={"query": QUERY, "variables": {"ids": chunk}},
                                      timeout=300)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            r.raise_for_status()
            data = r.json()
            if "errors" in data and not data.get("data"):
                raise SystemExit(f"GraphQL error: {data['errors'][:2]}")
            for e in data["data"]["entries"] or []:
                if e is not None:
                    fo.write(json.dumps(e) + "\n")
            fo.flush()
            print(f"{min(k + batch, len(todo))}/{len(todo)}", flush=True)


def select_entries(meta_path, out_path, validation=False):
    """Entries of the metadata to build -> out_path (sorted IDs): X-ray diffraction as the only method and resolution
    <= 3.0 A at two decimals; unless validation, also at least one protein entity and one-character chain IDs."""
    ids = set()
    for line in open(meta_path):
        e = json.loads(line)
        info = e.get("rcsb_entry_info") or {}
        res = (info.get("resolution_combined") or [None])[0]
        if [x["method"] for x in e.get("exptl") or []] != ["X-RAY DIFFRACTION"] or res is None or round(res, 2) > 3.0:
            continue
        chains = [c for p in e.get("polymer_entities") or [] for c in
                  ((p.get("entity_poly") or {}).get("pdbx_strand_id") or "").split(",")]
        if validation or (info.get("polymer_entity_count_protein") and all(len(c) <= 1 for c in chains)):
            ids.add(e["rcsb_id"].lower())
    with open(out_path, "w") as f:
        f.write("\n".join(sorted(ids)) + "\n")
    print(f"{len(ids)} entries to build -> {out_path}")


def fetch_rcsb(ids_path, out, kind="cif", par=6):
    """files.rcsb.org/download/<ID>.<kind>.gz -> out/<id>.<kind>.gz; non-empty files are kept; IDs that fail or do not
    exist go to out/_missing.txt."""
    os.makedirs(out, exist_ok=True)
    missing = os.path.join(out, "_missing.txt")

    def fetch(x):
        lc, f = x.lower(), os.path.join(out, f"{x.lower()}.{kind}.gz")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            return
        for t in (1, 2, 3):
            code = subprocess.run(["curl", "-s", "-w", "%{http_code}", "-o", f + ".part",
                                   f"https://files.rcsb.org/download/{x.upper()}.{kind}.gz"], capture_output=True,
                                  text=True).stdout
            if code == "200":
                os.replace(f + ".part", f)
                return
            if os.path.exists(f + ".part"):
                os.remove(f + ".part")
            if code == "404":
                open(missing, "a").write(f"{lc} 404\n")
                return
            time.sleep(t * 5)
        open(missing, "a").write(f"{lc} fail {code}\n")
    with ThreadPool(par) as p:
        p.map(fetch, [l.strip() for l in open(ids_path) if l.strip()])


# ---------------------------------------------------------------- binding-site builder
COV_CUT = 1.6        # A, ligand heavy atom to ATOM-record heavy atom; below = covalently bound
CONTACT_CUT = 4.0    # A, ligand must have a heavy atom within this distance of an ATOM-record heavy atom
MW_MIN, MW_MAX = 140.0, 800.0
VALID_RES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
             "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
COORD_MIN, COORD_MAX = -999.999, 9999.999   # representable in an 8.3f PDB field
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def std_pdb_line(a, serial):
    """Standard-column PDB line of a ligand atom (only to hand the residue to RDKit)."""
    name = a["name"]
    name4 = name.ljust(4) if (len(name) == 4 or len(a["elem"]) == 2) else (" " + name).ljust(4)
    resn = a["resn"] if len(a["resn"]) <= 3 else "UNL"
    return (f"HETATM{serial % 100000:>5} {name4}{a['alt']}{resn:>3} {a['chain'][:1]}{a['resi'] % 10000:>4}{(a['icode'] or ' ')[:1]}   "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['occ']:6.2f}{a['b']:6.2f}          {a['elem']:>2}")


def rdkit_mw(lig_atoms):
    block = "\n".join(std_pdb_line(a, i + 1) for i, a in enumerate(lig_atoms)) + "\nEND\n"
    try:
        m = Chem.MolFromPDBBlock(block)
    except Exception:
        m = None
    return None if m is None else Descriptors.MolWt(m)


def coords_representable(atoms):
    return all(COORD_MIN <= v <= COORD_MAX for a in atoms for v in (a["x"], a["y"], a["z"]))


def train_filter_excluded(ccd, remove_names):
    """Ligand filter of the training data (lenseek.exclusion.valid_ligand) on the full CCD id; a flag only."""
    return ccd in remove_names or re.search(r"^\d", ccd) is not None or len(ccd) < 2


def build_entry(path, excluded_ccd, remove_names=frozenset()):
    """Return (sites, candidates, atoms). sites: accepted ligands with pocket info; candidates: every HETATM decision."""
    raw, st = read_atoms(path)
    atoms = select_altloc(raw)
    prot = [a for a in atoms if a["rec"] == "ATOM"]
    cands, order = {}, []
    for i, a in enumerate(atoms):
        if a["rec"] == "HETATM" and a["resn"] not in WATER:
            k = (a["chain"], a["resi"], a["icode"], a["resn"])
            if k not in cands:
                cands[k] = []
                order.append(k)
            cands[k].append(i)
    if not prot:
        return [], [dict(key=k, reason="no_protein") for k in order], atoms
    P = np.array([[a["x"], a["y"], a["z"]] for a in prot])
    pheavy_idx = [i for i, a in enumerate(prot) if a["elem"] not in ("H", "D")]
    TH = cKDTree(P[pheavy_idx]) if pheavy_idx else None
    TA = cKDTree(P)
    XALL = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    TALL = cKDTree(XALL)
    sites, decisions = [], []
    for k in order:
        L = [atoms[i] for i in cands[k]]
        heavy = [a for a in L if a["elem"] not in ("H", "D")]
        d = dict(key=k, n_atoms=len(L), n_heavy=len(heavy), mw=None, dmin=None)
        if not any(a["elem"] == "C" for a in heavy):
            d["reason"] = "no_carbon"; decisions.append(d); continue
        mw = rdkit_mw(L)
        d["mw"] = mw
        if mw is None:
            d["reason"] = "rdkit_fail"; decisions.append(d); continue
        if mw < MW_MIN:
            d["reason"] = "mw_lt_140"; decisions.append(d); continue
        if mw >= MW_MAX:
            d["reason"] = "mw_ge_800"; decisions.append(d); continue
        if TH is None:
            d["reason"] = "no_protein"; decisions.append(d); continue
        X = np.array([[a["x"], a["y"], a["z"]] for a in heavy])
        dmin = float(TH.query(X, k=1)[0].min())
        d["dmin"] = dmin
        if dmin < COV_CUT:
            d["reason"] = "covalent"; decisions.append(d); continue
        if dmin > CONTACT_CUT:
            d["reason"] = "no_contact"; decisions.append(d); continue
        if k[3] in excluded_ccd:
            d["reason"] = "excluded_ccd"; decisions.append(d); continue
        # pocket: ATOM residues with any atom (incl. H) closer than r_vdw(lig atom) + 4.0 to any ligand atom (incl. H)
        XL = np.array([[a["x"], a["y"], a["z"]] for a in L])
        pres, near_chains = set(), {k[0]}
        lig_set = set(cands[k])
        for j, a in enumerate(L):
            cut = rvdw(a["elem"]) + POCKET_ADD - 1e-9
            for q in TA.query_ball_point(XL[j], cut):
                pa = prot[q]
                pres.add((pa["chain"], pa["resi"], pa["icode"]))
            for q in TALL.query_ball_point(XL[j], cut):
                if q not in lig_set:
                    near_chains.add(atoms[q]["chain"])
        d["reason"] = "ok"
        d["pocket"] = pres
        d["suffix"] = "".join(sorted(near_chains))
        d["atoms"] = L
        decisions.append(d)
        sites.append(d)
    for n, s in enumerate(sites, 1):
        s["idx"] = n
        s["train_filter_excluded"] = train_filter_excluded(s["key"][3], remove_names)
    return sites, decisions, atoms


def pocket_std_keys(site, atoms):
    """Pocket residues keyed RES+chain+%04d over standard residues (no insertion code)."""
    keys = set()
    for a in atoms:
        if a["rec"] == "ATOM" and (a["chain"], a["resi"], a["icode"]) in site["pocket"] and a["resn"] in VALID_RES:
            keys.add(f"{a['resn']}{a['chain']}{a['resi']:04d}")
    return keys


def sign_loss_atoms(site, atoms):
    """Pocket-residue atoms with a coordinate <= -100 or >= 1000 A (8 columns in the 8.3f field)."""
    n = 0
    for a in atoms:
        if a["rec"] == "ATOM" and (a["chain"], a["resi"], a["icode"]) in site["pocket"]:
            if any(len(f"{v:8.3f}".strip()) >= 8 for v in (a["x"], a["y"], a["z"])):
                n += 1
    return n


def atom_chain_seqs(atoms):
    """Per-chain sequence of a built entry for MMseqs2 (identity), from its mmCIF atoms: gemmi tabulated amino acids,
    one-letter, the 20 standard letters, residues keyed by (chain, resSeq, iCode) in (resSeq, first appearance) order,
    chains with >= 10 residues. (pipeline.chain_seqs reads a written protein.pdb in file order, non-standard as X.)"""
    first, names = {}, {}
    for i, a in enumerate(atoms):
        if a["rec"] != "ATOM":
            continue
        k = (a["chain"], a["resi"], a["icode"])
        if k not in first:
            first[k] = i
            names[k] = a["resn"]
    per_chain = {}
    for k in sorted(first, key=lambda k: (k[0], k[1], first[k])):
        t = gemmi.find_tabulated_residue(names[k])
        if t is None or not t.is_amino_acid():
            continue
        c = gemmi.one_letter_code([names[k]]).upper()
        if c in AA20:
            per_chain.setdefault(k[0], []).append(c)
    return {ch: "".join(v) for ch, v in per_chain.items() if len(v) >= 10}


def superseded_ids(path):
    try:
        doc = gemmi.cif.read(path)
        tab = doc.sole_block().find("_pdbx_database_PDB_obs_spr.", ["replace_pdb_id"])
        out = set()
        for row in tab:
            for x in row[0].replace("'", "").split():
                out.add(x.lower())
        return out
    except Exception:
        return set()


def build_one(pdbid):
    path = os.path.join(cifdir, f"{pdbid}.cif.gz")
    if not os.path.exists(path):
        return dict(pdbid=pdbid, status="no_file")
    try:
        sites, dec, atoms = build_entry(path, EXCL, REMOVE)
    except Exception as e:
        return dict(pdbid=pdbid, status=f"error:{type(e).__name__}:{str(e)[:80]}")
    chains = {a["chain"] for a in atoms}
    if any(len(c) != 1 for c in chains):
        return dict(pdbid=pdbid, status="multichar_chain")
    if any(not (-999 <= a["resi"] <= 9999) for a in atoms):
        return dict(pdbid=pdbid, status="resnum_out_of_range")
    sup = superseded_ids(path) & BSITE
    if sup:
        return dict(pdbid=pdbid, status="supersedes_bsitepdb:" + ",".join(sorted(sup)))
    seqs = atom_chain_seqs(atoms)
    rows = []
    for s in sites:
        std_old = pocket_std_keys(s, atoms)            # RES+chain+%04d (no insertion code)
        std = {(a["chain"], a["resi"], a["icode"]) for a in atoms
               if a["rec"] == "ATOM" and (a["chain"], a["resi"], a["icode"]) in s["pocket"] and a["resn"] in VALID_RES}
        n_all = len(s["pocket"])
        n_icode = sum(1 for k in std if k[2])
        per_chain = {}
        for k in std:
            per_chain[k[0]] = per_chain.get(k[0], 0) + 1
        pchains = "".join(sorted(per_chain))
        ranked = sorted(per_chain.items(), key=lambda t: (-t[1], t[0]))
        primary = next((c for c, _ in ranked if c in seqs), "")
        ccd = s["key"][3]
        tf = train_filter_excluded(ccd, REMOVE)
        why = ("exclusion_list" if ccd in REMOVE else "numeric_start" if ccd[:1].isdigit() else "short" if len(ccd) < 2 else "") if tf else ""
        rows.append(dict(site_id=f"{pdbid}/ligand_pocket_{s['idx']}_{s['suffix']}", pdbid=pdbid, idx=s["idx"], suffix=s["suffix"],
                         ligand_ccd=ccd, ligand_chain=s["key"][0], ligand_resnum=s["key"][1], ligand_icode=s["key"][2],
                         ligand_atoms=s["n_atoms"], ligand_heavy_atoms=s["n_heavy"], rdkit_mw=round(s["mw"], 2),
                         dmin_protein=round(s["dmin"], 3), n_res_std=len(std), n_res_std_noicode=len(std_old),
                         n_icode_res=n_icode, n_res_all=n_all, pocket_chains=pchains,
                         primary_chain=primary, train_filter_excluded=int(tf), train_filter_reason=why,
                         ccd_5char=int(len(ccd) > 3), coord_sign_loss_atoms=sign_loss_atoms(s, atoms)))
    return dict(pdbid=pdbid, status="ok", sites=rows, seqs=seqs,
                cands=[(d["key"], d["reason"]) for d in dec])


cifdir = out_root = root = out = bsdb = None      # settings of the Pool workers, set by set_globals
EXCL, BSITE, REMOVE, sel = set(), set(), set(), {}


def set_globals(g):
    """Pool initializer: the worker functions (build_one, write_one, convert_train_one) read module globals."""
    globals().update(g)


def cmd_build(a):
    from lenseek import exclusion as X
    cifdir, outdir, nproc = a.cifdir, os.path.join(a.out, "build"), a.nproc
    os.makedirs(outdir, exist_ok=True)
    EXCL = {l.split()[0] for l in open(a.exclusion) if l.strip() and not l.startswith("#")}
    BSITE = {l.strip().lower() for l in open(a.bsitepdb_ids) if l.strip()}
    REMOVE = {x.strip() for x in (X.Crystallization_aids + "," + X.Ligand_exclusion + "," + X.glycans).split(",")}
    ids = [l.strip().lower() for l in open(a.ids) if l.strip()]
    cols = ["site_id", "pdbid", "idx", "suffix", "ligand_ccd", "ligand_chain", "ligand_resnum", "ligand_icode",
            "ligand_atoms", "ligand_heavy_atoms", "rdkit_mw", "dmin_protein", "n_res_std", "n_res_std_noicode", "n_icode_res",
            "n_res_all", "pocket_chains",
            "primary_chain", "train_filter_excluded", "train_filter_reason", "ccd_5char", "coord_sign_loss_atoms"]
    g = dict(cifdir=cifdir, EXCL=EXCL, BSITE=BSITE, REMOVE=REMOVE)
    with Pool(nproc, set_globals, (g,)) as p, open(os.path.join(outdir, "all_sites.tsv"), "w") as fs, \
            open(os.path.join(outdir, "entries.tsv"), "w") as fe, open(os.path.join(outdir, "chains.tsv"), "w") as fc, \
            open(os.path.join(outdir, "candidates.tsv"), "w") as fd:
        fs.write("\t".join(cols) + "\n"); fe.write("pdbid\tstatus\tn_sites\n"); fc.write("chain_id\tsequence\n")
        fd.write("pdbid\tchain\tresnum\ticode\tccd\treason\n")
        for i, r in enumerate(p.imap_unordered(build_one, ids, chunksize=4)):
            fe.write(f"{r['pdbid']}\t{r['status']}\t{len(r.get('sites', []))}\n")
            for s in r.get("sites", []):
                fs.write("\t".join(str(s[c]) for c in cols) + "\n")
            for ch, sq in r.get("seqs", {}).items():
                fc.write(f"{r['pdbid']}_{ch}\t{sq}\n")
            for k, why in r.get("cands", []):
                fd.write(f"{r['pdbid']}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{why}\n")
            if i % 1000 == 0:
                print(i, flush=True)


# ---------------------------------------------------------------- identity to the training chains, selection
def make_fasta(_args):
    """seq_map.tsv (chain -> unique-sequence id), holdout_query.fasta (pocket chains of the sites with 10-65 standard
    residues), holdout_primary.fasta (their primary chains)."""
    build, sdir = _args.build, _args.sdir
    os.makedirs(sdir, exist_ok=True)
    seqs = {}
    with open(os.path.join(build, "chains.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seqs[r["chain_id"]] = r["sequence"]
    pocket_chains, primary_chains = set(), set()
    with open(os.path.join(build, "all_sites.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not (10 <= int(r["n_res_std"]) <= 65):
                continue
            for c in r["pocket_chains"]:
                cid = f"{r['pdbid']}_{c}"
                if cid in seqs:
                    pocket_chains.add(cid)
            if r["primary_chain"]:
                primary_chains.add(f"{r['pdbid']}_{r['primary_chain']}")
    rep = {}
    for cid in sorted(pocket_chains | primary_chains):
        rep.setdefault(seqs[cid], f"s{len(rep):06d}")
    with open(os.path.join(sdir, "seq_map.tsv"), "w") as f:
        f.write("chain_id\tseq_id\tlength\n")
        for cid in sorted(pocket_chains | primary_chains):
            f.write(f"{cid}\t{rep[seqs[cid]]}\t{len(seqs[cid])}\n")
    inv = {v: k for k, v in rep.items()}
    q = sorted({rep[seqs[c]] for c in pocket_chains})
    p = sorted({rep[seqs[c]] for c in primary_chains})
    for name, ids in (("holdout_query.fasta", q), ("holdout_primary.fasta", p)):
        with open(os.path.join(sdir, name), "w") as f:
            for i in ids:
                f.write(f">{i}\n{inv[i]}\n")
    print(f"pocket chains {len(pocket_chains)} -> {len(q)} unique sequences; primary chains {len(primary_chains)} -> {len(p)} unique")


def train_chain_sequences(_args):
    """Training-chain FASTA from the BsitePDB protein.pdb files: gemmi tabulated amino acids, one-letter, the 20
    standard letters, chains with >= 10 residues, headers <pdbid>_<chain>."""
    ids_path, root, out = _args.ids_path, _args.root, _args.out
    n = 0
    with open(out, "w") as f:
        for pid in [l.strip() for l in open(ids_path) if l.strip()]:
            for h, s in entry_chain_seqs(root, pid):
                f.write(f">{h}\n{s}\n"); n += 1
    print("chains:", n)


def cmd_identity(a):
    sdir, mm, th = os.path.join(a.out, "seqs"), os.path.join(a.out, "mm"), str(a.threads)
    make_fasta(argparse.Namespace(build=os.path.join(a.out, "build"), sdir=sdir))
    train = a.train_fasta or os.path.join(sdir, "train_full.fasta")
    if not a.train_fasta:
        train_chain_sequences(argparse.Namespace(ids_path=a.train_ids, root=a.bsdb_root, out=train))
    os.makedirs(mm, exist_ok=True)
    with open(os.path.join(mm, "search.log"), "w") as lg:
        subprocess.run([MMSEQS, "easy-search", os.path.join(sdir, "holdout_query.fasta"), train,
                        os.path.join(mm, "holdout_vs_train.m8"), os.path.join(mm, "tmp_s"), "--threads", th, "-s", "7.5",
                        "--max-seqs", "2000", "-e", "10", "--format-output",
                        "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"], stdout=lg, stderr=lg, check=True)
    with open(os.path.join(mm, "cluster.log"), "w") as lg:
        subprocess.run([MMSEQS, "easy-cluster", os.path.join(sdir, "holdout_primary.fasta"), os.path.join(mm, "clu80"),
                        os.path.join(mm, "tmp_c"), "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0", "--threads", th],
                       stdout=lg, stderr=lg, check=True)
    for t in ("tmp_s", "tmp_c"):
        shutil.rmtree(os.path.join(mm, t), ignore_errors=True)


def select_sites(_args):
    """Identity strata, de-duplication, stratified cap and the selection manifest (manifest_all.tsv, manifest.tsv)."""
    build, sdir, mmdir, meta_path, out = _args.build, _args.sdir, _args.mmdir, _args.meta_path, _args.out
    SEED, CAP = 20261102, 2000

    meta = {}
    for l in open(meta_path):
        e = json.loads(l)
        info = e.get("rcsb_entry_info") or {}
        acc = e.get("rcsb_accession_info") or {}
        meta[e["rcsb_id"].lower()] = dict(release_date=(acc.get("initial_release_date") or "")[:10],
                                          deposit_date=(acc.get("deposit_date") or "")[:10],
                                          resolution=(info.get("resolution_combined") or [None])[0])
    seq_id = {}
    for r in csv.DictReader(open(os.path.join(sdir, "seq_map.tsv")), delimiter="\t"):
        seq_id[r["chain_id"]] = r["seq_id"]

    # identity of each query sequence to the training chains under the three coverage definitions
    best = defaultdict(lambda: {"A": (0.0, ""), "B": (0.0, ""), "C": (0.0, "")})
    cols = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits".split(",")
    with open(os.path.join(mmdir, "holdout_vs_train.m8")) as f:
        for line in f:
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            fid, qc, tc = float(r["fident"]), float(r["qcov"]), float(r["tcov"]),
            b = best[r["query"]]
            if fid > b["A"][0]:
                b["A"] = (fid, r["target"])
            if qc >= 0.8 and fid > b["B"][0]:
                b["B"] = (fid, r["target"])
            if max(qc, tc) >= 0.5 and fid > b["C"][0]:
                b["C"] = (fid, r["target"])

    clu = {}
    for line in open(os.path.join(mmdir, "clu80_cluster.tsv")):
        rep, mem = line.split()
        clu[mem] = rep
    rep_ids = {r: f"C{i:05d}" for i, r in enumerate(sorted(set(clu.values())))}

    def stratum(x):
        return "lt30" if x < 0.30 else ("30to80" if x < 0.80 else "ge80")

    sites = []
    for r in csv.DictReader(open(os.path.join(build, "all_sites.tsv")), delimiter="\t"):
        if not (10 <= int(r["n_res_std"]) <= 65):
            continue
        pid = r["pdbid"]
        if not r["primary_chain"]:
            continue
        ident = {}
        for d in "ABC":
            v, tgt, via = 0.0, "", ""
            for c in r["pocket_chains"]:
                sid = seq_id.get(f"{pid}_{c}")
                if sid is None:
                    continue
                x, t = best[sid][d] if sid in best else (0.0, "")
                if x > v:
                    v, tgt, via = x, t, c
            ident[d] = (v, tgt, via)
        psid = seq_id[f"{pid}_{r['primary_chain']}"]
        m = meta.get(pid, {})
        r.update(release_date=m.get("release_date", ""), deposit_date=m.get("deposit_date", ""),
                 resolution=m.get("resolution", ""), cluster80=rep_ids[clu[psid]],
                 ident_A=round(ident["A"][0], 3), nearest_train_A=ident["A"][1],
                 ident_B=round(ident["B"][0], 3), nearest_train_B=ident["B"][1],
                 ident_C=round(ident["C"][0], 3), nearest_train_C=ident["C"][1], query_chain_C=ident["C"][2],
                 stratum=stratum(ident["C"][0]), stratum_B=stratum(ident["B"][0]), stratum_A=stratum(ident["A"][0]))
        sites.append(r)

    # de-duplication: one site per (80 % cluster, CCD); best resolution, earliest release, PDB id, ligand number
    groups = defaultdict(list)
    for s in sites:
        groups[(s["cluster80"], s["ligand_ccd"])].append(s)
    reps = []
    for k, g in groups.items():
        g.sort(key=lambda s: (float(s["resolution"]) if s["resolution"] not in ("", None) else 99.0, s["release_date"],
                              s["pdbid"], int(s["idx"])))
        for i, s in enumerate(g):
            s["dedup_group_size"] = len(g)
            s["dedup_representative"] = int(i == 0)
        reps.append(g[0])

    # stratified cap with round-robin over clusters inside each stratum
    by_st = defaultdict(list)
    for s in reps:
        by_st[s["stratum"]].append(s)
    order = ["lt30", "30to80", "ge80"]
    N = {k: len(by_st[k]) for k in order}
    if sum(N.values()) <= CAP:
        quota = dict(N)
    else:
        quota = {"lt30": 667, "30to80": 667, "ge80": 666}
        while True:
            short = {k: quota[k] - N[k] for k in order if N[k] < quota[k]}
            if not short:
                break
            free = sum(short.values())
            for k in short:
                quota[k] = N[k]
            rest = [k for k in order if N[k] > quota[k]]
            if not rest:
                break
            add, rem = divmod(free, len(rest))
            for i, k in enumerate(rest):
                quota[k] += add + (1 if i < rem else 0)
    rng = np.random.default_rng(SEED)
    selected = set()
    for k in order:
        pool = sorted(by_st[k], key=lambda s: s["site_id"])
        perm = [pool[i] for i in rng.permutation(len(pool))]
        first_rank, members = {}, defaultdict(list)
        for rank, s in enumerate(perm):
            first_rank.setdefault(s["cluster80"], rank)
            members[s["cluster80"]].append(s)
        clusters = sorted(members, key=lambda c: first_rank[c])
        take, rnd = [], 0
        while len(take) < quota[k]:
            added = False
            for c in clusters:
                if rnd < len(members[c]):
                    take.append(members[c][rnd]); added = True
                    if len(take) >= quota[k]:
                        break
            if not added:
                break
            rnd += 1
        for s in take:
            selected.add(s["site_id"])
        w = N[k] / len(take) if take else 0
        for s in by_st[k]:
            s["sampling_weight"] = round(w, 4) if s["site_id"] in selected else ""
    for s in sites:
        s["selected"] = int(s["site_id"] in selected)
        s["atoms_le_m100"] = s["coord_sign_loss_atoms"]  # read as written in the strict layout
        s.setdefault("sampling_weight", "")

    cols_out = ["site_id", "pdbid", "release_date", "deposit_date", "resolution", "ligand_ccd", "ligand_chain", "ligand_resnum",
                "ligand_icode", "idx", "suffix", "pocket_chains", "primary_chain", "n_res_std", "n_res_std_noicode", "n_icode_res",
                "n_res_all", "ligand_atoms",
                "ligand_heavy_atoms", "rdkit_mw", "dmin_protein", "cluster80", "dedup_group_size", "dedup_representative",
                "stratum", "ident_C", "nearest_train_C", "query_chain_C", "ident_A", "nearest_train_A", "stratum_A",
                "ident_B", "nearest_train_B", "stratum_B", "train_filter_excluded", "train_filter_reason", "ccd_5char",
                "atoms_le_m100", "selected", "sampling_weight"]
    os.makedirs(out, exist_ok=True)
    for name, rows in (("manifest_all.tsv", sites), ("manifest.tsv", [s for s in sites if s["selected"]])):
        with open(os.path.join(out, name), "w") as f:
            f.write("\t".join(cols_out) + "\n")
            for s in sorted(rows, key=lambda s: s["site_id"]):
                f.write("\t".join(str(s.get(c, "")) for c in cols_out) + "\n")
    print("valid sites", len(sites), "| dedup groups", len(groups), "| per stratum after dedup", N, "| quota", quota,
          "| selected", len(selected))


def write_one(pid):
    """Rebuild one entry; the selected sites must reappear with the same number, chain suffix, CCD, chain and residue
    number and fit the 8.3f coordinate field; then write them in strict PDB columns."""
    rows = sel[pid]
    sites, dec, atoms = build_entry(os.path.join(cifdir, f"{pid}.cif.gz"), EXCL)
    by_idx = {s["idx"]: s for s in sites}
    want = set()
    for r in rows:
        s = by_idx.get(int(r["idx"]))
        k = s["key"] if s else None
        if s is None or s["suffix"] != r["suffix"] or k[3] != r["ligand_ccd"] or k[0] != r["ligand_chain"] or str(k[1]) != r["ligand_resnum"]:
            return (pid, "MISMATCH")
        want.add(s["idx"])
    prot = [a for a in atoms if a["rec"] == "ATOM"]
    if not coords_representable(prot + [a for s in sites if s["idx"] in want for a in s["atoms"]]):
        return (pid, "COORD_OUT_OF_RANGE")
    write_entry_strict(out_root, pid, sites, atoms, only_idx=want)
    return (pid, "ok")


def cmd_write(a):
    cifdir, out_root = a.cifdir, os.path.join(a.out, "BSDB_HO")
    EXCL = {l.split()[0] for l in open(a.exclusion) if l.strip() and not l.startswith("#")}
    sel = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.out, "sel", "manifest.tsv")), delimiter="\t"):
        sel[r["pdbid"]].append(r)
    os.makedirs(out_root, exist_ok=True)
    g = dict(cifdir=cifdir, EXCL=EXCL, sel=sel, out_root=out_root)
    with Pool(a.nproc, set_globals, (g,)) as p, open(os.path.join(a.out, "write_report.tsv"), "w") as f:
        f.write("pdbid\tstatus\n")
        for pid, status in p.imap_unordered(write_one, sorted(sel), chunksize=4):
            f.write(f"{pid}\t{status}\n")


# ---------------------------------------------------------------- embeddings and features
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


MIN_NUM_PTS = 10


_NUM = re.compile(r"-?\d+\.\d{3}")


class LoudParseError(ValueError):
    pass


def coords(line):
    """x, y, z as the three 3-decimal numbers in columns 27-54 (strict or one-column-left layout); raises otherwise."""
    v = _NUM.findall(line[26:54])
    if len(v) != 3:
        raise ValueError(f"cannot read three coordinates from: {line.rstrip()}")
    return float(v[0]), float(v[1]), float(v[2])


def residue_records(pdb_path):
    """ATOM records grouped by (chain, resSeq, iCode), in file order."""
    res, order = {}, []
    with open(pdb_path) as f:
        for ln, line in enumerate(f, 1):
            if not line.startswith("ATOM"):
                continue
            try:
                chain, resseq, icode = line[21], int(line[22:26]), line[26].strip()
                resn, name = line[17:20].strip(), line[12:16].strip()
                xyz = np.array(coords(line))
            except ValueError as e:
                raise LoudParseError(f"{pdb_path}:{ln}: {e}") from e
            key = (chain, resseq, icode)
            r = res.get(key)
            if r is None:
                r = res[key] = {"resn": resn, "atoms": {}}
                order.append(key)
            elif r["resn"] != resn:
                raise LoudParseError(f"{pdb_path}:{ln}: two residue names at {key}: {r['resn']} / {resn}")
            if name in r["atoms"]:
                raise LoudParseError(f"{pdb_path}:{ln}: duplicate atom {name} at {key}")
            r["atoms"][name] = xyz
    return res, order


def chain_sequences(pdb_path):
    """{chain: sequence}, {chain: [residue keys]} over the 20 standard residues; order (resSeq, first appearance)."""
    res, order = residue_records(pdb_path)
    rank = {k: i for i, k in enumerate(order)}
    per_chain = defaultdict(list)
    for k in order:
        if res[k]["resn"] in VALID_RES and k[0].strip():
            per_chain[k[0]].append(k)
    keys, seqs = {}, {}
    for ch in sorted(per_chain):
        ks = sorted(per_chain[ch], key=lambda k: (k[1], rank[k]))
        keys[ch] = ks
        seqs[ch] = "".join(THREE_TO_ONE[res[k]["resn"]] for k in ks)
    return seqs, keys, res


def pocket_keys(pocket_path):
    keys, names = set(), {}
    with open(pocket_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[17:20].strip() in VALID_RES:
                k = (line[21], int(line[22:26]), line[26].strip())
                keys.add(k)
                names[k] = line[17:20].strip()
    return keys, names


def featurize_site(pocket_path, protein_path, ankh_dir, get_frames):
    """Return (node_feats, rotations, translations, node_keys) or raise LoudParseError / return (None, reason)."""
    pdbid = os.path.basename(os.path.dirname(protein_path))
    pk, pnames = pocket_keys(pocket_path)
    if len(pk) < MIN_NUM_PTS:
        return None, f"residues<{MIN_NUM_PTS} ({len(pk)})"
    seqs, keys, res = chain_sequences(protein_path)
    nodes = sorted(pk, key=lambda k: f"{pnames[k]}{k[0]}{k[1]:04d}{k[2]}")
    n = len(nodes)
    node_feats = np.empty((n, 768), dtype=np.float32)
    cn, ca, cc = (np.empty((n, 3), dtype=np.float32) for _ in range(3))
    emb, index = {}, {}
    for ch in {k[0] for k in nodes}:
        f = os.path.join(ankh_dir, f"{pdbid}_chain_{ch}.npy")
        if not os.path.exists(f):
            return None, f"missing embedding {os.path.basename(f)}"
        emb[ch] = np.load(f)
        if ch not in keys or emb[ch].shape[0] != len(keys[ch]):
            return None, f"embedding length mismatch chain {ch}"
        index[ch] = {k: i for i, k in enumerate(keys[ch])}
    for i, k in enumerate(nodes):
        r = res.get(k)
        if r is None or r["resn"] != pnames[k]:
            return None, f"pocket residue {k} not in protein.pdb"
        node_feats[i] = emb[k[0]][index[k[0]][k]]
        try:
            cn[i], ca[i], cc[i] = r["atoms"]["N"], r["atoms"]["CA"], r["atoms"]["C"]
        except KeyError as e:
            return None, f"missing backbone atom {e} in {k}"
    translations, rotations = get_frames(cn, ca, cc)
    return (node_feats, rotations, translations, nodes), ""


def cmd_embed(a):
    import torch
    from lenseek.ankh import embed_sequence, load_ankh
    root, out_dir = os.path.join(a.out, "BSDB_HO"), os.path.join(a.out, "ankh_holdout")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_ankh(device, a.ankh_dir)

    def get_ankh_embed(seq, name, device):
        """The embedding, or None (with a message) if the model call fails."""
        try:
            return embed_sequence(seq, model, tok, device)
        except Exception as e:
            print(f"get_ankh_embed failed for {name}: {e}")
            return None

    t0, n, nres, fails = time.time(), 0, 0, []
    prots = sorted(glob.glob(os.path.join(root, "*", "protein.pdb")))
    for k, prot in enumerate(prots):
        pid = os.path.basename(os.path.dirname(prot))
        seqs = chain_sequences(prot)[0]
        for ch, seq in seqs.items():
            out = os.path.join(out_dir, f"{pid}_chain_{ch}.npy")
            if os.path.exists(out):
                continue
            emb = get_ankh_embed(seq, f"{pid}_chain_{ch}", device)
            if emb is None:
                fails.append(f"{pid}_chain_{ch}"); continue
            np.save(out, emb)
            n += 1; nres += len(seq)
        if k % 200 == 0:
            print(f"{k}/{len(prots)} proteins, {n} chains, {nres} residues, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {n} chains, {nres} residues, {time.time() - t0:.0f} s, failures {len(fails)} {fails[:20]}", flush=True)


def cmd_features(a):
    from lenseek.featurize import residue_frames
    query_root, ankh_dir = a.query_root or os.path.join(a.out, "BSDB_HO"), os.path.join(a.out, "ankh_holdout")
    queries, skipped = [], []
    for pocket in sorted(glob.glob(os.path.join(query_root, "*", "*pocket*.pdb"))):
        protein = os.path.join(os.path.dirname(pocket), "protein.pdb")
        try:
            out, why = featurize_site(pocket, protein, ankh_dir, residue_frames)
        except LoudParseError as e:
            skipped.append((pocket, f"parse error: {e}"))
            print("PARSE ERROR", e, flush=True)
            continue
        if out is None:
            skipped.append((pocket, why))
            continue
        h, R, T, nodes = out
        queries.append({"path": pocket, "h": h, "R": R, "T": T,
                        "residues": [f"{c}:{r}{i}" for c, r, i in nodes]})
    with open(os.path.join(a.out, "query_cache_holdout.pkl"), "wb") as f:
        pickle.dump({"queries": queries, "skipped": skipped, "node_features": "ankh",
                     "featuriser": "scripts/holdout.py features (strict columns, insertion codes kept)"},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"queries cached: {len(queries)}; skipped: {len(skipped)}")
    for p, why in skipped[:60]:
        print(f"  skip {os.path.relpath(p, query_root)}: {why}")
    # manifest.tsv = selection manifest + featurisation status (the scored sites are the featurised sites)
    feat = {os.path.relpath(q["path"], query_root): q for q in queries}
    skip = {os.path.relpath(p, query_root): why for p, why in skipped}
    sel_rows = list(csv.DictReader(open(os.path.join(a.out, "sel", "manifest.tsv")), delimiter="\t"))
    cols = list(sel_rows[0]) + ["featurised", "skip_reason", "n_nodes"]
    with open(os.path.join(a.out, "manifest.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for s in sel_rows:
            rel = os.path.join(s["pdbid"], f"ligand_pocket_{s['idx']}_{s['suffix']}.pdb")
            q = feat.get(rel)
            s.update(featurised=int(q is not None), skip_reason=skip.get(rel, ""), n_nodes=q["h"].shape[0] if q else "")
            f.write("\t".join(str(s[c]) for c in cols) + "\n")


# ---------------------------------------------------------------- 50 % sequence clusters
STD20 = {k for k, v in AA.items() if k not in ("MSE", "SEC", "PYL")}
CLUSTER_LEVELS, CLUSTER_MIN_COV = (50, 30), 0.50


def pocket_counts_icode(pocket_file):
    """Pocket residues per chain: distinct (chain, resSeq, iCode) over standard-residue ATOM lines."""
    keys = set()
    for l in open(pocket_file):
        if l.startswith("ATOM") and l[17:20].strip() in STD20:
            keys.add((l[21], int(l[22:26]), l[26].strip()))
    return collections.Counter(k[0] for k in keys)


def sites_from_manifest(manifest, root):
    out, mism = [], []
    for r in read_tsv(manifest):
        if r.get("featurised", "1") != "1":
            continue
        key = r["site_id"] + ".pdb"
        cnt = pocket_counts_icode(os.path.join(root, key))
        order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        if order and order[0][0] != r["primary_chain"]:
            mism.append((key, order[0][0], r["primary_chain"]))
        out.append(dict(key=key, chains=",".join(f"{c}:{n}" for c, n in sorted(cnt.items())),
                        primary_chain=r["primary_chain"]))
    return out, mism


def cmd_clusters(a):
    """Chain sequences of every entry under the root -> chains.fasta; all-vs-all MMseqs2 (or --m8); chains linked at
    fident >= level over >= 50 % of the shorter chain, single linkage; a site joins the cluster of its primary
    chain (then the chain with most pocket residues); clusters c000, c001, ... by size, then by smallest chain name."""
    root = a.root or os.path.join(a.out, "BSDB_HO")
    a.out_dir = os.path.join(a.out, "clusters")
    os.makedirs(a.out_dir, exist_ok=True)
    sites, mism = sites_from_manifest(a.manifest or os.path.join(a.out, "manifest.tsv"), root)
    if mism:
        print(f"[warn] {len(mism)} sites whose top pocket chain differs from manifest primary_chain; the manifest "
              f"value is used first: {mism[:10]}")
    with open(os.path.join(a.out_dir, "sites.tsv"), "w") as f:
        f.write("key\tchains\n")
        for s in sites:
            f.write(f"{s['key']}\t{s['chains']}\n")

    fasta = os.path.join(a.out_dir, "chains.fasta")
    pdbs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "protein.pdb")))
    seqs = {}
    with open(fasta, "w") as f:
        for p in pdbs:
            for c, s in chain_seqs(os.path.join(root, p, "protein.pdb")).items():
                f.write(f">{p}_{c}\n{s}\n")
                seqs[f"{p}_{c}"] = s
    ver = subprocess.run([MMSEQS, "version"], capture_output=True, text=True).stdout.strip()
    if a.m8:
        m8 = a.m8
        open(os.path.join(a.out_dir, "all_vs_all.cmd"), "w").write(f"# reused search {os.path.abspath(m8)}\n")
    else:
        m8 = os.path.join(a.out_dir, "all_vs_all.m8")
        tmp = os.path.join(a.out_dir, "mmseqs_tmp")
        cmd = [MMSEQS, "easy-search", fasta, fasta, m8, tmp, "--format-output",
               "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits", "-e", "1e-3", "--threads", str(a.threads)]
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(["rm", "-rf", tmp])
        open(os.path.join(a.out_dir, "all_vs_all.cmd"), "w").write(f"# MMseqs2 {ver}\n" + " ".join(cmd) + "\n")
    hits = [l.split("\t")[:6] for l in open(m8)]

    for pct in CLUSTER_LEVELS:
        root_of = single_linkage(seqs, hits, pct, CLUSTER_MIN_COV)
        rows = []
        for s in sites:
            pdb = s["key"].split("/")[0]
            order = sorted([kv.split(":") for kv in s["chains"].split(",") if kv], key=lambda kv: (-int(kv[1]), kv[0]))
            if s.get("primary_chain"):   # manifest value first, then the counts
                order = [[s["primary_chain"], "0"]] + [kv for kv in order if kv[0] != s["primary_chain"]]
            main = order[0][0] if order else ""
            chain = next((f"{pdb}_{c}" for c, _ in order if f"{pdb}_{c}" in seqs), f"{pdb}_{main}")
            rows.append(dict(query_key=s["key"], pdb=pdb, main_chain=main, cluster_chain=chain,
                             root=root_of.get(chain, chain)))
        size = collections.Counter(r["root"] for r in rows)
        n_e = {rt: len({r["pdb"] for r in rows if r["root"] == rt}) for rt in size}
        # ties in size are ordered by the smallest chain name of the component (independent of the m8 line order)
        comp_min = {}
        for c, r in root_of.items():
            comp_min[r] = min(comp_min.get(r, c), c)
        roots = sorted(size, key=lambda r: (-size[r], comp_min.get(r, r)))
        name = {r: f"c{i:03d}" for i, r in enumerate(roots)}
        out = os.path.join(a.out_dir, f"seq_clusters_{pct}.tsv")
        with open(out, "w") as f:
            f.write(f"# chains linked when they align with fident >= {pct / 100} over >= {CLUSTER_MIN_COV} of the shorter "
                    f"chain ({os.path.basename(m8)}); single linkage; pocket -> cluster of its main chain (most pocket "
                    f"residues); {ver}\n")
            f.write("query_key\tpdb\tmain_chain\tcluster_chain\tcluster\tcluster_n_queries\tcluster_n_entries\n")
            for r in rows:
                f.write(f"{r['query_key']}\t{r['pdb']}\t{r['main_chain']}\t{r['cluster_chain']}\t{name[r['root']]}\t"
                        f"{size[r['root']]}\t{n_e[r['root']]}\n")
        fb = sum(r["cluster_chain"] != r["pdb"] + "_" + r["main_chain"] for r in rows)
        print(f"{pct}%: {len(rows)} sites -> {len(size)} clusters; fallback main chains {fb}; {ver}")


# ---------------------------------------------------------------- descriptive strata
def convert_train_one(pid):
    """BsitePDB protein.pdb (no insertion-code column) -> standard columns, a blank iCode inserted at column 27."""
    src = os.path.join(root, pid, "protein.pdb")
    dst = os.path.join(out, f"{pid}.pdb.gz")
    if os.path.exists(dst):
        return pid, "exists"
    if not os.path.exists(src):
        return pid, "missing"
    n = 0
    with open(src) as f, gzip.open(dst, "wt", compresslevel=3) as g:
        for l in f:
            if l.startswith("ATOM"):
                l = l.rstrip("\n")
                g.write(l[:26] + " " + l[26:] + "\n"); n += 1
            elif l.startswith(("TER", "END")):
                g.write(l)
    return pid, f"ok {n}"


def strata_structure(_args):  # shared: scripts/site_annotation.py strata
    """Foldseek easy-search (3Di+AA, --alignment-type 2, -e 10) of every chain of the selected proteins against the
    training chains; site value = maximum query TM-score over its pocket chains."""
    man, root, tdb, work, out, threads = _args.man, _args.root, _args.tdb, _args.work, _args.out, _args.threads
    FS = _args.foldseek
    sites = list(csv.DictReader(open(man), delimiter="\t"))
    qdir = os.path.join(work, "query_pdb")
    os.makedirs(qdir, exist_ok=True)
    for pid in sorted({s["pdbid"] for s in sites}):
        dst = os.path.join(qdir, f"{pid}.pdb")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(root, pid, "protein.pdb")), dst)
    m8 = os.path.join(work, "holdout_vs_train_tm.m8")
    fields = "query,target,fident,alnlen,qlen,tlen,qtmscore,ttmscore,alntmscore,evalue,qcov,tcov"
    subprocess.run([FS, "easy-search", qdir, tdb, m8, os.path.join(work, "tmp"), "--alignment-type", "2", "-e", "10",
                    "--chain-name-mode", "1", "--threads", threads, "--format-output", fields], check=True,
                   stdout=open(os.path.join(work, "search.log"), "w"), stderr=subprocess.STDOUT)
    best = {}
    for line in open(m8):
        p = line.rstrip("\n").split("\t")
        q, t, qtm = p[0], p[1], float(p[6])
        qp, qc = q.rsplit("_", 1)
        tp, tc = t.rsplit("_", 1)
        key = (qp.replace(".pdb", ""), qc)
        if qtm > best.get(key, (-1.0, ""))[0]:
            best[key] = (qtm, f"{tp.replace('.pdb', '')}_{tc}")

    def stratum(x):
        return "lt0.5" if x < 0.5 else ("0.5to0.8" if x < 0.8 else "ge0.8")

    with open(out, "w") as f:
        f.write("site_id\tmax_tm_train\tnearest_train_chain_tm\tquery_chain_tm\tstructure_stratum\n")
        for s in sites:
            v, near, via = 0.0, "", ""
            for c in s["pocket_chains"]:
                x, t = best.get((s["pdbid"], c), (0.0, ""))
                if x > v:
                    v, near, via = x, t, c
            f.write(f"{s['site_id']}\t{round(v, 4)}\t{near}\t{via}\t{stratum(v)}\n")
    print(f"query chains with a hit: {len(best)}; sites: {len(sites)}")


def strata_ligand(_args):  # shared: scripts/site_annotation.py strata
    """Ligand novelty: same CCD as a database site / new CCD with maximum Dice >= 0.46 (Morgan r2, 2048 bits) to a
    database ligand / below; CCDs missing from the SMILES cache are fetched from RCSB (network) and saved."""
    import datetime
    import requests
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    man, codes_path, cache_path, out, fetched_path = (_args.man, _args.codes_path, _args.cache_path, _args.out,
                                                      _args.fetched_path)
    THR = 0.46
    train_ccd = {r["ccd"] for r in csv.DictReader(open(codes_path), delimiter="\t") if r["set"] == "db"}
    smiles = {r["comp_id"]: r["smiles"] for r in csv.DictReader(open(cache_path), delimiter="\t")}
    sites = list(csv.DictReader(open(man), delimiter="\t"))
    need = sorted({s["ligand_ccd"] for s in sites} - set(smiles))
    Q = "query($ids:[String!]!){ chem_comps(comp_ids:$ids){ chem_comp{ id } rcsb_chem_comp_descriptor{ SMILES } } }"
    got = {}
    for k in range(0, len(need), 200):
        r = requests.post("https://data.rcsb.org/graphql", json={"query": Q, "variables": {"ids": need[k:k + 200]}}, timeout=300)
        r.raise_for_status()
        for c in r.json()["data"]["chem_comps"] or []:
            if c and c.get("rcsb_chem_comp_descriptor") and c["rcsb_chem_comp_descriptor"].get("SMILES"):
                got[c["chem_comp"]["id"]] = c["rcsb_chem_comp_descriptor"]["SMILES"]
    with open(fetched_path, "w") as f:
        f.write(f"# RCSB GraphQL rcsb_chem_comp_descriptor.SMILES fetched {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write("comp_id\tsmiles\n")
        for c in sorted(got):
            f.write(f"{c}\t{got[c]}\n")
    smiles.update(got)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(c):
        s = smiles.get(c)
        m = Chem.MolFromSmiles(s) if s else None
        return gen.GetFingerprint(m) if m is not None else None

    tr_codes = sorted(c for c in train_ccd if fp(c) is not None)
    tr_fps = [fp(c) for c in tr_codes]
    print(f"training CCDs {len(train_ccd)}, with fingerprint {len(tr_codes)}; evaluation-set CCDs {len({s['ligand_ccd'] for s in sites})}, "
          f"fetched SMILES {len(got)} of {len(need)} missing")
    res = {}
    for c in sorted({s["ligand_ccd"] for s in sites}):
        f = fp(c)
        if f is None:
            res[c] = ("", "", "same_ccd" if c in train_ccd else "no_fingerprint")
            continue
        d = np.array(DataStructs.BulkDiceSimilarity(f, tr_fps))
        j = int(d.argmax())
        mx = float(d[j])
        st = "same_ccd" if c in train_ccd else ("new_ccd_similar" if mx >= THR else "new_ccd_novel")
        res[c] = (round(mx, 4), tr_codes[j], st)
    with open(out, "w") as f:
        f.write("site_id\tligand_ccd\tccd_in_training\tmax_dice_train\tnearest_train_ccd\tligand_stratum\n")
        for s in sites:
            c = s["ligand_ccd"]
            mx, near, st = res[c]
            f.write(f"{s['site_id']}\t{c}\t{int(c in train_ccd)}\t{mx}\t{near}\t{st}\n")


def train_foldseek_db(ids, bsdb_root, sd, foldseek, nproc):  # shared: scripts/site_annotation.py strata
    """Foldseek database of the chains of the training proteins ids (protein.pdb in standard columns) -> sd/train_db."""
    out = os.path.join(sd, "train_pdb")
    os.makedirs(out, exist_ok=True)
    with Pool(nproc, set_globals, (dict(root=bsdb_root, out=out),)) as p:
        bad = [r for r in p.imap_unordered(convert_train_one, ids, chunksize=32)
               if not r[1].startswith(("ok", "exists"))]
    print("training proteins missing:", len(bad))
    tdb = os.path.join(sd, "train_db", "train")
    os.makedirs(os.path.dirname(tdb), exist_ok=True)
    subprocess.run([foldseek, "createdb", out, tdb, "--chain-name-mode", "1"], check=True)
    return tdb


def cmd_strata(a):
    sd, man = os.path.join(a.out, "strata"), os.path.join(a.out, "sel", "manifest.tsv")
    os.makedirs(sd, exist_ok=True)
    tdb = a.train_db or train_foldseek_db([l.strip() for l in open(a.train_ids) if l.strip()], a.bsdb_root, sd,
                                          a.foldseek, a.nproc)
    strata_structure(argparse.Namespace(man=man, root=os.path.join(a.out, "BSDB_HO"), tdb=tdb, work=os.path.join(sd, "foldseek"),
                                        out=os.path.join(sd, "structure.tsv"), threads=str(a.threads), foldseek=a.foldseek))
    strata_ligand(argparse.Namespace(man=man, codes_path=a.ligand_codes, cache_path=a.smiles, out=os.path.join(sd, "ligand.tsv"),
                                     fetched_path=os.path.join(sd, "fetched_smiles.tsv")))


# ---------------------------------------------------------------- construction counts
def cmd_counts(a):
    """Counts of Supplementary Section S5 from the outputs of build, select, features and strata."""
    path = lambda given, *rel: given or os.path.join(a.out, *rel)
    ent = read_tsv(path(a.entries, "build", "entries.tsv"))
    valid = read_tsv(path(a.manifest_all, "sel", "manifest_all.tsv"))
    feat = [r for r in read_tsv(path(a.manifest, "manifest.tsv")) if r["featurised"] == "1"]
    struct = read_tsv(path(a.structure, "strata", "structure.tsv"))
    lig = read_tsv(path(a.ligand, "strata", "ligand.tsv"))
    dedup = [r for r in valid if r["dedup_representative"] == "1"]
    sampled = [r for r in valid if r["selected"] == "1"]
    n = lambda rows, col: len({r[col] for r in rows})
    by = lambda label, rows, col, keys: [(f"{label} {k}", sum(r[col] == k for r in rows)) for k in keys]
    ident = ["lt30", "30to80", "ge80"]
    rows = ([("entries built", sum(r["status"] == "ok" for r in ent)),
             ("valid sites", len(valid)), ("valid sites: entries", n(valid, "pdbid")),
             ("de-duplicated sites", len(dedup)), ("de-duplicated sites: 80 % clusters", n(dedup, "cluster80"))] +
            by("de-duplicated sites: identity", dedup, "stratum", ident) +
            [("sampled sites", len(sampled)), ("sampled sites: entries", n(sampled, "pdbid")),
             ("sampled sites: 80 % clusters", n(sampled, "cluster80"))] +
            by("sampled sites: identity", sampled, "stratum", ident) +
            by("sampled sites: max TM-score", struct, "structure_stratum", ["lt0.5", "0.5to0.8", "ge0.8"]) +
            by("sampled sites: ligand", lig, "ligand_stratum",
               ["same_ccd", "new_ccd_similar", "new_ccd_novel", "no_fingerprint"]) +
            [("featurised sites", len(feat))] + by("featurised sites: identity", feat, "stratum", ident))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "counts.tsv"), "w") as f:
        f.write("count\tvalue\n" + "".join(f"{k}\t{v}\n" for k, v in rows))
    w = max(len(k) for k, _ in rows)
    print("\n".join(f"{k:{w}s}  {v:,}" for k, v in rows))


# ---------------------------------------------------------------- the build rules on entries that BsitePDB contains
def bsdb_sites(pdbid):
    """BsitePDB sites of one entry: residue name, number and first-atom coordinates of each ligand file (BsitePDB
    columns) and the (chain, number) residues of its binding-site file."""
    out = []
    for fn in sorted(glob.glob(os.path.join(bsdb, pdbid, "ligand_[0-9]*_*.pdb"))):
        L = [l for l in open(fn) if l.startswith(("HETATM", "ATOM"))]
        pk = fn.replace("/ligand_", "/ligand_pocket_")
        pres = {(l[21], int(l[22:26])) for l in open(pk) if l.startswith("ATOM")} if os.path.exists(pk) else set()
        if not L:
            out.append(dict(resn="", resi=None, xyz=None, pocket=pres)); continue
        l = L[0]
        out.append(dict(resn=l[17:20].strip(), resi=int(l[22:26]),
                        xyz=(float(l[29:37]), float(l[37:45]), float(l[45:53])), pocket=pres))
    return out


def validate_one(pdbid):
    """Every ligand decision of build_entry, matched to an unused BsitePDB ligand with the same residue number, the
    same name (UNL for a five-character CCD) and a first atom within 0.0015 A; an accepted, matched site gets the
    Jaccard index of its (chain, number) binding-site residues. BsitePDB ligands left unmatched are added."""
    path = os.path.join(cifdir, f"{pdbid}.cif.gz")
    if not os.path.exists(path):
        return [(pdbid, "", "", "", "", "no_file", 0, "")]
    try:
        sites, dec, atoms = build_entry(path, EXCL)
    except Exception as e:
        return [(pdbid, "", "", "", "", f"error:{type(e).__name__}", 0, "")]
    ref = bsdb_sites(pdbid) if os.path.isdir(os.path.join(bsdb, pdbid)) else []
    rows, used = [], set()
    for d in dec:
        k = d["key"]
        a0 = next(a for a in atoms if a["rec"] == "HETATM" and (a["chain"], a["resi"], a["icode"], a["resn"]) == k)
        match = None
        for j, r in enumerate(ref):
            if j in used or r["xyz"] is None:
                continue
            if r["resi"] == k[1] and (r["resn"] == k[3] or (len(k[3]) > 3 and r["resn"] == "UNL")) and \
                    max(abs(r["xyz"][0] - a0["x"]), abs(r["xyz"][1] - a0["y"]), abs(r["xyz"][2] - a0["z"])) < 0.0015:
                match = j
                break
        if match is not None:
            used.add(match)
        jac = ""
        if d["reason"] == "ok" and match is not None:
            mine, theirs = {(c, r) for c, r, _ in d["pocket"]}, ref[match]["pocket"]
            u = mine | theirs
            jac = round(len(mine & theirs) / len(u), 4) if u else 1.0
        rows.append((pdbid, k[0], k[1], k[2], k[3], d["reason"], int(match is not None), jac))
    rows += [(pdbid, "?", r["resi"], "", r["resn"], "bsdb_unmatched", 1, "")
             for j, r in enumerate(ref) if j not in used]
    return rows


def cmd_validate(a):
    """Recall of the BsitePDB sites of the entries in --ids and the mean Jaccard index of the recovered sites."""
    ids = [l.strip().lower() for l in open(a.ids) if l.strip()]
    g = dict(cifdir=a.cifdir, bsdb=a.bsdb_root,
             EXCL={l.split()[0] for l in open(a.exclusion) if l.strip() and not l.startswith("#")})
    d = os.path.join(a.out, "validate")
    os.makedirs(d, exist_ok=True)
    with Pool(a.nproc, set_globals, (g,)) as p:
        rows = [r for rows in p.imap(validate_one, ids, chunksize=4) for r in rows]
    with open(os.path.join(d, "sites.tsv"), "w") as f:
        f.write("pdbid\tchain\tresi\ticode\tresn\treason\tin_bsdb\tjaccard\n")
        f.write("".join("\t".join(map(str, r)) + "\n" for r in rows))
    ref = [r for r in rows if r[6] == 1]
    hit = [r for r in ref if r[5] == "ok"]
    summary = [("entries", len(ids)),
               ("entries without a file or failing to build", sum(r[5] == "no_file" or r[5].startswith("error")
                                                                  for r in rows)),
               ("BsitePDB sites", len(ref)), ("recovered by the build rules", len(hit)),
               ("recall", len(hit) / len(ref) if ref else np.nan),
               ("mean Jaccard index of the recovered sites", float(np.mean([r[7] for r in hit])) if hit else np.nan)]
    with open(os.path.join(d, "summary.tsv"), "w") as f:
        f.write("count\tvalue\n" + "".join(f"{k}\t{v}\n" for k, v in summary))
    print("\n".join(f"{k:43s} {v:.4f}" if isinstance(v, float) else f"{k:43s} {v:,}" for k, v in summary))


# ---------------------------------------------------------------- score: the one run over the sites
def checkpoint(given, asset):
    """A checkpoint path: the one given, else the released asset (lenseek.assets: $LENSEEK_HOME,
    $LENSEEK_ASSET_MIRROR)."""
    if given:
        return given
    from lenseek.assets import fetch
    return str(fetch(asset))


def cmd_score(a):
    path = lambda given, *rel: given or os.path.join(a.out, *rel)  # noqa: E731
    out = os.path.abspath(path(a.run, "run"))
    strict_dir = a.strict_eval_dir or os.path.join(os.environ.get("LENSEEK_EVAL_ROOT", ""), "out_strict30_s0")
    written = "written by `bash examples/holdout.sh {}` below $LENSEEK_OUT"
    need = [("--query-cache", path(a.query_cache, "query_cache_holdout.pkl"), written.format("features")),
            ("--manifest", path(a.manifest, "manifest.tsv"), written.format("features")),
            ("--clusters", path(a.clusters, "clusters", "seq_clusters_50.tsv"), written.format("clusters"))]
    if not a.prepare_only:
        for flag, v in (("--main-eval-dir", a.main_eval_dir), ("--dt-dir", a.dt_dir), ("--bsdb-root", a.bsdb_root)):
            if not v:
                sys.exit(f"{flag} not given and its environment variable is not set (score -h)")
        need.append(("--root", path(a.root, "BSDB_HO"), written.format("write")))
        for flag, d, files, source in (
                ("--main-eval-dir", a.main_eval_dir, ("db_paths.txt", "db_mu.npy", "db_mask.npy"),
                 "$LENSEEK_EVAL_DIR, database encodings of scripts/evaluate.py"),
                ("--strict-eval-dir", strict_dir, ("db_paths.txt", "db_mu.npy", "db_mask.npy"),
                 "$LENSEEK_EVAL_ROOT/out_strict30_s0, database encodings of scripts/evaluate.py"),
                ("--dt-dir", a.dt_dir, ("db_desc.npy",), "$LENSEEK_DT_DIR, scripts/deeplytough.py assemble"),
                ("--dt-queries", path(a.dt_queries, "deeplytough"), ("q_desc.npy", "q_paths.txt"),
                 "DT_QUERIES in examples/holdout.sh")):
            need += [(flag, os.path.join(d, f), source) for f in files]
    for flag, p, source in need:            # every input before the run folder is created
        if not os.path.exists(p):
            die(f"{flag}: {p} does not exist ({source})")
    if not os.path.isdir(os.path.dirname(out)):
        die(f"--run: {os.path.dirname(out)} does not exist")
    os.mkdir(out)                           # never overwrite a run
    cq = pickle.load(open(path(a.query_cache, "query_cache_holdout.pkl"), "rb"))["queries"]
    c_keys = [site_key(q["path"]) for q in cq]
    mk = {r["site_id"] + ".pdb": r for r in read_tsv(path(a.manifest, "manifest.tsv")) if r.get("featurised") == "1"}
    clus = {r["query_key"]: r["cluster"] for r in read_tsv(path(a.clusters, "clusters", "seq_clusters_50.tsv"))}
    qrows = [dict(q_idx=i, key=k, entry=mk[k]["pdbid"].lower(), cluster=clus[k], ligand=mk[k]["ligand_ccd"].strip(),
                  stratum=mk[k]["stratum"]) for i, k in enumerate(c_keys)]
    q_tsv = os.path.join(out, "queries.tsv")
    with open(q_tsv, "w") as f:
        f.write("q_idx\tkey\tentry\tcluster\tligand\tstratum\n")
        for r in qrows:
            f.write(f"{r['q_idx']}\t{r['key']}\t{r['entry']}\t{r['cluster']}\t{r['ligand']}\t{r['stratum']}\n")
    if a.prepare_only:
        return
    root = path(a.root, "BSDB_HO")
    models = [dict(name="main_s0", ckpt=checkpoint(a.main_checkpoint, "checkpoint_main_s0"), db_dir=a.main_eval_dir),
              dict(name="strict30_s0", ckpt=checkpoint(a.strict_checkpoint, "checkpoint_strict30_s0"),
                   db_dir=strict_dir)]
    db_paths = open(os.path.join(a.main_eval_dir, "db_paths.txt")).read().split()
    ref_db = [site_key(p) for p in db_paths]
    if [site_key(p) for p in open(os.path.join(strict_dir, "db_paths.txt")).read().split()] != ref_db:
        sys.exit(f"{strict_dir}/db_paths.txt does not list the database sites of {a.main_eval_dir} in the same order")
    n_db = len(ref_db)
    log(f"{len(qrows)} queries, {len({r['entry'] for r in qrows})} entries, {len(set(clus.values()))} clusters; "
        f"DB {n_db} sites")

    # encodings and distances of both models (database encodings as stored; scripts/evaluate.py)
    import torch
    from evaluate import load_encoder, write_sites
    from lenseek.search import chamfer_rows_cdist
    torch.set_num_threads(a.threads)
    sites = [(q["h"], q["T"], q["R"]) for q in cq]
    dists = {}
    for m in models:
        mdir = os.path.join(out, f"model_{m['name']}")
        os.mkdir(mdir)
        enc, _ = load_encoder(m["ckpt"], a.device)
        q_mu, q_mask = write_sites(mdir, "q", enc, sites, [q["path"] for q in cq])
        dist = chamfer_rows_cdist(q_mu, q_mask, np.load(os.path.join(m["db_dir"], "db_mu.npy"), mmap_mode="r"),
                                  np.load(os.path.join(m["db_dir"], "db_mask.npy")), a.device, progress=True)
        if not np.isfinite(dist).all():
            sys.exit(f"{int((~np.isfinite(dist)).sum())} non-finite distances")
        dists[m["name"]] = os.path.join(mdir, "dist.npy")
        np.save(dists[m["name"]], dist)
        log(f"{m['name']}: distance matrix {dist.shape}")
    dt_q = path(a.dt_queries, "deeplytough")
    qd = np.load(os.path.join(dt_q, "q_desc.npy")).astype(np.float64)
    qdp = [site_key(p) for p in open(os.path.join(dt_q, "q_paths.txt")).read().split()]
    pos = {k: i for i, k in enumerate(qdp)}
    X = np.load(os.path.join(a.dt_dir, "db_desc.npy")).astype(np.float64)
    okd = np.isfinite(X).all(1)
    Q = np.full((len(c_keys), X.shape[1]), np.nan)
    for i, k in enumerate(c_keys):
        if k in pos:
            Q[i] = qd[pos[k]]
    okq = np.isfinite(Q).all(1)
    d = np.full((len(c_keys), n_db), np.nan, np.float32)
    x2 = (Q[okq] ** 2).sum(1)[:, None]
    y2 = (X[okd] ** 2).sum(1)[None, :]
    d[np.ix_(okq, okd)] = np.sqrt(np.clip(x2 + y2 - 2.0 * Q[okq] @ X[okd].T, 0, None) + 1e-8).astype(np.float32)
    dists["deeplytough"] = os.path.join(out, "dist_deeplytough.npy")
    np.save(dists["deeplytough"], d)
    log(f"DeeplyTough: {int(okq.sum())}/{len(c_keys)} queries with a descriptor")

    # ProBiS / G-LoSA pair lists: top-100 of main_s0 (distance, DB index), DeeplyTough (distance, DB index), and
    # their union; structure paths below --root and --bsdb-root
    K = 100
    D = np.load(dists["main_s0"], mmap_mode="r")
    query_path = lambda q: os.path.join(root, site_key(q["path"]))  # noqa: E731
    target_path = lambda j: os.path.join(a.bsdb_root, ref_db[j])  # noqa: E731
    ptsv = os.path.join(out, f"pairs_main_s0_top{K}.tsv")
    with open(ptsv, "w") as f:
        f.write("query_idx\tquery_path\trank\tdb_idx\ttarget_path\tdist\n")
        for i, q in enumerate(cq):
            row = np.asarray(D[i], dtype=np.float64)
            order = np.argsort(np.where(np.isfinite(row), row, np.inf), kind="stable")[:K]
            qp = query_path(q)
            for rnk, j in enumerate(order, 1):
                f.write(f"{i}\t{qp}\t{rnk}\t{int(j)}\t{target_path(j)}\t{row[j]:.6f}\n")
    union = {}
    for l in open(ptsv).read().splitlines()[1:]:
        p_ = l.split("\t")
        union[(p_[1], p_[4])] = [p_[2], ""]
    Ddt = np.load(dists["deeplytough"], mmap_mode="r")
    with open(os.path.join(out, f"pairs_deeplytough_top{K}.tsv"), "w") as f:
        f.write("query_idx\tquery_path\trank\tdb_idx\ttarget_path\tdist\n")
        for i, q in enumerate(cq):
            if not okq[i]:
                continue
            d_ = np.asarray(Ddt[i], np.float64)
            idx = np.nonzero(np.isfinite(d_))[0]
            o = idx[np.lexsort((idx, d_[idx]))][:K]
            qp = query_path(q)
            for rnk, j in enumerate(o, 1):
                tp = target_path(j)
                f.write(f"{i}\t{qp}\t{rnk}\t{int(j)}\t{tp}\t{d_[j]:.9f}\n")
                union.setdefault((qp, tp), ["", ""])[1] = str(rnk)
    utsv = os.path.join(out, "pairs_union_for_labels.tsv")
    with open(utsv, "w") as f:
        f.write("query_path\ttarget_path\trank_main_s0\trank_deeplytough\n")
        for (qp, tp), (r1, r3) in sorted(union.items()):
            f.write(f"{qp}\t{tp}\t{r1}\t{r3}\n")
    log(f"  union to label {utsv}: {len(union)} pairs")


# ---------------------------------------------------------------- CLI
def cmd_entries(a):
    m = os.path.join(a.out, "meta")
    os.makedirs(m, exist_ok=True)
    rcsb_search_window(argparse.Namespace(frm=a.date_from, to=a.date_to, out=os.path.join(m, "ids_window.txt")))
    rcsb_entry_meta(argparse.Namespace(ids_path=os.path.join(m, "ids_window.txt"),
                                       out_path=os.path.join(m, "meta_window.jsonl"), batch=150))
    select_entries(os.path.join(m, "meta_window.jsonl"), os.path.join(m, "entries_to_build.txt"), a.validation)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    excl = os.path.join(REPO, "data", "exclusion_ccd.txt")
    spec = {
        "entries": [("--date-from", dict(default="2023-11-07")), ("--date-to", dict(default="2026-10-31")),
                    ("--validation", dict(action="store_true", help="entries of validate: method and resolution only"))],
        "fetch": [("--ids", dict(default=None, help="default OUT/meta/entries_to_build.txt"))],
        "build": [("--ids", dict(default=None)), ("--cifdir", dict(default=None)), ("--exclusion", dict(default=excl)),
                  ("--bsitepdb-ids", dict(required=True, help="PDB IDs of BsitePDB and of the training list")),
                  ("--nproc", dict(type=int, default=8))],
        "identity": [("--train-fasta", dict(default=None)), ("--train-ids", dict(default=None)),
                     ("--bsdb-root", dict(default=os.environ.get("LENSEEK_BSDB_ROOT"))), ("--threads", dict(type=int, default=16))],
        "select": [("--meta", dict(default=None, help="default OUT/meta/meta_window.jsonl"))],
        "write": [("--cifdir", dict(default=None)), ("--exclusion", dict(default=excl)), ("--nproc", dict(type=int, default=8))],
        "embed": [("--ankh-dir", dict(default=None))],
        "features": [("--query-root", dict(default=None))],
        "clusters": [("--manifest", dict(default=None, help="default OUT/manifest.tsv (written by features)")),
                     ("--root", dict(default=None)), ("--m8", dict(default=None, help="reuse an all-vs-all search")),
                     ("--threads", dict(type=int, default=2))],
        "strata": [("--train-db", dict(default=None)), ("--train-ids", dict(default=None)),
                   ("--bsdb-root", dict(default=os.environ.get("LENSEEK_BSDB_ROOT"))),
                   ("--foldseek", dict(default=os.environ.get("FOLDSEEK", "foldseek"))),
                   ("--ligand-codes", dict(required=True, help="TSV set, idx, ccd ('db' rows: database sites)")),
                   ("--smiles", dict(required=True, help="TSV comp_id, smiles")), ("--threads", dict(type=int, default=6)),
                   ("--nproc", dict(type=int, default=8))],
        "counts": [("--entries", dict(default=None, help="default OUT/build/entries.tsv")),
                   ("--manifest-all", dict(default=None, help="default OUT/sel/manifest_all.tsv")),
                   ("--manifest", dict(default=None, help="default OUT/manifest.tsv (written by features)")),
                   ("--structure", dict(default=None, help="default OUT/strata/structure.tsv")),
                   ("--ligand", dict(default=None, help="default OUT/strata/ligand.tsv"))],
        "validate": [("--ids", dict(required=True, help="PDB IDs of entries that BsitePDB covers")),
                     ("--cifdir", dict(required=True, help="<id>.cif.gz of these entries")),
                     ("--bsdb-root", dict(default=os.environ.get("LENSEEK_EVAL_PERIOD_ROOT"),
                                          help="BsitePDB files of these entries (default $LENSEEK_EVAL_PERIOD_ROOT)")),
                     ("--exclusion", dict(default=excl)), ("--nproc", dict(type=int, default=8))],
        "score": [("--run", dict(default=None, help="new output folder (default OUT/run)")),
                  ("--prepare-only", dict(action="store_true", help="stop after queries.tsv (nothing encoded)")),
                  ("--query-cache", dict(default=None, help="default OUT/query_cache_holdout.pkl")),
                  ("--manifest", dict(default=None, help="default OUT/manifest.tsv")),
                  ("--clusters", dict(default=None, help="default OUT/clusters/seq_clusters_50.tsv")),
                  ("--root", dict(default=None, help="structures of the sites (default OUT/BSDB_HO)")),
                  ("--main-checkpoint", dict(default=None, help="default: asset checkpoint_main_s0")),
                  ("--strict-checkpoint", dict(default=None, help="default: asset checkpoint_strict30_s0")),
                  ("--main-eval-dir", dict(default=os.environ.get("LENSEEK_EVAL_DIR"),
                                           help="database encoding of the original model (default $LENSEEK_EVAL_DIR)")),
                  ("--strict-eval-dir", dict(default=None, help="database encoding of the model trained without "
                                                                "homologous entries (default "
                                                                "$LENSEEK_EVAL_ROOT/out_strict30_s0)")),
                  ("--dt-dir", dict(default=os.environ.get("LENSEEK_DT_DIR"),
                                    help="DeeplyTough database descriptors db_desc.npy (default $LENSEEK_DT_DIR)")),
                  ("--dt-queries", dict(default=None, help="DeeplyTough descriptors of the scored sites, "
                                                           "q_desc.npy and q_paths.txt (default OUT/deeplytough)")),
                  ("--bsdb-root", dict(default=os.environ.get("LENSEEK_BSDB_ROOT"),
                                       help="database structures, target paths of the pair lists "
                                            "(default $LENSEEK_BSDB_ROOT)")),
                  ("--device", dict(default="cpu", choices=["cpu", "cuda"])), ("--threads", dict(type=int, default=8))]}
    for name, opts in spec.items():
        p = sub.add_parser(name)
        p.add_argument("--out", default=OUT)
        for flag, kw in opts:
            p.add_argument(flag, **kw)
    a = ap.parse_args()
    if a.cmd in ("build", "write"):
        a.cifdir = a.cifdir or os.path.join(a.out, "raw", "cif")
    if a.cmd in ("fetch", "build"):
        a.ids = a.ids or os.path.join(a.out, "meta", "entries_to_build.txt")
    if a.cmd == "fetch":
        fetch_rcsb(a.ids, os.path.join(a.out, "raw", "cif"), "cif", 6)
    elif a.cmd == "select":
        select_sites(argparse.Namespace(build=os.path.join(a.out, "build"), sdir=os.path.join(a.out, "seqs"),
                                        mmdir=os.path.join(a.out, "mm"), out=os.path.join(a.out, "sel"),
                                        meta_path=a.meta or os.path.join(a.out, "meta", "meta_window.jsonl")))
    else:
        dict(entries=cmd_entries, build=cmd_build, identity=cmd_identity, write=cmd_write, embed=cmd_embed,
             features=cmd_features, clusters=cmd_clusters, strata=cmd_strata, counts=cmd_counts,
             validate=cmd_validate, score=cmd_score)[a.cmd](a)


if __name__ == "__main__":
    main()
