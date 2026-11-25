#!/usr/bin/env python
"""Extract the binding site around a ligand and search the database with it, in one command.

    python scripts/search_ligand.py complex.cif --ligand ATP:A:501 --top-k 100 --out hits.tsv
    python scripts/search_ligand.py protein.pdb --ligand ligand.pdb --device cuda

The binding site is every ATOM-record residue with any atom closer than r_vdW(ligand atom) + 4.0 A to any ligand atom
(hydrogens included; r_vdW from the RDKit periodic table), the rule the database sites were built with
(lenseek.pocket). Alternate locations: blank plus 'A', or the lowest label when there is no 'A'.

STRUCTURE is a PDB or mmCIF file of the protein (first model). --ligand is either a file holding the ligand's coordinates
(PDB or mmCIF; its HETATM records, or all atoms if it has none), or a HETATM residue of STRUCTURE given as CCD[:chain[:number]],
which must match exactly one residue.

The pocket and the protein are written to --workdir/<name>/ in standard PDB columns and passed to lenseek-search; every other
option is forwarded to it unchanged (--top-k, --out, --db, --device, --embeddings, ...). Unless --embeddings is given, --embed
is added so that a missing embedding is computed. Needs the `pocket` extra (gemmi, rdkit, scipy).
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lenseek.cli import search as lenseek_search  # noqa: E402
from lenseek.cli.common import die  # noqa: E402
from lenseek.pocket import POCKET_ADD, WATER, read_atoms, rvdw, select_altloc, write_entry_strict  # noqa: E402


def resolve_ligand(spec, atoms):
    """Ligand atoms: from a coordinate file, or the single HETATM residue of the structure matching CCD[:chain[:number]]."""
    if os.path.isfile(spec):
        raw, _ = read_atoms(spec)
        lig = select_altloc(raw)
        het = [a for a in lig if a["rec"] == "HETATM" and a["resn"] not in WATER]
        return het or [a for a in lig if a["resn"] not in WATER]
    parts = spec.split(":")
    if not parts[0] or len(parts) > 3:
        die(f"--ligand {spec!r}: expected a file or CCD[:chain[:number]]")
    ccd, chain, number = parts[0].upper(), parts[1] if len(parts) > 1 else None, int(parts[2]) if len(parts) > 2 else None
    groups = {}
    for a in atoms:
        if a["rec"] == "HETATM" and a["resn"] == ccd and (chain is None or a["chain"] == chain) and \
                (number is None or a["resi"] == number):
            groups.setdefault((a["chain"], a["resi"], a["icode"]), []).append(a)
    if not groups:
        die(f"--ligand {spec!r}: no matching HETATM residue in the structure")
    if len(groups) > 1:
        die(f"--ligand {spec!r} matches {len(groups)} residues ({', '.join(f'{c}:{n}{i}' for c, n, i in groups)}); "
            f"give CCD:chain:number")
    return next(iter(groups.values()))


def pocket_of(lig, atoms):
    """Pocket residue keys and the chains near the ligand (the file-name suffix), as the database builder computes them."""
    prot = [a for a in atoms if a["rec"] == "ATOM"]
    if not prot:
        die("the structure has no ATOM records")
    xyz = lambda rows: np.array([[a["x"], a["y"], a["z"]] for a in rows])
    protein_tree, all_tree = cKDTree(xyz(prot)), cKDTree(xyz(atoms))
    lig_ids = {id(a) for a in lig}
    residues, chains = set(), {a["chain"] for a in lig if a["chain"].strip()}
    for a, p in zip(lig, xyz(lig)):
        cut = rvdw(a["elem"]) + POCKET_ADD - 1e-9
        for q in protein_tree.query_ball_point(p, cut):
            residues.add((prot[q]["chain"], prot[q]["resi"], prot[q]["icode"]))
        for q in all_tree.query_ball_point(p, cut):
            if id(atoms[q]) not in lig_ids:
                chains.add(atoms[q]["chain"])
    return residues, "".join(sorted(chains))


def extract(structure, ligand, name, workdir):
    """Write <workdir>/<name>/{protein,ligand_1_X,ligand_pocket_1_X}.pdb and return the directory and pocket size."""
    raw, _ = read_atoms(structure)
    atoms = select_altloc(raw)
    lig = resolve_ligand(ligand, atoms)
    if not lig:
        die(f"--ligand {ligand!r}: no ligand atoms found")
    residues, suffix = pocket_of(lig, atoms)
    if not residues:
        die("no protein residue is within the cutoff of the ligand")
    first = lig[0]
    site = dict(idx=1, suffix=suffix, key=(first["chain"], first["resi"], first["icode"], first["resn"]), atoms=lig,
                pocket=residues)
    write_entry_strict(workdir, name, [site], atoms)
    return os.path.join(workdir, name), len(residues)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="PDB or mmCIF file of the protein")
    ap.add_argument("--ligand", required=True, help="ligand coordinate file, or CCD[:chain[:number]] in the structure")
    ap.add_argument("--name", default=None, help="entry name of the query (default: the structure's file stem or directory)")
    ap.add_argument("--workdir", default="lenseek_pockets", help="where the extracted pocket is written (default ./lenseek_pockets)")
    args, forward = ap.parse_known_args(argv)
    if not os.path.isfile(args.structure):
        die(f"structure file not found: {args.structure}")
    if not os.path.isfile(args.ligand) and (os.sep in args.ligand or os.path.splitext(args.ligand)[1].lower() in
                                            (".pdb", ".ent", ".cif", ".mmcif", ".gz")):
        die(f"--ligand {args.ligand}: file not found (give a ligand file or CCD[:chain[:number]])")
    name = args.name or lenseek_entry_name(args.structure)
    site_dir, n_res = extract(args.structure, args.ligand, name, args.workdir)
    print(f"[search_ligand] {n_res} pocket residues around the ligand -> {site_dir}", file=sys.stderr)
    if not any(a == "--embeddings" or a.startswith("--embeddings=") or a == "--embed" for a in forward):
        forward = ["--embed"] + forward
    lenseek_search.main([site_dir] + forward)


def lenseek_entry_name(path):
    from lenseek.featurize import default_entry_name
    return default_entry_name(path)


if __name__ == "__main__":
    main()
