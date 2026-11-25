"""Binding site around a ligand, with the rules the database sites and the binding sites from later PDB entries were
built with.

Atoms are read from the first model of a PDB or mmCIF file (gemmi); one alternate location is kept per residue
position (blank plus 'A', else the lowest label). The binding site is every ATOM-record residue with any atom
(hydrogens included) closer than r_vdW(ligand atom) + 4.0 A to any ligand atom, r_vdW from the RDKit periodic table.
Sites are written in the BsitePDB file names (protein.pdb, ligand_<n>_<chains>.pdb, ligand_pocket_<n>_<chains>.pdb) with
standard PDB columns and insertion codes. Used by scripts/search_ligand.py and scripts/holdout.py; needs the `pocket`
extra (gemmi, rdkit).
"""
import os

import gemmi
from rdkit import Chem

PT = Chem.GetPeriodicTable()
WATER = {"HOH", "DOD", "WAT"}
POCKET_ADD = 4.0     # A, added to the ligand atom's vdW radius
_RV = {}


def rvdw(elem):
    if elem not in _RV:
        try:
            _RV[elem] = PT.GetRvdw(PT.GetAtomicNumber(elem.capitalize()))
        except Exception:
            _RV[elem] = 2.0
    return _RV[elem]


def read_atoms(path):
    """First model of an mmCIF/PDB file as a list of atom dicts (auth chain / seq id / comp id, ATOM vs HETATM)."""
    st = gemmi.read_structure(path)
    model = st[0]
    atoms = []
    for ch in model:
        for res in ch:
            rec = "HETATM" if res.het_flag == "H" else "ATOM"
            for at in res:
                p = at.pos
                atoms.append(dict(rec=rec, name=at.name, alt=at.altloc if at.altloc != "\0" else " ",
                                  resn=res.name, chain=ch.name, resi=res.seqid.num,
                                  icode=res.seqid.icode if res.seqid.icode != " " else "",
                                  x=p.x, y=p.y, z=p.z, occ=at.occ, b=at.b_iso, elem=at.element.name.upper()))
    return atoms, st


def select_altloc(atoms):
    """Keep blank-altloc atoms plus one label per residue position: 'A' if present, else the lowest label."""
    labels = {}
    for a in atoms:
        if a["alt"] != " ":
            labels.setdefault((a["chain"], a["resi"], a["icode"]), set()).add(a["alt"])
    keep = {k: ("A" if "A" in v else min(v)) for k, v in labels.items()}
    out = []
    for a in atoms:
        if a["alt"] == " " or keep.get((a["chain"], a["resi"], a["icode"])) == a["alt"]:
            out.append(a)
    return out


def pdb_line_strict(rec, serial, a, chain=None):
    name = a["name"]
    name4 = name.ljust(4) if (len(name) == 4 or len(a["elem"]) == 2) else (" " + name).ljust(4)
    resn = a["resn"] if len(a["resn"]) <= 3 else "UNL"
    ch = a["chain"] if chain is None else chain
    ic = (a["icode"] or " ")[:1]
    return (f"{rec:<6}{serial % 100000:>5} {name4}{a['alt']}{resn:>3} {ch[:1]}{a['resi']:>4}{ic}   "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['occ']:6.2f}{a['b']:6.2f}          {a['elem']:>2}  ")


def ter_line_strict(serial, a, chain=None):
    resn = a["resn"] if len(a["resn"]) <= 3 else "UNL"
    ch = a["chain"] if chain is None else chain
    return f"TER   {serial % 100000:>5}      {resn:>3} {ch[:1]}{a['resi']:>4}{(a['icode'] or ' ')[:1]}".ljust(80)


def write_entry_strict(outdir, pdbid, sites, atoms, only_idx=None):
    """Write protein.pdb and the chosen sites' ligand / pocket files (BsitePDB file names, strict PDB columns)."""
    d = os.path.join(outdir, pdbid)
    os.makedirs(d, exist_ok=True)
    prot = [a for a in atoms if a["rec"] == "ATOM"]

    def block(rec, rows, chain=None):
        out, ser, prev = [], 0, None
        for a in rows:
            if prev is not None and (chain is None and a["chain"] != prev["chain"]):
                ser += 1
                out.append(ter_line_strict(ser, prev, chain))
            ser += 1
            out.append(pdb_line_strict(rec, ser, a, chain))
            prev = a
        if prev is not None:
            ser += 1
            out.append(ter_line_strict(ser, prev, chain))
        return out

    with open(os.path.join(d, "protein.pdb"), "w") as f:
        f.write("\n".join(block("ATOM", prot) + ["END".ljust(80)]) + "\n")
    written = []
    for s in sites:
        if only_idx is not None and s["idx"] not in only_idx:
            continue
        tag = f"{s['idx']}_{s['suffix']}"
        ll = block("HETATM", s["atoms"], chain=" ")
        if len(s["key"][3]) > 3:
            ll.insert(0, f"REMARK 999 LIGAND_CCD {s['key'][3]}".ljust(80))
        with open(os.path.join(d, f"ligand_{tag}.pdb"), "w") as f:
            f.write("\n".join(ll + ["END".ljust(80)]) + "\n")
        pk = [a for a in prot if (a["chain"], a["resi"], a["icode"]) in s["pocket"]]
        with open(os.path.join(d, f"ligand_pocket_{tag}.pdb"), "w") as f:
            f.write("\n".join(block("ATOM", pk) + ["END".ljust(80)]) + "\n")
        written.append(tag)
    return written
