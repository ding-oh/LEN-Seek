"""PDB readers for binding-site featurization.

Two modes:

``strict`` (default, the reader for new data)
    * coordinates are read from the PDB-standard columns 31-54 *or* from the one-column-left layout of the BsitePDB
      files used to build the LEN-Seek database (no insertion-code column). Values are written as ``%8.3f``, so the
      three numbers are found unambiguously in columns 27-54; anything else falls back to the standard columns;
    * residues are keyed by (chain, residue number, insertion code), so ``60A`` and ``60D`` stay different residues;
    * one alternate location per residue (blank plus 'A', or blank plus the lowest label if there is no 'A');
    * only the first MODEL is read; HETATM records are ignored (as in training);
    * any ATOM record that cannot be read raises :class:`PDBParseError` instead of being skipped, and so do two
      different residue names under one residue key (typically a file that lost its insertion codes).

``legacy`` (reproduction of the released inputs)
    The fixed-column reader the released training features and search database were built with: fixed standard
    columns, residues keyed by (chain, number) only, unreadable atoms skipped, last duplicate atom wins. The two
    readers can read different coordinates from files in the original BsitePDB column layout (README, Usage:
    Search). Use it to regenerate those inputs exactly as built.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

STANDARD_RESIDUES = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
VALID_RES = frozenset(STANDARD_RESIDUES)
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
MODES = ("strict", "legacy")

_XYZ = re.compile(r"-?\d+\.\d{3}")


class PDBParseError(ValueError):
    """An ATOM record or residue that the strict reader refuses to guess about."""


@dataclass(frozen=True, order=True)
class ResidueId:
    """Residue position in a structure. ``icode`` is '' when there is no insertion code."""

    chain: str
    number: int
    icode: str = ""

    def __str__(self) -> str:
        return f"{self.chain}:{self.number}{self.icode}"


@dataclass
class Residue:
    rid: ResidueId
    name: str
    atoms: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict)


@dataclass
class Structure:
    """Residues of the ATOM records of one model, in file order."""

    residues: "OrderedDict[ResidueId, Residue]" = field(default_factory=OrderedDict)
    mode: str = "strict"
    source: str = ""

    def chains(self) -> List[str]:
        out: List[str] = []
        for rid in self.residues:
            if rid.chain not in out:
                out.append(rid.chain)
        return out


def atom_xyz(line: str) -> Tuple[float, float, float]:
    """x, y, z of an ATOM/HETATM line in the standard layout or the one-column-left BsitePDB layout."""
    m = _XYZ.findall(line[26:54])
    if len(m) == 3:
        return float(m[0]), float(m[1]), float(m[2])
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"parser mode must be one of {MODES}, got {mode!r}")


# --------------------------------------------------------------------------- legacy reader

def parse_pdb_legacy(path: str, strict_coordinates: bool = False) -> Optional[Dict[str, Dict[int, dict]]]:
    """Legacy reader: {chain: {resnum: {'atoms': {name: xyz}}}} or None if unreadable.

    ``strict_coordinates=True`` keeps every other behaviour of this reader (residue keys without insertion codes, last
    duplicate atom wins) but reads the coordinates with :func:`atom_xyz`, as the strict reader does. This is the reader
    of the database's standard edition: the sites and residues of the paper edition, the coordinates of
    :func:`atom_xyz`."""
    struct: Dict[str, Dict[int, dict]] = {}
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception:
        return None
    for line in lines:
        if line.startswith("ATOM"):
            try:
                chain = line[21]
                res_seq = int(line[22:26])
                atom = line[12:16].strip()
                if chain not in struct:
                    struct[chain] = {}
                if res_seq not in struct[chain]:
                    struct[chain][res_seq] = {"atoms": {}}
                if strict_coordinates:
                    x, y, z = atom_xyz(line)
                else:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                struct[chain][res_seq]["atoms"][atom] = np.array([x, y, z])
            except (ValueError, IndexError):
                continue
    return struct


# --------------------------------------------------------------------------- strict reader

def _strict_records(path: str) -> List[Tuple[int, str]]:
    """(line number, line) of the ATOM records of the first model."""
    out = []
    seen_model = False
    with open(path, "r") as f:
        for no, line in enumerate(f, 1):
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM"):
                out.append((no, line.rstrip("\n")))
    return out


def _altloc_choice(labels: List[str]) -> Optional[str]:
    """Alternate location kept for a residue: 'A' if present, else the lowest label (blank atoms are always kept)."""
    named = sorted({a for a in labels if a.strip()})
    if not named:
        return None
    return "A" if "A" in named else named[0]


def read_structure(path: str, mode: str = "strict") -> Structure:
    """Read the ATOM records of a PDB file into a :class:`Structure`."""
    _check_mode(mode)
    st = Structure(mode=mode, source=str(path))
    if mode == "legacy":
        legacy = parse_pdb_legacy(path)
        if legacy is None:
            raise PDBParseError(f"{path}: cannot be read")
        names = _legacy_residue_names(path)
        for chain, residues in legacy.items():
            for num, rec in residues.items():
                rid = ResidueId(chain, num, "")
                res = Residue(rid, names.get((chain, num), "UNK"))
                for atom, xyz in rec["atoms"].items():
                    res.atoms[atom] = xyz
                st.residues[rid] = res
        return st

    records = _strict_records(path)
    parsed = []
    for no, line in records:
        if len(line) < 54:
            raise PDBParseError(f"{path}:{no}: ATOM record shorter than 54 columns: {line!r}")
        try:
            name = line[12:16].strip()
            alt = line[16]
            resname = line[17:20].strip()
            chain = line[21].strip()
            number = int(line[22:26])
            icode = line[26].strip()
            xyz = atom_xyz(line)
        except ValueError as exc:
            raise PDBParseError(f"{path}:{no}: unreadable ATOM record ({exc}): {line!r}") from None
        if not name:
            raise PDBParseError(f"{path}:{no}: ATOM record without an atom name: {line!r}")
        parsed.append((no, name, alt, resname, ResidueId(chain, number, icode), xyz))

    alts: Dict[ResidueId, List[str]] = {}
    for _, _, alt, _, rid, _ in parsed:
        alts.setdefault(rid, []).append(alt)
    keep_alt = {rid: _altloc_choice(labels) for rid, labels in alts.items()}

    for no, name, alt, resname, rid, xyz in parsed:
        if alt.strip() and alt != keep_alt[rid]:
            continue
        res = st.residues.get(rid)
        if res is None:
            res = st.residues[rid] = Residue(rid, resname)
        elif res.name != resname:
            raise PDBParseError(
                f"{path}:{no}: residue {rid} is both {res.name} and {resname}. Two residues share chain, number and "
                f"insertion code; the file has probably lost its insertion-code column. Renumber or restore it.")
        if name in res.atoms:
            raise PDBParseError(
                f"{path}:{no}: atom {name} of residue {rid} appears twice without an alternate-location label; the "
                f"file has probably lost its altloc column. Keep one conformation per atom.")
        res.atoms[name] = np.array(xyz)
    return st


def _legacy_residue_names(path: str) -> Dict[Tuple[str, int], str]:
    names: Dict[Tuple[str, int], str] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    names.setdefault((line[21], int(line[22:26])), line[17:20].strip())
                except ValueError:
                    continue
    return names


# --------------------------------------------------------------------------- pocket residues

@dataclass(frozen=True)
class PocketResidue:
    """A pocket residue. ``key`` is the sort key of the residue order used for every LEN-Seek input."""

    name: str
    rid: ResidueId

    @property
    def key(self) -> str:
        # NAME + chain + number as %04d (e.g. 'TYRH0060'), followed by the insertion code if there is one.
        return f"{self.name}{self.rid.chain}{self.rid.number:04d}{self.rid.icode}"


def pocket_residues(path: str, mode: str = "strict") -> List[PocketResidue]:
    """Standard residues listed in a pocket file, unique, sorted by :attr:`PocketResidue.key`.

    The order matters only for sites with more than 65 residues (the first 65 are used) and for floating-point
    summation order; it is the order of every training site and database entry.
    """
    _check_mode(mode)
    found: Dict[str, PocketResidue] = {}
    with open(path) as f:
        for no, line in enumerate(f, 1):
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname not in VALID_RES:
                continue
            if mode == "legacy":
                rid = ResidueId(line[21].strip(), int(line[22:26].strip()), "")
            else:
                try:
                    rid = ResidueId(line[21].strip(), int(line[22:26]), line[26].strip() if len(line) > 26 else "")
                except ValueError as exc:
                    raise PDBParseError(f"{path}:{no}: unreadable residue number ({exc}): {line!r}") from None
            pr = PocketResidue(resname, rid)
            found.setdefault(pr.key, pr)
    return [found[k] for k in sorted(found)]


# --------------------------------------------------------------------------- chain sequences (Ankh input)

def chain_sequences(structure: Structure) -> "OrderedDict[str, List[ResidueId]]":
    """Residues of each chain in sequence order: standard residues of the ATOM records, sorted by number then
    insertion code, chains sorted by name, blank chain IDs skipped. ``sequence_string`` gives the one-letter string.
    This is the sequence the per-chain Ankh embedding files are indexed by."""
    per_chain: Dict[str, List[Tuple[int, str, ResidueId]]] = {}
    for rid, res in structure.residues.items():
        if res.name not in THREE_TO_ONE or not rid.chain.strip():
            continue
        per_chain.setdefault(rid.chain, []).append((rid.number, rid.icode, rid))
    out: "OrderedDict[str, List[ResidueId]]" = OrderedDict()
    for chain in sorted(per_chain):
        out[chain] = [r for _, _, r in sorted(per_chain[chain], key=lambda x: (x[0], x[1]))]
    return out


def sequence_string(structure: Structure, chain: str) -> str:
    return "".join(THREE_TO_ONE[structure.residues[r].name] for r in chain_sequences(structure)[chain])


def legacy_chain_sequences(path: str) -> "OrderedDict[str, List[Tuple[int, str]]]":
    """Chain residue lists of the legacy reader (the database embeddings): per chain, unique residue numbers of
    standard ATOM records (first residue name seen), sorted by number."""
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    seen = set()
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            res_name = line[17:20].strip()
            if res_name not in THREE_TO_ONE:
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                res_num = int(line[22:26].strip())
            except ValueError:
                continue
            if (chain_id, res_num) not in seen:
                seen.add((chain_id, res_num))
                chain_residues.setdefault(chain_id, []).append((res_num, res_name))
    out: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()
    for chain_id in sorted(chain_residues):
        out[chain_id] = sorted(chain_residues[chain_id], key=lambda x: x[0])
    return out
