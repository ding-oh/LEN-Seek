"""Binding site -> model inputs: per-residue node features h, C-alpha positions T and residue frames R.

A site is given by a pocket file (its ATOM records name the pocket residues) and the protein file the residues are
read from. Node features are rows of a per-chain protein-language-model embedding (Ankh-base for every released
model), indexed by the residue's position in the chain sequence of :func:`lenseek.io.chain_sequences`.

``featurize_site`` is the path for new inputs (strict PDB reader). ``featurize_site_legacy`` is the featurization
that built the training and database inputs and regenerates them exactly; the two readers can read different
coordinates from files in the original BsitePDB column layout.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import MIN_RESIDUES
from .io import pdb as pdbio


class FeaturizeError(ValueError):
    """The site cannot be featurized (too few residues, missing backbone atom, embedding mismatch, ...)."""


@dataclass
class SiteFeatures:
    """Model inputs of one binding site. Rows follow ``keys`` (sorted residue keys)."""

    site_id: str
    keys: List[str]
    h: np.ndarray            # (n, node_dim) float32
    T: np.ndarray            # (n, 3) float32, C-alpha positions in the input frame
    R: np.ndarray            # (n, 3, 3) float32, residue frames, axes as columns
    meta: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)


def residue_frames(coords_n: np.ndarray, coords_ca: np.ndarray, coords_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translations (C-alpha) and frames: e1 = CA->C, e2 = CA->N orthogonalised against e1, e3 = e1 x e2.

    Computed in float32 with torch, as for the training and database inputs; R has the axes as columns.
    """
    coords_n = np.asarray(coords_n, dtype=np.float32)
    coords_ca = np.asarray(coords_ca, dtype=np.float32)
    coords_c = np.asarray(coords_c, dtype=np.float32)
    T = coords_ca
    with torch.no_grad():
        e1 = F.normalize(torch.from_numpy(coords_c - coords_ca).float(), dim=-1)
        v2 = torch.from_numpy(coords_n - coords_ca).float()
        u2 = v2 - torch.sum(v2 * e1, dim=-1, keepdim=True) * e1
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=2).cpu().numpy()
    return T, R


# --------------------------------------------------------------------------- embeddings

EmbeddingLookup = Union[str, Mapping[str, np.ndarray], Callable[[str, str], np.ndarray]]


def embedding_file(emb_dir: str, entry: str, chain: str) -> str:
    """Per-chain embedding file name used by every LEN-Seek tool: ``<entry>_chain_<chain>.npy``."""
    return os.path.join(emb_dir, f"{entry}_chain_{chain}.npy")


def embedding_sidecar(path: str) -> str:
    """``<file>.json`` next to an embedding file: SHA-256 of the chain sequence it was computed from, and the model."""
    return path + ".json"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def write_embedding_sidecar(path: str, sequence: str, model: str) -> None:
    with open(embedding_sidecar(path), "w") as f:
        json.dump({"sequence_sha256": sequence_sha256(sequence), "length": len(sequence), "model": model}, f)
        f.write("\n")


def embedding_matches(path: str, sequence: str) -> Optional[bool]:
    """Whether an embedding file was computed from ``sequence`` according to its sidecar; None without a sidecar
    (a file made outside LEN-Seek, checked by length only)."""
    side = embedding_sidecar(path)
    if not os.path.exists(side):
        return None
    try:
        with open(side) as f:
            return json.load(f).get("sequence_sha256") == sequence_sha256(sequence)
    except (OSError, ValueError, AttributeError):
        return False


def _check_embedding_sequence(path: str, sequence: str, entry: str, chain: str, protein_path: str) -> None:
    if embedding_matches(path, sequence) is False:
        raise FeaturizeError(
            f"{path} was computed from a different sequence than chain {chain} of {protein_path} (another protein "
            f"with the entry name {entry!r}?). Recompute it (lenseek-search --embed, or lenseek-embed --overwrite), "
            f"or give each protein a unique directory name or its own --embeddings directory")


def _chain_embedding(source: EmbeddingLookup, entry: str, chain: str) -> np.ndarray:
    if isinstance(source, str):
        path = embedding_file(source, entry, chain)
        if not os.path.exists(path):
            raise FeaturizeError(f"missing embedding file {path} (run lenseek-embed on the protein first)")
        return np.load(path)
    if callable(source):
        return np.asarray(source(entry, chain))
    if chain not in source:
        raise FeaturizeError(f"no embedding for chain {chain!r} of {entry}")
    return np.asarray(source[chain])


def default_entry_name(protein_path: str) -> str:
    """Entry name embedding files are looked up by: the protein file's directory name (BsitePDB layout
    ``<entry>/protein.pdb``), or the file stem when the file is not called protein.pdb / <dir>_protein.pdb."""
    base = os.path.basename(protein_path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(protein_path)))
    if base in ("protein.pdb", f"{parent}_protein.pdb"):
        return parent
    stem = os.path.splitext(base)[0]
    return stem[:-len("_protein")] if stem.endswith("_protein") else stem


# --------------------------------------------------------------------------- strict featurization

def featurize_site(pocket_path: str, protein_path: str, embeddings: EmbeddingLookup, *, mode: str = "strict",
                   entry: Optional[str] = None, site_id: Optional[str] = None,
                   structure: Optional[pdbio.Structure] = None, node_dim: Optional[int] = None) -> SiteFeatures:
    """Featurize one site with the strict (default) or legacy reader.

    embeddings: directory of ``<entry>_chain_<C>.npy`` files, a {chain: (L, D) array} mapping, or a callable
    (entry, chain) -> array. ``structure`` may be passed to reuse a parsed protein for several pockets.
    """
    if mode == "legacy":
        if not isinstance(embeddings, str):
            raise ValueError("legacy featurization reads embeddings from a directory")
        return featurize_site_legacy(pocket_path, protein_path, embeddings, site_id=site_id, entry=entry)
    entry = entry or default_entry_name(protein_path)
    site_id = site_id or default_site_id(pocket_path)
    residues = pdbio.pocket_residues(pocket_path, mode="strict")
    if len(residues) < MIN_RESIDUES:
        raise FeaturizeError(f"{pocket_path}: {len(residues)} standard residues (< {MIN_RESIDUES})")
    st = structure if structure is not None else pdbio.read_structure(protein_path, mode="strict")
    seqs = pdbio.chain_sequences(st)

    chains_needed = sorted({r.rid.chain for r in residues})
    emb: Dict[str, np.ndarray] = {}
    index: Dict[str, Dict[pdbio.ResidueId, int]] = {}
    for ch in chains_needed:
        if ch not in seqs:
            raise FeaturizeError(f"{pocket_path}: chain {ch!r} has no standard residues in {protein_path}")
        e = _chain_embedding(embeddings, entry, ch)
        path = embedding_file(embeddings, entry, ch) if isinstance(embeddings, str) else None
        if path and os.path.exists(embedding_sidecar(path)):
            seq = "".join(pdbio.THREE_TO_ONE[st.residues[r].name] for r in seqs[ch])
            _check_embedding_sequence(path, seq, entry, ch, protein_path)
        if e.ndim != 2 or e.shape[0] != len(seqs[ch]):
            raise FeaturizeError(f"{entry} chain {ch}: embedding has shape {e.shape} but the chain sequence has "
                                 f"{len(seqs[ch])} residues; recompute it from this protein file (lenseek-embed)")
        if node_dim is not None and e.shape[1] != node_dim:
            raise FeaturizeError(f"{entry} chain {ch}: embedding width {e.shape[1]} != model node_dim {node_dim}")
        emb[ch] = e
        index[ch] = {rid: i for i, rid in enumerate(seqs[ch])}

    n = len(residues)
    width = next(iter(emb.values())).shape[1]
    h = np.empty((n, width), dtype=np.float32)
    xyz = {a: np.empty((n, 3), dtype=np.float32) for a in ("N", "CA", "C")}
    for i, r in enumerate(residues):
        res = st.residues.get(r.rid)
        if res is None:
            raise FeaturizeError(f"{pocket_path}: residue {r.name} {r.rid} not found in {protein_path}")
        if res.name != r.name:
            raise FeaturizeError(f"{pocket_path}: residue {r.rid} is {r.name} in the pocket file but {res.name} "
                                 f"in {protein_path}")
        for a in ("N", "CA", "C"):
            if a not in res.atoms:
                raise FeaturizeError(f"{pocket_path}: residue {r.name} {r.rid} has no backbone atom {a}")
            xyz[a][i] = res.atoms[a]
        h[i] = emb[r.rid.chain][index[r.rid.chain][r.rid]]
    T, R = residue_frames(xyz["N"], xyz["CA"], xyz["C"])
    return SiteFeatures(site_id, [r.key for r in residues], h, T, R,
                        meta={"pocket": str(pocket_path), "protein": str(protein_path), "parser": "strict"})


def default_site_id(pocket_path: str) -> str:
    """'<entry dir>/<pocket file stem>', e.g. '7uad/ligand_pocket_1_A'."""
    return f"{os.path.basename(os.path.dirname(os.path.abspath(pocket_path)))}/" \
           f"{os.path.splitext(os.path.basename(pocket_path))[0]}"


# --------------------------------------------------------------------------- legacy featurization (database inputs)

def _legacy_seq_map(protein_path: str) -> Dict[str, Dict[str, int]]:
    """{chain: {resdef: index in the chain sequence}} over the legacy chain sequences."""
    mapping: Dict[str, Dict[str, int]] = {}
    for chain, residues in pdbio.legacy_chain_sequences(protein_path).items():
        mapping[chain] = {f"{name}{chain}{num:04d}": i for i, (num, name) in enumerate(residues)}
    return mapping


def legacy_resdefs(pocket_path: str) -> List[str]:
    """Legacy pocket residue keys: sorted {RESNAME + chain + %04d number} of standard ATOM records."""
    with open(pocket_path) as f:
        lines = f.readlines()
    return sorted({f"{l[17:20].strip()}{l[21].strip()}{int(l[22:26].strip()):04d}"
                   for l in lines if l.startswith("ATOM") and l[17:20].strip() in pdbio.VALID_RES})


def featurize_site_legacy(pocket_path: str, protein_path: str, ankh_dir: str, *, site_id: Optional[str] = None,
                          entry: Optional[str] = None, structure: Optional[dict] = None) -> SiteFeatures:
    """Featurization of the training and database inputs, with failures raised as FeaturizeError.

    Regenerates those inputs exactly with the fixed-column reader the released features and database were built with.
    """
    site_id = site_id or default_site_id(pocket_path)
    entry = entry or os.path.basename(os.path.dirname(protein_path))
    resdefs = legacy_resdefs(pocket_path)
    if len(resdefs) < MIN_RESIDUES:
        raise FeaturizeError(f"{pocket_path}: {len(resdefs)} standard residues (< {MIN_RESIDUES})")
    st = structure if structure is not None else pdbio.parse_pdb_legacy(protein_path)
    if st is None:
        raise FeaturizeError(f"cannot read {protein_path}")
    mapping = _legacy_seq_map(protein_path)
    chains = {r[3] for r in resdefs}
    emb: Dict[str, np.ndarray] = {}
    for ch in chains:
        path = embedding_file(ankh_dir, entry, ch)
        if not os.path.exists(path):
            raise FeaturizeError(f"missing embedding file {path}")
        emb[ch] = np.load(path)
        if ch not in mapping:
            raise FeaturizeError(f"chain {ch} found in the pocket but not in the protein sequence map")
        if os.path.exists(embedding_sidecar(path)):
            seq = "".join(pdbio.THREE_TO_ONE[name] for _, name in pdbio.legacy_chain_sequences(protein_path)[ch])
            _check_embedding_sequence(path, seq, entry, ch, protein_path)
        if len(mapping[ch]) != emb[ch].shape[0]:
            raise FeaturizeError(f"length mismatch for {entry} chain {ch}: sequence {len(mapping[ch])} vs "
                                 f"embedding {emb[ch].shape[0]}")
    n = len(resdefs)
    width = next(iter(emb.values())).shape[1]
    h = np.empty((n, width), dtype=np.float32)
    rows = []
    for rdef in resdefs:
        ch = rdef[3]
        if rdef not in mapping[ch]:
            raise FeaturizeError(f"residue {rdef} of the pocket is not in the protein sequence map")
        idx = mapping[ch][rdef]
        if idx >= emb[ch].shape[0]:
            raise FeaturizeError(f"index {idx} out of bounds for chain {ch}")
        rows.append(emb[ch][idx])
    h[:] = np.array(rows, dtype=np.float32)
    coords = {a: np.empty((n, 3), dtype=np.float32) for a in ("N", "CA", "C")}
    for i, rdef in enumerate(resdefs):
        try:
            res = st[rdef[3]][int(rdef[4:])]
        except KeyError:
            raise FeaturizeError(f"residue {rdef} not found in the protein structure") from None
        for a in ("N", "CA", "C"):
            if a not in res["atoms"]:
                raise FeaturizeError(f"missing backbone atom '{a}' in residue {rdef}")
            coords[a][i] = res["atoms"][a]
    T, R = residue_frames(coords["N"], coords["CA"], coords["C"])
    return SiteFeatures(site_id, list(resdefs), h, T, R,
                        meta={"pocket": str(pocket_path), "protein": str(protein_path), "parser": "legacy"})


# --------------------------------------------------------------------------- directory layouts

def find_sites(root: str, pattern: str = "*pocket*.pdb") -> List[Tuple[str, str]]:
    """(pocket, protein) pairs of a directory in the BsitePDB layout: ``<root>/<entry>/{protein.pdb,
    <entry>_protein.pdb}`` next to pocket files matching ``pattern``. Sorted by pocket path."""
    out = []
    for pocket in sorted(glob.glob(os.path.join(root, "*", pattern))):
        d = os.path.dirname(pocket)
        entry = os.path.basename(d)
        protein = os.path.join(d, f"{entry}_protein.pdb")
        if not os.path.exists(protein):
            protein = os.path.join(d, "protein.pdb")
        if os.path.basename(pocket) in ("protein.pdb", f"{entry}_protein.pdb"):
            continue
        out.append((pocket, protein))
    return out
