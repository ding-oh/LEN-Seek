"""Embedding files are reused only for the chain sequence they were computed from: two different proteins with the
same entry name and the same chain length must not share an embedding silently. A stand-in replaces Ankh-base."""
import shutil

import numpy as np
import pytest

from conftest import EXAMPLES
from lenseek import ankh
from lenseek.featurize import FeaturizeError, embedding_file, featurize_site
from lenseek.io.pdb import THREE_TO_ONE


def fake_embed(sequence, model=None, tokenizer=None, device="cpu"):
    """Deterministic stand-in for Ankh-base: row i depends on residue i's letter and on i."""
    codes = np.frombuffer(sequence.encode(), dtype=np.uint8).astype(np.float32)
    return codes[:, None] + np.arange(len(sequence), dtype=np.float32)[:, None] * 1e-3 + np.zeros(768, np.float32)


@pytest.fixture
def fake_ankh(monkeypatch):
    monkeypatch.setattr(ankh, "load_ankh", lambda *a, **k: (None, None))
    monkeypatch.setattr(ankh, "embed_sequence", fake_embed)


def two_proteins_same_name(tmp_path):
    """projA/site and projB/site: the 7uad example, and a copy with one residue outside the pocket renamed."""
    src = EXAMPLES / "queries" / "7uad"
    a, b = tmp_path / "projA" / "site", tmp_path / "projB" / "site"
    for d in (a, b):
        d.mkdir(parents=True)
        shutil.copy(src / "ligand_pocket_1_A.pdb", d)
    shutil.copy(src / "protein.pdb", a)
    pocket = {l[21:27] for l in open(src / "ligand_pocket_1_A.pdb") if l.startswith("ATOM")}
    lines = open(src / "protein.pdb").read().splitlines(keepends=True)
    target, name = next((l[21:27], l[17:20]) for l in lines
                        if l.startswith("ATOM") and l[17:20] in THREE_TO_ONE and l[21:27] not in pocket)
    new = "ALA" if name != "ALA" else "GLY"
    (b / "protein.pdb").write_text("".join(l[:17] + new + l[20:] if l.startswith("ATOM") and l[21:27] == target
                                           else l for l in lines))
    return a, b


def test_same_name_same_length_other_sequence(tmp_path, fake_ankh):
    a, b = two_proteins_same_name(tmp_path)
    seq_a = ankh.chain_sequences_of(str(a / "protein.pdb"))["A"]
    seq_b = ankh.chain_sequences_of(str(b / "protein.pdb"))["A"]
    assert len(seq_a) == len(seq_b) and seq_a != seq_b
    emb = str(tmp_path / "embeddings")
    path = embedding_file(emb, "site", "A")

    ankh.embed_protein(str(a / "protein.pdb"), emb, "site")
    ankh.embed_protein(str(b / "protein.pdb"), emb, "site")      # same file name and length, other sequence
    assert np.array_equal(np.load(path), fake_embed(seq_b))      # recomputed for B, not A's file reused

    s = featurize_site(str(b / "ligand_pocket_1_A.pdb"), str(b / "protein.pdb"), emb)
    assert len(s) > 0
    for mode in ("strict", "legacy"):                             # A's search must refuse B's embedding
        with pytest.raises(FeaturizeError, match="different sequence"):
            featurize_site(str(a / "ligand_pocket_1_A.pdb"), str(a / "protein.pdb"), emb, mode=mode)


def test_file_without_sidecar_is_recomputed(tmp_path, fake_ankh):
    a, b = two_proteins_same_name(tmp_path)
    emb = tmp_path / "embeddings"
    emb.mkdir()
    seq_b = ankh.chain_sequences_of(str(b / "protein.pdb"))["A"]
    np.save(embedding_file(str(emb), "site", "A"), np.zeros((len(seq_b), 768), np.float32))   # unknown origin
    ankh.embed_protein(str(b / "protein.pdb"), str(emb), "site")
    assert np.array_equal(np.load(embedding_file(str(emb), "site", "A")), fake_embed(seq_b))


def test_example_embeddings_match_their_structures():
    from lenseek.featurize import embedding_matches

    files = sorted((EXAMPLES / "embeddings").glob("*_chain_*.npy"))
    assert len(files) == 5
    for f in files:
        entry, chain = f.stem.rsplit("_chain_", 1)
        protein = str(EXAMPLES / "queries" / entry / "protein.pdb")
        for mode in ("strict", "legacy"):
            assert embedding_matches(str(f), ankh.chain_sequences_of(protein, mode=mode)[chain]) is True
