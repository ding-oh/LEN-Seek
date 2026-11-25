"""Per-chain Ankh-base embeddings (node features of every released LEN-Seek model).

The embedding of a chain is the last hidden state of the Ankh-base T5 encoder over the chain's one-letter sequence
(characters as words, special tokens added), without the trailing special token: shape (L, 768), float32.
The node features of the training sites and of the search database were computed this way.

The weights (``ElnaggarLab/ankh-base`` on Hugging Face, about 2.9 GB) are downloaded on first use. They are
licensed CC BY-NC-SA 4.0 by their authors (model card), independently of the license of this repository.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .featurize import embedding_file, embedding_matches, write_embedding_sidecar
from .io import pdb as pdbio

ANKH_REPO = "ElnaggarLab/ankh-base"
ANKH_REVISION = "d99cb6b966530dfc2ae96bc69d9255c2a07308b0"   # Hugging Face commit pinned for this release
_cache: Dict[Tuple[str, str], tuple] = {}


def load_ankh(device: str = "cpu", model_dir: Optional[str] = None, revision: str = ANKH_REVISION):
    """(model, tokenizer) on ``device``. ``model_dir``: local copy of the Hugging Face repository (optional;
    ``$LENSEEK_ANKH_DIR`` is used when set)."""
    from transformers import AutoTokenizer, T5EncoderModel

    model_dir = model_dir or os.environ.get("LENSEEK_ANKH_DIR")
    key = (str(device), model_dir or f"{ANKH_REPO}@{revision}")
    if key in _cache:
        return _cache[key]
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=ANKH_REPO, revision=revision, allow_patterns=[
            "config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5EncoderModel.from_pretrained(model_dir, use_safetensors=False)
    model.to(device).eval()
    _cache[key] = (model, tokenizer)
    return model, tokenizer


@torch.no_grad()
def embed_sequence(sequence: str, model, tokenizer, device: str = "cpu") -> np.ndarray:
    """(len(sequence), 768) float32 embedding of one chain."""
    ids = tokenizer.batch_encode_plus([list(sequence)], add_special_tokens=True, padding=True,
                                      is_split_into_words=True, return_tensors="pt")
    out = model(input_ids=ids["input_ids"].to(device), attention_mask=ids["attention_mask"].to(device))
    return out.last_hidden_state[0, :len(sequence)].float().cpu().numpy()


def chain_sequences_of(protein_path: str, mode: str = "strict") -> Dict[str, str]:
    """One-letter sequence of every chain, in the residue order the featurizer indexes embeddings by."""
    if mode == "legacy":
        return {ch: "".join(pdbio.THREE_TO_ONE[name] for _, name in res)
                for ch, res in pdbio.legacy_chain_sequences(protein_path).items()}
    st = pdbio.read_structure(protein_path, mode="strict")
    return {ch: pdbio.sequence_string(st, ch) for ch in pdbio.chain_sequences(st)}


def embed_protein(protein_path: str, out_dir: str, entry: str, *, mode: str = "strict", device: str = "cpu",
                  overwrite: bool = False, model_dir: Optional[str] = None, chains=None) -> Dict[str, str]:
    """Write ``<out_dir>/<entry>_chain_<C>.npy`` for every chain of a protein file; returns {chain: path}.

    Each file gets a sidecar ``<file>.json`` with the SHA-256 of the chain sequence and the model. An existing file
    is kept only when its sidecar matches the chain's sequence (unless ``overwrite``); otherwise it is recomputed."""
    os.makedirs(out_dir, exist_ok=True)
    seqs = chain_sequences_of(protein_path, mode=mode)
    local = model_dir or os.environ.get("LENSEEK_ANKH_DIR")
    model_name = f"{ANKH_REPO} (local copy {local})" if local else f"{ANKH_REPO}@{ANKH_REVISION}"
    written = {}
    model = tok = None
    for ch, seq in seqs.items():
        if chains is not None and ch not in chains:
            continue
        path = embedding_file(out_dir, entry, ch)
        if os.path.exists(path) and not overwrite:
            match = embedding_matches(path, seq)
            if match:
                try:
                    if np.load(path, mmap_mode="r").shape[0] == len(seq):
                        written[ch] = path
                        continue
                except Exception:
                    pass
            elif match is False:
                print(f"[lenseek-embed] recomputing {path}: it was computed from a different sequence (another "
                      f"protein with the entry name {entry!r}?)", file=sys.stderr)
        if model is None:
            model, tok = load_ankh(device, model_dir)
        np.save(path, embed_sequence(seq, model, tok, device))
        write_embedding_sidecar(path, seq, model_name)
        written[ch] = path
    return written
