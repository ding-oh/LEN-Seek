"""Helpers shared by the console commands."""
from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional, Tuple

from ..assets import AssetError, fetch
from ..featurize import find_sites
from ..resources import model_registry


def die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def set_threads(n: Optional[int]) -> None:
    if n:
        import torch
        torch.set_num_threads(int(n))


def check_device(device: Optional[str]) -> None:
    """Stop with a message when a CUDA device is asked for but torch cannot use one."""
    if device and str(device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            die(f"--device {device}: CUDA is not available in this torch build or on this machine; use --device cpu "
                f"or install a CUDA build of torch")


def resolve_checkpoint(checkpoint: Optional[str], model: Optional[str]) -> str:
    """--checkpoint path, or the asset of a released model name (default: the registry's default model)."""
    if checkpoint:
        return checkpoint
    reg = model_registry()
    name = model or next((k for k, v in reg.items() if isinstance(v, dict) and v.get("default")), "main_s0")
    if name not in reg:
        die(f"unknown model {name!r}; released models: {', '.join(reg)}")
    try:
        return str(fetch(reg[name]["asset"]))
    except AssetError as e:
        die(str(e))


def expand_sites(inputs: List[str], protein: Optional[str] = None,
                 pattern: str = "*pocket*.pdb") -> List[Tuple[str, str]]:
    """(pocket, protein) pairs from pocket files, entry directories or a BsitePDB-layout root."""
    pairs: List[Tuple[str, str]] = []
    for item in inputs:
        if os.path.isdir(item):
            local = sorted(p for p in glob.glob(os.path.join(item, pattern)) if not p.endswith("protein.pdb"))
            if local:
                for p in local:
                    pairs.append((p, protein or _protein_next_to(p)))
            else:
                pairs.extend((pk, protein or pr) for pk, pr in find_sites(item, pattern))
        else:
            for p in sorted(glob.glob(item)) or [item]:
                pairs.append((p, protein or _protein_next_to(p)))
    for pk, pr in pairs:
        if not os.path.exists(pk):
            die(f"pocket file not found: {pk}")
        if not os.path.exists(pr):
            die(f"protein file not found for {pk}: {pr} (pass --protein)")
    return pairs


def _protein_next_to(pocket: str) -> str:
    d = os.path.dirname(os.path.abspath(pocket))
    entry = os.path.basename(d)
    cand = os.path.join(d, f"{entry}_protein.pdb")
    return cand if os.path.exists(cand) else os.path.join(d, "protein.pdb")
