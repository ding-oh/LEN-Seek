"""Provenance stored with every checkpoint: switches, full run config, code hash, library versions."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess


def code_tree_hash(code_dir: str) -> str:
    """sha256 over the sorted relative paths and contents of every .py file below code_dir (no __pycache__)."""
    files = []
    for root, dirs, names in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    h = hashlib.sha256()
    for rel in sorted(os.path.relpath(p, code_dir).replace(os.sep, "/") for p in files):
        with open(os.path.join(code_dir, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def versions() -> dict:
    """Library versions without creating a CUDA context (GPU names from nvidia-smi when available)."""
    out = {"python": platform.python_version(), "host": socket.gethostname()}
    try:
        import torch
        out.update(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    except ImportError:
        pass
    for mod in ("accelerate", "numpy", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        out["gpus"] = [l.strip() for l in smi.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        out["gpus"] = []
    return out


def collect(run, **extra) -> dict:
    """JSON-safe provenance of a training run (stored as checkpoint['run_info'])."""
    from . import __version__
    from .config import run_config_to_dict, switches_of
    from .geometry import KNN_RULE

    pkg = os.path.dirname(os.path.abspath(__file__))
    info = {"switches": {**switches_of(run), "DIRECTION_FRAME": run.model.direction_frame},
            "config": run_config_to_dict(run), "lenseek_version": __version__, "code_hash": code_tree_hash(pkg),
            "knn_rule": KNN_RULE, "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("LENSEEK_")},
            "versions": versions(), **extra}
    return json.loads(json.dumps(info, default=str))
