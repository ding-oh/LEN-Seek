"""Load LEN-Seek checkpoints.

A checkpoint is the dict written by train.py: ``{'epoch', 'model_state_dict', 'optimizer_state_dict'[, 'run_info']}``
(a bare state dict is accepted too). The state dict loads unchanged into :class:`lenseek.model.VAE`; a ``module.``
prefix from DistributedDataParallel is stripped.

Model settings are resolved in this order: an explicit ``ModelConfig``; the switches stored in the checkpoint's
``run_info``, when it has one (lenseek.train writes it); the released-model registry (lenseek/configs/models.yaml),
matched by SHA-256; otherwise the defaults, with a warning.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from .config import ModelConfig, model_config_from_dict, normalize_edge_drop
from .model import VAE
from .resources import model_registry


def sha256sum(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class CheckpointInfo:
    path: str
    sha256: str
    name: Optional[str]                 # registry name if the file is a released checkpoint
    epoch: Optional[int]
    config_source: str
    run_info: Optional[Dict[str, Any]] = None
    registry_entry: Dict[str, Any] = field(default_factory=dict)


def _switches_to_model_config(sw: Dict[str, Any]) -> ModelConfig:
    cfg = ModelConfig()
    cfg.edge_drop = normalize_edge_drop(sw.get("EDGE_DROP", []))
    cfg.fullgraph = bool(sw.get("FULLGRAPH", False))
    cfg.stopgrad = bool(sw.get("STOPGRAD", False))
    cfg.nonadapted = bool(sw.get("NONADAPTED", False))
    cfg.node_const = bool(sw.get("NODE_CONST", False))
    cfg.node_dim = int(sw.get("NODE_DIM", 768))
    if "DIRECTION_FRAME" in sw:
        cfg.direction_frame = sw["DIRECTION_FRAME"]
    return cfg


def find_registered(sha: str) -> Tuple[Optional[str], Dict[str, Any]]:
    for name, entry in model_registry().items():
        if isinstance(entry, dict) and entry.get("sha256") == sha:
            return name, entry
    return None, {}


def _state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _check_shapes(cfg: ModelConfig, state: Dict[str, torch.Tensor], path: str) -> None:
    node_in = state["feature_adaptor.0.weight"].shape[1]
    out_node = state["out_node.weight"].shape[0]
    if node_in != cfg.node_dim:
        raise ValueError(f"{path}: node feature width {node_in} but the model config says {cfg.node_dim}")
    if (out_node == cfg.node_dim and out_node != cfg.hidden) != cfg.nonadapted:
        raise ValueError(f"{path}: decoder node head of width {out_node} does not match nonadapted={cfg.nonadapted}")


def load_model(path: str, model_cfg: Optional[ModelConfig] = None, *, device: str = "cpu",
               verify_registry: bool = True) -> Tuple[VAE, ModelConfig, CheckpointInfo]:
    """Build the VAE for a checkpoint file and load its weights. Returns (model in eval mode, config, info)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path} (lenseek-fetch downloads the released ones)")
    sha = sha256sum(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    run_info = obj.get("run_info") if isinstance(obj, dict) else None
    name, entry = find_registered(sha)
    if model_cfg is not None:
        cfg, source = model_cfg, "explicit ModelConfig"
    elif run_info and "switches" in run_info:
        cfg, source = _switches_to_model_config(run_info["switches"]), "checkpoint run_info"
    elif name is not None:
        cfg = model_config_from_dict(entry.get("model"))
        source = f"registry entry '{name}' (lenseek/configs/models.yaml)"
    else:
        cfg, source = ModelConfig(), "defaults (unregistered checkpoint without run_info)"
        warnings.warn(f"{path}: not a released checkpoint and stores no switches; assuming the main-model "
                      f"settings. Pass a ModelConfig if it was trained with ablation switches.", stacklevel=2)
    if entry.get("superseded") and verify_registry:
        warnings.warn(f"{path} is the superseded LEN-Seek {name} checkpoint (not rotation invariant). Use it only "
                      f"to reproduce its own results.", stacklevel=2)
    state = _state_dict(obj)
    _check_shapes(cfg, state, path)
    model = VAE.from_config(cfg)
    model.load_state_dict(state)
    model.to(device).eval()
    info = CheckpointInfo(path=path, sha256=sha, name=name, epoch=obj.get("epoch") if isinstance(obj, dict) else None,
                          config_source=source, run_info=run_info, registry_entry=entry)
    return model, cfg, info
