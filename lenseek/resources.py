"""Locate the YAML registries shipped with the package (lenseek/configs/models.yaml, lenseek/configs/assets.yaml)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


def config_dir() -> Path:
    """``$LENSEEK_CONFIG_DIR``, else the ``configs/`` directory inside the package."""
    env = os.environ.get("LENSEEK_CONFIG_DIR")
    if env:
        return Path(env)
    packaged = Path(__file__).resolve().parent / "configs"
    if (packaged / "models.yaml").exists():
        return packaged
    raise FileNotFoundError("lenseek/configs/ not found: set LENSEEK_CONFIG_DIR to a directory with models.yaml and "
                            "assets.yaml")


def repo_root() -> Path:
    """Root of the source checkout (parent of the package)."""
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def load_yaml(name: str) -> Dict[str, Any]:
    import yaml

    with open(config_dir() / name) as f:
        return yaml.safe_load(f) or {}


def model_registry() -> Dict[str, Any]:
    return load_yaml("models.yaml")


def asset_registry() -> Dict[str, Any]:
    return load_yaml("assets.yaml")
