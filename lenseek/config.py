"""Model and training configuration.

Every default below is the value used to train the released main model (``main_s0``).

The training runs of the paper were launched with ``LENSEEK_<NAME>=<value>`` switches. :func:`apply_switch` accepts
exactly those names, so a run can be re-created from its recorded arguments, e.g. ``LENSEEK_EDGE_DROP=dir`` or
``LENSEEK_W_FAPE=0``. ``data/runs.yaml`` holds every training run of the paper: the main-model recipe once, then each
run as its differences from it (:func:`load_run`); :func:`load_run_config` also reads a single-run YAML.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Fixed input conventions (not configurable: the released checkpoints depend on them).
NUM_PTS = 65          # residues per site after padding / truncation
MIN_RESIDUES = 10     # sites with fewer standard residues are rejected
EDGE_DIM = 39         # 32 RBF + 3 direction + 4 quaternion
RBF_COUNT = 32
RBF_MAX = 20.0        # Angstrom
EDGE_GROUPS = ("rbf", "dir", "quat")


@dataclass
class ModelConfig:
    """Architecture and graph construction. Must match the checkpoint that is loaded."""

    node_dim: int = 768            # Ankh-base embedding width (1280 / 1536 for the ESM node-feature ablations)
    edge_dim: int = EDGE_DIM
    hidden: int = 512              # GNN_H
    latent: int = 256              # GNN_Z
    layers: int = 4                # GNN_LAYERS (encoder and decoder each)
    heads: int = 8                 # GNN_HEADS
    dropout: float = 0.1           # GNN_DROP
    k_neighbors: int = 15          # K_NEIGHBORS, C-alpha kNN graph of the encoder
    logvar_min: float = -2.0
    logvar_max: float = 2.0
    # Ablation switches (defaults = main model).
    edge_drop: Tuple[str, ...] = ()   # edge feature groups set to zero, subset of EDGE_GROUPS; the kNN graph is kept
    fullgraph: bool = False           # encoder attends over all residue pairs instead of the kNN graph
    nonadapted: bool = False          # decoder reconstructs the raw node features (out_node: hidden -> node_dim)
    node_const: bool = False          # node features replaced by zeros (structure-only ablation)
    stopgrad: bool = False            # detach the adaptor output when it is the reconstruction target (training only)
    # 'local' = direction of j seen from residue i in i's own frame, R_i^T (x_j - x_i) / d_ij (paper eq. 10; every
    # model of this release). 'global' = the same vector in the laboratory frame, as used by checkpoint v1.0.
    direction_frame: str = "local"

    def __post_init__(self):
        self.edge_drop = normalize_edge_drop(self.edge_drop)
        if self.direction_frame not in ("local", "global"):
            raise ValueError(f"direction_frame must be 'local' or 'global', got {self.direction_frame!r}")


@dataclass
class TrainConfig:
    """Optimisation, loss weights and schedules."""

    epochs: int = 200
    batch_size: int = 16           # per process (GPU); main_s0: 16 per GPU x 2 GPUs x 16 accumulation = 512
    grad_accum: int = 16
    start_lr: float = 1e-3         # linear decay to end_lr over the epochs, set once per epoch
    end_lr: float = 1e-4
    max_grad: float = 1.0          # gradient-norm clipping
    beta_max: float = 0.5          # KL weight: 0 before kl_start, then linear ramp to beta_max over half a cycle
    kl_cycles: int = 1
    kl_start: int = 50
    free_bits: float = 12.0        # per-residue KL floor (nats, summed over latent dimensions)
    w_kl: float = 1.0
    w_mse: float = 1.0
    w_cos: float = 40.0
    w_norm: float = 4.0
    w_fape: float = 20.0
    w_rmsd: float = 5.0
    seed: int = 0
    save_every: int = 10           # write ep<N>.pth every N epochs (0 = never); final.pth is written after the last epoch
    subset: int = 0                # >0: train on the first N sites only (smoke tests)


@dataclass
class RunConfig:
    """A training run: model + training settings + data. ``name`` is used for the output directory."""

    name: str = "main_s0"
    data: str = ""                 # feature store (lenseek-featurize) or training pickle
    exclude_entries: str = ""      # optional file of PDB entry IDs whose sites are left out (strict30_s0)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""


def normalize_edge_drop(value: Any) -> Tuple[str, ...]:
    """Comma string or iterable -> edge groups in canonical order; unknown names are an error."""
    if value is None:
        return ()
    if isinstance(value, str):
        names = [g.strip() for g in value.split(",") if g.strip()]
    else:
        names = [str(g).strip() for g in value if str(g).strip()]
    unknown = sorted(set(names) - set(EDGE_GROUPS))
    if unknown:
        raise ValueError(f"unknown edge group(s) {unknown}; allowed: {', '.join(EDGE_GROUPS)}")
    return tuple(g for g in EDGE_GROUPS if g in names)


# --------------------------------------------------------------------------- LENSEEK_* switches of the paper runs

# switch name (without the LENSEEK_ prefix) -> (section, attribute)
SWITCHES: Dict[str, Tuple[str, str]] = {
    "EDGE_DROP": ("model", "edge_drop"),
    "FULLGRAPH": ("model", "fullgraph"),
    "STOPGRAD": ("model", "stopgrad"),
    "NONADAPTED": ("model", "nonadapted"),
    "NODE_CONST": ("model", "node_const"),
    "NODE_DIM": ("model", "node_dim"),
    "W_KL": ("train", "w_kl"),
    "W_MSE": ("train", "w_mse"),
    "W_COS": ("train", "w_cos"),
    "W_NORM": ("train", "w_norm"),
    "W_FAPE": ("train", "w_fape"),
    "W_RMSD": ("train", "w_rmsd"),
    "SAVE_EVERY": ("train", "save_every"),
    "SEED": ("train", "seed"),
    "EPOCHS": ("train", "epochs"),
    "BATCH": ("train", "batch_size"),
    "ACCUM": ("train", "grad_accum"),
    "SUBSET": ("train", "subset"),
}
_BOOL_SWITCHES = {"FULLGRAPH", "STOPGRAD", "NONADAPTED", "NODE_CONST"}


def _parse_bool(name: str, raw: str) -> bool:
    raw = str(raw).strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name}={raw!r}: expected 0 or 1")
    return raw == "1"


def apply_switch(run: RunConfig, assignment: str) -> None:
    """Apply one ``LENSEEK_<NAME>=<value>`` (or ``<NAME>=<value>``) switch to ``run`` in place."""
    name, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"switch {assignment!r} is not NAME=value")
    name = name.strip().upper()
    if name.startswith("LENSEEK_"):
        name = name[len("LENSEEK_"):]
    if name not in SWITCHES:
        raise ValueError(f"unknown switch {name!r}; known: {', '.join(SWITCHES)}")
    section, attr = SWITCHES[name]
    target = run.model if section == "model" else run.train
    if name == "EDGE_DROP":
        value: Any = normalize_edge_drop(raw)
    elif name in _BOOL_SWITCHES:
        value = _parse_bool(name, raw)
    else:
        current = getattr(target, attr)
        value = type(current)(float(raw)) if isinstance(current, int) and not isinstance(current, bool) else type(current)(raw)
    setattr(target, attr, value)


def switches_of(run: RunConfig) -> Dict[str, Any]:
    """Model and loss switches of a run under their LENSEEK_* names (the run_info that lenseek.train stores)."""
    out = {}
    for name, (section, attr) in SWITCHES.items():
        if name in ("SAVE_EVERY", "SEED", "EPOCHS", "BATCH", "ACCUM", "SUBSET"):
            continue
        value = getattr(run.model if section == "model" else run.train, attr)
        out[name] = list(value) if isinstance(value, tuple) else value
    return out


# --------------------------------------------------------------------------- YAML

def _update_dataclass(obj, values: Mapping[str, Any], where: str) -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise ValueError(f"unknown key {where}.{key}; known: {', '.join(sorted(names))}")
        if key == "edge_drop":
            value = normalize_edge_drop(value)
        setattr(obj, key, value)
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()


def model_config_from_dict(values: Optional[Mapping[str, Any]]) -> ModelConfig:
    cfg = ModelConfig()
    _update_dataclass(cfg, values or {}, "model")
    return cfg


def expand_path(value: str) -> str:
    """``~`` and environment variables of a config path; an unset ``$LENSEEK_HOME`` is ``~/.cache/lenseek``, the
    default of the rest of the package (lenseek.assets.home)."""
    from .assets import home

    if "LENSEEK_HOME" not in os.environ:
        value = re.sub(r"\$\{LENSEEK_HOME\}|\$LENSEEK_HOME\b", lambda _: str(home()), value)
    return os.path.expandvars(os.path.expanduser(value))


def default_runs_file() -> str:
    """``data/runs.yaml`` of the source checkout: every training run of the paper."""
    from .resources import repo_root

    return str(repo_root() / "data" / "runs.yaml")


def _read_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _table_run(raw: Mapping[str, Any], path: str, name: Optional[str]) -> Dict[str, Any]:
    """The ``recipe`` of a run table with the keys of run ``name`` merged in: ``model`` and ``train`` key by key,
    every other key replaced."""
    unknown = set(raw) - {"recipe", "runs"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}; a run table has recipe and runs")
    runs = raw.get("runs") or {}
    if name is None:
        raise ValueError(f"{path} is a run table; name a run: {', '.join(runs)}")
    if name not in runs:
        raise ValueError(f"{path}: no run {name!r}; runs: {', '.join(runs)}")
    merged = dict(raw.get("recipe") or {})
    for key, value in (runs[name] or {}).items():
        if key in ("model", "train"):
            merged[key] = {**(merged.get(key) or {}), **(value or {})}
        else:
            merged[key] = value
    merged["name"] = name
    return merged


def run_names(path: Optional[str] = None) -> List[str]:
    """Names of the runs of a run table (default data/runs.yaml), in file order."""
    return list(_read_yaml(path or default_runs_file()).get("runs") or {})


def load_run(name: str, path: Optional[str] = None, overrides: Iterable[str] = ()) -> RunConfig:
    """One run of a run table (default data/runs.yaml), e.g. ``load_run("strict30_s0")``."""
    return load_run_config(path or default_runs_file(), overrides, name=name)


def load_run_config(path: str, overrides: Iterable[str] = (), name: Optional[str] = None) -> RunConfig:
    """Read one training run: run ``name`` of a run table (``recipe`` + ``runs``, as data/runs.yaml), or a single-run
    YAML (keys name, data, exclude_entries, model, train, notes) when ``name`` is None. ``exclude_entries`` is relative
    to the YAML file. ``overrides`` are LENSEEK_* switches applied afterwards."""
    raw = _read_yaml(path)
    if "runs" in raw or "recipe" in raw:
        raw = _table_run(raw, path, name)
    elif name is not None:
        raise ValueError(f"{path} is a single-run config, not a run table; it has no run {name!r}")
    run = RunConfig()
    unknown = set(raw) - {"name", "data", "exclude_entries", "model", "train", "notes"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    run.name = str(raw.get("name", run.name))
    run.data = expand_path(str(raw.get("data", "") or ""))
    run.notes = str(raw.get("notes", "") or "")
    ex = str(raw.get("exclude_entries", "") or "")
    if ex and not os.path.isabs(ex):
        ex = os.path.join(os.path.dirname(os.path.abspath(path)), ex)
    run.exclude_entries = ex
    _update_dataclass(run.model, raw.get("model") or {}, "model")
    _update_dataclass(run.train, raw.get("train") or {}, "train")
    for sw in overrides:
        apply_switch(run, sw)
    run.model.__post_init__()
    return run


def run_config_to_dict(run: RunConfig) -> Dict[str, Any]:
    d = dataclasses.asdict(run)
    d["model"]["edge_drop"] = list(run.model.edge_drop)
    return d
