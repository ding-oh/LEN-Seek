"""Run configurations and checkpoints.

Run configs: every training run of data/runs.yaml resolves to its recorded settings, field by field; a single-run YAML
gives the same run; the LENSEEK_* switch names map onto the config.

The released checkpoints load unchanged: the model has their parameter names and shapes
(tests/data/state_dict_keys.tsv).
"""
import copy
import hashlib
import json
import warnings

import pytest
import torch

from conftest import ROOT
from lenseek.config import (apply_switch, load_run, load_run_config, ModelConfig, run_config_to_dict, run_names,
                            RunConfig, switches_of)
from lenseek.model import VAE
from lenseek.resources import model_registry
from lenseek.train import get_beta, get_lr


# ---------------------------------------------------------------- config
# Resolved settings of every training run of the paper, in the order of data/runs.yaml: the values that differ from
# RunConfig(), with data below $LENSEEK_HOME/assets and the exclusion lists by SHA-256. RESOLVED_SHA256 is the SHA-256
# of the canonical JSON (sorted keys, no spaces) of all resolved runs with LENSEEK_HOME=/LENSEEK_HOME.
S30 = "sha256:17b732562a4c2e4076d5daea0bec101891dc9d3421b61720868346827143c1fd"
S50 = "sha256:c154edc868f59114bf4e1cd30e7d87382d327f73601e50a4a2f9f986a0c01901"
G3, NO_GEOM, PLM = ["rbf", "dir", "quat"], {"w_fape": 0.0, "w_rmsd": 0.0}, {"batch_size": 32, "grad_accum": 8}
PAPER_RUNS = {
    "main_s0": {},
    "main_s1": {"train": {"seed": 1}},
    "main_s2": {"train": {"seed": 2}},
    "strict30_s0": {"exclude_entries": S30},
    "strict50_s0": {"exclude_entries": S50},
    "abl_nodir": {"model": {"edge_drop": ["dir"]}},
    "abl_norbf": {"model": {"edge_drop": ["rbf"]}},
    "abl_noquat": {"model": {"edge_drop": ["quat"]}},
    "abl_ankh_only": {"model": {"edge_drop": G3}, "train": NO_GEOM},
    "abl_ankh_fullgraph": {"model": {"edge_drop": G3, "fullgraph": True}, "train": NO_GEOM},
    "abl_ankh_fullgraph_na": {"model": {"edge_drop": G3, "fullgraph": True, "nonadapted": True}, "train": NO_GEOM},
    "abl_noFAPE": {"train": {"w_fape": 0.0}},
    "abl_noRMSD": {"train": {"w_rmsd": 0.0}},
    "abl_noCosNorm": {"train": {"w_cos": 0.0, "w_norm": 0.0}},
    "abl_beta0": {"train": {"batch_size": 32, "w_kl": 0.0}},
    "abl_nonadapted": {"model": {"nonadapted": True}, "train": {"batch_size": 32}},
    "abl_stopgrad": {"model": {"stopgrad": True}},
    "plm_esm2": {"data": "train_features_esm2", "model": {"node_dim": 1280}, "train": PLM},
    "plm_esm3": {"data": "train_features_esm3", "model": {"node_dim": 1536}, "train": PLM},
    "plm_esmc": {"data": "train_features_esmc", "model": {"node_dim": 960}, "train": PLM},
    "plm_none": {"model": {"node_const": True}, "train": {**PLM, "w_mse": 0.0, "w_cos": 0.0, "w_norm": 0.0}},
    "abl_onehot": {"data": "train_features_onehot", "train": {"batch_size": 32}},
    "abl_onehot_na": {"data": "train_features_onehot", "model": {"nonadapted": True}},
    "abl_onehot_fullgraph_na": {"data": "train_features_onehot",
                                "model": {"edge_drop": G3, "fullgraph": True, "nonadapted": True}, "train": NO_GEOM},
    "filter_free_s0": {"data": "train_features_filter_free"},
}
RESOLVED_SHA256 = "9b3d37e720f0d72f6ac34ac47331066365d3555424c2f0f6f0fc0b3e5c9d10b1"


def resolved(name):
    d = run_config_to_dict(load_run(name))
    if d["exclude_entries"]:
        d["exclude_entries"] = "sha256:" + hashlib.sha256(open(d["exclude_entries"], "rb").read()).hexdigest()
    return d


def test_run_table_gives_the_recorded_settings(monkeypatch):
    monkeypatch.setenv("LENSEEK_HOME", "/LENSEEK_HOME")
    got = {name: resolved(name) for name in run_names()}
    assert list(got) == list(PAPER_RUNS)
    default = run_config_to_dict(RunConfig())
    for name, diff in PAPER_RUNS.items():
        exp = copy.deepcopy(default)
        exp.update(name=name, data="/LENSEEK_HOME/assets/" + diff.get("data", "train_features"),
                   exclude_entries=diff.get("exclude_entries", ""))
        for section in ("model", "train"):
            exp[section].update(diff.get(section, {}))
        assert set(got[name]) == set(exp)
        for key, value in exp.items():
            if isinstance(value, dict):
                assert set(got[name][key]) == set(value), (name, key)
                for k, v in value.items():
                    g = got[name][key][k]
                    assert g == v and type(g) is type(v), (name, key, k, g, v)
            else:
                assert got[name][key] == value, (name, key)
    canon = json.dumps(got, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == RESOLVED_SHA256


def test_all_run_configs_load():
    for name in run_names():
        t = load_run(name).train
        assert t.batch_size * t.grad_accum in (256, 512)      # 2 GPUs: 16 x 16 or 32 x 8; 1 GPU: 32 x 16


def test_single_run_yaml(tmp_path):
    """A single-run YAML (lenseek-train --config) gives the same run as the run table."""
    p = tmp_path / "abl.yaml"
    p.write_text("name: abl_nodir\ndata: ${LENSEEK_HOME}/assets/train_features\nmodel: {edge_drop: [dir]}\n")
    assert load_run_config(str(p)) == load_run("abl_nodir")
    with pytest.raises(ValueError, match="name a run"):
        load_run_config(str(ROOT / "data" / "runs.yaml"))
    with pytest.raises(ValueError, match="no run 'main_s9'"):
        load_run("main_s9")
    with pytest.raises(ValueError, match="single-run"):
        load_run_config(str(p), name="abl_nodir")


@pytest.mark.parametrize("name, expect", [
    ("main_s0", {"EDGE_DROP": [], "FULLGRAPH": False, "W_FAPE": 20.0, "W_KL": 1.0}),
    ("abl_nodir", {"EDGE_DROP": ["dir"]}),
    ("abl_ankh_only", {"EDGE_DROP": ["rbf", "dir", "quat"], "FULLGRAPH": False, "W_FAPE": 0.0, "W_RMSD": 0.0}),
    ("abl_ankh_fullgraph", {"EDGE_DROP": ["rbf", "dir", "quat"], "FULLGRAPH": True, "W_FAPE": 0.0, "W_RMSD": 0.0}),
    ("abl_stopgrad", {"STOPGRAD": True}),
    ("abl_nonadapted", {"NONADAPTED": True}),
    ("abl_noCosNorm", {"W_COS": 0.0, "W_NORM": 0.0, "W_MSE": 1.0}),
    ("abl_beta0", {"W_KL": 0.0}),
    ("plm_none", {"NODE_CONST": True, "W_MSE": 0.0, "W_COS": 0.0, "W_NORM": 0.0}),
    ("plm_esm2", {"NODE_DIM": 1280}),
    ("plm_esmc", {"NODE_DIM": 960}),
])
def test_switches_of_runs(name, expect):
    sw = switches_of(load_run(name))
    for k, v in expect.items():
        assert sw[k] == v, (name, k)


def test_switch_overrides_like_run_scripts():
    run = RunConfig()
    for s in ("LENSEEK_EDGE_DROP=quat, rbf", "LENSEEK_FULLGRAPH=1", "W_KL=0", "LENSEEK_BATCH=32", "LENSEEK_SEED=2"):
        apply_switch(run, s)
    assert run.model.edge_drop == ("rbf", "quat") and run.model.fullgraph
    assert run.train.w_kl == 0.0 and run.train.batch_size == 32 and run.train.seed == 2
    with pytest.raises(ValueError):
        apply_switch(run, "LENSEEK_EDGE_DROP=rbf,_dir")
    with pytest.raises(ValueError):
        apply_switch(run, "LENSEEK_STOPGRAD=yes")
    with pytest.raises(ValueError):
        apply_switch(run, "LENSEEK_UNKNOWN=1")


def test_strict30_excludes_entries():
    run = load_run("strict30_s0")
    with open(run.exclude_entries) as f:
        assert len([l for l in f if l.strip()]) == 5963


def test_schedules():
    t = RunConfig().train
    assert get_beta(49, t) == 0.0 and get_beta(50, t) == 0.0
    assert get_beta(125, t) == pytest.approx(0.5) and get_beta(200, t) == pytest.approx(0.5)
    assert get_beta(87, t) == pytest.approx(0.5 * 37 / 75)
    assert get_lr(200, t) == pytest.approx(1e-4) and get_lr(1, t) == pytest.approx(1e-3 - 9e-4 / 200)


# ---------------------------------------------------------------- checkpoint
def expected_keys():
    rows = [l.rstrip("\n").split("\t") for l in open(ROOT / "tests" / "data" / "state_dict_keys.tsv")][1:]
    return {k: tuple(int(x) for x in s.split("x")) for k, s in rows}


def test_architecture_matches_released_state_dict():
    """Recorded from the main_s0 checkpoint: 166 tensors; this guards key names without downloading it."""
    sd = VAE.from_config(ModelConfig()).state_dict()
    exp = expected_keys()
    assert list(sd) == list(exp)
    assert {k: tuple(v.shape) for k, v in sd.items()} == exp


def test_nonadapted_head_shape():
    sd = VAE.from_config(ModelConfig(nonadapted=True)).state_dict()
    assert tuple(sd["out_node.weight"].shape) == (768, 512)


def test_registry_entries():
    reg = model_registry()
    assert {"main_s0", "strict30_s0", "v1.0"} <= set(reg)
    assert reg["v1.0"]["model"]["direction_frame"] == "global" and reg["v1.0"]["superseded"]
    for name in ("main_s0", "strict30_s0", "v1.0"):
        assert len(reg[name]["sha256"]) == 64 and reg[name]["epoch"] == 200


@pytest.mark.assets
def test_main_checkpoint_loads_strictly(main_checkpoint):
    from lenseek.checkpoint import load_model

    model, cfg, info = load_model(main_checkpoint)
    assert info.name == "main_s0" and info.epoch == 200
    assert cfg == ModelConfig()
    raw = torch.load(main_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    fresh = VAE.from_config(cfg)
    fresh.load_state_dict(raw, strict=True)  # raises on any mismatch
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw[k])


def test_unregistered_checkpoint_warns(tmp_path):
    from lenseek.checkpoint import load_model

    p = tmp_path / "random.pth"
    torch.save({"epoch": 1, "model_state_dict": VAE.from_config(ModelConfig()).state_dict()}, p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model, cfg, info = load_model(str(p))
    assert info.name is None and any("not a released checkpoint" in str(x.message) for x in w)


def test_run_info_switches_are_used(tmp_path):
    from lenseek.checkpoint import load_model

    cfg = ModelConfig(edge_drop=("dir",))
    p = tmp_path / "abl.pth"
    torch.save({"epoch": 200, "model_state_dict": VAE.from_config(cfg).state_dict(),
                "run_info": {"switches": {"EDGE_DROP": ["dir"], "FULLGRAPH": False}}}, p)
    _, got, info = load_model(str(p))
    assert got.edge_drop == ("dir",) and info.config_source == "checkpoint run_info"
