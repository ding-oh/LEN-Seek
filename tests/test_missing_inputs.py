"""Missing or unreleased inputs stop with one line before any download or launch (training features, a CUDA device,
the scaling study's added sites, the inputs of a result of analysis/run.py), which
takes a variable that is not set from the unpacked asset that holds it;
the asset helper prints full descriptions, says when it links a file from a directory mirror and refuses a directory
unpacked from another version of an archive; the notebook runs the results of analysis/run.py and is saved without
outputs; the reconstruction set of eval_extras holds relative site paths."""
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import ROOT


def run(cmd, tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("LENSEEK_")}
    e.update(PATH=os.path.dirname(sys.executable) + os.pathsep + e.get("PATH", ""), LENSEEK_HOME=str(tmp_path / "home"),
             **env)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=e, timeout=300)


def test_run_config_home_default(monkeypatch, tmp_path):
    from lenseek.config import load_run

    monkeypatch.delenv("LENSEEK_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert load_run("main_s0").data == str(tmp_path / ".cache/lenseek/assets/train_features")
    monkeypatch.setenv("LENSEEK_HOME", str(tmp_path / "lh"))
    assert load_run("main_s0").data == str(tmp_path / "lh") + "/assets/train_features"


def test_train_without_features(monkeypatch, tmp_path):
    from lenseek.cli import train

    monkeypatch.setenv("LENSEEK_HOME", str(tmp_path))
    single = tmp_path / "my_run.yaml"
    single.write_text("name: my_run\ndata: ${LENSEEK_HOME}/assets/train_features\n")
    for argv in (["--run", "main_s0"], ["--config", str(single)]):
        with pytest.raises(SystemExit) as e:
            train.main(argv)
        assert "lenseek-fetch train_features" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        train.main(["--run", "main_s9"])
    assert "no run 'main_s9'" in str(e.value.code)


def test_train_sh_stops_before_launch(tmp_path):
    single = tmp_path / "my_run.yaml"
    single.write_text("name: my_run\ndata: ${LENSEEK_HOME}/assets/train_features\n")
    for arg in ("main_s0", str(single)):
        p = run(["bash", "examples/train.sh", arg], tmp_path)
        assert p.returncode != 0 and "lenseek-fetch train_features" in p.stderr
        assert "Traceback" not in p.stderr and len(p.stderr.strip().splitlines()) == 1


def test_cuda_device_unavailable(monkeypatch, capsys):
    import torch
    from lenseek.cli import search
    from lenseek.cli.common import check_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    check_device("cpu")
    with pytest.raises(SystemExit) as e:
        search.main([str(ROOT / "data" / "example" / "queries"), "--device", "cuda"])
    assert e.value.code == 2 and "CUDA is not available" in capsys.readouterr().err


def test_scaling_real150k_needs_added(tmp_path):
    p = run(["bash", "examples/scaling.sh", "accuracy", "real", "real150k"], tmp_path, LENSEEK_OUT=str(tmp_path))
    assert p.returncode == 2 and "scaling/added" in p.stderr and "Traceback" not in p.stderr


def test_run_skips_a_result_without_its_inputs(tmp_path):
    p = run([sys.executable, "analysis/run.py", "fig3"], tmp_path, LENSEEK_OUT=str(tmp_path / "out"))
    assert p.returncode == 3 and "Traceback" not in p.stderr
    assert p.stdout.strip() == ("[skip] fig3: $LENSEEK_EVAL_DIR not set (or lenseek-fetch eval_main_s0); "
                                "$LENSEEK_PAIRS not set (or lenseek-fetch labels_probis_glosa)")


def test_run_takes_unset_variables_from_unpacked_assets(tmp_path):
    import yaml

    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from run import FROM_ASSETS

    reg = yaml.safe_load(open(os.path.join(ROOT, "lenseek", "configs", "assets.yaml")))["assets"]
    assert all(reg[a]["file"] and (reg[a]["kind"] == "tar" or not sub) for a, sub in FROM_ASSETS.values())
    cmd = [sys.executable, "analysis/run.py", "--print", "table1"]
    p = run(cmd, tmp_path, LENSEEK_OUT=str(tmp_path / "out"))
    assert "$LENSEEK_SCAN_DIR not set (or lenseek-fetch labels_probis_glosa)" in p.stdout
    assert not (tmp_path / "home").exists()                 # looking for assets creates nothing
    labels = tmp_path / "home" / "assets" / "labels_probis_glosa"
    for sub in ("scan_probis", "scan_glosa", "pairs"):
        (labels / reg["labels_probis_glosa"]["root"] / sub).mkdir(parents=True)
    (labels / ".unpacked").write_text("")
    p = run(cmd, tmp_path, LENSEEK_OUT=str(tmp_path / "out"))
    assert "LENSEEK_SCAN_DIR" not in p.stdout.split("$ python")[0] and "LENSEEK_PAIRS_DIR" not in p.stdout
    p = run(cmd, tmp_path, LENSEEK_OUT=str(tmp_path / "out"), LENSEEK_SCAN_DIR=str(tmp_path / "mine"))
    assert f"$LENSEEK_SCAN_DIR={tmp_path / 'mine'} does not exist" in p.stdout     # a set variable wins
    derived = tmp_path / "home" / "assets" / "derived_tables"
    strata = derived / reg["derived_tables"]["root"] / "site_annotation" / "protein_strata.tsv"
    strata.parent.mkdir(parents=True)
    strata.write_text("x\n")
    (derived / ".unpacked").write_text("")
    p = run(cmd, tmp_path, LENSEEK_OUT=str(tmp_path / "out"))
    copy = tmp_path / "out" / "site_annotation" / "protein_strata.tsv"
    assert copy.read_text() == "x\n" and not copy.is_symlink() and "protein_strata.tsv does not exist" not in p.stdout


def test_fetch_list_and_mirror_link(tmp_path):
    cfg, mirror = tmp_path / "cfg", tmp_path / "mirror"
    cfg.mkdir()
    mirror.mkdir()
    (mirror / "tiny.bin").write_bytes(b"lenseek")
    desc = "a test asset whose description is longer than seventy characters, so that it would have been cut"
    (cfg / "assets.yaml").write_text(json.dumps({"assets": {"tiny": {
        "description": desc, "file": "tiny.bin", "size": 7, "sha256": hashlib.sha256(b"lenseek").hexdigest(),
        "url": None, "kind": "file"}}}))
    env = dict(LENSEEK_CONFIG_DIR=str(cfg), LENSEEK_ASSET_MIRROR=str(mirror))
    p = run([sys.executable, "-m", "lenseek.cli.fetch", "--list"], tmp_path, **env)
    assert p.returncode == 0 and desc in p.stdout
    p = run([sys.executable, "-m", "lenseek.cli.fetch", "tiny"], tmp_path, **env)
    assert p.returncode == 0 and "linking tiny from" in p.stderr
    assert (tmp_path / "home" / "assets" / "tiny.bin").is_symlink()


def test_fetch_unreachable_url_is_one_line(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "assets.yaml").write_text(json.dumps({"assets": {"tiny": {
        "file": "tiny.bin", "size": 7, "sha256": None, "url": (tmp_path / "missing.bin").as_uri(), "kind": "file"}}}))
    p = run([sys.executable, "-m", "lenseek.cli.fetch", "tiny"], tmp_path, LENSEEK_CONFIG_DIR=str(cfg))
    assert p.returncode == 1 and "Traceback" not in p.stderr and "tiny: download from" in p.stderr
    assert not list((tmp_path / "home" / "assets").glob("tiny.bin*"))


def test_fetch_refuses_a_stale_unpack(monkeypatch, tmp_path):
    import io
    import tarfile

    from lenseek import assets

    tar = tmp_path / "mirror" / "tiny.tar"
    tar.parent.mkdir()
    with tarfile.open(tar, "w") as t:
        info = tarfile.TarInfo("tiny/x.txt")
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    reg = {"tiny": dict(description="t", file="tiny.tar", size=tar.stat().st_size, url=None, kind="tar", root="tiny",
                        sha256=hashlib.sha256(tar.read_bytes()).hexdigest())}
    monkeypatch.setattr(assets, "registry", lambda: reg)
    monkeypatch.setenv("LENSEEK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LENSEEK_ASSET_MIRROR", str(tar.parent))
    assert (assets.fetch("tiny", quiet=True) / "x.txt").read_text() == "x"
    (tmp_path / "home" / "assets" / "tiny" / ".unpacked").write_text("0" * 64)
    with pytest.raises(assets.AssetError, match="another version"):
        assets.fetch("tiny", quiet=True)


def test_notebook_runs_the_results_of_run_py():
    sys.path.insert(0, str(ROOT / "analysis"))
    import run as R

    nb = json.load(open(ROOT / "notebooks" / "reproduce_figures.ipynb"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    called = re.findall(r"^result\('([^']+)'", "\n".join("".join(c["source"]) for c in code), flags=re.M)
    assert called == [r[0] for r in R.RESULTS]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code)


@pytest.mark.assets
def test_recon_set_has_relative_site_paths():
    from lenseek.assets import AssetError, fetch
    from lenseek.data import load_legacy_pickle

    try:
        root = fetch("eval_extras", quiet=True)
    except (AssetError, OSError) as e:
        pytest.skip(f"eval_extras not available: {e}")
    paths = load_legacy_pickle(str(root / "reconstruction_set.pkl")).path
    assert len(paths) == 1310 and all(re.fullmatch(r"[0-9a-z]{4}/ligand_pocket_\d+_[A-Za-z0-9]+\.pdb", p) for p in paths)
