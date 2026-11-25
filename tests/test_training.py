"""Training smoke test on CPU: one epoch over a few synthetic sites writes a loadable checkpoint with provenance."""
import json

import numpy as np
import pytest

from conftest import synthetic_site
from lenseek.config import RunConfig
from lenseek.data import FeatureStore, SiteDataset


def _store(tmp_path, sizes=(12, 20, 70, 30)):
    rng = np.random.default_rng(0)
    hs, Ts, Rs, ids = [], [], [], []
    for i, n in enumerate(sizes):
        h, T, R, _ = synthetic_site(rng, n)
        hs.append(h), Ts.append(T), Rs.append(R), ids.append(f"e{i:03d}/ligand_pocket_1_A")
    path = str(tmp_path / "features")
    FeatureStore.write_arrays(path, hs, Ts, Rs, ids)
    return path


def test_dataset_sampling(tmp_path):
    ds = SiteDataset(FeatureStore(_store(tmp_path)))
    np.random.seed(0)
    h, t, mask, r = ds[0]                 # 12 residues: wrap padding
    assert h.shape == (65, 768) and int(mask.sum()) == 12 and np.array_equal(t[12].numpy(), t[0].numpy())
    h, t, mask, r = ds[2]                 # 70 residues: 65 drawn without replacement
    assert int(mask.sum()) == 65 and len({tuple(x) for x in t.numpy().round(4)}) == 65


@pytest.mark.slow
def test_one_epoch_on_cpu(tmp_path):
    pytest.importorskip("accelerate")
    from lenseek.checkpoint import load_model
    from lenseek.train import main

    run = RunConfig(name="smoke", data=_store(tmp_path))
    run.train.epochs, run.train.batch_size, run.train.grad_accum, run.train.save_every = 1, 2, 1, 1
    out = str(tmp_path / "run")
    main(run, out, progress=False)
    hist = [json.loads(l) for l in open(f"{out}/loss_history.jsonl")]
    assert len(hist) == 1 and np.isfinite(hist[0]["total_loss"])
    model, cfg, info = load_model(f"{out}/final.pth")
    assert info.config_source == "checkpoint run_info" and info.epoch == 1
    assert info.run_info["switches"]["W_FAPE"] == 20.0
