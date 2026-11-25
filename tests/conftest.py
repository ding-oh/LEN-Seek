"""Shared fixtures. Tests run on CPU; the ones that need the released checkpoint are marked ``assets`` and are
skipped when it is neither cached in $LENSEEK_HOME nor reachable (URL or $LENSEEK_ASSET_MIRROR)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(int(os.environ.get("LENSEEK_TEST_THREADS", "2")))

EXAMPLES = ROOT / "data" / "example"


@pytest.fixture(scope="session")
def main_checkpoint():
    from lenseek.assets import AssetError, fetch

    try:
        return str(fetch("checkpoint_main_s0", quiet=True))
    except (AssetError, OSError) as e:
        pytest.skip(f"main_s0 checkpoint not available: {e}")


@pytest.fixture(scope="session")
def main_model(main_checkpoint):
    from lenseek.checkpoint import load_model

    return load_model(main_checkpoint)


@pytest.fixture(scope="session")
def random_model():
    """Untrained VAE with the main-model architecture (invariance holds for any weights)."""
    from lenseek.config import ModelConfig
    from lenseek.model import VAE

    torch.manual_seed(0)
    cfg = ModelConfig()
    return VAE.from_config(cfg).eval(), cfg


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def synthetic_site(rng, n, node_dim=768):
    """A site with a plausible backbone: random C-alpha walk, N and C placed around each C-alpha."""
    from lenseek.featurize import residue_frames

    ca = np.cumsum(rng.normal(scale=2.2, size=(n, 3)), axis=0)
    n_at = ca + rng.normal(scale=1.0, size=(n, 3))
    c_at = ca + rng.normal(scale=1.0, size=(n, 3))
    T, R = residue_frames(n_at, ca, c_at)
    h = rng.normal(size=(n, node_dim)).astype(np.float32)
    return h, T, R, (n_at, ca, c_at)
