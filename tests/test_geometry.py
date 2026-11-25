"""Geometry of the encoder and the loss: rotation invariance, the Kabsch RMSD, the kNN graph.

A rigid motion of a binding site must not change its encoding (paper eq. 10: directions in the residue frame).

Small sites (<= 15 residues) carry one kNN self-edge per residue; its RBF feature depends on the floating-point
residue of torch.cdist on the diagonal, so their latents move by up to a few 1e-3 under rotation (the same noise
separates CPU and GPU encodings). Chamfer distances stay many orders of magnitude below typical nearest-neighbour
distances (tens).

Kabsch RMSD of the training loss: zero for a rigidly moved copy, equal to a numpy reference otherwise.

Applying the optimal rotation transposed (P @ R instead of P @ R^T) gives a non-zero RMSD for an exact copy; the last
Kabsch test checks this.

kNN graph: ties at +inf are ranked by column index, so a site with <= 15 residues gets one self-edge per residue
(the graph every model was trained with on GPUs); larger sites never have self-edges.
"""
import numpy as np
import pytest
import torch

from conftest import EXAMPLES, random_rotation, synthetic_site
from lenseek.config import ModelConfig
from lenseek.encode import Encoder, pad_site
from lenseek.featurize import residue_frames
from lenseek.geometry import get_edges, knn_indices
from lenseek.loss import kabsch_rmsd
from lenseek.search import chamfer_pair


# ---------------------------------------------------------------- rotation invariance
MAXABS_TOL = 5e-3
CD_TOL = 1e-4


def _rotated(rng, atoms):
    Q = random_rotation(rng)
    t = rng.normal(size=3) * 20.0
    n_at, ca, c_at = atoms
    return residue_frames(n_at @ Q.T + t, ca @ Q.T + t, c_at @ Q.T + t)


@pytest.mark.parametrize("n", [11, 15, 16, 40, 70])
def test_invariance_any_weights(random_model, n):
    model, cfg = random_model
    enc = Encoder(model, cfg)
    rng = np.random.default_rng(n)
    h, T, R, atoms = synthetic_site(rng, n)
    for _ in range(3):
        T2, R2 = _rotated(rng, atoms)
        a, b = enc.encode([(h, T, R), (h, T2, R2)], warn_truncation=False)
        assert np.abs(a - b).max() < MAXABS_TOL
        assert chamfer_pair(a, b) < CD_TOL


def test_global_direction_frame_is_not_invariant(random_model):
    """Direction features in the laboratory frame change under rotation."""
    model, _ = random_model
    cfg = ModelConfig(direction_frame="global")
    enc = Encoder(model, cfg)
    rng = np.random.default_rng(1)
    h, T, R, atoms = synthetic_site(rng, 30)
    T2, R2 = _rotated(rng, atoms)
    a, b = enc.encode([(h, T, R), (h, T2, R2)])
    assert np.abs(a - b).max() > 1e-2


@pytest.mark.assets
def test_invariance_released_model_keeps_top1(main_model):
    """Rotating each example query leaves its encoding and its top-1 hit in the example database unchanged."""
    from lenseek.featurize import featurize_site, find_sites
    from lenseek.latentdb import LatentDB
    from lenseek.search import chamfer_matrix

    model, cfg, _ = main_model
    enc = Encoder(model, cfg)
    db = LatentDB.load(str(EXAMPLES / "db" / "example_db.npz"))
    rng = np.random.default_rng(0)
    for pocket, protein in find_sites(str(EXAMPLES / "queries")):
        s = featurize_site(pocket, protein, str(EXAMPLES / "embeddings"))
        # rotate the backbone atoms rigidly: positions and frames follow
        Q = random_rotation(rng)
        t = rng.normal(size=3) * 30.0
        T2 = s.T @ Q.T.astype(np.float32) + t.astype(np.float32)
        R2 = np.einsum("ij,njk->nik", Q.astype(np.float32), s.R)
        a, b = enc.encode([s, (s.h, T2, R2)])
        assert chamfer_pair(a, b) < CD_TOL
        d = chamfer_matrix([a, b], db)
        assert d[0].argmin() == d[1].argmin()


# ---------------------------------------------------------------- kabsch
def numpy_kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T                    # rotates column vectors: q = R p
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def _batch(P, Q, n_valid=None):
    n = len(P)
    mask = torch.ones(1, n, dtype=torch.bool)
    if n_valid is not None:
        mask[0, n_valid:] = False
    return torch.tensor(P[None], dtype=torch.float64), torch.tensor(Q[None], dtype=torch.float64), mask


def test_rigid_copy_has_zero_rmsd():
    rng = np.random.default_rng(0)
    for _ in range(5):
        P = rng.normal(size=(30, 3)) * 5
        Q = P @ random_rotation(rng).T + rng.normal(size=3) * 10
        rmsd = kabsch_rmsd(*_batch(P, Q)).item()
        assert rmsd < 1e-3          # sqrt(eps) floor of the loss (eps = 1e-8)


def test_matches_numpy_reference():
    rng = np.random.default_rng(1)
    for _ in range(5):
        P = rng.normal(size=(25, 3)) * 5
        Q = P @ random_rotation(rng).T + rng.normal(scale=0.7, size=P.shape)
        ours = kabsch_rmsd(*_batch(P, Q)).item()
        assert abs(ours - numpy_kabsch_rmsd(P, Q)) < 1e-6


def test_padding_is_ignored():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(20, 3)) * 5
    Q = P @ random_rotation(rng).T + rng.normal(scale=0.5, size=P.shape)
    full = kabsch_rmsd(*_batch(P[:14], Q[:14])).item()
    padded = kabsch_rmsd(*_batch(P, Q, n_valid=14)).item()
    assert abs(full - padded) < 1e-6


def test_transposed_application_was_wrong():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(30, 3)) * 5
    Q = P @ random_rotation(rng).T
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ U.T
    correct = np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean())
    transposed = np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean())
    assert correct < 1e-6 and transposed > 1.0


# ---------------------------------------------------------------- knn
def reference_ties_by_column(D, k):
    """Brute force: sort each row by (value, column)."""
    out = []
    for row in D:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))
        out.append(order[:k])
    return np.array(out)


def test_knn_indices_rank_ties_by_column():
    rng = np.random.default_rng(0)
    D = rng.integers(0, 5, size=(64, 65)).astype(np.float32)
    D[rng.random(D.shape) < 0.4] = np.inf
    for k in (1, 5, 15):
        got = knn_indices(torch.from_numpy(D), k).numpy()
        assert np.array_equal(got, reference_ties_by_column(D, k))


def _graph(n):
    rng = np.random.default_rng(n)
    h, T, R, _ = synthetic_site(rng, n)
    _, T, R, M = pad_site(h, T, R)
    T = torch.from_numpy(T)[None]
    R = torch.from_numpy(R)[None]
    M = torch.from_numpy(M)[None]
    T = T - (T * M[..., None]).sum(1, keepdim=True) / M.sum(1, keepdim=True)[..., None]
    e, mask = get_edges(T, R, M.unsqueeze(2) & M.unsqueeze(1))
    return e[0], mask[0], M[0]


@pytest.mark.parametrize("n", [10, 12, 15])
def test_small_sites_have_self_edges(n):
    e, mask, M = _graph(n)
    diag = torch.diagonal(mask)[:n]
    assert bool(diag.all())
    # a valid residue never connects to padding
    assert not bool(mask[:n, n:].any()) and not bool(mask[n:, :].any())
    # every valid pair is connected (n - 1 < 15 neighbours available)
    assert bool(mask[:n, :n].all())


@pytest.mark.parametrize("n", [16, 30, 65])
def test_larger_sites_have_no_self_edges(n):
    e, mask, M = _graph(n)
    assert not bool(torch.diagonal(mask).any())
    deg = mask[:n, :n].sum(1)
    assert int(deg.min()) >= 15          # 15 nearest neighbours, symmetrised


def test_knn_is_independent_of_residue_order():
    rng = np.random.default_rng(3)
    h, T, R, _ = synthetic_site(rng, 13)
    perm = rng.permutation(13)
    graphs = []
    for order in (np.arange(13), perm):
        _, Tp, Rp, M = pad_site(h[order], T[order], R[order])
        Tt = torch.from_numpy(Tp)[None]
        Mt = torch.from_numpy(M)[None]
        _, mask = get_edges(Tt, torch.from_numpy(Rp)[None], Mt.unsqueeze(2) & Mt.unsqueeze(1))
        graphs.append(mask[0, :13, :13].numpy())
    inv = np.argsort(perm)
    assert np.array_equal(graphs[0], graphs[1][np.ix_(inv, inv)])
