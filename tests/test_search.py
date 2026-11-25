"""Search: latent database, Chamfer distances (ragged sites and the padded per-query form of the evaluation folders),
deterministic top-k, and the example files (the queries featurize; the example database holds every expected hit).
The example search end to end is tests/test_example_search.py.
"""
import csv

import numpy as np
import pytest

from conftest import EXAMPLES
from lenseek.featurize import featurize_site, find_sites
from lenseek.latentdb import LatentDB
from lenseek.search import chamfer_matrix, chamfer_pair, chamfer_rows_cdist, topk


# ---------------------------------------------------------------- search
def random_db(rng, n_sites=40, dim=16):
    lat = [rng.normal(size=(int(rng.integers(10, 66)), dim)).astype(np.float32) for _ in range(n_sites)]
    return LatentDB.from_latents(lat, [f"e{i:03d}/ligand_pocket_1_A" for i in range(n_sites)], {"model": "test"}), lat


def test_chamfer_matrix_matches_bruteforce():
    rng = np.random.default_rng(0)
    db, lat = random_db(rng)
    queries = [rng.normal(size=(n, 16)).astype(np.float32) for n in (10, 23, 50)]
    D = chamfer_matrix(queries, db, chunk_rows=200, query_batch=2)
    ref = np.array([[chamfer_pair(q, s) for s in lat] for q in queries])
    assert np.allclose(D, ref, rtol=1e-4, atol=1e-4)


def test_padded_rows_match_bruteforce():
    """chamfer_rows_cdist on padded, masked sites equals the Chamfer distance of the unpadded point sets."""
    rng = np.random.default_rng(5)
    sizes, n, d = [3, 7, 5], 8, 4
    queries = [rng.normal(size=(k, d)).astype(np.float32) for k in (6, 2)]
    sites = [rng.normal(size=(k, d)).astype(np.float32) for k in sizes]
    pad = lambda x: np.vstack([x, np.full((n - len(x), d), 50.0, np.float32)])  # noqa: E731
    mask = lambda pts: np.stack([np.arange(n) < len(x) for x in pts])  # noqa: E731
    D = chamfer_rows_cdist(np.stack([pad(q) for q in queries]), mask(queries), np.stack([pad(x) for x in sites]),
                           mask(sites))
    ref = np.array([[chamfer_pair(q, x) for x in sites] for q in queries])
    assert D.shape == (2, 3) and D.dtype == np.float32 and np.allclose(D, ref, rtol=1e-5)


def test_self_distance_is_zero_and_top1_is_self():
    rng = np.random.default_rng(1)
    db, lat = random_db(rng)
    D = chamfer_matrix(lat[:5], db)
    for i in range(5):
        assert D[i, i] < 1e-3
        assert topk(D[i], 1)[0] == i


def test_topk_ties_by_index():
    row = np.array([3.0, 1.0, 2.0, 1.0, 1.0, 0.5], dtype=np.float32)
    assert topk(row, 3).tolist() == [5, 1, 3]
    assert topk(row, 10).tolist() == [5, 1, 3, 4, 2, 0]


@pytest.mark.parametrize("ext", ["", ".npz"])
def test_save_load_roundtrip(tmp_path, ext):
    rng = np.random.default_rng(2)
    db, lat = random_db(rng, 12)
    path = str(tmp_path / ("db" + ext))
    db.save(path)
    db2 = LatentDB.load(path)
    assert db2.site_ids == db.site_ids and len(db2) == 12
    for i in range(12):
        assert np.array_equal(db2.site(i), lat[i])
    assert db2.meta["model"] == "test"


def test_patch_overlay(tmp_path):
    rng = np.random.default_rng(3)
    db, lat = random_db(rng, 10)
    new = [lat[i] + 1.0 for i in (2, 7)]
    patch = str(tmp_path / "patch.npz")
    LatentDB.write_patch(patch, db, [2, 7], new, meta={"name": "t", "edition": "standard"})
    q = [lat[2]]
    before = chamfer_matrix(q, db)
    db.apply_patch(patch)
    after = chamfer_matrix(q, db)
    assert np.array_equal(db.site(7), new[1]) and db.meta["edition"] == "standard"
    assert after[0, 2] > before[0, 2] + 1.0
    untouched = [i for i in range(10) if i not in (2, 7)]
    assert np.allclose(after[0, untouched], before[0, untouched])
    bad = LatentDB.from_latents([x[:-1] if i == 2 else x for i, x in enumerate(lat)], db.site_ids)
    with pytest.raises(ValueError):
        bad.apply_patch(patch)


def test_from_padded_drops_padding():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=(3, 65, 8)).astype(np.float32)
    mask = np.zeros((3, 65), bool)
    for i, n in enumerate((10, 30, 65)):
        mask[i, :n] = True
    db = LatentDB.from_padded(mu, mask, ["a", "b", "c"])
    assert db.sizes.tolist() == [10, 30, 65]
    assert np.array_equal(db.site(1), mu[1, :30])


# ---------------------------------------------------------------- end to end
def expected_hits():
    out = {}
    for row in csv.DictReader(open(EXAMPLES / "expected_hits.tsv"), delimiter="\t"):
        out.setdefault(row["query"], []).append((row["site_id"], float(row["chamfer_distance_paper"])))
    return out


def test_example_files_featurize():
    pairs = find_sites(str(EXAMPLES / "queries"))
    assert len(pairs) == 4
    for pocket, protein in pairs:
        s = featurize_site(pocket, protein, str(EXAMPLES / "embeddings"))
        assert s.h.shape == (len(s), 768) and s.T.shape == (len(s), 3) and s.R.shape == (len(s), 3, 3)
        assert np.allclose(np.einsum("nij,nik->njk", s.R, s.R), np.eye(3), atol=1e-5)   # orthonormal frames
        assert 10 <= len(s) <= 65


def test_example_db_contains_expected_hits():
    db = LatentDB.load(str(EXAMPLES / "db" / "example_db.npz"))
    assert 100 <= len(db) <= 1000 and db.dim == 256
    for hits in expected_hits().values():
        for site_id, _ in hits:
            db.index_of(site_id)
