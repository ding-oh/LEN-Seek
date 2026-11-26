"""Analysis and evaluation scripts on synthetic inputs: the run configurations of the variants and of the
retraining without the ligand filters, evaluating a checkpoint on a feature store (scripts/evaluate.py, also in a
database order that lists sites the store lacks, with rotated queries and as sites added to an evaluated database),
the paired contrasts and the out-of-fold Table 1 interface of ablations.py, the --db-entries option of
fig4_ligand.py grid, the rotation summary of model_figures.py, the distance quartiles of scripts/deeplytough.py
assemble, the G-LoSA ROC-AUC of baselines.py on the common pair set, the readout of the retraining (including a
retrained model identical to the original model: every difference 0, interval [0, 0]), the shared helpers of
analysis/common.py, the result list of analysis/run.py, every analysis script loading with its helpers, the identity
strata of scripts/site_annotation.py, and the entry selection, the construction counts and the check of the build
rules against BsitePDB of scripts/holdout.py."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from lenseek.config import load_run, switches_of
from lenseek.exclusion import valid_ligand

QS = ROOT / "data" / "query_sets"
EX = ROOT / "data" / "example"
A = ROOT / "analysis"
S = ROOT / "scripts"
sys.path.insert(0, str(A))


def script(name, *args, folder=A, **kw):
    return subprocess.run([sys.executable, str(folder / name), *map(str, args)], check=True, stdout=subprocess.DEVNULL,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)), **kw)


@pytest.mark.parametrize("name, expect", [
    ("abl_norbf", {"EDGE_DROP": ["rbf"], "FULLGRAPH": False, "W_FAPE": 20.0}),
    ("abl_noquat", {"EDGE_DROP": ["quat"], "FULLGRAPH": False, "W_RMSD": 5.0}),
    ("abl_ankh_fullgraph_na", {"EDGE_DROP": ["rbf", "dir", "quat"], "FULLGRAPH": True, "NONADAPTED": True,
                               "W_FAPE": 0.0, "W_RMSD": 0.0, "W_COS": 40.0}),
    ("filter_free_s0", {"EDGE_DROP": [], "FULLGRAPH": False, "NONADAPTED": False, "W_FAPE": 20.0, "W_KL": 1.0}),
    ("strict50_s0", {"EDGE_DROP": [], "W_FAPE": 20.0}),
])
def test_added_run_configs(name, expect):
    run = load_run(name)
    sw = switches_of(run)
    for k, v in expect.items():
        assert sw[k] == v, (name, k)
    assert run.train.batch_size * run.train.grad_accum * 2 == 512 and run.train.seed == 0


def test_filter_free_run_differs_from_main_only_in_data():
    a = load_run("main_s0")
    b = load_run("filter_free_s0")
    assert switches_of(a) == switches_of(b) and a.train == b.train and a.model == b.model
    assert a.data != b.data and not b.exclude_entries


def test_strict50_list_is_inside_strict30():
    run = load_run("strict50_s0")
    s50 = {l.strip() for l in open(run.exclude_entries) if l.strip()}
    s30 = {l.strip() for l in open(ROOT / "data" / "strict30_exclude_entries.txt") if l.strip()}
    assert len(s50) == 1502 and s50 <= s30


def test_query_ligand_table():
    t = pd.read_csv(QS / "query_ligands.tsv", sep="\t", comment="#", dtype=str, keep_default_na=False)
    assert len(t) == 481 and t.query_key.is_unique
    assert all((f == "1") == (not valid_ligand(c)) for c, f in zip(t.ccd, t.filtered))
    excl = {l.split()[0].lower() for l in open(QS / "primary_exclude_pdb.txt") if l.strip() and not l.startswith("#")}
    f = t[t.filtered == "1"]
    assert len(f) == 75 and int((~f.entry.str.lower().isin(excl)).sum()) == 64
    assert set(t.stratum) == {"drug-like", "cofactor / endogenous", "additive / other"}


def test_filter_reason_matches_valid_ligand():
    """scripts/filter_free_data.py reason against lenseek.exclusion.valid_ligand, in its own process (scripts/ on
    sys.path)."""
    code = f"""
import sys
sys.path.insert(0, {str(S)!r})
import pandas as pd
import filter_free_data as N
from lenseek.exclusion import valid_ligand
assert N.reason("GOL")[0] == "exclusion_list" and N.reason("1N1") == ("digit_start", "digit_start")
assert N.reason("ATP") == ("valid", "valid")
for c in pd.read_csv({str(QS / "query_ligands.tsv")!r}, sep="\\t", comment="#", dtype=str).ccd:
    assert valid_ligand(c) == (N.reason(c)[0] == "valid"), c
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))


def test_evaluate_on_a_feature_store(tmp_path, random_model):
    """scripts/evaluate.py with a database feature store whose sites are the queries: same latents, zero
    self-distance."""
    import pickle

    import torch
    from conftest import synthetic_site
    from lenseek.data import FeatureStore
    model, _ = random_model
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, tmp_path / "m.pth")
    rng = np.random.default_rng(0)
    sites = [synthetic_site(rng, n)[:3] for n in (12, 30, 70)]
    ids = ["1abc/ligand_pocket_1_A", "1abc/ligand_pocket_2_B", "2xyz/ligand_pocket_1_A"]
    FeatureStore.write_arrays(str(tmp_path / "db"), [s[0] for s in sites], [s[1] for s in sites], [s[2] for s in sites],
                              ids)
    pickle.dump({"queries": [dict(path=f"/q/{i}.pdb", h=h, T=T, R=R) for i, (h, T, R) in zip(ids, sites)]},
                open(tmp_path / "q.pkl", "wb"))
    script("evaluate.py", "--checkpoint", tmp_path / "m.pth", "--features", tmp_path / "db", "--queries",
           tmp_path / "q.pkl", "--out", tmp_path / "out", "--threads", "1", "--rotations", "2", folder=S)
    o = tmp_path / "out"
    assert [l.strip() for l in open(o / "db_paths.txt")] == [f"{i}.pdb" for i in ids]
    q, d = np.load(o / "q_mu.npy"), np.load(o / "db_mu.npy")
    assert np.array_equal(q, d) and np.load(o / "q_mask.npy").sum(1).tolist() == [12, 30, 65]
    dist = np.load(o / "dist.npy")
    assert dist.shape == (3, 3) and np.all(np.abs(np.diag(dist)) < 1e-3) and (np.argmin(dist, 1) == [0, 1, 2]).all()
    rot = pd.read_csv(o / "rotation.tsv", sep="\t")      # the encodings do not depend on the orientation
    assert rot.site.tolist() == [f"/q/{i}.pdb" for i in ids for _ in range(2)] and rot.rotation.tolist() == [0, 1] * 3
    assert (rot.rank_of_top1 == 1).all() and rot.top1.tolist() == [0, 0, 1, 1, 2, 2]
    assert (rot.chamfer_to_unrotated < 1e-3).all() and (rot.chamfer_to_top1 < 1e-3).all()
    # a database order with a site the store lacks: NaN column, all-False mask row; encoded sites in db_subset/
    order = [f"{ids[0]}.pdb", "9zzz/ligand_pocket_1_A.pdb", f"{ids[1]}.pdb", f"{ids[2]}.pdb"]
    (tmp_path / "order.txt").write_text("\n".join(order) + "\n")
    script("evaluate.py", "--checkpoint", tmp_path / "m.pth", "--features", tmp_path / "db", "--queries",
           tmp_path / "q.pkl", "--out", tmp_path / "out2", "--threads", "1", "--db-order",
           tmp_path / "order.txt", folder=S)
    o2 = tmp_path / "out2"
    d2 = np.load(o2 / "dist.npy")
    assert d2.shape == (3, 4) and np.isnan(d2[:, 1]).all() and np.array_equal(d2[:, [0, 2, 3]], dist)
    assert not np.load(o2 / "db_mask.npy")[1].any() and [l.strip() for l in open(o2 / "db_paths.txt")] == order
    assert np.load(o2 / "db_subset" / "db_index.npy").tolist() == [0, 2, 3]
    assert np.array_equal(np.load(o2 / "db_subset" / "db_mu.npy"), d)
    # the store as sites added to the database of out/: its query encodings, fp16 latents, the same distances
    script("evaluate.py", "--checkpoint", tmp_path / "m.pth", "--features", tmp_path / "db", "--added", o,
           "--out", tmp_path / "added", "--threads", "1", folder=S)
    a = tmp_path / "added"
    assert np.array_equal(np.load(a / "added_mu.npy"), d.astype(np.float16))
    assert np.array_equal(np.load(a / "added_mask.npy"), np.load(o / "db_mask.npy"))
    assert np.array_equal(np.load(a / "dist_added.npy"), dist)
    assert [l.strip() for l in open(a / "added_keys.txt")] == [f"{i}.pdb" for i in ids]
    t = json.load(open(a / "encode_time.json"))
    assert t["added_sites"] == 3 and t["threads"] == 1 and t["encode_seconds"] > 0


# --------------------------------------------------------------------------- synthetic per-query values
def _queries():
    return pd.read_csv(QS / "query_ligands.tsv", sep="\t", comment="#", dtype=str, keep_default_na=False)


def _primary(t):
    excl = {l.split()[0].lower() for l in open(QS / "primary_exclude_pdb.txt") if l.strip() and not l.startswith("#")}
    return t[~t.entry.str.lower().isin(excl)]


def _per_query(rng, shift_f=0.0):
    """Per-query whole-database ligand ROC-AUC of the 481 queries (fig4_ligand.py grid columns); F queries + shift_f."""
    t = _queries()
    auc = rng.uniform(0.3, 0.9, len(t))
    auc = np.where(t.filtered == "1", auc + shift_f, auc)
    n_pos = rng.integers(0, 50, len(t))
    return pd.DataFrame(dict(query_key=t.query_key, lenseek_raw__auc=auc, n_pos=n_pos, n_db=114661))


def _did(tmp_path, retrain, seeds, name):
    out = tmp_path / name
    script("filter_retrain.py", "did", "--retrain", retrain, "--base", tmp_path / "base.csv", "--sets",
           tmp_path / "sets", "--out", out, "--n-boot", "200", *sum([["--seed", s] for s in seeds], []))
    j = json.load(open(out / "did.json"))
    return j, {r["set"]: r for r in j["rows"] if r["unit"] == "cluster50"}


@pytest.fixture(scope="module")
def did_inputs(tmp_path_factory):
    """Per-query files of an original model, of retrained models (+0.10 on the F queries) and of two further seeds
    (+-0.004), and set (b) without four R queries of one entry."""
    import filter_retrain as FR
    tmp = tmp_path_factory.mktemp("did")
    rng = np.random.default_rng(0)
    base = _per_query(rng)
    base.to_csv(tmp / "base.csv", index=False)
    d = FR.prepare(base, FR.COL, "primary")
    (tmp / "sets").mkdir()
    ex = d[d.group == "R"].head(4)
    pd.DataFrame(dict(query_key=ex.query_key, protein=ex.entry, group=ex.group)).to_csv(
        tmp / "sets" / "did_exclude_ge80.tsv", sep="\t", index=False)
    isF = (_queries().filtered == "1").to_numpy()
    for name, shift in [("plus10", 0.10), ("seed_p", 0.004), ("seed_m", -0.004)]:
        x = base.copy()
        x[FR.COL] = np.where(isF, x[FR.COL] + shift, x[FR.COL])
        x.to_csv(tmp / f"{name}.csv", index=False)
    return tmp


def test_retrained_model_identical_to_main(did_inputs):
    t = did_inputs
    j, r = _did(t, t / "base.csv", [t / "base.csv", t / "seed_p.csv", t / "seed_m.csv"], "fake_three_seeds")
    for row in j["rows"]:
        assert row["DiD"] == 0 and row["ci_lo"] == 0 and row["ci_hi"] == 0
    assert j["verdict_code"] == "not_detected" and r["a_primary"]["exceeds_2sd"] is False


def test_retrained_model_better_on_filtered_queries(did_inputs):
    t = did_inputs
    j, r = _did(t, t / "plus10.csv", [t / "base.csv", t / "seed_p.csv", t / "seed_m.csv"], "plus10")
    assert r["a_primary"]["DiD"] == pytest.approx(0.10) and r["b_primary_minus_ge80"]["DiD"] == pytest.approx(0.10)
    assert j["verdict_code"] == "detected"


def test_did_needs_three_seeds(did_inputs):
    t = did_inputs
    with pytest.raises(subprocess.CalledProcessError):
        _did(t, t / "plus10.csv", [t / "base.csv", t / "seed_p.csv"], "plus10_two_seeds")


def test_filter_compare_shift():
    import filter_retrain as FR
    rng = np.random.default_rng(1)
    d = FR.prepare(_per_query(rng, shift_f=-0.2), FR.COL, "primary")
    assert len(d) == 448
    r = FR.compare(d, FR.COL, "cluster50", n_boot=500)
    assert r["D"] < -0.1 and r["ci_hi"] < 0 and r["lower_in_F"]


def test_paired_ligand_contrast(tmp_path):
    """ablations.contrast on fig4_ligand.py grid files: a shift of 0.02 in every query gives 0.02 [0.02, 0.02]."""
    import ablations as AB
    t = _primary(_queries())
    v = np.random.default_rng(2).uniform(0.4, 0.8, len(t))
    for run, add in [("main_s0", 0.0), ("abl_x", -0.02)]:
        (tmp_path / run).mkdir()
        pd.DataFrame(dict(query_key=t.query_key, protein=t.entry.str.lower(), has_query_fp=True, n_pos=5, n_db=100,
                          lenseek_raw__auc=v + add)).to_csv(tmp_path / run / "ligand_grid_per_query.csv", index=False)
    AB.PATHS.update(fig=str(tmp_path), out=str(tmp_path))
    c = AB.contrast("lig_auc", [("main_s0", 1), ("abl_x", -1)])
    assert c["value"] == pytest.approx(0.02) and c["ci_lo"] == pytest.approx(0.02) and c["ci_hi"] == pytest.approx(0.02)
    assert c["n_queries"] == len(t) and not c["missing"]
    assert AB.NAMED["D_E2"][:2] == ([("main_s0", 1), ("plm_esm2", -1)], "language_model")


def test_paired_db_entries_contrast(tmp_path, capsys):
    """ablations.py table, the ligand ROC-AUC among the database sites of an entry list (fig4_ligand.py grid
    --db-entries): original model minus a run 0.02 lower in every query is 0.02 [0.02, 0.02]; both levels are printed."""
    import ablations as AB
    t = _primary(_queries())
    v = np.random.default_rng(4).uniform(0.4, 0.8, len(t))
    col = "lenseek_raw__auc_db_entries"
    for run, add in [("main_s0", 0.0), ("strict30_s0", -0.02)]:
        (tmp_path / run).mkdir()
        pd.DataFrame(dict(query_key=t.query_key, protein=t.entry.str.lower(), has_query_fp=True, n_pos=5, n_db=100,
                          lenseek_raw__auc=v, **{col: v + add})).to_csv(tmp_path / run / "ligand_grid_per_query.csv",
                                                                        index=False)
        pd.DataFrame([dict(stratum="all", column=col, entry_mean=0.6 + add)]).to_csv(
            tmp_path / run / "ligand_grid_summary.csv", index=False)
    pd.DataFrame(dict(pdb=t.entry.str.lower().unique(), stratum_either50="<30")).to_csv(
        tmp_path / "strata.tsv", sep="\t", index=False)
    AB.PATHS.update(fig=str(tmp_path), out=str(tmp_path), strata=str(tmp_path / "strata.tsv"))
    AB._ID.clear()
    f = AB.build_frame(["main_s0", "strict30_s0"])
    x = f[(f.run == "strict30_s0") & (f.metric == "raw_auc_db_entries") & (f.stratum == "all")].iloc[0]
    assert x.value == pytest.approx(0.02) and x.ci_lo == pytest.approx(0.02) and x.ci_hi == pytest.approx(0.02)
    assert x.n_queries == len(t)
    AB._ID.clear()
    AB.db_entries_lines(f)
    assert "strict30_s0 0.580, main_s0 0.600; main_s0 - strict30_s0 +0.020 [0.020, 0.020]" in capsys.readouterr().out


def test_ligand_grid_db_entries(monkeypatch):
    """fig4_ligand.py grid on one synthetic query: --db-entries ranks the listed database sites alone."""
    import fig4_ligand as F4
    from common import mw_auc

    class Homology:
        def identity(self, qi):
            return np.zeros(n)

        def unrelated(self, qi, ident):
            return np.ones(n, bool), True
    rng = np.random.default_rng(3)
    n = 40
    d = rng.uniform(10, 60, n)
    dice = np.array([0.9, 0.5, 0.47, 0.1, 0.2, 0.3])          # query ligand against the six fingerprinted codes
    db_col = rng.integers(0, 6, n)
    entries = np.arange(n) % 3 == 0
    monkeypatch.setattr(F4, "G", dict(
        extra={}, q_code=np.array(["AAA"], dtype=object), q_keys=["1abc/ligand_pocket_1_A.pdb"], q_size=np.array([20]),
        db_size=np.full(n, 20), q_amb=np.array([0]), dice={"AAA": dice}, ls=d[None], dt=d[None],
        has=np.ones(n, bool), db_col=db_col, db_code=np.array(["X"] * n, dtype=object), H=Homology(),
        w_dup=np.ones(n), amb=np.zeros(n, bool), dt_ok=np.ones(n, bool), entries=entries))
    r = F4.per_query(0)
    lab = dice[db_col] >= 0.46
    assert r["db_entries_n_db"] == entries.sum() and r["db_entries_n_pos"] == lab[entries].sum()
    assert r["lenseek_raw__auc_db_entries"] == pytest.approx(mw_auc(-d[entries], lab[entries]))


def test_rotation_summary(tmp_path):
    """model_figures.py rotation: searches whose top-1 result is that of the unrotated site, median and maximum
    Chamfer distance to the unrotated encoding."""
    pd.DataFrame(dict(site=np.repeat(["1abc/ligand_pocket_1_A.pdb", "2xyz/ligand_pocket_1_A.pdb"], 3),
                      rotation=[0, 1, 2] * 2, chamfer_to_unrotated=[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-4], top1=3,
                      rank_of_top1=[1, 1, 1, 1, 2, 1], chamfer_to_top1=1.0)).to_csv(tmp_path / "rotation.tsv",
                                                                                     sep="\t", index=False)
    subprocess.run([sys.executable, str(A / "model_figures.py"), "rotation", "--table", str(tmp_path / "rotation.tsv")],
                   check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=str(ROOT),
                                                                   LENSEEK_OUT=str(tmp_path / "out")))
    j = json.load(open(tmp_path / "out" / "model_figures" / "rotation.json"))
    assert j == dict(sites=2, rotations=3, searches=6, top1_unchanged=5, chamfer_median=pytest.approx(3.5e-5),
                     chamfer_max=6e-4)


def test_deeplytough_assemble_quartiles(tmp_path):
    """scripts/deeplytough.py assemble: descriptors in the evaluation order, NaN for a site without one, and the
    quartiles of the query-database distances in dist_quartiles.txt."""
    ev, rd = tmp_path / "ev", tmp_path / "runs"
    ev.mkdir()
    rd.mkdir()
    (ev / "q_paths.txt").write_text("\n".join(f"/q/1abc/ligand_pocket_{i}_A.pdb" for i in (1, 2)) + "\n")
    (ev / "db_paths.txt").write_text("\n".join(f"/db/{p}/ligand_pocket_1_A.pdb" for p in ("2aaa", "2bbb", "2ccc",
                                                                                           "2ddd")) + "\n")
    x = np.random.default_rng(5).normal(size=(6, 128))
    np.savez(rd / "desc_t.npz", set=np.array(["q", "q", "db", "db", "db", "db"]), idx=np.array([0, 1, 0, 1, 2, 3]),
             desc=(x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32),
             ok=np.array([1, 1, 1, 1, 0, 1], bool))
    script("deeplytough.py", "assemble", rd, "--eval-dir", ev, "--out", tmp_path / "out", folder=S)
    D = np.load(tmp_path / "out" / "dist.npy")
    assert D.shape == (2, 4) and np.isnan(D[:, 2]).all() and np.isfinite(D[:, [0, 1, 3]]).all()
    line = open(tmp_path / "out" / "dist_quartiles.txt").read()
    assert line.startswith("distance quartiles over 6 query-database pairs: ")
    q = np.percentile(D[np.isfinite(D)].astype(np.float64), [25, 50, 75])
    assert [float(v) for v in line.split(":")[1].split()] == pytest.approx(q, abs=5e-5)


def test_out_of_fold_top1_of_table1(tmp_path):
    """ablations.py reads table1_concordance.py's top1_oof.csv and refuses one whose top-1 MCC disagrees with
    table1A.csv."""
    import ablations as AB
    d = tmp_path / "abl_x"
    d.mkdir()
    keys = _primary(_queries()).query_key.tolist()[:40]
    truth = np.tile([1.0, 0.0, 1.0, np.nan], 10)
    pred = np.tile([True, False, False, True], 10)
    pd.DataFrame(dict(q_idx=range(40), query_key=keys, protein=[k.split("/")[0] for k in keys], cluster="c001",
                      cd=1.0, truth=truth, fold=np.arange(40) % 5, fold_threshold=1.0, pred=pred)).to_csv(
        d / "top1_oof.csv", index=False)
    tp, fp, fn, tn = 10, 0, 10, 10
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    row = dict(method="LEN-Seek", k=1, MCC=mcc, MCC_ci_lo=0.1, MCC_ci_hi=0.9)
    pd.DataFrame([row, dict(row, k=5)]).to_csv(d / "table1A.csv", index=False)
    AB.PATHS.update(t1=str(tmp_path))
    top1 = AB.oof_top1("abl_x")
    assert len(top1) == 30 and list(top1.columns) == ["query_key", "protein", "cluster", "fold", "truth", "pred"]
    assert AB.levels("abl_x")["mcc_top1"] == pytest.approx(mcc)
    pd.DataFrame([dict(row, MCC=0.25)]).to_csv(d / "table1A.csv", index=False)
    with pytest.raises(SystemExit):
        AB.oof_top1("abl_x")


def test_glosa_auc_on_the_common_pair_set(monkeypatch):
    """baselines.py compare, G-LoSA labels: the database sites every method can score with a G-LoSA score (here a
    site without a DeeplyTough descriptor and a failed G-LoSA pair are left out); GA >= 0.59 in double precision, so a
    stored float32 0.59 is negative."""
    import common

    class Ranking:
        def ranking(self, qi, names):
            return dict(zip(names, map(np.array, ([.1, .3, .7, .2, .4, .5], [.5] * 6))))
    monkeypatch.setattr(common, "G", dict(
        status=np.ones((1, 6), np.uint8), raw=np.array([[10., 0, 0, 10, 0, 0]]), gst=np.array([[1, 1, 1, 1, 1, 3]]),
        ga=np.array([[.8, .59, .2, .7, .1, .9]], np.float32), q_keys=["1abc/ligand_pocket_1_A.pdb"], n_db=6,
        dbidx=np.arange(6), ls={"lenseek_main_s0": np.array([[1., 2, 3, 4, 5, 6]])}, bl=Ranking(),
        dt=np.array([[.1, .2, np.nan, .3, .5, .6]])))
    r, _ = common.probis_query(0)
    assert r["glosa"] == dict(n_db=4, n_pos=2, lenseek_main_s0__auc=0.75, deeplytough__auc=0.75, seqid__auc=0.0,
                              random__auc=0.5)
    assert r["excluded"] == "" and r["n_db"] == 5 and r["n_pos"] == 2


# --------------------------------------------------------------------------- common.py, run.py, script imports
def test_sequence_clusters():
    import common
    c = pd.read_csv(QS / "seq_clusters_50.tsv", sep="\t", comment="#", dtype=str)
    k = c.query_key.iloc[0]
    assert common.cluster_of(k) == c.cluster.iloc[0]
    one = c.groupby("pdb").cluster.nunique()
    pdb = one[one == 1].index[0]
    assert common.cluster_of(pdb.upper()) == c[c.pdb == pdb].cluster.iloc[0]
    assert common.protein_of("7QVA/ligand_pocket_1_A.pdb") == "7qva"
    assert common.site_key("/x/y/7qva/ligand_pocket_1_A.pdb") == "7qva/ligand_pocket_1_A.pdb"
    assert common.site_key is common.pipeline.site_key and common.tie_key is common.pipeline.tie_key   # one definition
    with pytest.raises(SystemExit):
        common.cluster_of("0zzz/ligand_pocket_1_A.pdb")


def test_threshold_grid():
    """make_grid: the multiples of the step as the nearest doubles; a negated score (Folddisco) gets the distinct
    lattice values of its top-1 d and one step below, so a site is called similar at its own score."""
    import common
    assert common.make_grid(np.array([0.00031]), 0.0001).tolist() == [0.0, 0.0001, 0.0002, 0.0003, 0.0004]
    assert common.make_grid(np.array([0.0003, 0.089]), 0.001)[89] == 0.089
    d1 = np.array([-(12.3456 + 1e-7 * 3), -(5.0001 + 1e-7), -(12.3456 + 1e-7 * 7), 1000.0])
    g = common.make_grid(d1, 0.0001)
    assert g.tolist() == [-12.3457, -12.3456, -5.0001, 1000.0]
    assert (d1[:3, None] <= g[None, :]).argmax(1).tolist() == [1, 2, 1]


def test_cluster_bootstrap_weights():
    """One resample-count matrix for every cluster bootstrap: each row draws as many clusters as there are."""
    import common
    cl, W = common.boot_weights(["c2", "c1", "c2", "c3"], n_boot=50, seed=0)
    assert cl == ["c1", "c2", "c3"] and W.shape == (50, 3) and (W.sum(1) == 3).all()
    assert np.array_equal(W, common.boot_weights(["c3", "c1", "c2"], n_boot=50, seed=0)[1])


def test_run_lists_every_result():
    p = subprocess.run([sys.executable, str(A / "run.py"), "--list"], capture_output=True, text=True, check=True)
    names = [l.split()[0] for l in p.stdout.splitlines()]
    assert {"table1", "fig3", "fig4", "table2", "table3", "figS1", "figS2", "figS3", "cases", "ablations",
            "filter-retrain", "holdout-table", "derivation"} <= set(names)
    assert len(names) == len(set(names))
    import run
    for name, _, steps, *_ in run.RESULTS:
        for s, args in ([] if callable(steps) else steps):
            assert (A / s).exists(), (name, s)


@pytest.mark.parametrize("name", sorted(p.name for p in A.glob("*.py") if p.name != "common.py"))
def test_script_imports(name):
    """Every analysis script loads with its helpers from analysis/common.py."""
    script(name, "--help")


def test_holdout_script_imports():
    script("holdout.py", "--help", folder=S)


# --------------------------------------------------------------------------- strata and later PDB entries
def test_identity_strata(tmp_path):
    """scripts/site_annotation.py identity-strata: best hit with query or target coverage >= 0.5 (fident, then bit
    score), binned; a protein without such a hit is < 30 %."""
    for p in ("1abc", "2abc", "3abc"):
        (tmp_path / "q" / p).mkdir(parents=True)
    hits = [("1abc_A", "9xyz_A", 0.82, 0.9, 0.2, 300), ("1abc_B", "9xyz_B", 0.82, 0.3, 0.6, 310),
            ("1abc_A", "8xyz_A", 0.95, 0.4, 0.4, 500), ("2abc_A", "9xyz_A", 0.35, 0.6, 0.6, 90)]
    (tmp_path / "m8").write_text("".join(f"{q}\t{t}\t{f}\t100\t120\t130\t{qc}\t{tc}\t1e-5\t{b}\n"
                                         for q, t, f, qc, tc, b in hits))
    subprocess.run([sys.executable, str(S / "site_annotation.py"), "identity-strata", "--m8", str(tmp_path / "m8"),
                    "--query-root", str(tmp_path / "q")], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, PYTHONPATH=str(ROOT), LENSEEK_OUT=str(tmp_path / "out")))
    t = pd.read_csv(tmp_path / "out" / "site_annotation" / "protein_strata.tsv", sep="\t", dtype=str,
                    keep_default_na=False)
    assert t.pdb.tolist() == ["1abc", "2abc", "3abc"] and t.stratum_either50.tolist() == ["80-95", "30-40", "<30"]
    assert t.besthit_either50.tolist() == ["1abc_B>9xyz_B", "2abc_A>9xyz_A", ""]


def test_holdout_entries_to_build(tmp_path):
    """scripts/holdout.py select_entries: X-ray diffraction only, resolution <= 3.0 A at two decimals; for the binding
    sites from later PDB entries also a protein entity and one-character chain IDs."""
    pytest.importorskip("gemmi")
    pytest.importorskip("rdkit")
    sys.path.insert(0, str(S))
    import holdout

    def entry(i, method=("X-RAY DIFFRACTION",), res=2.0, protein=1, chains="A,B"):
        return dict(rcsb_id=i, exptl=[dict(method=m) for m in method],
                    rcsb_entry_info=dict(resolution_combined=[res] if res else None,
                                         polymer_entity_count_protein=protein),
                    polymer_entities=[dict(entity_poly=dict(pdbx_strand_id=chains))])
    meta = [entry("1AAA"), entry("1BBB", res=3.004), entry("1CCC", res=3.006),
            entry("1DDD", method=("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION")), entry("1EEE", protein=0),
            entry("1FFF", chains="A,AA"), entry("1GGG", res=None), entry("1HHH", method=("ELECTRON MICROSCOPY",))]
    (tmp_path / "meta.jsonl").write_text("".join(json.dumps(e) + "\n" for e in meta))
    for validation, want in [(False, ["1aaa", "1bbb"]), (True, ["1aaa", "1bbb", "1eee", "1fff"])]:
        holdout.select_entries(tmp_path / "meta.jsonl", tmp_path / "ids.txt", validation)
        assert open(tmp_path / "ids.txt").read().split() == want


# --------------------------------------------------------------------------- scripts/holdout.py counts and validate
def _tsv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def _counts(path):
    return {k: v for k, v in (l.rstrip("\n").split("\t") for l in list(open(path))[1:])}


def test_holdout_counts(tmp_path):
    """scripts/holdout.py counts on small construction outputs in the --out layout."""
    _tsv(tmp_path / "build" / "entries.tsv", ["pdbid", "status", "n_sites"],
         [("1abc", "ok", 2), ("2abc", "ok", 0), ("3abc", "multichar_chain", 0), ("4abc", "ok", 1), ("5abc", "ok", 1)])
    _tsv(tmp_path / "sel" / "manifest_all.tsv",
         ["site_id", "pdbid", "cluster80", "stratum", "dedup_representative", "selected"],
         [("1abc/ligand_pocket_1_A", "1abc", "C00000", "lt30", 1, 1),
          ("1abc/ligand_pocket_2_A", "1abc", "C00000", "lt30", 0, 0),
          ("4abc/ligand_pocket_1_B", "4abc", "C00001", "ge80", 1, 1),
          ("5abc/ligand_pocket_1_A", "5abc", "C00001", "30to80", 1, 0)])
    _tsv(tmp_path / "manifest.tsv", ["site_id", "stratum", "featurised"],
         [("1abc/ligand_pocket_1_A", "lt30", 0), ("4abc/ligand_pocket_1_B", "ge80", 1)])
    _tsv(tmp_path / "strata" / "structure.tsv", ["site_id", "structure_stratum"],
         [("1abc/ligand_pocket_1_A", "lt0.5"), ("4abc/ligand_pocket_1_B", "ge0.8")])
    _tsv(tmp_path / "strata" / "ligand.tsv", ["site_id", "ligand_stratum"],
         [("1abc/ligand_pocket_1_A", "new_ccd_novel"), ("4abc/ligand_pocket_1_B", "same_ccd")])
    script("holdout.py", "counts", "--out", tmp_path, folder=S)
    c = {k: int(v) for k, v in _counts(tmp_path / "counts.tsv").items()}
    assert c["entries built"] == 4 and c["valid sites"] == 4 and c["valid sites: entries"] == 3
    assert c["de-duplicated sites"] == 3 and c["de-duplicated sites: 80 % clusters"] == 2
    assert [c[f"de-duplicated sites: identity {s}"] for s in ("lt30", "30to80", "ge80")] == [1, 1, 1]
    assert c["sampled sites"] == 2 and c["sampled sites: entries"] == 2 and c["sampled sites: 80 % clusters"] == 2
    assert c["sampled sites: max TM-score lt0.5"] == 1 and c["sampled sites: max TM-score 0.5to0.8"] == 0
    assert c["sampled sites: ligand new_ccd_novel"] == 1 and c["sampled sites: ligand no_fingerprint"] == 0
    assert c["featurised sites"] == 1 and c["featurised sites: identity ge80"] == 1


def _example_cif(entry, out):
    """mmCIF of an example entry: protein.pdb, its ligand with the residue number of the ligand file, a copy of the
    ligand 60 A away (residue 900, no contact) and a water."""
    import gzip

    import gemmi
    q = EX / "queries" / entry
    st = gemmi.read_structure(str(q / "protein.pdb"))
    lig = gemmi.read_structure(str(next(q.glob("ligand_[0-9]*.pdb"))))
    ch = gemmi.Chain("L")
    for shift in (0.0, 60.0):
        for res in lig[0][0]:
            r = res.clone()
            r.het_flag = "H"
            if shift:
                r.seqid = gemmi.SeqId(900, " ")
            for a in r:
                a.pos = gemmi.Position(a.pos.x + shift, a.pos.y, a.pos.z)
            ch.add_residue(r)
    w = gemmi.Residue()
    w.name, w.het_flag, w.seqid = "HOH", "H", gemmi.SeqId(950, " ")
    at = gemmi.Atom()
    at.name, at.element, at.pos = "O", gemmi.Element("O"), gemmi.Position(0, 0, 0)
    w.add_atom(at)
    ch.add_residue(w)
    st[0].add_chain(ch)
    st.setup_entities()
    out.mkdir(exist_ok=True)
    with gzip.open(out / f"{entry}.cif.gz", "wt") as f:
        f.write(st.make_mmcif_document().as_string())


def test_holdout_validate(tmp_path):
    """scripts/holdout.py validate on an example entry rebuilt from its files: the site is recovered with the same
    binding-site residues from the BsitePDB copy of its ligand and binding-site files (coordinates one column to the
    left), the far copy is not a match, a missing file is counted, and the exclusion list E removes the site."""
    pytest.importorskip("gemmi")
    pytest.importorskip("rdkit")
    _example_cif("7uad", tmp_path / "cif")
    (tmp_path / "bsdb" / "7uad").mkdir(parents=True)
    for f in (EX / "queries" / "7uad").glob("ligand_*.pdb"):
        lines = [l[:26] + l[27:] if l.startswith(("ATOM", "HETATM")) else l for l in open(f)]
        (tmp_path / "bsdb" / "7uad" / f.name).write_text("".join(lines))
    (tmp_path / "ids.txt").write_text("7uad\n9zzz\n")
    args = ["--ids", tmp_path / "ids.txt", "--cifdir", tmp_path / "cif", "--bsdb-root", tmp_path / "bsdb",
            "--nproc", "1"]
    script("holdout.py", "validate", "--out", tmp_path / "a", *args, folder=S)
    s = _counts(tmp_path / "a" / "validate" / "summary.tsv")
    assert int(s["entries"]) == 2 and int(s["entries without a file or failing to build"]) == 1
    assert int(s["BsitePDB sites"]) == 1 and int(s["recovered by the build rules"]) == 1
    assert float(s["recall"]) == 1.0 and float(s["mean Jaccard index of the recovered sites"]) == 1.0
    rows = pd.read_csv(tmp_path / "a" / "validate" / "sites.tsv", sep="\t", dtype=str, keep_default_na=False)
    far = rows[rows.resi == "900"]
    assert len(far) == 1 and far.reason.iloc[0] == "no_contact" and far.in_bsdb.iloc[0] == "0"
    code = rows[rows.reason == "ok"].resn.iloc[0]
    (tmp_path / "excl.txt").write_text(f"# test list\n{code} 9\n")
    script("holdout.py", "validate", "--out", tmp_path / "b", "--exclusion", tmp_path / "excl.txt", *args, folder=S)
    s = _counts(tmp_path / "b" / "validate" / "summary.tsv")
    assert int(s["BsitePDB sites"]) == 1 and int(s["recovered by the build rules"]) == 0
