#!/usr/bin/env python
"""Run the analysis behind a reported result by its name.

    python analysis/run.py --list                  # every result and the manuscript item it produces
    python analysis/run.py table1 [extra args]     # the script and subcommand(s) of one result
    python analysis/run.py --all                   # every result whose inputs are present, in dependency order
    python analysis/run.py --print table1          # the commands of a result and its missing inputs, nothing run

A result runs one or more steps of the scripts in analysis/, which turn intermediate files into the reported numbers
and figures; the intermediate files are written by the scripts in scripts/ (examples/*.sh). Default arguments come
from the environment variables below (the scripts read most of them themselves; run.py fills in the rest). Extra
arguments are appended to every step of the result. A result whose inputs are missing (a variable that is not set, a
path that does not exist, an output of another result, an executable that is not found) is skipped with the list of
what is missing; run by name, it then exits with status 3. --print shows the commands in any case, with the missing
inputs above them. --all ends with a table of the results that ran, were skipped (and why) or failed.

Environment:
  LENSEEK_OUT              outputs and the intermediate files of scripts/ (default ./out)
  LENSEEK_EVAL_DIR         evaluation outputs of LEN-Seek (asset eval_main_s0; scripts/evaluate.py)
  LENSEEK_EVAL_ROOT        folders out_<run>/ of the variants, the seeds and the model trained without the ligand
                           filters (scripts/evaluate.py); a run whose folder is not there is read from the asset that
                           holds it (EVAL_FROM_ASSETS: the two one-hot models of Supplementary Table S15)
  LENSEEK_PAIRS            scored top-100 pairs of LEN-Seek (scripts/labels.py tables)
  LENSEEK_SCAN_DIR, LENSEEK_GLOSA_SCAN_DIR, LENSEEK_PAIRS_DIR   database-wide ProBiS and G-LoSA scans and their site
                           lists (scripts/labels.py)
  LENSEEK_STRAT_DIR        MMseqs2 searches of the evaluation chains against the training chains (asset eval_extras)
  LENSEEK_REFERENCE_DIR    CCD SMILES and component tables, SIFTS and Pfam tables, database ligand codes (asset
                           reference_data)
  LENSEEK_QUERY_ROOT, LENSEEK_BSDB_ROOT   query and database structures (BsitePDB layout)
  LENSEEK_DT_DIR           DeeplyTough outputs (scripts/deeplytough.py assemble)
  LENSEEK_FOLDDISCO_DIR    Folddisco runs prefilter/, matching/, blosum62/, entries/ (scripts/folddisco.py assemble;
                           default $LENSEEK_OUT/folddisco)
  LENSEEK_BASELINE_DIR     sequence-identity and random baselines: top100/ of scripts/baseline_features.py (default
                           $LENSEEK_OUT/baselines)
  LENSEEK_NA_EVAL_DIR, LENSEEK_NA_PAIRS   evaluation outputs and scored pairs of the Non-adapted Model
  LENSEEK_CKPT, LENSEEK_RECON_DATA        LEN-Seek checkpoint and reconstruction set (figS3)
  LENSEEK_TIMING_CPU, LENSEEK_TIMING_GPU, LENSEEK_TIMING_DT   timing runs of Table 2 and Supplementary Table S17 when
                           they are not in $LENSEEK_OUT/timing (scripts/timing.py; the CPU folder holds the Folddisco
                           timing too)
  LENSEEK_FILTER_SETS      query sets of the retraining readout (default $LENSEEK_OUT/filter_retrain/sets)
  LENSEEK_SCALING_DIR, LENSEEK_SCALING_TIME_DIR, LENSEEK_SINGLE_VECTOR_DIR   outputs of scripts/scaling.py (accuracy,
                           timing) and scripts/single_vector.py (defaults $LENSEEK_OUT/scaling, the same,
                           $LENSEEK_OUT/single_vector); LENSEEK_SCALING_ADDED  encodings of the sites removed by the
                           ligand filters (table3, real150k; scripts/evaluate.py --added, default
                           $LENSEEK_OUT/scaling/added)
  LENSEEK_HOLDOUT_RUN, LENSEEK_HOLDOUT_LABELS, LENSEEK_HOLDOUT_THRESHOLDS, LENSEEK_HOLDOUT_MANIFEST   binding sites
                           from later PDB entries: the one scoring run (scripts/holdout.py score), its ProBiS results
                           (scripts/labels.py tables), the thresholds (holdout_score.py thresholds) and the manifest of
                           the sites (defaults below $LENSEEK_OUT/holdout and $LENSEEK_OUT/holdout_score); read only:
                           the metrics are recomputed
  FOLDSEEK                 Foldseek executable (cases)
A variable that is not set points to the released asset that holds its input once `lenseek-fetch <asset>` has unpacked
it below $LENSEEK_HOME/assets/ (FROM_ASSETS below; lenseek/configs/assets.yaml): the evaluation outputs, reference
tables and strata searches of the release, and the labels, comparison-method outputs, timing logs, scoring outputs
of the binding sites from later PDB entries, derived tables and BsitePDB structures of the data records. The
intermediate files read from $LENSEEK_OUT (below) that are missing there are copied from the derived tables
(OUT_FROM_DERIVED; the degeneracy checks of the two one-hot models from asset eval_onehot_2x2), file by file, so that
nothing is written into the asset.
Intermediate files read from $LENSEEK_OUT: cache/site_annot/ (scripts/site_annotation.py), site_annotation/
(protein_strata.tsv and query_strata.csv of scripts/site_annotation.py identity-strata and strata), cache/pfam/ (SIFTS
and Pfam tables when $LENSEEK_REFERENCE_DIR/pfam/ is absent), database_scan/ (scripts/database_scan.py), timing/
(scripts/timing.py), filter_free_data/identity/ (scripts/filter_free_data.py identity), labels/null/ (scripts/labels.py
null).
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROM_ASSETS = {                                     # variable: (asset, path inside the unpacked asset)
    "LENSEEK_EVAL_DIR": ("eval_main_s0", ""),
    "LENSEEK_STRAT_DIR": ("eval_extras", ""),
    "LENSEEK_RECON_DATA": ("eval_extras", "reconstruction_set.pkl"),
    "LENSEEK_CKPT": ("checkpoint_main_s0", ""),
    "LENSEEK_BSDB_ROOT": ("bsitepdb", "BSDB_231103"),
    "LENSEEK_EVAL_PERIOD_ROOT": ("bsitepdb", "BSDB_EX"),
    "LENSEEK_QUERY_ROOT": ("eval_queries", ""),
    "LENSEEK_REFERENCE_DIR": ("reference_data", ""),
    "LENSEEK_PAIRS": ("labels_probis_glosa", "pairs_main_s0.scored.tsv"),
    "LENSEEK_SCAN_DIR": ("labels_probis_glosa", "scan_probis"),
    "LENSEEK_GLOSA_SCAN_DIR": ("labels_probis_glosa", "scan_glosa"),
    "LENSEEK_PAIRS_DIR": ("labels_probis_glosa", "pairs"),
    "LENSEEK_DT_DIR": ("comparison_methods", "deeplytough"),
    "LENSEEK_FOLDDISCO_DIR": ("comparison_methods", "folddisco"),
    "LENSEEK_BASELINE_DIR": ("comparison_methods", "baselines"),
    "LENSEEK_TIMING_CPU": ("timing_logs", "cpu"),
    "LENSEEK_TIMING_GPU": ("timing_logs", "gpu"),
    "LENSEEK_TIMING_DT": ("timing_logs", "dt"),
    "LENSEEK_HOLDOUT_RUN": ("holdout_outputs", "run"),
    "LENSEEK_HOLDOUT_LABELS": ("holdout_outputs", "labels.scored.tsv"),
    "LENSEEK_HOLDOUT_THRESHOLDS": ("holdout_outputs", "thresholds"),
    "LENSEEK_HOLDOUT_MANIFEST": ("holdout_outputs", "manifest.tsv"),
    "LENSEEK_EVAL_ROOT": ("derived_tables", "eval"),
    "LENSEEK_NA_EVAL_DIR": ("derived_tables", "eval/out_abl_nonadapted"),
    "LENSEEK_NA_PAIRS": ("derived_tables", "pairs_abl_nonadapted.scored.tsv"),
    "LENSEEK_FILTER_SETS": ("derived_tables", "filter_retrain/sets"),
    "LENSEEK_SCALING_DIR": ("derived_tables", "scaling"),
    "LENSEEK_SCALING_ADDED": ("derived_tables", "scaling/added"),
    "LENSEEK_SINGLE_VECTOR_DIR": ("derived_tables", "single_vector"),
}
OUT_FROM_DERIVED = ("cache/site_annot", "site_annotation", "database_scan", "timing/dt_prep.csv", "labels/null",
                    "holdout/mm", "holdout/seqs", "ablations")   # paths below $LENSEEK_OUT and the derived tables
# the evaluation outputs (eval/out_<run>/) and degeneracy checks (ablations/<run>/) of the two one-hot models of
# Supplementary Table S15 are in asset eval_onehot_2x2
EVAL_FROM_ASSETS = {"abl_onehot_na": "eval_onehot_2x2", "abl_onehot_fullgraph_na": "eval_onehot_2x2"}
OUT_FROM_ONEHOT = ("ablations",)                    # paths below $LENSEEK_OUT and that asset
EVAL_DIRS = {}                                      # run: its out_<run>/ in the fetched asset


def assets_to_env():
    """Set every variable of FROM_ASSETS that is not set to its fetched asset; the steps run here inherit it."""
    try:
        from lenseek.assets import home, registry
    except ImportError:                             # not installed: the package of this source tree
        sys.path.insert(0, os.path.dirname(HERE))
        try:
            from lenseek.assets import home, registry
        except ImportError:
            return
    reg = registry()

    def unpacked(asset, sub=""):
        entry = reg.get(asset) or {}
        if entry.get("kind") == "file":                     # a single file, e.g. a checkpoint
            path = os.path.join(home(), "assets", entry.get("file") or "")
            return path if entry.get("file") and os.path.isfile(path) else None
        d = os.path.join(home(), "assets", asset)
        path = os.path.normpath(os.path.join(d, entry.get("root") or "", sub))
        return path if os.path.exists(os.path.join(d, ".unpacked")) and os.path.exists(path) else None

    for v, (asset, sub) in FROM_ASSETS.items():
        path = None if os.environ.get(v) else unpacked(asset, sub)
        if path:
            os.environ[v] = path
    for run, asset in EVAL_FROM_ASSETS.items():
        path = unpacked(asset, os.path.join("eval", f"out_{run}"))
        if path:
            EVAL_DIRS[run] = path
    for asset, rels in (("derived_tables", OUT_FROM_DERIVED), ("eval_onehot_2x2", OUT_FROM_ONEHOT)):
        root, copied = unpacked(asset), 0
        for rel in rels if root else ():
            src = os.path.join(root, rel)
            files = [src] if os.path.isfile(src) else [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs]
            for f in files:
                dst = os.path.join(os.path.abspath(os.environ.get("LENSEEK_OUT") or "out"), os.path.relpath(f, root))
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(f, dst)
                    copied += 1
        if copied:
            print(f"[run] copied {copied} intermediate files of asset {asset} into $LENSEEK_OUT", file=sys.stderr)


assets_to_env()
OUT = os.path.abspath(os.environ.get("LENSEEK_OUT") or "out")
ENV = os.environ.get
BASELINE = ENV("LENSEEK_BASELINE_DIR") or os.path.join(OUT, "baselines")    # input top100/
REFERENCE_DIR = ENV("LENSEEK_REFERENCE_DIR") or ""
SKIPPED = 3


def out(*p):
    return os.path.join(OUT, *p)


def var(name):
    """The value of an environment variable, or the text $name when it is not set (shown by --print)."""
    return ENV(name) or f"${name}"


def top100(name):
    return os.path.join(BASELINE, "top100", f"{name}_top100.tsv")


def methods(all_baselines=True):
    m = ["--method", "DeeplyTough", "0.0001", os.path.join(var("LENSEEK_DT_DIR"), "dist.npy")]
    if all_baselines:
        m += ["--method", "Sequence-identity baseline", "0.001", top100("seqid"),
              "--method", "Random ranking", "0.001", top100("random")]
    return m


def grid(run):
    return out("fig4", run, "ligand_grid_per_query.csv")


def eval_dir(run):
    """out_<run>/ of $LENSEEK_EVAL_ROOT, else of the fetched asset that holds it (EVAL_FROM_ASSETS)."""
    d = os.path.join(var("LENSEEK_EVAL_ROOT"), f"out_{run}")
    return EVAL_DIRS[run] if run in EVAL_DIRS and not os.path.exists(os.path.join(d, "dist.npy")) else d


STRICT30_EXCLUDED = ["--db-entries", os.path.join(os.path.dirname(HERE), "data", "strict30_exclude_entries.txt")]


FOLDDISCO = ENV("LENSEEK_FOLDDISCO_DIR") or out("folddisco")
FD_RUNS = {"prefilter": "folddisco", "matching": "matching", "blosum62": "blosum62", "entries": "entries"}
FD_TOP100 = os.path.join(FOLDDISCO, "prefilter", "top100_pairs.tsv")
FD_LIST = ["--method", "Folddisco", "0.0001", FD_TOP100]           # table1_concordance.py: the primary run
FD_DIRS = sum([["--method", f"{name}={os.path.join(FOLDDISCO, r)}"] for r, name in FD_RUNS.items()], [])
FD_DIST = [os.path.join(FOLDDISCO, r, "dist.npy") for r in FD_RUNS]


ANNOT = [out("cache", "site_annot", f) for f in ("query_sites.tsv", "db_sites.tsv")]
ID_STRATA = [out("site_annotation", "protein_strata.tsv")]
STRATA = ID_STRATA + [out("site_annotation", "query_strata.csv")]
SEED_DIRS = {s: os.path.join(ENV("LENSEEK_EVAL_ROOT") or "", f"out_{s}") for s in ("main_s1", "main_s2")}
SEEDS = sum([["--seed", f"{s}={d}"] for s, d in SEED_DIRS.items() if os.path.exists(os.path.join(d, "dist.npy"))], [])
PFAM_FILES = ("pdb_chain_pfam.tsv.gz", "Pfam-A.clans.tsv.gz")
PFAM = next(([os.path.join(d, f) for f in PFAM_FILES]
             for d in (os.path.join(REFERENCE_DIR, "pfam"), out("cache", "pfam"))
             if all(os.path.exists(os.path.join(d, f)) for f in PFAM_FILES)),
            [out("cache", "pfam", f) for f in PFAM_FILES])     # the copy of reference_data, else the download
SCAN = [out("database_scan", f) for f in ("scan_db.tsv", "scan_queries.tsv", "eval_period.json")]
TIMING = dict(cpu=ENV("LENSEEK_TIMING_CPU") or out("timing"), gpu=ENV("LENSEEK_TIMING_GPU") or out("timing"),
              dt=ENV("LENSEEK_TIMING_DT") or out("timing"))
HOLDOUT = dict(run=ENV("LENSEEK_HOLDOUT_RUN") or out("holdout", "run"),
               labels=ENV("LENSEEK_HOLDOUT_LABELS") or out("holdout", "labels", "tables",
                                                           "pairs_union_for_labels.scored.tsv"),
               thresholds=ENV("LENSEEK_HOLDOUT_THRESHOLDS") or out("holdout_score", "thresholds"),
               manifest=ENV("LENSEEK_HOLDOUT_MANIFEST") or out("holdout", "manifest.tsv"))
FACTORIAL_DIST = [os.path.join(eval_dir(r), "dist.npy") for r in EVAL_FROM_ASSETS]
FETCH_HINT = {f: EVAL_FROM_ASSETS[r] for r, f in zip(EVAL_FROM_ASSETS, FACTORIAL_DIST)}   # missing file: its asset
NOTES = []


def ablation_steps():
    """Per run with out_<run>/ in $LENSEEK_EVAL_ROOT: latent-collapse check, whole-database ligand metrics,
    out-of-fold concordance in the format of Table 1 (non-collapsed runs); then the sites common to the language-model
    variants, the same concordance of each run restricted to them and the rows of Supplementary Table S14. Outputs that
    exist are reused. The runs of group factorial are those of the factorial result."""
    sys.path.insert(0, HERE)
    from common import RUNS
    root = var("LENSEEK_EVAL_ROOT")
    for run, _, _, group, _ in RUNS:
        if group == "factorial":
            continue
        ev = os.path.join(root, f"out_{run}")
        if not os.path.exists(os.path.join(ev, "dist.npy")):
            NOTES.append(f"ablations: {run} has no {ev}/dist.npy")
            continue
        deg = out("ablations", run, "degeneracy.json")
        if not os.path.exists(deg):
            yield "ablations.py", ["check", "--eval-dir", ev, "--tag", run]
        if not os.path.exists(out("fig4", run, "ligand_grid_summary.csv")):
            yield "fig4_ligand.py", ["grid", "--eval-dir", ev] + (STRICT30_EXCLUDED if run == "strict30_s0" else [])
        collapsed = os.path.exists(deg) and json.load(open(deg))["au"] == 0
        if not collapsed and not os.path.exists(out("table1", run, "top1_oof.csv")):
            yield "table1_concordance.py", ["table1", "--eval-dir", ev, "--out", out("table1", run), "--repeats", "0",
                                            "--strata-methods", "none"]
    yield "ablations.py", ["common-sites", "--make-restricted"]
    for d in sorted(glob.glob(out("ablations", "common_sites_eval", "out_*__common_sites"))):
        run = os.path.basename(d)[len("out_"):]
        if not os.path.exists(out("table1", run, "top1_oof.csv")):
            yield "table1_concordance.py", ["table1", "--eval-dir", d, "--out", out("table1", run), "--repeats", "0",
                                            "--strata-methods", "none"]
    yield "ablations.py", ["table"]


def factorial_steps():
    """Per model of Supplementary Table S15 and per seed of the original model, as in ablation_steps (outputs that
    exist are reused): latent-collapse check (the models), whole-database ligand metrics, out-of-fold concordance in
    the format of Table 1; then the values and differences of the table."""
    sys.path.insert(0, HERE)
    from common import FACTORIAL
    models = [r for r, *_ in FACTORIAL]
    for run in models + ["main_s0", "main_s1", "main_s2"]:
        ev = eval_dir(run)
        if run in models and not os.path.exists(out("ablations", run, "degeneracy.json")):
            yield "ablations.py", ["check", "--eval-dir", ev, "--tag", run]
        if not os.path.exists(out("fig4", run, "ligand_grid_summary.csv")):
            yield "fig4_ligand.py", ["grid", "--eval-dir", ev]
        if not os.path.exists(out("table1", run, "top1_oof.csv")):
            yield "table1_concordance.py", ["table1", "--eval-dir", ev, "--out", out("table1", run), "--repeats", "0",
                                            "--strata-methods", "none"]
    yield "factorial.py", sum([["--model", f"{r}={eval_dir(r)}"] for r in models], []) + \
        sum([["--seed", f"{s}={eval_dir(s)}"] for s in ("main_s1", "main_s2")], []) + FD_DIRS


def filter_steps():
    """Filtered-ligand queries against the rest for LEN-Seek and the model trained without the filters, and the
    difference in differences against the three seeds of the original model (whole-database ligand ROC-AUC per query of
    fig4_ligand.py grid)."""
    yield "filter_retrain.py", ["compare", "--per-query", grid("main_s0"), "--tag", "main_s0"]
    if not os.path.exists(grid("filter_free_s0")):
        NOTES.append("filter-retrain: no ligand metrics of filter_free_s0 (run the ablations result first)")
        return
    yield "filter_retrain.py", ["compare", "--per-query", grid("filter_free_s0"), "--tag", "filter_free_s0"]
    sets = ENV("LENSEEK_FILTER_SETS") or out("filter_retrain", "sets")
    if not os.path.exists(os.path.join(sets, "did_exclude_ge80.tsv")):
        m8 = out("filter_free_data", "identity", "eval_vs_new.m8")
        if not os.path.exists(m8):
            NOTES.append(f"filter-retrain: no query sets in {sets} and no {m8} (scripts/filter_free_data.py identity)")
            return
        yield "filter_retrain.py", ["sets", "--out", sets]
    missing = [s for s in ("main_s0", "main_s1", "main_s2") if not os.path.exists(grid(s))]
    if missing:
        NOTES.append(f"filter-retrain: no ligand metrics of {', '.join(missing)} (run the ablations result first)")
        return
    seeds = sum([["--seed", grid(s)] for s in ("main_s0", "main_s1", "main_s2")], [])
    yield "filter_retrain.py", ["did", "--retrain", grid("filter_free_s0"), "--base", grid("main_s0"),
                                "--sets", sets] + seeds


def baseline_ligand_steps():
    """The whole-database ligand metrics of the other seeds of the original model (fig4_ligand.py grid; outputs that
    exist are reused, as in the ablations), for the seed SD of the detection rule; then the two baselines and the paired ligand
    differences."""
    seeds = []
    for s, ev in SEED_DIRS.items():
        if not os.path.exists(out("fig4", s, "ligand_grid_summary.csv")):
            if not (os.path.exists(os.path.join(ev, "dist.npy")) and ENV("LENSEEK_DT_DIR")):
                NOTES.append(f"baseline-ligand: no ligand metrics of {s} ({ev}/dist.npy and $LENSEEK_DT_DIR needed); "
                             "detection without that seed")
                continue
            yield "fig4_ligand.py", ["grid", "--eval-dir", ev]
        seeds += ["--seed-grid", f"{s}={grid(s)}"]
    yield "baselines.py", ["ligand"] + seeds


def table2_steps():
    yield "table2_cost.py", ["build", "--cpu-dir", TIMING["cpu"], "--gpu-dir", TIMING["gpu"], "--dt-dir", TIMING["dt"],
                             "--dt-prep", out("timing", "dt_prep.csv")]


def scaling_dirs():
    acc = ENV("LENSEEK_SCALING_DIR") or out("scaling")
    return ["--acc", acc, "--time", ENV("LENSEEK_SCALING_TIME_DIR") or acc,
            "--ann", ENV("LENSEEK_SINGLE_VECTOR_DIR") or out("single_vector")]


def holdout_steps():
    mcc = out("holdout_score", "mcc_holdout.json")
    yield "holdout_score.py", ["ligand", "--run", HOLDOUT["run"], "--manifest", HOLDOUT["manifest"],
                               "--db-paths", os.path.join(var("LENSEEK_EVAL_DIR"), "db_paths.txt")]
    yield "holdout_score.py", ["mcc", "--run", HOLDOUT["run"], "--pairs", HOLDOUT["labels"],
                               "--db-paths", os.path.join(var("LENSEEK_EVAL_DIR"), "db_paths.txt"),
                               "--thresholds", HOLDOUT["thresholds"], "--out", mcc]
    yield "holdout_score.py", ["table", "--mcc", mcc, "--manifest", HOLDOUT["manifest"]]


# name, manuscript item, steps (list of (script, args) or a generator function), inputs: environment variables,
# files (intermediate files and outputs of other results), executables (environment variable, default name)
RESULTS = [
    ("derivation", "Supplementary Figure S7, Section S1 and Table S6: derivation of the training data and the "
     "evaluation data, redundancy of the training data, ligands removed by the digit filter, evaluation sites with "
     "filtered ligands",
     [("database_derivation.py", [])], ["LENSEEK_EVAL_DIR", "LENSEEK_REFERENCE_DIR"],
     SCAN + [out("cache", "site_annot", f) for f in ("db_sites.tsv", "db80_cluster.tsv", "db30_cluster.tsv")], []),
    ("table1", "Table 1 (fixed threshold); Supplementary Table S7 A (out-of-fold thresholds); Table S8 A (G-LoSA "
     "labels, with the paired LEN-Seek - Folddisco MCC difference); Table S13 A; evaluation-data rows of Table S10 B; "
     "the 50 shuffled splits of Section S2",
     [("table1_concordance.py", ["table1", "--paired", "Folddisco"] + methods() + FD_LIST)],
     ["LENSEEK_EVAL_DIR", "LENSEEK_SCAN_DIR", "LENSEEK_GLOSA_SCAN_DIR", "LENSEEK_PAIRS_DIR", "LENSEEK_DT_DIR"],
     [top100(b) for b in ("seqid", "random")] + [FD_TOP100] + STRATA, []),
    ("counts", "Supplementary Table S9: search results classified as similar at the fixed thresholds, and their "
     "overlaps",
     [("table1_concordance.py", ["counts"] + methods(False) + FD_LIST)],
     ["LENSEEK_EVAL_DIR", "LENSEEK_SCAN_DIR", "LENSEEK_GLOSA_SCAN_DIR", "LENSEEK_PAIRS_DIR", "LENSEEK_DT_DIR",
      "LENSEEK_REFERENCE_DIR"], [out("table1", "thresholds.csv"), FD_TOP100] + ANNOT, []),
    ("fullscan", "Supplementary Table S8 caption: PR-AUC (and ROC-AUC) of LEN-Seek against ProBiS over the whole "
     "database",
     [("table1_concordance.py", ["fullscan"])], ["LENSEEK_EVAL_DIR", "LENSEEK_SCAN_DIR", "LENSEEK_PAIRS_DIR"], [], []),
    ("unrelated", "Supplementary Table S8 caption: ProBiS and G-LoSA positive labels among unrelated pairs of the "
     "database-wide scans",
     [("table1_concordance.py", ["unrelated"])], ["LENSEEK_SCAN_DIR", "LENSEEK_GLOSA_SCAN_DIR"],
     [out("labels", "null", "null_set.npz")], []),
    ("rotation", "Supplementary Section S10: the 481 evaluation sites rotated three times (top-1 results unchanged, "
     "Chamfer distance to the unrotated encoding)",
     [("model_figures.py", ["rotation"])], ["LENSEEK_EVAL_DIR"],
     [os.path.join(ENV("LENSEEK_EVAL_DIR") or "", "rotation.tsv")], []),
    ("fig3", "Figure 4 (Chamfer distance against ProBiS Z and G-LoSA GA) and Supplementary Figure S4 (MCC against "
     "the threshold)",
     [("score_agreement.py", ["corr", "--name", "fig3"]),
      ("score_agreement.py", ["mcc", "--name", "figS4", "--threshold", "42.5"])],
     ["LENSEEK_EVAL_DIR", "LENSEEK_PAIRS"], [], []),
    ("nonadapted", "Supplementary Figures S5 and S6 and Tables S4 and S5 (Non-adapted Model, Chamfer distance 48.5)",
     [("score_agreement.py", ["corr", "--name", "figS6", "--eval-dir", var("LENSEEK_NA_EVAL_DIR"),
                              "--pairs", var("LENSEEK_NA_PAIRS")]),
      ("score_agreement.py", ["mcc", "--name", "figS5", "--threshold", "48.5",
                              "--eval-dir", var("LENSEEK_NA_EVAL_DIR"), "--pairs", var("LENSEEK_NA_PAIRS")]),
      ("score_agreement.py", ["table", "--threshold", "48.5", "--eval-dir", var("LENSEEK_NA_EVAL_DIR")])],
     ["LENSEEK_NA_EVAL_DIR", "LENSEEK_NA_PAIRS", "LENSEEK_SCAN_DIR", "LENSEEK_PAIRS_DIR"], [], []),
    ("fig4", "Figure 5 and Supplementary Table S3 (ligand similarity of the pairs classified as similar)",
     [("fig4_ligand.py", ["kde"])],
     ["LENSEEK_EVAL_DIR", "LENSEEK_PAIRS", "LENSEEK_QUERY_ROOT", "LENSEEK_BSDB_ROOT", "LENSEEK_REFERENCE_DIR"], [], []),
    ("ligand-grid", "Supplementary Table S13 B ligand ROC-AUC and RE@1% (LEN-Seek, DeeplyTough, Folddisco; ligand "
     "ROC-AUC of the sequence-identity baseline); ligand values of seed 0 in Table S7 B; RE@1% of the Folddisco runs "
     "(caption of Table S13); ligand ROC-AUC among the sites excluded from the training of the model trained without "
     "homologous entries (Section S5)",
     [("fig4_ligand.py", ["grid"] + FD_DIRS + STRICT30_EXCLUDED)],
     ["LENSEEK_EVAL_DIR", "LENSEEK_DT_DIR", "LENSEEK_STRAT_DIR", "LENSEEK_REFERENCE_DIR"],
     ANNOT + [out("cache", "site_annot", "db80_cluster.tsv")] + PFAM + FD_DIST, []),
    ("baseline-probis", "Supplementary Table S13 B ProBiS columns (ProBiS over the whole database) of LEN-Seek, "
     "DeeplyTough, Folddisco and the sequence-identity and random baselines, the paired LEN-Seek - Folddisco "
     "differences of its caption and their detection, the ProBiS ROC-AUC of the Folddisco runs (caption of Table S13) "
     "and the share of the database the pre-filter returns (Section S6); Supplementary Table S8 B (G-LoSA and ProBiS "
     "ROC-AUC over the whole database)",
     [("baselines.py", ["compare"] + SEEDS + FD_DIRS)],
     ["LENSEEK_EVAL_DIR", "LENSEEK_SCAN_DIR", "LENSEEK_GLOSA_SCAN_DIR", "LENSEEK_PAIRS_DIR", "LENSEEK_DT_DIR",
      "LENSEEK_STRAT_DIR"],
     ANNOT + PFAM + FD_DIST + [os.path.join(FOLDDISCO, "prefilter", "per_query.tsv")], []),
    ("baseline-ligand", "Supplementary Table S13 B ligand ROC-AUC and RE@1% of the sequence-identity baseline and the "
     "random ranking; the paired ligand differences of LEN-Seek from Folddisco (caption of Table S13) and their "
     "detection; ligand values of seeds 1 and 2 in Table S7 B",
     baseline_ligand_steps, ["LENSEEK_EVAL_DIR", "LENSEEK_STRAT_DIR", "LENSEEK_REFERENCE_DIR"],
     [grid("main_s0")] + ANNOT + PFAM + ID_STRATA, []),
    ("strata", "Supplementary Table S10 A (ligand ROC-AUC by query identity, structure, ligand and site size)",
     [("fig4_ligand.py", ["strata"])],
     ["LENSEEK_EVAL_DIR"], [grid("main_s0")] + STRATA, []),
    ("figS1", "Supplementary Figure S1 (residues per binding site)",
     [("model_figures.py", ["site-size"])], ["LENSEEK_BSDB_ROOT"], [], []),
    ("figS2", "Supplementary Figure S2 (learning rate and KL weight per epoch)",
     [("model_figures.py", ["schedule"])], [], [], []),
    ("figS3", "Figure 3 (node-feature reconstruction) and Supplementary Figure S3 (structure reconstruction)",
     [("model_figures.py", ["recon"]), ("model_figures.py", ["plot"])], ["LENSEEK_CKPT", "LENSEEK_RECON_DATA"],
     [], []),
    ("cases", "Figures 6 and 7 (the cases shown and their values, and the ranked candidates) and the false positives "
     "of the top-100 results with a G-LoSA GA-score of 0.59 or higher (Discussion)",
     [("cases.py", [])],
     ["LENSEEK_EVAL_DIR", "LENSEEK_PAIRS", "LENSEEK_STRAT_DIR", "LENSEEK_REFERENCE_DIR", "LENSEEK_QUERY_ROOT",
      "LENSEEK_BSDB_ROOT"],
     [out("table1", f) for f in ("threshold_cv.csv", "thresholds.csv")] + ANNOT +
     [out("cache", "site_annot", f) for f in ("db_chains.fasta", "eval_vs_db_sens.m8")], [("FOLDSEEK", "foldseek")]),
    ("table2", "Table 2 (per-query cost), its caption values and the break-even query count; Supplementary Table S17 "
     "and Section S9",
     table2_steps, [], [TIMING["cpu"], TIMING["gpu"], TIMING["dt"], out("timing", "dt_prep.csv")], []),
    ("ablations", "Supplementary Table S14 and Section S7 (variants, sites common to the language-model variants); "
     "seed rows of Table S7 B; the model trained without homologous entries ranked among the sites excluded from its "
     "training (Section S5)",
     ablation_steps, ["LENSEEK_EVAL_ROOT", "LENSEEK_SCAN_DIR", "LENSEEK_PAIRS_DIR", "LENSEEK_DT_DIR",
                      "LENSEEK_STRAT_DIR", "LENSEEK_REFERENCE_DIR"], ANNOT + PFAM + ID_STRATA, []),
    ("factorial", "Supplementary Table S15 and Section S7 (node features and geometry in trained models: the 2 x 2 of "
     "Ankh or one-hot node features by geometry, its values and active latent units, differences and interaction)",
     factorial_steps, ["LENSEEK_EVAL_DIR", "LENSEEK_EVAL_ROOT", "LENSEEK_SCAN_DIR", "LENSEEK_GLOSA_SCAN_DIR",
                       "LENSEEK_PAIRS_DIR", "LENSEEK_DT_DIR", "LENSEEK_STRAT_DIR", "LENSEEK_REFERENCE_DIR"],
     ANNOT + PFAM + ID_STRATA + FD_DIST + FACTORIAL_DIST, []),
    ("filter-retrain", "Supplementary Section S1, model trained without the ligand filters: queries with filtered "
     "ligands against the others, and the difference in differences",
     filter_steps, ["LENSEEK_EVAL_DIR"], [grid("main_s0")], []),
    ("table3", "Supplementary Table S16 and the values of Section S8 (single-vector and two-stage retrieval, larger "
     "databases, ligand-similar fraction of the two-stage search); two-stage row of Table S17 and the two-stage "
     "search of the main text (Computational Efficiency)",
     [("table3_two_stage.py", scaling_dirs())], ["LENSEEK_EVAL_DIR", "LENSEEK_QUERY_ROOT", "LENSEEK_REFERENCE_DIR"],
     [os.path.join(ENV("LENSEEK_SCALING_DIR") or out("scaling"), "real", "recall_per_query.csv.gz")], []),
    ("holdout-table", "Supplementary Table S11 A-C and the rows of the later PDB entries in Table S10 B, from the "
     "stored outputs of the one scoring run",
     holdout_steps, ["LENSEEK_EVAL_DIR", "LENSEEK_REFERENCE_DIR"],
     list(HOLDOUT.values()) + ANNOT + [out("holdout", "mm", "holdout_vs_train_uncapped.m8"),
                                       out("holdout", "seqs", "seq_map.tsv")], []),
]
BY_NAME = {r[0]: r for r in RESULTS}


def missing_inputs(name):
    _, _, _, env, files, exes = BY_NAME[name]
    miss = []
    for v in env:
        if not ENV(v):
            miss.append(f"${v} not set" + (f" (or lenseek-fetch {FROM_ASSETS[v][0]})" if v in FROM_ASSETS else ""))
        elif not os.path.exists(ENV(v)):
            miss.append(f"${v}={ENV(v)} does not exist")
    miss += [f"{f} does not exist" + (f" (or lenseek-fetch {FETCH_HINT[f]})" if f in FETCH_HINT else "")
             for f in files if not os.path.exists(f)]
    for v, default in exes:
        exe = ENV(v) or default
        if not shutil.which(exe):
            miss.append(f"executable {exe} not found (${v})")
    return miss


def run(name, extra=(), dry=False):
    """'ran', 'failed' or 'skipped: ...' for one result; dry: print the commands (and the missing inputs) only."""
    _, item, steps, *_ = BY_NAME[name]
    miss = missing_inputs(name)
    if miss and not dry:
        print(f"[skip] {name}: " + "; ".join(miss), flush=True)
        return "skipped: " + "; ".join(miss)
    print(f"== {name}: {item}", flush=True)
    if miss:
        print("[missing] " + "; ".join(miss), flush=True)
    for script, args in (steps() if callable(steps) else steps):
        cmd = [sys.executable, os.path.join(HERE, script)] + [str(x) for x in args] + list(extra)
        print("$ " + " ".join(f"\"{x}\"" if x.startswith("$") else shlex.quote(x)  # $NAME of var() stays expandable
                              for x in ["python", f"analysis/{script}"] + cmd[2:]), flush=True)
        if dry:
            continue
        if subprocess.run(cmd).returncode != 0:
            print(f"[failed] {name}: {script} {args[0] if args else ''}", flush=True)
            return "failed"
    return "skipped: " + "; ".join(miss) if miss else "ran"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="result name (see --list)")
    ap.add_argument("--list", action="store_true", help="print the result names and manuscript items")
    ap.add_argument("--all", action="store_true", help="run every result whose inputs are present")
    ap.add_argument("--print", action="store_true", help="print the commands without running them")
    a, extra = ap.parse_known_args()
    if a.list:
        w = max(len(r[0]) for r in RESULTS)
        for name, item, *_ in RESULTS:
            print(f"{name:{w}s}  {item}")
        return
    if a.all:
        status = [(name, run(name, dry=a.print)) for name, *_ in RESULTS]
        print("\nresult" + " " * 14 + "status")
        for name, s in status:
            print(f"{name:20s}{s}")
        for n in NOTES:
            print("note:", n)
        sys.exit(1 if any(s == "failed" for _, s in status) else 0)
    if a.name not in BY_NAME:
        ap.error(f"unknown result {a.name!r}; see --list" if a.name else "give a result name, --list or --all")
    s = run(a.name, extra, dry=a.print)
    for n in NOTES:
        print("note:", n)
    sys.exit(0 if s == "ran" else SKIPPED if s.startswith("skipped") else 1)


if __name__ == "__main__":
    main()
