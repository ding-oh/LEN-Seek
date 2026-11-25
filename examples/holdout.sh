#!/bin/bash
# Binding sites from later PDB entries: construction from the PDB entries released after the database was compiled,
# their counts and the check of the build rules against BsitePDB, the thresholds fixed on the evaluation data, the one
# scoring run, its ProBiS labels and its metrics (Supplementary Section S5, Table S11 and the rows of the later PDB
# entries in Table S10 B). The sites are scored once: run `score` a single time, into a new output directory.
#
# Produces  $LENSEEK_OUT/holdout/        meta/, raw/cif/, build/, seqs/, mm/, sel/, BSDB_HO/, ankh_holdout/,
#                                        query_cache_holdout.pkl, manifest.tsv, clusters/, strata/, counts.tsv,
#                                        run/ (the scoring run), labels/ (store/, tables/), validate/ (entries of the
#                                        evaluation period in evaluation_period/)
#           $LENSEEK_OUT/holdout_score/  thresholds/, the per-query ligand metrics (*_per_query.csv),
#                                        mcc_holdout.json, mcc_strata.csv, table_S14{A,B,C}.csv, table1B_holdout.csv
# Inputs    see `python analysis/run.py -h` (thresholds, results) and the docstrings of scripts/holdout.py and
#           scripts/labels.py (construction, score, labels); read here: BSITEPDB_IDS and TRAIN_IDS (build, identity,
#           strata), LIGAND_CODES and CCD_SMILES (strata: database ligand CCDs, CCD SMILES table),
#           LENSEEK_EVAL_PERIOD_ROOT (validate), DT_QUERIES (score: DeeplyTough q_desc.npy and q_paths.txt of the
#           scored sites) and SEQID_M8 (results: MMseqs2 easy-search of seqs/holdout_query.fasta against the
#           training chains, -s 7.5 --max-seqs 200000 -e 10 --min-seq-id 0.25).
# Outputs   below $LENSEEK_OUT (default out/ in the repository).
#
#   bash examples/holdout.sh entries             # RCSB search, entry metadata and meta/entries_to_build.txt
#   bash examples/holdout.sh fetch|build|identity|select|write|embed|features|clusters|strata
#   bash examples/holdout.sh counts              # the counts of Supplementary Section S5
#   bash examples/holdout.sh validate-entries    # the same for the entries of the evaluation period
#   bash examples/holdout.sh validate            # build rules against BsitePDB on those entries (recall, Jaccard)
#   bash examples/holdout.sh thresholds          # thresholds on the evaluation data (before any scoring)
#   bash examples/holdout.sh score-check         # queries.tsv only, nothing encoded
#   bash examples/holdout.sh score               # the one scoring run
#   sbatch --array=0-39%8 examples/holdout.sh labels-score   # ProBiS scores of the union of the top-100 lists
#   bash examples/holdout.sh labels              # label table of the union
#   bash examples/holdout.sh results             # ligand metrics, top-k MCC, P@1, Table S11
#                                                #   (= python analysis/run.py holdout-table)
#
#SBATCH --job-name=lenseek_holdout
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export LENSEEK_OUT=$(realpath -m "${LENSEEK_OUT:-$PWD/out}")
B="python scripts/holdout.py"
S="python analysis/holdout_score.py"
L="python -u scripts/labels.py"
H=$LENSEEK_OUT/holdout
V=$H/evaluation_period
STEP=${1:?step name, see the header}; shift
case "$STEP" in
  entries|fetch|select|write|embed|features|clusters|counts) $B "$STEP" ;;
  build)    $B build --bsitepdb-ids "${BSITEPDB_IDS:?}" ;;
  identity) $B identity --train-ids "${TRAIN_IDS:?}" ;;
  strata)   $B strata --train-ids "${TRAIN_IDS:?}" --ligand-codes "${LIGAND_CODES:?}" --smiles "${CCD_SMILES:?}" ;;
  validate-entries) $B entries --date-from 2023-07-21 --date-to 2023-11-06 --validation --out "$V" ;;
  validate) $B fetch --out "$V"
            $B validate --ids "$V/meta/entries_to_build.txt" --cifdir "$V/raw/cif" \
                        --bsdb-root "${LENSEEK_EVAL_PERIOD_ROOT:?}" ;;
  thresholds) $S thresholds ;;
  score-check) $B score --run "$H/run_check" --prepare-only ;;
  score)    $B score --dt-queries "${DT_QUERIES:?}" ;;
  labels-score) $L score --tools probis --pairs "$H/run/pairs_union_for_labels.tsv" --store "$H/labels/store" \
                  --ntasks "${SLURM_ARRAY_TASK_COUNT:-1}" --workers "${SLURM_CPUS_PER_TASK:-16}" ;;
  labels)   $L tables "$H/run/pairs_union_for_labels.tsv" --store "$H/labels/store" --scan "" \
              --out "$H/labels/tables" ;;
  results)  $S ligand --run "$H/run" --manifest "$H/manifest.tsv" --seqid-m8 "${SEQID_M8:?}"
            $S mcc --run "$H/run" --pairs "$H/labels/tables/pairs_union_for_labels.scored.tsv"
            $S table --manifest "$H/manifest.tsv" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
