#!/bin/bash
# Structural labels and the comparison methods: ProBiS 2.4.7 and G-LoSA 2.2 over every query-site pair of the search
# database (481 x 114,661), the scored top-100 lists behind Table 1, Figures 4-7 and the Supplementary tables, the
# unrelated pairs, the DeeplyTough descriptors, distances and top-100 list with the checks of Supplementary
# Section S6, and the four Folddisco runs of Supplementary Section S6 (prefilter, matching, blosum62, entries). One step
# per call; the scan steps are SLURM array tasks (or a loop over the chunks).
#
# Produces  $LENSEEK_OUT/labels/      sites/, census.tsv, scan/ (all_status.npy, all_raw.npy), scan_glosa/
#                                     (all_status.npy, all_ga.npy), pairs/, store/, tables/<list>.scored.tsv, null/
#           $LENSEEK_OUT/deeplytough/ tasks/, runs/, q_desc.npy, db_desc.npy, dist.npy, failures.tsv, top100_pairs.tsv,
#                                     tough_check/, box/, overlap/, ood/
#           $LENSEEK_OUT/folddisco/   work/ (rewritten files, manifests, indexes), runs/<run>/, and per run
#                                     <run>/dist.npy, per_query.tsv, top100_pairs.tsv, q_paths.txt, db_paths.txt
# Inputs    see `python analysis/run.py -h` and the docstrings of scripts/labels.py (the tools: PROBIS_BIN, GLOSA_DIR,
#           JAVA_BIN, built by step build-glosa), scripts/deeplytough.py and scripts/folddisco.py; read here: DT_PYTHON
#           (python of the DeeplyTough environment, Python 3.6; dt-* steps), TOUGH_DIR (TOUGH-M1 files, dt-checks) and
#           FOLDDISCO (the Folddisco 3.0.0 executable, default `folddisco` on PATH; fd-* steps).
# Outputs   below $LENSEEK_OUT (default out/ in the repository); scripts/labels.py, scripts/deeplytough.py and
#           scripts/folddisco.py document every step.
#
#   bash examples/labels.sh build-glosa glosa_v2.2.tar.gz   # compile G-LoSA and its Java helpers into $GLOSA_DIR
#   bash examples/labels.sh sites                           # site lists and per-site census
#   sbatch --array=0-114%8 examples/labels.sh scan-probis   # ProBiS scan, one database chunk per task
#   bash examples/labels.sh assemble                        # the ProBiS scan as one array (scan/)
#   sbatch --array=0-114%8 examples/labels.sh scan-glosa    # G-LoSA scan
#   bash examples/labels.sh assemble-glosa
#   bash examples/labels.sh pairs [EVAL_DIR TAG]            # top-100 list of a model (default: LEN-Seek, main_s0)
#   sbatch --array=0-39%8 examples/labels.sh score LIST...  # G-LoSA scores of listed pairs (TOOLS=probis,glosa
#                                                           #   for pairs outside the scan; STORE=<store>)
#   bash examples/labels.sh tables LIST...                  # scored tables (ProBiS results from the scan)
#   bash examples/labels.sh null                            # unrelated pairs (their positive labels:
#                                                           #   python analysis/run.py unrelated)
#   bash examples/labels.sh dt-tasks                        # DeeplyTough task files
#   sbatch --array=0-15 examples/labels.sh dt-run           # database shards; TAG=q sbatch ... for the queries
#   bash examples/labels.sh dt-assemble                     # descriptors in evaluation order, dist.npy, top-100 list
#   bash examples/labels.sh dt-checks                       # TOUGH-M1 test, ligand box, overlap, query statistics
#   bash examples/labels.sh fd-prepare                      # Folddisco input files: site files and entries rewritten
#   bash examples/labels.sh fd-index                        # Folddisco indexes of the database sites and entries
#   sbatch examples/labels.sh fd-search RUN                 # one run: prefilter, matching or blosum62
#   sbatch --array=0-2 examples/labels.sh fd-search entries # the entries run in three parts
#   bash examples/labels.sh fd-assemble                     # distance matrix and top-100 list of each of the four runs
# Every array task writes only its own output, so tasks can be resubmitted; finished chunks and scored pairs are
# skipped. Run the scans and the pair scores on one CPU type (ProBiS limits its clique search by CPU time, so the CPU
# can change a result), and add site-specific options (partition, CPU constraint) to the sbatch line.
#
#SBATCH --job-name=lenseek_labels
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
L="python -u scripts/labels.py"
D=scripts/deeplytough.py
F="python -u scripts/folddisco.py"
export LENSEEK_OUT=$(realpath -m "${LENSEEK_OUT:-$REPO/out}")
OUT=$LENSEEK_OUT
export LENSEEK_TMP=${LENSEEK_TMP:-${TMPDIR:-/tmp}} OMP_NUM_THREADS=1
NCPU=${SLURM_CPUS_PER_TASK:-16}
STEP=${1:?step name, see the header}; shift
case "$STEP" in
  build-glosa) $L build-glosa "${1:?G-LoSA tarball}" ;;
  sites)       $L sites "${LENSEEK_EVAL_DIR:?}"
               $L census --workers "$NCPU" ;;
  scan-probis) $L scan-probis --workers "$NCPU" ;;
  assemble)    $L assemble ;;
  scan-glosa)  $L scan-glosa --workers "$NCPU" ;;
  assemble-glosa) $L assemble-glosa ;;
  pairs)       $L pairs "${1:-${LENSEEK_EVAL_DIR:?}}" "$OUT/labels/pairs/${2:-main_s0}_top100.tsv" ;;
  score)       $L score --tools "${TOOLS:-glosa}" --store "${STORE:-$OUT/labels/store}" \
                 --ntasks "${SLURM_ARRAY_TASK_COUNT:-1}" --workers "$NCPU" --pairs "$@" ;;
  tables)      $L tables --store "${STORE:-$OUT/labels/store}" "$@" ;;
  null)        $L smiles
               $L null "${LENSEEK_STRAT_DIR:?}/eval_vs_train_uncapped.m8" ;;
  dt-tasks)    "${DT_PYTHON:?}" $D tasks ;;
  dt-run)      TAG=${TAG:-db_$(printf %02d "${SLURM_ARRAY_TASK_ID:?array task id or TAG}")}
               # a writable local working directory: the DeeplyTough code writes next to the current directory
               W=$(mktemp -d "${TMPDIR:-/tmp}/dt_XXXX"); trap 'rm -rf "$W"' EXIT
               (cd "$W" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${DT_PYTHON:?}" "$REPO/$D" run \
                  --tasks "$OUT/deeplytough/tasks/$TAG.jsonl" --tag "$TAG" \
                  --workers "$NCPU") ;;
  dt-assemble) OMP_NUM_THREADS=1 "${DT_PYTHON:?}" $D assemble "$OUT/deeplytough/runs"
               python $D top100 ;;
  dt-checks)   "${DT_PYTHON:?}" $D tough-check --sanity-dir "${TOUGH_DIR:?}"
               "${DT_PYTHON:?}" $D box
               python $D overlap --sanity-dir "$TOUGH_DIR"
               python $D ood --sanity-dir "$TOUGH_DIR" ;;
  fd-prepare)  $F prepare --entries --workers "$NCPU" ;;
  fd-index)    $F index --threads "$NCPU"
               $F index --entries --threads "$NCPU" ;;
  fd-search)   $F search --run "${1:?run: prefilter, matching, blosum62 or entries}" --threads "$NCPU" \
                 --part "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" ;;
  fd-assemble) for r in prefilter matching blosum62 entries; do $F assemble --run $r --workers "$NCPU"; done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
