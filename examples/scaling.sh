#!/bin/bash
# Larger databases: the single-vector baselines (Supplementary Table S15 A), the two-stage search on databases from the
# database to 10^7 sites (recall, time, memory: Supplementary Section S8 and Table S15 B) and the
# tables. A feasibility measurement, not a search mode of the package.
#
# Produces  $LENSEEK_OUT/single_vector/      cache/ (residue CSR, full-search top 1,000), exact_scan.tsv, pooling.tsv,
#                                             faiss_*.tsv, rerank_latency.tsv, twostage_residue.tsv, fde*.tsv
#           $LENSEEK_SCALING_ADDED           added_mu.npy, added_mask.npy, added_keys.txt, dist_added.npy,
#                                             encode_time.json (the sites removed by the ligand filters)
#           $LENSEEK_OUT/scaling/<db>/       gen.json, gt_top1000.npz, build_<tag>.json, cands_np16.npz, stage1.json,
#                                             recall_per_query.csv.gz, search_T<threads>.*, exact_T*, exact_gpu*.json
#           $LENSEEK_OUT/table3_two_stage/   table3.csv, tableS9_A.csv, tableS9_B.csv, ligand_ratio.csv and the tables
#                                             behind them
# Inputs    see `python analysis/run.py -h` (table3) and the docstrings of scripts/scaling.py, scripts/single_vector.py
#           and scripts/evaluate.py (LENSEEK_EVAL_DIR with the latents of scripts/evaluate.py, db_mu.npy included;
#           asset eval_main_s0 has no db_mu.npy, so every step but table3 needs a run of scripts/evaluate.py); read
#           here: LENSEEK_SCALING_THREADS (default 32, threads of the added and accuracy steps), LENSEEK_SCALING_LOCAL
#           (node-local scratch: about 150 GB for a 10^7-site database), LENSEEK_SCALING_REAL_DIST (default
#           $LENSEEK_EVAL_DIR/dist.npy), LENSEEK_SCALING_ADDED (default $LENSEEK_OUT/scaling/added), LENSEEK_CKPT
#           (added: the LEN-Seek checkpoint) and the feature store of the sites removed by the ligand filters (added:
#           $LENSEEK_OUT/filter_free_data/sites/added of scripts/filter_free_data.py sites). A 10^7-site database needs
#           more memory than a workstation; the study ran on cluster nodes, the timings on otherwise idle ones.
# Outputs   below $LENSEEK_OUT (default out/ in the repository).
#
#   bash examples/scaling.sh single-vector      # scripts/single_vector.py prep ... fde-faiss (prep: also scaling's input)
#   bash examples/scaling.sh prep               # site clusters (50 % clusters of the database chains), generator
#                                               #   parameters
#   bash examples/scaling.sh added              # encodings and query distances of the sites removed by the
#                                               #   ligand filters (real150k)
#   bash examples/scaling.sh accuracy [db ...]  # generate, index, stage-1 candidates (one thread at 10^7 sites), recall
#   bash examples/scaling.sh timing [db ...]    # store, index build time (16 threads), two-stage time with the accuracy
#                                               #   index at 1 thread (candidate lists checked against accuracy's) and
#                                               #   16 threads, exhaustive scan
#   bash examples/scaling.sh gpu                # exhaustive scan on one GPU
#   bash examples/scaling.sh table3             # Table S15, ligand fraction (= python analysis/run.py table3)
# db: real real150k chim3e5 chim1e6 chim1e7 copy1e6 copy1e7 (default all)
set -euo pipefail
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then sed -n '2,36p' "$0"; exit 2; fi
cd "$(dirname "$0")/.."
export LENSEEK_OUT=${LENSEEK_OUT:-$PWD/out}
export LENSEEK_SCALING_REAL_DIST=${LENSEEK_SCALING_REAL_DIST:-${LENSEEK_EVAL_DIR:-}/dist.npy}
T=${LENSEEK_SCALING_THREADS:-32}
L=${LENSEEK_SCALING_LOCAL:-${TMPDIR:-/tmp}/lenseek_scaling}
export LENSEEK_SCALING_ADDED=${LENSEEK_SCALING_ADDED:-$LENSEEK_OUT/scaling/added}
A=$LENSEEK_SCALING_ADDED
SC="python -u scripts/scaling.py"
step=$1; shift
dbs=${*:-real real150k chim3e5 chim1e6 chim1e7 copy1e6 copy1e7}
case $step in accuracy|timing)
  if [[ " $dbs " == *" real150k "* ]] && [ ! -f "$A/added_mu.npy" ]; then
    echo "error: $step: real150k needs $A (run the added step first)" >&2
    exit 2
  fi ;;
esac
case $step in
  single-vector)
    for s in prep exact pooling faiss residue fde fde-faiss; do python -u scripts/single_vector.py $s; done ;;
  prep)
    $SC clusters
    $SC params --threads "$T" ;;
  added)
    python -u scripts/evaluate.py --checkpoint "${LENSEEK_CKPT:?}" --added "${LENSEEK_EVAL_DIR:?}" \
      --features "$LENSEEK_OUT/filter_free_data/sites/added" --out "$A" --threads "$T" ;;
  accuracy)
    for db in $dbs; do
      case $db in chim1e7|copy1e7) t1=1;; *) t1=$T;; esac
      $SC generate "$db" --threads "$T"
      $SC index "$db" --threads "$T"
      $SC stage1 "$db" --threads "$t1"
      $SC recall "$db"
    done ;;
  timing)
    for db in $dbs; do
      $SC generate "$db" --store-only
      $SC index "$db" --tag time --index "$L/$db/index_time.faiss"     # searches use the accuracy index
      $SC time "$db" --threads 1 --ref-cands "$LENSEEK_OUT/scaling/$db/cands_np16.npz"
      $SC time "$db"
      $SC exact "$db"
      case $db in chim1e7|copy1e6|copy1e7) qs=sub48;; *) qs=all;; esac
      $SC exact "$db" --threads 1 --queries "$qs"
    done ;;
  gpu)    $SC gpu ;;
  table3) python analysis/run.py table3 ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
