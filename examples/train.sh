#!/bin/bash
# Train one model of the paper (GPU; about two days on two GPUs). data/runs.yaml holds every training run.
#
#   examples/train.sh <run> [training features]      # a run of data/runs.yaml, e.g. main_s0
#   examples/train.sh <file>.yaml [training features] # a single-run YAML (lenseek-train --config)
#
# main_s0 is LEN-Seek; its training features are the asset train_features (LEN-Seek data record) or the training
# pickle. Each run stops after a fixed 200 epochs and keeps the last checkpoint in runs/<run>/final.pth. The number of
# processes is the one that makes the global batch 512 (batch_size x processes x grad_accum): two GPUs for most runs,
# one for the runs with batch_size 32 and grad_accum 16. Needs the train extra (accelerate).
# Each run is then evaluated like the original model (eval dir in the database order of the original model) before
# `python analysis/run.py ablations` reads it:
#   python scripts/evaluate.py --checkpoint runs/<run>/final.pth --features <node features of the database> \
#       --queries <query cache> --out $LENSEEK_EVAL_ROOT/out_<run> --db-order $LENSEEK_EVAL_DIR/db_paths.txt
# (a language-model variant lacks the sites without a chain embedding; they get NaN distances). The original model's
# eval dir also gets the rotated searches read by `python analysis/run.py rotation` (add --rotations: 3 rotations per
# query).
# Training data of filter_free_s0 (trained without the ligand filters), and the identity of the evaluation proteins to
# the entries it adds (read by `python analysis/run.py filter-retrain`):
#   python scripts/filter_free_data.py sites --embeddings <per-chain Ankh files>
#   python scripts/filter_free_data.py build --base <training features>
#   python scripts/filter_free_data.py identity
#   examples/train.sh filter_free_s0 $LENSEEK_OUT/filter_free_data/features
set -euo pipefail
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
RUN=${1:?run name in data/runs.yaml (e.g. main_s0) or a single-run YAML}
DATA=${2:+$(abspath "$2")}
case "$RUN" in
  *.yaml|*.yml) CFG=$(abspath "$RUN"); RUN= ;;
  *)            CFG=data/runs.yaml ;;
esac
cd "$(dirname "$0")/.."
# resolve the run and check its training data once here, not in every training process; prints name and processes
OUT=$(python -c 'import sys
from lenseek.cli.train import check_data, select_run
run = select_run(sys.argv[1], sys.argv[2] or None)
check_data(sys.argv[3] or run.data)
per_process = run.train.batch_size * run.train.grad_accum
if 512 % per_process:
    sys.exit(f"error: batch_size x grad_accum = {per_process} does not divide the global batch 512; use lenseek-train")
print(run.name, 512 // per_process)' "$CFG" "$RUN" "$DATA")
read -r NAME NPROC <<< "$OUT"
command -v accelerate >/dev/null || { echo "error: accelerate not found; install the train extra (pip install -e \".[train]\")" >&2; exit 2; }
MULTI=
if [ "$NPROC" -gt 1 ]; then MULTI=--multi_gpu; fi
accelerate launch $MULTI --num_processes "$NPROC" --mixed_precision no -m lenseek.cli.train --config "$CFG" \
    ${RUN:+--run "$RUN"} ${DATA:+--data "$DATA"} --out "runs/$NAME"
