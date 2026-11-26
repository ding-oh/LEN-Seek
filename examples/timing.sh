#!/bin/bash
# The CPU timing runs of Table 2 and Supplementary Table S17 on one node that runs no other job: the ProBiS surface
# extraction of the database on its own (the ProBiS preprocessing time), ProBiS, G-LoSA, LEN-Seek per query, the
# LEN-Seek preprocessing of 500 random database entries, DeeplyTough (its pipeline per query, the network alone, the
# search) and
# Folddisco (its index, then one process per query), one after another with the same 20 queries (seed 0). The node
# is allocated whole (--exclusive) and each part is confined with taskset to the first 16 CPUs of the allocation, the
# cores a 16-CPU job gets on an idle node, so process pools, OpenMP / BLAS and JVMs see 16 cores. The parts are the
# steps of examples/analysis.sh timing; its GPU steps (lenseek-gpu, prep-gpu, on a GPU node) and dt-prep complete the
# inputs of Table 2 and Table S17 (python analysis/run.py table2).
#
#   sbatch examples/timing.sh        # from the repository root; add site options (partition, CPU type) to the line
#
# Produces  $LENSEEK_OUT/timing/ (default out/timing/ in the repository): the files of scripts/timing.py probis-prep,
#           probis, glosa, lenseek --device cpu, lenseek-prep --sample 500, deeplytough, dt-net, dt-search and
#           folddisco (listed in its docstring); <step>.log of each part; loadavg_trace.txt (/proc/loadavg every
#           30 s); steps.txt (host, CPU model, CPU list, and the start, end, exit code and load average of each part)
# Inputs    those of the timing steps of examples/analysis.sh (the docstring of scripts/timing.py lists them)
#
#SBATCH --job-name=lenseek_timing
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -u
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
export LENSEEK_OUT=$(realpath -m "${LENSEEK_OUT:-$REPO/out}")
O=$LENSEEK_OUT/timing
mkdir -p "$O"
( while :; do echo "$(date -Is) $(cat /proc/loadavg)"; sleep 30; done ) > "$O/loadavg_trace.txt" 2>&1 &
TRACE=$!
trap 'kill $TRACE 2>/dev/null' EXIT
CPUS=$(python -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:16])))") || CPUS=0-15
echo "$(date -Is) host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)" \
     "nproc=$(nproc) taskset=$CPUS" > "$O/steps.txt"
step() {   # step NAME: examples/analysis.sh timing NAME on the first 16 CPUs, log $O/NAME.log
  local rc
  echo "$(date -Is) start $1 load $(cat /proc/loadavg)" >> "$O/steps.txt"
  taskset -c "$CPUS" bash examples/analysis.sh timing "$1" > "$O/$1.log" 2>&1
  rc=$?
  echo "$(date -Is) end $1 rc=$rc load $(cat /proc/loadavg)" >> "$O/steps.txt"
}
for s in probis-prep probis glosa lenseek-cpu prep-cpu deeplytough dt-net dt-search folddisco; do step "$s"; done
echo "$(date -Is) $(cat /proc/loadavg) (end)" >> "$O/loadavg_trace.txt"
cat "$O/steps.txt"
