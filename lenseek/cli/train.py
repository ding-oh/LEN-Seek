"""lenseek-train: train a LEN-Seek model, one run of data/runs.yaml or a run given as a single-run YAML.

    lenseek-train --run main_s0 --data features/ --out runs/main_s0
    lenseek-train --config my_run.yaml
    accelerate launch --multi_gpu --num_processes 2 -m lenseek.cli.train --run main_s0 ...

--run names a run of the run table --config (default data/runs.yaml: the main-model recipe and every training run of
the paper as its differences from it). --config <file> without --run reads a single-run YAML with the keys name, data,
exclude_entries, model and train (lenseek.config). Extra LENSEEK_<SWITCH>=<value> arguments override the run with the
switch names of the paper's runs (EDGE_DROP, FULLGRAPH, STOPGRAD, NONADAPTED, NODE_CONST, NODE_DIM, W_KL, W_MSE, W_COS,
W_NORM, W_FAPE, W_RMSD, SAVE_EVERY, SEED, EPOCHS, BATCH, ACCUM, SUBSET). The global batch is batch x processes x
accumulation (512 in every run of the paper); --allow-global-batch accepts other values.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

from ..config import RunConfig, default_runs_file, load_run_config


def check_data(path: str) -> None:
    """Stop with one line when the training data of a run is missing (examples/train.sh calls this before launching)."""
    if not path:
        sys.exit("error: no training data: set `data:` in the run config or pass --data")
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found: fetch the training features with `lenseek-fetch train_features`, or "
                 f"give --data <feature store or training pickle>")


def select_run(config: str, run: Optional[str] = None, switches: Iterable[str] = ()) -> RunConfig:
    """Run ``run`` of the run table ``config``, or the single-run YAML ``config`` when ``run`` is None; stops with one
    line when the file or the run does not exist."""
    if not os.path.isfile(config):
        sys.exit(f"error: run config {config} not found")
    try:
        return load_run_config(config, switches, name=run)
    except ValueError as e:
        sys.exit(f"error: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenseek-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run name in the run table --config, e.g. main_s0")
    ap.add_argument("--config", default=default_runs_file(),
                    help="run table (default: data/runs.yaml of the repository) or a single-run YAML (without --run)")
    ap.add_argument("--data", default=None, help="feature store or training pickle (overrides the run's `data`)")
    ap.add_argument("--out", default=None, help="run directory (default runs/<name>)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (ep<N>.pth)")
    ap.add_argument("--allow-global-batch", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("switches", nargs="*", help="LENSEEK_<SWITCH>=<value> overrides")
    args = ap.parse_args(argv)
    run = select_run(args.config, args.run, args.switches)
    if args.data:
        run.data = args.data
    check_data(run.data)
    nproc = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = run.train.batch_size * nproc * run.train.grad_accum
    if global_batch != 512 and not args.allow_global_batch:
        sys.exit(f"error: global batch {run.train.batch_size} x {nproc} processes x {run.train.grad_accum} = "
                 f"{global_batch}, not 512 as in the paper (BATCH/ACCUM switches, or --allow-global-batch)")
    from ..train import main as train_main

    train_main(run, args.out or os.path.join("runs", run.name), resume=args.resume, progress=not args.no_progress)


if __name__ == "__main__":
    main()
