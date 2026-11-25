"""Training loop of the paper's runs (Hugging Face Accelerate; single or multi GPU).

AdamW with gradient accumulation, learning rate set once per epoch by linear decay, KL weight beta from
:func:`get_beta`, gradient-norm clipping on synchronisation steps, NaN batches skipped, ``ep<N>.pth`` every
``save_every`` epochs and ``final.pth`` after the last epoch, each with the run's provenance (lenseek.runinfo). No
validation set is used and no checkpoint is selected: ``final.pth`` is simply the state after epoch ``epochs``.

Launch: ``accelerate launch [--multi_gpu --num_processes 2] -m lenseek.cli.train --run <name>`` (the console command
``lenseek-train`` runs the same code in one process; the paper's runs are in data/runs.yaml).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import RunConfig, TrainConfig, switches_of
from .data import SiteDataset, open_training_data
from .geometry import edges_for
from .loss import vae_loss
from .model import VAE
from . import runinfo


def get_beta(epoch: int, cfg: TrainConfig) -> float:
    """KL weight: 0 before kl_start, then a linear ramp over half a cycle to beta_max, then beta_max."""
    if epoch < cfg.kl_start:
        return 0.0
    total_anneal_epochs = cfg.epochs - cfg.kl_start
    cycle_len = total_anneal_epochs // cfg.kl_cycles
    if cycle_len == 0:
        return cfg.beta_max
    cycle_epoch = (epoch - cfg.kl_start) % cycle_len
    beta = min(1.0, cycle_epoch / (cycle_len * 0.5)) * cfg.beta_max
    if (epoch - cfg.kl_start) // cycle_len >= cfg.kl_cycles:
        beta = cfg.beta_max
    return beta


def get_lr(epoch: int, cfg: TrainConfig) -> float:
    frac = epoch / cfg.epochs
    return cfg.start_lr * (1 - frac) + cfg.end_lr * frac


def site_entries(source):
    """PDB entry (directory name) of every site of a FeatureStore or training pickle, in order."""
    if hasattr(source, "site_ids"):
        return [s.split("/")[0] for s in source.site_ids]
    return [os.path.basename(os.path.dirname(p)) for p in source.path]


def entry_filter(source, exclude_file: str):
    """Indices of the sites whose entry is not listed in ``exclude_file`` (one PDB ID per line), in order."""
    with open(exclude_file) as f:
        excluded = {l.strip().lower() for l in f if l.strip()}
    return [i for i, e in enumerate(site_entries(source)) if e.lower() not in excluded]


def center(t: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Subtract each site's mean C-alpha position (training-time normalisation)."""
    mask_float = mask.float().unsqueeze(-1)
    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=eps)
    t_mean = (t * mask_float).sum(dim=1, keepdim=True) / num_valid
    return (t - t_mean) * mask_float


class LossTracker:
    def __init__(self):
        self.losses = {}

    def update(self, d):
        for k, v in d.items():
            self.losses.setdefault(k, []).append(v)

    def averages(self):
        return {k: sum(v) / len(v) if v else 0 for k, v in self.losses.items()}


def train_epoch(model, dataloader, optimizer, accelerator, kl_beta, run: RunConfig, progress: bool = True):
    model.train()
    tracker = LossTracker()
    mcfg, tcfg = run.model, run.train
    iterator = dataloader
    if progress and accelerator.is_main_process:
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="Training", leave=False, mininterval=30)
    for batch in iterator:
        # accumulate(): gradients sync and the optimizer steps only every grad_accum micro-batches,
        # so zero_grad() comes after step(), never before the forward pass.
        with accelerator.accumulate(model):
            h_true, trans, mask, rots = batch
            t = center(trans, mask)
            r = rots
            if mcfg.node_const:
                h_true = torch.zeros_like(h_true)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            e, knn_mask = edges_for(mcfg, t, r, mask_2d)
            h_pred, t_pred, r_pred, mu, logvar, z, h_target = model(h_true, e, mask, knn_mask)
            losses = vae_loss(h_true=h_true if mcfg.nonadapted else h_target, t_true=t, r_true=r,
                              h_pred=h_pred, t_pred=t_pred, r_pred=r_pred, mu=mu, logvar=logvar, mask=mask,
                              kl_beta=kl_beta, cfg=tcfg)
            total_loss = losses["total_loss"]
            if torch.isnan(total_loss):
                if accelerator.is_main_process:
                    print("[lenseek.train] NaN loss; batch skipped", flush=True)
                continue
            accelerator.backward(total_loss)
            if tcfg.max_grad > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), tcfg.max_grad)
            optimizer.step()
            optimizer.zero_grad()
        tracker.update({k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
    return tracker.averages()


def save_checkpoint(model, optimizer, epoch, path, accelerator, run_info=None):
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"epoch": epoch, "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "run_info": run_info}, path)
        accelerator.print(f"[lenseek.train] checkpoint saved to {path}", flush=True)


def main(run: RunConfig, out_dir: str, resume: Optional[str] = None, progress: bool = True) -> None:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    tcfg = run.train
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=tcfg.grad_accum, kwargs_handlers=[ddp_kwargs])
    set_seed(tcfg.seed)
    global_batch = tcfg.batch_size * accelerator.num_processes * tcfg.grad_accum
    accelerator.print(f"[lenseek.train] run {run.name}: per-process batch {tcfg.batch_size} x {accelerator.num_processes} "
                      f"processes x accumulation {tcfg.grad_accum} = global {global_batch}", flush=True)
    accelerator.print(f"[lenseek.train] switches {json.dumps(switches_of(run))}", flush=True)

    if not run.data:
        sys.exit("[lenseek.train] no training data: set `data:` in the run config or pass --data")
    source = open_training_data(run.data)
    dataset = SiteDataset(source)
    if run.exclude_entries:
        keep = entry_filter(source, run.exclude_entries)
        dataset = torch.utils.data.Subset(dataset, keep)
        accelerator.print(f"[lenseek.train] {len(keep)} of {len(source)} sites kept after excluding the entries in "
                          f"{run.exclude_entries}", flush=True)
    if tcfg.subset > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(tcfg.subset, len(dataset))))
        accelerator.print(f"[lenseek.train] SUBSET: training on {len(dataset)} sites", flush=True)

    info = runinfo.collect(run, data_path=run.data, n_sites=len(dataset), num_processes=accelerator.num_processes,
                           global_batch=global_batch)
    model = VAE.from_config(run.model)
    dataloader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=tcfg.start_lr)

    start_epoch = 1
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        old = (ckpt.get("run_info") or {}).get("switches")
        if old is not None and old != info["switches"]:
            sys.exit(f"[lenseek.train] resume switches differ from {resume}: {old} vs {info['switches']}")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        info["resumed_from"] = os.path.abspath(resume)
        accelerator.print(f"[lenseek.train] resumed from {resume} at epoch {start_epoch}", flush=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    os.makedirs(out_dir, exist_ok=True)
    history = os.path.join(out_dir, "loss_history.jsonl")
    if accelerator.is_main_process:
        with open(os.path.join(out_dir, "run_info.jsonl"), "a") as f:
            f.write(json.dumps({"start_epoch": start_epoch, **info}) + "\n")

    for epoch in range(start_epoch, tcfg.epochs + 1):
        kl_beta = get_beta(epoch, tcfg)
        lr = get_lr(epoch, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        avg = train_epoch(model, dataloader, optimizer, accelerator, kl_beta, run, progress)
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}: LR={lr:.6f}, Beta={kl_beta:.4f}", flush=True)
            for k, v in avg.items():
                print(f"  {k}: {v:.4f}", flush=True)
            with open(history, "a") as f:
                f.write(json.dumps({"epoch": epoch, "lr": lr, "beta": kl_beta, **avg}) + "\n")
            if tcfg.save_every > 0 and epoch % tcfg.save_every == 0:
                save_checkpoint(model, optimizer, epoch, os.path.join(out_dir, f"ep{epoch}.pth"), accelerator, info)
            if epoch == tcfg.epochs:
                save_checkpoint(model, optimizer, epoch, os.path.join(out_dir, "final.pth"), accelerator, info)
