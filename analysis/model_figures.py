#!/usr/bin/env python
"""Figures of the model input, the training schedule and the reconstruction: residues per binding site
(Supplementary Figure S1), learning rate and KL weight per epoch (Figure S2), node-feature reconstruction (main-text
Figure 3) and structure reconstruction (Figure S3); and the rotation check of the encodings (Supplementary Section
S10).

Subcommands:
  site-size  Figure S1: residues per pocket file (every ligand_pocket_*.pdb under the database root), a residue being
             a distinct (chain ID, residue number + insertion code) on the ATOM lines; histogram with 1-wide bins,
             mean and variance (ddof 0). The counts are cached in $LENSEEK_OUT/cache/site_sizes_<root>.csv.
  schedule   Figure S2: lenseek.train.get_lr / get_beta at every epoch with the lenseek.config.TrainConfig defaults
             (the settings of the original model).
  recon      encodes and decodes every site of the reconstruction set with a checkpoint (CPU). The latent vector is
             sampled from the posterior (z = mu + sigma * eps, numpy and torch seeded before the model is built;
             sites with more than 65 residues are subsampled by the dataset with numpy). Per residue: cosine
             similarity between the adapted 512-d node features and their reconstruction (the raw features for a
             non-adapted model). Per site: FAPE (lenseek.loss.fape_loss, clamp 10 A) and the C-alpha RMSD after
             Kabsch superposition (reconstruction rotated onto the input structure).
  plot       from recon.npz: Figure 3, A the cosine-similarity histogram with its mean and median, B the first two
             principal components of the adapted node features and of their reconstruction (10,000 residues drawn
             with default_rng(0) without replacement, PCA fitted on both together, the sign of each component set so
             that the reconstruction has a positive mean on it; filled kernel density contours);
             and Figure S3: FAPE and RMSD histograms and the sites at the first quartile and the median of the RMSD
             (sorted RMSD at int(0.25 n) and int(0.5 n)).
  rotation   from the rotated searches of scripts/evaluate.py --rotations (<eval dir>/rotation.tsv): the searches whose
             top-1 result is that of the unrotated site (rank_of_top1 = 1) and the median and maximum Chamfer distance
             between the encodings of a site and of its rotated copy.

Inputs: see `python analysis/run.py -h`; read here: for recon LENSEEK_CKPT (--ckpt) and LENSEEK_RECON_DATA (--dataset,
the featurised reconstruction set in training-pickle format, lenseek.data.load_legacy_pickle: reconstruction_set.pkl
of asset eval_extras); for rotation LENSEEK_EVAL_DIR (--table, default $LENSEEK_EVAL_DIR/rotation.tsv).
Outputs ($LENSEEK_OUT/model_figures/): figS1_site_size.{png,pdf,csv}, figS1_stats.csv, figS2_schedule.{png,pdf,csv},
recon.npz, recon_set_counts.csv, fig3_recon.{png,pdf}, fig3_stats.csv, figS3_struct.{png,pdf}, figS3_stats.csv,
figS3_per_site.csv, rotation.json.

    python analysis/model_figures.py site-size
    python analysis/model_figures.py schedule
    python analysis/model_figures.py recon --ckpt lenseek_main_s0.pth
    python analysis/model_figures.py plot
    python analysis/model_figures.py rotation
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, OUT_ROOT, apply_style, require_path, savefig  # noqa: E402

OUT = os.path.join(OUT_ROOT, "model_figures")
# print sizes: figures are drawn at their placement width (TeX points), so an N-pt font prints at N pt
PT_PER_IN, COLUMNWIDTH_MAIN, TEXTWIDTH_SUPP = 72.27, 254.65216, 469.75502
XMIN = 0.96                                 # left limit of the cosine histogram
FIG3_PX = (2221, 1047)                      # pixel size of Figure 3 at its placement width (\columnwidth)
FS_S1 = dict(xlabel=8.4, ticks=7.2, legend=8.4, annotation=8.4)
FS_S2 = dict(title=5.2, xlabel=4.8, ylabel=4.8, ticks=4.8, panel=6.1)
FS_S3 = dict(title=7.1, xlabel=5.8, ylabel=5.8, ticks=3.9, legend=6.4, panel=6.8, title_3d=6.8, ticks_3d=4.8)
FS_FIG3 = dict(title=6.33, xlabel=6.17, ylabel=6.13, ticks=4.65, legend=4.60, panel=8.15)
LW_FIG3, LW_LINES_FIG3 = 0.47, 0.8
PCA_RESIDUES, PCA_SEED = 10000, 0


# ---------------------------------------------------------------- Figure S1
def count_residues(path):
    res = set()
    try:
        with open(path) as f:
            for l in f:
                if l.startswith("ATOM"):
                    res.add((l[21], l[22:27]))               # chain, resSeq + iCode
    except OSError:
        return path, -1
    return path, len(res)


def cmd_site_size(a):
    require_path(a.bsdb_root, "--bsdb-root", "LENSEEK_BSDB_ROOT")
    cache = os.path.join(CACHE, f"site_sizes_{os.path.basename(os.path.normpath(a.bsdb_root))}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        files = sorted(glob.glob(os.path.join(a.bsdb_root, "*", "ligand_pocket_*.pdb")))
        print(f"counting residues in {len(files)} pocket files ...")
        with Pool(a.workers) as pool:
            df = pd.DataFrame(pool.map(count_residues, files, chunksize=500), columns=["path", "n_res_all"])
        os.makedirs(CACHE, exist_ok=True)
        df.to_csv(cache, index=False)
    df = df[df.n_res_all >= 0]
    n = df.n_res_all.to_numpy()
    df.to_csv(os.path.join(OUT, "figS1_site_size.csv"), index=False)
    st = dict(n_sites=len(n), mean=n.mean(), variance=n.var(), min=int(n.min()), max=int(n.max()))
    pd.DataFrame([st]).to_csv(os.path.join(OUT, "figS1_stats.csv"), index=False)
    print(pd.DataFrame([st]).T.to_string(header=False))

    fs = FS_S1
    apply_style(base=fs["ticks"], lw=0.8)
    w = 0.7 * TEXTWIDTH_SUPP
    fig = plt.figure(figsize=(w / PT_PER_IN, w * 288 / 374 / PT_PER_IN))
    ax = fig.add_axes([0.125, 0.11, 0.85, 0.865])
    ax.hist(n, bins=np.arange(0, n.max() + 2) - 0.5, edgecolor="black", linewidth=0.6, color="tab:blue")
    ax.axvline(n.mean(), color="red", linestyle="--", linewidth=0.8, label=f"Mean: {n.mean():.2f}")
    ax.text(n.mean() + 0.4, ax.get_ylim()[1] * 0.93, f"Mean: {n.mean():.2f}", color="red",
            fontsize=fs["annotation"], va="bottom")
    ax.text(0.03, 0.87, f"Variance: {n.var():.2f}", color="blue", transform=ax.transAxes, fontsize=fs["annotation"])
    ax.set_xlabel("Number of residues", fontsize=fs["xlabel"], labelpad=1)
    ax.legend(fontsize=fs["legend"], loc="upper right")
    ax.tick_params(labelsize=fs["ticks"])
    savefig(fig, os.path.join(OUT, "figS1_site_size"))


# ---------------------------------------------------------------- Figure S2
def cmd_schedule(a):
    from lenseek.config import TrainConfig
    from lenseek.train import get_beta, get_lr
    cfg = TrainConfig()
    ep = np.arange(1, cfg.epochs + 1)
    s = pd.DataFrame({"epoch": ep, "lr_config": [get_lr(int(e), cfg) for e in ep],
                      "beta_config": [get_beta(int(e), cfg) for e in ep]})
    s.to_csv(os.path.join(OUT, "figS2_schedule.csv"), index=False)
    print(f"beta first reaches its maximum at epoch {int(s.epoch[np.argmax(s.beta_config.to_numpy())])}; "
          f"lr epoch 1 = {s.lr_config.iloc[0]:.6g}, last = {s.lr_config.iloc[-1]:.6g}")

    fs = FS_S2
    apply_style(base=fs["ticks"], lw=0.3)
    w = 0.9 * TEXTWIDTH_SUPP
    fig = plt.figure(figsize=(w / PT_PER_IN, w * 572 / 1334 / PT_PER_IN))
    axes = [fig.add_axes([0.058, 0.115, 0.43, 0.8]), fig.add_axes([0.557, 0.115, 0.43, 0.8])]
    axes[0].plot(s.epoch, s.lr_config, color="#3a76af", linewidth=1.0)
    axes[1].plot(s.epoch, s.beta_config, color="#c43a31", linewidth=1.0)
    for ax, title, ylab, lab in [(axes[0], "LR schedule", "Learning Rate", "A"),
                                 (axes[1], r"$\beta$ schedule", r"KL Weight ($\beta$)", "B")]:
        ax.set_title(title, fontsize=fs["title"], pad=3)
        ax.set_xlabel("Epoch", fontsize=fs["xlabel"], labelpad=1)
        ax.set_ylabel(ylab, fontsize=fs["ylabel"], labelpad=2)
        ax.set_xticks(np.arange(0, 201, 25))
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
        ax.tick_params(labelsize=fs["ticks"], length=2, pad=1.5)
        fig.text(ax.get_position().x0 - 0.047, 0.985, lab, fontsize=fs["panel"], va="top")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axes[0].yaxis.get_offset_text().set_fontsize(fs["ticks"])
    savefig(fig, os.path.join(OUT, "figS2_schedule"))


# ---------------------------------------------------------------- reconstruction pass
def kabsch_align(P, Q):
    """P superposed onto Q (rows are points): (P - mean P) R^T + mean Q."""
    Pm, Qm = P.mean(0, keepdims=True), Q.mean(0, keepdims=True)
    H = (P - Pm).T @ (Q - Qm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Vt[-1, :] *= d
    R = Vt.T @ U.T                       # rotates column vectors: q = R p
    return (P - Pm) @ R.T + Qm


def rmsd(A, B):
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def cmd_recon(a):
    import torch
    import lenseek
    from lenseek.checkpoint import load_model
    from lenseek.config import NUM_PTS
    from lenseek.data import SiteDataset, load_legacy_pickle
    from lenseek.geometry import KNN_RULE, edges_for
    from lenseek.loss import fape_loss
    require_path(a.ckpt, "--ckpt", "LENSEEK_CKPT")
    require_path(a.dataset, "--dataset", "LENSEEK_RECON_DATA")
    torch.set_num_threads(a.threads)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, cfg, info = load_model(a.ckpt, None, device="cpu")
    print(f'model {info.name or "(unregistered)"}; settings from {info.config_source} | kNN rule: {KNN_RULE}')
    src = load_legacy_pickle(a.dataset)
    ds = SiteDataset(src, num_pts=getattr(src, "pocPts", NUM_PTS))
    n = len(ds)
    paths = list(getattr(src, "path", [""] * n))
    print(f"{n} sites from {a.dataset}; lenseek {lenseek.__version__}; ckpt {a.ckpt}")

    cos_all, rid_all, ht_all, hp_all = [], [], [], []
    fape, rm, nres, T_true, T_pred, masks = [], [], [], [], [], []
    pad = lambda X: np.pad(X, ((0, 65 - len(X)), (0, 0))).astype(np.float32)  # noqa: E731
    with torch.no_grad():
        for s in range(0, n, a.batch):
            items = [ds[i] for i in range(s, min(s + a.batch, n))]
            h, t, mask, r = (torch.stack([it[j] for it in items]) for j in range(4))
            if cfg.node_const:
                h = torch.zeros_like(h)
            e, knn = edges_for(cfg, t, r, mask.unsqueeze(2) & mask.unsqueeze(1))
            h_pred, t_pred, r_pred, mu, logvar, z, h_tgt = model(h, e, mask, knn)
            for b in range(len(items)):
                m = mask[b].numpy().astype(bool)
                ht = (h if cfg.nonadapted else h_tgt)[b][mask[b]]
                hp = h_pred[b][mask[b]]
                cos_all.append(torch.nn.functional.cosine_similarity(hp, ht, dim=-1).numpy())
                rid_all.append(np.full(m.sum(), s + b, dtype=np.int32))
                ht_all.append(ht.numpy().astype(np.float16))
                hp_all.append(hp.numpy().astype(np.float16))
                fape.append(fape_loss(t_pred[b:b + 1], r_pred[b:b + 1], t[b:b + 1], r[b:b + 1], mask[b:b + 1]).item())
                Q, P = t[b].numpy()[m].astype(np.float64), t_pred[b].numpy()[m].astype(np.float64)
                c = kabsch_align(P, Q)
                rm.append(rmsd(c, Q))
                nres.append(int(m.sum()))
                T_true.append(pad(Q)); T_pred.append(pad(c)); masks.append(m)
            print(f"  {min(s + a.batch, n)}/{n}", end="\r", flush=True)
    meta = dict(ckpt=os.path.abspath(a.ckpt), model=info.name or "", settings=info.config_source,
                dataset=os.path.abspath(a.dataset), batch=a.batch, seed=a.seed, lenseek_version=lenseek.__version__,
                knn_rule=KNN_RULE)
    f = os.path.join(OUT, "recon.npz")
    np.savez_compressed(f, cos=np.concatenate(cos_all), res_site=np.concatenate(rid_all),
                        h_true=np.concatenate(ht_all), h_pred=np.concatenate(hp_all), fape=np.array(fape),
                        rmsd=np.array(rm), n_res=np.array(nres), paths=np.array(paths), T_true=np.stack(T_true),
                        T_pred=np.stack(T_pred), mask=np.stack(masks), meta_json=np.array(json.dumps(meta)))
    cos = np.concatenate(cos_all)
    print(f"\nwrote {f}\n  residues {len(cos)}: cosine mean {cos.mean():.4f} median {np.median(cos):.4f}\n"
          f"  sites {len(fape)}: FAPE mean {np.mean(fape):.3f} median {np.median(fape):.3f}; "
          f"RMSD mean {np.mean(rm):.3f} median {np.median(rm):.3f}")


# ---------------------------------------------------------------- Figures 3 and S3 from recon.npz
def plot_fig3(d):
    """Figure 3: A, cosine similarity of the reconstructed node features; B, their first two principal components."""
    from sklearn.decomposition import PCA
    cos = d["cos"].astype(np.float64)
    T, P = d["h_true"].astype(np.float32), d["h_pred"].astype(np.float32)
    n = min(PCA_RESIDUES, len(T))
    idx = np.random.default_rng(PCA_SEED).choice(len(T), n, replace=False)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(np.concatenate([T[idx], P[idx]], axis=0))
    Z *= np.where(Z[n:].mean(0) < 0, -1.0, 1.0)     # component signs: the reconstruction has a positive mean on each
    zt, zp = Z[:n], Z[n:]
    W, H = FIG3_PX
    st = dict(n_residues=len(cos), n_sites=len(np.unique(d["res_site"])), target_dim=T.shape[1],
              mean=cos.mean(), median=np.median(cos), p01=np.percentile(cos, 1), min=cos.min(),
              below_xmin=int((cos < XMIN).sum()), pca_residues=n, pca_seed=PCA_SEED,
              pc1_explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
              pc2_explained_variance_ratio=float(pca.explained_variance_ratio_[1]),
              **{f"pc{j + 1}_{w}_{s}": float(f(z[:, j])) for j in range(2) for w, z in (("original", zt), ("recon", zp))
                 for s, f in (("mean", np.mean), ("sd", np.std))})
    pd.DataFrame([st]).to_csv(os.path.join(OUT, "fig3_stats.csv"), index=False)
    print(pd.DataFrame([st]).T.to_string(header=False))

    fs, lw = FS_FIG3, LW_FIG3
    apply_style(base=fs["ticks"], lw=lw)
    fig = plt.figure(figsize=(COLUMNWIDTH_MAIN / PT_PER_IN, COLUMNWIDTH_MAIN * H / W / PT_PER_IN))
    # A: axes frame at pixels x 218.5-1167.5, y 128.5-813.5 of the W x H figure
    ax = fig.add_axes([218.5 / W, 1 - 813.5 / H, 949 / W, 685 / H])
    sns.histplot(cos, kde=True, bins=50, stat="density", ax=ax, line_kws={"linewidth": 0.6},
                 edgecolor="black", linewidth=0.3)
    ax.axvline(cos.mean(), color="red", linestyle="--", linewidth=LW_LINES_FIG3, label=f"Mean: {cos.mean():.3f}")
    ax.axvline(np.median(cos), color="green", linestyle=":", linewidth=LW_LINES_FIG3,
               label=f"Median: {np.median(cos):.3f}")
    ax.set_xlim(XMIN - 0.00074, 1.0 + 0.00052)
    ax.set_xticks(np.round(np.arange(XMIN, 1.0 + 1e-9, 0.01), 2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_title("Reconstruction Cosine Similarity", fontsize=fs["title"], pad=3)
    ax.set_xlabel("Cosine Similarity", fontsize=fs["xlabel"], labelpad=2)
    ax.set_ylabel("Density", fontsize=fs["ylabel"], labelpad=1)
    ax.legend(loc="upper left", fontsize=fs["legend"], handlelength=2.2, borderpad=0.4).get_frame().set_linewidth(lw)
    ax.tick_params(labelsize=fs["ticks"], width=lw, length=2.5, pad=1.5)
    fig.text(3 / W, 1 - 25 / H, "A", fontsize=fs["panel"], fontweight="bold", va="top", ha="left")
    # B: axes frame at pixels x 1449.5-2125.5, y 128.5-833.5; PC1 ticks -1, 0, 1 at x 1570.5, 1775.5, 1981.5 and PC2
    # ticks -1 ... 1 at y 815.5 ... 195.5
    bx = fig.add_axes([1449.5 / W, 1 - 833.5 / H, 676 / W, 705 / H])
    sns.kdeplot(x=zt[:, 0], y=zt[:, 1], ax=bx, cmap="Blues", fill=True)
    sns.kdeplot(x=zp[:, 0], y=zp[:, 1], ax=bx, cmap="Reds", fill=True, alpha=0.5)
    bx.set_xlim(-1.0 - 121 / 205.5, 1.0 + 144 / 205.5)
    bx.set_ylim(-1.0 - 18 / 310.0, 1.0 + 67 / 310.0)
    bx.set_xticks([-1.0, 0.0, 1.0])
    bx.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    bx.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\u2212")))
    bx.set_title("Data Distribution", fontsize=fs["title"], pad=3)
    bx.set_xlabel("PC1", fontsize=fs["xlabel"], labelpad=2)
    bx.set_ylabel("PC2", fontsize=fs["ylabel"], labelpad=1)
    bx.tick_params(labelsize=fs["ticks"], width=lw, length=2.5, pad=1.5)
    for s in bx.spines.values():
        s.set_linewidth(lw)
    handles = [Line2D([], [], marker="o", color="blue", linestyle="", markersize=3.6, label="Original"),
               Line2D([], [], marker="x", color="red", linestyle="", markersize=3.6, markeredgewidth=1.0,
                      label="Recon")]
    bx.legend(handles=handles, loc="upper left", fontsize=fs["legend"], borderpad=0.4,
              handletextpad=0.4).get_frame().set_linewidth(lw)
    fig.text(1268 / W, 1 - 25 / H, "B", fontsize=fs["panel"], fontweight="bold", va="top", ha="left")
    savefig(fig, os.path.join(OUT, "fig3_recon"), dpi=W / (COLUMNWIDTH_MAIN / PT_PER_IN))


def plot_structure(d):
    fape, rm = d["fape"], d["rmsd"]
    pd.DataFrame({"path": d["paths"], "n_res": d["n_res"], "fape": fape, "rmsd": rm}).to_csv(
        os.path.join(OUT, "figS3_per_site.csv"), index=False)
    order = np.argsort(rm, kind="stable")
    n = len(rm)
    picks = {"Q1": order[min(int(n * 0.25), n - 1)], "Median": order[min(int(n * 0.5), n - 1)]}
    st = dict(n_sites=n, fape_mean=fape.mean(), fape_median=np.median(fape), rmsd_mean=rm.mean(),
              rmsd_median=np.median(rm), **{f"{k}_site": str(d["paths"][i]) for k, i in picks.items()},
              **{f"{k}_rmsd": rm[i] for k, i in picks.items()})
    pd.DataFrame([st]).to_csv(os.path.join(OUT, "figS3_stats.csv"), index=False)
    print(pd.DataFrame([st]).T.to_string(header=False))

    fs = FS_S3
    apply_style(base=fs["ticks"], lw=0.6)
    fig = plt.figure(figsize=(TEXTWIDTH_SUPP / PT_PER_IN, TEXTWIDTH_SUPP * 1845 / 2002 / PT_PER_IN))
    for j, (vals, title, xl) in enumerate([(fape, "FAPE Loss Distribution", "FAPE (Å)"),
                                           (rm, "Kabsch-RMSD Distribution", "RMSD (Å)")]):
        ax = fig.add_axes([0.047 + 0.493 * j, 0.602, 0.45, 0.342])
        sns.histplot(vals, kde=True, bins=30, ax=ax, edgecolor="black", linewidth=0.4, line_kws={"linewidth": 0.8})
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=0.7, label=f"Mean: {vals.mean():.2f}")
        ax.axvline(np.median(vals), color="green", linestyle=":", linewidth=0.7, label=f"Median: {np.median(vals):.2f}")
        ax.set_title(title, fontsize=fs["title"], pad=5)
        ax.set_xlabel(xl, fontsize=fs["ylabel"], labelpad=2)
        ax.set_ylabel("Count", fontsize=fs["ylabel"], labelpad=2)
        ax.tick_params(labelsize=fs["ticks"], width=0.6, length=2, pad=1.5)
        ax.legend(fontsize=fs["legend"], loc="upper right", handlelength=2.5)
        fig.text(0.02 + 0.493 * j, 0.962, "AB"[j], fontsize=fs["panel"], fontweight="bold", va="top")
    for j, key in enumerate(["Q1", "Median"]):
        i = picks[key]
        m = d["mask"][i]
        A, B = d["T_true"][i][m], d["T_pred"][i][m]
        with plt.rc_context({"grid.linewidth": 0.35, "axes.linewidth": 0.4}):
            ax = fig.add_axes([0.0 + 0.485 * j, -0.005, 0.5, 0.5], projection="3d")
        ax.set_box_aspect(None, zoom=0.95)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color="blue", s=4, label="Original", alpha=0.7, depthshade=False)
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color="red", s=4, marker="x", label="Recon", alpha=0.9,
                   linewidths=0.6, depthshade=False)
        for setter, lab in [(ax.set_xlabel, "X"), (ax.set_ylabel, "Y"), (ax.set_zlabel, "Z")]:
            setter(lab, fontsize=fs["ticks_3d"], labelpad=-3)
        ax.tick_params(labelsize=fs["ticks_3d"], pad=-1.5, width=0.4)
        fig.text(0.25 + 0.5 * j, 0.535, f"Structure Reconstruction {key}\nRMSD = {rm[i]:.3f} Å",
                 ha="center", va="top", fontsize=fs["title_3d"], linespacing=1.3)
        if j == 0:
            fig.text(0.02, 0.535, "C", fontsize=fs["panel"], fontweight="bold", va="top")
        else:
            ax.legend(fontsize=fs["legend"], loc="upper right", bbox_to_anchor=(0.99, 0.535),
                      bbox_transform=fig.transFigure, markerscale=1.5)
    savefig(fig, os.path.join(OUT, "figS3_struct"))


def set_counts(d, q_paths):
    """Sites and PDB entries of the reconstruction set, and how many evaluation sites it includes."""
    key = lambda p: "/".join(str(p).split("/")[-2:])  # noqa: E731
    sites = {key(p) for p in d["paths"]}
    r = dict(sites=len(sites), entries=len({k.split("/")[0] for k in sites}))
    print(f"reconstruction set: {r['sites']:,} sites from {r['entries']} PDB entries")
    if q_paths and os.path.exists(q_paths):
        q = [key(line.strip()) for line in open(q_paths) if line.strip()]
        r.update(evaluation_sites=len(q), evaluation_sites_included=sum(k in sites for k in q))
        print(f"  includes {r['evaluation_sites_included']} of the {len(q)} sites of the evaluation data")
    pd.DataFrame([r]).to_csv(os.path.join(OUT, "recon_set_counts.csv"), index=False)


def cmd_plot(a):
    d = np.load(require_path(a.recon, "--recon", "LENSEEK_OUT"))
    set_counts(d, a.q_paths)
    plot_fig3(d)
    plot_structure(d)


def cmd_rotation(a):
    t = pd.read_csv(require_path(a.table, "--table", "LENSEEK_EVAL_DIR"), sep="\t")
    cd, same = t.chamfer_to_unrotated, int((t.rank_of_top1 == 1).sum())
    r = dict(sites=int(t.site.nunique()), rotations=int(t.rotation.nunique()), searches=len(t), top1_unchanged=same,
             chamfer_median=float(cd.median()), chamfer_max=float(cd.max()))
    json.dump(r, open(os.path.join(OUT, "rotation.json"), "w"), indent=1)
    print(f"top-1 result unchanged in {same:,} of {len(t):,} searches ({t.site.nunique()} sites x "
          f"{t.rotation.nunique()} rotations)")
    print(f"Chamfer distance between a site and its rotated copy: median {cd.median():.0e}, maximum {cd.max():.0e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("site-size", help="Figure S1")
    p.add_argument("--bsdb-root", default=os.environ.get("LENSEEK_BSDB_ROOT"),
                   help="database structures (default $LENSEEK_BSDB_ROOT)")
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("schedule", help="Figure S2")
    p = sub.add_parser("recon", help="reconstruction pass -> recon.npz")
    p.add_argument("--ckpt", default=os.environ.get("LENSEEK_CKPT"), help="checkpoint (default $LENSEEK_CKPT)")
    p.add_argument("--dataset", default=os.environ.get("LENSEEK_RECON_DATA"),
                   help="featurised reconstruction set (default $LENSEEK_RECON_DATA)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("plot", help="Figures 3 and S3 from recon.npz")
    p.add_argument("--recon", default=os.path.join(OUT, "recon.npz"))
    p.add_argument("--q-paths", default=os.path.join(os.environ.get("LENSEEK_EVAL_DIR") or "", "q_paths.txt"),
                   help="evaluation query list, to count the evaluation sites in the set "
                        "(default $LENSEEK_EVAL_DIR/q_paths.txt)")
    p = sub.add_parser("rotation", help="rotated searches of scripts/evaluate.py --rotations")
    p.add_argument("--table", default=os.path.join(os.environ.get("LENSEEK_EVAL_DIR") or "", "rotation.tsv"),
                   help="rotation.tsv (default $LENSEEK_EVAL_DIR/rotation.tsv)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"site-size": cmd_site_size, "schedule": cmd_schedule, "recon": cmd_recon, "plot": cmd_plot,
     "rotation": cmd_rotation}[a.cmd](a)


if __name__ == "__main__":
    main()
