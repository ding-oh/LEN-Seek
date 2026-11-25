#!/usr/bin/env python
"""Ligand similarity of the retrieved binding sites: main-text Figure 5, Supplementary Table S3, the whole-database
ligand metrics and the ligand strata (Supplementary Table S10 A).

Ligand of a site = residue name of the first HETATM line of its ligand file (ligand_<n>_<chains>.pdb next to
ligand_pocket_<n>_<chains>.pdb). SMILES come from the wwPDB CCD snapshot (ccd_smiles.tsv of asset reference_data);
fingerprints are Morgan radius 2, 2,048 bits, compared by the Dice similarity. Codes without SMILES and codes RDKit
cannot parse are written to a *_dropped_codes.tsv file.

Subcommands
  kde     Figure 5 and Table S3. The top-100 results of each primary query by Chamfer distance (ties by database index)
          are split by LEN-Seek (similar: distance <= --threshold, 42.5) and, for the same pairs, by ProBiS (similar:
          Z >= 2; pairs without an alignment are dissimilar, failed comparisons are dropped). Per group: the Dice
          distribution (kernel density estimate, clipped to [0, 1]), the share of pairs with Dice >= 0.46 and >= 0.6
          (cluster-bootstrap 95 % intervals, 1,000 resamples), and the Table S3 bins [a, a + 0.1) (last bin closed).
  grid    Whole-database ligand retrieval of one model (the ligand columns of Supplementary Table S13 B for LEN-Seek,
          DeeplyTough, Folddisco and the sequence-identity baseline; the ligand values of Supplementary Table S7 B and
          Table S14 for the seeds and every variant; the RE@1% of the Folddisco runs in the caption of Table S13).
          Every database site with a ligand fingerprint is ranked for every primary query by
            lenseek_raw  -d(q, s), the Chamfer distance
            dt_raw       -d_DT(q, s), DeeplyTough (sites with a descriptor)
            seqid        identity of the main chains (MMseqs2 search of the evaluation chains, 0 below 25 %)
            <name>_raw   -d of a further method (--method NAME=DIR, DIR/dist.npy: the Folddisco runs of
                         scripts/folddisco.py assemble; every site is scored)
          Labels: Dice >= 0.46 (main), >= 0.60, >= 0.80 and the identical CCD code. Per query: ROC-AUC (mid-ranks),
          RE@1% (true-positive rate at a false-positive rate of 1 % divided by 0.01, NA below 5 positives), EF@1%;
          variants with Dice >= 0.46: homology-unrelated database sites only (identity < 30 % and another Pfam clan,
          both main chains annotated), ambiguous sites removed (summarised over non-ambiguous queries), sites weighted
          by 1 / size of their (80 % chain cluster, CCD) group, and, for LEN-Seek, the size-matched ranking of the
          sites within max(2, 10 %) residues of the query size. Queries with a fingerprint and at least one positive
          and one negative site are summarised by entry-weighted means (sites of a PDB entry averaged first), query
          means and cluster-weighted means with 95 % intervals from 2,000 bootstrap resamples of the 50 % sequence
          clusters (seed 0), for all queries and by ligand base rate (< 1 %, >= 1 % positives) and site size
          (10-19, 20-29, >= 30 residues). With --db-entries F, also the ROC-AUC of lenseek_raw ranked among the
          database sites of the PDB entries listed in F only (lenseek_raw__auc_db_entries;
          data/strict30_exclude_entries.txt: the sites excluded from the training of the model trained without
          homologous entries, Supplementary Section S5; the paired difference between two models is made by
          ablations.py table).
  strata  Supplementary Table S10 A (ligand ROC-AUC): the grid values of LEN-Seek by query identity to the
          training proteins (< 30 %, >= 30 %), the homology-unrelated database restriction, base rate, site size, the
          maximum Foldseek TM-score to a training structure and the novelty of the query ligand, beside DeeplyTough
          and the sequence-identity baseline; Folddisco (the grid's folddisco_raw, when present) in the rows of all
          queries and of the identity strata. Same summary as grid (2,000 resamples, the count of the printed
          intervals).

Inputs: see `python analysis/run.py -h`; read here: --method and --db-entries (grid: further comparison methods, the
entry list) and, for strata, --protein-strata and --query-strata (identity, TM-score and ligand strata of
scripts/site_annotation.py identity-strata and strata, default $LENSEEK_OUT/site_annotation/protein_strata.tsv and
query_strata.csv).

Outputs ($LENSEEK_OUT/fig4/<model>/, default ./out/fig4/<model>/; <model> = eval dir name without 'out_')
  kde     fig4_ligand_dice.{png,pdf}, fig4_dice_points.csv.gz, fig4_dice_summary.csv, tableS3_dice_bins.csv,
          fig4_dropped_codes.tsv
  grid    ligand_grid_per_query.csv, ligand_grid_summary.csv (stratum x column -> entry / query / cluster means and
          intervals), ligand_grid_counts.json
  strata  tableS12A_strata_long.csv, tableS12A_strata.csv

usage: python analysis/fig4_ligand.py kde
       python analysis/fig4_ligand.py grid [--eval-dir DIR] [--method folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter ...] \\
           [--db-entries data/strict30_exclude_entries.txt]
       python analysis/fig4_ligand.py strata
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rdkit import DataStructs, RDLogger  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ANNOT, OUT_ROOT, PROTEIN_STRATA, QUERY_STRATA, SNAP_META,  # noqa: E402
                    EvalOutputs, Homology, attach_pairs, auc_from_ranks, bootstrap_proteins, chain_name, cluster_of,
                    die, enrich, entry_weighted, fingerprints, identity_strata, ligand_code, load_cache, load_pairs,
                    mw_auc, probis_truth, protein_of, require_path, site_key, site_tables, structure_ligand_strata,
                    topk_rows)

RDLogger.DisableLog("rdApp.*")
LABELS = {"d046": 0.46, "d060": 0.60, "d080": 0.80}
RE_MIN_POS = 5
SIZE = [(10, 20, "10-19"), (20, 30, "20-29"), (30, 66, ">=30")]
BASE = [(0.0, 0.01, "rare <1%"), (0.01, 1.01, "common >=1%")]
SCORES = ["lenseek_raw", "dt_raw", "seqid"]
ID_GE30 = ["30-40", "40-50", "50-80", "80-95", ">=95"]
G = {}


# ---------------------------------------------------------------- helpers
def check_pairs(pairs, ev, rows, tol=1e-3):
    """TSV order, distances and ranks must agree with dist.npy and the tie-broken ranks."""
    qi = {k: i for i, k in enumerate(ev.q_keys)}
    ti = {k: i for i, k in enumerate(ev.db_keys)}
    p = pairs[pairs.query_key.isin({ev.q_keys[i] for i in ev.sel})]
    if (~p.target_key.isin(ti)).any():
        die(f"{int((~p.target_key.isin(ti)).sum())} TSV targets are not in {ev.dir}/db_paths.txt")
    qa = p.query_key.map(qi).to_numpy()
    ta = p.target_key.map(ti).to_numpy()
    if "query_idx" in p and (p.query_idx.to_numpy() != qa).any():
        die("TSV query_idx does not follow q_paths.txt order (TSV made from another eval dir?)")
    if "db_idx" in p and (p.db_idx.to_numpy() != ta).any():
        die("TSV db_idx does not follow db_paths.txt order (TSV made from another eval dir?)")
    if "dist" in p:
        dn = ev.dist[qa, ta].astype(np.float64)
        dd = np.abs(p.dist.to_numpy(dtype=float) - dn)
        if (dd > tol * (1 + np.abs(dn))).any():
            die(f"TSV dist disagrees with dist.npy (max |diff| {dd.max():.3g}); the TSV belongs to another model or run")
    if "rank" in p:
        m = rows.merge(p[["query_key", "target_key", "rank"]], on=["query_key", "target_key"], suffixes=("", "_tsv"))
        if (m["rank"] != m["rank_tsv"]).any():
            die(f"TSV rank disagrees with the tie-broken rank for {int((m['rank'] != m['rank_tsv']).sum())} pairs")


def weighted_auc(score, lab, w):
    """ROC-AUC with site weights w (ties count one half)."""
    wp, wn = w[lab].sum(), w[~lab].sum()
    if wp == 0 or wn == 0:
        return np.nan
    o = np.argsort(score, kind="mergesort")
    s, l, ww = score[o], lab[o], w[o]
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    gneg = np.bincount(gid, np.where(~l, ww, 0.0))
    gpos = np.bincount(gid, np.where(l, ww, 0.0))
    below = np.cumsum(gneg) - gneg
    return float((gpos * (below + 0.5 * gneg)).sum() / (wp * wn))


def write_dropped(path, ids, cache, fps):
    """Codes not in the CCD snapshot, in it without SMILES, or with SMILES that RDKit cannot parse."""
    ids = [i for i in ids if i]
    lists = [("not_in_ccd", sorted(i for i in ids if i not in cache)),
             ("no_smiles_in_ccd", sorted(i for i in ids if i in cache and not cache[i])),
             ("rdkit_failed", sorted(i for i in ids if cache.get(i) and fps.get(i) is None))]
    meta = json.load(open(SNAP_META)) if os.path.exists(SNAP_META) else {}
    with open(path, "w") as f:
        f.write(f"# CCD snapshot: {meta.get('url', '?')} (server {meta.get('server_last_modified', '?')}, "
                f"md5 {meta.get('md5', '?')})\n# reason\tcomp_id\tn_sites\n")
        for reason, lst in lists:
            for i in lst:
                f.write(f"{reason}\t{i}\t\n")
    return [i for _, lst in lists for i in lst]


# ---------------------------------------------------------------- outputs
def out_dir(tag):
    """Output folder $LENSEEK_OUT/fig4/<tag>/ (created)."""
    d = os.path.join(OUT_ROOT, "fig4", tag)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- kde: Figure 5 and Table S3
GROUPS = ["ProBiS True", "ProBiS False", "LEN-Seek True", "LEN-Seek False"]
PALETTE = {"ProBiS True": "blue", "ProBiS False": "skyblue", "LEN-Seek True": "red", "LEN-Seek False": "salmon"}
FS = dict(title=7.1, xlabel=7.1, ylabel=5.8, ticks=6.7, legend=5.6)    # print sizes (pt) at \columnwidth
FIGSIZE = (254.65216 / 72.27, 254.65216 * 1111 / 1676 / 72.27)          # \columnwidth of the main text, inches


def pct_stat(S):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"pct": 100 * S["hit"] / S["n"]}


def cmd_kde(a):
    require_path(a.query_root, "--query-root", "LENSEEK_QUERY_ROOT")
    require_path(a.db_root, "--db-root", "LENSEEK_BSDB_ROOT")
    ev = EvalOutputs(a.eval_dir)
    od = out_dir(ev.tag)
    rows = topk_rows(ev, a.topk)
    pairs = load_pairs(a.pairs)
    check_pairs(pairs, ev, rows)
    m = attach_pairs(rows, pairs)
    y, counts = probis_truth(m), m.probis_class.value_counts().to_dict()
    keep = np.isfinite(y)
    pools = [rows.assign(Method="LEN-Seek", score=rows.cd,
                         Group=np.where(rows.cd <= a.threshold, "LEN-Seek True", "LEN-Seek False")),
             m[keep].assign(Method="ProBiS", score=m.probis_z[keep],
                            Group=np.where(y[keep] == 1, "ProBiS True", "ProBiS False"))]
    print(f"[ProBiS pool] top-{a.topk} rows {len(m)}: classes {counts}; kept {int(keep.sum())}")
    pts = pd.concat([x[["query_key", "protein", "target_key", "Method", "Group", "score"]] for x in pools],
                    ignore_index=True)
    qlig = {k: ligand_code(a.query_root, k) for k in pts.query_key.unique()}
    tlig = {k: ligand_code(a.db_root, k) for k in pts.target_key.unique()}
    cache = load_cache()
    ids = {v for v in list(qlig.values()) + list(tlig.values()) if v}
    fps = fingerprints(ids, cache)
    dropped = write_dropped(os.path.join(od, "fig4_dropped_codes.tsv"), ids, cache, fps)
    pts["query_ligand"] = pts.query_key.map(qlig)
    pts["target_ligand"] = pts.target_key.map(tlig)
    qfp = pts.query_ligand.map(lambda x: fps.get(x) if x else None)
    tfp = pts.target_ligand.map(lambda x: fps.get(x) if x else None)
    ok = qfp.notna() & tfp.notna()
    print(f"pairs with both fingerprints: {ok.sum()}/{len(pts)}; ligand codes without a fingerprint: {len(dropped)}")
    pts = pts[ok].copy()
    pts["Dice"] = [DataStructs.DiceSimilarity(x, z) for x, z in zip(qfp[ok], tfp[ok])]
    pts.to_csv(os.path.join(od, "fig4_dice_points.csv.gz"), index=False)

    summ, bins = [], []
    edges = np.round(np.arange(0, 1.01, 0.1), 2)
    for g in GROUPS:
        sub = pts[pts.Group == g]
        d = sub.Dice.to_numpy()
        r = dict(Group=g, n=len(d), n_queries=sub.query_key.nunique(), n_proteins=sub.protein.nunique())
        for cut, name in [(a.dice_cut, "pct_ge_cut"), (0.6, "pct_ge_0p6")]:
            r[name] = 100 * (d >= cut).mean()
            pq = pd.DataFrame({"query_key": sub.query_key.to_numpy(), "protein": sub.protein.to_numpy(), "n": 1,
                               "hit": (d >= cut).astype(int)})
            r[name + "_ci_lo"], r[name + "_ci_hi"] = bootstrap_proteins(pq, pct_stat, n=a.n_boot)["pct"]
        summ.append(r)
        h = np.histogram(d, bins=edges)[0]
        bins.append(dict(Group=g, n=len(d), **{f"{edges[i]:.1f}-{edges[i + 1]:.1f}": 100 * h[i] / len(d)
                                                for i in range(10)}))
    summ, bins = pd.DataFrame(summ), pd.DataFrame(bins)
    s = bins[[c for c in bins.columns if c[0].isdigit()]].sum(1)
    if ((s - 100).abs() > 0.05).any():
        die("Table S3 rows do not sum to 100 %")
    summ.to_csv(os.path.join(od, "fig4_dice_summary.csv"), index=False)
    bins.to_csv(os.path.join(od, "tableS3_dice_bins.csv"), index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(bins.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    b = FS["ticks"]
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": b, "axes.titlesize": b, "axes.labelsize": b,
                                "xtick.labelsize": b, "ytick.labelsize": b, "legend.fontsize": b, "axes.linewidth": 0.6,
                                "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                                "ytick.major.size": 2.5, "lines.linewidth": 1.0, "figure.dpi": 100, "pdf.fonttype": 42})
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.08, 0.135, 0.88, 0.77])
    for g in GROUPS:
        sns.kdeplot(x=pts.Dice[pts.Group == g], fill=True, alpha=0.1, linewidth=1.1, color=PALETTE[g],
                    clip=(0.0, 1.0), ax=ax)
    ax.axvline(a.dice_cut, color="black", linestyle="--", linewidth=0.7)
    tt = ax.set_title("KDE Estimation of Ligand Similarity - Binding Site Similarity Distribution",
                      fontsize=FS["title"], pad=4)
    ax.set_xlabel("Dice Similarity", fontsize=FS["xlabel"], labelpad=1)
    ax.set_ylabel("Estimated density", fontsize=FS["ylabel"], labelpad=1)
    ax.tick_params(labelsize=FS["ticks"], length=0, pad=1)
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    handles = [Rectangle((0, 0), 1, 1, facecolor=matplotlib.colors.to_rgba(PALETTE[g], 0.15), edgecolor=PALETTE[g],
                         linewidth=1.6) for g in GROUPS]
    leg = ax.legend(handles, GROUPS, loc="upper center", fontsize=FS["legend"], frameon=True, fancybox=True,
                    edgecolor="#999999", handlelength=2.2, handleheight=0.9, borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(1.0)
    fig.canvas.draw()
    w = tt.get_window_extent(fig.canvas.get_renderer()).width          # shrink the title to the figure width
    if w > 0.98 * fig.bbox.width:
        tt.set_fontsize(tt.get_fontsize() * 0.98 * fig.bbox.width / w)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(od, f"fig4_ligand_dice.{ext}"), dpi=300)
    plt.close(fig)
    print("->", od)


# ---------------------------------------------------------------- grid: whole-database ligand metrics
def per_query(qi):
    g = G
    code = g["q_code"][qi]
    r = dict(q_idx=int(qi), query_key=g["q_keys"][qi], protein=protein_of(g["q_keys"][qi]), query_ligand=code,
             size=int(g["q_size"][qi]), query_ambiguous=int(g["q_amb"][qi]), has_query_fp=bool(code in g["dice"]))
    if not r["has_query_fp"]:
        return r
    d = np.asarray(g["ls"][qi], dtype=np.float64)
    dd = np.asarray(g["dt"][qi], dtype=np.float64)
    dice = np.full(len(d), -1.0)
    dice[g["has"]] = g["dice"][code][g["db_col"][g["has"]]]
    labs = {k: dice >= c for k, c in LABELS.items()}
    labs["same_ccd"] = g["db_code"] == code
    lab46 = labs["d046"]
    ident = g["H"].identity(qi)
    unrel, q_pfam = g["H"].unrelated(qi, ident)
    r["query_has_pfam"] = int(q_pfam)
    sc = {"lenseek_raw": -d, "dt_raw": -dd, "seqid": ident}
    base_set = {"lenseek": g["has"], "dt": g["has"] & g["dt_ok"], "seqid": g["has"]}
    for x, D in g["extra"].items():
        sc[f"{x}_raw"], base_set[x] = -np.asarray(D[qi], dtype=np.float64), g["has"]
    for name in SCORES + [f"{x}_raw" for x in g["extra"]]:
        S = base_set[name.split("_")[0]] & np.isfinite(sc[name])
        s = sc[name]
        pfx = name.split("_")[0]
        r[f"{pfx}_n_db"] = int(S.sum())
        for k, L in labs.items():
            r[f"{pfx}_n_pos_{k}"] = int(L[S].sum())
        rk = rankdata(s[S])
        for k, L in labs.items():
            r[f"{name}__auc" if k == "d046" else f"{name}__auc_{k}"] = auc_from_ranks(rk, L[S])
        if name != "seqid":
            re1, ef1, _ = enrich(s[S], lab46[S])
            r[f"{name}__re1"] = re1 if int(lab46[S].sum()) >= RE_MIN_POS else np.nan
            r[f"{name}__ef1"] = ef1
        r[f"{name}__auc_unrel"] = mw_auc(s[S & unrel], lab46[S & unrel]) if q_pfam else np.nan
        r[f"{name}__auc_noamb"] = mw_auc(s[S & ~g["amb"]], lab46[S & ~g["amb"]])
        r[f"{name}__auc_dupw"] = weighted_auc(s[S], lab46[S], g["w_dup"][S])
        if name == "lenseek_raw":
            band = S & (np.abs(g["db_size"] - g["q_size"][qi]) <= max(2, 0.1 * g["q_size"][qi]))
            r[f"{name}__auc_sizematched"] = mw_auc(s[band], lab46[band])
    S = base_set["lenseek"] & np.isfinite(sc["lenseek_raw"])
    if g["entries"] is not None:
        E = S & g["entries"]
        r["db_entries_n_db"], r["db_entries_n_pos"] = int(E.sum()), int(lab46[E].sum())
        r["lenseek_raw__auc_db_entries"] = mw_auc(sc["lenseek_raw"][E], lab46[E])
    r["n_db"] = r["lenseek_n_db"]
    r["n_pos"] = r["lenseek_n_pos_d046"]
    r["base_rate"] = r["n_pos"] / max(1, r["n_db"])
    return r


def query_strata(pq):
    yield "all", np.ones(len(pq), bool)
    for lo, hi, name in BASE:
        yield f"base rate {name}", ((pq.base_rate >= lo) & (pq.base_rate < hi)).to_numpy()
    for lo, hi, name in SIZE:
        yield f"size {name}", ((pq["size"] >= lo) & (pq["size"] < hi)).to_numpy()


def summarize(pq, cols, n_boot, strata=True):
    out = []
    for stratum, sel in (query_strata(pq) if strata else [("all", np.ones(len(pq), bool))]):
        d = pq[sel]
        if not len(d):
            continue
        res = entry_weighted(d.assign(cluster=d.cluster50), cols, n_boot=n_boot)
        out += [dict(stratum=stratum, column=c, **res[c]) for c in cols if c in res]
    return pd.DataFrame(out)


def usable_queries(pq):
    """Queries with a ligand fingerprint, at least one ligand-similar and one other database site."""
    return pq[pq.has_query_fp & (pq.n_pos > 0) & (pq.n_pos < pq.n_db)].copy()


def cmd_grid(a):
    require_path(a.dt_dir, "--dt-dir", "LENSEEK_DT_DIR")
    ev = EvalOutputs(a.eval_dir)
    od = out_dir(ev.tag)
    if [site_key(p) for p in open(os.path.join(a.dt_dir, "db_paths.txt")).read().split()] != ev.db_keys:
        die("DeeplyTough DB order differs from the LEN-Seek DB order")
    qs, ds = site_tables()
    if list(ds.key) != ev.db_keys or list(qs.key) != ev.q_keys:
        die("site annotation tables are not in the order of the evaluation outputs")
    cache = load_cache()
    db_code = ds.ligand.to_numpy(dtype=object)
    q_code = qs.ligand.to_numpy(dtype=object)
    ids = {c for c in list(db_code) + list(q_code) if c}
    fps = fingerprints(ids, cache)
    uniq = sorted(i for i in ids if fps.get(i) is not None)
    col = {c: j for j, c in enumerate(uniq)}
    db_col = np.array([col.get(c, -1) for c in db_code])
    has = db_col >= 0
    entries = None
    if a.db_entries:
        listed = {l.strip().lower() for l in open(require_path(a.db_entries, "--db-entries", "")) if l.strip()}
        entries = np.array([protein_of(k) in listed for k in ev.db_keys])
    dice = {c: np.array(DataStructs.BulkDiceSimilarity(fps[c], [fps[u] for u in uniq]), dtype=np.float64)
            for c in set(q_code[ev.sel]) if c and fps.get(c) is not None}
    cl80 = pd.read_csv(os.path.join(ANNOT, "db80_cluster.tsv"), sep="\t", header=None, names=["rep", "mem"])
    rep = dict(zip(cl80.mem, cl80.rep))
    mc = [chain_name(k, c) for k, c in zip(ds.key, ds.main_chain)]
    grp = pd.Series([f"{rep.get(c, c)}|{l}" for c, l in zip(mc, db_code)])
    w_dup = (1.0 / grp.map(grp.value_counts())).to_numpy(dtype=np.float64)
    H = Homology(ev.q_keys, ev.db_keys)
    dt_ok = np.isfinite(np.load(os.path.join(a.dt_dir, "db_desc.npy"), mmap_mode="r")).all(1)
    extra = {}
    for s in a.method:
        x, d = s.split("=", 1)
        if not x.isalnum() or x in ("lenseek", "dt", "seqid") or x in extra:
            die(f"--method {x}: give a new name of letters and digits")
        if [site_key(p) for p in open(os.path.join(d, "db_paths.txt")).read().split()] != ev.db_keys:
            die(f"{d}: DB order differs from the LEN-Seek DB order")
        extra[x] = np.load(os.path.join(d, "dist.npy"), mmap_mode="r")
    G.update(extra=extra, q_code=q_code, q_keys=ev.q_keys, q_size=ev.sizes("q"), db_size=ev.sizes("db"),
             q_amb=qs.ambiguous.to_numpy(), dice=dice, ls=ev.dist,
             dt=np.load(os.path.join(a.dt_dir, "dist.npy"), mmap_mode="r"),
             has=has, db_col=db_col, db_code=db_code, H=H, w_dup=w_dup, amb=ds.ambiguous.to_numpy().astype(bool),
             dt_ok=dt_ok, entries=entries)
    with Pool(a.workers) as p:
        pq = pd.DataFrame(p.map(per_query, list(ev.sel), chunksize=4))
    pq["cluster50"] = pq.query_key.map(cluster_of)
    pq.to_csv(os.path.join(od, "ligand_grid_per_query.csv"), index=False)
    usable = usable_queries(pq)
    counts = dict(n_queries=int(len(pq)), no_query_fingerprint=int((~pq.has_query_fp).sum()),
                  no_positive=int((pq.has_query_fp & (pq.n_pos.fillna(0) == 0)).sum()), used=int(len(usable)),
                  re1_na_lt5_pos=int((usable.n_pos < RE_MIN_POS).sum()),
                  db_sites=len(ev.db_keys), db_without_fingerprint=int((~has).sum()),
                  db_without_dt_descriptor=int((~dt_ok).sum()), db_ambiguous=int(G["amb"].sum()),
                  queries_ambiguous=int(usable.query_ambiguous.sum()),
                  queries_with_pfam=int(usable.query_has_pfam.sum()),
                  db_sites_with_pfam_main_chain=int(H.db_has_pfam.sum()),
                  dup_groups=int(grp.nunique()), dup_effective_db=float(w_dup.sum()),
                  queries_without_fp=sorted(pq.loc[~pq.has_query_fp, "query_key"]),
                  queries_without_positive=sorted(pq.loc[pq.has_query_fp & (pq.n_pos.fillna(0) == 0), "query_key"]))
    if entries is not None:
        counts.update(db_entries_sites=int(entries.sum()), db_entries_with_fingerprint=int((entries & has).sum()))
    json.dump(counts, open(os.path.join(od, "ligand_grid_counts.json"), "w"), indent=1)
    print({k: v for k, v in counts.items() if not isinstance(v, list)})
    metric_cols = [c for c in usable.columns if "__" in c]
    noamb = [c for c in metric_cols if c.endswith("__auc_noamb")]
    summ = pd.concat([summarize(usable, [c for c in metric_cols if c not in noamb], a.n_boot),
                      summarize(usable[usable.query_ambiguous == 0], noamb, a.n_boot, strata=False)],
                     ignore_index=True)
    summ["score"] = summ.column.str.split("__").str[0]
    summ["metric"] = summ.column.str.split("__").str[1]
    summ["label"] = np.where(summ.score == "seqid", "sequence baseline", "reported")
    summ.to_csv(os.path.join(od, "ligand_grid_summary.csv"), index=False)
    show = summ[summ.stratum == "all"].set_index("column")
    print(show[["n_queries", "n_entries", "entry_mean", "entry_ci_lo", "entry_ci_hi", "query_mean", "cluster_mean"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    if "lenseek_raw__auc_db_entries" in show.index:
        x = show.loc["lenseek_raw__auc_db_entries"]
        print(f"ligand ROC-AUC among the database sites of {os.path.basename(a.db_entries)}: {x.entry_mean:.3f} "
              f"({x.n_queries} queries)")
    print("->", od)


# ---------------------------------------------------------------- strata: Supplementary Table S10 A
STRATA_SCORES = [("LEN-Seek", "lenseek_raw"), ("DeeplyTough", "dt_raw"), ("Folddisco", "folddisco_raw"),
                 ("Seq. identity", "seqid")]
ID_ROWS = ["All queries", "Query identity to training <30%", "Query identity to training >=30%"]


def cmd_strata(a):
    ev = EvalOutputs(a.eval_dir)
    od = out_dir(ev.tag)
    f = os.path.join(od, "ligand_grid_per_query.csv")
    if not os.path.exists(f):
        die(f'{f} missing: run "fig4_ligand.py grid" first')
    u = usable_queries(pd.read_csv(f))
    u["id_stratum"] = u.protein.map(identity_strata(a.protein_strata)).fillna("missing")
    scores = [(name, s) for name, s in STRATA_SCORES if f"{s}__auc" in u.columns]
    rows = [(ID_ROWS[0], np.ones(len(u), bool), "__auc"),
            (ID_ROWS[1], (u.id_stratum == "<30").to_numpy(), "__auc"),
            (ID_ROWS[2], u.id_stratum.isin(ID_GE30).to_numpy(), "__auc"),
            ("Database <30% identity and other Pfam clan", np.ones(len(u), bool), "__auc_unrel")]
    rows += [(f'Ligand base rate {"<1%" if lo == 0 else ">=1%"}',
              ((u.base_rate >= lo) & (u.base_rate < hi)).to_numpy(), "__auc") for lo, hi, _ in BASE]
    rows += [(f"Site size {name} residues", ((u["size"] >= lo) & (u["size"] < hi)).to_numpy(), "__auc")
             for lo, hi, name in SIZE]
    no_seqid = set()
    u = u.merge(structure_ligand_strata(a.query_strata), on="q_idx", how="left", validate="1:1")
    for lab, col, val in [("Maximum TM-score to training <0.5", "structure_stratum", "TM < 0.5"),
                          ("Maximum TM-score 0.5-0.8", "structure_stratum", "TM 0.5-0.8"),
                          ("Maximum TM-score >=0.8", "structure_stratum", "TM >= 0.8"),
                          ("Query ligand: same component in training", "ligand_stratum", "same_ccd"),
                          ("Query ligand: new component, Dice >=0.46 to a training ligand", "ligand_stratum",
                           "new_ccd_similar")]:
        rows.append((lab, (u[col] == val).to_numpy(), "__auc"))
        no_seqid.add(lab)
    long, table = [], []
    for lab, sel, suffix in rows:
        d = u[sel]
        res = entry_weighted(d.assign(cluster=d.cluster50), [s + suffix for _, s in scores], n_boot=a.n_boot)
        long += [dict(row=lab, column=c, **v) for c, v in res.items()]
        r0 = res[f"lenseek_raw{suffix}"]
        t = {"Stratum": lab, "Queries / entries": f"{r0['n_queries']} / {r0['n_entries']}",
             "LEN-Seek": f"{r0['entry_mean']:.3f} [{r0['entry_ci_lo']:.3f}, {r0['entry_ci_hi']:.3f}]"}
        for name, s in scores[1:]:
            t[name] = "--" if (s == "seqid" and lab in no_seqid) or (s == "folddisco_raw" and lab not in ID_ROWS) \
                else f"{res[s + suffix]['entry_mean']:.3f}"
        table.append(t)
    pd.DataFrame(long).to_csv(os.path.join(od, "tableS12A_strata_long.csv"), index=False)
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(od, "tableS12A_strata.csv"), index=False)
    print(table.to_string(index=False))
    print("->", od)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kde", help="Figure 5 and Table S3")
    p.add_argument("--pairs", default=os.environ.get("LENSEEK_PAIRS"), help="scored top-100 pairs; $LENSEEK_PAIRS")
    p.add_argument("--threshold", type=float, default=42.5, help="Chamfer distance called similar (<=)")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--dice-cut", type=float, default=0.46)
    p.add_argument("--n-boot", type=int, default=1000, help="resamples of the intervals (default 1,000)")
    p.add_argument("--query-root", default=os.environ.get("LENSEEK_QUERY_ROOT"), help="$LENSEEK_QUERY_ROOT")
    p.add_argument("--db-root", default=os.environ.get("LENSEEK_BSDB_ROOT"), help="$LENSEEK_BSDB_ROOT")
    p = sub.add_parser("grid", help="whole-database ligand metrics of one model")
    p.add_argument("--dt-dir", default=os.environ.get("LENSEEK_DT_DIR"), help="DeeplyTough outputs; $LENSEEK_DT_DIR")
    p.add_argument("--n-boot", type=int, default=2000, help="resamples of the intervals (default 2,000)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--method", action="append", default=[],
                   help="further comparison method: name=<dir with dist.npy, db_paths.txt> (smaller = more similar), "
                        "e.g. folddisco=$LENSEEK_FOLDDISCO_DIR/prefilter")
    p.add_argument("--db-entries", default=None,
                   help="PDB entries whose database sites are also ranked alone (data/strict30_exclude_entries.txt)")
    p = sub.add_parser("strata", help="Supplementary Table S10 A (reads the grid output)")
    p.add_argument("--protein-strata", default=PROTEIN_STRATA,
                   help="identity strata (scripts/site_annotation.py identity-strata)")
    p.add_argument("--query-strata", default=QUERY_STRATA,
                   help="TM-score and ligand strata (scripts/site_annotation.py strata)")
    p.add_argument("--n-boot", type=int, default=2000, help="resamples of the intervals (default 2,000)")
    for p in sub.choices.values():
        p.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"), help="$LENSEEK_EVAL_DIR")
    a = ap.parse_args()
    {"kde": cmd_kde, "grid": cmd_grid, "strata": cmd_strata}[a.cmd](a)


if __name__ == "__main__":
    main()
