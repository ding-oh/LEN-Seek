"""lenseek-search: find the database binding sites closest to query binding sites.

    lenseek-search data/example/queries --embeddings data/example/embeddings --db data/example/db/example_db.npz --top-k 10

Each query is a pocket file (ATOM records of the pocket residues) next to its protein file (protein.pdb or
<entry>_protein.pdb); a directory argument expands to every pocket file in it (or in its <entry>/ subdirectories).
Node features are read from per-chain Ankh embeddings (<entry>_chain_<C>.npy) in --embeddings; --embed computes the
missing ones with Ankh-base (CPU works for a few proteins; the weights are downloaded on first use).

Without --db the released database of the model is fetched (lenseek/configs/assets.yaml). The default standard
edition is the paper edition with the rows of the sites whose N, CA or C coordinates differ between the two PDB readers
re-encoded from the coordinates of the strict reader; --edition paper uses the database exactly as built, and
--edition paper --parser legacy reproduces the rankings of the paper.

--parser strict (default) reads standard PDB columns and insertion codes and is the reader for new data; --parser
legacy is the fixed-column reader the released features and database were built with. The two readers can read
different coordinates from files in the original BsitePDB column layout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from ..assets import AssetError, fetch
from ..checkpoint import load_model
from ..encode import Encoder
from ..featurize import FeaturizeError, default_entry_name, featurize_site
from ..io.pdb import PDBParseError
from ..latentdb import LatentDB
from ..resources import model_registry
from ..search import chamfer_matrix, topk
from .common import check_device, die, expand_sites, resolve_checkpoint, set_threads


def build_parser():
    ap = argparse.ArgumentParser(prog="lenseek-search", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="pocket files and/or directories")
    ap.add_argument("--protein", help="protein file for every pocket (default: protein.pdb next to each pocket)")
    ap.add_argument("--pattern", default="*pocket*.pdb", help="pocket file pattern inside directories")
    ap.add_argument("--embeddings", default=None, help="directory of <entry>_chain_<C>.npy files "
                    "(default: $LENSEEK_HOME/embeddings)")
    ap.add_argument("--embed", action="store_true", help="compute missing embeddings with Ankh-base")
    ap.add_argument("--model", default=None, help="released model name (lenseek/configs/models.yaml); default main_s0")
    ap.add_argument("--checkpoint", default=None, help="checkpoint file (overrides --model)")
    ap.add_argument("--db", default=None, help="latent database (directory or .npz); default: released DB of the model")
    ap.add_argument("--edition", choices=["standard", "paper"], default="standard",
                    help="edition of the released DB (default: standard; ignored with --db)")
    ap.add_argument("--patch", default=None, help="extra patch (.npz) to overlay on the database")
    ap.add_argument("--parser", choices=["strict", "legacy"], default="strict",
                    help="PDB reader: strict (default, standard PDB columns and insertion codes) or legacy (the "
                         "fixed-column reader the released features and database were built with)")
    ap.add_argument("--top-k", type=int, default=100, help="hits reported per query (default: 100)")
    ap.add_argument("--out", default=None, help="TSV of hits (default: stdout)")
    ap.add_argument("--save-latents", default=None, help="write the query latents to this .npz")
    ap.add_argument("--device", default="cpu", help="torch device for embedding and encoding, e.g. cuda (default: cpu)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    ap.add_argument("--batch", type=int, default=32, help="sites encoded per batch (default: 32)")
    return ap


def open_db(args, model_name):
    if args.db:
        db = LatentDB.load(args.db)
        if args.patch:
            db.apply_patch(args.patch)
        return db
    entry = model_registry().get(model_name, {}) if model_name else {}
    db_asset = entry.get("db_asset")
    if not db_asset:
        die(f"no released database for model {model_name or '(unregistered checkpoint)'}; pass --db")
    try:
        db = LatentDB.load(str(fetch(db_asset)))
        if args.edition == "standard" and entry.get("db_patch_asset"):
            db.apply_patch(str(fetch(entry["db_patch_asset"])))
    except AssetError as e:
        die(str(e))
    if args.patch:
        db.apply_patch(args.patch)
    return db


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_device(args.device)
    set_threads(args.threads)
    t0 = time.time()
    pairs = expand_sites(args.inputs, args.protein, args.pattern)
    if not pairs:
        die("no pocket files found")
    emb_dir = args.embeddings or os.path.join(os.environ.get("LENSEEK_HOME", os.path.expanduser("~/.cache/lenseek")),
                                              "embeddings")
    ckpt = resolve_checkpoint(args.checkpoint, args.model)
    model, cfg, info = load_model(ckpt, device=args.device)
    db = open_db(args, info.name)   # before --embed, so that a missing checkpoint or database stops at once
    if args.embed:
        from ..ankh import embed_protein
        for protein in sorted({pr for _, pr in pairs}):
            embed_protein(protein, emb_dir, default_entry_name(protein), mode=args.parser, device=args.device)
    sites, failed = [], []
    for pocket, protein in pairs:
        try:
            sites.append(featurize_site(pocket, protein, emb_dir, mode=args.parser, node_dim=cfg.node_dim))
        except (FeaturizeError, PDBParseError, ValueError) as e:
            failed.append((pocket, str(e)))
    for pocket, why in failed:
        print(f"[skip] {pocket}: {why}", file=sys.stderr)
    if not sites:
        die("no query could be featurized", 1)
    latents = Encoder(model, cfg, device=args.device, batch_size=args.batch).encode(sites)
    if args.save_latents:
        np.savez(args.save_latents, site_ids=np.array([s.site_id for s in sites]),
                 offsets=np.concatenate([[0], np.cumsum([len(x) for x in latents])]), mu=np.concatenate(latents))
    if db.dim != latents[0].shape[1]:
        die(f"database latent width {db.dim} != model latent width {latents[0].shape[1]}")
    if db.meta.get("checkpoint_sha256") and db.meta["checkpoint_sha256"] != info.sha256:
        print(f"warning: the database was encoded with checkpoint {db.meta['checkpoint_sha256'][:12]}..., "
              f"the queries with {info.sha256[:12]}...; distances are only meaningful for the same model",
              file=sys.stderr)
    dist = chamfer_matrix(latents, db, threads=args.threads)
    out = open(args.out, "w") if args.out else sys.stdout
    out.write("query\trank\tsite_id\tchamfer_distance\n")
    for qi, s in enumerate(sites):
        for r, j in enumerate(topk(dist[qi], args.top_k), 1):
            out.write(f"{s.site_id}\t{r}\t{db.site_ids[j]}\t{dist[qi, j]:.4f}\n")
    if args.out:
        out.close()
    print(f"[lenseek-search] {len(sites)} queries x {len(db)} sites ({db.meta.get('edition', 'custom')} edition, "
          f"model {info.name or 'custom'}) in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
