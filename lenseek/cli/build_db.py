"""lenseek-build-db: encode a feature store (or a structure root) into a latent database.

    lenseek-build-db --features features/ --out mydb/ [--model main_s0 | --checkpoint ckpt.pth] [--device cuda]
    lenseek-build-db --root structures/ --embeddings embeddings/ --out mydb.npz

The released database was built this way from the feature store of the 114,661 training sites. Sites keep the
feature-store order; site ids are '<entry>/<pocket file stem>'.
"""
from __future__ import annotations

import argparse
import sys
import time

from ..checkpoint import load_model
from ..data import FeatureStore
from ..encode import Encoder
from ..featurize import FeaturizeError, featurize_site, find_sites
from ..io.pdb import PDBParseError
from ..latentdb import LatentDB
from .common import check_device, die, resolve_checkpoint, set_threads


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenseek-build-db", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", help="feature store directory (lenseek-featurize)")
    ap.add_argument("--root", help="structure root (BsitePDB layout), featurized with the strict reader")
    ap.add_argument("--embeddings", help="embedding directory (with --root)")
    ap.add_argument("--pattern", default="*pocket*.pdb")
    ap.add_argument("--model", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True, help="output: directory, or a file name ending in .npz")
    ap.add_argument("--device", default="cpu", help="torch device for encoding, e.g. cuda (default: cpu)")
    ap.add_argument("--batch", type=int, default=32, help="sites encoded per batch (default: 32)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    args = ap.parse_args(argv)
    check_device(args.device)
    set_threads(args.threads)
    ckpt = resolve_checkpoint(args.checkpoint, args.model)
    model, cfg, info = load_model(ckpt, device=args.device)
    enc = Encoder(model, cfg, device=args.device, batch_size=args.batch)
    t0 = time.time()
    if args.features:
        fs = FeatureStore(args.features)
        ids = fs.site_ids
        latents = []
        for s in range(0, len(fs), 1024):
            latents += enc.encode([fs.site(i) for i in range(s, min(s + 1024, len(fs)))], warn_truncation=False)
            print(f"  {len(latents)}/{len(fs)} sites ({time.time() - t0:.0f} s)", file=sys.stderr)
    elif args.root and args.embeddings:
        sites = []
        for pocket, protein in find_sites(args.root, args.pattern):
            try:
                sites.append(featurize_site(pocket, protein, args.embeddings, node_dim=cfg.node_dim))
            except (FeaturizeError, PDBParseError, ValueError) as e:
                print(f"[skip] {pocket}: {e}", file=sys.stderr)
        ids = [s.site_id for s in sites]
        latents = enc.encode(sites)
    else:
        die("give --features, or --root and --embeddings")
    db = LatentDB.from_latents(latents, ids, meta={"model": info.name or "custom", "checkpoint_sha256": info.sha256,
                                                   "edition": "custom", "built_with": "lenseek-build-db"})
    db.save(args.out)
    print(f"[lenseek-build-db] {len(db)} sites -> {args.out} in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
