"""lenseek-embed: per-chain Ankh-base embeddings of protein files.

    lenseek-embed path/to/1abc/protein.pdb --out embeddings/            # one protein
    lenseek-embed structures/ --out embeddings/ --device cuda           # every <entry>/protein.pdb below a root

Writes <out>/<entry>_chain_<C>.npy (float32, L x 768) for every chain with standard residues; <entry> is the protein
file's directory name (BsitePDB layout) or its file stem. Sequences come from the ATOM records with the same reader
(--parser) the featurizer will use, so lengths always match. Each file gets a sidecar <file>.json with the SHA-256
of the chain sequence; an existing file is kept only when its sidecar matches the chain, otherwise it is recomputed.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from ..featurize import default_entry_name
from .common import check_device, set_threads


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenseek-embed", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="protein files or roots in the BsitePDB layout")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--entry", default=None, help="entry name (only with a single protein file)")
    ap.add_argument("--parser", choices=["strict", "legacy"], default="strict")
    ap.add_argument("--device", default="cpu", help="torch device for Ankh-base, e.g. cuda (default: cpu)")
    ap.add_argument("--ankh-dir", default=None, help="local copy of ElnaggarLab/ankh-base (default: download)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    args = ap.parse_args(argv)
    check_device(args.device)
    set_threads(args.threads)
    from ..ankh import embed_protein

    proteins = []
    for item in args.inputs:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, "*", "protein.pdb")) +
                           glob.glob(os.path.join(item, "*", "*_protein.pdb")))
            if os.path.exists(os.path.join(item, "protein.pdb")):
                found.insert(0, os.path.join(item, "protein.pdb"))
            proteins += found
        else:
            proteins.append(item)
    if args.entry and len(proteins) != 1:
        sys.exit("error: --entry needs exactly one protein file")
    t0 = time.time()
    n = 0
    for p in proteins:
        entry = args.entry or default_entry_name(p)
        try:
            written = embed_protein(p, args.out, entry, mode=args.parser, device=args.device,
                                    overwrite=args.overwrite, model_dir=args.ankh_dir)
            n += len(written)
            print(f"{entry}: chains {', '.join(sorted(written))}", file=sys.stderr)
        except Exception as e:  # report and continue with the next protein
            print(f"[skip] {p}: {e}", file=sys.stderr)
    print(f"[lenseek-embed] {n} chain embeddings for {len(proteins)} proteins in {time.time() - t0:.1f} s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
