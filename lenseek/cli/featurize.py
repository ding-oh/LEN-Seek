"""lenseek-featurize: build a feature store (model inputs of many sites) for training or for lenseek-build-db.

    lenseek-featurize --root structures/ --embeddings embeddings/ --out features/ [--filters bsitepdb-v1]
    lenseek-featurize --from-legacy-pickle training_pickle.pkl --out features/

--root: BsitePDB layout, <root>/<entry>/protein.pdb with pocket files (default pattern ligand_pocket*.pdb) and, for
the database filters, ligand files ligand_<n>_<chains>.pdb.
--filters bsitepdb-v1 applies the site filters that defined the released training set and search database:
  * at least 10 pocket residues among the 20 standard amino acids (other ATOM residues of the pocket file, such as
    nucleotides, SEC or UNK, are ignored, as in the released data),
  * the ligand file's first HETATM residue passes lenseek.exclusion.valid_ligand,
  * per-chain embeddings exist and match the protein sequence; N, CA and C present for every pocket residue.
--parser legacy is the fixed-column reader the released training features and database were built with and reproduces
those inputs exactly; the default --parser strict reads standard PDB columns and insertion codes and is the reader for
new data. The two readers can read different coordinates from files in the original BsitePDB column layout (README,
Usage: Search).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

from ..data import FeatureStore, convert_legacy_pickle
from ..exclusion import valid_ligand
from ..featurize import FeaturizeError, featurize_site, featurize_site_legacy, find_sites
from ..io.pdb import PDBParseError, STANDARD_RESIDUES


def onehot_features(residue_names, width: int = 768):
    """One-hot of the 20 standard residues (order of lenseek.io.pdb.STANDARD_RESIDUES) in the first 20 columns."""
    out = np.zeros((len(residue_names), width), dtype=np.float32)
    out[np.arange(len(residue_names)), [STANDARD_RESIDUES.index(n) for n in residue_names]] = 1.0
    return out


def bsitepdb_v1_filter(pocket: str) -> str:
    """'' if the site passes the database filters that do not need featurization, else the reason.

    Non-standard ATOM residues in the pocket file do not exclude a site: featurization keeps the standard residues only.
    """
    ligand_file = pocket.replace("ligand_pocket_", "ligand_")
    name = None
    if os.path.exists(ligand_file):
        with open(ligand_file) as f:
            for line in f:
                if line.startswith("HETATM"):
                    name = line[17:20].strip()
                    break
    if name is None or not valid_ligand(name):
        return "no valid ligand"
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenseek-featurize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="structure root (BsitePDB layout)")
    ap.add_argument("--embeddings", help="directory of <entry>_chain_<C>.npy files")
    ap.add_argument("--out", required=True, help="output feature-store directory")
    ap.add_argument("--pattern", default="ligand_pocket*.pdb")
    ap.add_argument("--parser", choices=["strict", "legacy"], default="strict")
    ap.add_argument("--filters", choices=["none", "bsitepdb-v1"], default="none")
    ap.add_argument("--node-features", choices=["embedding", "onehot"], default="embedding",
                    help="onehot: residue-type one-hot in the first 20 of --onehot-width columns (abl_onehot ablation)")
    ap.add_argument("--onehot-width", type=int, default=768)
    ap.add_argument("--from-legacy-pickle", default=None,
                    help="convert a training pickle (lenseek.data.load_legacy_pickle) instead")
    ap.add_argument("--failures", default=None, help="TSV of skipped sites and reasons (default <out>/failures.tsv)")
    args = ap.parse_args(argv)

    if args.from_legacy_pickle:
        fs = convert_legacy_pickle(args.from_legacy_pickle, args.out)
        print(f"[lenseek-featurize] {len(fs)} sites converted to {args.out}", file=sys.stderr)
        return
    if not args.root or not args.embeddings:
        sys.exit("error: --root and --embeddings are required (or --from-legacy-pickle)")
    t0 = time.time()
    sites, failures = [], []
    for pocket, protein in find_sites(args.root, args.pattern):
        if args.filters == "bsitepdb-v1":
            why = bsitepdb_v1_filter(pocket)
            if why:
                failures.append((pocket, why))
                continue
        try:
            if args.parser == "legacy":
                sites.append(featurize_site_legacy(pocket, protein, args.embeddings))
            else:
                sites.append(featurize_site(pocket, protein, args.embeddings))
        except (FeaturizeError, PDBParseError, ValueError, KeyError) as e:
            failures.append((pocket, str(e)))
    if not sites:
        sys.exit("error: no site could be featurized")
    if args.node_features == "onehot":
        for s in sites:
            s.h = onehot_features([k[:3] for k in s.keys], args.onehot_width)
    FeatureStore.write(args.out, sites, meta={"root": os.path.abspath(args.root), "parser": args.parser,
                                              "filters": args.filters})
    fpath = args.failures or os.path.join(args.out, "failures.tsv")
    with open(fpath, "w") as f:
        f.write("pocket\treason\n")
        for p, why in failures:
            f.write(f"{p}\t{why}\n")
    reasons = Counter(w.split(":")[0] if args.filters == "none" else w for _, w in failures)
    print(f"[lenseek-featurize] {len(sites)} sites written, {len(failures)} skipped ({fpath}) in "
          f"{time.time() - t0:.0f} s; {dict(reasons.most_common(5))}", file=sys.stderr)


if __name__ == "__main__":
    main()
