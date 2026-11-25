#!/usr/bin/env python
"""Evaluate a checkpoint: encode the query sites and the database, and rank every database site for every query.

    python scripts/evaluate.py --checkpoint lenseek_main_s0.pth --features features/ --queries query_cache.pkl \
        --out eval/out_main_s0 [--rotations 3] [--db-order DB_PATHS] [--device cuda]
    python scripts/evaluate.py --checkpoint lenseek_main_s0.pth --features filter_free_data/sites/added \
        --added eval/out_main_s0 --out scaling/added --threads 32

Writes the evaluation folder that the analysis scripts read:
  q_paths.txt, q_mu.npy (n_q, 65, 256) float32, q_mask.npy (n_q, 65) bool     query sites
  db_paths.txt, db_mu.npy, db_mask.npy                                        database sites
  dist.npy (n_q, n_db) float32                                                Chamfer distances
  rotation.tsv (--rotations N)                                                rotated queries, one row per rotation
Sites are encoded with lenseek.encode.Encoder (wrap padding to 65 residues, posterior means mu; model settings from the
checkpoint, lenseek.checkpoint.load_model). Chamfer distance = mean over the query residues of the squared distance to
the nearest site residue plus the same from the site side, computed per query with torch.cdist on the padded arrays,
padded residues masked (lenseek.search.chamfer_rows_cdist; lenseek.search.chamfer_matrix computes the same distance on
unpadded sites).
--rotations N: every query site is turned rigidly N times (C-alpha positions T -> T Q^T, local frames R -> Q R; Q from
the QR decomposition of a 3 x 3 standard normal matrix with the signs fixed to a rotation, numpy default_rng(0), drawn
query by query), encoded and searched again. rotation.tsv: site, rotation, chamfer_to_unrotated (Chamfer distance
between the encodings of the rotated and the unrotated site), top1 (database index of the unrotated site's nearest site,
ties to the lower index), rank_of_top1 (rank of top1 in the search of the rotated site, ties to the lower index; 1 =
the same top-1 result) and chamfer_to_top1 (rotated site to top1).
--db-order: the database order of the folder (a db_paths.txt; the original model's for the variants), for a feature
store that lacks some of its sites (the language-model variants have no node features for the sites without a chain
embedding). The folder then lists every site of that order: dist.npy has NaN columns and db_mask.npy all-False rows
for the sites not encoded; the encoded sites are in db_subset/ (db_mu.npy, db_mask.npy, db_paths.txt and
db_index.npy, their positions in the order).
--features: feature store of the database (lenseek-featurize) or training pickle. --queries: query cache (pickle
{'queries': [{'path', 'h', 'T', 'R'}]}) or feature store. --queries-only encodes the queries alone.
--added EVAL_DIR: the sites of --features are sites added to the database of the evaluation folder EVAL_DIR (the sites
removed by the ligand filters, scripts/filter_free_data.py sites, for the real150k database of scripts/scaling.py);
the query encodings of EVAL_DIR are reused and --out gets added_mu.npy (float16, the precision of the residue store
of scripts/scaling.py), added_mask.npy, added_keys.txt, dist_added.npy (queries x added sites, from the float32
encodings) and encode_time.json (encode_seconds: wall time of the encoding, threads, added_sites).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import die, site_key  # noqa: E402
from lenseek.data import FeatureStore, open_training_data  # noqa: E402
from lenseek.search import chamfer_rows_cdist  # noqa: E402

SEED = 0                              # rotations
BATCH = 32                            # sites per encoder batch


def load_encoder(checkpoint, device="cpu"):
    """Encoder of a checkpoint and its CheckpointInfo."""
    from lenseek.checkpoint import load_model
    from lenseek.encode import Encoder
    model, cfg, info = load_model(checkpoint, device=device)
    print(f"checkpoint {info.sha256[:12]} ({info.config_source})", flush=True)
    return Encoder(model, cfg, device=device, batch_size=BATCH), info


def write_sites(out, prefix, enc, sites, paths):
    """Encode sites -> <prefix>_mu.npy, <prefix>_mask.npy, <prefix>_paths.txt; returns mu, mask."""
    mu, mask = enc.encode_padded(sites)
    np.save(os.path.join(out, f"{prefix}_mu.npy"), mu)
    np.save(os.path.join(out, f"{prefix}_mask.npy"), mask)
    with open(os.path.join(out, f"{prefix}_paths.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")
    return mu, mask


def read_queries(path):
    """Query inputs [(h, T, R)] and pocket paths of a query cache or a feature store."""
    if os.path.isdir(path):
        fs = FeatureStore(path)
        return [fs.site(i) for i in range(len(fs))], [f"{s}.pdb" for s in fs.site_ids]
    queries = pickle.load(open(path, "rb"))["queries"]
    return [(q["h"], q["T"], q["R"]) for q in queries], [q["path"] for q in queries]


def feature_kind(mats):
    """'onehot' if every row is a 0/1 residue indicator in the first 20 columns, else 'dense'."""
    kinds = set()
    for h in mats:
        oh = (h.shape[1] >= 20 and not np.any(h[:, 20:]) and np.all((h[:, :20] == 0) | (h[:, :20] == 1))
              and np.all(h[:, :20].sum(1) == 1))
        kinds.add("onehot" if oh else "dense")
    if len(kinds) != 1:
        sys.exit(f"node features mix one-hot and dense rows: {sorted(kinds)}")
    return kinds.pop()


def check_features(enc, sites, label):
    widths = {h.shape[1] for h, _, _ in sites}
    if widths != {enc.cfg.node_dim}:
        sys.exit(f"{label}: node feature widths {sorted(widths)} != model node_dim {enc.cfg.node_dim}")


def order_positions(order, paths):
    """Positions of the sites `paths` in the database order `order` (site keys); exits unless each is listed once and
    they follow the order."""
    pos = {site_key(p): i for i, p in enumerate(order)}
    miss = [p for p in paths if site_key(p) not in pos]
    if miss:
        sys.exit(f"{len(miss)} encoded database sites are not in --db-order, e.g. {miss[0]}")
    idx = np.array([pos[site_key(p)] for p in paths], dtype=np.int64)
    if len(idx) and np.any(np.diff(idx) <= 0):
        sys.exit("the database sites of --features do not follow the order of --db-order")
    return idx


def write_in_order(out, order, idx, dist, db_mask):
    """dist.npy, db_mask.npy and db_paths.txt in the database order: NaN distances and all-False masks for the sites
    that were not encoded."""
    d = np.full((dist.shape[0], len(order)), np.nan, dtype=np.float32)
    d[:, idx] = dist
    np.save(os.path.join(out, "dist.npy"), d)
    mk = np.zeros((len(order), db_mask.shape[1]), dtype=bool)
    mk[idx] = db_mask
    np.save(os.path.join(out, "db_mask.npy"), mk)
    with open(os.path.join(out, "db_paths.txt"), "w") as f:
        f.write("\n".join(order) + "\n")


def random_rotation(rng):
    """3 x 3 rotation (float32): QR decomposition of a standard normal matrix, signs fixed to a proper rotation."""
    Q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(r))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def rotation_test(a, enc, queries, q_paths, q_mu, q_mask, dist, db_mu, db_mask, idx):
    """--rotations: rotation.tsv (module docstring); dist and db_mu hold the encoded database sites, idx their
    positions in the database order."""
    rng = np.random.default_rng(SEED)
    items, owner = [], []
    for i, (h, T, R) in enumerate(queries):
        for k in range(a.rotations):
            Q = random_rotation(rng)
            items.append((h, T @ Q.T, np.einsum("ij,njk->nik", Q, R)))
            owner.append((i, k))
    r_mu, r_mask = enc.encode_padded(items)
    r_dist = chamfer_rows_cdist(r_mu, r_mask, db_mu, db_mask, a.device, progress=True)
    ranks, cds = [], []
    with open(os.path.join(a.out, "rotation.tsv"), "w") as f:
        f.write("site\trotation\tchamfer_to_unrotated\ttop1\trank_of_top1\tchamfer_to_top1\n")
        for j, (i, k) in enumerate(owner):
            t, r = int(np.argmin(dist[i])), r_dist[j]
            cds.append(chamfer_rows_cdist(r_mu[j:j + 1], r_mask[j:j + 1], q_mu[i:i + 1], q_mask[i:i + 1],
                                          a.device)[0, 0])
            ranks.append(int((r < r[t]).sum() + (r[:t] == r[t]).sum()) + 1)
            f.write(f"{q_paths[i]}\t{k}\t{cds[-1]:.6g}\t{idx[t]}\t{ranks[-1]}\t{r[t]:.4f}\n")
    print(f"rotations: top-1 unchanged in {ranks.count(1)} of {len(ranks)} searches; Chamfer distance to the unrotated "
          f"encoding: median {np.median(cds):.2g}", flush=True)


def encode_added(a, enc):
    """--added: encodings and query distances of the sites of --features, added to the database of a.added."""
    q_mu, q_mask = (np.load(os.path.join(a.added, f"q_{x}.npy")) for x in ("mu", "mask"))
    db = open_training_data(a.features)
    sites = [db.site(i) for i in range(len(db))]
    check_features(enc, sites[:1], "added sites")
    keys = [f"{s}.pdb" for s in db.site_ids] if hasattr(db, "site_ids") else list(db.path)
    t0 = time.time()
    mu, mask = enc.encode_padded(sites)
    seconds = time.time() - t0
    np.save(os.path.join(a.out, "added_mu.npy"), mu.astype(np.float16))
    np.save(os.path.join(a.out, "added_mask.npy"), mask)
    with open(os.path.join(a.out, "added_keys.txt"), "w") as f:
        f.write("\n".join(keys) + "\n")
    print(f"added sites encoded: {mu.shape}, {seconds:.0f} s", flush=True)
    np.save(os.path.join(a.out, "dist_added.npy"), chamfer_rows_cdist(q_mu, q_mask, mu, mask, a.device, progress=True))
    json.dump(dict(encode_seconds=seconds, threads=a.threads, added_sites=len(keys), device=a.device),
              open(os.path.join(a.out, "encode_time.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", default=None, help="feature store (or training pickle) of the search database")
    ap.add_argument("--queries", default=None, help="query cache (pickle) or feature store of the query sites")
    ap.add_argument("--out", required=True, help="evaluation folder")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--queries-only", action="store_true", help="encode the queries only")
    ap.add_argument("--rotations", type=int, nargs="?", const=3, default=0, metavar="N",
                    help="also search every query again after N random rotations (N default 3) -> rotation.tsv")
    ap.add_argument("--db-order", default=None, help="db_paths.txt of the full database order (sites missing from "
                                                     "--features get NaN distances)")
    ap.add_argument("--added", default=None, metavar="EVAL_DIR",
                    help="encode --features as sites added to the database of EVAL_DIR (its query encodings)")
    a = ap.parse_args(argv)
    if not a.queries_only and not a.features:
        sys.exit("--features is required unless --queries-only")
    if a.queries_only and (a.db_order or a.rotations):
        sys.exit("--rotations and --db-order need the database encoding (drop --queries-only)")
    if a.added and (a.queries or a.queries_only or a.db_order or a.rotations):
        sys.exit("--added reuses the query encodings of its evaluation folder: no --queries, --queries-only, "
                 "--rotations or --db-order")
    if not a.added and not a.queries:
        sys.exit("--queries is required unless --added")
    for flag, path in (("--checkpoint", a.checkpoint), ("--queries", a.queries), ("--features", a.features),
                       ("--db-order", a.db_order), ("--added", a.added)):
        if path and not os.path.exists(path):
            die(f"{flag} {path} does not exist")
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    enc, _ = load_encoder(a.checkpoint, a.device)
    if a.added:
        return encode_added(a, enc)
    queries, q_paths = read_queries(a.queries)
    check_features(enc, queries, "queries")
    q_mu, q_mask = write_sites(a.out, "q", enc, queries, q_paths)
    print(f"queries encoded: {q_mu.shape}", flush=True)
    if a.queries_only:
        return
    db = open_training_data(a.features)
    n_db = len(db)
    sample = sorted(set(range(min(100, n_db))) | set(np.linspace(0, n_db - 1, 200).astype(int).tolist()))
    if feature_kind([db.site(i)[0] for i in sample]) != feature_kind([h for h, _, _ in queries]):
        sys.exit("node features of the database and the queries differ (one-hot vs dense)")
    sites = [db.site(i) for i in range(n_db)]
    check_features(enc, sites[:1], "database")
    db_paths = [f"{s}.pdb" for s in db.site_ids] if hasattr(db, "site_ids") else list(db.path)
    del db
    order = open(a.db_order).read().split() if a.db_order else db_paths
    idx = order_positions(order, db_paths)
    subset = len(idx) < len(order)
    db_dir = os.path.join(a.out, "db_subset") if subset else a.out
    os.makedirs(db_dir, exist_ok=True)
    db_mu, db_mask = write_sites(db_dir, "db", enc, sites, db_paths)
    del sites
    print(f"database encoded: {db_mu.shape}", flush=True)
    dist = chamfer_rows_cdist(q_mu, q_mask, db_mu, db_mask, a.device, progress=True)
    if subset:
        np.save(os.path.join(db_dir, "db_index.npy"), idx)
        write_in_order(a.out, order, idx, dist, db_mask)
        print(f"database order of {a.db_order}: {len(idx)} of {len(order)} sites encoded, the others NaN", flush=True)
    else:
        np.save(os.path.join(a.out, "dist.npy"), dist)
    print(f"distance matrix: {dist.shape}", flush=True)
    if a.rotations:
        rotation_test(a, enc, queries, q_paths, q_mu, q_mask, dist, db_mu, db_mask, idx)


if __name__ == "__main__":
    main()
