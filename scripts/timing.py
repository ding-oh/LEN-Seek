#!/usr/bin/env python
"""Timing runs behind main-text Table 2 and Supplementary Table S17 (computational cost of ProBiS, G-LoSA, DeeplyTough,
Folddisco and LEN-Seek).

All tools search the same queries (a random sample of the evaluation queries: --n-queries 20, --seed 0) against the
whole database on one node with --threads 16 cores (DeeplyTough: its pipeline on one core, the network and the search
also timed on 16); times are wall-clock per query. analysis/table2_cost.py builds Table 2 and Supplementary Table S17
from the files written here (OUT = --out, default $LENSEEK_OUT/timing):

  probis-prep   ProBiS preprocessing: -extract a surface (.srf) of every database site (--threads processes), the
                extraction of probis without the queries, in a run of its own; Table 2 takes the ProBiS
                preprocessing time from it. -> OUT/probis_prep.tsv, probis_prep_summary.txt
  probis        ProBiS database mode: -extract a surface (.srf) of every database site and of the queries (--threads
                processes), then per query -surfdb against all database surfaces (-ncpu --threads) and -results.
                -> OUT/probis_extract.tsv, probis_surfdb.tsv, probis_summary.txt
  glosa         G-LoSA: staged pocket copies and chemical features of every site (--threads BatchCF JVMs), then per
                query glosa against all database pockets (--threads processes).
                -> OUT/glosa_full.tsv, glosa_full_ga.npz, glosa_summary.txt
  lenseek       LEN-Seek per query, on CPU or GPU: pass A embeds every chain of the query protein with Ankh, encodes
                the site and scans the database (exact Chamfer, top-100); pass B repeats encode + scan with the
                embeddings on disk. -> OUT/lenseek_<device>_timing.tsv, lenseek_<device>_top100.npz
  lenseek-prep  LEN-Seek database preprocessing (Ankh embedding, featurization, encoding), each stage timed, over
                the whole database or (--sample) a random sample of entries.
                -> OUT/lenseek_prep_<full|sample<n>_seed<s>>_chains.tsv, _sites.tsv, _summary.txt
  deeplytough   DeeplyTough per query with its public pipeline: the queries one at a time through scripts/deeplytough.py
                run (a pool of one process, the network on one thread). -> OUT/dt_queries.jsonl (task file),
                timing_q.tsv (sid, t_feat, t_vox, t_net), desc_q.npz, fail_q.tsv
  dt-net        the DeeplyTough network alone on one 8 x 24^3 input (batch 1) with 1 and 16 threads: one warm-up pass,
                then the mean of three. -> OUT/dt_net.tsv (seconds per pocket, one column per thread count)
  dt-search     DeeplyTough search per query with 16 BLAS threads: one descriptor against all database descriptors +
                top-100. -> OUT/search_timing.tsv
  dt-prep       DeeplyTough featurization cost of the database from the worker timing tables of its runs
                (scripts/deeplytough.py run). -> OUT/dt_prep.csv
  folddisco     Folddisco: the index of the database pocket files rewritten by scripts/folddisco.py prepare (--work)
                built with --threads threads, then per query one process with the prefilter command line of
                scripts/folddisco.py and --threads threads, loading the index included. -> OUT/folddisco_index.tsv
                (wall and CPU seconds and peak memory of the build), folddisco_timing.tsv (per query: exit status,
                wall and CPU seconds, database sites returned), folddisco_queries/ (outputs and logs)

The LEN-Seek scan uses chamfer_block, the CSR Chamfer kernel of scripts/pipeline.py, on the CPU and a padded form of it
on the GPU (scan_gpu); both use the precomputed database norms, do not copy the database per query and give the
distances of lenseek.search.chamfer_matrix up to float rounding.
External tools for probis-prep / probis / glosa: PROBIS_BIN, GLOSA_DIR (compiled helpers in cf/ and cfbatch/,
scripts/labels.py build-glosa), GLOSA_BIN, JAVA_BIN; node-local scratch $LENSEEK_TMP, else $TMPDIR; for folddisco
FOLDDISCO (the executable, default `folddisco` on PATH) and --work on a local disk. lenseek / lenseek-prep need a
checkpoint, the CSR latent database of the scan (db_res.npy, db_off.npy, db_norm.npy; examples/scaling.sh
single-vector writes it from the database encodings db_mu.npy and db_mask.npy of scripts/evaluate.py, and asset
eval_main_s0 has no db_mu.npy) and the Ankh-base weights.
deeplytough, dt-net and dt-search run in the DeeplyTough environment (scripts/deeplytough.py: DEEPLYTOUGH_HOME,
MGL_ENV) from a writable working directory; dt-search reads the descriptors of scripts/deeplytough.py assemble.
Query and database site lists default to the lists of scripts/labels.py sites.

usage:
  python scripts/timing.py probis-prep
  python scripts/timing.py probis
  python scripts/timing.py glosa
  python scripts/timing.py lenseek --checkpoint CKPT --latent-db DIR --device cpu
  python scripts/timing.py lenseek --checkpoint CKPT --latent-db DIR --device cuda --threads 6       (GPU node)
  python scripts/timing.py lenseek-prep --checkpoint CKPT --device cpu --sample 500
  python scripts/timing.py lenseek-prep --checkpoint CKPT --device cuda --threads 6                  (GPU node)
  $DT_PYTHON scripts/timing.py deeplytough
  $DT_PYTHON scripts/timing.py dt-net
  $DT_PYTHON scripts/timing.py dt-search
  python scripts/timing.py dt-prep --runs RUN_DIR [RUN_DIR ...]
  python scripts/timing.py folddisco
"""
import argparse
import glob
import json
import os
import platform
import random
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import (GLOSA_CF_CP, GLOSA_DIR, JAVA, JAVA_OPTS, PROBIS, cf_of, chamfer_block, cpu_model, die,  # noqa: E402
                      glosa_compare, pocket_chains, pocket_protein, probis_motif, read_list, run, scratch_base,
                      stage_name, stage_protein, strip_inner_ter)

OUT_ROOT = os.environ.get("LENSEEK_OUT") or "out"
SITES = os.path.join(OUT_ROOT, "labels", "sites")
DT_NETWORK = os.path.join("networks", "deeplytough_toughm1_test.pth.tar")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
# input options of the subcommands: the files they must hold (none: the path itself) and the step that writes them
INPUTS = {"queries": ((), "bash examples/labels.sh sites"), "db": ((), "bash examples/labels.sh sites"),
          "latent_db": (("db_res.npy", "db_off.npy", "db_norm.npy"), "bash examples/scaling.sh single-vector"),
          "q_desc": ((), "bash examples/labels.sh dt-assemble"), "db_desc": ((), "bash examples/labels.sh dt-assemble"),
          "runs": ((), "bash examples/labels.sh dt-run"), "work": ((), "bash examples/labels.sh fd-prepare")}


# ---------------------------------------------------------------- ProBiS and G-LoSA per query
class Clock:
    """Wall time and measured CPU time (self + reaped children) of a block."""
    def __enter__(self):
        self.w0 = time.perf_counter()
        s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        self.c0 = s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime
        return self

    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.w0
        s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        self.cpu = s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime - self.c0


def sample_queries(qfile, nq, seed):
    queries = [ln.strip() for ln in open(qfile) if ln.strip()]
    return queries, sorted(random.Random(seed).sample(range(len(queries)), nq))


def summary(outdir, name, line):
    """Print a SUMMARY line and append it to OUT/<name>_summary.txt."""
    print(line, flush=True)
    with open(os.path.join(outdir, f"{name}_summary.txt"), "a") as f:
        f.write(line + "\n")


def extract_one(arg):
    i, site, tag, srfdir, tmpdir = arg
    prot = pocket_protein(site)
    if not (os.path.isfile(site) and os.path.isfile(prot)):
        return i, site, "", "", float("nan"), float("nan"), "missing"
    w = tempfile.mkdtemp(prefix="x_", dir=tmpdir)
    try:
        t0 = time.perf_counter()
        motif, shifts, _ = probis_motif(site, prot)
        chains, _ = pocket_chains(site)
        name = stage_name(site, tag)
        stage_protein(prot, os.path.join(w, name), shifts)
        t_stage = time.perf_counter() - t0
        srf = os.path.join(srfdir, f"{tag}{i:06d}.srf")
        t0 = time.perf_counter()
        rc, _, _, _, to = run([PROBIS, "-extract", "-f1", name, "-c1", chains, "-motif1", motif, "-srffile", srf], w,
                              3600)
        dt = time.perf_counter() - t0
        ok = os.path.isfile(srf) and os.path.getsize(srf) > 0
        return i, site, srf, chains, t_stage, dt, "ok" if ok else ("timeout" if to else f"fail rc={rc}")
    finally:
        shutil.rmtree(w, ignore_errors=True)


def probis_db(a, prep=False):
    """probis; with prep=True probis-prep: the database extraction alone, written as probis_prep*."""
    outdir, dbfile, threads = a.out, a.db, a.threads
    queries, qs = ([], []) if prep else sample_queries(a.queries, a.n_queries, a.seed)
    dbs = [ln.strip() for ln in open(dbfile) if ln.strip()]
    scratch = tempfile.mkdtemp(prefix="probis_", dir=scratch_base())
    srfdir = os.path.join(scratch, "srf")
    os.makedirs(srfdir)
    try:
        jobs = [(i, s, "t", srfdir, scratch) for i, s in enumerate(dbs)] + [(i, queries[i], "q", srfdir, scratch) for i in qs]
        with Clock() as ce:
            with ProcessPoolExecutor(threads) as ex:
                rows = list(ex.map(extract_one, jobs, chunksize=64))
        with open(os.path.join(outdir, "probis_prep.tsv" if prep else "probis_extract.tsv"), "w") as f:
            f.write("idx\tsite\tsrf\tchains\tstage_s\textract_s\tstatus\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        db_rows = [r for r in rows[:len(dbs)] if r[6] == "ok"]
        bad = [r for r in rows if r[6] != "ok"]
        with open(os.path.join(scratch, "srfs.txt"), "w") as f:
            for r in db_rows:
                f.write(f"{r[2]} {r[3]}\n")
        qrows = {r[0]: r for r in rows[len(dbs):]}
        summary(outdir, "probis_prep" if prep else "probis",
                f"SUMMARY probis_db extract: {len(db_rows)}/{len(dbs)} DB .srf ok; wall {ce.wall:.0f}s, measured CPU "
                f"{ce.cpu:.0f} core-s ({ce.cpu / 3600:.2f} core-h) with {threads} processes; per site mean stage "
                f"{np.mean([r[4] for r in db_rows]):.4f}s extract {np.mean([r[5] for r in db_rows]):.4f}s; srf "
                f"{sum(os.path.getsize(os.path.join(srfdir, x)) for x in os.listdir(srfdir)) / 1e9:.2f} GB; "
                f"failed: {[(r[1], r[6]) for r in bad]}")
        if prep:
            return
        out = []
        for qi in qs:
            r = qrows[qi]
            if r[6] != "ok":
                out.append((qi, queries[qi], 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), r[6]))
                continue
            w = tempfile.mkdtemp(prefix="q_", dir=scratch)
            site = queries[qi]
            prot = pocket_protein(queries[qi])
            motif, shifts, _ = probis_motif(site, prot)
            name = stage_name(site, "q")
            stage_protein(prot, os.path.join(w, name), shifts)
            with Clock() as c1:
                rc1, _, _, _, _ = run([PROBIS, "-surfdb", "-sfile", os.path.join(scratch, "srfs.txt"), "-f1", r[2],
                                       "-c1", r[3], "-nosql", "q.nosql", "-ncpu", str(threads)], w, 7200)
            nrows = sum(1 for _ in open(os.path.join(w, "q.nosql"))) if os.path.exists(os.path.join(w, "q.nosql")) else 0
            with Clock() as c2:
                rc2, _, _, _, _ = run([PROBIS, "-results", "-z_score", "-1000", "-f1", name, "-c1", r[3], "-motif1",
                                       motif, "-nosql", "q.nosql", "-json", "q.json"], w, 3600)
            out.append((qi, queries[qi], len(db_rows), nrows, c1.wall, c1.cpu, c2.wall, c2.cpu, f"rc {rc1}/{rc2}"))
            print(f"query {qi}: surfdb wall {c1.wall:.1f}s cpu {c1.cpu:.0f} core-s ({nrows} nosql rows); results "
                  f"{c2.wall:.1f}s", flush=True)
            shutil.rmtree(w, ignore_errors=True)
        with open(os.path.join(outdir, "probis_surfdb.tsv"), "w") as f:
            f.write("q_idx\tquery\tn_targets\tnosql_rows\tsurfdb_wall_s\tsurfdb_core_s\tresults_wall_s\tresults_core_s\t"
                    "note\n")
            for r in out:
                f.write("\t".join(str(x) for x in r) + "\n")
        ok = [r for r in out if r[4] == r[4]]
        tw = np.array([r[4] + r[6] for r in ok])
        tc = np.array([r[5] + r[7] for r in ok])
        summary(outdir, "probis",
                f"SUMMARY probis_db per query ({len(ok)} queries, -ncpu {threads}): wall mean {tw.mean():.1f}s median "
                f"{np.median(tw):.1f}s min {tw.min():.1f} max {tw.max():.1f}; measured CPU mean {tc.mean():.0f} core-s; "
                f"per comparison {tw.mean() / len(db_rows) * 1e3:.3f} ms wall, {tc.mean() / len(db_rows):.4f} core-s")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def probis_prep(a):
    """probis-prep (see the module docstring)."""
    probis_db(a, prep=True)


def stage_copy(arg):
    src, dst = arg
    if not os.path.isfile(src):
        return None
    strip_inner_ter(src, dst)
    return dst


def batchcf(lst_and_dir):
    lst, wd = lst_and_dir
    cp = GLOSA_CF_CP + ":" + os.path.join(GLOSA_DIR, "cfbatch")
    rc, _, err, _, to = run([JAVA, *JAVA_OPTS, "-cp", cp, "BatchCF", lst], wd, 7200)
    return rc, to, err.strip()[-200:]


G = {}


def g_init(db_files, scratch):
    G["db"] = db_files
    G["w"] = tempfile.mkdtemp(prefix="gw_", dir=scratch)


def g_pair(arg):
    q, j = arg
    st, ga, _ = glosa_compare(q, G["db"][j], G["w"], 600)
    return j, st, ga


def glosa_full(a):
    outdir, qfile, dbfile, nq, seed, threads = a.out, a.queries, a.db, a.n_queries, a.seed, a.threads
    queries, qs = sample_queries(qfile, nq, seed)
    dbs = [ln.strip() for ln in open(dbfile) if ln.strip()]
    scratch = tempfile.mkdtemp(prefix="glosa_", dir=scratch_base())
    assert "pdb" not in scratch   # AssignChemicalFeatures splits the path on the regex ".pdb"
    sd = os.path.join(scratch, "s")
    os.makedirs(sd)
    try:
        jobs = [(s, os.path.join(sd, f"t{i:06d}.pdb")) for i, s in enumerate(dbs)] + \
               [(queries[i], os.path.join(sd, f"q{i:03d}.pdb")) for i in qs]
        with Clock() as cs:
            with ProcessPoolExecutor(threads) as ex:
                staged = list(ex.map(stage_copy, jobs, chunksize=256))
        files = [f for f in staged if f]
        lists = []
        for k in range(threads):
            lst = os.path.join(scratch, f"cf_{k:02d}.txt")
            open(lst, "w").write("\n".join(files[k::threads]) + "\n")
            lists.append((lst, scratch))
        with Clock() as cf:
            with ProcessPoolExecutor(threads) as ex:
                cfres = list(ex.map(batchcf, lists))
        db_files = staged[:len(dbs)]
        q_files = dict(zip(qs, staged[len(dbs):]))
        nocf = sum(1 for f in files if cf_of(f) is None)
        summary(outdir, "glosa",
                f"SUMMARY glosa prep: {len(files)} pockets staged wall {cs.wall:.0f}s CPU {cs.cpu:.0f} core-s; "
                f"features ({threads} BatchCF JVMs) wall {cf.wall:.0f}s CPU {cf.cpu:.0f} core-s; without features "
                f"{nocf}; missing {sum(1 for f in staged if f is None)}; BatchCF rc {[r[0] for r in cfres]}")
        rows = []
        gas = {}
        for qi in qs:
            q = q_files[qi]
            ga = np.full(len(dbs), np.nan, np.float32)
            st = np.zeros(len(dbs), np.uint8)
            with Clock() as c:
                with ProcessPoolExecutor(threads, initializer=g_init, initargs=(db_files, scratch)) as ex:
                    for j, s, g in ex.map(g_pair, [(q, j) for j in range(len(dbs))], chunksize=256):
                        st[j], ga[j] = s, g
            gas[qi] = ga
            cnt = {c_: int((st == c_).sum()) for c_ in range(1, 8) if (st == c_).any()}
            rows.append((qi, queries[qi], len(dbs), c.wall, c.cpu, cnt))
            print(f"query {qi}: glosa vs {len(dbs)} DB wall {c.wall:.1f}s CPU {c.cpu:.0f} core-s; status {cnt}", flush=True)
        np.savez_compressed(os.path.join(outdir, "glosa_full_ga.npz"), q_idx=np.array(qs),
                            ga=np.stack([gas[q] for q in qs]))
        with open(os.path.join(outdir, "glosa_full.tsv"), "w") as f:
            f.write("q_idx\tquery\tn_db\twall_s\tcore_s\tstatus_counts\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        w = np.array([r[3] for r in rows])
        cc = np.array([r[4] for r in rows])
        summary(outdir, "glosa",
                f"SUMMARY glosa per query ({len(rows)} queries, {threads} processes): wall mean {w.mean():.1f}s median "
                f"{np.median(w):.1f}s min {w.min():.1f} max {w.max():.1f}; measured CPU mean {cc.mean():.0f} core-s; "
                f"per comparison {w.mean() / len(dbs) * 1e3:.3f} ms wall, {cc.mean() / len(dbs):.4f} core-s")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- LEN-Seek per query
def lenseek_timing(args):
    """LEN-Seek per query: pass A embeds the query chains with Ankh, encodes and scans; pass B repeats
    encode + scan with the embeddings on disk (see the module docstring)."""
    import torch
    ckpt, cache_dir, qfile, nq, seed, threads, outdir = (args.checkpoint, args.latent_db, args.queries, args.n_queries,
                                                         args.seed, args.threads, args.out)
    torch.set_num_threads(threads)
    os.makedirs(outdir, exist_ok=True)

    def site_chunks(off, max_res):
        out, s, N = [], 0, len(off) - 1
        while s < N:
            e = int(np.searchsorted(off, off[s] + max_res, side="right")) - 1
            e = min(max(e, s + 1), N)
            out.append((s, e))
            s = e
        return out

    def scan_gpu(q):
        qt = torch.from_numpy(np.ascontiguousarray(q)).to(gdev)
        qn = (qt * qt).sum(1)
        D = torch.addmm(norms_g.unsqueeze(0), qt, res_g.T, beta=1.0, alpha=-2.0)
        D.add_(qn.unsqueeze(1)).clamp_(min=0.0)
        out = torch.empty(Ns, device=gdev)
        for s_ in range(0, Ns, 16384):
            e_ = min(Ns, s_ + 16384)
            Dp = D[:, idx_g[s_:e_]]
            v = valid_g[s_:e_]
            a2b = Dp.masked_fill(~v.unsqueeze(0), float("inf")).min(dim=2).values.mean(dim=0)
            b2a = (Dp.min(dim=0).values * v).sum(1) / v.sum(1)
            out[s_:e_] = a2b + b2a
        torch.cuda.synchronize()
        return out.cpu().numpy()

    def scan_one(q, res_t, off, norms_t, chunks):
        """q: (n, 256) float32 numpy (valid query residues). Returns (N,) float32 Chamfer distances."""
        if DEVN == "cuda":
            return scan_gpu(q)
        N = len(off) - 1
        out = np.empty(N, dtype=np.float32)
        qt = torch.from_numpy(np.ascontiguousarray(q))
        qoff = np.array([0, len(q)])
        for s, e in chunks:
            out[s:e] = chamfer_block(qt, qoff, res_t[off[s]:off[e]], off[s:e + 1] - off[s], norms_t[off[s]:off[e]])[0]
        return out

    t0 = time.perf_counter()
    res = np.load(os.path.join(cache_dir, "db_res.npy"))
    off = np.load(os.path.join(cache_dir, "db_off.npy"))
    norms = np.load(os.path.join(cache_dir, "db_norm.npy"))
    res_t, norms_t = torch.from_numpy(res), torch.from_numpy(norms)
    chunks = site_chunks(off, 65536)
    DEVN = args.device
    if DEVN == "cuda":
        gdev = torch.device("cuda")
        Ns, Ls = len(off) - 1, np.diff(off)
        ar = np.arange(int(Ls.max()))
        valid_np = ar[None, :] < Ls[:, None]
        idx_np = np.where(valid_np, off[:-1, None] + ar[None, :], 0)
        res_g = torch.from_numpy(res).to(gdev)
        norms_g = torch.from_numpy(norms).to(gdev)
        idx_g, valid_g = torch.from_numpy(idx_np).to(gdev), torch.from_numpy(valid_np).to(gdev)
        torch.cuda.synchronize()
    t_db = time.perf_counter() - t0
    print(f"DB loaded: {len(off) - 1} sites, {res.shape[0]} residues, {res.nbytes / 1e9:.2f} GB in {t_db:.1f}s", flush=True)

    t0 = time.perf_counter()
    from lenseek.ankh import chain_sequences_of, embed_sequence, load_ankh
    from lenseek.checkpoint import load_model
    from lenseek.encode import Encoder
    from lenseek.featurize import FeaturizeError, featurize_site
    from lenseek.io.pdb import PDBParseError
    t_import = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, cfg, _ = load_model(ckpt, device=DEVN)
    ENC = Encoder(model, cfg, device=DEVN)
    t_model = time.perf_counter() - t0
    t0 = time.perf_counter()
    ankh_model, ankh_tok = load_ankh(DEVN, model_dir=args.ankh_dir)
    t_ankh_load = time.perf_counter() - t0
    print(f"imports {t_import:.1f}s, LEN-Seek model {t_model:.1f}s, Ankh model {t_ankh_load:.1f}s "
          f"(ok={ankh_model is not None}); torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)

    print(f"device: {DEVN} ({torch.cuda.get_device_name(0) if DEVN == 'cuda' else 'cpu'}; torch.cuda.is_available()="
          f"{torch.cuda.is_available()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); host "
          f"{socket.gethostname()}", flush=True)
    queries = [ln.strip() for ln in open(qfile) if ln.strip()]
    qs = sorted(random.Random(seed).sample(range(len(queries)), nq))
    A = {}
    for qi in qs:   # pass A: with Ankh
        pocket = queries[qi]
        protein = os.path.join(os.path.dirname(pocket), "protein.pdb")
        pdbid = os.path.basename(os.path.dirname(protein))
        adir = tempfile.mkdtemp(prefix="ankh_")
        t_all = time.perf_counter()
        seqs = chain_sequences_of(protein, mode="legacy")
        nres = 0
        for ch, seq in seqs.items():
            emb = embed_sequence(seq, ankh_model, ankh_tok, DEVN)
            np.save(os.path.join(adir, f"{pdbid}_chain_{ch}.npy"), emb)
            nres += len(seq)
        t_ankh = time.perf_counter() - t_all
        t0 = time.perf_counter()
        try:
            q = ENC.encode([featurize_site(pocket, protein, adir, mode="legacy", node_dim=cfg.node_dim)],
                           warn_truncation=False)[0]
        except (FeaturizeError, PDBParseError, ValueError):
            q = None
        if DEVN == "cuda":
            torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0
        if q is None:
            A[qi] = (pocket, len(seqs), nres, 0, t_ankh, t_enc, float("nan"), float("nan"), adir, None)
            continue
        q = q.astype(np.float32)
        t0 = time.perf_counter()
        d = scan_one(q, res_t, off, norms_t, chunks)
        top = np.lexsort((np.arange(len(d)), d))[:100]
        t_scan = time.perf_counter() - t0
        A[qi] = (pocket, len(seqs), nres, len(q), t_ankh, t_enc, t_scan, time.perf_counter() - t_all, adir, top)
        print(f"A query {qi} ({pdbid}, {len(seqs)} chains, {nres} aa, {len(q)} pocket res): ankh {t_ankh:.2f}s encode "
              f"{t_enc:.2f}s scan {t_scan:.2f}s total {A[qi][7]:.2f}s", flush=True)
    rows = []
    for qi in qs:   # pass B: embeddings cached on disk
        pocket, nch, nres, npr, t_ankh, t_enc, t_scan, t_with, adir, topA = A[qi]
        protein = os.path.join(os.path.dirname(pocket), "protein.pdb")
        t_all = time.perf_counter()
        try:
            q = ENC.encode([featurize_site(pocket, protein, adir, mode="legacy", node_dim=cfg.node_dim)],
                           warn_truncation=False)[0]
        except (FeaturizeError, PDBParseError, ValueError):
            q = None
        if DEVN == "cuda":
            torch.cuda.synchronize()
        t_enc2 = time.perf_counter() - t_all
        if q is None or topA is None:
            rows.append((qi, pocket, nch, nres, npr, t_ankh, t_enc, t_scan, t_with, t_enc2, float("nan"), float("nan"),
                         "fail"))
            continue
        q = q.astype(np.float32)
        t0 = time.perf_counter()
        d = scan_one(q, res_t, off, norms_t, chunks)
        top = np.lexsort((np.arange(len(d)), d))[:100]
        t_scan2 = time.perf_counter() - t0
        t_wo = time.perf_counter() - t_all
        rows.append((qi, pocket, nch, nres, npr, t_ankh, t_enc, t_scan, t_with, t_enc2, t_scan2, t_wo,
                     "ok" if (top == topA).all() else "top100_differs"))
        print(f"B query {qi}: encode {t_enc2:.2f}s scan {t_scan2:.2f}s total {t_wo:.2f}s", flush=True)
        shutil.rmtree(adir, ignore_errors=True)
    with open(os.path.join(outdir, f"lenseek_{DEVN}_timing.tsv"), "w") as f:
        f.write("q_idx\tquery\tn_chains\tn_aa\tn_pocket_res\tA_ankh_s\tA_encode_s\tA_scan_s\tA_total_with_ankh_s\t"
                "B_encode_s\tB_scan_s\tB_total_without_ankh_s\tstatus\n")
        for r in rows:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    ok = [r for r in rows if r[12] in ("ok", "top100_differs")]
    if ok:
        w, wo = np.array([r[8] for r in ok]), np.array([r[11] for r in ok])
        print(f"SUMMARY lenseek_{DEVN} ({threads} threads, {len(ok)} queries, Ankh on {DEVN}): with Ankh mean "
              f"{w.mean():.3f}s median {np.median(w):.3f}s (excl. first query mean {w[1:].mean():.3f}s); without Ankh "
              f"mean {wo.mean():.3f}s median {np.median(wo):.3f}s min {wo.min():.3f}s max {wo.max():.3f}s; one-time: DB "
              f"load {t_db:.1f}s, model {t_model:.1f}s, Ankh load {t_ankh_load:.1f}s; same top-100 in A and B: "
              f"{sum(r[12] == 'ok' for r in ok)}/{len(ok)}")
    np.savez(os.path.join(outdir, f"lenseek_{DEVN}_top100.npz"), q_idx=np.array([r[0] for r in rows]),
             top=np.array([A[r[0]][9] for r in rows]))
    print("LENSEEK_TIMING_DONE")


# ---------------------------------------------------------------- LEN-Seek database preprocessing
def lenseek_db_prep_timing(a):
    """LEN-Seek database preprocessing: Ankh, featurization and encoding of every site, each
    stage timed separately (see the module docstring)."""
    import torch

    torch.set_num_threads(a.threads)
    t_imp = time.perf_counter()
    from lenseek.ankh import chain_sequences_of, embed_sequence, load_ankh
    from lenseek.checkpoint import load_model
    from lenseek.encode import Encoder
    from lenseek.featurize import MIN_RESIDUES, FeaturizeError, featurize_site_legacy, legacy_resdefs
    from lenseek.io.pdb import parse_pdb_legacy
    t_imp = time.perf_counter() - t_imp
    dev = torch.device(a.device)

    def cpu_s():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    sites = [ln.strip() for ln in open(a.db) if ln.strip()]
    by_entry = defaultdict(list)
    for j, s in enumerate(sites):
        by_entry[os.path.dirname(s)].append(j)
    entries = sorted(by_entry)
    if a.sample:
        entries = sorted(random.Random(a.seed).sample(entries, a.sample))
        tag = f"sample{a.sample}_seed{a.seed}"
    else:
        tag = "full"
    os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print(f"[{tag}] {len(entries)} entries, {sum(len(by_entry[e]) for e in entries)} sites; device {a.device} ({gpu}); "
          f"threads {a.threads}; host {os.uname().nodename}; cpu {cpu_model()}; imports {t_imp:.1f}s", flush=True)

    t0, c0 = time.perf_counter(), cpu_s()
    model, cfg, _ = load_model(a.checkpoint, device=a.device)
    ENC = Encoder(model, cfg, device=a.device, batch_size=a.batch)
    ankh_model, ankh_tok = load_ankh(a.device, model_dir=a.ankh_dir)
    sync()
    t_load, c_load = time.perf_counter() - t0, cpu_s() - c0
    assert ankh_model is not None, "Ankh model not loaded"

    scratch = tempfile.mkdtemp(prefix="lenseek_prep_", dir=scratch_base())
    T = {"ankh": [0.0, 0.0], "featurize": [0.0, 0.0], "encode": [0.0, 0.0]}   # wall, cpu
    chains_out, sites_out = [], []
    pending = []      # (site index, ftr)

    def flush_batch():
        if not pending:
            return
        t0, c0 = time.perf_counter(), cpu_s()
        ENC.encode_padded([f for _, f in pending])
        sync()
        T["encode"][0] += time.perf_counter() - t0
        T["encode"][1] += cpu_s() - c0
        pending.clear()

    t_all, c_all = time.perf_counter(), cpu_s()
    for n_e, ed in enumerate(entries, 1):
        prot = pocket_protein(sites[by_entry[ed][0]])
        pdbid = os.path.basename(ed)
        adir = os.path.join(scratch, pdbid)
        os.makedirs(adir, exist_ok=True)
        # 1) Ankh for every chain of the protein
        seqs = chain_sequences_of(prot, mode="legacy") if os.path.isfile(prot) else {}
        for ch, seq in seqs.items():
            t0, c0 = time.perf_counter(), cpu_s()
            emb = embed_sequence(seq, ankh_model, ankh_tok, a.device)
            sync()
            dt, dc = time.perf_counter() - t0, cpu_s() - c0
            T["ankh"][0] += dt
            T["ankh"][1] += dc
            if emb is not None:
                np.save(os.path.join(adir, f"{pdbid}_chain_{ch}.npy"), emb)
            chains_out.append((pdbid, ch, len(seq), dt, dc, "ok" if emb is not None else "fail"))
        # 2) featurize every site of the entry, 3) encode in batches
        t0, c0 = time.perf_counter(), cpu_s()
        structure = parse_pdb_legacy(prot) if seqs else None
        T["featurize"][0] += time.perf_counter() - t0
        T["featurize"][1] += cpu_s() - c0
        for j in by_entry[ed]:
            pocket = sites[j]
            t0, c0 = time.perf_counter(), cpu_s()
            ftr, why, resdef = None, "", []
            try:
                resdef = legacy_resdefs(pocket)
                if len(resdef) < MIN_RESIDUES:
                    why = "too_few_residues"
                elif structure is None:
                    why = "no_structure"
                else:
                    ftr = featurize_site_legacy(pocket, prot, adir, entry=pdbid, structure=structure)
                    why = "ok"
            except FeaturizeError:
                why = "featurize_failed"
            except Exception as ex:  # noqa: BLE001
                why = f"error {type(ex).__name__}"
            dt, dc = time.perf_counter() - t0, cpu_s() - c0
            T["featurize"][0] += dt
            T["featurize"][1] += dc
            sites_out.append((j, len(resdef), dt, dc, why))
            if ftr is not None:
                pending.append((j, ftr))
                if len(pending) >= a.batch:
                    flush_batch()
        shutil.rmtree(adir, ignore_errors=True)
        if n_e % 2000 == 0:
            el = time.perf_counter() - t_all
            print(f"  {n_e}/{len(entries)} entries, {el:.0f}s; ankh {T['ankh'][0]:.0f}s featurize {T['featurize'][0]:.0f}s "
                  f"encode {T['encode'][0]:.0f}s", flush=True)
    flush_batch()
    t_all, c_all = time.perf_counter() - t_all, cpu_s() - c_all
    shutil.rmtree(scratch, ignore_errors=True)

    with open(os.path.join(a.out, f"lenseek_prep_{tag}_chains.tsv"), "w") as f:
        f.write("entry\tchain\tlength\twall_s\tcpu_s\tstatus\n")
        for r in chains_out:
            f.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    with open(os.path.join(a.out, f"lenseek_prep_{tag}_sites.tsv"), "w") as f:
        f.write("db_idx\tn_res\tfeaturize_wall_s\tfeaturize_cpu_s\tstatus\n")
        for r in sites_out:
            f.write("\t".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    n_ok = sum(1 for r in sites_out if r[4] == "ok")
    summ = (f"SUMMARY lenseek-prep {tag}: device {a.device} ({gpu}), threads {a.threads}, host {os.uname().nodename}, "
            f"cpu {cpu_model()}; entries {len(entries)}, chains {len(chains_out)} "
            f"(failed {sum(1 for r in chains_out if r[5] != 'ok')}), sites {len(sites_out)} (encoded {n_ok}; not encoded: "
            f"{dict((k, sum(1 for r in sites_out if r[4] == k)) for k in set(r[4] for r in sites_out) if k != 'ok')}); "
            f"one-time model + Ankh load {t_load:.1f}s wall / {c_load:.1f} cpu-s; "
            f"ankh {T['ankh'][0]:.1f}s wall / {T['ankh'][1]:.1f} cpu-s; featurize {T['featurize'][0]:.1f}s wall / "
            f"{T['featurize'][1]:.1f} cpu-s; encode (batch {a.batch}) {T['encode'][0]:.1f}s wall / {T['encode'][1]:.1f} "
            f"cpu-s; total loop {t_all:.1f}s wall / {c_all:.1f} cpu-s")
    open(os.path.join(a.out, f"lenseek_prep_{tag}_summary.txt"), "w").write(summ + "\n")
    print(summ, flush=True)
    print("LENSEEK_PREP_DONE", flush=True)


def blas_threads(n):
    """Run this command again (unbuffered) with n OpenMP / BLAS threads unless they are set already: numpy reads them
    when it is imported."""
    env = {v: str(n) for v in THREAD_VARS}
    if any(os.environ.get(k) != x for k, x in env.items()):
        os.execve(sys.executable, [sys.executable, "-u"] + sys.argv, dict(os.environ, **env))


# ---------------------------------------------------------------- DeeplyTough
def dt_timing(a):
    """DeeplyTough per query: a task file with one line per sampled query (sid q<index>:<pdbid>, as in
    scripts/deeplytough.py tasks), run by scripts/deeplytough.py run with --workers processes, the network on --threads
    threads and as many OpenMP / BLAS threads."""
    queries, qs = sample_queries(a.queries, a.n_queries, a.seed)
    os.makedirs(a.out, exist_ok=True)
    tasks = os.path.join(a.out, "dt_queries.jsonl")
    with open(tasks, "w") as f:
        for i in qs:
            f.write(json.dumps({"sid": f"q{i}:{os.path.basename(os.path.dirname(queries[i]))}",
                                "protein": pocket_protein(queries[i]), "pockets": [["q", i, queries[i]]]}) + "\n")
    subprocess.run([sys.executable, os.path.join(HERE, "deeplytough.py"), "run", "--tasks", tasks, "--tag", "q",
                    "--workers", str(a.workers), "--threads", str(a.threads), "--out", a.out],
                   env=dict(os.environ, **{v: str(a.threads) for v in THREAD_VARS}), check=True)


def dt_net(a):
    """The DeeplyTough network alone on one random 8 x 24^3 input (batch 1) for each thread count: one warm-up pass,
    then the mean wall time of three passes."""
    import contextlib
    import io
    import torch
    home = os.environ.get("DEEPLYTOUGH_HOME", "")
    net = os.path.join(home, DT_NETWORK)
    if not os.path.isfile(net):
        die(f"DEEPLYTOUGH_HOME={home!r} is not a DeeplyTough checkout ({DT_NETWORK} missing)")
    sys.path.insert(0, os.path.join(home, "deeplytough"))
    from engine.predictor import load_model
    with contextlib.redirect_stdout(io.StringIO()):   # create_model prints the whole network
        m, _ = load_model(net, "cpu")
    m.eval()
    sec = {}
    for th in a.threads:
        torch.set_num_threads(th)
        x = torch.from_numpy(np.random.RandomState(0).rand(1, 8, 24, 24, 24).astype(np.float32))
        with torch.no_grad():
            m(x[:1])  # warm-up
            t0 = time.time()
            for _ in range(3):
                m(x)
            sec[f"threads_{th}_s"] = (time.time() - t0) / 3
        print("threads %d batch 1: %.3f s per pocket" % (th, sec[f"threads_{th}_s"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dt_net.tsv"), "w") as f:
        f.write("# host %s cpu %s\n" % (platform.node(), cpu_model()))
        f.write("\t".join(sec) + "\n" + "\t".join("%.6f" % v for v in sec.values()) + "\n")


def dt_search(a):
    """DeeplyTough search per query: distances of one descriptor to every database descriptor (bag_distances in
    float64) + top-100 with database-index tie-break; also the one-off cost of loading the database matrix."""
    blas_threads(a.threads)
    q = np.load(a.q_desc).astype(np.float64)
    t1 = time.time()
    db = np.load(a.db_desc)
    ok = np.isfinite(db).all(1)
    D = db[ok].astype(np.float64)
    idx = np.where(ok)[0]
    D2 = (D ** 2).sum(1)
    t_load = time.time() - t1
    rows = []
    for i in np.where(np.isfinite(q).all(1))[0]:
        s = time.perf_counter()
        x = q[i]
        d = np.sqrt(np.clip((x ** 2).sum() + D2 - 2.0 * (D @ x), 0, None) + 1e-8)
        idx[np.lexsort((idx, d))[:100]]   # the top-100 is part of the timed search
        rows.append((i, time.perf_counter() - s))
    cpu = cpu_model()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "search_timing.tsv"), "w") as f:
        f.write("# host %s cpu %s OMP_NUM_THREADS=%s db=%d load_db_s=%.3f\n" % (
            platform.node(), cpu, os.environ.get("OMP_NUM_THREADS"), len(idx), t_load))
        f.write("query_idx\tsearch_s\n")
        for i, t in rows:
            f.write("%d\t%.6f\n" % (i, t))
    t = np.array([r[1] for r in rows])
    print("host %s (%s): %d queries x %d DB descriptors, per query median %.4f s, mean %.4f s, max %.4f s; "
          "load DB %.2f s" % (platform.node(), cpu, len(rows), len(idx), np.median(t), t.mean(), t.max(), t_load))


def dt_prep(a):
    """Worker time of the DeeplyTough featurization (protein preparation, voxelization, network on one thread),
    summed over the structures with at least one descriptor, for the queries (timing_q.tsv) and the database
    (timing_db_*.tsv, timing_missing*.tsv). A table that occurs in several run folders is counted once (first
    folder)."""
    tabs, seen = [], set()
    for d in a.runs:
        for fn in sorted(glob.glob(os.path.join(d, "timing_*.tsv"))):
            tag = os.path.basename(fn)[7:-4]
            if tag not in seen:
                seen.add(tag)
                tabs.append(pd.read_csv(fn, sep="\t", dtype={"route": str}).assign(tag=tag))
    t = pd.concat(tabs, ignore_index=True)
    rows = []
    for name, sel in [("queries", t.tag.eq("q")), ("database", t.tag.str.startswith(("db_", "missing")))]:
        s = t[sel]
        ok = s[s.nok > 0]
        tf, tv, tn = (ok[c].to_numpy(dtype=float) for c in ("t_feat", "t_vox", "t_net"))
        rows.append(dict(set=name, structures=len(s), structures_with_descriptor=len(ok),
                         pockets_requested=int(s.npockets.sum()), descriptors=int(ok.nok.sum()),
                         prep_core_h=tf.sum() / 3600, vox_core_h=tv.sum() / 3600, net_core_h=tn.sum() / 3600,
                         total_core_h=(tf.sum() + tv.sum() + tn.sum()) / 3600))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "dt_prep.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- Folddisco
def folddisco_timing(a):
    """Folddisco: the pocket index built in a run of its own, then the sampled queries in index order, one Folddisco
    process each (see the module docstring)."""
    from folddisco import build_index, query_one
    _, qs = sample_queries(a.queries, a.n_queries, a.seed)
    n_db = len(read_list(a.db))
    work, out = os.path.abspath(a.work), os.path.abspath(a.out)
    qdir = os.path.join(out, "folddisco_queries")
    os.makedirs(qdir, exist_ok=True)
    head = f"# host {socket.gethostname()} cpu {cpu_model()} threads {a.threads}\n"
    idx = os.path.join(out, "folddisco_index", "pockets")
    ix = build_index(work, "pockets", a.threads, idx)
    with open(os.path.join(out, "folddisco_index.tsv"), "w") as f:
        f.write(head + "\t".join(ix) + "\n" + "\t".join(str(v) for v in ix.values()) + "\n")
    t = pd.DataFrame([query_one("prefilter", q, None, work, idx, os.path.join(qdir, f"q{q:03d}.tsv"), a.threads, n_db)
                      for q in qs])
    t["n_returned"] = np.where(t.n_lines > 0, t.n_lines - 1, -1)
    with open(os.path.join(out, "folddisco_timing.tsv"), "w") as f:
        f.write(head)
        t.drop(columns="n_lines").to_csv(f, sep="\t", index=False)
    print(f"SUMMARY folddisco ({a.threads} threads): index {ix['n_files']} files, wall {ix['wall_s']:.1f}s, CPU "
          f"{ix['cpu_s']:.0f}s, peak {ix['max_rss_kb'] / 1e6:.1f} GB; per query ({len(t)} queries, non-zero exit "
          f"{int((t.rc != 0).sum())}): wall mean {t.wall_s.mean():.3f}s median {t.wall_s.median():.3f}s, database sites "
          f"returned median {t.n_returned.median():.0f}")


def check_inputs(a):
    """Stop with one '[error]' line at the first input of the subcommand that does not exist."""
    if getattr(a, "checkpoint", None) and not os.path.exists(a.checkpoint):
        die(f"--checkpoint {a.checkpoint} does not exist (LENSEEK_CKPT in examples/analysis.sh; CHECKPOINTS.md)")
    for name, (files, step) in INPUTS.items():
        value = getattr(a, name, None)
        flag = "--" + name.replace("_", "-")
        for path in [value] if isinstance(value, str) else value or []:
            for p in [os.path.join(path, f) for f in files] or [path]:
                if not os.path.exists(p):
                    die(f"{flag}: {p} does not exist (written by `{step}` below $LENSEEK_OUT, or give {flag})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True   # the keyword argument needs Python 3.7; the DeeplyTough modes run in Python 3.6
    out = os.path.join(OUT_ROOT, "timing")

    def common(p, queries=True, threads=16):
        if queries:
            p.add_argument("--queries", default=os.path.join(SITES, "all_query_sites.txt"),
                           help="query pocket files, one per line (protein.pdb next to each)")
            p.add_argument("--n-queries", type=int, default=20)
            p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=threads)
        p.add_argument("--out", default=out)

    for name, fn in (("probis-prep", probis_prep), ("probis", probis_db), ("glosa", glosa_full)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--db", default=os.path.join(SITES, "all_db_sites.txt"), help="database pocket files")
        common(p, queries=name != "probis-prep")
    p = sub.add_parser("lenseek")
    p.set_defaults(fn=lenseek_timing)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-db", required=True, help="CSR latent database: db_res.npy, db_off.npy, db_norm.npy")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--ankh-dir", default=None, help="local copy of ElnaggarLab/ankh-base")
    common(p)
    p = sub.add_parser("lenseek-prep")
    p.set_defaults(fn=lenseek_db_prep_timing)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--db", default=os.path.join(SITES, "all_db_sites.txt"), help="database pocket files")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--sample", type=int, default=0, help="random sample of this many entries (0: every entry)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ankh-dir", default=None, help="local copy of ElnaggarLab/ankh-base")
    common(p, queries=False)
    p = sub.add_parser("deeplytough")
    p.set_defaults(fn=dt_timing)
    common(p, threads=1)
    p.add_argument("--workers", type=int, default=1, help="processes of the DeeplyTough pool")
    p = sub.add_parser("dt-net")
    p.set_defaults(fn=dt_net)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 16])
    p.add_argument("--out", default=out)
    p = sub.add_parser("dt-search")
    p.set_defaults(fn=dt_search)
    dt = os.path.join(OUT_ROOT, "deeplytough")
    p.add_argument("--q-desc", default=os.path.join(dt, "q_desc.npy"))
    p.add_argument("--db-desc", default=os.path.join(dt, "db_desc.npy"))
    p.add_argument("--threads", type=int, default=16, help="OpenMP / BLAS threads")
    p.add_argument("--out", default=out)
    p = sub.add_parser("dt-prep")
    p.set_defaults(fn=dt_prep)
    p.add_argument("--runs", nargs="+", default=[os.path.join(dt, "runs")],
                   help="DeeplyTough run folders with timing_*.tsv (scripts/deeplytough.py run)")
    p.add_argument("--out", default=out)
    p = sub.add_parser("folddisco")
    p.set_defaults(fn=folddisco_timing)
    p.add_argument("--db", default=os.path.join(SITES, "all_db_sites.txt"), help="database pocket files")
    p.add_argument("--work", default=os.path.join(OUT_ROOT, "folddisco", "work"),
                   help="rewritten files of scripts/folddisco.py prepare (on a local disk)")
    common(p)
    a = ap.parse_args()
    check_inputs(a)
    if a.cmd in ("probis-prep", "probis", "glosa"):
        os.makedirs(a.out, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
