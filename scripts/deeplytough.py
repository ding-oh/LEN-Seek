"""DeeplyTough baseline: descriptors, distance matrix and top-100 list, and the checks of Supplementary Section S6.

DeeplyTough (Simonovsky and Meyers, J. Chem. Inf. Model. 2020) is run with its public code and the TOUGH-M1 network on
the evaluation queries and the search database of LEN-Seek. The outputs of `assemble` (dist.npy, db_desc.npy,
failures.tsv) and `top100` (top100_pairs.tsv, labelled by scripts/labels.py) give the DeeplyTough rows of the
Supplementary tables; the other subcommands give the numbers of Supplementary Section S6.

  tasks        one JSON line per structure with all of its pockets (queries: q.jsonl; database: db_00..db_15.jsonl,
               structures shuffled with seed 0 and dealt round-robin)
  run          128-d descriptor per pocket with the DeeplyTough code path: protein.pdb -> misc.utils.remove_water_and_hets
               -> MGLTools prepare_receptor4.py -U nphs_lps_waters -A hydrogens (obabel -h fallback) -> HTMD atom-type
               channels x radii; pocket -> convex-hull centroid (misc.utils.center_from_pdb_file); 8 x 24^3 volume
               (VoxelizedDataset._extract_volume, 1 A voxels) -> network -> L2 normalisation. The BsitePDB files are
               first rewritten in standard PDB columns (normalize_pdb; atoms and coordinates unchanged, altLoc blanked)
               and atoms that cannot reach the box are left out (same volume). One process per structure.
  assemble     descriptors of several run directories (first one wins) in the row order of q_paths.txt /
               db_paths.txt; dist.npy = engine.predictor.bag_distances, sqrt(max(|x|^2 + |y|^2 - 2 x.y, 0) + 1e-8),
               float64 -> float32, NaN for sites without a descriptor; failures.tsv gives the reason of each;
               dist_quartiles.txt: quartiles of the distances over all query-database pairs with a descriptor
  top100       each query's 100 nearest database sites (ascending distance, ties to the lower database index; sites
               without a descriptor never listed), in the pair-list format of scripts/labels.py pairs
  tough-check  the network on the TOUGH-M1 test fold with the authors' preprocessed inputs and their code
               (matchers.DeeplyTough); ROC-AUC against the published scores; quartiles of the descriptor distances
               (as bag_distances) over all pairs of the test-fold pockets
  box          share of each query ligand's heavy atoms inside the voxelised 24 A box
  overlap      highest sequence identity of a query site's chains to any TOUGH-M1 structure (MMseqs2 easy-search -s 7.5
               --max-seqs 2000 -e 10, hits with query or target coverage >= 0.5; all TOUGH-M1 structures, an upper
               bound on the training fold); sites and proteins at >= 30 % and >= 80 %
  ood          ligand heavy atoms, chains of the residues within 4.5 A of the ligand and chains of the structure, for
               TOUGH-M1 and the queries

Inputs: LENSEEK_EVAL_DIR (q_paths.txt, db_paths.txt; or --eval-dir), LENSEEK_QUERY_ROOT and LENSEEK_BSDB_ROOT (BsitePDB
layout <root>/<pdb>/{protein.pdb, ligand_pocket_<n>_<chains>.pdb, ligand_<n>_<chains>.pdb}); for run, tough-check and box
the DeeplyTough environment: DEEPLYTOUGH_HOME = checkout of https://github.com/BenevolentAI/DeeplyTough at 7536c5bb
(network networks/deeplytough_toughm1_test.pth.tar, md5 0e17028117fb2415d749837b0decb9a0), Python 3.6.15, htmd 1.13.10,
the pinned requirements.txt, mdtraj 1.9.3, se3cnn 6b976bea, lie_learn 0746908, openbabel 3.1.1 (OBABEL); MGL_ENV = a
separate environment with bioconda mgltools=1.5.6 alone (bin/pythonsh); LENSEEK_TMP scratch. se3cnn caches its kernel bases in
./cache: run from a writable local directory. TOUGH-M1 checks (--sanity-dir S): S/dl/lists/TOUGH-M1_{positive,pocket}.list
(OSF lists), S/dl/official/ext/TOUGH-M1_dataset (official dataset, <code5>/<code5>.pdb, <code5>00.pdb), S/tough (DeeplyTough
Zenodo dt_tough.zip: TOUGH-M1/TOUGH-M1_dataset/<code5>/<code5>_out/pockets/, processed/htmd/...). overlap needs MMSEQS
(MMseqs2 18.8cc5c; default mmseqs) and Biopython; ood needs scipy only.
Outputs ($LENSEEK_OUT/deeplytough/, default ./out/deeplytough/): tasks/, runs/{desc,fail,timing}_<tag>.*, q_desc.npy,
db_desc.npy, dist.npy, q_paths.txt, db_paths.txt, failures.tsv, dist_quartiles.txt, top100_pairs.tsv, tough_check/,
box/, overlap/, ood/.

    python scripts/deeplytough.py tasks
    python scripts/deeplytough.py run --tasks out/deeplytough/tasks/q.jsonl --tag q --workers 16   (DeeplyTough env)
    OMP_NUM_THREADS=1 python scripts/deeplytough.py assemble out/deeplytough/runs                  (DeeplyTough env)
    python scripts/deeplytough.py top100
    python scripts/deeplytough.py tough-check --sanity-dir S                                      (DeeplyTough env)
"""
import argparse
import collections
import glob
import importlib
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import REPO, read_list, site_key, under  # noqa: E402

OUT = os.path.join(os.environ.get("LENSEEK_OUT", "out"), "deeplytough")
DT_ROOT = os.environ.get("DEEPLYTOUGH_HOME", "")
if DT_ROOT:
    sys.path.insert(0, os.path.join(DT_ROOT, "deeplytough"))
MGL_ENV = os.environ.get("MGL_ENV", "")
PYTHONSH = os.path.join(MGL_ENV, "bin", "pythonsh")
PREP_REC = os.path.join(MGL_ENV, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
OBABEL = os.environ.get("OBABEL", shutil.which("obabel") or "obabel")
TOUGH_TEST = os.path.join("results", "ToughM1-DeeplyTough-deeplytough_toughm1_test.csv")
NETWORK = os.path.join("networks", "deeplytough_toughm1_test.pth.tar")
BOX = 24
_G = {}


# ------------------------------------------------------------------ paths
def env_dir(var, what=""):
    v = os.environ.get(var)
    if not v:
        sys.exit("[error] environment variable %s is not set%s" % (var, (" (" + what + ")") if what else ""))
    if not os.path.isdir(v):
        sys.exit("[error] %s=%r is not a directory" % (var, v))
    return os.path.abspath(v)


def eval_dir(arg=None):
    """Directory with q_paths.txt / db_paths.txt: the argument if given, else $LENSEEK_EVAL_DIR."""
    return arg or env_dir("LENSEEK_EVAL_DIR", "directory with q_paths.txt and db_paths.txt")


def pdb_id(p):
    return site_key(p).split("/")[0]


def deeplytough_home():
    d = env_dir("DEEPLYTOUGH_HOME", "DeeplyTough checkout")
    if not os.path.isdir(os.path.join(d, "deeplytough")):
        sys.exit("[error] DEEPLYTOUGH_HOME=%r has no deeplytough/ package directory" % d)
    return d


def ligand_file(pocket):
    return os.path.join(os.path.dirname(pocket), os.path.basename(pocket).replace("ligand_pocket_", "ligand_"))


# ------------------------------------------------------------------ tasks
def cmd_tasks(a):
    ev = eval_dir(a.eval_dir)
    qroot = env_dir("LENSEEK_QUERY_ROOT", "query structures")
    dbroot = env_dir("LENSEEK_BSDB_ROOT", "database structures")
    q, db = read_list(os.path.join(ev, "q_paths.txt")), read_list(os.path.join(ev, "db_paths.txt"))
    os.makedirs(a.out, exist_ok=True)

    def group(paths, st, root):
        g = collections.OrderedDict()
        for i, p in enumerate(paths):
            mp_ = under(p, root)
            g.setdefault(os.path.dirname(mp_), []).append([st, i, mp_])
        return [{"sid": st + ":" + os.path.basename(d), "protein": os.path.join(d, "protein.pdb"), "pockets": v}
                for d, v in g.items()]

    def dump(tasks, fn):
        with open(os.path.join(a.out, fn), "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
    tq = group(q, "q", qroot)
    dump(tq, "q.jsonl")
    tdb = group(db, "db", dbroot)
    random.Random(0).shuffle(tdb)
    for s in range(a.nshard):
        dump(tdb[s::a.nshard], "db_%02d.jsonl" % s)
    print("q structures", len(tq), "pockets", len(q), "| db structures", len(tdb), "pockets", len(db), "| shards", a.nshard)


# ------------------------------------------------------------------ run (DeeplyTough environment)
def _init_worker(net, threads):
    import torch
    torch.set_num_threads(threads)
    if "model" not in _G:  # normally inherited from the parent through fork (no NFS reads per worker)
        import contextlib
        import io
        from engine.predictor import load_model
        with contextlib.redirect_stdout(io.StringIO()):  # create_model prints the whole network
            model, args = load_model(net, "cpu")
        model.eval()
        _G.update(model=model, args=args)


_NUM3 = re.compile(r"-?\d+\.\d{3}")
_NUM2 = re.compile(r"-?\d+\.\d{2}")


def normalize_pdb(src, dst):
    """Rewrite a BsitePDB file as column-exact PDB. BsitePDB puts x/y/z/occ/B one column left of the PDB standard, so a
    column parser reads x <= -100 A as positive and fails on y/z <= -100 A; coordinates are %8.3f, so the first three
    3-decimal numbers after the residue number are unambiguous even when fields touch. The altLoc column is blanked (a
    stray 'A' makes MGLTools emit 5-character hydrogen names that HTMD cannot read).
    Returns (n_atoms, n_lines that a standard-column reader would read differently or reject, n_altloc_blanked)."""
    out, n, nbad, nalt = [], 0, 0, 0
    for L in open(src):
        if not L.startswith(("ATOM  ", "HETATM")):
            if L.startswith(("TER", "END", "MODEL", "ENDMDL")):
                out.append(L.rstrip("\n")[:80])
            continue
        L = L.rstrip("\n")
        m = list(_NUM3.finditer(L, 26))[:3]
        if len(m) < 3:
            raise ValueError("cannot parse coordinates: %r" % L)
        x, y, z = (float(v.group()) for v in m)
        rest = L[m[2].end():]
        oc = _NUM2.findall(rest)[:2]
        occ = float(oc[0]) if len(oc) > 0 else 1.0
        bf = float(oc[1]) if len(oc) > 1 else 0.0
        el = L[76:78].strip() if len(L) >= 77 else ""
        if not el.isalpha():
            el = "".join(c for c in L[12:14] if c.isalpha())[:1]
        try:
            std = (float(L[30:38]), float(L[38:46]), float(L[46:54]))
            if std != (x, y, z):
                nbad += 1
        except ValueError:
            nbad += 1
        alt = L[16]
        if alt != " ":
            nalt += 1
        out.append("%-6s%5s %-4s %3s %1s%4s%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (
            L[0:6], L[6:11], L[12:16], L[17:20], L[21], L[22:26], L[26] if L[26].isalpha() else " ",
            x, y, z, occ, bf, el.upper()[:2]))
        n += 1
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\nEND\n")
    return n, nbad, nalt


def _mgl(pdb_in, outdir, timeout):
    cmd = [PYTHONSH, PREP_REC, "-r", pdb_in, "-U", "nphs_lps_waters", "-A", "hydrogens"]
    r = subprocess.run(cmd, cwd=outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("prepare_receptor4 rc=%d: %s" % (r.returncode, r.stdout.decode(errors="replace")[-300:]))


def featurize_protein(pdb_path, tmp, timeout):
    """misc.utils.htmd_featurizer for one protein: (channels, coords, route)."""
    import htmd.molecule.molecule as htmdmol
    import htmd.molecule.voxeldescriptors as htmdvox
    from misc.utils import remove_water_and_hets

    outdir = os.path.join(tmp, "out")
    os.makedirs(outdir)  # (tmp is created here when missing)
    d1 = os.path.join(tmp, "in1")
    os.makedirs(d1)
    pdb1 = os.path.join(d1, os.path.basename(pdb_path))
    remove_water_and_hets(pdb_path, pdb1)
    route = "mgl"
    try:
        _mgl(pdb1, outdir, timeout)
    except Exception:
        route = "obabel+mgl"
        d2 = os.path.join(tmp, "in2")
        os.makedirs(d2)
        pdb2 = os.path.join(d2, os.path.basename(pdb_path))
        subprocess.run([OBABEL, pdb1, "-O", pdb2, "-h"], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout)
        _mgl(pdb2, outdir, timeout)
    pdbqt = os.path.join(outdir, os.path.basename(pdb_path)) + "qt"
    if not os.path.exists(pdbqt) and os.path.exists(pdbqt.replace(".pdb", "_model1.pdb")):
        os.rename(pdbqt.replace(".pdb", "_model1.pdb"), pdbqt)
    mol = htmdmol.Molecule(pdbqt)
    channels = htmdvox._getAtomtypePropertiesPDBQT(mol)
    sigmas = htmdvox._getRadii(mol)
    channels = sigmas[:, np.newaxis] * channels.astype(float)
    coords = mol.coords[:, :, mol.frame]
    return channels, coords, route


class _Vox:
    """VoxelizedDataset._extract_volume / _getOccupancyC without a dataset; bound by _bind_vox() in the parent before the
    worker pool is forked, so this module imports without DeeplyTough on the path."""
    _getOccupancyC = None
    extract = None


def _bind_vox():
    if _Vox.extract is None:
        from engine.datasets import VoxelizedDataset as _VD
        _Vox._getOccupancyC = staticmethod(_VD._getOccupancyC)
        _Vox.extract = _VD._extract_volume


def process(task):
    import torch
    import torch.nn.functional as nnf
    from misc.utils import center_from_pdb_file
    _bind_vox()

    model, args = _G["model"], _G["args"]
    box = args.patch_size
    res = []   # (set, idx, desc or None, center or None, status)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="dt_", dir=os.environ.get("LENSEEK_TMP", None))
    feat_err, route, natoms, diag = None, "", 0, [0, 0, 0, 0, 0]
    try:
        try:
            if not os.path.exists(task["protein"]):
                raise FileNotFoundError("protein missing")
            os.makedirs(os.path.join(tmp, "norm"))
            prot = os.path.join(tmp, "norm", "protein.pdb")
            diag[0], diag[1], diag[2] = normalize_pdb(task["protein"], prot)
            channels, coords, route = featurize_protein(prot, os.path.join(tmp, "feat"), timeout=1800)
            natoms = coords.shape[0]
        except Exception as e:
            feat_err = "%s: %s" % (type(e).__name__, str(e).replace("\t", " ").replace("\n", " ")[:300])
        t_feat = time.time() - t0

        vols, keep = [], []
        t1 = time.time()
        for k, (st, idx, pocket) in enumerate(task["pockets"]):
            if feat_err is not None:
                res.append((st, idx, None, None, "featurize_failed|" + feat_err))
                continue
            if not os.path.exists(pocket):
                res.append((st, idx, None, None, "pocket_file_missing"))
                continue
            pk = os.path.join(tmp, "norm", "pocket_%d.pdb" % k)
            try:
                _, b, _ = normalize_pdb(pocket, pk)
                diag[3] += b
                diag[4] += int(b > 0)
            except Exception as e:
                res.append((st, idx, None, None, "pocket_parse_failed|%s" % str(e)[:200]))
                continue
            center = center_from_pdb_file(pk)
            if center is None:
                res.append((st, idx, None, None, "pocket_center_none"))
                continue
            center = np.asarray(center, dtype=np.float64)
            # atoms farther than (box half-diagonal + 8 A) from the centre cannot reach any voxel (5 A truncation)
            r = 0.5 * np.sqrt(3) * box + 8.0
            m = np.sum((coords.astype(np.float64) - center) ** 2, axis=1) < r * r
            c, ch = coords[m], channels[m]
            if c.shape[0] == 0:
                res.append((st, idx, None, center, "empty_box"))
                continue
            vol, _, _ = _Vox.extract(_Vox, c, ch, center, [box] * 3, 1.0)
            vols.append(vol)
            keep.append(len(res))
            res.append((st, idx, None, center, "ok"))
        t_vox = time.time() - t1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    t2 = time.time()
    if vols:
        with torch.no_grad():
            x = torch.from_numpy(np.stack(vols))
            out = []
            for b in range(0, x.shape[0], 30):
                o = model(x[b:b + 30])
                if args.l2_normed_descriptors:
                    o = nnf.normalize(o)
                out.append(o.cpu().float())
            out = torch.cat(out).view(len(vols), -1).numpy()
        for k, j in enumerate(keep):
            st, idx, _, center, status = res[j]
            res[j] = (st, idx, out[k], center, status)
    t_net = time.time() - t2
    timing = dict(sid=task["sid"], natoms=natoms, npockets=len(task["pockets"]), nok=len(vols), route=route,
                  t_feat=t_feat, t_vox=t_vox, t_net=t_net, prot_atoms_in=diag[0], prot_misparse=diag[1],
                  prot_altloc=diag[2], pocket_misparse=diag[3], pockets_with_misparse=diag[4])
    return res, timing


def cmd_run(a):
    if not DT_ROOT or not os.path.isdir(os.path.join(DT_ROOT, "deeplytough")):
        sys.exit("[error] DEEPLYTOUGH_HOME=%r is not a DeeplyTough checkout (deeplytough/ missing)" % DT_ROOT)
    if not MGL_ENV or not os.path.isfile(PYTHONSH) or not os.path.isfile(PREP_REC):
        sys.exit("[error] MGL_ENV=%r is not an MGLTools environment (bin/pythonsh and prepare_receptor4.py needed)"
                 % MGL_ENV)
    net = os.path.join(DT_ROOT, NETWORK)
    tasks = [json.loads(l) for l in open(a.tasks)]
    # largest proteins first (load balance; output order does not matter)
    tasks.sort(key=lambda t: -(os.path.getsize(t["protein"]) if os.path.exists(t["protein"]) else 0))
    os.makedirs(a.out, exist_ok=True)
    print("tasks", len(tasks), "pockets", sum(len(t["pockets"]) for t in tasks), "workers", a.workers,
          "net", net, flush=True)
    sets, idxs, descs, centers, oks = [], [], [], [], []
    ftsv = open(os.path.join(a.out, "fail_%s.tsv" % a.tag), "w")
    ttsv = open(os.path.join(a.out, "timing_%s.tsv" % a.tag), "w")
    ttsv.write("sid\tnatoms\tnpockets\tnok\troute\tt_feat\tt_vox\tt_net\tprot_atoms_in\tprot_misparse\tprot_altloc"
               "\tpocket_misparse\tpockets_with_misparse\n")
    # The model and every module the workers need are loaded here and inherited through fork (workers recycled every
    # 200 tasks); no forward pass in the parent, so no OpenMP pool exists at fork time.
    import contextlib
    import io
    import torch
    for mod in ("torch.nn.functional", "htmd.molecule.molecule", "htmd.molecule.readers",
                "htmd.molecule.voxeldescriptors", "mdtraj", "Bio.PDB", "misc.utils"):
        importlib.import_module(mod)
    from engine.predictor import load_model
    _bind_vox()
    torch.set_num_threads(a.threads)
    with contextlib.redirect_stdout(io.StringIO()):
        _m, _a = load_model(net, "cpu")
    _m.eval()
    _G.update(model=_m, args=_a)
    T0 = last_print = time.time()
    with mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(net, a.threads),
                                     maxtasksperchild=200) as pool:
        for n, (res, tm) in enumerate(pool.imap_unordered(process, tasks, chunksize=1)):
            ttsv.write("{sid}\t{natoms}\t{npockets}\t{nok}\t{route}\t{t_feat:.3f}\t{t_vox:.3f}\t{t_net:.3f}\t{prot_atoms_in}"
                       "\t{prot_misparse}\t{prot_altloc}\t{pocket_misparse}\t{pockets_with_misparse}\n".format(**tm))
            for st, idx, d, c, status in res:
                sets.append(st)
                idxs.append(idx)
                ok = d is not None
                oks.append(ok)
                descs.append(d if ok else np.full(128, np.nan, np.float32))
                centers.append(c if c is not None else np.full(3, np.nan))
                if not ok:
                    ftsv.write("%s\t%d\t%s\t%s\n" % (st, idx, tm["sid"], status))
            ftsv.flush()
            ttsv.flush()
            if n % 100 == 0 or time.time() - last_print > 300:
                last_print = time.time()
                print("%d/%d proteins  %.0f s" % (n + 1, len(tasks), time.time() - T0), flush=True)
    ftsv.close()
    ttsv.close()
    np.savez(os.path.join(a.out, "desc_%s.npz" % a.tag), set=np.array(sets), idx=np.array(idxs, dtype=np.int64),
             desc=np.stack(descs).astype(np.float32), center=np.stack(centers).astype(np.float64),
             ok=np.array(oks, dtype=bool))
    print("done %d pockets (%d ok) in %.0f s" % (len(oks), sum(oks), time.time() - T0), flush=True)


# ------------------------------------------------------------------ assemble, top-100
def bag_distances(x, y):
    """engine.predictor.bag_distances in numpy float64."""
    x2 = (x ** 2).sum(1)[:, None]
    y2 = (y ** 2).sum(1)[None, :]
    return np.sqrt(np.clip(x2 + y2 - 2.0 * x @ y.T, 0, None) + 1e-8)


def quartiles(d):
    """25th, 50th and 75th percentiles of the finite values of d (float64)."""
    d = np.asarray(d, dtype=np.float64).ravel()
    return np.percentile(d[np.isfinite(d)], [25, 50, 75])


def cmd_assemble(a):
    ev = eval_dir(a.eval_dir)
    paths = {"q": read_list(os.path.join(ev, "q_paths.txt")), "db": read_list(os.path.join(ev, "db_paths.txt"))}
    desc = {st: np.full((len(p), 128), np.nan, np.float32) for st, p in paths.items()}
    have = {st: np.zeros(len(p), bool) for st, p in paths.items()}
    for rd in a.run_dirs:   # a pocket with a descriptor in several run directories keeps the first one
        for fn in sorted(glob.glob(os.path.join(rd, "desc_*.npz"))):
            z = np.load(fn)
            for st in ("q", "db"):
                m = (z["set"] == st) & z["ok"]
                for i, d in zip(z["idx"][m], z["desc"][m]):
                    if not have[st][i]:
                        desc[st][i], have[st][i] = d, True
            print(fn, len(z["idx"]), int(z["ok"].sum()))
    reasons = {}
    for rd in a.run_dirs:
        for fn in sorted(glob.glob(os.path.join(rd, "fail_*.tsv"))):
            for l in open(fn):
                st, idx, sid, status = l.rstrip("\n").split("\t", 3)
                reasons.setdefault((st, int(idx)), (sid, status))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "failures.tsv"), "w") as f:   # failures count only without any descriptor
        f.write("set\tidx\tpath\tstructure\treason\n")
        for st in ("q", "db"):
            for i in np.where(~have[st])[0]:
                sid, status = reasons.get((st, int(i)), ("-", "not_processed"))
                f.write("%s\t%d\t%s\t%s\t%s\n" % (st, i, paths[st][i], sid, status))
    for st in ("q", "db"):
        np.save(os.path.join(a.out, "%s_desc.npy" % st), desc[st])
        with open(os.path.join(a.out, "%s_paths.txt" % st), "w") as f:
            f.write("\n".join(paths[st]) + "\n")
    okq, okd = np.isfinite(desc["q"]).all(1), np.isfinite(desc["db"]).all(1)
    dist = np.full((len(okq), len(okd)), np.nan, np.float32)
    dist[np.ix_(okq, okd)] = bag_distances(desc["q"][okq].astype(np.float64),
                                           desc["db"][okd].astype(np.float64)).astype(np.float32)
    np.save(os.path.join(a.out, "dist.npy"), dist)
    line = "distance quartiles over %d query-database pairs: %.4f %.4f %.4f" % ((okq.sum() * okd.sum(),)
                                                                              + tuple(quartiles(dist)))
    open(os.path.join(a.out, "dist_quartiles.txt"), "w").write(line + "\n")
    print("descriptors: q %d/%d, db %d/%d; failures %d -> %s" % (okq.sum(), len(okq), okd.sum(), len(okd),
                                                                 (~okq).sum() + (~okd).sum(), a.out))
    print(line)


def cmd_top100(a):
    qroot = env_dir("LENSEEK_QUERY_ROOT", "query structures")
    dbroot = env_dir("LENSEEK_BSDB_ROOT", "database structures")
    q, db = read_list(os.path.join(a.dt_dir, "q_paths.txt")), read_list(os.path.join(a.dt_dir, "db_paths.txt"))
    D = np.load(os.path.join(a.dt_dir, "dist.npy"), mmap_mode="r")
    assert D.shape == (len(q), len(db))
    out = a.out or os.path.join(a.dt_dir, "top100_pairs.tsv")
    skipped = []
    with open(out, "w") as f:
        f.write("query_idx\tquery_path\trank\tdb_idx\ttarget_path\tdist\n")
        for i in range(len(q)):
            row = np.asarray(D[i], dtype=np.float64)
            cand = np.where(np.isfinite(row))[0]
            if not len(cand):         # query without a descriptor: no rows
                skipped.append(i)
                continue
            order = cand[np.argsort(row[cand], kind="stable")][:a.k]
            qp = under(q[i], qroot)
            for r, j in enumerate(order, 1):
                f.write("%d\t%s\t%d\t%d\t%s\t%.6f\n" % (i, qp, r, j, under(db[j], dbroot), row[j]))
    print("%s: %d queries x top%d (queries without descriptor: %s)" % (out, len(q) - len(skipped), a.k, skipped))


# ------------------------------------------------------------------ Supplementary Section S6 checks
def cmd_tough_check(a):
    from sklearn.metrics import roc_auc_score
    dt = deeplytough_home()
    from matchers import DeeplyTough
    S, root = a.sanity_dir, os.path.join(a.sanity_dir, "tough")
    pub = [(x, y, float(s)) for x, y, s in (l.strip().split(",") for l in open(os.path.join(dt, TOUGH_TEST)))]
    pos = {tuple(l.split()[:2]) for l in open(os.path.join(S, "dl", "lists", "TOUGH-M1_positive.list"))}
    pk = {l.split()[0]: int(l.split()[1]) for l in open(os.path.join(S, "dl", "lists", "TOUGH-M1_pocket.list"))}
    codes = sorted({p[0] for p in pub} | {p[1] for p in pub})
    labels = np.array([(x, y) in pos for x, y, _ in pub])
    pub_s = np.array([s for _, _, s in pub])
    entries = [{"code5": c, "protein_htmd": "%s/processed/htmd/TOUGH-M1/TOUGH-M1_dataset/%s/%s.npz" % (root, c, c),
                "pocket": "%s/TOUGH-M1/TOUGH-M1_dataset/%s/%s_out/pockets/pocket%d_vert.pqr" % (root, c, c, pk[c] - 1)}
               for c in codes]
    m = DeeplyTough(os.path.join(dt, "networks", "deeplytough_toughm1_test.pth.tar"), device="cpu", batch_size=30,
                    nworkers=a.workers)
    entries = m.precompute_descriptors(entries)
    td = {e["code5"]: e for e in entries}
    ours = np.asarray(m.pair_match([(td[x], td[y]) for x, y, _ in pub]), dtype=float)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "tough_scores.npy"), ours)
    ok = np.isfinite(ours)
    d = np.abs(ours[ok] - pub_s[ok])
    desc = np.stack([e["descriptor"].numpy()[0] for e in entries]).astype(np.float64)
    td_all = bag_distances(desc, desc)[np.triu_indices(len(desc), 1)]
    lines = ["TOUGH-M1 test fold: pairs %d, scored %d (score = -distance)" % (len(pub), ok.sum()),
             "  |score - published|: max %.3g, mean %.3g; Pearson r %.6f" % (
                 d.max(), d.mean(), np.corrcoef(ours[ok], pub_s[ok])[0, 1]),
             "  ROC-AUC published (same pairs) %.4f | this run %.4f   (positives %d, negatives %d)" % (
                 roc_auc_score(labels[ok], pub_s[ok]), roc_auc_score(labels[ok], ours[ok]), labels[ok].sum(),
                 (~labels[ok]).sum()),
             "  descriptor distance quartiles over all %d pairs of the %d pockets: %.4f %.4f %.4f" % (
                 (len(td_all), len(desc)) + tuple(quartiles(td_all)))]
    open(os.path.join(a.out, "tough_check.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def heavy_coords(fn):
    xyz = []
    for l in open(fn):
        if l.startswith(("ATOM", "HETATM")):
            el = l[76:78].strip() or l[12:14].strip()[0]
            if el.upper() != "H":
                xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return np.array(xyz)


def cmd_box(a):
    deeplytough_home()
    from misc.utils import center_from_pdb_file
    ev, qroot = eval_dir(a.eval_dir), env_dir("LENSEEK_QUERY_ROOT", "query structures")
    tmp = tempfile.mkdtemp()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i, p in enumerate(read_list(os.path.join(ev, "q_paths.txt"))):
        pf = under(p, qroot)
        pn, ln = os.path.join(tmp, "p.pdb"), os.path.join(tmp, "l.pdb")
        normalize_pdb(pf, pn)              # the same rewrite as `run`
        normalize_pdb(ligand_file(pf), ln)
        c = np.asarray(center_from_pdb_file(pn))
        lig = heavy_coords(ln)
        lo, hi = c - BOX // 2, c + (BOX // 2 - 1)   # voxel centres span lo..hi, each voxel +-0.5 A
        rows.append((i, p, len(lig), np.all((lig >= lo - 0.5) & (lig <= hi + 0.5), axis=1).mean()))
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.out, "box_check.tsv"), "w") as f:
        f.write("query_idx\tquery_path\tlig_heavy\tlig_in_box\n")
        f.writelines("%d\t%s\t%d\t%.6f\n" % r for r in rows)
    v = np.array([r[3] for r in rows])
    print("ligand heavy atoms inside the 24 A box: entirely inside %d/%d, >= 90%% %d, min %.3f" % (
        (v == 1).sum(), len(v), (v >= 0.9).sum(), v.min()))


def cmd_overlap(a):
    from Bio.Data.SCOPData import protein_letters_3to1
    ev = eval_dir(a.eval_dir)
    tough_dir = os.path.join(a.sanity_dir, "dl", "official", "ext", "TOUGH-M1_dataset")
    std = set("ACDEFGHIKLMNPQRSTVWY")
    os.makedirs(a.out, exist_ok=True)

    def seq_of(fn):   # first model, one letter per residue, modified residues as their parent, standard 20 only
        seen, seq = set(), []
        for L in open(fn):
            if L.startswith("ENDMDL"):
                break
            if not L.startswith(("ATOM  ", "HETATM")) or (L[21], L[22:27]) in seen:
                continue
            seen.add((L[21], L[22:27]))
            aa = protein_letters_3to1.get(L[17:20].strip().upper(), "")
            if len(aa) == 1 and aa.upper() in std:
                seq.append(aa.upper())
        return "".join(seq)
    tfa = os.path.join(a.out, "toughm1.fasta")
    if not os.path.exists(tfa):
        with open(tfa, "w") as f:
            for code in sorted(os.listdir(tough_dir)):
                fn = os.path.join(tough_dir, code, code + ".pdb")
                s = seq_of(fn) if os.path.exists(fn) else ""
                if len(s) >= 10:
                    f.write(">%s\n%s\n" % (code, s))
    m8 = os.path.join(a.out, "eval_vs_toughm1.m8")
    if not os.path.exists(m8):
        subprocess.run([os.environ.get("MMSEQS", "mmseqs"), "easy-search", a.eval_fasta, tfa, m8,
                        os.path.join(a.out, "tmp"), "--threads", str(a.threads), "-s", "7.5", "--max-seqs", "2000",
                        "-e", "10", "--format-output", "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hits = collections.defaultdict(list)
    for L in open(m8):
        q, t, fid, _, _, _, qc, tc = L.split("\t")[:8]
        if float(qc) >= 0.5 or float(tc) >= 0.5:
            hits[q].append((float(fid), t))
    rows, prots = [], collections.defaultdict(float)
    for i, p in enumerate(read_list(os.path.join(ev, "q_paths.txt"))):
        chains = os.path.basename(p)[:-4].split("_")[-1]   # ligand_pocket_<n>_<chains>.pdb
        v, tb = 0.0, ""
        for c in chains:
            for fid, t in hits.get("%s_%s" % (pdb_id(p), c), []):
                if fid > v:
                    v, tb = fid, t
        rows.append((i, p, chains, v, tb))
        prots[pdb_id(p)] = max(prots[pdb_id(p)], v)
    with open(os.path.join(a.out, "query_vs_toughm1.tsv"), "w") as f:
        f.write("query_idx\tquery_path\tchains\tmaxid\tbest_hit\n")
        f.writelines("%d\t%s\t%s\t%.3f\t%s\n" % r for r in rows)
    v, pv = [r[3] for r in rows], list(prots.values())
    line = "either coverage >= 0.5   sites >=30%%: %3d/%d, >=80%%: %3d/%d | proteins >=30%%: %3d/%d, >=80%%: %3d/%d" % (
        sum(x >= 0.3 for x in v), len(v), sum(x >= 0.8 for x in v), len(v), sum(x >= 0.3 for x in pv), len(pv),
        sum(x >= 0.8 for x in pv), len(pv))
    open(os.path.join(a.out, "overlap_summary.txt"), "w").write(line + "\n")
    print(line)


def heavy_atoms(fn, bsite):
    """Heavy atoms of the first model: coordinates and residue keys (chain, number, icode, name); BsitePDB files are read
    through normalize_pdb."""
    if bsite:
        fd, tmp = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        normalize_pdb(fn, tmp)
        fn = tmp
    xyz, keys = [], []
    for L in open(fn):
        if L.startswith("ENDMDL"):
            break
        if not L.startswith(("ATOM  ", "HETATM")):
            continue
        el = L[76:78].strip().upper() if len(L) > 77 else ""
        if el in ("H", "D") or (not el and L[12:16].strip()[:1] in ("H", "D")):
            continue
        try:
            xyz.append((float(L[30:38]), float(L[38:46]), float(L[46:54])))
        except ValueError:
            continue
        keys.append((L[21], L[22:26], L[26], L[17:20]))
    if bsite:
        os.remove(fn)
    return np.array(xyz, dtype=float).reshape(-1, 3), keys


def measure(job):
    """(set, ligand heavy atoms, chains of the residues within 4.5 A of the ligand, chains of the structure)."""
    from scipy.spatial import cKDTree
    tag, prot, lig, bsite = job
    try:
        L, _ = heavy_atoms(lig, bsite)
        P, pk = heavy_atoms(prot, bsite)
        res = {pk[j][:3] for h in cKDTree(P).query_ball_point(L, 4.5) for j in h}
        return tag, len(L), len({r[0] for r in res}), len({k[0] for k in pk})
    except Exception as e:  # noqa: BLE001
        return tag, None, str(e)[:80]


def cmd_ood(a):
    ev, qroot = eval_dir(a.eval_dir), env_dir("LENSEEK_QUERY_ROOT", "query structures")
    pk = {l.split()[0]: int(l.split()[1]) for l in open(os.path.join(a.sanity_dir, "dl/lists/TOUGH-M1_pocket.list"))}
    T = os.path.join(a.sanity_dir, "dl/official/ext/TOUGH-M1_dataset")
    jobs = [("toughm1", os.path.join(T, c, c + ".pdb"), os.path.join(T, c, c + "00.pdb"), False) for c in sorted(pk)]
    for p in read_list(os.path.join(ev, "q_paths.txt")):
        f = under(p, qroot)
        jobs.append(("queries", os.path.join(os.path.dirname(f), "protein.pdb"), ligand_file(f), True))
    with mp.Pool(a.workers) as pool:
        res = pool.map(measure, jobs, chunksize=20)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ood_per_site.tsv"), "w") as f:
        f.write("set\tlig_heavy\tpocket_chains\tstructure_chains\n")
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in res if r[1] is not None)
    lines = ["per-site failures: %d %s" % (sum(r[1] is None for r in res), [r for r in res if r[1] is None][:3])]
    for st in ("toughm1", "queries"):
        x = np.array([r[1:] for r in res if r[0] == st and r[1] is not None], dtype=float)
        lines.append("%-8s n=%d | ligand heavy atoms %.1f [%.1f-%.1f] | pocket spans >=2 chains: %.1f%% | structure "
                     "chains >1: %.1f%%" % (st, len(x), np.median(x[:, 0]), np.percentile(x[:, 0], 5),
                                            np.percentile(x[:, 0], 95), 100 * (x[:, 1] >= 2).mean(),
                                            100 * (x[:, 2] > 1).mean()))
    open(os.path.join(a.out, "ood_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("tasks", help="task files (one JSON line per structure)")
    s.add_argument("--eval-dir", default=None, help="q_paths.txt / db_paths.txt (default $LENSEEK_EVAL_DIR)")
    s.add_argument("--nshard", type=int, default=16)
    s.add_argument("--out", default=os.path.join(OUT, "tasks"))
    s = sub.add_parser("run", help="descriptors for the pockets of one task file (DeeplyTough environment)")
    s.add_argument("--tasks", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", default=os.path.join(OUT, "runs"))
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("assemble", help="descriptors in evaluation order, dist.npy, failures.tsv")
    s.add_argument("run_dirs", nargs="+", help="run output directories (desc_*.npz, fail_*.tsv); first one wins")
    s.add_argument("--eval-dir", default=None)
    s.add_argument("--out", default=OUT)
    s = sub.add_parser("top100", help="top-100 pair list for the ProBiS / G-LoSA labels")
    s.add_argument("--dt-dir", default=OUT, help="assemble output (default $LENSEEK_OUT/deeplytough)")
    s.add_argument("--out", default=None, help="default <dt-dir>/top100_pairs.tsv")
    s.add_argument("--k", type=int, default=100)
    for name, hlp in (("tough-check", "TOUGH-M1 test-fold ROC-AUC (DeeplyTough environment)"),
                      ("box", "query ligands inside the 24 A box (DeeplyTough environment)"),
                      ("overlap", "sequence identity of the query sites to TOUGH-M1"),
                      ("ood", "ligand size and chain make-up of TOUGH-M1 and the queries")):
        s = sub.add_parser(name, help=hlp)
        if name != "box":
            s.add_argument("--sanity-dir", required=True, help="TOUGH-M1 files (layout in the module docstring)")
        if name != "tough-check":
            s.add_argument("--eval-dir", default=None)
        if name == "overlap":
            s.add_argument("--eval-fasta", default=os.path.join(REPO, "data", "query_sets", "eval_chains.fasta"))
            s.add_argument("--threads", type=int, default=8)
        elif name != "box":
            s.add_argument("--workers", type=int, default=2 if name == "tough-check" else 4)
        s.add_argument("--out", default=os.path.join(OUT, name.replace("-", "_")))
    a = ap.parse_args()
    if not a.cmd:
        ap.error("a subcommand is required")
    {"tasks": cmd_tasks, "run": cmd_run, "assemble": cmd_assemble, "top100": cmd_top100, "tough-check": cmd_tough_check,
     "box": cmd_box, "overlap": cmd_overlap, "ood": cmd_ood}[a.cmd](a)


if __name__ == "__main__":
    main()
