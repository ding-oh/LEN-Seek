"""Helpers shared by the scripts in scripts/: input paths and site keys, chain sequences and sequence clusters,
tie-breaks, the staging and comparison helpers of ProBiS and G-LoSA, and the CSR Chamfer kernels of the scaling runs.

The scripts insert their own directory on sys.path and import what they need from here (`from pipeline import ...`).
analysis/common.py takes die, require_path, site_key, read_tsv, tie_key, strict_order and TIE_SEED from here, so that a
producer and a result script join their files by the same site key and break ties of a ranking the same way.
scripts/deeplytough.py imports it in the Python 3.6 environment of DeeplyTough, so the module level uses the standard
library, numpy and torch only (gemmi is imported inside the function that needs it).
"""
import collections
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ---------------------------------------------------------------- inputs, site keys, logging
def die(msg):
    """Stop with one '[error]' line."""
    sys.exit(f"[error] {msg}")


def require_path(value, flag, var):
    """A required input path: given by flag or the environment variable var, and existing."""
    if not value:
        die(f"{flag} not given and ${var} is not set")
    if not os.path.exists(value):
        die(f"{flag} {value} does not exist (set {flag} or ${var})")
    return value


def site_key(path):
    """'<pdbid>/<pocket file>' of a site path: sites are matched by their last two path components (surrounding
    whitespace removed); ValueError with fewer than two components."""
    parts = str(path).strip().rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot derive <pdbid>/<file> from path: {path!r}")
    return "/".join(parts[-2:])


def under(p, root_dir):
    """The site `p` placed under `root_dir`."""
    return os.path.join(root_dir, site_key(p))


def read_list(fn):
    """Non-empty lines of a text file, stripped."""
    return [ln.strip() for ln in open(fn) if ln.strip()]


def read_tsv(path):
    """Rows of a TSV with a header row as dicts; '#' lines skipped, short rows padded with ''."""
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    return [dict(zip(rows[0], r + [""] * (len(rows[0]) - len(r)))) for r in rows[1:]]


def log(*a):
    """Print a time-stamped line with the 1-minute load average."""
    print(time.strftime("%m-%d %H:%M:%S"), f"[load {os.getloadavg()[0]:.1f}]", *a, flush=True)


def status_gb(field):
    """A memory field of /proc/self/status in GB (VmHWM peak, VmRSS current resident memory); -1 when absent."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) / 1e6
    return -1.0


def peak_rss_gb():
    """Peak resident memory of this process (GB)."""
    return status_gb("VmHWM")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


# ---------------------------------------------------------------- chain sequences and sequence clusters
AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
      "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
      "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}


def chain_seqs(pdb_file):
    """{chain: sequence} over the ATOM records, one letter per (chain, resSeq + iCode) in file order, non-standard
    residues X, chains of >= 10 residues."""
    seq = collections.OrderedDict()
    seen = set()
    for l in open(pdb_file, errors="replace"):
        if not l.startswith("ATOM"):
            continue
        key = (l[21], l[22:27])
        if key in seen:
            continue
        seen.add(key)
        seq.setdefault(l[21], []).append(AA.get(l[17:20].strip(), "X"))
    return {c: "".join(s) for c, s in seq.items() if len(s) >= 10}


def entry_chain_seqs(root, pdb_id):
    """[(<pdbid>_<chain>, sequence)] of <root>/<pdbid>/protein.pdb read with gemmi: tabulated amino acids, one-letter,
    the 20 standard letters, chains with >= 10 residues."""
    import gemmi
    st = gemmi.read_structure(os.path.join(root, pdb_id, "protein.pdb"))
    out = []
    for ch in st[0]:
        names = [r.name for r in ch if gemmi.find_tabulated_residue(r.name) is not None
                 and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        seq = "".join(c for c in gemmi.one_letter_code(names).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        if len(seq) >= 10:
            out.append((f"{pdb_id}_{ch.name}", seq))
    return out


def single_linkage(names, hits, pct, min_cov):
    """Single-linkage clusters of the chains `names`: two chains are linked by an MMseqs2 hit (rows query, target,
    fident, alnlen, qlen, tlen) with fident >= pct / 100 over >= min_cov of the shorter chain. Returns {chain: root
    chain of its connected component}."""
    parent = {c: c for c in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for q, t, fid, aln, ql, tl in hits:
        if q == t or q not in parent or t not in parent:
            continue
        if float(fid) >= pct / 100 and int(aln) >= min_cov * min(int(ql), int(tl)):
            ra, rb = find(q), find(t)
            if ra != rb:
                parent[ra] = rb
    return {c: find(c) for c in parent}


def site_chain(key, main_chain, chains, known):
    """'<pdb>_<chain>' of the first chain of a site that is in known: the main chain, then the other chains by
    binding-site residues (ties alphabetical); None when no chain is (site annotation columns key, main_chain,
    chains = 'A:12,B:3')."""
    pdb = key.split("/")[0]
    order = sorted((c.split(":") for c in chains.split(",") if c), key=lambda c: (-int(c[1]), c[0]))
    names = [main_chain] + [c for c, _ in order if c != main_chain]
    return next((f"{pdb}_{c}" for c in names if f"{pdb}_{c}" in known), None)


# ---------------------------------------------------------------- ranking with fixed tie-breaks
TIE_SEED = 1


def tie_key(qi, n):
    """Fixed random tie-break key of query qi over n database sites (seed TIE_SEED, qi)."""
    return np.random.default_rng(np.random.SeedSequence([TIE_SEED, int(qi)])).random(n)


def strict_order(score, tie):
    """Indices sorted by (score desc, tie asc)."""
    return np.lexsort((tie, -score))


def topk_stable(row, k):
    """Indices of the k smallest values, ties broken by index."""
    k = min(k, len(row))
    part = np.argpartition(row, k - 1)[:k]
    thr = row[part].max()
    cand = np.flatnonzero(row <= thr)
    o = np.lexsort((cand, row[cand]))
    return cand[o[:k]]


# ---------------------------------------------------------------- ProBiS and G-LoSA: tools, staging, comparison
# AssignChemicalFeatures.java and AssignSecondaryStructures.java both define a top-level class `Atom`, so the helpers
# are compiled into their own class directory (cf/). The locale is pinned because AssignChemicalFeatures formats
# coordinates with java.text.DecimalFormat.
PROBIS = os.environ.get("PROBIS_BIN", "probis")
GLOSA_DIR = os.environ.get("GLOSA_DIR", "")
GLOSA = os.environ.get("GLOSA_BIN") or os.path.join(GLOSA_DIR, "glosa")
GLOSA_CF_CP = os.path.join(GLOSA_DIR, "cf")
JAVA = os.environ.get("JAVA_BIN", "java")
JAVA_OPTS = ["-Duser.language=en", "-Duser.country=US"]
POCKET_RE = re.compile(r"ligand_pocket_(\d+)_([A-Za-z0-9]+)\.pdb$")
ST_OK, ST_NAN, ST_MISSING, ST_TIMEOUT, ST_FAIL, ST_CF = 1, 2, 3, 4, 5, 6   # G-LoSA comparison status codes


def pocket_protein(pocket):
    """protein.pdb of a pocket's BsitePDB entry."""
    return os.path.join(os.path.dirname(pocket), "protein.pdb")


def run(cmd, cwd, timeout):
    """Run cmd in its own process group; kill the whole group on timeout. (rc, stdout, stderr, seconds, timed_out)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, time.time() - t0, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = p.communicate()
        return -9, out, err, time.time() - t0, True


def file_chains(path):
    """Chain ids in ATOM records, in order of first appearance."""
    seen = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and len(line) > 21:
                c = line[21]
                if c not in seen:
                    seen.append(c)
    return "".join(seen)


def pocket_chains(path):
    """Chains for ProBiS -c1/-c2: the ligand_pocket_<n>_<CHAINS>.pdb suffix (BsitePDB naming), checked against the
    ATOM records; falls back to the chains present in the file."""
    fc = file_chains(path)
    m = POCKET_RE.search(os.path.basename(path))
    if m and set(m.group(2)) <= set(fc):
        return m.group(2), "name"
    return fc, "file"


def stage_name(path, tag):
    """<pdbid>_<tag>_<pocketstem>.pdb : ProBiS trims protein names to 4 letters (= pdb id)."""
    pdbid = os.path.basename(os.path.dirname(path))
    stem = os.path.basename(path)[:-4] if path.endswith(".pdb") else os.path.basename(path)
    # ProBiS treats every -f1/-f2 name containing "srf" or "SRF" as a surface (.srf) file (args.cc:719-723), e.g.
    # PDB 1srf; such names would fail ("fail_compare"), so mask them.
    return f"{pdbid}_{tag}_{stem}.pdb".replace("srf", "s_f").replace("SRF", "S_F")


def pocket_residues(pocket):
    """Pocket residues in order of first appearance: list of (chain, resi:int, icode, resname), unique per (chain,
    residue-number field, insertion code)."""
    seen, out = set(), []
    with open(pocket) as f:
        for ln in f:
            if ln.startswith("ATOM") and len(ln) > 26:
                key = (ln[21], ln[22:26], ln[26])
                if key not in seen:
                    seen.add(key)
                    out.append((ln[21], int(ln[22:26]), ln[26], ln[17:20]))
    return out


def probis_motif(pocket, protein):
    """ProBiS motif of a pocket. ProBiS never reads residues with a non-blank insertion code (molecule.cc ~480), so they
    are left out. The motif parser splits on '-' (ranges), so a negative number -n would select residue n (motif.cc
    ~140): a pocket chain with negative numbers is shifted as a whole in the staged copy of protein.pdb by
    K = 1 - min(residue number in the chain) (ProBiS only uses number differences) and the motif uses the shifted
    numbers. Returns (motif, shifts {chain: K}, n_icode_dropped)."""
    res = pocket_residues(pocket)
    blank = [(c, n) for c, n, ic, _ in res if ic == " "]
    n_icode = sum(1 for r in res if r[2] != " ")
    neg_chains = {c for c, n in blank if n < 0}
    shifts = {}
    if neg_chains:
        mins = {}
        with open(protein) as f:
            for ln in f:
                if ln.startswith(("ATOM", "HETATM")) and len(ln) > 26 and ln[21] in neg_chains:
                    v = int(ln[22:26])
                    mins[ln[21]] = min(mins.get(ln[21], v), v)
        shifts = {c: 1 - mins[c] for c in neg_chains}
    by = {}
    for c, n in blank:
        lst = by.setdefault(c, [])
        v = n + shifts.get(c, 0)
        if v not in lst:
            lst.append(v)
    motif = "[" + " or ".join(f":{c} and ({','.join(str(x) for x in sorted(v))})" for c, v in by.items()) + "]"
    return motif, shifts, n_icode


def normalize_line(ln):
    """BsitePDB writes the coordinates one column to the left of the PDB standard (x in columns 30-37 instead of
    31-38). ProBiS (molecule.cc:516-518), glosa.cpp (818-820) and AssignChemicalFeatures (java:95-97) read the standard
    columns, so an 8-character coordinate (<= -100 or >= 1000 A) would be read wrongly. One space is inserted at index
    29 on lines detected as shifted (decimal points at 33/41/49); values are unchanged. Returns (line, changed)."""
    if ln.startswith(("ATOM", "HETATM")) and len(ln) > 50 and ln[33] == "." and ln[41] == "." and ln[49] == ".":
        return ln[:29] + " " + ln[29:], True
    return ln, False


def stage_protein(src, dst, shifts):
    """Copy of protein.pdb in standard columns; with shifts {chain: K}, K is added to the residue number of every
    ATOM/HETATM/ANISOU/TER record of that chain (all other bytes unchanged)."""
    with open(src) as f, open(dst, "w") as g:
        for ln in f:
            ln = normalize_line(ln)[0]
            if ln.startswith(("ATOM", "HETATM", "ANISOU", "TER")) and len(ln) > 26 and ln[21] in shifts \
                    and ln[22:26].strip().lstrip("-").isdigit():
                v = int(ln[22:26]) + shifts[ln[21]]
                if v > 9999:
                    raise ValueError(f"shifted residue number {v} > 9999")
                ln = ln[:22] + f"{v:4d}" + ln[26:]
            g.write(ln)


def strip_inner_ter(src, dst):
    """Copy of a pocket file for G-LoSA: ATOM lines only, in standard columns, without the TER lines between chains
    (G-LoSA and AssignChemicalFeatures stop reading at the first line starting with 'T' or 'E'), residues renumbered
    1..N in file order, unique per (chain, residue number, insertion code), insertion code cleared
    (AssignChemicalFeatures groups atoms into residues only by a change of the residue-number field, java ~78)."""
    with open(src) as f:
        lines = [normalize_line(ln)[0] for ln in f if ln.startswith("ATOM")]
    key2new, out = {}, []
    for ln in lines:
        key = (ln[21], ln[22:26], ln[26])
        if key not in key2new:
            key2new[key] = len(key2new) + 1
        out.append(ln[:22] + f"{key2new[key]:4d}" + " " + ln[27:])
    with open(dst, "w") as g:
        g.writelines(out)
        g.write("TER\nEND\n")


def scratch_base():
    """Node-local scratch: $LENSEEK_TMP, else $TMPDIR, else /tmp. AssignChemicalFeatures splits paths on the regex
    '.pdb', so the path must not contain 'pdb'."""
    return os.environ.get("LENSEEK_TMP") or os.environ.get("TMPDIR") or "/tmp"


def cf_of(p):
    """Chemical-feature file (<stem>-cf.pdb) of a staged pocket, None if it is missing or empty."""
    if p is None:
        return None
    c = p[:-4] + "-cf.pdb"
    return c if os.path.exists(c) and os.path.getsize(c) > 0 else None


def glosa_compare(q, t, w, timeout):
    """One G-LoSA comparison of two staged pockets with their feature files: (status, ga, sec)."""
    if q is None or t is None:
        return ST_MISSING, np.nan, 0.0
    qc, tc = cf_of(q), cf_of(t)
    if qc is None or tc is None:
        return ST_CF, np.nan, 0.0
    t0 = time.perf_counter()
    rc, out, err, _, to = run([GLOSA, "-s1", q, "-s1cf", qc, "-s2", t, "-s2cf", tc, "-o", "0"], w, timeout)
    sec = time.perf_counter() - t0
    if to:
        return ST_TIMEOUT, np.nan, sec
    for line in out.splitlines():
        if "GA-score:" in line:
            try:
                ga = float(line.split("GA-score:")[1].strip())
            except ValueError:
                return ST_FAIL, np.nan, sec
            return (ST_NAN, np.nan, sec) if ga != ga else (ST_OK, ga, sec)
    return ST_FAIL, np.nan, sec


# ---------------------------------------------------------------- CSR Chamfer kernels of the scaling runs
# Kept outside the package: scripts/scaling.py and scripts/single_vector.py measure and scale these implementations
# (size-class buckets, fp16 storage, precomputed norms); they give the distances of lenseek.search up to float rounding.
SIZE_CLASSES = np.array([12, 16, 20, 24, 28, 32, 40, 48, 56, 65])


def set_threads(n):
    """Thread count of torch and, when installed, FAISS."""
    torch.set_num_threads(n)
    try:
        import faiss
        faiss.omp_set_num_threads(n)
    except ImportError:
        pass


def bucketize(X, o, base=0, dtype=torch.float32):
    """Sites of a CSR run grouped into size classes, each wrap-padded to its class size P (wrap duplicates change no
    minimum; the means use the mask): list of (site ids + base, X (S, P, 256), mask (S, P), n (S,))."""
    n = np.diff(o)
    cls = np.searchsorted(SIZE_CLASSES, n)
    out = []
    for ci in np.unique(cls):
        ids = np.flatnonzero(cls == ci)
        P = int(SIZE_CLASSES[ci])
        j = np.arange(P)
        idx = o[ids, None] + (j[None, :] % n[ids, None])
        Xb = torch.from_numpy(np.ascontiguousarray(X[idx.ravel()]).reshape(len(ids), P, -1)).to(dtype)
        m = torch.from_numpy((j[None, :] < n[ids, None]).astype(np.float32))
        out.append((ids + base, Xb, m, torch.from_numpy(n[ids].astype(np.float32))))
    return out


def pad_queries(qres, qoff, sel):
    """Queries sel of a CSR run, wrap-padded to the longest: X (B, mq, 256), mask (B, mq), n (B,)."""
    n = np.diff(qoff)[sel]
    mq = int(n.max())
    j = np.arange(mq)
    idx = qoff[sel][:, None] + (j[None, :] % n[:, None])
    X = torch.from_numpy(np.asarray(qres[idx.ravel()]).reshape(len(sel), mq, -1))
    return X, torch.from_numpy((j[None, :] < n[:, None]).astype(np.float32)), torch.from_numpy(n.astype(np.float32))


@torch.no_grad()
def chamfer_dense(qX, qm, qn, X, m, n, max_cols=32768):
    """B padded queries vs S padded sites of one size class -> (B, S) torch; X may be float16 (upcast per step)."""
    B, mq, dim = qX.shape
    S, P, _ = X.shape
    qf = qX.reshape(B * mq, dim)
    q2 = (qf * qf).sum(1)
    out = torch.empty(B, S)
    step = max(1, max_cols // P)
    for s in range(0, S, step):
        x = X[s:s + step]
        if x.dtype != torch.float32:
            x = x.float()
        sb = x.shape[0]
        xf = x.reshape(sb * P, dim)
        D = torch.addmm((xf * xf).sum(1).unsqueeze(0), qf, xf.T, beta=1.0, alpha=-2.0)
        D.add_(q2.unsqueeze(1)).clamp_(min=0.0)
        D = D.view(B, mq, sb, P)
        a2b = (D.amin(3) * qm[:, :, None]).sum(1) / qn[:, None]
        b2a = (D.amin(1) * m[None, s:s + step]).sum(2) / n[None, s:s + step]
        out[:, s:s + sb] = a2b + b2a
    return out


def chamfer_scan(qres, qoff, buckets, out, qbatch=64):
    """Exhaustive scan: exact Chamfer of all queries vs the sites of `buckets` (from bucketize) into out (Q, N)
    (columns = site ids)."""
    order = np.argsort(np.diff(qoff), kind="stable")          # similar sizes in a batch -> little query padding
    qb = [order[b:b + qbatch] for b in range(0, len(qoff) - 1, qbatch)]
    qpad = [pad_queries(qres, qoff, sel) for sel in qb]
    for ids, Xb, m, n in buckets:
        for sel, (qX, qm, qn) in zip(qb, qpad):
            out[np.ix_(sel, ids)] = chamfer_dense(qX, qm, qn, Xb, m, n).numpy()
    return out


def chamfer_block(q, qoff, d, doff, dn=None):
    """Exact Chamfer between Q stacked query sites and S stacked database sites (torch float32, offsets numpy):
    mean_i min_j |q_i - d_j|^2 + mean_j min_i |q_i - d_j|^2 -> (Q, S) numpy float32."""
    qn = (q * q).sum(1)
    if dn is None:
        dn = (d * d).sum(1)
    D = torch.addmm(dn.unsqueeze(0), q, d.T, beta=1.0, alpha=-2.0)
    D.add_(qn.unsqueeze(1)).clamp_(min=0.0)
    D = D.numpy()
    qs, ds = qoff[:-1], doff[:-1]
    nq, nd = np.diff(qoff).astype(np.float32), np.diff(doff).astype(np.float32)
    rowmin = np.minimum.reduceat(D, ds, axis=1)
    a2b = np.add.reduceat(rowmin, qs, axis=0) / nq[:, None]
    colmin = np.minimum.reduceat(D, qs, axis=0)
    b2a = np.add.reduceat(colmin, ds, axis=1) / nd[None, :]
    return (a2b + b2a).astype(np.float32)


def rerank(q, cand, X, loc, dn=None, k=None):
    """Exact Chamfer re-rank of the candidate sites of one query: q (n, 256) numpy, X the candidates' residue vectors
    stacked in cand order (torch float32, candidate j = rows loc[j]:loc[j+1]), dn their squared norms (computed when
    None) -> the first k candidates by distance, ties by site id, and their distances."""
    cd = chamfer_block(torch.from_numpy(np.ascontiguousarray(q)), np.array([0, len(q)]), X, loc, dn=dn)[0]
    o = np.lexsort((cand, cd))[:k]
    return cand[o], cd[o]


def candidates(D, I, ntot):
    """Approximate a2b score of the hit sites: per query residue i, d_i(s) = best hit distance to site s, or tau_i (its
    r-th hit distance) when s was not hit; score(s) = sum_i (d_i(s) - tau_i) over the residues that hit s (same order
    as mean_i d_i(s)). I holds site ids. Returns the hit sites ordered by score, ties by site id."""
    tau = D[:, -1:]
    valid = I >= 0
    rows_ = np.broadcast_to(np.arange(I.shape[0])[:, None], I.shape)[valid]
    sites, dd = I[valid], (D - tau)[valid]
    key = rows_.astype(np.int64) * ntot + sites
    o_ = np.lexsort((dd, key))
    key, dd, sites = key[o_], dd[o_], sites[o_]
    first = np.r_[True, key[1:] != key[:-1]]
    us, inv = np.unique(sites[first], return_inverse=True)
    sc = np.bincount(inv, weights=dd[first], minlength=len(us))
    return us[np.lexsort((us, sc))]
