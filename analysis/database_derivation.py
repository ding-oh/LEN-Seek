#!/usr/bin/env python
"""Derivation of the training data and the evaluation data (Supplementary Figure S7, Section S1 and Table S6), from the
per-site checks of scripts/database_scan.py sites.

Training data (Figure S7 A). Every binding site of the entries released up to July 20, 2023 is counted at the first
of these steps that removes it:
  digit          the CCD code of the site's ligand (first HETATM residue of ligand_<n>_<chains>.pdb) starts with a digit
  exclusion      the code is on an exclusion list of the pipeline (lenseek.exclusion: crystallization aids, additives,
                 glycans)
  no ligand      the ligand file has no HETATM record
  one character  the code has one character
  < 10 residues  fewer than 10 standard residues; a residue is RESNAME + chain + %04d of the ATOM records
  featurization  the featurization of the training data fails (readable protein.pdb, sequence map, N, CA and C in the
                 fixed-column reader of the training code; a failure whose N, CA and C atoms the strict reader finds
                 is counted as backbone atoms not read by the fixed-column reader)
The sites in $LENSEEK_EVAL_DIR/db_paths.txt form the search database (= the training set); a site that passes every
step but is not in it would have failed the checks of the Ankh files (residual).
Evaluation data (Figure S7 B). The binding sites of the evaluation period (entries released July 21 to November 6, 2023)
without the entries removed by the MMseqs2 identity filter against the training data; the kept entries are those of
$LENSEEK_QUERY_ROOT. Sites with fewer than 10 standard residues or a failed featurization (same checks) are removed;
no ligand filter is applied. The rest are the sites of the evaluation data ($LENSEEK_EVAL_DIR/q_paths.txt), and
without the entries of data/query_sets/primary_exclude_pdb.txt the primary queries. Queries whose ligand the database
filters would remove are listed with their ligand kind.
Composition of the database (Table S6). A site belongs to the sequence cluster of its main chain (most binding-site
residues): MMseqs2 easy-cluster of the database chains at 30 % and at 80 % identity (scripts/site_annotation.py
db-search), both with -c 0.8 --cov-mode 0; and to its ligand component (CCD code). Chains shorter than 10 residues are
not clustered, so their sites form a group of their own.
Ligand kinds. Each CCD code gets the kind of the first matching rule, from the SMILES of the CCD snapshot and the
chemical-component name and type of the same CCD release (ccd_name_type.tsv of asset reference_data; --ccd-cif derives
the table from a components.cif.gz instead, and a fresh CCD download can differ slightly from that release):
  crystallization aid; cofactor or pigment (COFACTORS below, a pyridoxal-phosphate core, >= 4 five-membered N rings,
  or >= 4 conjugated acyclic C=C); metal or inorganic (a metal atom or no carbon); additive exclusion list (drug-like
  with >= 20 heavy atoms and an aromatic atom); nucleotide or nucleoside; saccharide; lipid or fatty acid (six CH2 in
  a row or a steroid nucleus); amino acid or peptide (CCD type); fragment (<= 10 heavy atoms); polyethylene glycol
  oligomer; phosphorylated metabolite (P, <= 20 heavy atoms, no aromatic atom); drug-like (11-70 heavy atoms);
  other (> 70). A code without parsable SMILES is decided by the lists alone.

Inputs: see `python analysis/run.py -h`; read here: the per-site checks of scripts/database_scan.py sites (--scan-dir:
scan_db.tsv, scan_queries.tsv, eval_period.json; default $LENSEEK_OUT/database_scan) and, optionally, a CCD
components.cif.gz (--ccd-cif).
Outputs ($LENSEEK_OUT/database_derivation/): flow_database.tsv, flow_queries.tsv, composition.tsv, digit_by_kind.tsv,
filtered_queries.tsv, counts.tsv, figS8_derivation.pdf and .png (Supplementary Figure S7); with --ccd-cif also
ccd_name_type.tsv.

    python analysis/database_derivation.py
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ANNOT, OUT_ROOT, PRIMARY_EXCLUDE_FILE, REFERENCE_DIR, SNAP, apply_style, die,  # noqa: E402
                    require_path, savefig, site_key)
from lenseek.exclusion import CRYSTALLIZATION_AIDS_SET, GLYCANS_SET, LIGAND_EXCLUSION_SET  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = os.path.join(OUT_ROOT, "database_derivation")
SCAN = os.path.join(OUT_ROOT, "database_scan")
MIN_RESIDUES = 10
TEXTWIDTH_SUPP, PT_PER_IN = 469.75502, 72.27


# ---------------------------------------------------------------- ligand kinds
METALS = {3, 4, 11, 12, 13, 19, 20} | set(range(21, 32)) | set(range(37, 51)) | set(range(55, 85)) | set(range(87, 104))
PAT = {k: Chem.MolFromSmarts(v) for k, v in dict(
    pyrimidine="[#6]1~[#7]~[#6]~[#7]~[#6]~[#6]~1", ring5o="[#8;R]1~[#6;R]~[#6;R]~[#6;R]~[#6;R]~1",
    sugar6="[#6;R;!a]1~[#8;R;!a]~[#6;R;!a]~[#6;R;!a]~[#6;R;!a]~[#6;R;!a]~1", oh="[OX2H]",
    lipid="[CH2][CH2][CH2][CH2][CH2][CH2]", plp="[CH3]c1ncc(CO[P](~O)(~O)~O)c([#6])c1[OX2]",
    polyene="[C;!R]=[C;!R]-[C;!R]=[C;!R]-[C;!R]=[C;!R]-[C;!R]=[C;!R]", nring5="[#7;R]1~[#6;R]~[#6;R]~[#6;R]~[#6;R]~1",
    peg="[OX2][CH2][CH2][OX2][CH2][CH2][OX2]",
    steroid="[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~1~2"
).items()}
# enzyme cofactors, coenzymes and prosthetic groups (CCD codes, with common redox and truncated forms)
COFACTORS = set("""
FAD FDA FAE FAS 6FA FMN FNR FMA RBF
NAD NAI NAP NDP NAJ NAQ NAE NHD NHO NZQ ENA ADJ NA0 DND NMN NXX
HEM HEC HEA HEB HDD HDM HAS HEO HEV HNI 1FH 2FH DHE VER SRM HCO
COA ACO CAO SCA MCA 1VU HXC CO8 MLC CAA FCX BCO 3HC SOP DCC CMC COS COO MYA PLM_COA
SAM SAH SFG SMM SAI 5GP_NO
PLP PMP PLR PDP P1T PXP PLG LLP_NO
TPP TDP TPW TZD TDM TPU 8EF
BTN BTI B12 CNC COB B1Z 5AD
H4B HBI BH4 THF THG FOL DHF THH MEF FFO 5MF TMF C2F
PQQ F42 FO1 MGD PGD MTE MTV MCN LPA LPB LIP
GSH GTT GTB GDS GSN
UQ1 UQ2 UQ5 UQ6 UQ7 UQ8 UQ9 U10 PL9 MQ7 MQ8 MQ9 MQE
RET RTL CRT
CLA CHL BCL BCB PHO BPH BPB CL1 CL2 F43
ASC TOQ TQQ TRQ TPQ
IHP
""".split())


class LigandKinds:
    """kind(code) -> (kind, CCD name) by the rules of the module docstring."""

    def __init__(self, cif=None):
        s = pd.read_csv(require_path(SNAP, "ccd_smiles.tsv", "LENSEEK_REFERENCE_DIR"), sep="\t", dtype=str,
                        keep_default_na=False)
        self.smiles = dict(zip(s.comp_id, s.smiles))
        self.meta = self._cif_meta(cif) if cif else self._snapshot_meta()

    @staticmethod
    def _snapshot_meta():
        """comp_id -> (name, type) of the CCD release of the snapshot (names with a line break are quoted fields)."""
        f = require_path(os.path.join(REFERENCE_DIR, "ccd_name_type.tsv"), "ccd_name_type.tsv", "LENSEEK_REFERENCE_DIR")
        m = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        return {r.comp_id: (r["name"], r["type"]) for _, r in m.iterrows()}

    @staticmethod
    def _cif_meta(cif):
        """comp_id -> (name, type) read from a components.cif.gz, also written to ccd_name_type.tsv."""
        import gemmi
        if not os.path.exists(cif):
            die(f"--ccd-cif {cif} does not exist")
        out = {}
        for b in gemmi.cif.read(cif):
            cid = b.find_value("_chem_comp.id")
            if cid is not None:
                g = [b.find_value(t) for t in ("_chem_comp.name", "_chem_comp.type")]
                out[gemmi.cif.as_string(cid)] = tuple(gemmi.cif.as_string(v) if v is not None else "" for v in g)
        os.makedirs(OUT, exist_ok=True)
        pd.DataFrame([(k, *v) for k, v in out.items()], columns=["comp_id", "name", "type"]).to_csv(
            os.path.join(OUT, "ccd_name_type.tsv"), sep="\t", index=False)
        return out

    def kind(self, code):
        name, typ = self.meta.get(code, ("", ""))
        s = self.smiles.get(code, "")
        m = Chem.MolFromSmiles(s) if s else None
        ha = m.GetNumHeavyAtoms() if m is not None else None
        if code in CRYSTALLIZATION_AIDS_SET:
            return "crystallization aid", name
        if code in COFACTORS or (m is not None and (
                m.HasSubstructMatch(PAT["plp"]) or m.HasSubstructMatch(PAT["polyene"]) or
                len({tuple(sorted(x)) for x in m.GetSubstructMatches(PAT["nring5"])}) >= 4)):
            return "cofactor / pigment", name
        if m is None:
            if code in LIGAND_EXCLUSION_SET:
                return "additive / other (Ligand_exclusion)", name
            if code in GLYCANS_SET or "saccharide" in typ.lower():
                return "saccharide / glycan", name
            return "no SMILES", name
        z = {x.GetAtomicNum() for x in m.GetAtoms()}
        arom = any(x.GetIsAromatic() for x in m.GetAtoms())
        if (z & METALS) or 6 not in z:
            return "metal / inorganic", name
        if code in LIGAND_EXCLUSION_SET:
            return ("drug-like" if ha >= 20 and arom else "additive / other (Ligand_exclusion)"), name
        has_p = 15 in z
        pyr = m.HasSubstructMatch(PAT["pyrimidine"])
        if (has_p and pyr) or (pyr and m.HasSubstructMatch(PAT["ring5o"])):
            return "nucleotide / nucleoside", name
        if code in GLYCANS_SET or "saccharide" in typ.lower() or (
                (m.HasSubstructMatch(PAT["sugar6"]) or m.HasSubstructMatch(PAT["ring5o"])) and
                len(m.GetSubstructMatches(PAT["oh"])) >= 3 and not arom and not has_p):
            return "saccharide / glycan", name
        if m.HasSubstructMatch(PAT["lipid"]) or m.HasSubstructMatch(PAT["steroid"]):
            return "lipid / fatty acid", name
        if "peptide" in typ.lower():
            return "amino acid / peptide", name
        if ha <= 10:
            return "fragment (<=10 heavy atoms)", name
        if m.HasSubstructMatch(PAT["peg"]) and not arom and 7 not in z:
            return "additive / other (PEG)", name
        if has_p and not arom and ha <= 20:
            return "metabolite (phosphorylated)", name
        if ha <= 70:
            return "drug-like", name
        return "other / large", name


# ---------------------------------------------------------------- flow
DB_STEPS = [("digit", "CCD code starts with a digit", lambda S: S.lig_digit == 1),
            ("exclusion", "CCD code on an exclusion list", lambda S: S.exclusion_list != ""),
            ("noligand", "no HETATM in the ligand file", lambda S: S.lig_none == 1),
            ("onechar", "one-character CCD code", lambda S: S.lig_len1 == 1),
            ("lt10", "< 10 standard residues", lambda S: S.n_std_res < MIN_RESIDUES),
            ("feat", "featurization failure", lambda S: S.feat_fail == 1)]
FEAT = ["backbone lost by the PDB reader", "residue not in the sequence map", "backbone atoms absent"]  # feat_detail
FEAT_LABEL = dict(zip(FEAT, ["backbone atoms not read by the fixed-column reader", "residue not in the sequence map",
                             "backbone atoms absent"]))


def read_scan(path):
    if not os.path.exists(path):
        die(f"{path} missing: run scripts/database_scan.py sites first")
    S = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in ["n_std_res", "lig_none", "lig_digit", "lig_len1", "valid_ligand", "feat_fail"]:
        S[c] = S[c].astype(int)
    return S


def flow_database(S, db_keys):
    """Remaining sites and entries after each step, each site counted at the first step that removes it."""
    S["in_db"] = S.key.isin(db_keys)
    first = pd.Series("", index=S.index)
    for k, _, rule in DB_STEPS:
        first[(first == "") & rule(S)] = k
    S["first_reason"] = np.where(S.in_db, "in DB", first)
    rows = [dict(step=0, stage="binding-site files", removed="", remaining=len(S), entries=S.entry.nunique(),
                 detail="")]
    rem = pd.Series(True, index=S.index)
    for i, (k, lab, rule) in enumerate(DB_STEPS, 1):
        hit = rem & rule(S)
        det = ""
        if k == "digit":
            det = f"{S[hit].ligand.nunique()} CCD codes"
        elif k == "exclusion":
            det = "; ".join(f"{n} {int((hit & (S.exclusion_list == n)).sum())}" for n in
                            ["Crystallization_aids", "Ligand_exclusion", "glycans"])
        elif k == "feat":
            det = "; ".join(f"{FEAT_LABEL.get(n, n)} {int((hit & (S.feat_detail == n)).sum())}"
                            for n in FEAT + ["protein.pdb unreadable"])
        rem &= ~rule(S)
        rows.append(dict(step=i, stage=lab, removed=int(hit.sum()), remaining=int(rem.sum()),
                         entries=S[rem].entry.nunique(), detail=det))
    rows.append(dict(step=len(rows), stage="Ankh embedding file missing or of another length (residual)",
                     removed=int((rem & ~S.in_db).sum()), remaining=int((rem & S.in_db).sum()),
                     entries=S[rem & S.in_db].entry.nunique(), detail=""))
    rows.append(dict(step=len(rows), stage="search database = training set", removed="", remaining=int(S.in_db.sum()),
                     entries=S[S.in_db].entry.nunique(),
                     detail=f"sites of db_paths.txt failing a step: {int((S.in_db & ~rem).sum())}"))
    return pd.DataFrame(rows)


def flow_queries(Q, ev_period, q_keys, primary_excl):
    """Panel B: evaluation-period sites -> kept entries -> < 10 residues -> featurization -> all -> primary."""
    lt10 = Q.n_std_res < MIN_RESIDUES
    feat = ~lt10 & (Q.feat_fail == 1)
    ok = Q[~lt10 & ~feat]
    prim = ok[~ok.entry.str.lower().isin(primary_excl)]
    fd = Q[feat].feat_detail.value_counts()
    rows = [dict(step=0, stage="evaluation-period binding-site files", removed="",
                 remaining=ev_period["eval_period_sites"], entries=ev_period["eval_period_entries"], detail=""),
            dict(step=1, stage="entries > 80 % identical to the training data (MMseqs2)",
                 removed=ev_period["eval_period_sites"] - len(Q), remaining=len(Q), entries=Q.entry.nunique(),
                 detail=f'{ev_period["eval_period_entries"] - Q.entry.nunique()} entries'),
            dict(step=2, stage="< 10 standard residues", removed=int(lt10.sum()), remaining=int((~lt10).sum()),
                 entries=Q[~lt10].entry.nunique(), detail=""),
            dict(step=3, stage="featurization failure", removed=int(feat.sum()), remaining=len(ok),
                 entries=ok.entry.nunique(), detail="; ".join(f"{FEAT_LABEL[n]} {int(fd.get(n, 0))}" for n in FEAT)),
            dict(step=4, stage="entries >= 80 % identical to a training protein over >= 80 % of the query",
                 removed=len(ok) - len(prim), remaining=len(prim), entries=prim.entry.nunique(),
                 detail=f"{ok.entry.nunique() - prim.entry.nunique()} entries")]
    if set(ok.key) != set(q_keys):
        print(f"[warn] the {len(ok)} sites passing every step differ from the {len(q_keys)} sites of q_paths.txt")
    return pd.DataFrame(rows), ok.assign(primary=~ok.entry.str.lower().isin(primary_excl))


def composition(annot_dir, c30_file):
    """Sites per 30 % cluster, 80 % cluster and CCD code of the database sites."""
    sites = pd.read_csv(os.path.join(annot_dir, "db_sites.tsv"), sep="\t", dtype=str, keep_default_na=False)
    sites["chain_id"] = sites.key.str.split("/").str[0] + "_" + sites.main_chain

    def clusters(path):
        if not os.path.exists(path):
            die(f"{path} missing (scripts/site_annotation.py db-search)")
        c = pd.read_csv(path, sep="\t", header=None, names=["rep", "mem"], dtype=str)
        return sites.chain_id.map(dict(zip(c.mem, c.rep))).fillna("nochain_" + sites.chain_id)

    groups = [("30 % sequence cluster", clusters(c30_file)),
              ("80 % sequence cluster", clusters(os.path.join(annot_dir, "db80_cluster.tsv"))),
              ("ligand component (CCD code)", sites.ligand)]
    rows = []
    for name, g in groups:
        sz = g.value_counts()
        rows.append(dict(grouping=name, n_sites=len(sites), n_groups=len(sz), median_sites=float(sz.median()),
                         max_sites=int(sz.max()), largest=sz.index[0]))
    return pd.DataFrame(rows)


def draw_flow(fa, fb, nf, nfp, path_noext):
    """Supplementary Figure S7: two columns of boxes, removals to the right of the arrows (counts from the flow tables)."""
    apply_style(base=6.0, lw=0.6)
    W, H = TEXTWIDTH_SUPP / PT_PER_IN, 4.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    FS = 6.0
    MW, SW, GAP, BH, SH, DY, TOP = 1.30, 1.72, 0.12, 0.48, 0.46, 0.68, 3.42

    def box(x, y, w, h, text, fc="white", bold=False, ha="center"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0", fc=fc, ec="black", lw=0.6))
        ax.text(x if ha == "center" else x - w / 2 + 0.05, y, text, ha=ha, va="center", fontsize=FS,
                fontweight="bold" if bold else "normal", linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                                      shrinkA=0, shrinkB=0, mutation_scale=6))

    def column(xl, label, title, main, sides):
        x0 = xl + MW / 2
        ax.text(xl, TOP + BH / 2 + 0.12, label, ha="left", va="bottom", fontsize=FS + 2, fontweight="bold")
        ax.text(xl + 0.17, TOP + BH / 2 + 0.12, title, ha="left", va="bottom", fontsize=FS + 1, fontweight="bold")
        ys = [TOP - i * DY for i in range(len(main))]
        for i, (txt, bold) in enumerate(main):
            box(x0, ys[i], MW, BH, txt, fc="#eeeeee" if bold else "white", bold=bold)
            if i < len(main) - 1:
                arrow(x0, ys[i] - BH / 2, x0, ys[i + 1] + BH / 2)
                ym = (ys[i] + ys[i + 1]) / 2
                xs = xl + MW + GAP + SW / 2
                box(xs, ym, SW, SH, sides[i], ha="left")
                arrow(x0, ym, xs - SW / 2, ym)

    f = lambda v: f"{int(v):,}"
    det = lambda s: dict((p.rsplit(" ", 1)[0], int(p.rsplit(" ", 1)[1])) for p in s.split("; "))
    A, B = fa.set_index("step"), fb.set_index("step")
    ex = det(A.loc[2, "detail"])
    main_a = [(f'BsitePDB, released up\nto July 20, 2023\n{f(A.loc[0, "remaining"])} binding sites\n'
               f'{f(A.loc[0, "entries"])} entries', True),
              (f'{f(A.loc[1, "remaining"])} sites', False), (f'{f(A.loc[3, "remaining"])} sites', False),
              (f'{f(A.loc[5, "remaining"])} sites', False),
              (f'Training data\n{f(A.loc[8, "remaining"])} sites\n{f(A.loc[8, "entries"])} entries', True)]
    sides_a = [f'Ligand code starts with a digit\n−{f(A.loc[1, "removed"])} sites '
               f'({f(A.loc[1, "detail"].split()[0])} codes)',
               f'Ligand code on exclusion lists\n−{f(A.loc[2, "removed"])}: additives '
               f'{f(ex["Ligand_exclusion"])},\nglycans {f(ex["glycans"])}; no ligand record −{f(A.loc[3, "removed"])}',
               f'< 10 standard residues\n−{f(A.loc[5, "removed"])} sites',
               f'Featurization failure\n−{f(A.loc[6, "removed"])} sites']
    column(0.05, "A", "Training data", main_a, sides_a)
    main_b = [(f'BsitePDB, released\nJuly 21 to Nov. 6, 2023\n{f(B.loc[0, "remaining"])} binding sites\n'
               f'{f(B.loc[0, "entries"])} entries', True),
              (f'{f(B.loc[1, "remaining"])} sites\n{f(B.loc[1, "entries"])} entries', False),
              (f'{f(B.loc[2, "remaining"])} sites', False),
              (f'Evaluation data\n{f(B.loc[3, "remaining"])} sites\n{f(B.loc[3, "entries"])} entries', True),
              (f'Queries\n{f(B.loc[4, "remaining"])} sites\n{f(B.loc[4, "entries"])} entries', True)]
    sides_b = [f'Entries >80% identical to the\ntraining data (MMseqs2)\n'
               f'−{f(B.loc[0, "entries"] - B.loc[1, "entries"])} entries (−{f(B.loc[1, "removed"])} sites)',
               f'< 10 standard residues\n−{f(B.loc[2, "removed"])} sites',
               f'Featurization failure\n−{f(B.loc[3, "removed"])} sites',
               f"Entries ≥80% identical to a training\nprotein over ≥80% of their sequence\n"
               f'({B.loc[3, "entries"] - B.loc[4, "entries"]} entries) −{f(B.loc[4, "removed"])} sites']
    column(3.30, "B", "Evaluation data", main_b, sides_b)
    ax.text(0.05, 0.06, f'No ligand filter is applied to the evaluation data: {nf} of the {f(B.loc[3, "remaining"])} '
            f'sites ({nfp} of the {f(B.loc[4, "remaining"])} queries) carry a ligand that the ligand filters of '
            f"panel A remove.", ha="left", va="bottom", fontsize=FS - 0.5, color="#333333")
    savefig(fig, path_noext)


def cmd_flow(a):
    require_path(a.eval_dir, "--eval-dir", "LENSEEK_EVAL_DIR")
    os.makedirs(OUT, exist_ok=True)
    S = read_scan(os.path.join(a.scan_dir, "scan_db.tsv"))
    Q = read_scan(os.path.join(a.scan_dir, "scan_queries.tsv"))
    ev_period = json.load(open(os.path.join(a.scan_dir, "eval_period.json")))
    db_keys = {site_key(p) for p in open(os.path.join(a.eval_dir, "db_paths.txt")).read().split()}
    q_keys = {site_key(p) for p in open(os.path.join(a.eval_dir, "q_paths.txt")).read().split()}
    excl = {l.split()[0].lower() for l in open(PRIMARY_EXCLUDE_FILE) if l.strip() and not l.startswith("#")}
    fa = flow_database(S, db_keys)
    fb, ok = flow_queries(Q, ev_period, q_keys, excl)
    fa.to_csv(os.path.join(OUT, "flow_database.tsv"), sep="\t", index=False)
    fb.to_csv(os.path.join(OUT, "flow_queries.tsv"), sep="\t", index=False)
    comp = composition(a.annot_dir, os.path.join(a.annot_dir, "db30_cluster.tsv"))
    comp.to_csv(os.path.join(OUT, "composition.tsv"), sep="\t", index=False)

    LK = LigandKinds(a.ccd_cif)
    dg = S[S.lig_digit == 1]
    kinds = {c: LK.kind(c) for c in sorted(set(dg.ligand) | set(ok.ligand)) if c}
    dg = dg.assign(kind=dg.ligand.map(lambda c: kinds[c][0]))
    top = dg.groupby(["kind", "ligand"]).size().rename("n").reset_index().sort_values(["n", "ligand"],
                                                                                    ascending=[False, True])
    dk = dg.groupby("kind").agg(sites=("key", "size"), ccd_codes=("ligand", "nunique"))
    dk["share"] = dk.sites / len(dg)
    dk["top_codes"] = [", ".join(f"{r.ligand} ({kinds[r.ligand][1][:40]}) {r.n}" for r in
                                 top[top.kind == k].head(5).itertuples()) for k in dk.index]
    dk = dk.sort_values("sites", ascending=False).reset_index()
    dk.to_csv(os.path.join(OUT, "digit_by_kind.tsv"), sep="\t", index=False, float_format="%.4f")
    fq = ok[ok.valid_ligand == 0].assign(kind=lambda d: d.ligand.map(lambda c: kinds[c][0] if c else "no ligand"))
    fq[["key", "entry", "ligand", "exclusion_list", "lig_digit", "kind", "primary"]].sort_values("key").to_csv(
        os.path.join(OUT, "filtered_queries.tsv"), sep="\t", index=False)

    A, B = fa.set_index("step"), fb.set_index("step")
    det = lambda s: dict((p.rsplit(" ", 1)[0], int(p.rsplit(" ", 1)[1])) for p in s.split("; "))
    ex, fd, fdq = det(A.loc[2, "detail"]), det(A.loc[6, "detail"]), det(B.loc[3, "detail"])
    lig_only = ((S.lig_digit == 1) | (S.exclusion_list != "")) & (S.n_std_res >= MIN_RESIDUES) & (S.feat_fail == 0)
    C = [("training period: binding sites", A.loc[0, "remaining"]), ("training period: entries", A.loc[0, "entries"]),
         ("removed: CCD code starts with a digit", A.loc[1, "removed"]),
         ("  CCD codes", int(A.loc[1, "detail"].split()[0])),
         ("removed: exclusion lists", A.loc[2, "removed"]), ("  Ligand_exclusion", ex["Ligand_exclusion"]),
         ("  glycans", ex["glycans"]), ("  Crystallization_aids", ex["Crystallization_aids"]),
         ("removed: no HETATM record", A.loc[3, "removed"]), ("removed: one-character code", A.loc[4, "removed"]),
         ("removed: < 10 standard residues", A.loc[5, "removed"]),
         ("removed: featurization failure", A.loc[6, "removed"])] + \
        [(f"  {FEAT_LABEL[n]}", fd[FEAT_LABEL[n]]) for n in FEAT] + \
        [("residual (Ankh files)", A.loc[7, "removed"]),
         ("search database: sites", A.loc[8, "remaining"]), ("search database: entries", A.loc[8, "entries"]),
         ("removed by the ligand filters and no other step", int(lig_only.sum()))]
    for r in comp.itertuples():
        C += [(f"{r.grouping}: groups", r.n_groups), (f"{r.grouping}: median sites", r.median_sites),
              (f"{r.grouping}: max sites", r.max_sites), (f"{r.grouping}: largest", r.largest)]
    C += [(f"digit-code sites: {r.kind}", f"{r.sites} ({r.share:.3f})") for r in dk.itertuples()]
    C += [("evaluation period: binding sites", B.loc[0, "remaining"]),
          ("evaluation period: entries", B.loc[0, "entries"]),
          ("removed by identity: entries", B.loc[0, "entries"] - B.loc[1, "entries"]),
          ("removed by identity: sites", B.loc[1, "removed"]), ("kept: sites", B.loc[1, "remaining"]),
          ("kept: entries", B.loc[1, "entries"]), ("queries removed: < 10 standard residues", B.loc[2, "removed"]),
          ("queries removed: featurization failure", B.loc[3, "removed"])] + \
         [(f"  {FEAT_LABEL[n]}", fdq[FEAT_LABEL[n]]) for n in FEAT[:2]] + \
         [("evaluation data: sites", B.loc[3, "remaining"]), ("evaluation data: entries", B.loc[3, "entries"]),
          ("evaluation data equals q_paths.txt", set(ok.key) == q_keys),
          ("identity criterion of the primary queries: sites", B.loc[4, "removed"]),
          ("identity criterion of the primary queries: entries", B.loc[3, "entries"] - B.loc[4, "entries"]),
          ("primary queries: sites", B.loc[4, "remaining"]), ("primary queries: entries", B.loc[4, "entries"]),
          ("queries with a filtered ligand", len(fq)),
          ("primary queries with a filtered ligand", int(fq.primary.sum()))]
    C += [(f"filtered-ligand queries: {k}", n) for k, n in fq.kind.value_counts().items()]
    pd.DataFrame(C, columns=["count", "value"]).to_csv(os.path.join(OUT, "counts.tsv"), sep="\t", index=False)
    print(fa.to_string(index=False))
    print(fb.to_string(index=False))
    print(pd.DataFrame(C, columns=["count", "value"]).to_string(index=False))
    draw_flow(fa, fb, len(fq), int(fq.primary.sum()), os.path.join(OUT, "figS8_derivation"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", default=SCAN,
                    help="per-site checks of scripts/database_scan.py sites (default $LENSEEK_OUT/database_scan)")
    ap.add_argument("--annot-dir", default=ANNOT,
                    help="site annotation of scripts/site_annotation.py (default $LENSEEK_OUT/cache/site_annot)")
    ap.add_argument("--eval-dir", default=os.environ.get("LENSEEK_EVAL_DIR"),
                    help="evaluation outputs with db_paths.txt, q_paths.txt (default $LENSEEK_EVAL_DIR)")
    ap.add_argument("--ccd-cif", help="derive the CCD names and types from this components.cif.gz instead of "
                                      "$LENSEEK_REFERENCE_DIR/ccd_name_type.tsv")
    cmd_flow(ap.parse_args())


if __name__ == "__main__":
    main()
