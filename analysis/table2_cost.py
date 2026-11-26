#!/usr/bin/env python
"""Main-text Table 2 and Supplementary Table S17 (computational cost of ProBiS, G-LoSA, DeeplyTough, Folddisco and
LEN-Seek; Supplementary Section S9) from the timing runs.

Reads the files written by scripts/timing.py and writes one row per (tool, quantity):
the preprocessing time (Folddisco: the index build), the comparisons per query, the mean and median wall time per
query and per comparison, the LEN-Seek times with and without the Ankh embedding of the query on CPU and GPU, the
speed-ups over ProBiS and G-LoSA and the break-even, the number of queries above which LEN-Seek (preprocessing +
queries) takes less time than ProBiS. The ProBiS preprocessing time is that of the extraction run on its own
(probis-prep).
The CPU preprocessing of the whole database is extrapolated from the CPU sample: Ankh time linear in chain length
(chain lengths of the full GPU run), featurization and encoding proportional to the number of sites.

Inputs (scripts/timing.py <mode> writes each file):
  --cpu-dir     16-core run: probis_prep_summary.txt (probis-prep), probis_surfdb.tsv (probis), glosa_full.tsv,
                glosa_summary.txt (glosa), lenseek_cpu_timing.tsv (lenseek), lenseek_prep_sample*_summary.txt,
                lenseek_prep_sample*_chains.tsv (lenseek-prep --sample), folddisco_index.tsv, folddisco_timing.tsv
                (folddisco)
  --gpu-dir     GPU run: lenseek_cuda_timing.tsv (lenseek), lenseek_prep_full_summary.txt, lenseek_prep_full_chains.tsv
                (lenseek-prep)
  --dt-dir      DeeplyTough run of the same queries: timing_q.tsv (deeplytough), dt_net.tsv (dt-net, the network
                alone on 16 threads), search_timing.tsv (dt-search)
  --dt-prep     dt_prep.csv (dt-prep)
Each directory defaults to $LENSEEK_OUT/timing. Output: $LENSEEK_OUT/table2/table2.csv (tool, quantity, value).

usage:
  python analysis/table2_cost.py build --cpu-dir A --gpu-dir B --dt-dir DT
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_ROOT, die  # noqa: E402


def one_file(d, pattern):
    f = sorted(glob.glob(os.path.join(d, pattern)))
    if len(f) != 1:
        die(f"expected one {pattern} in {d}, found {len(f)}")
    return f[0]


def grab(text, pattern, src):
    m = re.search(pattern, text)
    if not m:
        die(f"{src}: no match for {pattern!r}")
    return float(m.group(1))


def build(a):
    rows = []

    def add(tool, quantity, value):
        rows.append(dict(tool=tool, quantity=quantity, value=float(value)))

    rd = lambda f: pd.read_csv(f, sep="\t", comment="#")  # noqa: E731
    summ = " ".join(open(one_file(a.cpu_dir, f)).read() for f in ("probis_prep_summary.txt", "glosa_summary.txt"))
    full = open(one_file(a.gpu_dir, "lenseek_prep_full_summary.txt")).read()
    sample = open(one_file(a.cpu_dir, "lenseek_prep_sample*_summary.txt")).read()
    n_db = grab(full, r"sites ([0-9]+) \(", "lenseek_prep_full_summary.txt")

    p = rd(one_file(a.cpu_dir, "probis_surfdb.tsv"))
    pw = p.surfdb_wall_s + p.results_wall_s
    probis_prep = grab(summ, r"probis_db extract: .*? wall ([0-9.]+)s", a.cpu_dir)
    add("ProBiS", "preprocessing_s", probis_prep)
    add("ProBiS", "comparisons_per_query", p.n_targets.iloc[0])
    add("ProBiS", "query_mean_s", pw.mean())
    add("ProBiS", "query_median_s", pw.median())
    add("ProBiS", "per_comparison_ms", pw.mean() / p.n_targets.iloc[0] * 1e3)

    g = rd(one_file(a.cpu_dir, "glosa_full.tsv"))
    add("G-LoSA", "preprocessing_s", grab(summ, r"glosa prep: .*? staged wall ([0-9.]+)s", a.cpu_dir)
        + grab(summ, r"glosa prep: .*? features \([0-9]+ BatchCF JVMs\) wall ([0-9.]+)s", a.cpu_dir))
    add("G-LoSA", "comparisons_per_query", g.n_db.iloc[0])
    add("G-LoSA", "query_mean_s", g.wall_s.mean())
    add("G-LoSA", "query_median_s", g.wall_s.median())
    add("G-LoSA", "per_comparison_ms", g.wall_s.mean() / g.n_db.iloc[0] * 1e3)

    # DeeplyTough: protein preparation + voxelization + network (1 thread) + search (16 threads) per query
    t = rd(one_file(a.dt_dir, "timing_q.tsv"))
    sf = one_file(a.dt_dir, "search_timing.tsv")
    s = rd(sf).set_index("query_idx").search_s
    n16 = rd(one_file(a.dt_dir, "dt_net.tsv")).threads_16_s.iloc[0]
    q = t.sid.str.split(":").str[0].str[1:].astype(int)
    srch = s.loc[q].to_numpy()
    total = t.t_feat + t.t_vox + t.t_net + srch
    dtp = pd.read_csv(a.dt_prep).set_index("set")
    add("DeeplyTough", "preprocessing_core_h", dtp.loc["database", "total_core_h"])
    add("DeeplyTough", "comparisons_per_query", grab(open(sf).readline(), r"db=([0-9]+)", sf))
    add("DeeplyTough", "query_mean_s", total.mean())
    add("DeeplyTough", "query_median_s", total.median())
    add("DeeplyTough", "protein_preparation_mean_s", t.t_feat.mean())
    add("DeeplyTough", "query_mean_s_network_16_threads", (t.t_feat + t.t_vox + n16 + srch).mean())

    fi = rd(one_file(a.cpu_dir, "folddisco_index.tsv"))
    ft = rd(one_file(a.cpu_dir, "folddisco_timing.tsv"))
    add("Folddisco", "preprocessing_s", fi.wall_s.iloc[0])
    add("Folddisco", "query_mean_s", ft.wall_s.mean())
    add("Folddisco", "query_median_s", ft.wall_s.median())

    lc = rd(one_file(a.cpu_dir, "lenseek_cpu_timing.tsv"))
    lg = rd(one_file(a.gpu_dir, "lenseek_cuda_timing.tsv"))
    for dev, x in [("CPU", lc), ("GPU", lg)]:
        add("LEN-Seek", f"{dev}_query_mean_s", x.A_total_with_ankh_s.mean())
        add("LEN-Seek", f"{dev}_query_median_s", x.A_total_with_ankh_s.median())
        add("LEN-Seek", f"{dev}_query_without_ankh_mean_s", x.B_total_without_ankh_s.mean())
        add("LEN-Seek", f"{dev}_query_without_ankh_median_s", x.B_total_without_ankh_s.median())
        add("LEN-Seek", f"{dev}_per_comparison_us", x.A_total_with_ankh_s.mean() / n_db * 1e6)
    add("LEN-Seek", "comparisons_per_query", n_db)
    for k, pat in [("ankh", r"; ankh ([0-9.]+)s wall"), ("featurize", r"featurize ([0-9.]+)s wall"),
                   ("encode", r"encode \(batch [0-9]+\) ([0-9.]+)s wall"), ("total", r"total loop ([0-9.]+)s wall")]:
        add("LEN-Seek", f"GPU_preprocessing_{k}_s", grab(full, pat, "lenseek_prep_full_summary.txt"))
    # CPU preprocessing of the whole database, extrapolated from the sample
    sc = rd(one_file(a.cpu_dir, "lenseek_prep_sample*_chains.tsv"))
    sc = sc[sc.status == "ok"]
    fc = rd(one_file(a.gpu_dir, "lenseek_prep_full_chains.tsv"))
    fc = fc[fc.status == "ok"]
    ankh = float(np.polyval(np.polyfit(sc.length.to_numpy(float), sc.wall_s.to_numpy(float), 1),
                            fc.length.to_numpy(float)).sum())
    scale = n_db / grab(sample, r"sites ([0-9]+) \(", "lenseek-prep sample summary")
    feat = grab(sample, r"featurize ([0-9.]+)s wall", "lenseek-prep sample summary") * scale
    enc = grab(sample, r"encode \(batch [0-9]+\) ([0-9.]+)s wall", "lenseek-prep sample summary") * scale
    for k, v in [("ankh", ankh), ("featurize", feat), ("encode", enc), ("total", ankh + feat + enc)]:
        add("LEN-Seek", f"CPU_preprocessing_{k}_s", v)
    n_star = (ankh + feat + enc - probis_prep) / (pw.mean() - lc.A_total_with_ankh_s.mean())
    add("LEN-Seek", "break_even_queries_vs_probis", np.floor(n_star) + 1)
    add("LEN-Seek", "speedup_vs_probis", pw.mean() / lc.A_total_with_ankh_s.mean())
    add("LEN-Seek", "speedup_vs_glosa", g.wall_s.mean() / lc.A_total_with_ankh_s.mean())
    od = os.path.join(OUT_ROOT, "table2")
    os.makedirs(od, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(od, "table2.csv"), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    timing = os.path.join(OUT_ROOT, "timing")
    p = sub.add_parser("build")
    p.add_argument("--cpu-dir", default=timing,
                   help="16-core run: ProBiS (extraction alone and per query), G-LoSA, LEN-Seek CPU, preprocessing "
                        "sample, Folddisco")
    p.add_argument("--gpu-dir", default=timing, help="GPU run: LEN-Seek per query and full preprocessing")
    p.add_argument("--dt-dir", default=timing, help="DeeplyTough per-query run")
    p.add_argument("--dt-prep", default=os.path.join(timing, "dt_prep.csv"))
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
