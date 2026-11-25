"""scripts/folddisco.py: the script loads (--help); on synthetic inputs, the residue order of the rewritten files, the
residues Folddisco reads from them, the matched residues of an entries output row, and assemble (distance matrix,
per-query table and top-100 list) for the prefilter and the entries run."""
import gzip
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

S = ROOT / "scripts"
sys.path.insert(0, str(S))
import folddisco as FD  # noqa: E402


def atom(serial, name, num, chain, rec="ATOM  ", res="ALA"):
    xyz = f"{1.0:8.3f}{2.0:8.3f}{-3.0:8.3f}"
    return f"{rec}{serial:5d} {name:<4s} {res} {chain}{num:4d}    {xyz}  1.00  0.00           C"


def test_script_imports():
    subprocess.run([sys.executable, str(S / "folddisco.py"), "--help"], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, PYTHONPATH=str(ROOT)))


def test_residue_order_of_the_rewritten_files():
    """Residues sorted by (chain, number, insertion code), atoms of a residue in file order, only the first MODEL."""
    lines = ["MODEL        1", atom(1, " N", 7, "B"), atom(2, " CA", 7, "B"), atom(3, " N", 3, "A"), "TER",
             atom(4, " CA", 3, "A"), atom(5, " O", 7, "B"), atom(6, " C1", 100, "A", rec="HETATM", res="LIG"),
             "ENDMDL", "MODEL        2", atom(7, " N", 1, "A")]
    rec, keys = FD.residue_sorted(lines)
    assert keys == [("A", 3, " "), ("A", 100, " "), ("B", 7, " ")]
    assert [int(r[6:11]) for r in rec] == [3, 4, 6, 1, 2, 5]


def test_residues_folddisco_reads():
    """ATOM records only; a residue needs N and CA; residues of two chains with the same number are one residue (the
    first chain's); the last atom of the file is never part of a kept residue; negative numbers are not read."""
    lines = [atom(1, " N", -2, "A"), atom(2, " CA", -2, "A"),
             atom(3, " N", 3, "A"), atom(4, " CA", 3, "A"), atom(5, " C", 3, "A"),
             atom(6, " CA", 4, "A"),                                               # no N
             atom(7, " N", 9, "A"), atom(8, " CA", 9, "B"),                         # chains A and B, number 9
             atom(9, " C1", 50, "A", rec="HETATM", res="LIG"),
             atom(10, " CA", 12, "B"), atom(11, " N", 12, "B")]                     # N is the last atom
    assert FD.folddisco_residues(lines) == [("A", 3), ("A", 9)]
    assert FD.res_string([("A", 3), ("B", 12)]) == "A_3,B_12"


def test_matched_residues_of_an_entries_row():
    assert FD.parse_res("A57,_,B102") == {"A_57", "B_102"}
    assert FD.parse_res("AA_250, B7") == {"AA_250", "B_7"}
    assert FD.parse_res("") == set()


def write_out(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


@pytest.fixture
def lists(tmp_path):
    q = [str(tmp_path / "q" / p) for p in ("1abc/ligand_pocket_1_A.pdb", "2xyz/ligand_pocket_1_B.pdb")]
    db = [str(tmp_path / "db" / p) for p in ("3aaa/ligand_pocket_1_A.pdb", "3aaa/ligand_pocket_2_A.pdb",
                                             "4bbb/ligand_pocket_1_C.pdb")]
    (tmp_path / "q.txt").write_text("\n".join(q) + "\n")
    (tmp_path / "db.txt").write_text("\n".join(db) + "\n")
    work = tmp_path / "out" / "work"
    work.mkdir(parents=True)
    pd.DataFrame(dict(key=[f"{i:06d}" for i in range(3)], path=db, ok=1, residues="",
                      residues_all=["A_10,A_11", "A_11,A_12,A_13", "C_5"], error="")).to_csv(
        work / "manifest_db_pockets.tsv", sep="\t", index=False)
    return tmp_path


def assemble(tmp, run):
    subprocess.run([sys.executable, str(S / "folddisco.py"), "assemble", "--queries", str(tmp / "q.txt"), "--db",
                    str(tmp / "db.txt"), "--out", str(tmp / "out"), "--run", run, "--workers", "1"], check=True,
                   stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    od = tmp / "out" / run
    return np.load(od / "dist.npy"), pd.read_csv(od / "per_query.tsv", sep="\t"), \
        pd.read_csv(od / "top100_pairs.tsv", sep="\t", float_precision="round_trip")


def test_assemble_prefilter(lists):
    """d = -(idf + 1e-7 node_count) for the returned sites, 1000 for the others and for a query without output; the
    top-100 list orders by d, then database index."""
    write_out(str(lists / "out" / "runs" / "prefilter" / "q000.tsv.gz"), ["tid", "idf", "node_count"],
              [("db_pockets/000002.pdb", 12.5, 3), ("db_pockets/000000.pdb", 12.5, 4)])
    D, pq, top = assemble(lists, "prefilter")
    assert D.dtype == np.float64 and D.shape == (2, 3)
    assert D[0].tolist() == [-(12.5 + 1e-7 * 4), 1000.0, -(12.5 + 1e-7 * 3)] and (D[1] == 1000.0).all()
    assert pq.status.tolist() == ["ok", "no output"] and pq.n_returned.tolist() == [2, 0]
    assert top.db_idx.tolist() == [0, 2, 1, 0, 1, 2] and top["rank"].tolist() == [1, 2, 3, 1, 2, 3]
    assert top.dist.tolist() == D[0, [0, 2, 1]].tolist() + [1000.0] * 3


def test_assemble_entries(lists):
    """Each match goes to the site of its entry with most of its matched residues; a site keeps its first match; a
    match that shares no residue with a site is not used."""
    write_out(str(lists / "out" / "runs" / "entries" / "q000.tsv.gz"),
              ["tid", "match_score", "rmsd", "matching_residues"],
              [("db_entries/3aaa.pdb", 9.0, 0.5, "A10,A12,A13"), ("db_entries/3aaa.pdb", 8.0, 0.2, "A10,A11"),
               ("db_entries/3aaa.pdb", 7.0, 0.1, "A12"), ("db_entries/4bbb.pdb", 6.0, 1500.0, "C6")])
    D, pq, _ = assemble(lists, "entries")
    assert D[0].tolist() == [-8.0 + 1e-7 * 0.2, -9.0 + 1e-7 * 0.5, 1000.0] and (D[1] == 1000.0).all()
    assert pq.n_rows.tolist() == [4, 0] and pq.n_returned.tolist() == [2, 0]
