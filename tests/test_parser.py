"""PDB reading and the site filter.

PDB reading: coordinates <= -100 A in both column layouts, insertion codes, alternate locations, loud failures.

--filters bsitepdb-v1 keeps sites whose pocket file also lists nucleotides or other non-standard residues.

908 of the 114,661 database sites have such ATOM records (DNA, RNA, SEC, UNK); the released data keeps them and
featurizes their standard residues only.
"""
import numpy as np
import pytest

from lenseek.cli.featurize import bsitepdb_v1_filter
from lenseek.io import pdb as pdbio


# ---------------------------------------------------------------- parser
def std_line(serial, name, resn, chain, num, icode, x, y, z, alt=" ", rec="ATOM  "):
    """PDB-standard ATOM record (x, y, z in columns 31-54)."""
    return (f"{rec}{serial:5d} {name:<4s}{alt}{resn:>3s} {chain}{num:4d}{icode:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {name[0]}")


def bsitepdb_line(serial, name, resn, chain, num, x, y, z):
    """BsitePDB layout: no insertion-code column, so x, y, z start one column early (30-53)."""
    return (f"ATOM  {serial:5d} {name:<4s} {resn:>3s} {chain}{num:4d}   "[:29] +
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {name[0]}")


def write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_layouts_have_expected_columns():
    s = std_line(1, "N", "LEU", "A", 213, "", -100.490, 17.523, 118.070)
    b = bsitepdb_line(1, "N", "LEU", "A", 213, -100.490, 17.523, 118.070)
    assert s[30:38] == "-100.490" and b[29:37] == "-100.490"


@pytest.mark.parametrize("xyz", [(-100.490, 17.523, 118.070), (52.262, 58.290, -101.112), (-999.999, -100.0, -100.001),
                                 (1.5, -2.25, 3.0)])
def test_atom_xyz_both_layouts(xyz):
    s = std_line(1, "CA", "GLY", "A", 5, "", *xyz)
    b = bsitepdb_line(1, "CA", "GLY", "A", 5, *xyz)
    assert pdbio.atom_xyz(s) == pytest.approx(xyz)
    assert pdbio.atom_xyz(b) == pytest.approx(xyz)


def test_legacy_reader_reads_fixed_columns(tmp_path):
    """BsitePDB layout: reading fixed columns, the legacy reader takes x <= -100 without its sign and skips an atom
    whose y or z is <= -100; the strict reader reads all three coordinates."""
    lines = [bsitepdb_line(1, "N", "LEU", "A", 213, -100.490, 17.523, 118.070),
             bsitepdb_line(2, "CA", "LEU", "A", 213, 52.262, 58.290, -101.112),
             bsitepdb_line(3, "C", "LEU", "A", 213, 10.0, 11.0, 12.0)]
    p = write(tmp_path, "shifted.pdb", lines)
    legacy = pdbio.parse_pdb_legacy(p)["A"][213]["atoms"]
    assert legacy["N"][0] == pytest.approx(100.490)          # columns 31-38 without the sign
    assert "CA" not in legacy                                 # unreadable in fixed columns: skipped
    strict = pdbio.read_structure(p).residues[pdbio.ResidueId("A", 213, "")].atoms
    assert strict["N"] == pytest.approx([-100.490, 17.523, 118.070])
    assert strict["CA"] == pytest.approx([52.262, 58.290, -101.112])


def test_standard_layout_same_in_both_modes(tmp_path):
    lines = [std_line(i + 1, a, "SER", "B", 7, "", -150.0 + i, 2.0 * i, -3.5) for i, a in enumerate(("N", "CA", "C"))]
    p = write(tmp_path, "std.pdb", lines)
    legacy = pdbio.parse_pdb_legacy(p)["B"][7]["atoms"]
    strict = pdbio.read_structure(p).residues[pdbio.ResidueId("B", 7, "")].atoms
    for a in ("N", "CA", "C"):
        assert np.array_equal(legacy[a], strict[a])


def test_insertion_codes_are_kept_apart(tmp_path):
    """Thrombin-style numbering: Tyr 60A and Trp 60D are different residues."""
    lines = []
    for k, (resn, ic) in enumerate((("TYR", "A"), ("TRP", "D"))):
        for j, a in enumerate(("N", "CA", "C")):
            lines.append(std_line(3 * k + j + 1, a, resn, "H", 60, ic, float(k), float(j), 0.0))
    p = write(tmp_path, "icode.pdb", lines)
    st = pdbio.read_structure(p)
    assert st.residues[pdbio.ResidueId("H", 60, "A")].name == "TYR"
    assert st.residues[pdbio.ResidueId("H", 60, "D")].name == "TRP"
    keys = [r.key for r in pdbio.pocket_residues(p)]
    assert keys == ["TRPH0060D", "TYRH0060A"]
    seq = pdbio.chain_sequences(st)["H"]
    assert [r.icode for r in seq] == ["A", "D"]
    # the legacy reader merges both into residue 60 (the later atoms overwrite the earlier ones)
    legacy = pdbio.parse_pdb_legacy(p)
    assert list(legacy["H"]) == [60]


def test_lost_insertion_codes_fail_loudly(tmp_path):
    """A file without the insertion-code column (BsitePDB layout) puts both residues under one key."""
    lines = [bsitepdb_line(1, "N", "TYR", "H", 60, 1.0, 1.0, 1.0), bsitepdb_line(2, "N", "TRP", "H", 60, 2.0, 2.0, 2.0)]
    p = write(tmp_path, "collision.pdb", lines)
    with pytest.raises(pdbio.PDBParseError, match="insertion"):
        pdbio.read_structure(p)


def test_duplicate_atom_without_altloc_fails_with_the_reason(tmp_path):
    """Tools that drop the altloc column leave two copies of an atom; the strict reader names the cause."""
    lines = [std_line(1, "OG", "SER", "A", 77, "", 0.0, 0.0, 0.0), std_line(2, "OG", "SER", "A", 77, "", 1.0, 1.0, 1.0)]
    p = write(tmp_path, "dup.pdb", lines)
    with pytest.raises(pdbio.PDBParseError, match="alternate-location"):
        pdbio.read_structure(p)


def test_unreadable_record_fails_loudly(tmp_path):
    bad = std_line(1, "CA", "GLY", "A", 5, "", 1.0, 2.0, 3.0)
    bad = bad[:30] + "  abc.de" + bad[38:]
    p = write(tmp_path, "bad.pdb", [bad])
    with pytest.raises(pdbio.PDBParseError):
        pdbio.read_structure(p)
    assert pdbio.parse_pdb_legacy(p) == {"A": {5: {"atoms": {}}}}   # legacy reader: silently empty


def test_altloc_keeps_A(tmp_path):
    lines = [std_line(1, "N", "SER", "A", 3, "", 0.0, 0.0, 0.0),
             std_line(2, "CA", "SER", "A", 3, "", 1.0, 0.0, 0.0),
             std_line(3, "OG", "SER", "A", 3, "", 9.0, 9.0, 9.0, alt="B"),
             std_line(4, "OG", "SER", "A", 3, "", 2.0, 2.0, 2.0, alt="A")]
    p = write(tmp_path, "alt.pdb", lines)
    res = pdbio.read_structure(p).residues[pdbio.ResidueId("A", 3, "")]
    assert res.atoms["OG"] == pytest.approx([2.0, 2.0, 2.0])


def test_negative_residue_numbers(tmp_path):
    lines = [std_line(1, "CA", "ALA", "A", -5, "", 0.0, 0.0, 0.0), std_line(2, "CA", "GLY", "A", 3, "", 1.0, 0.0, 0.0)]
    p = write(tmp_path, "neg.pdb", lines)
    keys = [r.key for r in pdbio.pocket_residues(p)]
    assert keys == ["ALAA-005", "GLYA0003"]
    seq = pdbio.chain_sequences(pdbio.read_structure(p))["A"]
    assert [r.number for r in seq] == [-5, 3]


def test_only_first_model(tmp_path):
    lines = ["MODEL        1", std_line(1, "CA", "ALA", "A", 1, "", 0.0, 0.0, 0.0), "ENDMDL",
             "MODEL        2", std_line(1, "CA", "ALA", "A", 1, "", 5.0, 5.0, 5.0), "ENDMDL"]
    p = write(tmp_path, "models.pdb", lines)
    assert pdbio.read_structure(p).residues[pdbio.ResidueId("A", 1, "")].atoms["CA"] == pytest.approx([0, 0, 0])


# ---------------------------------------------------------------- featurize filter
def _atom(serial, name, resname, chain, resnum, x):
    return (f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00\n")


def test_nonstandard_pocket_residue_kept(tmp_path):
    pocket = tmp_path / "ligand_pocket_1_A.pdb"
    pocket.write_text(_atom(1, "CA", "ALA", "A", 10, 0.0) + _atom(2, "P", "DG", "B", 3, 5.0)
                      + _atom(3, "CA", "SEC", "A", 11, 3.8))
    (tmp_path / "ligand_1_A.pdb").write_text("HETATM    1  C1  ATP A 401       1.000   1.000   1.000  1.00  0.00\n")
    assert bsitepdb_v1_filter(str(pocket)) == ""


def test_invalid_ligand_still_rejected(tmp_path):
    pocket = tmp_path / "ligand_pocket_1_A.pdb"
    pocket.write_text(_atom(1, "CA", "ALA", "A", 10, 0.0))
    (tmp_path / "ligand_1_A.pdb").write_text("HETATM    1  C1  1PE A 401       1.000   1.000   1.000  1.00  0.00\n")
    assert bsitepdb_v1_filter(str(pocket)) == "no valid ligand"
