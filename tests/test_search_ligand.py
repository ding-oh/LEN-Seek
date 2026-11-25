"""scripts/search_ligand.py: the pocket extracted around a ligand equals the shipped pocket of the same evaluation site."""
import importlib.util
import os

import pytest

from conftest import EXAMPLES, ROOT

pytest.importorskip("gemmi")
pytest.importorskip("rdkit")
pytest.importorskip("scipy")


def load_script():
    spec = importlib.util.spec_from_file_location("search_ligand", ROOT / "scripts" / "search_ligand.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def residues(path):
    return sorted({(line[21], line[22:27].strip()) for line in open(path) if line.startswith("ATOM")})


@pytest.mark.parametrize("entry,suffix", [("7uad", "A"), ("8aga", "A"), ("8h43", "AB"), ("8ho6", "A")])
def test_extracted_pocket_matches_shipped(entry, suffix, tmp_path):
    q = EXAMPLES / "queries" / entry
    site_dir, n_res = load_script().extract(str(q / "protein.pdb"), str(q / f"ligand_1_{suffix}.pdb"), entry, str(tmp_path))
    extracted = os.path.join(site_dir, f"ligand_pocket_1_{suffix}.pdb")
    assert residues(extracted) == residues(q / f"ligand_pocket_1_{suffix}.pdb")
    assert n_res == len(residues(extracted))


def test_ligand_selector_must_be_unique(tmp_path):
    mod = load_script()
    atoms = [dict(rec="HETATM", resn="ATP", chain=c, resi=501, icode="", elem="C", x=0.0, y=0.0, z=0.0) for c in "AB"]
    with pytest.raises(SystemExit):
        mod.resolve_ligand("ATP", atoms)


def test_missing_files_stop_with_a_message(tmp_path, capsys):
    mod = load_script()
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path / "missing.cif"), "--ligand", "ATP"])
    assert "structure file not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        mod.main([str(EXAMPLES / "queries" / "7uad" / "protein.pdb"), "--ligand", str(tmp_path / "missing.pdb")])
    assert "file not found" in capsys.readouterr().err
