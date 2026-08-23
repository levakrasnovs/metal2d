"""Regression tests for externally reported GitHub issues."""
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest
from rdkit import Chem

import metal2d
from metal2d.cli import main


def test_import_does_not_suppress_rdkit_logging_or_change_depictor_preference():
    script = """
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
RDLogger.EnableLog('rdApp.*')
rdDepictor.SetPreferCoordGen(True)
import metal2d
print(rdDepictor.GetPreferCoordGen())
Chem.MolFromSmiles('invalid_smiles_xyz')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True,
        env=env, check=True)
    assert result.stdout.strip() == "True"
    assert "SMILES Parse Error" in result.stderr


def test_compare_out_uses_unique_paths_for_multiple_molecules(tmp_path):
    source = tmp_path / "two.smi"
    source.write_text(
        "[Cl-]->[Pt+2]<-N mol1\n[Br-]->[Pd+2]<-P mol2\n",
        encoding="utf-8")
    output = tmp_path / "comparison.svg"
    assert main(["compare", str(source), "-o", str(output),
                 "--engines", "metal2d", "--size", "160", "--quiet"]) == 0
    assert not output.exists()
    assert (tmp_path / "comparison_000_mol1.svg").exists()
    assert (tmp_path / "comparison_001_mol2.svg").exists()


def test_prepare_haptic_without_conformer_has_actionable_error():
    smiles = (
        "CCCCn1cc[n+]2-c3ccccc3[O-]->[Ir+3]3456(<-[Cl-])(<-[c-]12)"
        "<-[c]1(C)[c]->3(C)[c]->4(C)[c-]->5(-c2ccc(-c3ccccc3)cc2)"
        "[c]->61C")
    mol = Chem.MolFromSmiles(smiles)
    assert mol.GetNumConformers() == 0
    with pytest.raises(ValueError, match=r"call metal2d\.depict\(\) first"):
        metal2d.prepare_for_drawing(mol)


def test_cluster_failure_is_logged_and_strict_mode_reraises(monkeypatch, caplog):
    from metal2d import cluster

    def fail(_mol):
        raise RuntimeError("sentinel cluster bug")

    monkeypatch.setattr(cluster, "has_cluster", fail)
    mol = Chem.MolFromSmiles("[Cl-]->[Pt+2]<-N")
    with caplog.at_level(logging.WARNING, logger="metal2d.core"):
        out = metal2d.depict(mol)
    assert out.GetNumConformers() == 1
    assert "cluster layout failed" in caplog.text
    assert "sentinel cluster bug" in caplog.text
    with pytest.raises(RuntimeError, match="sentinel cluster bug"):
        metal2d.depict(mol, strict=True)


def test_csv_column_works_for_draw_compare_and_auto_detection(tmp_path):
    source = tmp_path / "complexes.csv"
    source.write_text(
        "name,ligand_smiles,complex_smiles\n"
        "test1,c1ccccc1,[Cl-]->[Pt+2]<-N\n", encoding="utf-8")

    auto = tmp_path / "auto"
    named = tmp_path / "named"
    compared = tmp_path / "compared"
    assert main(["draw", str(source), "--outdir", str(auto), "--quiet"]) == 0
    assert main(["draw", str(source), "--column", "complex_smiles",
                 "--outdir", str(named), "--quiet"]) == 0
    assert main(["compare", str(source), "--column", "complex_smiles",
                 "--outdir", str(compared), "--engines", "metal2d",
                 "--size", "160", "--quiet"]) == 0
    assert len(list(auto.glob("*.svg"))) == 1
    assert len(list(named.glob("*.svg"))) == 1
    assert len(list(compared.glob("*.svg"))) == 1
