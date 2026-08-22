"""Public T-REX API and command-line smoke tests."""
import metal2d
from metal2d.cli import main


CISPLATIN = (
    "Pt{+2} | L=[ SMILES:[Cl-], SMILES:[Cl-], SMILES:N, SMILES:N ] "
    "| MAP:{ (1:1, 3:1), (2:1, 4:1) } | G:sqpl"
)


def test_public_trex_api_builds_and_depicts():
    parsed = metal2d.parse_trex(CISPLATIN)
    built = metal2d.mol_from_trex(parsed)
    depicted = metal2d.depict_trex(parsed)
    unified = metal2d.depict_input(CISPLATIN)

    assert built.mol.GetNumAtoms() == 5
    assert depicted.GetNumConformers() == 1
    assert unified.GetNumConformers() == 1
    assert metal2d.classify_topology(parsed).geometry == "sqpl"


def test_cli_draw_accepts_trex_string(tmp_path):
    outdir = tmp_path / "direct"
    assert main(["draw", CISPLATIN, "--outdir", str(outdir), "--quiet"]) == 0
    assert list(outdir.glob("*.svg"))


def test_cli_draw_accepts_trex_file(tmp_path):
    source = tmp_path / "complexes.trex"
    source.write_text("# cisplatin\n" + CISPLATIN + "\n", encoding="utf-8")
    outdir = tmp_path / "file"
    assert main(["draw", str(source), "--outdir", str(outdir), "--quiet"]) == 0
    assert len(list(outdir.glob("*.svg"))) == 1
