"""Generate the T-REX example figures used in the README.

Run from the repository root:

    python examples/trex_examples.py
"""
from pathlib import Path

import metal2d


PPY = "[c-]1ccccc1-c1ccccn1"

TREX_EXAMPLES = {
    "trex_cisplatin": ("cisplatin",
        "Pt{+2} | L=[ SMILES:[Cl-], SMILES:[Cl-], SMILES:N, SMILES:N ] "
        "| MAP:{ (1:1, 3:1), (2:1, 4:1) } | G:sqpl"
    ),
    "trex_transplatin": ("transplatin",
        "Pt{+2} | L=[ SMILES:[Cl-], SMILES:[Cl-], SMILES:N, SMILES:N ] "
        "| MAP:{ (1:1, 2:1), (3:1, 4:1) } | G:sqpl"
    ),
    "trex_fac_ir_ppy3": ("fac-Ir(ppy)3",
        f"Ir{{+3}} | L=[ SMILES:{PPY}, SMILES:{PPY}, SMILES:{PPY} ] "
        "| MAP:{ (1:1, 2:12), (2:1, 3:12), (3:1, 1:12) } | G:O"
    ),
    "trex_mer_ir_ppy3": ("mer-Ir(ppy)3",
        f"Ir{{+3}} | L=[ SMILES:{PPY}, SMILES:{PPY}, SMILES:{PPY} ] "
        "| MAP:{ (1:1, 2:1), (1:12, 3:12), (3:1, 2:12) } | G:O"
    ),
}


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "images"
    output_dir.mkdir(exist_ok=True)
    for name, (title, value) in TREX_EXAMPLES.items():
        path = output_dir / f"{name}.svg"
        metal2d.draw_trex(value, path, size=(600, 600), title=title)
        print(path)


if __name__ == "__main__":
    main()
