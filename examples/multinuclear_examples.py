"""Generate the multinuclear comparison figures used in the README.

Each SVG contains RDKit Compute2DCoords, RDKit CoordGen and metal2d panels.
Run from the repository root:

    python examples/multinuclear_examples.py
"""
from pathlib import Path

import metal2d


EXAMPLES = (
    (
        "multinuclear_example1",
        "Bridged diiridium complex",
        "C[c]12->[Ir+3]3456(<-[Cl-])(<-[n]7ccc[n]8->[Ir+3]9%10%11%12"
        "(<-[Cl-])(<-[n]%13ccc[n]->3c%13-c78)<-[c]3(C)[c]->9(C)[c]->%10(C)"
        "[c-]->%11(C)[c]->%123C)<-[c]1(C)[c]->4(C)[c-]->5(C)[c]->62C",
    ),
    (
        "multinuclear_example2",
        "Diiron carbonyl-phosphazane complex",
        "[Fe+]%10%11%12%13%14%40%41%42<-[C-]#[O+]."
        "[Fe+]%20%21%22%23%24%40%50%51%60."
        "[cH-]%15%10[cH]%11[cH]%12[cH]%13[cH]%14%15."
        "[cH-]%25%20[cH]%21[cH]%22[cH]%23[cH]%24%25."
        "C%41%50=[N+](C)C.O=C%42%51.P->%6012CN3CN(CN(C3)C1)C2",
    ),
    (
        "multinuclear_example3",
        "Sulfur-bridged diruthenium complex",
        "C1(N2CCCCC2)[S-]->[Ru+3]234(<-[S]5->[Ru+3]6"
        "(<-[S-]C=5N5CCCCC5)(<-[S]->2=C(N2CCCCC2)[S-]->3)"
        "(<-[S-]C(N2CCCCC2)=[S]->6)<-[S]=C(N2CCCCC2)[S-]->4)<-[S]=1",
    ),
)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "images"
    output_dir.mkdir(exist_ok=True)
    for name, title, smiles in EXAMPLES:
        path = output_dir / f"{name}.svg"
        metal2d.compare(smiles, path, title=title)
        print(path)


if __name__ == "__main__":
    main()
