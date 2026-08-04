"""metal2d - readable 2D depictions of coordination and organometallic complexes.

    from rdkit import Chem
    import metal2d

    mol = Chem.MolFromSmiles("[Cl-]->[Pt+2]12<-[S-]C(=N[N]->1=Cc1cccc[n]->21)N")
    coords = metal2d.depict(mol)          # a copy carrying a 2D conformer
    metal2d.draw(coords, "complex.svg")   # or .png

`depict` only produces coordinates, so the result works with any renderer.
`draw` is the convenience wrapper. `prepare_for_drawing` returns the molecule as
it is actually drawn - dative bonds turned into plain lines and eta-bonded
groups collapsed to a single bond at the ring centre - which a custom renderer
needs in order not to draw six lines to one arene.
"""
from .core import (METALS, METAL_COLOR, ML, LB, HAPTO_R, MAX_REACH,
                   depict, draw, find_metal, read_molecules, style_options)
from .core import _drawing_mol as prepare_for_drawing
from .metrics import score, evaluate
from .compare import compare

__version__ = "0.2.0"

__all__ = ["depict", "draw", "prepare_for_drawing", "style_options",
           "find_metal", "read_molecules", "score", "evaluate", "compare",
           "METALS", "METAL_COLOR", "ML", "LB", "HAPTO_R", "MAX_REACH",
           "__version__"]
