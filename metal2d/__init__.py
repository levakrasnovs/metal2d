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
                   depict, depict_input, mol_from_input, draw, find_metal,
                   read_molecules, style_options)
from .core import _drawing_mol as prepare_for_drawing
from .trex import (Trex, TrexMol, classify_topology, depict_trex, draw_trex,
                   is_trex, mol_from_trex, parse_trex)
from .metrics import score, evaluate
from .compare import compare

__version__ = "0.3.1"

__all__ = ["depict", "depict_input", "mol_from_input", "draw",
           "prepare_for_drawing", "style_options",
           "find_metal", "read_molecules", "score", "evaluate", "compare",
           "Trex", "TrexMol", "is_trex", "parse_trex", "mol_from_trex",
           "classify_topology", "depict_trex", "draw_trex",
           "METALS", "METAL_COLOR", "ML", "LB", "HAPTO_R", "MAX_REACH",
           "__version__"]
