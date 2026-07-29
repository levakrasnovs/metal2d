"""
compare.py - side-by-side pictures of the same complex from several depiction
engines, each panel labelled with its readability metrics.

    python compare.py "[Cl-]->[Ru+2]1..."              -> comparison.svg
    python compare.py "SMILES" -o dppb.svg --png       -> also writes dppb.png
    python compare.py data.smi --index 0 3 7           -> one file per structure
    python compare.py data.csv --index 12 --engines coordgen,metal2d

As a library:

    from compare import compare
    compare("[Cl-]->[Ru+2]1...", "out.svg")
    compare(mol, "out.svg", engines=("coordgen", "metal2d"))

Output is SVG, which needs nothing beyond RDKit and numpy. PNG conversion uses
cairosvg or Pillow if either is installed.
"""
import os
import re
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

from . import core as metal2d
from .metrics import score, read_molecules

PANEL = 640          # px per panel
HEADER = 62          # px reserved for the labels above each panel
GAP = 10

LABELS = {"compute2dcoords": "RDKit Compute2DCoords",
          "coordgen": "RDKit CoordGen",
          "metal2d": "metal2d"}


# --------------------------------------------------------------------------- #
#  one panel
# --------------------------------------------------------------------------- #
def _depiction(mol, engine):
    """Molecule with 2D coordinates, plus the molecule as it will be drawn."""
    if engine == "compute2dcoords":
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(m)
        return m, m
    if engine == "coordgen":
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        rdCoordGen.AddCoords(m)
        return m, m
    if engine == "metal2d":
        g = metal2d.depict(mol)
        return g, metal2d._drawing_mol(g)
    raise ValueError("unknown engine %r" % engine)


def _panel_svg(mol, engine, size):
    """Raw SVG for one engine. RDKit engines are drawn with stock settings, so
    the picture is what a user actually gets out of the box."""
    geom, drawn = _depiction(mol, engine)
    if engine == "metal2d":
        d = rdMolDraw2D.MolDraw2DSVG(size, size)
        if hasattr(metal2d, "style_options"):
            metal2d.style_options(d.drawOptions())
        else:                       # older metal2d.py without the helper
            o = d.drawOptions()
            o.addStereoAnnotation = False
            o.bondLineWidth = 2
            o.scaleBondWidth = False
            o.additionalAtomLabelPadding = 0.04
            o.updateAtomPalette({6: (0.1, 0.1, 0.1), 7: (0.15, 0.15, 0.9),
                                 8: (0.9, 0.1, 0.1), 16: (0.8, 0.7, 0.0),
                                 17: (0.1, 0.8, 0.1), 0: (0.1, 0.1, 0.1)})
            for z in metal2d.METALS:
                o.updateAtomPalette({z: metal2d.METAL_COLOR})
        try:
            prepared = rdMolDraw2D.PrepareMolForDrawing(drawn, kekulize=False,
                                                        wedgeBonds=False)
        except Exception:
            prepared = drawn
        d.DrawMolecule(prepared)
    else:
        d = rdMolDraw2D.MolDraw2DSVG(size, size)
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(d, drawn)
        except Exception:
            d.DrawMolecule(drawn)
    d.FinishDrawing()
    return d.GetDrawingText(), drawn


def _inner(svg):
    """Strip the wrapper so a panel can be nested inside a bigger SVG."""
    body = svg[svg.index(">", svg.index("<svg")) + 1:]
    return body[:body.rindex("</svg>")]


def _split_crossings(mol):
    """(inside one ligand, between ligands). A cage such as PTA cannot be drawn
    flat without self-crossings, and that is the depiction engine's doing, not
    the arrangement's - worth separating before blaming the layout."""
    import numpy as np
    mi = metal2d.find_metal(mol)
    if mi is None:
        return 0, 0
    c = mol.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                  for i in range(mol.GetNumAtoms())])
    em = Chem.RWMol(mol)
    for b in list(em.GetAtomWithIdx(mi).GetBonds()):
        em.RemoveBond(mi, b.GetOtherAtomIdx(mi))
    frag_of = {}
    for k, f in enumerate(Chem.GetMolFrags(em.GetMol())):
        for a in f:
            frag_of[a] = k
    segs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    def side(p, q, r):
        return np.sign((q - p)[0] * (r - p)[1] - (q - p)[1] * (r - p)[0])

    same = diff = 0
    from .metrics import _haptic_pairs
    forgive = _haptic_pairs(mol)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if set(segs[i]) & set(segs[j]) or (i, j) in forgive:
                continue
            a, b = P[segs[i][0]], P[segs[i][1]]
            c_, d_ = P[segs[j][0]], P[segs[j][1]]
            if side(a, b, c_) * side(a, b, d_) < 0 and \
               side(c_, d_, a) * side(c_, d_, b) < 0:
                if frag_of.get(segs[i][0]) == frag_of.get(segs[j][0]):
                    same += 1
                else:
                    diff += 1
    return same, diff


# --------------------------------------------------------------------------- #
#  the figure
# --------------------------------------------------------------------------- #
def compare(mol, path="comparison.svg",
            engines=("compute2dcoords", "coordgen", "metal2d"),
            size=PANEL, title=None, png=False):
    """Write a side-by-side comparison figure. `mol` may be a Mol or a SMILES."""
    if isinstance(mol, str):
        smi, mol = mol, Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError("could not parse SMILES: %s" % smi)

    panels, results = [], []
    for eng in engines:
        svg, drawn = _panel_svg(mol, eng, size)
        s = score(drawn)
        panels.append(_inner(svg))
        results.append((eng, s, _split_crossings(drawn)))

    best = min(s["crossings"] for _, s, _ in results)
    W = size * len(engines) + GAP * (len(engines) - 1)
    H = size + HEADER + (26 if title else 0)
    top = 26 if title else 0

    out = ["<svg xmlns='http://www.w3.org/2000/svg' "
           "xmlns:xlink='http://www.w3.org/1999/xlink' "
           "width='%d' height='%d' viewBox='0 0 %d %d'>" % (W, H, W, H),
           "<rect width='%d' height='%d' fill='white'/>" % (W, H)]
    if title:
        out.append("<text x='12' y='19' font-family='sans-serif' font-size='16' "
                   "fill='#333'>%s</text>" % _escape(title))

    for i, (eng, s, (same, diff)) in enumerate(results):
        x = i * (size + GAP)
        good = s["crossings"] == best and s["overlaps"] == 0
        colour = "#0a7a30" if good else "#b00000"
        note = "%d bond crossings, %d atom overlaps" % (s["crossings"], s["overlaps"])
        if same and s["crossings"]:
            note += "  (%d inside one ligand)" % same
        out.append("<text x='%d' y='%d' font-family='sans-serif' font-size='20' "
                   "font-weight='bold' fill='#111'>%s</text>"
                   % (x + 14, top + 26, _escape(LABELS.get(eng, eng))))
        out.append("<text x='%d' y='%d' font-family='sans-serif' font-size='14' "
                   "fill='%s'>%s</text>" % (x + 14, top + 48, colour, _escape(note)))
        out.append("<g transform='translate(%d,%d)'>%s</g>"
                   % (x, top + HEADER, panels[i]))
        if i:
            out.append("<line x1='%d' y1='0' x2='%d' y2='%d' stroke='#dddddd' "
                       "stroke-width='2'/>" % (x - GAP // 2, x - GAP // 2, H))
    out.append("</svg>")

    with open(path, "w") as fh:
        fh.write("\n".join(out))

    if png:
        _to_png(path)
    return path, results


def _escape(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _to_png(svg_path, scale=2):
    png_path = os.path.splitext(svg_path)[0] + ".png"
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=scale,
                         background_color="white")
        return png_path
    except ImportError:
        pass
    print("  (install cairosvg for PNG output; the SVG is complete either way)",
          file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
