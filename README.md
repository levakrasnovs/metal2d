# metal2d

Readable 2D depictions of coordination and organometallic complexes for RDKit.

RDKit's coordinate generators are built for organic molecules. They have no
notion of a coordination sphere, no idea that six dative bonds to one atom
should fan out, and no idea that five bonds to the same ring mean one
η-interaction. Complexes come out as a knot around the metal.

`metal2d` does not replace the depiction engine. It cuts the ligands off, lets
CoordGen draw each of them on its own — which it does well, they are ordinary
organic fragments — and then arranges them around the metal itself.

---



## Examples

Each figure shows the same molecule from the two RDKit generators and from
`metal2d`, with the readability metrics printed above each panel.

**Tris(dichloro-dipyridophenazine)ruthenium(II)** — three large fused ligands on
one centre.

![tris-dppz ruthenium](https://raw.githubusercontent.com/levakrasnovs/metal2d/main/images/example1.svg)

```
Clc1cc2nc3c4ccc[n]5->[Ru+2]67(<-[n]8cccc(c3nc2cc1Cl)c8c45)(<-[n]1cccc2c3nc4cc(Cl)c(Cl)cc4nc3c3ccc[n]->6c3c21)<-[n]1cccc2c3nc4cc(Cl)c(Cl)cc4nc3c3ccc[n]->7c3c21
```

**Ruthenium bis(dppm) methylimidazole-thiolate** — two bidentate phosphines with
eight phenyl rings between them.

![ruthenium bis-dppm](https://raw.githubusercontent.com/levakrasnovs/metal2d/main/images/example2.svg)

```
Cn1cc[n]2->[Ru+2]34(<-[S-]c12)(<-[P](C[P]->3(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1)<-[P](C[P]->4(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1
```

**Cp\*-iridium(III) iminopyridine chloride** — an η⁵ ring, drawn as a single bond
to the ring centre rather than five bonds to five carbons.

![Cp\* iridium](https://raw.githubusercontent.com/levakrasnovs/metal2d/main/images/example3.svg)

```
CC(C)c1cccc(C(C)C)c1/[N]1=C/c2ccc3cc(F)ccc3[n]2->[Ir+3]<-12345(<-[Cl-])<-[c]1(C)[c]->2(C)[c]->3(C)[c-]->4(-c2ccc(-c3ccccc3)cc2)[c]->51C
```

---



## Install

```bash
pip install metal2d
```

Only RDKit and numpy are required. `pip install metal2d[png]` adds cairosvg for
PNG output, `metal2d[progress]` adds a nicer progress bar.

## Use

```python
from rdkit import Chem
import metal2d

mol = Chem.MolFromSmiles("[Cl-]->[Pt+2]12<-[S-]C(=N[N]->1=Cc1cccc[n]->21)Nc1ccccc1")

coords = metal2d.depict(mol)          # a copy carrying a 2D conformer
metal2d.draw(coords, "complex.svg")   # or .png
```

`depict()` only produces coordinates, so the result can go to any renderer, into
`MolsToGridImage`, or out to a molfile. `draw()` is the convenience wrapper.

From the command line, on a SMILES string, a `.smi`/`.csv` list or an SDF:

```bash
metal2d draw "CC(C)(C)c1cc[n]2->[Ru+2]34..."
metal2d draw complexes.smi --png --outdir figures
metal2d draw library.sdf --index 0 5 12
```

Input format does not matter: incoming coordinates are discarded and
regenerated. What is required is that the metal–donor bonds are present in the
connection table, as dative bonds (`->`, `<-`) or plain ones. Complexes written
as separated ions (`[Ru+2].c1ccncc1...`) carry no coordination information and
fall through to plain CoordGen.

---



## Benchmarks

Measured on every tenth structure of **[tmQM](https://github.com/uiocompcat/tmQM/blob/master/tmQM/tmQM_y.csv)** — 10,083 mononuclear transition-metal complexes drawn from the Cambridge Structural Database. A public dataset, and one the algorithm was never tuned on:


|                       | bond crossings | atom overlaps | donors facing away | clean drawings |
| --------------------- | -------------- | ------------- | ------------------ | -------------- |
| RDKit Compute2DCoords | 4.43           | 2.57          | 54.5%              | 28.8%          |
| RDKit CoordGen        | 3.93           | 0.51          | 55.2%              | 49.1%          |
| **metal2d**           | **0.81**       | **0.06**      | **27.5%**          | **69.8%**      |


```bash
metal2d metrics tmqm.csv --step 10
```

A 300-structure subset ships with the package, so the numbers can be checked
without downloading anything:

```bash
metal2d metrics "$(python -c 'import metal2d,os;print(os.path.join(os.path.dirname(metal2d.__file__),"data","sample.smi"))')"
```

It is stratified across 23 metals and deliberately weighted towards the awkward
cases — 14% of it carries η-bonded groups, coordination numbers run from 1 to 13
and denticities from 1 to 8 — so it scores harsher than tmQM:


| on the bundled sample | bond crossings | atom overlaps | clean drawings |
| --------------------- | -------------- | ------------- | -------------- |
| RDKit CoordGen        | 5.60           | 0.63          | 44.0%          |
| **metal2d**           | **0.65**       | **0.02**      | **75.3%**      |


`examples/make_sample.py` regenerates it from a database CSV, deterministically.

## What it actually does

1. Cut every bond from the metal, giving one fragment per ligand.
2. Depict each fragment on its own with CoordGen.
3. Collapse η-bonded groups — Cp, Cp\*, arenes, allyl — into a single
   pseudo-donor at the ring centroid.
4. Fold each chelate into a conformation that can actually chelate. A free
  2,2'-bipyridine is drawn *s-trans*, with its nitrogens pointing apart, which
   no metal position can satisfy. In two dimensions, rotating about a bond is a
   reflection, so the fix is to search reflections for the one that puts the
   donors on a small circle.
5. Place the metal. For three or more donors it goes at the centre of the circle
   through them, the only point equidistant from all of them. For one or two,
   on an arc of slots.
6. Share the 360° around the metal in proportion to how wide each ligand really
   is, then relax rotations to clear the remaining collisions.



## Known limitations

- **Over-long metal–donor bonds.** When several bulky ligands compete for room,
they get pushed outward and the bonds to the metal are drawn visibly longer
than an ordinary bond — noticeably more so than with CoordGen. This is the main
remaining defect.
- **Bulky monodentate donors.** A triarylphosphine puts three rings on one atom
1.5 bond lengths from the centre; it will subtend more than 120° no matter how
the ligands are shared out. Bidentate phosphines are fine.
- **Large wrapping ligands.** Peptide conjugates and macrocyclic chelators that
envelop the metal are handled better by plain CoordGen, which has dedicated
macrocycle support. This is where most of the remaining losses sit.
- **Polynuclear complexes: partly.** Two metals joined by a flexible linker, or
bridged by a ligand that chelates each of them separately, are laid out one
centre at a time and come out fine. Metals bonded to each other are handled by a
dedicated cluster layout: the pair is placed as a rigid core, bridging donors go
on the perpendicular bisector between them, and the terminal ligands of the two
halves are mirrored rather than shared out independently. Not handled: cores of
three or more metals, and pairs held close by a short rigid bridge with no
metal–metal bond, where the second centre can land almost on top of the first.
- Cage ligands such as PTA or adamantane cannot be drawn flat without
self-crossings at all. `metal2d metrics` reports how many crossings lie inside
a single ligand, so this can be told apart from a bad arrangement.



## A note on measuring depictions

Counting bond crossings naively penalises correct organometallic drawing. The
bond from a metal to the centre of an η-bonded ring **must** cross that ring's
perimeter to get there, so every η group adds one unavoidable crossing, which
on a haptic-rich set is enough to swing the clean-drawing rate substantially.
`metal2d metrics` recognises the centroid bond and forgives that single crossing —
and only that one: a haptic bond crossing anything else still counts.