"""T-REX input support for the local metal2d renderer.

The parser follows the public T-REX-Full grammar for monometallic complexes.
SMILES ligand payloads are converted to one RDKit molecule with donor->metal
dative bonds.  The trans-pair map is kept separately and used to orient the
ligands after metal2d has made their normal, chemically readable 2D layouts.

This module deliberately does not import the external ``trex`` package: the
renderer must keep using the files in this working directory while they are
being developed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


_HEAD = re.compile(r"^([A-Z][a-z]?)\s*\{\s*([+-]?\d+)(?:\s*,\s*S\s*=\s*(\d+))?\s*\}$")


@dataclass(frozen=True, order=True)
class Site:
    """One coordination site: 1-based ligand and ligand-atom indices."""

    ligand: int
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class Ligand:
    kind: str
    text: str


@dataclass
class Trex:
    metal: str
    oxidation_state: int
    spin: int | None = None
    ligands: list[Ligand] = field(default_factory=list)
    pairs: list[tuple[Site, Site]] = field(default_factory=list)
    singles: list[Site] = field(default_factory=list)
    geometry: str | None = None
    chirality: str | None = None

    @property
    def sites(self) -> list[Site]:
        return [s for pair in self.pairs for s in pair] + list(self.singles)

    @property
    def coordination_number(self) -> int:
        return len(self.sites)


@dataclass
class TrexMol:
    mol: Chem.Mol
    trex: Trex
    metal_idx: int
    ligand_atoms: list[list[int]]
    site_atoms: dict[Site, tuple[int, ...]]
    ligand_base_offsets: list[int] = field(default_factory=list)
    ligand_pos1b_to_global: list[list[int]] = field(default_factory=list)
    dative_bond_ids: list[int] = field(default_factory=list)
    # Official trex2mol represents these by ZERO bonds.  Keeping the same
    # information out-of-graph prevents those virtual edges from joining
    # otherwise independent ligand fragments in the 2D layout.
    trans_site_atoms: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list)


@dataclass(frozen=True)
class TopologyClass:
    """Geometry/isomer labels derived from the T-REX trans-pair graph.

    ``fac_mer`` is intentionally optional: IUPAC's fac/mer descriptors only
    apply when the six sites split into two chemically equivalent classes of
    three.  ``ligand_modes`` records local facial/meridional coordination of
    tridentate ligands independently of that global descriptor.
    """

    geometry: str | None
    fac_mer: str | None
    ligand_modes: tuple[tuple[int, str], ...] = ()
    bis_tridentate: str | None = None
    trans_signature: tuple[tuple[str, str], ...] = ()
    reason: str = ""


def is_trex(text: str) -> bool:
    """Cheap, conservative format detection suitable for CLI/API dispatch."""
    if not isinstance(text, str):
        return False
    first = text.split("|", 1)[0].strip()
    return bool(_HEAD.match(first) and "L=" in text and "MAP:" in text)


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split outside (), [] and {}; required for commas inside SMILES."""
    out, buf = [], []
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    matching = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != matching[ch]:
                raise ValueError("unbalanced brackets in T-REX input")
            stack.pop()
        if ch == separator and not stack:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
        else:
            buf.append(ch)
    if quote or stack:
        raise ValueError("unbalanced quote or brackets in T-REX input")
    token = "".join(buf).strip()
    if token:
        out.append(token)
    return out


def _site(text: str) -> Site:
    if ":" not in text:
        raise ValueError("MAP site must be ligand:atom, got %r" % text.strip())
    lig_s, atom_s = (x.strip() for x in text.split(":", 1))
    try:
        lig = int(lig_s)
        if atom_s.startswith("["):
            if not atom_s.endswith("]"):
                raise ValueError
            atoms = tuple(int(x.strip()) for x in
                          _split_top_level(atom_s[1:-1], ","))
        else:
            atoms = (int(atom_s),)
    except (TypeError, ValueError):
        raise ValueError("bad MAP site %r" % text.strip()) from None
    if lig < 1 or not atoms or any(a < 1 for a in atoms):
        raise ValueError("MAP indices are 1-based and must be positive: %r" % text.strip())
    return Site(lig, tuple(sorted(set(atoms))))


def _map(body: str) -> tuple[list[tuple[Site, Site]], list[Site]]:
    pair_chunks, remainder = [], []
    depth = 0
    start = None
    for i, ch in enumerate(body):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced MAP parentheses")
            if depth == 0 and start is not None:
                pair_chunks.append(body[start:i + 1])
                remainder.append(body[:start] if len(pair_chunks) == 1 else "")
                start = None
    if depth:
        raise ValueError("unbalanced MAP parentheses")

    pairs = []
    for chunk in pair_chunks:
        tokens = _split_top_level(chunk[1:-1], ",")
        if len(tokens) != 2:
            raise ValueError("a MAP pair must contain exactly two sites: %s" % chunk)
        pairs.append((_site(tokens[0]), _site(tokens[1])))

    rest = body
    for chunk in pair_chunks:
        rest = rest.replace(chunk, " ", 1)
    # A semicolon is conventional but accepting a comma-only remainder makes
    # the parser tolerant of strings emitted by early T-REX prototypes.
    if ";" in rest:
        rest = rest.split(";", 1)[1]
    rest = rest.strip(" ,;")
    singles = [_site(x) for x in _split_top_level(rest, ",")] if rest else []
    return pairs, singles


def parse_trex(text: str) -> Trex:
    """Parse a monometallic T-REX-Full string without canonicalizing it."""
    blocks = _split_top_level(text.strip(), "|")
    if not blocks:
        raise ValueError("empty T-REX input")
    head = _HEAD.match(blocks[0])
    if not head:
        raise ValueError("bad T-REX header %r" % blocks[0])
    result = Trex(head.group(1), int(head.group(2)),
                  int(head.group(3)) if head.group(3) else None)
    seen = set()
    for block in blocks[1:]:
        if block.startswith("L="):
            if "L" in seen:
                raise ValueError("duplicate ligand block")
            seen.add("L")
            rhs = block[2:].strip()
            if not (rhs.startswith("[") and rhs.endswith("]")):
                raise ValueError("bad ligand block %r" % block)
            for token in _split_top_level(rhs[1:-1], ","):
                if ":" not in token:
                    raise ValueError("ligand payload needs a type tag: %r" % token)
                kind, payload = token.split(":", 1)
                kind = kind.strip().upper()
                payload = payload.strip()
                if kind != "SMILES":
                    raise NotImplementedError(
                        "T-REX payload %s is not supported yet; use SMILES" % kind)
                if not payload:
                    raise ValueError("empty ligand payload")
                result.ligands.append(Ligand(kind, payload))
        elif block.startswith("MAP:"):
            if "MAP" in seen:
                raise ValueError("duplicate MAP block")
            seen.add("MAP")
            rhs = block[4:].strip()
            if not (rhs.startswith("{") and rhs.endswith("}")):
                raise ValueError("bad MAP block %r" % block)
            result.pairs, result.singles = _map(rhs[1:-1].strip())
        elif block.startswith("G:"):
            result.geometry = block[2:].strip() or None
        elif block.startswith("X:"):
            result.chirality = block[2:].strip() or None
        else:
            raise ValueError("unknown T-REX block %r" % block)
    if "L" not in seen or "MAP" not in seen:
        raise ValueError("T-REX input requires both L=[...] and MAP:{...}")
    if not result.sites:
        raise ValueError("MAP contains no coordination sites")
    if len(set(result.sites)) != len(result.sites):
        raise ValueError("a coordination site occurs more than once in MAP")
    return result


def _written_order_to_rdidx(smiles: str) -> list[int]:
    """Official T-REX SMILES-position -> RDKit-index convention.

    T-REX atom positions refer to the written order of the canonical ligand
    payload, not necessarily the current RDKit atom numbering.  Atom maps let
    RDKit expose that permutation without modifying the ligand used below.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    canonical = Chem.MolToSmiles(mapped, canonical=True, isomericSmiles=True)
    ids = [int(x) - 1 for x in re.findall(r":(\d+)\]", canonical)]
    if not ids and mapped.GetNumAtoms() == 1:
        return [0]
    if len(ids) != mapped.GetNumAtoms():
        ranks = Chem.CanonicalRankAtoms(mol)
        return sorted(range(mol.GetNumAtoms()), key=lambda i: (ranks[i], i))
    return ids


def _site_equivalence_classes(t: Trex) -> dict[Site, str]:
    """Return chemical equivalence classes for coordination sites.

    Atom symbols alone are insufficient: two nitrogens in an unsymmetrical
    diimine may be inequivalent, while the two nitrogens of bpy are related by
    a ligand graph automorphism.  RDKit symmetry ranks (``breakTies=False``)
    provide exactly the distinction needed here, combined with canonical
    ligand identity so equivalent sites on separate ligand copies collapse.
    """
    ligand_info = []
    for ligand in t.ligands:
        mol = Chem.MolFromSmiles(ligand.text)
        if mol is None:
            raise ValueError("invalid ligand SMILES: %s" % ligand.text)
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        order = _written_order_to_rdidx(ligand.text)
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False,
                                             includeChirality=True))
        ligand_info.append((identity, order, ranks))

    raw = {}
    for site in t.sites:
        identity, order, ranks = ligand_info[site.ligand - 1]
        site_ranks = tuple(sorted(ranks[order[pos - 1]] for pos in site.atoms))
        raw[site] = (identity, site_ranks, len(site.atoms))
    unique = {key: "S%d" % (i + 1)
              for i, key in enumerate(sorted(set(raw.values()), key=repr))}
    return {site: unique[key] for site, key in raw.items()}


def classify_topology(value: str | Trex | TrexMol) -> TopologyClass:
    """Classify fac/mer and tridentate modes from trans relations.

    The result follows the IUPAC/T-REX topological definition.  No coordinate
    thresholds are used, so distorted octahedra classify consistently.
    """
    if isinstance(value, TrexMol):
        t = value.trex
    elif isinstance(value, Trex):
        t = value
    else:
        t = parse_trex(value)

    geometry_text = (t.geometry or "").strip().lower()
    octahedral = (t.coordination_number == 6 and len(t.pairs) == 3 and
                  not t.singles and geometry_text in
                  ("", "o", "oh", "oct", "octahedral"))
    if not octahedral:
        return TopologyClass(t.geometry, None, reason=
                             "fac/mer requires an octahedral six-site map")

    classes = _site_equivalence_classes(t)
    signature = tuple(sorted(tuple(sorted((classes[a], classes[b])))
                             for a, b in t.pairs))
    counts = {cls: list(classes.values()).count(cls)
              for cls in set(classes.values())}
    fac_mer = None
    reason = "site equivalence does not form an A3B3 partition"
    if sorted(counts.values()) == [3, 3]:
        within = [classes[a] == classes[b] for a, b in t.pairs]
        if not any(within):
            fac_mer = "fac"
            reason = "all three trans pairs are A-B"
        elif sum(within) == 2:
            same_types = [tuple(sorted((classes[a], classes[b])))
                          for a, b in t.pairs if classes[a] == classes[b]]
            if len(set(same_types)) == 2:
                fac_mer = "mer"
                reason = "trans signature is A-A, A-B, B-B"

    # Local coordination mode of every tridentate ligand is independent of
    # the global A3B3 descriptor: zero internal trans pairs is facial, one is
    # meridional.
    ligand_modes = []
    for li in range(1, len(t.ligands) + 1):
        sites = [s for s in t.sites if s.ligand == li]
        if len(sites) != 3:
            continue
        internal = sum(a.ligand == li and b.ligand == li for a, b in t.pairs)
        if internal == 0:
            ligand_modes.append((li, "fac"))
        elif internal == 1:
            ligand_modes.append((li, "mer"))

    bis_mode = None
    if len(t.ligands) == 2 and len(ligand_modes) == 2:
        modes = [mode for _, mode in ligand_modes]
        if modes == ["mer", "mer"]:
            bis_mode = "mer"
        elif modes == ["fac", "fac"]:
            # For ABA ligands the unique B donors distinguish trans-fac from
            # cis-fac.  Otherwise retain the rigorous, less specific fac-fac.
            unique_sites = []
            for li in (1, 2):
                sites = [s for s in t.sites if s.ligand == li]
                local_counts = {}
                for s in sites:
                    local_counts[classes[s]] = local_counts.get(classes[s], 0) + 1
                uniques = [s for s in sites if local_counts[classes[s]] == 1]
                unique_sites.append(uniques[0] if len(uniques) == 1 else None)
            if all(unique_sites):
                paired = any(set(pair) == set(unique_sites) for pair in t.pairs)
                bis_mode = "trans-fac" if paired else "cis-fac"
            else:
                bis_mode = "fac-fac"
        else:
            bis_mode = "fac-mer"

    return TopologyClass("octahedral", fac_mer, tuple(ligand_modes),
                         bis_mode, signature, reason)


def mol_from_trex(value: str | Trex) -> TrexMol:
    """Build the RDKit connectivity graph represented by T-REX.

    Atom indices in MAP are interpreted in ligand SMILES parse order, matching
    the 1-based per-ligand indexing defined by T-REX.
    """
    t = parse_trex(value) if isinstance(value, str) else value
    rw = Chem.RWMol()
    atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(t.metal)
    if not atomic_number:
        raise ValueError("unknown metal element %r" % t.metal)
    metal = Chem.Atom(atomic_number)
    metal.SetFormalCharge(t.oxidation_state)
    metal.SetNoImplicit(True)
    mi = rw.AddAtom(metal)
    ligand_atoms: list[list[int]] = []
    ligand_base_offsets: list[int] = []
    ligand_pos1b_to_global: list[list[int]] = []
    for li, ligand in enumerate(t.ligands, 1):
        piece = Chem.MolFromSmiles(ligand.text)
        if piece is None:
            raise ValueError("ligand %d has invalid SMILES: %s" % (li, ligand.text))
        before = rw.GetNumAtoms()
        rw.InsertMol(piece)
        atoms = list(range(before, before + piece.GetNumAtoms()))
        ligand_atoms.append(atoms)
        ligand_base_offsets.append(before)
        ligand_pos1b_to_global.append(
            [atoms[rdidx] for rdidx in _written_order_to_rdidx(ligand.text)])

    site_atoms: dict[Site, tuple[int, ...]] = {}
    dative_bond_ids: list[int] = []
    for site in t.sites:
        if site.ligand > len(ligand_atoms):
            raise ValueError("MAP refers to missing ligand %d" % site.ligand)
        positions = ligand_pos1b_to_global[site.ligand - 1]
        if any(i > len(positions) for i in site.atoms):
            raise ValueError("MAP atom index outside ligand %d" % site.ligand)
        global_atoms = tuple(sorted({positions[i - 1] for i in site.atoms}))
        site_atoms[site] = global_atoms
        for donor in global_atoms:
            if rw.GetBondBetweenAtoms(donor, mi) is None:
                rw.AddBond(donor, mi, Chem.BondType.DATIVE)
                bond = rw.GetBondBetweenAtoms(donor, mi)
                bond.SetBoolProp("TREX_is_coord", True)
                atom_text = (str(site.atoms[0]) if len(site.atoms) == 1 else
                             "[" + ",".join(map(str, site.atoms)) + "]")
                bond.SetProp("TREX_site", "%d:%s" % (site.ligand, atom_text))
                dative_bond_ids.append(bond.GetIdx())

    mol = rw.GetMol()
    status = Chem.SanitizeMol(mol, catchErrors=True)
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError("assembled T-REX molecule failed RDKit sanitization (%s)" % status)
    mol.SetProp("_T_REX", value if isinstance(value, str) else "")
    mol.SetProp("_T_REX_geometry", t.geometry or "")
    # Property names used by the official trex2mol implementation.
    mol.SetProp("TREX_Metal", t.metal)
    mol.SetIntProp("TREX_OxidationState", t.oxidation_state)
    if isinstance(value, str):
        mol.SetProp("TREX_String", value)
    if t.spin is not None:
        mol.SetIntProp("_T_REX_spin", t.spin)
        mol.SetIntProp("TREX_Spin", t.spin)
    trans_site_atoms = [(site_atoms[a], site_atoms[b]) for a, b in t.pairs]
    return TrexMol(mol, t, mi, ligand_atoms, site_atoms,
                   ligand_base_offsets, ligand_pos1b_to_global,
                   dative_bond_ids, trans_site_atoms)


def _site_angles(t: Trex) -> dict[Site, float]:
    """Choose a deterministic 2D projection that preserves every trans pair."""
    angles: dict[Site, float] = {}
    p = len(t.pairs)
    if p:
        # Independent trans axes are spread over a half-circle.  This gives the
        # familiar cross for square planar and three 60-degree axes for Oh.
        for i, (a, b) in enumerate(t.pairs):
            axis = 180.0 * i / p
            angles[a] = axis
            angles[b] = axis + 180.0

    singles = list(t.singles)
    if not singles:
        return angles
    if not angles:
        for i, site in enumerate(singles):
            angles[site] = 90.0 + 360.0 * i / len(singles)
        return angles

    # Put unpaired sites successively into the largest empty angular gaps.
    # This is O(n^2), but CN is <=7 in T-REX, so it is effectively constant.
    for site in singles:
        occupied = sorted(a % 360.0 for a in angles.values())
        gaps = [((occupied[(i + 1) % len(occupied)] - occupied[i]) % 360.0, i)
                for i in range(len(occupied))]
        gap, i = max(gaps)
        angles[site] = (occupied[i] + gap / 2.0) % 360.0
    return angles


def _circular_mean(values: Iterable[float]) -> float:
    vals = list(values)
    z = sum(complex(math.cos(math.radians(x)), math.sin(math.radians(x)))
            for x in vals)
    return math.degrees(math.atan2(z.imag, z.real)) if abs(z) > 1e-12 else vals[0]


def _apply_octahedral_stereo_bonds(mol: Chem.Mol, layout: TrexMol) -> Chem.Mol:
    """Add the conventional 2D perspective to a three-axis octahedron.

    One trans axis stays in the page.  Each of the other axes gets one solid
    and one hashed wedge, so opposite sites also point in opposite z
    directions.  This is presentation metadata only; MAP remains the source of
    coordination topology.
    """
    if len(layout.trex.pairs) != 3:
        return mol
    geometry = (layout.trex.geometry or "").lower()
    if geometry and geometry not in ("o", "oh", "oct", "octahedral"):
        return mol
    out = Chem.Mol(mol)
    for bond in out.GetBonds():
        if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE,
                                 Chem.BondDir.BEGINDASH):
            bond.SetBondDir(Chem.BondDir.NONE)

    conf = out.GetConformer()
    points = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                       for i in range(out.GetNumAtoms())])
    metal_xy = points[layout.metal_idx]
    def opposite_score(pair):
        vectors = []
        for site in pair:
            vectors.append(points[list(layout.site_atoms[site])].mean(0) - metal_xy)
        return -float(np.dot(vectors[0], vectors[1]) /
                      max(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]), 1e-9))
    topology = classify_topology(layout)
    if topology.fac_mer == "mer":
        classes = _site_equivalence_classes(layout.trex)
        cross_axes = [i for i, (a, b) in enumerate(layout.trex.pairs)
                      if classes[a] != classes[b]]
        flat_axis = cross_axes[0] if len(cross_axes) == 1 else max(
            range(3), key=lambda i: opposite_score(layout.trex.pairs[i]))
    else:
        flat_axis = max(range(3), key=lambda i: opposite_score(layout.trex.pairs[i]))

    flat_site = layout.trex.pairs[flat_axis][0]
    flat_vector = (points[list(layout.site_atoms[flat_site])].mean(0) -
                   metal_xy)
    flat_vector /= max(np.linalg.norm(flat_vector), 1e-9)

    # IUPAC's preferred OC-6 projection has the ordinary trans pair vertical.
    # Rotate the complete drawing (rather than individual ligands) so that the
    # already optimised layout, its symmetry and all bond lengths are retained.
    angle = math.atan2(flat_vector[1], flat_vector[0])
    turn = math.pi / 2.0 - angle
    ct, st = math.cos(turn), math.sin(turn)
    rotation = np.array([[ct, -st], [st, ct]])
    points = (points - metal_xy) @ rotation.T + metal_xy
    # Enantiomers use mirrored ligand placement, while the conventional OC-6
    # bond styles remain fixed (hashed above, solid below).
    if (layout.trex.chirality or "").strip().upper() in ("Λ", "LAMBDA"):
        points[:, 0] = 2.0 * metal_xy[0] - points[:, 0]
    for atom_idx, (x, y) in enumerate(points):
        old = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), old.z))

    for axis, (a, b) in enumerate(layout.trex.pairs):
        if axis == flat_axis:              # actual in-plane trans axis
            continue
        for site in (a, b):
            atoms = layout.site_atoms.get(site, ())
            if len(atoms) != 1:            # no atom-level wedge for eta sites
                continue
            donor = atoms[0]
            bond = out.GetBondBetweenAtoms(donor, layout.metal_idx)
            if bond is None:
                continue
            site_vector = points[list(atoms)].mean(0) - metal_xy
            # Preferred OC-6 perspective: the two upper bonds are hashed and
            # the two lower bonds are solid wedges.  fac/mer changes which
            # donors occupy these fixed positions, never the projection itself.
            upper = float(site_vector[1]) > 0.0
            # Coordination bonds retain their donor->metal direction. Bond
            # endpoints are immutable in RDKit's Python API; BEGINWEDGE and
            # BEGINDASH are therefore reversed later only in core._drawing_mol.
            bond.SetBoolProp("_TREX_wedgeFromMetal", True)
            bond.SetBondDir(Chem.BondDir.BEGINDASH if upper
                            else Chem.BondDir.BEGINWEDGE)
    return out


def _enforce_one_flat_trans_axis(mol: Chem.Mol, layout: TrexMol) -> Chem.Mol:
    """Make one MAP trans pair exactly collinear without introducing tangles."""
    if len(layout.trex.pairs) != 3:
        return mol
    try:
        from . import core
        from . import cluster
        base = Chem.Mol(mol)
        conf = base.GetConformer()
        xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                       for i in range(base.GetNumAtoms())], dtype=float)
        origin = xy[layout.metal_idx].copy()
        xy -= origin
        candidates = []

        def angle(v):
            return math.degrees(math.atan2(v[1], v[0]))

        topology = classify_topology(layout)
        site_classes = (_site_equivalence_classes(layout.trex)
                        if topology.fac_mer in ("fac", "mer") else {})
        preferred_pairs = set(range(len(layout.trex.pairs)))
        if topology.fac_mer == "mer":
            # The unique heterotypic A-B axis is the conventional in-plane
            # axis for mer-M(AB)3.  Choosing A-A or B-B instead is topologically
            # valid but visually piles the two same-type donors on one side.
            cross = {i for i, (a, b) in enumerate(layout.trex.pairs)
                     if site_classes[a] != site_classes[b]}
            if cross:
                preferred_pairs = cross

        for pair_index, (a, b) in enumerate(layout.trex.pairs):
            if pair_index not in preferred_pairs:
                continue
            if a.ligand == b.ligand:       # a chelate cannot be a trans axis
                continue
            va = xy[list(layout.site_atoms[a])].mean(0)
            vb = xy[list(layout.site_atoms[b])].mean(0)
            delta = (angle(va) + 180.0 - angle(vb) + 180.0) % 360.0 - 180.0
            ia = np.asarray(layout.ligand_atoms[a.ligand - 1], dtype=int)
            ib = np.asarray(layout.ligand_atoms[b.ligand - 1], dtype=int)
            for push in (0.0, 0.5, 1.0, 1.5, 2.0):
                trial = xy.copy()
                trial[ia] = trial[ia] @ core._rot(-delta / 2.0).T
                trial[ib] = trial[ib] @ core._rot(delta / 2.0).T
                for ids in (ia, ib):
                    centre = trial[ids].mean(0)
                    norm = np.linalg.norm(centre)
                    if push and norm > 1e-9:
                        trial[ids] += push * centre / norm
                candidate = Chem.Mol(base)
                cc = candidate.GetConformer()
                for i, (x, y) in enumerate(trial + origin):
                    cc.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
                q = (cluster._tangle(core._drawing_mol(candidate), strain=False),
                     cluster._tangle(core._drawing_mol(candidate), strain=True))
                candidates.append((q, push, pair_index, candidate))
        clean = [item for item in candidates if item[0][0] == 0]
        if clean:
            # In fac tris-chelates the three trans axes are symmetry-equivalent.
            # Their floating-point strain scores can differ only in the last
            # decimals, which used to make the selected in-plane axis jump when
            # unrelated depiction code changed.  Treat visually indistinguish-
            # able scores as a tie and retain MAP order as the stable tie-break.
            best_strain = min(item[0][1] for item in clean)
            tolerance = max(1e-6, abs(best_strain) * 1e-4)
            equivalent = [item for item in clean
                          if item[0][1] <= best_strain + tolerance]
            return min(equivalent,
                       key=lambda item: (item[2], item[1], item[0][1]))[3]
    except Exception:
        pass
    return mol


def _symmetrize_fac_tris_chelate(mol: Chem.Mol, layout: TrexMol) -> Chem.Mol:
    """Give fac-M(AB)3 with three identical ligands exact C3 symmetry.

    Independently optimizing the three chelates leaves tiny but very visible
    differences in their radii and orientation.  Build one undistorted sector
    and copy it by 120-degree rotations instead.  This is deliberately narrow:
    only three identical bidentate ligands whose MAP pairs are all A--B qualify.
    """
    from . import core
    t = layout.trex
    if len(t.ligands) != 3 or len(t.pairs) != 3:
        return mol
    if classify_topology(t).fac_mer != "fac":
        return mol
    if len({lig.kind + ":" + lig.text for lig in t.ligands}) != 1:
        return mol
    sites_by_lig = [[s for s in t.sites if s.ligand == li]
                    for li in (1, 2, 3)]
    if any(len(sites) != 2 or any(len(s.atoms) != 1 for s in sites)
           for sites in sites_by_lig):
        return mol
    roles = sorted({s.atoms for s in sites_by_lig[0]})
    if len(roles) != 2 or any(sorted(s.atoms for s in sites) != roles
                              for sites in sites_by_lig):
        return mol
    role = {atoms: i for i, atoms in enumerate(roles)}
    if any(role[a.atoms] == role[b.atoms] for a, b in t.pairs):
        return mol                         # mer contains A--A and B--B axes

    # Solve ligand rotations from MAP.  The two donor rays in the master
    # sector are 60 degrees apart; every A--B MAP pair must differ by 180.
    donor_angle = (90.0, 150.0)
    rotations = {1: 0.0}
    for _ in range(3):
        for a, b in t.pairs:
            if a.ligand in rotations and b.ligand not in rotations:
                rotations[b.ligand] = (rotations[a.ligand] +
                    donor_angle[role[a.atoms]] + 180.0 -
                    donor_angle[role[b.atoms]]) % 360.0
            elif b.ligand in rotations and a.ligand not in rotations:
                rotations[a.ligand] = (rotations[b.ligand] +
                    donor_angle[role[b.atoms]] + 180.0 -
                    donor_angle[role[a.atoms]]) % 360.0
    if len(rotations) != 3:
        return mol

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(out.GetNumAtoms())], dtype=float)
    origin = xy[layout.metal_idx].copy()
    master_ids = np.asarray(layout.ligand_atoms[0], dtype=int)
    master_sites = {role[s.atoms]: layout.site_atoms[s][0]
                    for s in sites_by_lig[0]}
    p0, p1 = xy[master_sites[0]], xy[master_sites[1]]
    source = p1 - p0
    radius = max(np.linalg.norm(source), 1e-6)  # 60-degree chord == radius
    target0 = origin + radius * np.array([
        math.cos(math.radians(donor_angle[0])),
        math.sin(math.radians(donor_angle[0]))])
    target1 = origin + radius * np.array([
        math.cos(math.radians(donor_angle[1])),
        math.sin(math.radians(donor_angle[1]))])
    target = target1 - target0
    turn = math.degrees(math.atan2(target[1], target[0]) -
                        math.atan2(source[1], source[0]))
    scale = np.linalg.norm(target) / np.linalg.norm(source)
    master = (xy[master_ids] - p0) @ core._rot(turn).T * scale + target0

    for li in (1, 2, 3):
        ids = np.asarray(layout.ligand_atoms[li - 1], dtype=int)
        if len(ids) != len(master_ids):
            return mol
        placed = (master - origin) @ core._rot(rotations[li]).T + origin
        xy[ids] = placed
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    return out


def _layout_mer_tris_chelate(mol: Chem.Mol, layout: TrexMol) -> Chem.Mol:
    """Balanced idealized projection for mer-M(AB)3.

    A regular six-ray projection can encode the mer signature as AABABB around
    the metal: opposite pairs are A-A, A-B and B-B, while each chelate still
    occupies two adjacent rays.  One chelate necessarily has the reverse sense;
    reflecting its master fragment keeps all three ligand bodies outside.
    """
    from . import core
    t = layout.trex
    if classify_topology(t).fac_mer != "mer" or len(t.ligands) != 3:
        return mol
    if len({lig.kind + ":" + lig.text for lig in t.ligands}) != 1:
        return mol
    classes = _site_equivalence_classes(t)
    class_names = sorted(set(classes.values()))
    if len(class_names) != 2:
        return mol
    ca, cb = class_names
    sites_by_lig = {li: [s for s in t.sites if s.ligand == li]
                    for li in (1, 2, 3)}
    if any(len(v) != 2 or {classes[s] for s in v} != {ca, cb}
           or any(len(s.atoms) != 1 for s in v)
           for v in sites_by_lig.values()):
        return mol

    cross = [pair for pair in t.pairs if classes[pair[0]] != classes[pair[1]]]
    aa = [pair for pair in t.pairs if classes[pair[0]] == classes[pair[1]] == ca]
    bb = [pair for pair in t.pairs if classes[pair[0]] == classes[pair[1]] == cb]
    if len(cross) != 1 or len(aa) != 1 or len(bb) != 1:
        return mol
    cross_a = next(s for s in cross[0] if classes[s] == ca)
    cross_b = next(s for s in cross[0] if classes[s] == cb)

    # A-B flat at 90/270.  The same-class partners are assigned so that every
    # physical ligand occupies adjacent 60-degree rays.
    target_angle = {cross_a: 90.0, cross_b: 270.0}
    a_for_cross_b = next(s for s in aa[0] if s.ligand == cross_b.ligand)
    a_other = next(s for s in aa[0] if s != a_for_cross_b)
    b_for_cross_a = next(s for s in bb[0] if s.ligand == cross_a.ligand)
    b_other = next(s for s in bb[0] if s != b_for_cross_a)
    target_angle.update({a_for_cross_b: 210.0, a_other: 30.0,
                         b_for_cross_a: 150.0, b_other: 330.0})

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(out.GetNumAtoms())], dtype=float)
    origin = xy[layout.metal_idx].copy()
    master_ids = np.asarray(layout.ligand_atoms[0], dtype=int)
    master_sites = {classes[s]: layout.site_atoms[s][0]
                    for s in sites_by_lig[1]}
    source_a, source_b = xy[master_sites[ca]], xy[master_sites[cb]]
    source_vec = source_b - source_a
    source_len = np.linalg.norm(source_vec)
    if source_len < 1e-8:
        return mol
    radius = source_len                 # chord of a 60-degree sector
    u = source_vec / source_len
    v = np.array([-u[1], u[0]])
    local = xy[master_ids] - source_a
    along, across = local @ u, local @ v

    for li in (1, 2, 3):
        ids = np.asarray(layout.ligand_atoms[li - 1], dtype=int)
        if len(ids) != len(master_ids):
            return mol
        site_a = next(s for s in sites_by_lig[li] if classes[s] == ca)
        site_b = next(s for s in sites_by_lig[li] if classes[s] == cb)
        pa = origin + radius * np.array([
            math.cos(math.radians(target_angle[site_a])),
            math.sin(math.radians(target_angle[site_a]))])
        pb = origin + radius * np.array([
            math.cos(math.radians(target_angle[site_b])),
            math.sin(math.radians(target_angle[site_b]))])
        target_vec = pb - pa
        scale = np.linalg.norm(target_vec) / source_len
        tu = target_vec / np.linalg.norm(target_vec)
        tv = np.array([-tu[1], tu[0]])
        candidates = [pa + scale * (along[:, None] * tu + sign *
                                    across[:, None] * tv)
                      for sign in (1.0, -1.0)]
        # Put the ligand body outside the coordination sphere.
        placed = max(candidates,
                     key=lambda p: np.linalg.norm(p.mean(0) - origin))
        xy[ids] = placed
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    return out


def _apply_topology(layout: TrexMol) -> Chem.Mol:
    from . import core

    mol = layout.mol
    # Start from the normal high-quality metal2d result.  The early prototype
    # disabled relaxation before imposing T-REX sectors; that happened to work
    # for four one-atom ligands but recreated severe contacts in tris-chelates
    # such as Ir(bpy)3.
    base = core.depict(mol, relax=True)
    result = Chem.Mol(base)
    conf = result.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(result.GetNumAtoms())], dtype=float)
    centre = xy[layout.metal_idx].copy()
    xy -= centre
    wanted = _site_angles(layout.trex)

    for li, atoms in enumerate(layout.ligand_atoms, 1):
        sites = [s for s in layout.trex.sites if s.ligand == li]
        if not sites:
            continue
        offsets = []
        for site in sites:
            donor = xy[list(layout.site_atoms[site])].mean(0)
            actual = math.degrees(math.atan2(donor[1], donor[0]))
            offsets.append((wanted[site] - actual + 180.0) % 360.0 - 180.0)
        angle = _circular_mean(offsets)
        result_xy = xy[atoms] @ core._rot(angle).T
        xy[atoms] = result_xy

    # Absolute orientation is arbitrary.  Try quarter turns and keep the one
    # with the tightest bounding box, which makes tall/wide ligands use the page.
    candidates = [xy @ core._rot(a).T for a in (0.0, 90.0, 180.0, 270.0)]
    xy = min(candidates, key=lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))

    # MAP is a topology constraint, not permission to make a worse picture.
    # Rigid site rotations can be geometrically over-constrained in a flat 2D
    # projection (notably three octahedral chelates).  Retain them when equally
    # clean -- cis/trans monodentate cases -- and otherwise fall back to the
    # ordinary relaxed layout.  This adds only two coordinate-only scores.
    try:
        from . import cluster
        topo_q = (cluster._tangle(core._drawing_mol(result), strain=False),
                  cluster._tangle(core._drawing_mol(result), strain=True))
        base_q = (cluster._tangle(core._drawing_mol(base), strain=False),
                  cluster._tangle(core._drawing_mol(base), strain=True))
        if base_q < topo_q:
            result = base
    except Exception:
        pass
    symmetric = _symmetrize_fac_tris_chelate(result, layout)
    if symmetric is result:
        symmetric = _layout_mer_tris_chelate(result, layout)
    if symmetric is result:
        result = _enforce_one_flat_trans_axis(result, layout)
    else:
        result = symmetric
    return _apply_octahedral_stereo_bonds(result, layout)


def depict_trex(value: str | Trex) -> Chem.Mol:
    """Parse, build and depict a T-REX complex with its topology preserved."""
    return _apply_topology(mol_from_trex(value))


def depict_input(value: str) -> Chem.Mol:
    """Unified string entry point: T-REX when detected, otherwise SMILES."""
    if is_trex(value):
        return depict_trex(value)
    from . import core
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError("input is neither valid T-REX nor valid SMILES")
    return core.depict(mol)


def draw_trex(value: str | Trex, path, size=(800, 800), title="") -> Chem.Mol:
    """Convenience API mirroring ``core.draw``; returns the depicted molecule."""
    from . import core
    mol = depict_trex(value)
    core.draw(mol, path, size=size, title=title)
    return mol


__all__ = ["Ligand", "Site", "Trex", "TrexMol", "is_trex", "parse_trex",
           "classify_topology", "mol_from_trex", "depict_trex",
           "depict_input", "draw_trex"]
