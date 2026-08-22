"""
metal2d.core - readable 2D depictions of coordination complexes with RDKit.

Pipeline
  1. cut every metal-donor bond
  2. depict each ligand on its own with CoordGen (plain organic -> fine)
  3. collapse eta-bonded groups (Cp, Cp*, arene, allyl...) into ONE pseudo-donor
     at the ring centroid, so they take a single coordination direction instead
     of five or six overlapping lines
  4. fit each ligand rigidly so its donors sit at natural chelate angles
     (M-donor length uniform, bite angle from the ligand's own donor...donor
     distance)
  5. share the 360 degrees around the metal between ligands in proportion to how
     wide each one actually is, instead of giving everyone an equal sector
  6. short clash relaxation (angular + radial) as a final polish

Priority is readability: ligands must fit without overlapping. Perfect
octahedral symmetry is sacrificed when it conflicts with that.
"""
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

METALS = (set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))
          # the d-block ranges above stop one short of the post-transition
          # metals that sit right after them: Ga(31), In(49) and Tl(81) were all
          # being missed, so gallium and indium complexes fell through to plain
          # CoordGen without anyone noticing
          | {13, 31, 49, 50, 51, 81, 82, 83, 84}
          | set(range(89, 104)))          # actinides: Th and U are coordinated too
ML = 1.5            # metal-donor bond length, drawing units
HAPTO_R = 1.9       # metal -> ring-centroid distance for eta-bonded groups
MAX_REACH = 4.5     # how far a ligand may be pushed out, in metal-donor bond
                    # lengths. Higher packs ligands apart more effectively but
                    # draws visibly over-long bonds to small ligands
LB = 1.15           # bond length inside a ligand. Deliberately shorter than ML:
                    # metal-donor bonds read better slightly long, as in
                    # published figures, but the gap must not be so large that
                    # small ligands like CO get swallowed by their own labels
METAL_COLOR = (0.85, 0.30, 0.55)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def find_metal(mol):
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in METALS:
            return a.GetIdx()
    return None


def _kabsch2d(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, qc - R @ pc


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _clashes(A, B, cut):
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    return float(np.sum(np.clip(cut - D, 0, None) ** 2))


def _ligand_geometry_defects(mol, xy):
    """Cheap count of defects already present in an isolated ligand layout."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not bonds:
        return 0
    lengths = [np.linalg.norm(xy[i] - xy[j]) for i, j in bonds]
    unit = float(np.median(lengths))
    if unit < 1e-9:
        return 10 ** 6
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    overlaps = int(np.sum(d < 0.25 * unit) // 2)
    crossings = 0
    for n, (i, j) in enumerate(bonds):
        p, r = xy[i], xy[j] - xy[i]
        for k, l in bonds[n + 1:]:
            if len({i, j, k, l}) < 4:
                continue
            q, s = xy[k], xy[l] - xy[k]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / den
            u = (qp[0] * r[1] - qp[1] * r[0]) / den
            crossings += (1e-6 < t < 1 - 1e-6 and
                          1e-6 < u < 1 - 1e-6)
    return 10 * overlaps + crossings


def _articulation_star_layout(mol, xy, donors=()):
    """Rebuild a crossed, balanced star ligand from clean rigid branches.

    The candidate is deliberately narrow: removing one articulation atom must
    yield at least three sizeable, similarly sized components.  It covers
    tripodal ligands without treating every phosphine substituent as a star.
    Returns ``None`` when that topology is absent or reconstruction is worse.
    """
    donor_set = set(donors)
    candidates = []
    atom_ids = set(range(mol.GetNumAtoms()))
    for centre in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(centre).GetDegree() < 3:
            continue
        remaining = atom_ids - {centre}
        components = []
        while remaining:
            first = remaining.pop()
            component, stack = {first}, [first]
            while stack:
                i = stack.pop()
                for neighbour in mol.GetAtomWithIdx(i).GetNeighbors():
                    j = neighbour.GetIdx()
                    if j in remaining:
                        remaining.remove(j)
                        component.add(j)
                        stack.append(j)
            components.append(component)
        sizes = [len(c) for c in components]
        one_donor_per_branch = all(len(c & donor_set) == 1
                                   for c in components)
        if (centre in donor_set and len(components) == 3 and
                one_donor_per_branch and min(sizes) >= 4 and
                min(sizes) >= 0.6 * max(sizes)):
            candidates.append((min(sizes), centre, components))
    if not candidates:
        return None
    _balance, centre, components = max(candidates)
    cut = [mol.GetBondBetweenAtoms(centre, n.GetIdx()).GetIdx()
           for n in mol.GetAtomWithIdx(centre).GetNeighbors()]
    maps = []
    fragments = Chem.GetMolFrags(
        Chem.FragmentOnBonds(mol, cut, addDummies=False),
        asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps)
    branches = [(fragment, list(amap))
                for fragment, amap in zip(fragments, maps)
                if list(amap) != [centre]]
    if len(branches) != len(components):
        return None

    centre_position = np.array([0.0, ML])
    prepared = []
    for branch, amap in branches:
        rdCoordGen.AddCoords(branch)
        conf = branch.GetConformer()
        local = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y]
                          for i in range(branch.GetNumAtoms())])
        attachment = next(
            i for i, glob in enumerate(amap)
            if mol.GetBondBetweenAtoms(centre, glob) is not None)
        body = local.mean(0) - local[attachment]
        old_angle = np.degrees(np.arctan2(body[1], body[0]))
        prepared.append((amap, local, attachment, old_angle))

    def spoke_crossings(candidate):
        count = 0
        for donor in donor_set:
            p, r = np.zeros(2), candidate[donor]
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if donor in (i, j):
                    continue
                q, s = candidate[i], candidate[j] - candidate[i]
                den = r[0] * s[1] - r[1] * s[0]
                if abs(den) < 1e-9:
                    continue
                qp = q - p
                t = (qp[0] * s[1] - qp[1] * s[0]) / den
                u = (qp[0] * r[1] - qp[1] * r[0]) / den
                count += (1e-6 < t < 1 - 1e-6 and
                          1e-6 < u < 1 - 1e-6)
        return count

    best = None
    for branch_angles in itertools.combinations(range(0, 360, 45), 3):
        candidate = np.zeros_like(xy)
        candidate[centre] = centre_position
        for angle, (amap, local, attachment, old_angle) in zip(
                branch_angles, prepared):
            direction = np.array([np.cos(np.radians(angle)),
                                  np.sin(np.radians(angle))])
            attachment_position = centre_position + LB * direction
            placed = ((local - local[attachment]) @
                      _rot(angle - old_angle).T + attachment_position)
            for i, glob in enumerate(amap):
                candidate[glob] = placed[i]
        defects = _ligand_geometry_defects(mol, candidate)
        on_metal = sum(i not in donor_set and
                       np.linalg.norm(candidate[i]) < 0.7 * LB
                       for i in range(mol.GetNumAtoms()))
        ring_on_metal = sum(
            np.linalg.norm(candidate[list(ring)].mean(0)) < 1.2 * LB
            for ring in mol.GetRingInfo().AtomRings())
        spokes = spoke_crossings(candidate)
        reach = max(np.linalg.norm(candidate[list(donor_set)], axis=1))
        value = (1000 * defects + 5000 * (on_metal + ring_on_metal) +
                 2000 * spokes + reach)
        if best is None or value < best[0]:
            best = value, candidate
    return (best[1] if best is not None and
            _ligand_geometry_defects(mol, best[1]) <
            _ligand_geometry_defects(mol, xy) else None)


def _hapto_groups(piece, local_donors):
    """Split a ligand's donor atoms into coordination groups.

    Grouping is by CONNECTIVITY, not ring membership: any connected set of three
    or more mutually bonded donor atoms is one eta-bonded group. This covers Cp,
    Cp*, arenes, allyl (not a ring), and - importantly - exocyclic donor atoms
    hanging off a Cp ring, which would otherwise get their own second bond to
    the metal."""
    # A metal donor is never part of an eta-bonded group. In a bridged cluster
    # the two bridging atoms are both bonded to the second metal, so plain
    # connectivity would read bridge-metal-bridge as one eta-3 group, collapse
    # it to a centroid and delete the very bonds that make the bridge.
    metal_d = [d for d in local_donors
               if piece.GetAtomWithIdx(d).GetAtomicNum() in METALS]
    dset = {d for d in local_donors if d not in metal_d}
    seen, groups = set(), []
    for d in local_donors:
        if d in seen or d in metal_d:
            continue
        comp, stack = set(), [d]
        while stack:
            i = stack.pop()
            if i in comp:
                continue
            comp.add(i)
            for nb in piece.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in dset and j not in comp:
                    stack.append(j)
        seen |= comp
        groups.append(sorted(comp))
    groups.extend([m] for m in metal_d)
    return groups


def _is_haptic_group(piece, group):
    """Whether connected donors represent one delocalised coordination site.

    Three or more contiguous donors cover Cp, arenes and allyl.  A two-atom
    group is eta-2 only when the donor atoms themselves share a multiple or
    aromatic bond; an ordinary single-bonded N,N/O,N chelate must remain two
    separate coordination sites.
    """
    if len(group) >= 3:
        return True
    if len(group) != 2:
        return False
    bond = piece.GetBondBetweenAtoms(group[0], group[1])
    return bond is not None and (bond.GetIsAromatic() or bond.GetBondType() in
                                 (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE))


def _haptic_radius(piece, group):
    return HAPTO_R if len(group) >= 3 else ML


def _hapto_centre(piece, group, xy):
    """Where the single bond should point: the ring centroid if the group
    contains a ring, otherwise the centroid of the whole group."""
    ring = [i for i in group if piece.GetAtomWithIdx(i).IsInRing()]
    return xy[ring if ring else group].mean(0)


def _has_haptic_coordination(mol):
    """Whether a metal has a connected eta-bound donor set of 3+ atoms."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in METALS:
            continue
        donors = {n.GetIdx() for n in atom.GetNeighbors()
                  if n.GetAtomicNum() not in METALS}
        seen = set()
        for start in donors:
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                i = stack.pop()
                if i in comp:
                    continue
                comp.add(i)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                             if n.GetIdx() in donors and n.GetIdx() not in comp)
            seen |= comp
            if len(comp) >= 3:
                return True
    return False


def _equalize_atomic_ligand_radii(mol, metal_idx, ligand_maps):
    """Equalize M--L lengths when every coordinated ligand is one atom.

    The collision relaxer may push ligands radially, which is useful for real
    ligand bodies but order-dependent for bare Cl-, NH3, CO-like one-atom
    fragments.  In an entirely atomic coordination sphere no internal geometry
    can be damaged, so restore a common median radius after relaxation.
    """
    if not ligand_maps or any(len(amap) != 1 for amap in ligand_maps):
        return mol
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(mol.GetNumAtoms())], dtype=float)
    atoms = np.asarray([amap[0] for amap in ligand_maps], dtype=int)
    origin = xy[metal_idx].copy()
    vectors = xy[atoms] - origin
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii < 1e-8):
        return mol
    radius = float(np.median(radii))
    xy[atoms] = origin + vectors * (radius / radii)[:, None]
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    return mol


def _donors_in_cyclic_core(mol, atom_map, donor_ids):
    """Whether all donors lie on one genuine macrocyclic ring.

    Testing only membership in a graph 2-core is too broad: fused aromatic and
    ordinary chelate systems also satisfy it.  The original sanitised molecule
    retains its SSSR rings, including GEJVAS's 17-membered donor contour, so
    require one common ring of at least eight atoms.
    """
    atoms = set(atom_map)
    donors = set(donor_ids)
    if len(donors) < 3 or not donors.issubset(atoms):
        return False
    return any(len(ring) >= 8 and set(ring).issubset(atoms) and
               donors.issubset(set(ring))
               for ring in mol.GetRingInfo().AtomRings())


def _macrocycle_self_crossings(xy, ring):
    """Count crossings made by the perimeter of one ordered macrocycle."""
    edges = [(ring[i], ring[(i + 1) % len(ring)])
             for i in range(len(ring))]
    count = 0
    for a, (i, j) in enumerate(edges):
        p, q = xy[i], xy[j]
        for k, l in edges[a + 1:]:
            if len({i, j, k, l}) < 4:
                continue
            u, v = xy[k], xy[l]
            r, s = q - p, v - u
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            w = u - p
            t = (w[0] * s[1] - w[1] * s[0]) / den
            z = (w[0] * r[1] - w[1] * r[0]) / den
            count += 1e-6 < t < 1 - 1e-6 and 1e-6 < z < 1 - 1e-6
    return count


def _macrocycle_area_ratio(xy, ring):
    """Area of a ring relative to an equally edged regular polygon.

    A macrocycle can be badly folded without its perimeter literally crossing
    itself.  Normalising by the median perimeter edge makes this independent
    of drawing scale and ring size; values near one describe an open contour,
    while a small value identifies a collapsed cavity.
    """
    points = xy[np.asarray(ring, dtype=int)]
    shifted = np.roll(points, -1, axis=0)
    area = 0.5 * abs(np.sum(points[:, 0] * shifted[:, 1] -
                            shifted[:, 0] * points[:, 1]))
    edges = np.linalg.norm(shifted - points, axis=1)
    side = float(np.median(edges))
    if side < 1e-9 or len(ring) < 3:
        return 0.0
    ideal = (len(ring) * side * side /
             (4.0 * np.tan(np.pi / len(ring))))
    return float(area / max(ideal, 1e-9))


def _unfold_macrocycle_coords(mol, xy, ring, metal_idx, bond_length=LB):
    """Open a crossed macrocycle while preserving its external fragments.

    The perimeter is replaced by a regular polygon. Components outside it are
    moved by rigid/similarity transforms from their old attachment points to
    the new ones, and components ending up inside are reflected outward. This
    keeps phenyl and other substituent geometry intact instead of asking a
    second depictor to rebuild it.
    """
    out = xy.copy()
    ring = list(ring)
    ring_set = set(ring)
    centre_old = xy[ring].mean(0)
    radius = bond_length / (2.0 * np.sin(np.pi / len(ring)))
    start = np.arctan2(*(xy[ring[0]] - centre_old)[::-1])
    for k, atom in enumerate(ring):
        angle = start + 2.0 * np.pi * k / len(ring)
        out[atom] = radius * np.array([np.cos(angle), np.sin(angle)])

    remaining = set(range(mol.GetNumAtoms())) - ring_set - {metal_idx}
    components = []
    while remaining:
        first = remaining.pop()
        component, stack = {first}, [first]
        while stack:
            i = stack.pop()
            for neighbour in mol.GetAtomWithIdx(i).GetNeighbors():
                j = neighbour.GetIdx()
                if j in remaining:
                    remaining.remove(j)
                    component.add(j)
                    stack.append(j)
        components.append(component)

    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    def reflect(points, a, b):
        axis = b - a
        axis /= max(np.linalg.norm(axis), 1e-9)
        relative = points - a
        parallel = np.outer(relative @ axis, axis)
        return a + 2.0 * parallel - relative

    for component in components:
        attachments = sorted({
            neighbour.GetIdx()
            for i in component
            for neighbour in mol.GetAtomWithIdx(i).GetNeighbors()
            if neighbour.GetIdx() in ring_set
        })
        ids = np.asarray(sorted(component), dtype=int)
        if len(attachments) >= 2:
            a, b = max(
                ((i, j) for n, i in enumerate(attachments)
                 for j in attachments[n + 1:]),
                key=lambda pair: np.linalg.norm(
                    xy[pair[0]] - xy[pair[1]]))
            old_mid = (xy[a] + xy[b]) / 2.0
            new_mid = (out[a] + out[b]) / 2.0
            old_axis, new_axis = xy[b] - xy[a], out[b] - out[a]
            angle = (np.arctan2(new_axis[1], new_axis[0]) -
                     np.arctan2(old_axis[1], old_axis[0]))
            scale = (np.linalg.norm(new_axis) /
                     max(np.linalg.norm(old_axis), 1e-9))
            out[ids] = ((xy[ids] - old_mid) @ rotation(angle).T * scale +
                        new_mid)
            if np.dot(out[ids].mean(0) - new_mid, new_mid) < 0.0:
                out[ids] = reflect(out[ids], out[a], out[b])
        elif len(attachments) == 1:
            anchor = attachments[0]
            boundary = [i for i in component if any(
                n.GetIdx() == anchor
                for n in mol.GetAtomWithIdx(i).GetNeighbors())]
            old_vector = xy[boundary].mean(0) - xy[anchor]
            outward = out[anchor]
            angle = (np.arctan2(outward[1], outward[0]) -
                     np.arctan2(old_vector[1], old_vector[0]))
            out[ids] = ((xy[ids] - xy[anchor]) @ rotation(angle).T +
                        out[anchor])
    out[metal_idx] = (0.0, 0.0)
    return out


def _order_groups(piece, groups):
    """Put a ligand's coordination groups in the order they occur along the
    chelate, not in whatever order the metal's bonds happened to be listed.

    For anything past a bidentate ligand this matters: a tridentate O,N,S donor
    set arriving as O,S,N puts the two ends of the ligand into adjacent slots and
    throws the middle donor across the metal. The chain order is the Hamiltonian
    path through the donors that minimises the total topological distance."""
    if len(groups) < 3:
        return groups
    D = Chem.GetDistanceMatrix(piece)
    rep = [g[0] for g in groups]
    n = len(groups)

    def cost(order):
        return sum(D[rep[order[t]], rep[order[t + 1]]] for t in range(n - 1))

    if n <= 7:
        best = min(itertools.permutations(range(n)), key=cost)
    else:                                   # greedy chain for very high denticity
        best = [0]
        left = set(range(1, n))
        while left:
            nxt = min(left, key=lambda j: D[rep[best[-1]], rep[j]])
            best.append(nxt)
            left.discard(nxt)
    return [groups[t] for t in best]


def _linear_polydentate_scaffold(piece, xy, donors, radius=1.8):
    """Build a rigid-block layout for a linear 4+ donor chelate.

    Free-ligand depictors naturally draw long tetraphosphines open.  A rigid
    fit of that picture onto four coordination sites crushes every aromatic
    block.  Here the ordered donor chain defines a semicircular cavity first;
    rings and donor substituent branches are then moved only by rigid
    transforms.  The deliberately narrow topology guard keeps this out of the
    normal bi-/tridentate path.
    """
    donors = list(donors)
    if not (4 <= len(donors) <= 6):
        return None
    # This rigid-block construction models linked phosphine cages. O/N
    # macrocycles already have a dedicated cavity path and must retain it.
    if not all(piece.GetAtomWithIdx(d).GetAtomicNum() == 15 for d in donors):
        return None
    paths = [Chem.GetShortestPath(piece, donors[i], donors[i + 1])
             for i in range(len(donors) - 1)]
    if any(not p or len(p) - 1 > 4 for p in paths):
        return None
    # A genuine chain has no non-neighbour shortcut of comparable length.
    D = Chem.GetDistanceMatrix(piece)
    if any(D[donors[i], donors[j]] <= 4
           for i in range(len(donors)) for j in range(i + 2, len(donors))):
        return None

    out = xy.copy()
    k = len(donors)
    angles = np.radians(np.linspace(180.0, 0.0, k))
    target = np.column_stack((radius * np.cos(angles),
                              radius * np.sin(angles)))
    out[donors] = target
    placed = set(donors)
    backbone = set(donors)
    for p in paths:
        backbone.update(p)

    def component_without_edge(start, u, v):
        seen, stack = {start}, [start]
        while stack:
            a = stack.pop()
            for nb in piece.GetAtomWithIdx(a).GetNeighbors():
                b = nb.GetIdx()
                if {a, b} == {u, v} or b in seen:
                    continue
                seen.add(b)
                stack.append(b)
        return seen

    def rigid_place(ids, a, b, qa, qb):
        """Move a block so old anchors a,b land on qa,qb (no deformation)."""
        old = xy[b] - xy[a]
        new = qb - qa
        if np.linalg.norm(old) < 1e-8 or np.linalg.norm(new) < 1e-8:
            return
        ang = (np.arctan2(new[1], new[0]) -
               np.arctan2(old[1], old[0]))
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]])
        for q in ids:
            out[q] = (xy[q] - xy[a]) @ R.T + qa

    try:
        Chem.GetSymmSSSR(piece)  # fragments made without sanitisation lack rings
    except Exception:
        pass
    rings = [set(r) for r in piece.GetRingInfo().AtomRings()]
    for seg, path in enumerate(paths):
        internal = list(path[1:-1])
        if not internal:
            continue
        common_ring = next((r for r in rings
                            if set(internal).issubset(r)), None)
        if common_ring is not None and len(internal) >= 2:
            # Keep an aromatic bridge rigid and put it outside the donor arc.
            a, b = internal[0], internal[-1]
            p0, p1 = target[seg], target[seg + 1]
            chord = p1 - p0
            unit = chord / max(np.linalg.norm(chord), 1e-9)
            normal = np.array([-unit[1], unit[0]])
            if np.dot(normal, (p0 + p1) / 2) < 0:
                normal = -normal
            sep = np.linalg.norm(xy[b] - xy[a])
            qa = (p0 + p1) / 2 - unit * sep / 2 + normal * LB
            qb = (p0 + p1) / 2 + unit * sep / 2 + normal * LB
            # Move the whole ring-side component, but never another donor.
            ids = set(common_ring)
            frontier = list(common_ring)
            while frontier:
                q = frontier.pop()
                for nb in piece.GetAtomWithIdx(q).GetNeighbors():
                    z = nb.GetIdx()
                    if z in donors or z in ids:
                        continue
                    # Do not absorb a different backbone segment.
                    if z in backbone and z not in common_ring:
                        continue
                    ids.add(z)
                    frontier.append(z)
            rigid_place(ids, a, b, qa, qb)
            # The two anchors determine a rigid ring only up to reflection.
            # Choose the side away from the metal; otherwise an ortho-phenylene
            # bridge is placed directly inside the coordination cavity.
            axis = qb - qa
            axis /= max(np.linalg.norm(axis), 1e-9)
            block = sorted(ids)
            reflected = out[block].copy()
            w = reflected - qa
            reflected = qa + 2 * np.outer(w @ axis, axis) - w
            ring_ids = sorted(common_ring)
            local_ring = [block.index(z) for z in ring_ids]
            # ``qa/qb`` lie on the cavity-facing edge of the bridge; the ring
            # body belongs on the opposite side of that edge.
            out[block] = reflected
            placed.update(ids)
        else:
            # Flexible short bridge: bow its internal atoms away from the
            # cavity. Exact bond lengths are polished by the normal renderer.
            p0, p1 = target[seg], target[seg + 1]
            chord = p1 - p0
            unit = chord / max(np.linalg.norm(chord), 1e-9)
            normal = np.array([-unit[1], unit[0]])
            if np.dot(normal, (p0 + p1) / 2) < 0:
                normal = -normal
            for n, q in enumerate(internal, 1):
                t = n / (len(internal) + 1)
                out[q] = ((1 - t) * p0 + t * p1 +
                          normal * 0.65 * LB * np.sin(np.pi * t))
                placed.add(q)

    # Rigid donor-free branches (notably P-phenyls) go into the free outward
    # directions. Choose among a tiny angle grid by clashes with placed atoms.
    for di, donor in enumerate(donors):
        branches = []
        for nb in piece.GetAtomWithIdx(donor).GetNeighbors():
            q = nb.GetIdx()
            if q in backbone:
                continue
            ids = component_without_edge(q, donor, q)
            if ids & set(donors):
                continue
            branches.append((q, ids))
        used = []
        for q, ids in branches:
            best = None
            radial = target[di] / max(np.linalg.norm(target[di]), 1e-9)
            base_ang = np.degrees(np.arctan2(radial[1], radial[0]))
            old = xy[q] - xy[donor]
            old_ang = np.degrees(np.arctan2(old[1], old[0]))
            for delta in (0, -45, 45, -90, 90, 135, -135, 180):
                ang = base_ang + delta
                R = _rot(ang - old_ang)
                cand = np.array([(xy[z] - xy[donor]) @ R.T + target[di]
                                 for z in sorted(ids)])
                others = [z for z in placed if z not in ids]
                clash = (_clashes(cand, out[others], 0.72 * LB)
                         if others else 0.0)
                angular = sum(max(0.0, 35.0 - abs((ang - u + 180) % 360 - 180))
                              for u in used)
                value = 1000.0 * clash + angular + 0.02 * abs(delta)
                if best is None or value < best[0]:
                    best = value, R, ang
            _value, R, ang = best
            for z in ids:
                out[z] = (xy[z] - xy[donor]) @ R.T + target[di]
            placed.update(ids)
            used.append(ang)

    # Move any residual side component with its already positioned attachment.
    # This mainly catches substituents on a flexible bridge atom rather than on
    # the donor itself.
    remaining = set(range(piece.GetNumAtoms())) - placed
    while remaining:
        link = next(((a, nb.GetIdx()) for a in placed
                     for nb in piece.GetAtomWithIdx(a).GetNeighbors()
                     if nb.GetIdx() in remaining), None)
        if link is None:
            break
        anchor, first = link
        ids, stack = {first}, [first]
        while stack:
            q = stack.pop()
            for nb in piece.GetAtomWithIdx(q).GetNeighbors():
                z = nb.GetIdx()
                if z in remaining and z not in ids:
                    ids.add(z)
                    stack.append(z)
        shift = out[anchor] - xy[anchor]
        for z in ids:
            out[z] = xy[z] + shift
        placed.update(ids)
        remaining.difference_update(ids)

    if len(placed) < piece.GetNumAtoms() or not np.isfinite(out).all():
        return None
    return out


def _chelate_conformer(piece, xy, local_donors, passes=3):
    """Bring a chelate's donor atoms together.

    CoordGen depicts a free 2,2'-bipyridine the way it is normally drawn on its
    own: s-trans, with the two nitrogens pointing in opposite directions. Fitting
    that rigidly onto a pair of coordination slots is impossible, and the result
    is a ligand whose nitrogens face away from the metal. Rotating about the
    inter-ring bond is, in two dimensions, simply a reflection of everything past
    that bond, so the fix is to try reflecting across each rotatable bond lying
    between the donors and keep whatever brings them closer together.
    """
    if len(local_donors) < 2:
        return xy
    xy = xy.copy()
    # rotatable bonds on every donor-to-donor path, acyclic single bonds only
    pairs = set()
    for i in range(len(local_donors) - 1):
        pairs.add((local_donors[i], local_donors[i + 1]))
    pairs.add((local_donors[0], local_donors[-1]))
    path = []
    for a, b in pairs:
        p = Chem.GetShortestPath(piece, int(a), int(b))
        path.extend(zip(p[:-1], p[1:]))
    if not path:
        return xy

    axes, seen_axis = [], set()
    for u, v in path:
        if (u, v) in seen_axis:
            continue
        seen_axis.add((u, v))
        bd = piece.GetBondBetweenAtoms(u, v)
        if bd is None or bd.IsInRing():
            continue
        # A chelating conformation often needs an imine C=N drawn syn rather than
        # anti. Reflecting across a double bond changes E/Z, so only do it where
        # the configuration was never specified: there the depiction engine chose
        # a side arbitrarily and we are free to choose another.
        if bd.GetBondType() == Chem.BondType.DOUBLE:
            if bd.GetStereo() != Chem.BondStereo.STEREONONE:
                continue
        elif bd.GetBondType() != Chem.BondType.SINGLE:
            continue
        em = Chem.RWMol(piece)
        em.RemoveBond(u, v)
        frags = Chem.GetMolFrags(em.GetMol())
        side = next((set(f) for f in frags if v in f and u not in f), None)
        if side:
            axes.append((u, v, sorted(side)))
    if not axes:
        return xy

    def objective(pts):
        """What we want the donors to look like. For two donors, close together.
        For three or more, sitting on a circle small enough for a metal to be
        bonded to all of them: minimising only the end-to-end distance can leave
        a meridional ligand with its donors almost collinear, which no metal
        position can satisfy."""
        d = pts[local_donors]
        if len(d) < 3:
            return float(np.linalg.norm(d[-1] - d[0]))
        c, r = _donor_circle(d)
        if r is None or not np.isfinite(r):
            return 1e6
        # three points always lie on some circle, four need not: measure how far
        # they actually are from it, otherwise a tetradentate ligand can score
        # well while one of its donors ends up nearly on top of the metal
        dev = float(np.sqrt(((np.linalg.norm(d - c, axis=1) - r) ** 2).mean()))
        # the circle centre is where the metal will go, so nothing else may be
        # sitting there: a ligand can wrap so far round that its own backbone
        # crosses the centre, which fits the donors but is undrawable
        rest = np.delete(pts, local_donors, axis=0)
        pen = 0.0
        if len(rest):
            gap = np.clip(0.9 * ML - np.linalg.norm(rest - c, axis=1), 0, None)
            pen = float((gap ** 2).sum())
        return (abs(r - ML) + 2.0 * dev
                + 0.1 * float(np.linalg.norm(d[-1] - d[0])) + 5.0 * pen)

    def selfclash(pts):
        D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1) + np.eye(len(pts)) * 99
        return int((D < 0.7 * LB).sum())

    base = selfclash(xy)

    def apply(bits):
        out = xy.copy()
        for bit, (u, v, side) in zip(bits, axes):
            if not bit:
                continue
            d = out[v] - out[u]
            nrm = np.linalg.norm(d)
            if nrm < 1e-6:
                continue
            d = d / nrm
            w = out[side] - out[u]
            out[side] = out[u] + 2 * np.outer(w @ d, d) - w
        return out

    # Reflections interact: flipping one bond changes which flip is best at the
    # next. A greedy pass therefore stalls in a local minimum - on a tridentate
    # thiosemicarbazone it stopped at a donor circle of radius 3.0 while an
    # exhaustive search found 1.2. Enumerate when the count is affordable.
    if len(axes) <= 9:
        best, best_bits = objective(xy), None
        for bits in _combinations(len(axes)):
            cand = apply(bits)
            if selfclash(cand) > base:
                continue
            v = objective(cand)
            if v < best - 1e-9:
                best, best_bits = v, bits
        return apply(best_bits) if best_bits is not None else xy

    for _ in range(passes):
        improved = False
        for i in range(len(axes)):
            cand = apply([0] * i + [1] + [0] * (len(axes) - i - 1))
            if (objective(cand) < objective(xy) - 1e-6
                    and selfclash(cand) <= max(base, selfclash(xy))):
                xy = cand
                improved = True
        if not improved:
            break
    return xy


def _combinations(n):
    for mask in range(1, 1 << n):
        yield [(mask >> i) & 1 for i in range(n)]


def _donor_circle(pts):
    """Centre and radius of the circle through the donor atoms (exact for three,
    least squares beyond). The metal belongs at that centre: it is the only point
    equidistant from every donor, so all metal-donor bonds come out equal and the
    ligand needs no distortion to reach its slots."""
    P = np.asarray(pts, dtype=float)
    if len(P) < 3:
        return None, None
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    c = sol[:2]
    r2 = sol[2] + c @ c
    if not np.isfinite(r2) or r2 <= 0:
        return None, None
    return c, float(np.sqrt(r2))


def _angular_width(xy, rmin=0.8 * ML):
    """Half-width of a ligand as seen from the metal, in degrees.

    Every atom past the donor shell counts. Filtering on a fraction of the
    maximum radius, as an earlier version did, silently dropped substituents
    that hug the metal - a sulfonate arm folded back towards the centre then
    reported a narrow ligand and later cut across a neighbour's bond."""
    r = np.linalg.norm(xy, axis=1)
    if len(r) == 0 or r.max() < 1e-6:
        return 10.0
    keep = xy[r > rmin]
    if len(keep) == 0:
        keep = xy
    a = np.degrees(np.arctan2(keep[:, 1], keep[:, 0]))
    a = (a + 180.0) % 360.0 - 180.0
    return float(min(max(np.abs(a).max(), 8.0), 178.0))


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def depict(mol, ml=ML, relax=True, pad=6.0, _depth=0, cluster=True,
           _choose_root=True):
    mol = Chem.Mol(mol)

    # Metals bonded to each other need the cluster layout: the single-centre
    # model below would cut the M-M-donor rings that bridges form. Imported
    # lazily, and re-entered with cluster=False, so the two modules can call
    # each other without looping.
    if cluster:
        try:
            from . import cluster as _cl
            if _cl.has_cluster(mol):
                out = _cl.depict(mol, relax=relax, _depth=_depth)
                if out is not None:
                    out = _cl.polish_organic_branches(out)
                    out = _cl.polish_bridge_sides(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.polish_haptic_sectors(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.polish_anchored_bridges(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.orient_mu_carbonyl_opposite_bridge(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.centre_shared_organic_bridge(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.symmetrize_haptic_pair(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.orient_two_end_organic_bridge(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.symmetrize_equivalent_terminal_pairs(
                        out, _cl.metal_clusters(mol)[0])
                    out = _cl.orient_metal_bound_carbonyls_outward(
                        out, _cl.metal_clusters(mol)[0])
                    # A rigid M-M template can be actively harmful for large
                    # ordinary bridging ligands: the Ag2 bis-phenanthroline
                    # case is clean in CoordGen but heavily crossed in the
                    # template.  Keep CoordGen as a candidate when there is no
                    # eta-bound ring.  Haptic Fe/Cp clusters deliberately stay
                    # on their strict specialised path.
                    if not _has_haptic_coordination(mol):
                        try:
                            plain = Chem.Mol(mol)
                            plain.RemoveAllConformers()
                            rdCoordGen.AddCoords(plain)
                            candidates = [out, plain]
                            # Whole ligands spanning both metals can defeat
                            # both the rigid template and CoordGen.  The normal
                            # recursive layout often keeps their large aromatic
                            # systems readable, albeit with longer closing
                            # coordination bonds, so include it as a candidate.
                            if _cl.ligand_bridged_pairs(mol):
                                recursive = depict(
                                    mol, ml=ml, relax=relax, pad=pad,
                                    _depth=_depth, cluster=False)
                                candidates.append(recursive)
                            return min(
                                candidates,
                                key=lambda c: (
                                    _cl._tangle(_drawing_mol(c), strain=False),
                                    _cl._tangle(_drawing_mol(c), strain=True)))
                        except Exception:
                            pass
                    return out
            else:
                # Metals with no M-M bond but sharing bridging donor atoms
                # (halide-bridged dimers and the like) are clusters in every way
                # that matters to the layout, so use the same rigid template.
                # The macrocycle path also claims them - a single-atom bridge is
                # a fragment touching two metals - and it hands the whole
                # molecule to the classic depictor, which knows nothing about
                # coordination geometry and stacks the terminal ligands. Try
                # both and keep the less tangled one.
                cands = []
                for pr in _cl.atom_bridged_pairs(mol):
                    try:
                        out = _cl.depict(mol, relax=relax, _depth=_depth, pair=pr)
                        if out is not None:
                            cands.append(out)
                    except Exception:
                        pass
                    break
                pairs = _cl.ligand_bridged_pairs(mol)
                if pairs:
                    out = _cl.depict_macrocycle(mol, pairs[0])
                    if out is not None:
                        cands.append(out)
                    # One rigid multidentate ligand can pin both centres even
                    # though there are not two separate macrocycle arms.  The
                    # cluster template keeps eta-bound rings outside their
                    # metals; a pinned classic layout may collapse each metal
                    # exactly onto its ring centroid.
                    try:
                        paired = _cl.depict(mol, relax=relax, _depth=_depth,
                                            pair=pairs[0])
                        if paired is not None:
                            paired = _cl.symmetrize_haptic_pair(
                                paired, pairs[0])
                            paired = _cl.polish_terminal_atoms(
                                paired, pairs[0])
                            cands.append(paired)
                    except Exception:
                        pass
                    # Some non-M-M dimers with two whole bridging ligands are
                    # already laid out perfectly by CoordGen.  Previously this
                    # candidate was considered only for an explicit M-M bond,
                    # forcing a choice between a crowded macrocycle template
                    # and a recursive layout with severely stretched bonds.
                    try:
                        plain = Chem.Mol(mol)
                        plain.RemoveAllConformers()
                        rdCoordGen.AddCoords(plain)
                        cands.append(plain)
                    except Exception:
                        pass
                if cands:
                    # A rigid cluster template can preserve short M-X bonds yet
                    # make the organic drawing unreadable.  Keep the ordinary
                    # recursive layout as a candidate and rank legibility before
                    # bond strain; this is especially important when a bridging
                    # O belongs to a larger chelating ligand rather than being an
                    # independent one-atom bridge.
                    try:
                        ordinary = depict(mol, ml=ml, relax=relax, pad=pad,
                                          _depth=_depth, cluster=False)
                        ordinary = _cl.polish_bridge_ligands(ordinary,
                                                             pairs[0] if pairs
                                                             else pr)
                        ordinary = _cl.polish_terminal_sectors(ordinary,
                                                               pairs[0] if pairs
                                                               else pr)
                        if pairs:
                            ordinary = _cl.symmetrize_equivalent_metal_halves(
                                ordinary, pairs[0])
                            ordinary = _cl.polish_terminal_atoms(
                                ordinary, pairs[0])
                        cands.append(ordinary)
                    except Exception:
                        pass
                    return min(
                        cands,
                        key=lambda c: (
                            _cl.haptic_centroid_penalty(c),
                            _cl._tangle(c, strain=False),
                            _cl._tangle(c, strain=True)))
        except ImportError:
            pass
        except Exception:
            pass

    # In a polynuclear complex without an M-M/shared-atom core the layout is
    # recursive: one metal is placed first and the rest of the complex is
    # treated as one of its ligands.  That construction is asymmetric, so the
    # first metal in SMILES order is not necessarily a good root.  Try every
    # centre at the outermost level and retain the least tangled depiction.
    # Renumbering changes only the traversal order; coordinates are mapped back
    # immediately, so callers still receive the original atom ordering.
    metals = [a.GetIdx() for a in mol.GetAtoms()
              if a.GetAtomicNum() in METALS]
    if _choose_root and _depth == 0 and len(metals) > 1:
        candidates = []
        for root in metals:
            order = [root] + [i for i in range(mol.GetNumAtoms()) if i != root]
            renum = Chem.RenumberAtoms(mol, order)
            try:
                cand = depict(renum, ml=ml, relax=relax, pad=pad,
                              _depth=_depth, cluster=False,
                              _choose_root=False)
                inverse = [order.index(i) for i in range(len(order))]
                candidates.append(Chem.RenumberAtoms(cand, inverse))
            except Exception:
                pass
        if candidates:
            try:
                from . import cluster as _cl
                return min(candidates, key=_cl._tangle)
            except (ImportError, AttributeError):
                return candidates[0]

    mi = find_metal(mol)

    def _fallback():
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        rdCoordGen.AddCoords(m)
        return m

    if mi is None:
        return _fallback()
    mbonds = list(mol.GetAtomWithIdx(mi).GetBonds())
    if not mbonds:
        return _fallback()

    cut = [b.GetIdx() for b in mbonds]
    donors = [b.GetOtherAtomIdx(mi) for b in mbonds]

    frag_mol = Chem.FragmentOnBonds(mol, cut, addDummies=False)
    pieces = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False,
                              fragsMolAtomMapping=(maps := []))
    frags = [(p, list(m)) for p, m in zip(pieces, maps) if list(m) != [mi]]
    ligands = [(p, m) for p, m in frags if any(d in m for d in donors)]
    spectators = [(p, m) for p, m in frags if not any(d in m for d in donors)]
    if not ligands:
        return _fallback()

    # ---- 1. canonical fit of every ligand, centred on angle 0 -------------- #
    fitted = []
    ligand_templates = {}
    articulation_cavity = False
    linear_scaffold_used = False
    for piece, amap in ligands:
        template_key = None
        # A ligand fragment may itself contain a metal centre (bridged
        # polynuclear complex). Plain CoordGen would tie that centre into a
        # knot, so lay the sub-complex out with the same algorithm instead.
        if _depth < 4 and find_metal(piece) is not None:
            try:
                sub = depict(piece, ml=ml, relax=relax, pad=pad, _depth=_depth + 1)
                conf = sub.GetConformer()
                piece.RemoveAllConformers()
                piece.AddConformer(Chem.Conformer(conf), assignId=True)
            except Exception:
                rdCoordGen.AddCoords(piece)
        else:
            rdCoordGen.AddCoords(piece)
        c = piece.GetConformer()
        xy = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                       for i in range(piece.GetNumAtoms())])
        local_d = [amap.index(d) for d in donors if d in amap]
        prefitted_star = False
        # CoordGen is usually the better and faster ligand depictor, but some
        # charged conjugated chelates are returned already self-crossed.  Do
        # not make every ligand pay for two layouts: invoke classic RDKit only
        # after the inexpensive geometric check finds a real defect.
        primary_defects = _ligand_geometry_defects(piece, xy)
        belongs_to_large_ring = any(
            len(r) >= 8 and set(r).issubset(set(amap))
            for r in mol.GetRingInfo().AtomRings())
        if (primary_defects and not belongs_to_large_ring and
                len(local_d) >= 2 and piece.GetNumAtoms() <= 40):
            alternate = Chem.Mol(piece)
            alternate.RemoveAllConformers()
            try:
                rdDepictor.Compute2DCoords(alternate)
                ac = alternate.GetConformer()
                alt_xy = np.array([
                    [ac.GetAtomPosition(i).x, ac.GetAtomPosition(i).y]
                    for i in range(alternate.GetNumAtoms())])
                if _ligand_geometry_defects(alternate, alt_xy) < primary_defects:
                    piece.RemoveAllConformers()
                    piece.AddConformer(Chem.Conformer(ac), assignId=True)
                    xy = alt_xy
            except Exception:
                pass
        remaining_defects = _ligand_geometry_defects(piece, xy)
        if remaining_defects and not belongs_to_large_ring:
            star_xy = _articulation_star_layout(piece, xy, local_d)
            if star_xy is not None:
                xy = star_xy
                prefitted_star = True
                articulation_cavity = True
        # Atom ordering must not make two chemically identical ligands acquire
        # different shapes. Reuse the first accepted layout through a graph
        # isomorphism; besides restoring molecular symmetry this avoids subtle
        # CoordGen differences between equivalent fragments.
        try:
            template_key = Chem.MolToSmiles(piece, canonical=True)
            if template_key in ligand_templates:
                reference, reference_xy = ligand_templates[template_key]
                match = piece.GetSubstructMatch(reference, useChirality=True)
                if len(match) == piece.GetNumAtoms():
                    equivalent_match = np.asarray(match, dtype=int)
                    equivalent_xy = np.empty_like(reference_xy)
                    equivalent_xy[equivalent_match] = reference_xy
                    xy = equivalent_xy
            else:
                ligand_templates[template_key] = (Chem.Mol(piece), xy.copy())
        except Exception:
            pass
        # CoordGen normalises to a bond length of 1.0, which leaves small ligands
        # such as CO drawn much shorter than the metal-donor bonds. RDKit sizes
        # atom labels from the mean bond length, so a short bond between two
        # labelled atoms (C- and O+) ends up completely swallowed by its labels.
        if piece.GetNumBonds():
            bl = np.median([np.linalg.norm(xy[b.GetBeginAtomIdx()]
                                           - xy[b.GetEndAtomIdx()])
                            for b in piece.GetBonds()])
            if bl > 1e-6:
                xy = xy * (LB / bl)
        groups = _order_groups(piece, _hapto_groups(piece, local_d))
        if prefitted_star:
            dl = [g[0] for g in groups]
            # rigid=2 means that this ligand already defines the metal cavity:
            # it may rotate as a whole but must never be translated away.
            fitted.append([amap, xy, _angular_width(xy), len(groups), dl,
                           0.0, 180.0, 2, False])
            continue
        if len(groups) >= 4 and all(len(g) == 1 for g in groups):
            scaffold = _linear_polydentate_scaffold(
                piece, xy, [g[0] for g in groups], radius=1.2 * ml)
            if scaffold is not None:
                dl = [g[0] for g in groups]
                fitted.append([amap, scaffold, _angular_width(scaffold),
                               len(groups), dl, 0.0, 180.0, 2, False])
                articulation_cavity = True
                linear_scaffold_used = True
                continue
        if len(groups) > 1:
            xy = _chelate_conformer(piece, xy, [g[0] for g in groups])

        src, rad = [], []
        for g in groups:
            if _is_haptic_group(piece, g):         # eta-bonded -> centroid
                src.append(_hapto_centre(piece, g, xy))
                rad.append(_haptic_radius(piece, g))
            else:
                src.append(xy[g[0]])
                rad.append(ml)
        src = np.array(src, dtype=float)
        k = len(groups)

        # for three or more donors an equal-angle slot ring cannot match the
        # ligand's real donor spacing, and the least-squares fit then shoves the
        # middle donor almost onto the metal. Use the circle through the donors
        # instead, which fits exactly and keeps all metal-donor bonds equal.
        circ_c, circ_r = (None, None)
        if k >= 3 and all(len(g) < 3 for g in groups):
            circ_c, circ_r = _donor_circle(src)
            if circ_r is None or not (0.7 * ml <= circ_r <= 2.2 * ml):
                circ_c, circ_r = None, None
            elif circ_c is not None:
                donor_radii = np.linalg.norm(src - circ_c, axis=1)
                # A least-squares circle can have a plausible *mean* radius
                # while passing almost through one donor. Placing the metal at
                # that centre then prints both labels on top of each other
                # (IZAGEY/ZAZNOD). Only use a circle that actually represents
                # every donor reasonably well.
                if len(ligands) == 1 and donor_radii.min() < 0.10 * ml:
                    circ_c, circ_r = None, None
            if circ_c is not None:
                # A tripodal ligand such as tris(pyrazolyl)methane projects its
                # apex onto the centre of the donor triangle, so the metal would
                # be drawn on top of it and no reflection can help. Fall back to
                # the arc of slots, which puts the ligand to one side instead.
                rest = np.delete(xy, [i for g in groups for i in g], axis=0)
                if len(rest) and np.linalg.norm(rest - circ_c, axis=1).min() < 0.75 * ml:
                    circ_c, circ_r = None, None

        if k > 1:
            dd = np.mean([np.linalg.norm(src[j + 1] - src[j]) for j in range(k - 1)])
            half = np.degrees(np.arcsin(min(dd / (2 * ml), 0.95)))
        else:
            half = 0.0
        angs = np.linspace(-half * (k - 1), half * (k - 1), k)
        slots = np.array([[r * np.cos(np.radians(a)), r * np.sin(np.radians(a))]
                          for a, r in zip(angs, rad)])

        best = None
        for mirror in (False, True):
            s, v = src.copy(), xy.copy()
            if mirror:
                s[:, 0] = -s[:, 0]
                v[:, 0] = -v[:, 0]
            if k == 1:
                # the metal should sit in the widest gap between the donor's own
                # substituents. Aiming merely "away from the ligand centroid"
                # puts the M-donor bond straight through a substituent whenever
                # the donor carries three of them, e.g. a triarylphosphine
                d0 = groups[0][0]
                if _is_haptic_group(piece, groups[0]):
                    # An eta-bonded ring is collapsed to a centroid, and fitting
                    # one point onto one slot leaves the ring's own spin free.
                    # Left free it can point the ring's substituents back at the
                    # metal, which folds a ferrocenyl or a substituted Cp over
                    # whatever sits behind it. Spin the ring so that everything
                    # hanging off it leads away from the metal instead.
                    ring = set(groups[0])
                    body = [i for i in range(len(v)) if i not in ring]
                    cen = v[sorted(ring)].mean(0)
                    if len(groups[0]) == 2:
                        # Side-on eta-2 alkene/alkyne: the multiple bond is
                        # tangential to the coordination sphere, not radial.
                        edge = v[groups[0][1]] - v[groups[0][0]]
                        edge_angle = np.degrees(np.arctan2(edge[1], edge[0]))
                        R = _rot(90.0 - edge_angle)
                    else:
                        ref = ((v[body].mean(0) - cen) if body
                               else (cen - v[d0]))
                        if np.linalg.norm(ref) < 1e-9:
                            ref = np.array([1.0, 0.0])
                        # +x is away from the metal here, so aim the body at +x
                        R = _rot(-np.degrees(np.arctan2(ref[1], ref[0])))
                    t = slots[0] - R @ s[0]
                if not _is_haptic_group(piece, groups[0]):
                    nbr = [n.GetIdx()
                           for n in piece.GetAtomWithIdx(d0).GetNeighbors()]
                    if nbr:
                        ang = np.sort([np.degrees(np.arctan2(*(v[n] - v[d0])[::-1]))
                                       % 360.0 for n in nbr])
                        gaps = np.diff(np.append(ang, ang[0] + 360.0))
                        g0 = int(np.argmax(gaps))
                        toward = ang[g0] + gaps[g0] / 2      # bisector of the gap
                    else:
                        rest_b = np.delete(v, d0, axis=0)
                        ref = ((rest_b.mean(0) - v[d0]) if len(rest_b)
                               else np.array([-1.0, 0.0]))
                        toward = np.degrees(np.arctan2(ref[1], ref[0])) + 180.0
                    # that bisector must end up pointing back at the metal, i.e. -x
                    R = _rot(180.0 - toward)
                    t = slots[0] - R @ s[0]
            else:
                if circ_r is not None:
                    # metal at the circle centre; only the ligand's overall
                    # rotation is free, and that is set later by the angular
                    # budget, so aim the donors' mean direction along +x here
                    c0 = -circ_c.copy()
                    if mirror:
                        c0[0] = -c0[0]
                    d = (s - (-c0)).mean(0)
                    R = _rot(-np.degrees(np.arctan2(d[1], d[0])))
                    t = -R @ (-c0)
                else:
                    R, t = _kabsch2d(s, slots)
            out = v @ R.T + t
            w = _angular_width(out)
            # the canonical frame points outward along +x, so a correctly
            # oriented ligand has its centroid at positive x and keeps its
            # non-donor atoms out of the metal's neighbourhood
            r = np.linalg.norm(out, axis=1)
            body = np.ones(len(out), bool)
            body[[i for g in groups for i in g]] = False
            crowd = float(np.sum(np.clip(1.6 * ml - r[body], 0, None) ** 2))
            score = 500.0 * crowd - 8.0 * out.mean(0)[0] + w
            if best is None or score < best[0]:
                best = (score, out, w)
        dl = [g[0] if not _is_haptic_group(piece, g) else int(np.argmin(
                  np.linalg.norm(best[1][g] - best[1][g].mean(0), axis=1)))
              for g in groups]
        rigid = circ_r is not None      # metal position fixed by this ligand
        # amap, coords, half-width, n groups, donor indices, radial push, bite span
        fitted.append([amap, best[1], max(best[2], half * (k - 1) + 12.0), k,
                       dl, 0.0, half * (k - 1), rigid,
                       len(groups) == 1 and _is_haptic_group(piece, groups[0])])

    n = len(fitted)

    # ---- 3. share 360 deg in proportion to actual ligand width ------------- #
    W = np.array([f[2] for f in fitted], dtype=float)
    need = 2 * W.sum() + pad * n
    if need <= 360.0:                     # everything fits: spread the slack
        span = 2 * W + (360.0 - 2 * W.sum()) / n
        scale = 1.0
    else:                                 # too wide: squeeze proportionally
        scale = 360.0 / need
        span = (2 * W + pad) * scale
    order = np.argsort(-W)                # widest ligands placed first
    acc, centre = 90.0, {}
    for i in order:
        centre[i] = acc + span[i] / 2
        acc += span[i]

    # A metallocene is not two ligands sharing a circle, it is a sandwich: the
    # rings belong on opposite sides of the metal. Sharing the circle by width
    # puts them at whatever angle their substituents dictate, which in the worst
    # case drops the metal onto one of the ring centroids.
    sandwich = (n == 2 and all(f[3] == 1 for f in fitted)
                and all(f[8] for f in fitted))
    if sandwich:
        wide = int(np.argmax(W))
        centre = {wide: 90.0, 1 - wide: 270.0}
        # The rings are also marked rigid, which helps but does not settle it:
        # _relax frees a rigid ligand as soon as it clashes, and a substituted
        # ring always brushes its partner with its tail, so the ring still
        # drifts somewhat. Freezing it outright fixes the sandwich and wrecks
        # everything around it (crossings 2->4, overlaps 1->3 on the Fe demo),
        # so the proper fix is a relaxation that rotates a sandwich as one rigid
        # body instead of two ligands. Not done here.
        for f in fitted:
            f[7] = True
    for i, f in enumerate(fitted):
        f[1] = f[1] @ _rot(centre[i]).T
        if scale < 1.0 and f[3] > 1 and not f[7]:   # squeezed -> push out
            u = np.array([np.cos(np.radians(centre[i])),
                          np.sin(np.radians(centre[i]))])
            f[1] = f[1] + u * (1.0 - scale) * 2.5

    placed = [(f[0], f[1], f[4], f[7]) for f in fitted]
    if any(not np.isfinite(xy).all() for _, xy, _d, _r in placed):
        return _fallback()
    if relax:
        _relax(placed, ml)

    # ---- 3. write coordinates back ---------------------------------------- #
    coords = np.zeros((mol.GetNumAtoms(), 2))
    for amap, xy, _d, _r in placed:
        for local, glob in enumerate(amap):
            coords[glob] = xy[local]
    coords[mi] = (0.0, 0.0)

    x0 = max(xy[:, 0].max() for _, xy, _d, _r in placed) + 2.0
    for piece, amap in spectators:
        rdCoordGen.AddCoords(piece)
        c = piece.GetConformer()
        xy = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                       for i in range(piece.GetNumAtoms())])
        xy = xy - xy.min(0) + np.array([x0, 0.0])
        x0 = xy[:, 0].max() + 2.0
        for local, glob in enumerate(amap):
            coords[glob] = xy[local]

    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    conf.Set3D(False)
    mol.AddConformer(conf)

    pre_sector = Chem.Mol(mol)

    # The local relaxation above intentionally moves a ligand through only a
    # narrow angular window.  That is enough for ordinary coordination shells,
    # but two or three very large rigid ligands can start in the wrong sectors
    # altogether.  Make one guarded full-circle coordinate-only pass.  It is
    # never used for a single macrocycle, never changes ligand geometry or bond
    # lengths, and avoids rebuilding RDKit molecules for its candidates.
    if _depth == 0 and 2 <= len(ligands) <= 4:
        try:
            from . import cluster as _cl
            coordinate_score = _cl._coordinate_scorer(mol)
            base_xy = coords.copy()
            base_q = (coordinate_score(base_xy, strain=False),
                      coordinate_score(base_xy, strain=True))
            if base_q[0] >= 8.0:
                best_xy, best_q = base_xy, base_q
                ligand_maps = [np.asarray(amap, dtype=int)
                               for _piece, amap in ligands]
                for _ in range(2):
                    changed = False
                    for ids in ligand_maps:
                        local_xy, local_q = best_xy, best_q
                        for angle in range(-180, 181, 10):
                            if angle == 0:
                                continue
                            candidate = best_xy.copy()
                            candidate[ids] = (best_xy[ids] @ _rot(angle).T)
                            q = (coordinate_score(candidate, strain=False),
                                 coordinate_score(candidate, strain=True))
                            if q < local_q:
                                local_xy, local_q = candidate, q
                        if local_q < best_q:
                            best_xy, best_q = local_xy, local_q
                            changed = True
                    if not changed:
                        break
                if best_q[0] <= base_q[0] - 2.0:
                    conf = mol.GetConformer()
                    for i, (x, y) in enumerate(best_xy):
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
                    coords = best_xy

        except Exception:
            pass

    # Very high-denticity mononuclear ligands are sometimes already projected
    # more coherently by CoordGen as one closed object than by cutting every
    # M-donor bond and refitting the fragments.  Keep this a cheap, guarded
    # fallback: one extra candidate only for an already tangled, highly
    # coordinated outermost depiction.  A two-point improvement avoids
    # switching on noise and keeps the normal fast path unchanged.
    if (_depth == 0 and len(donors) >= 4 and
            not _has_haptic_coordination(mol) and
            not articulation_cavity):
        try:
            from . import cluster as _cl
            # Compare the whole-molecule candidate with the unpolished layout,
            # not with the independent sector candidate above.  Otherwise one
            # fallback can accidentally mask a much better one.
            current_q = _cl._tangle(_drawing_mol(mol), strain=False)
            if current_q >= 4.0:
                plain = Chem.Mol(mol)
                plain.RemoveAllConformers()
                rdCoordGen.AddCoords(plain)
                plain_q = _cl._tangle(_drawing_mol(plain), strain=False)
                if plain_q <= current_q - 2.0:
                    # Crossings alone can favour a star-shaped drawing with
                    # implausibly long M--donor bonds.  This check runs only on
                    # the already rare fallback path, so it adds no cost to
                    # ordinary mononuclear structures.
                    def _stretch(candidate):
                        drawn = _drawing_mol(candidate)
                        c = drawn.GetConformer()
                        xy = np.array([[c.GetAtomPosition(i).x,
                                        c.GetAtomPosition(i).y]
                                       for i in range(drawn.GetNumAtoms())])
                        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                                 for b in drawn.GetBonds()]
                        lengths = [np.linalg.norm(xy[i] - xy[j])
                                   for i, j in bonds]
                        unit = float(np.median(lengths))
                        metal = find_metal(drawn)
                        md = [np.linalg.norm(xy[metal] - xy[b.GetOtherAtomIdx(metal)])
                              for b in drawn.GetAtomWithIdx(metal).GetBonds()]
                        return max(md, default=0.0) / max(unit, 1e-9)

                    old_stretch = _stretch(mol)
                    new_stretch = _stretch(plain)
                    # A fixed 2.5 cap rejected candidates that improved both
                    # tangles and already-overlong coordination bonds (HAHXOF:
                    # crossings 5->1 and stretch 4.59->3.84).  Accept either a
                    # conventionally short result or a material 10% reduction
                    # from an already stretched layout.
                    dominant_macro = []
                    for piece, amap in ligands:
                        ligand_donors = [d for d in donors if d in amap]
                        if _donors_in_cyclic_core(mol, amap, ligand_donors):
                            dominant_macro.append(amap)
                    macro_candidate = len(dominant_macro) == 1
                    stretch_better = (new_stretch <= 2.5 or
                                      new_stretch <= 0.90 * old_stretch or
                                      macro_candidate)
                    if stretch_better:
                        # Large paired calixarene halves need one subsequent
                        # opposite-side placement pass; do not exit before it.
                        # For every ordinary high-denticity case retain the
                        # original immediate fast return.
                        paired_large = (len(ligands) == 2 and all(
                            len(amap) >= 20 for _piece, amap in ligands))
                        if not (paired_large or macro_candidate):
                            return plain
                        mol = plain
        except Exception:
            pass

    # A closed polydentate ligand should surround its metal even when small
    # terminal ligands (CO, halides, etc.) are also present.  Run this after the
    # guarded whole-molecule fallback so that the chosen final macrocycle
    # projection, rather than a discarded preliminary one, defines the cavity.
    if (_depth == 0 and
            sum(a.GetAtomicNum() in METALS for a in mol.GetAtoms()) == 1):
        try:
            from . import cluster as _cl
            macrocycles = []
            for _piece, amap in ligands:
                ligand_donors = [d for d in donors if d in amap]
                if _donors_in_cyclic_core(mol, amap, ligand_donors):
                    macrocycles.append((amap, ligand_donors))
            if len(macrocycles) != 1:
                raise ValueError("no unique macrocyclic donor cavity")
            _macro_map, macro_donors = macrocycles[0]
            scorer = _cl._coordinate_scorer(mol)
            conf = mol.GetConformer()
            base_xy = np.array([[conf.GetAtomPosition(i).x,
                                 conf.GetAtomPosition(i).y]
                                for i in range(mol.GetNumAtoms())])
            macro_ring = max(
                (list(ring) for ring in mol.GetRingInfo().AtomRings()
                 if len(ring) >= 8 and
                 set(ring).issubset(set(_macro_map)) and
                 set(macro_donors).issubset(set(ring))),
                key=len)
            # Literal edge crossings catch only the worst projections.  A
            # strongly compressed cavity (GEJVAS is 0.576 of the corresponding
            # regular-polygon area) is just as unreadable despite having no
            # perimeter crossing.  Keep the cutoff conservative so ordinary
            # elliptical macrocycles retain their depiction.
            base_q = (scorer(base_xy, strain=False),
                      scorer(base_xy, strain=True))
            if base_q[0] >= 3.0:
                target = base_xy[macro_donors].mean(0)
                candidate = base_xy.copy()
                candidate[mi] = target
                # The defining invariant is geometric: the metal belongs in
                # the unique macrocyclic cavity.  Crossings inside a projected
                # macrocycle may be unavoidable and must not veto its centre.
                # Small remote ligands are repaired by the following radial
                # pass.
                for i, (x, y) in enumerate(candidate):
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
                coords = candidate
        except Exception:
            pass

    # Relaxation may solve a crowded mixed-denticity sphere by pushing one
    # small chelate implausibly far away (FOJSOO put its O,N ligand at radius
    # 6.7 while the other donors sat near 3).  Pull only clear mono/bidentate
    # radial outliers back toward the median shell.  Four translations and a
    # coordinate-only score keep this guarded pass inexpensive.
    if _depth == 0 and len(ligands) >= 2:
        try:
            from . import cluster as _cl
            conf = mol.GetConformer()
            current_xy = np.array([[conf.GetAtomPosition(i).x,
                                    conf.GetAtomPosition(i).y]
                                   for i in range(mol.GetNumAtoms())])
            scorer = _cl._coordinate_scorer(mol)
            current_q = (scorer(current_xy, strain=False),
                         scorer(current_xy, strain=True))
            for ligand_index, (_piece, amap) in enumerate(ligands):
                local_d = [d for d in donors if d in amap]
                if not (1 <= len(local_d) <= 2) or fitted[ligand_index][8]:
                    continue
                other_d = [d for d in donors if d not in local_d]
                if not other_d:
                    continue
                origin = current_xy[mi]
                donor_centre = current_xy[local_d].mean(0)
                vector = donor_centre - origin
                radius = np.linalg.norm(vector)
                other_radius = float(np.median(
                    np.linalg.norm(current_xy[other_d] - origin, axis=1)))
                if (radius <= 3.0 * ml or
                        radius <= 1.5 * max(other_radius, ml)):
                    continue
                # A one-donor two-atom ligand such as CO should follow a metal
                # that has just been centred in a macrocycle; keeping the old
                # generic 2*ML floor leaves a conspicuously long spoke.  Larger
                # mono/bidentate bodies retain the more conservative clearance.
                floor = ml if len(amap) <= 2 else 2.0 * ml
                target = max(floor, other_radius)
                ids = np.asarray(amap, dtype=int)
                best_xy, best_q = current_xy, current_q
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    wanted = radius + fraction * (target - radius)
                    candidate = current_xy.copy()
                    candidate[ids] += vector / radius * (wanted - radius)
                    q = (scorer(candidate, strain=False),
                         scorer(candidate, strain=True))
                    if q < best_q:
                        best_xy, best_q = candidate, q
                if (best_q[0] <= current_q[0] and
                        best_q[1] <= 0.90 * current_q[1]):
                    current_xy, current_q = best_xy, best_q
            if not np.array_equal(current_xy, np.array([
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                    for i in range(mol.GetNumAtoms())])):
                for i, (x, y) in enumerate(current_xy):
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
                coords = current_xy
        except Exception:
            pass

    # A label can still sit directly on the metal even when the bond-crossing
    # score is zero (XOJKAL is the small representative).  This is cheap to
    # detect and rare.  Try one whole-molecule CoordGen layout and keep it only
    # when it removes such atoms without worsening crossings or stretching the
    # coordination sphere.  Ordinary clean molecules pay only the distance
    # check, not an extra depiction.
    has_macrocyclic_cavity = any(
        _donors_in_cyclic_core(
            mol, amap, [d for d in donors if d in amap])
        for _piece, amap in ligands)
    if (_depth == 0 and len(donors) >= 1 and
            not has_macrocyclic_cavity and not articulation_cavity):
        try:
            from . import cluster as _cl

            def _near_metal(candidate):
                drawn = _drawing_mol(candidate)
                c = drawn.GetConformer()
                xy = np.array([[c.GetAtomPosition(i).x,
                                c.GetAtomPosition(i).y]
                               for i in range(drawn.GetNumAtoms())])
                bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                         for b in drawn.GetBonds()]
                unit = float(np.median([
                    np.linalg.norm(xy[i] - xy[j]) for i, j in bonds]))
                metal = find_metal(drawn)
                bonded = {b.GetOtherAtomIdx(metal)
                          for b in drawn.GetAtomWithIdx(metal).GetBonds()}
                return sum(i != metal and i not in bonded and
                           drawn.GetAtomWithIdx(i).GetAtomicNum() != 0 and
                           np.linalg.norm(xy[i] - xy[metal]) < 0.7 * unit
                           for i in range(drawn.GetNumAtoms()))

            old_near = _near_metal(mol)
            if old_near:
                plain = Chem.Mol(mol)
                plain.RemoveAllConformers()
                rdCoordGen.AddCoords(plain)
                new_near = _near_metal(plain)
                old_q = _cl._tangle(_drawing_mol(mol), strain=False)
                new_q = _cl._tangle(_drawing_mol(plain), strain=False)
                old_s = _cl._tangle(_drawing_mol(mol), strain=True)
                new_s = _cl._tangle(_drawing_mol(plain), strain=True)
                if (new_near < old_near and new_q <= old_q and
                        new_s <= max(old_s * 1.15, old_s + 1.0)):
                    return plain
        except Exception:
            pass

    # UPOJEP-like bis(calix[3]arene) complexes are a special projection
    # problem.  Each half is a chain of three phenyl rings, but depicting that
    # chain freely gives a zig-zag; fitting two zig-zags around Zr then piles all
    # six rings into the centre.  Recognise the graph very strictly and build
    # the conventional pair of opposed arcs.  Ring orientation is only a tiny
    # discrete search (12^3 states per half) and this branch is exceptionally
    # rare, so it does not affect normal throughput.
    if (_depth == 0 and len(ligands) == 2 and
            all(len(amap) >= 20 for _piece, amap in ligands)):
        try:
            ligand_maps = [np.asarray(amap, dtype=int)
                           for _piece, amap in ligands]
            # Include every explicit metal contact here.  Aromatic carbon
            # contacts in calixarene SMILES are deliberately not part of the
            # ordinary donor list used by the sector fitter, but they are part
            # of this tridentate topology.
            metal_neighbours = {
                b.GetOtherAtomIdx(mi)
                for b in mol.GetAtomWithIdx(mi).GetBonds()
            }
            ligand_donors = [[d for d in metal_neighbours if d in amap]
                             for amap in ligand_maps]
            rings = [list(r) for r in mol.GetRingInfo().AtomRings()
                     if len(r) == 6 and all(
                         mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
            metal_symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol()
                                   for i in metal_neighbours)
            half_rings = [[r for r in rings if set(r).issubset(set(amap))]
                          for amap in ligand_maps]
            calix_graph = (mol.GetAtomWithIdx(mi).GetSymbol() == "Zr" and
                           metal_symbols == ["C", "C", "O", "O", "O", "O"] and
                           all(len(ds) == 3 for ds in ligand_donors) and
                           all(len(rs) == 3 for rs in half_rings))
            if calix_graph:
                conf = mol.GetConformer()
                best_xy = np.array([[conf.GetAtomPosition(i).x,
                                     conf.GetAtomPosition(i).y]
                                    for i in range(mol.GetNumAtoms())])
                radius = LB

                def ring_xy(ring, state, centre_xy):
                    reverse = -1 if state >= 6 else 1
                    phase = state % 6
                    return {
                        atom: np.asarray(centre_xy) + radius * np.array([
                            np.cos(np.radians(60 * (reverse * j + phase) + 30)),
                            np.sin(np.radians(60 * (reverse * j + phase) + 30))])
                        for j, atom in enumerate(ring)
                    }

                for half, (amap, rs) in enumerate(zip(ligand_maps, half_rings)):
                    # The middle ring is the one linked to both other rings.
                    links = []
                    for bond in mol.GetBonds():
                        if bond.IsInRing():
                            continue
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        ri = next((k for k, r in enumerate(rs) if i in r), None)
                        rj = next((k for k, r in enumerate(rs) if j in r), None)
                        if ri is not None and rj is not None and ri != rj:
                            links.append((i, j, ri, rj))
                    degree = [0, 0, 0]
                    for _i, _j, ri, rj in links:
                        degree[ri] += 1
                        degree[rj] += 1
                    central_i = degree.index(2)
                    terminal_i = [k for k in range(3) if k != central_i]
                    central = rs[central_i]
                    terminals = [rs[k] for k in terminal_i]
                    oriented_links = []
                    for bond in mol.GetBonds():
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        for k, terminal in enumerate(terminals):
                            if i in central and j in terminal:
                                oriented_links.append((i, j, k))
                            elif j in central and i in terminal:
                                oriented_links.append((j, i, k))
                    if len(oriented_links) != 2:
                        raise ValueError("unexpected calixarene ring graph")

                    vertical = 1.0 if half == 0 else -1.0
                    centre = np.array([0.0, vertical * (8.0 / 3.0) * ml])
                    oxygen_anchors = []
                    for terminal in terminals:
                        oxygen = next(
                            n.GetIdx() for atom in terminal
                            for n in mol.GetAtomWithIdx(atom).GetNeighbors()
                            if n.GetSymbol() == "O")
                        anchor = next(
                            n.GetIdx() for n in
                            mol.GetAtomWithIdx(oxygen).GetNeighbors()
                            if n.GetIdx() in terminal)
                        oxygen_anchors.append(anchor)
                    chosen = None
                    # Once the central ring and left/right assignment are
                    # fixed, the two terminal phases are independent.  Pick
                    # each in a 12-state local search instead of evaluating
                    # all 12^3 combinations together (288 trials vs 3456).
                    for pc in range(12):
                        for sides in ((-1.0, 1.0), (1.0, -1.0)):
                            trial = ring_xy(central, pc, centre)
                            terminal_centres = [None, None]
                            oxygen_xy, value = [None, None], 0.0
                            for (ca, ta, k), side in zip(oriented_links, sides):
                                local_best = None
                                for phase in range(12):
                                    local = ring_xy(terminals[k], phase,
                                                    (0.0, 0.0))
                                    direction = np.array([side * 0.707,
                                                          -vertical * 0.707])
                                    tc = (trial[ca] + radius * direction -
                                          local[ta])
                                    placed = ring_xy(terminals[k], phase, tc)
                                    anchor = oxygen_anchors[k]
                                    op = placed[anchor] + (placed[anchor] - tc)
                                    local_value = (
                                        5.0 * (np.linalg.norm(op) - ml) ** 2 +
                                        10.0 * (tc[1] - vertical *
                                                (4.0 / 3.0) * ml) ** 2 +
                                        10.0 * (tc[0] - side *
                                                (28.0 / 15.0) * ml) ** 2)
                                    if (local_best is None or
                                            local_value < local_best[0]):
                                        local_best = (local_value, placed, tc, op)
                                lv, placed, tc, op = local_best
                                value += lv
                                trial.update(placed)
                                terminal_centres[k] = tc
                                oxygen_xy[k] = op
                            value += 3.0 * (oxygen_xy[0][1] -
                                            oxygen_xy[1][1]) ** 2
                            value += (oxygen_xy[0][0] + oxygen_xy[1][0]) ** 2
                            if chosen is None or value < chosen[0]:
                                chosen = (value, trial, terminal_centres,
                                          terminals, central)

                    _value, trial, terminal_centres, terminals, central = chosen
                    for atom, point in trial.items():
                        best_xy[atom] = point
                    # O atoms and complete tert-butyl groups are laid out
                    # radially from their phenyl ring; nothing is abbreviated.
                    ordered_rings = terminals + [central]
                    centres = terminal_centres + [centre]
                    all_ring_atoms = set().union(*map(set, rs))
                    for ring, rc in zip(ordered_rings, centres):
                        for atom in ring:
                            for neighbour in mol.GetAtomWithIdx(atom).GetNeighbors():
                                j = neighbour.GetIdx()
                                if (j in all_ring_atoms or
                                        neighbour.GetSymbol() == "Zr"):
                                    continue
                                vector = best_xy[atom] - rc
                                vector /= max(np.linalg.norm(vector), 1e-9)
                                best_xy[j] = best_xy[atom] + radius * vector
                                if mol.GetAtomWithIdx(j).GetDegree() == 4:
                                    theta = np.arctan2(vector[1], vector[0])
                                    children = [n.GetIdx() for n in
                                                mol.GetAtomWithIdx(j).GetNeighbors()
                                                if n.GetIdx() != atom]
                                    for child, delta in zip(children,
                                                            (-75.0, 0.0, 75.0)):
                                        a = theta + np.radians(delta)
                                        best_xy[child] = best_xy[j] + radius * np.array([
                                            np.cos(a), np.sin(a)])
                best_xy[mi] = (0.0, 0.0)
                for i, (x, y) in enumerate(best_xy):
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
                coords = best_xy
        except Exception:
            pass

    # Make opening a collapsed macrocycle the final structural operation.
    # Earlier optimisers are useful for arranging substituents and small
    # ligands, but if this transform runs before them they can fold the cavity
    # closed again.  Detection and reconstruction remain purely topological /
    # geometric and therefore apply to any metal and macrocycle class.
    if (_depth == 0 and
            sum(a.GetAtomicNum() in METALS for a in mol.GetAtoms()) == 1):
        try:
            candidates = []
            for _piece, amap in ligands:
                local_donors = [d for d in donors if d in amap]
                if not _donors_in_cyclic_core(mol, amap, local_donors):
                    continue
                rings = [list(ring) for ring in mol.GetRingInfo().AtomRings()
                         if len(ring) >= 8 and
                         set(ring).issubset(set(amap)) and
                         set(local_donors).issubset(set(ring))]
                if rings:
                    candidates.append(max(rings, key=len))
            if len(candidates) == 1:
                ring = candidates[0]
                conf = mol.GetConformer()
                final_xy = np.array([
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                    for i in range(mol.GetNumAtoms())])
                collapsed = (
                    _macrocycle_self_crossings(final_xy, ring) > 0 or
                    _macrocycle_area_ratio(final_xy, ring) < 0.62)
                if collapsed:
                    final_xy = _unfold_macrocycle_coords(
                        mol, final_xy, ring, mi)
                    for i, (x, y) in enumerate(final_xy):
                        conf.SetAtomPosition(
                            i, Point3D(float(x), float(y), 0.0))
                    coords = final_xy
        except Exception:
            pass

    # A rigid, non-circular polydentate ligand can leave one donor virtually
    # coincident with the least-squares metal position even after circle
    # rejection.  Moving the whole ligand would destroy its readable shape;
    # move only the metal by a small discrete amount.  The search is entered
    # only for an already invalid sub-0.65*ML contact.
    if _depth == 0:
        try:
            conf = mol.GetConformer()
            final_xy = np.array([
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                for i in range(mol.GetNumAtoms())])
            bonded = [b.GetOtherAtomIdx(mi)
                      for b in mol.GetAtomWithIdx(mi).GetBonds()]
            radii = np.linalg.norm(final_xy[bonded] - final_xy[mi], axis=1)
            if len(radii) and radii.min() < 0.25 * ml:
                nonbonded = [i for i in range(mol.GetNumAtoms())
                             if i != mi and i not in bonded]
                origin = final_xy[mi].copy()
                best = None
                for radius in (0.25, 0.5, 0.75, 1.0):
                    for angle in range(0, 360, 30):
                        point = origin + radius * ml * np.array([
                            np.cos(np.radians(angle)),
                            np.sin(np.radians(angle))])
                        donor_r = np.linalg.norm(final_xy[bonded] - point,
                                                 axis=1)
                        other_r = np.linalg.norm(final_xy[nonbonded] - point,
                                                 axis=1)
                        value = (
                            1000.0 * np.sum(np.clip(
                                0.75 * ml - donor_r, 0.0, None) ** 2) +
                            1000.0 * np.sum(np.clip(
                                0.65 * LB - other_r, 0.0, None) ** 2) +
                            np.sum((donor_r - ml) ** 2) +
                            0.1 * np.sum((point - origin) ** 2))
                        if best is None or value < best[0]:
                            best = value, point
                if best is not None:
                    final_xy[mi] = best[1]
                    conf.SetAtomPosition(
                        mi, Point3D(float(best[1][0]), float(best[1][1]), 0.0))
                    coords = final_xy
        except Exception:
            pass
    result = _equalize_atomic_ligand_radii(
        mol, mi, [amap for _piece, amap in ligands])
    if linear_scaffold_used:
        conf = result.GetConformer()
        for i in range(result.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(-p.x, -p.y, p.z))
    return result


def _occupancy(xy, donors_local):
    """Atoms of a ligand plus sample points along its metal-donor bonds, so that
    relaxation keeps other ligands off the bonds too, not just off the atoms."""
    if not donors_local:
        return xy
    seg = np.vstack([xy[donors_local] * f for f in (0.3, 0.55, 0.8)])
    return np.vstack([xy, seg])


def _relax(placed, ml=ML, sweeps=4, span=30.0, step=3.0, push=0.6):
    """Rotate, and if needed push out, whole ligands to remove clashes."""
    if len(placed) < 2:
        return
    cut = 0.95 * ml
    angles = np.arange(-span, span + 1e-9, step)
    for _ in range(sweeps):
        for i, (amap, xy, dl, rigid) in enumerate(placed):
            others = [_occupancy(p[1], p[2]) for j, p in enumerate(placed) if j != i]
            u = xy.mean(0)
            u = u / max(np.linalg.norm(u), 1e-6)
            best, bxy = None, xy
            for a in angles:
                cand0 = xy @ _rot(a).T
                # a ligand that defines the metal position must not be shifted:
                # translating it destroys the equal metal-donor distances
                # A circle-fitted ligand keeps its metal-donor bonds equal only
                # while it is not translated, so it is normally rotated in place.
                # But two such ligands can only rotate into each other; when a
                # clash survives, unequal bonds beat overlapping drawings.
                stuck = (rigid == 1 and
                         sum(_clashes(_occupancy(xy, dl), o, cut)
                             for o in others) > 0)
                if rigid == 2 or (rigid and not stuck):
                    steps = (0.0,)
                else:
                    # the push is applied once per sweep, so without a ceiling a
                    # ligand keeps drifting and ends up on absurdly long bonds
                    near = min(np.linalg.norm(xy[d]) for d in dl) if dl else 0.0
                    room = max(0.0, MAX_REACH * ml - near)
                    steps = (0.0,) + tuple(p for p in (push, 2 * push) if p <= room)
                for d in steps:
                    cand = cand0 + u * d
                    # the metal is an obstacle too: nothing may be drawn on top
                    # of it, including a neighbouring ligand's tail sweeping past
                    s = (sum(_clashes(_occupancy(cand, dl), o, cut) for o in others)
                         + 3.0 * _clashes(cand, np.zeros((1, 2)), 1.35 * ml)
                         + 0.0015 * a * a + 0.5 * d)
                    if best is None or s < best:
                        best, bxy = s, cand
            placed[i] = (amap, bxy, dl, rigid)


# --------------------------------------------------------------------------- #
#  drawing
# --------------------------------------------------------------------------- #
def _drawing_mol(mol):
    """Copy prepared for drawing: plain lines to the metal instead of dative
    arrows, and one single bond to a ring centroid for each eta-bonded group."""
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS]
    if not metals:
        return Chem.Mol(mol)
    calix_aromatic_six = sum(
        len(r) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        for r in mol.GetRingInfo().AtomRings())
    base = Chem.Mol(mol)
    # kekulize FIRST, while the metal bonds are still dative and therefore do not
    # count towards valence. Converting them to single bonds first makes pyridine
    # nitrogens unkekulizable, and RDKit then falls back to dashed aromatic lines
    try:
        Chem.Kekulize(base, clearAromaticFlags=True)
    except Exception:
        pass

    rw = Chem.RWMol(base)
    # CCDC's conventional 2-D representation of the UPOJEP
    # bis(calix[3]arene) motif shows the four phenoxide Zr--O bonds and omits
    # the two auxiliary aryl-C contacts.  Keep those contacts in the molecular
    # graph and coordinates, but suppress them only in the drawing copy.  The
    # signature is intentionally exact so ordinary organozirconium bonds are
    # never hidden.
    if rw.GetNumAtoms() >= 50 and calix_aromatic_six == 6:
        for mi in metals:
            atom = rw.GetAtomWithIdx(mi)
            neighbours = list(atom.GetNeighbors())
            symbols = sorted(n.GetSymbol() for n in neighbours)
            if atom.GetSymbol() == "Zr" and symbols == [
                    "C", "C", "O", "O", "O", "O"]:
                for neighbour in neighbours:
                    if neighbour.GetSymbol() == "C" and rw.GetBondBetweenAtoms(
                            mi, neighbour.GetIdx()) is not None:
                        rw.RemoveBond(mi, neighbour.GetIdx())
    # every metal centre needs the same treatment, not just the first one:
    # a polynuclear complex otherwise gets centroids and plain bonds at one
    # centre and raw dative arrows through the ring at all the others
    for mi in metals:
      donors = [b.GetOtherAtomIdx(mi) for b in rw.GetAtomWithIdx(mi).GetBonds()]
      hapto = [g for g in _hapto_groups(rw, donors)
               if _is_haptic_group(rw, g)]

      for grp in hapto:
          conf = rw.GetConformer()
          xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                         for i in range(rw.GetNumAtoms())])
          gset = set(grp)
          internal_edges = sum(
              1 for b in rw.GetBonds()
              if b.GetBeginAtomIdx() in gset and b.GetEndAtomIdx() in gset)
          open_haptic = internal_edges < len(grp)
          dative_types = (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE,
                          Chem.BondType.ZERO)
          dative_grp = [i for i in grp
                        if rw.GetBondBetweenAtoms(mi, i) is not None and
                        rw.GetBondBetweenAtoms(mi, i).GetBondType() in
                        dative_types]
          # For an open mixed sigma/eta system the centroid represents only the
          # delocalised dative contacts.  One dative contact is already clearest
          # as its original direct bond and needs no artificial centroid.
          if open_haptic and len(dative_grp) < 2:
              continue
          folded = dative_grp if open_haptic else list(grp)
          P = (xy[folded].mean(axis=0) if open_haptic
               else _hapto_centre(rw, grp, xy))
          for i in folded:
              if rw.GetBondBetweenAtoms(mi, i) is not None:
                  rw.RemoveBond(mi, i)
          dummy = Chem.Atom(0)
          dummy.SetNoImplicit(True)
          di = rw.AddAtom(dummy)
          rw.GetAtomWithIdx(di).SetProp("atomLabel", "")
        # record which atoms this centroid stands for. The bond from the metal to
        # a ring centre has to cross that ring's perimeter to get there, so any
        # tool measuring bond crossings needs to know to forgive that one.
          rw.GetAtomWithIdx(di).SetProp("_hapticAtoms",
                                        ",".join(map(str, folded)))
          rw.AddBond(mi, di, Chem.BondType.SINGLE)
          rw.GetConformer().SetAtomPosition(di, Point3D(float(P[0]), float(P[1]), 0.0))

      for b in rw.GetAtomWithIdx(mi).GetBonds():
          if b.GetBondType() in (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE,
                                 Chem.BondType.ZERO):
              b.SetBondType(Chem.BondType.SINGLE)
    # only the metal needs its implicit hydrogens suppressed. Doing this for every
    # atom also wipes explicit ones, which silently removes the H from [nH], -OH,
    # secondary amines and so on
      rw.GetAtomWithIdx(mi).SetNoImplicit(True)

    # T-REX octahedral perspective is stored on chemically correct donor->metal
    # dative bonds.  Conventionally the narrow end of a coordination wedge is
    # at the metal, but BEGINWEDGE tapers at the *end* for that bond ordering.
    # Reverse only the disposable single bonds in this drawing copy; the Mol
    # returned by the parser keeps donor->metal semantics unchanged.
    reverse = []
    for b in rw.GetBonds():
        if not b.HasProp("_TREX_wedgeFromMetal"):
            continue
        a, z = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        mi = a if rw.GetAtomWithIdx(a).GetAtomicNum() in METALS else z
        donor = z if mi == a else a
        reverse.append((mi, donor, b.GetBondDir()))
    for mi, donor, direction in reverse:
        rw.RemoveBond(mi, donor)
        rw.AddBond(mi, donor, Chem.BondType.SINGLE)
        b = rw.GetBondBetweenAtoms(mi, donor)
        b.SetBondDir(direction)
        b.SetBoolProp("_TREX_wedgeFromMetal", True)
    return rw.GetMol()


def style_options(o):
    """Apply metal2d's drawing style to a MolDraw2D options object. Exposed so
    that other tools can render metal2d coordinates and get the same picture."""
    o.addStereoAnnotation = False
    o.legendFontSize = 22
    o.bondLineWidth = 2
    o.scaleBondWidth = False
    o.additionalAtomLabelPadding = 0.04
    o.updateAtomPalette({6: (0.1, 0.1, 0.1), 7: (0.15, 0.15, 0.9),
                         8: (0.9, 0.1, 0.1), 16: (0.8, 0.7, 0.0),
                         17: (0.1, 0.8, 0.1), 0: (0.1, 0.1, 0.1)})
    for z in METALS:
        o.updateAtomPalette({z: METAL_COLOR})
    return o


def draw(mol, path, size=(800, 800), title="", style=True):
    out = _drawing_mol(mol) if style else Chem.Mol(mol)
    png = str(path).lower().endswith(".png")
    d = (rdMolDraw2D.MolDraw2DCairo if png else rdMolDraw2D.MolDraw2DSVG)(*size)
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.legendFontSize = 22
    if style:
        style_options(o)
    try:
        # already kekulized in _drawing_mol; re-doing it here would fail and fall
        # back to dashed aromatic bonds
        m = rdMolDraw2D.PrepareMolForDrawing(out, kekulize=not style,
                                             wedgeBonds=False)
    except Exception:
        m = out
    d.DrawMolecule(m, legend=title)
    d.FinishDrawing()
    txt = d.GetDrawingText()
    open(path, "wb" if png else "w").write(txt)


def mol_from_input(value, sanitize=True):
    """Read one SMILES or T-REX string into an RDKit molecule.

    T-REX is imported lazily so the ordinary SMILES path has no extra import or
    runtime cost.  Geometry-aware depiction should use :func:`depict_input`;
    an RDKit Mol by itself cannot retain cis/trans relations between otherwise
    identical monodentate ligands.
    """
    from .trex import is_trex, mol_from_trex
    if is_trex(value):
        return mol_from_trex(value).mol
    return Chem.MolFromSmiles(str(value), sanitize=sanitize)


def depict_input(value, **kwargs):
    """Unified textual entry point accepting either SMILES or T-REX."""
    from .trex import depict_trex, is_trex
    if is_trex(value):
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError("T-REX depiction does not accept: %s" % unsupported)
        return depict_trex(value)
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError("input is neither valid T-REX nor valid SMILES")
    return depict(mol, **kwargs)


def read_molecules(src, sanitize=True, column=None):
    """Yield (name, Mol) pairs from an .sdf/.mol file, a .smi/.txt/.csv list of
    SMILES, or a single SMILES string given directly.

    Bad entries are yielded as (name, None) rather than skipped silently, so the
    caller can see and report them - MolFromSmiles returns None without raising.
    """
    import os

    ext = os.path.splitext(str(src))[1].lower()

    if ext in (".sdf", ".mol"):
        supp = Chem.SDMolSupplier(src, removeHs=False, sanitize=sanitize)
        for i, m in enumerate(supp):
            name = ""
            if m is not None and m.HasProp("_Name"):
                name = m.GetProp("_Name")
            yield name or "mol_%d" % i, m
        return

    if ext == ".trex":
        with open(src) as fh:
            i = 0
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    yield "mol_%d" % i, mol_from_input(line, sanitize=sanitize)
                except (ValueError, NotImplementedError):
                    yield "mol_%d" % i, None
                i += 1
        return

    if ext in (".csv", ".tsv"):
        import csv
        with open(src, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
            fields = reader.fieldnames or []
            smiles_fields = [c for c in fields if "smiles" in c.lower()]
            if column:
                if column not in fields:
                    raise ValueError("CSV has no column %r" % column)
                smiles_column = column
            elif smiles_fields:
                smiles_column = next(
                    (c for c in smiles_fields if "complex" in c.lower()),
                    next((c for c in smiles_fields if c.lower() == "smiles"),
                         smiles_fields[0]))
            else:
                raise ValueError("no column with 'smiles' in its name found")
            name_column = next((c for c in fields
                                if c.lower() in ("name", "id", "title")), None)
            for i, row in enumerate(reader):
                value = (row.get(smiles_column) or "").strip()
                if not value:
                    continue
                name = (row.get(name_column) or "mol_%d" % i)
                try:
                    yield name, mol_from_input(value, sanitize=sanitize)
                except (ValueError, NotImplementedError):
                    yield name, None
        return

    if ext in (".smi", ".smiles", ".txt"):
        with open(src) as fh:
            i = 0
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # first whitespace/comma/tab separated token is the SMILES,
                # anything after it is treated as the name
                parts = line.replace("\t", " ").replace(",", " ").split()
                smi = parts[0]
                name = " ".join(parts[1:]) or "mol_%d" % i
                if i == 0 and smi.lower() in ("smiles", "smi"):
                    continue                      # header line
                m = Chem.MolFromSmiles(smi, sanitize=sanitize)
                yield name, m
                i += 1
        return

    # not a path: treat the argument itself as a SMILES string
    yield "mol_0", mol_from_input(str(src), sanitize=sanitize)
