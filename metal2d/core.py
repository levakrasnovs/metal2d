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
from rdkit.Chem import rdCoordGen
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


def _hapto_centre(piece, group, xy):
    """Where the single bond should point: the ring centroid if the group
    contains a ring, otherwise the centroid of the whole group."""
    ring = [i for i in group if piece.GetAtomWithIdx(i).IsInRing()]
    return xy[ring if ring else group].mean(0)


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
def depict(mol, ml=ML, relax=True, pad=6.0, _depth=0, cluster=True):
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
                    return out
        except ImportError:
            pass
        except Exception:
            pass

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
    for piece, amap in ligands:
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
        local_d = [amap.index(d) for d in donors if d in amap]
        groups = _order_groups(piece, _hapto_groups(piece, local_d))
        if len(groups) > 1:
            xy = _chelate_conformer(piece, xy, [g[0] for g in groups])

        src, rad = [], []
        for g in groups:
            if len(g) >= 3:                       # eta-bonded -> centroid
                src.append(_hapto_centre(piece, g, xy))
                rad.append(HAPTO_R)
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
                if len(groups[0]) >= 3:
                    # An eta-bonded ring is collapsed to a centroid, and fitting
                    # one point onto one slot leaves the ring's own spin free.
                    # Left free it can point the ring's substituents back at the
                    # metal, which folds a ferrocenyl or a substituted Cp over
                    # whatever sits behind it. Spin the ring so that everything
                    # hanging off it leads away from the metal instead.
                    ring = set(groups[0])
                    body = [i for i in range(len(v)) if i not in ring]
                    cen = v[sorted(ring)].mean(0)
                    ref = (v[body].mean(0) - cen) if body else (cen - v[d0])
                    if np.linalg.norm(ref) < 1e-9:
                        ref = np.array([1.0, 0.0])
                    # +x is away from the metal here, so aim the body at +x
                    R = _rot(-np.degrees(np.arctan2(ref[1], ref[0])))
                    t = slots[0] - R @ s[0]
                if len(groups[0]) < 3:
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
        dl = [g[0] if len(g) < 3 else int(np.argmin(
                  np.linalg.norm(best[1][g] - best[1][g].mean(0), axis=1)))
              for g in groups]
        rigid = circ_r is not None      # metal position fixed by this ligand
        # amap, coords, half-width, n groups, donor indices, radial push, bite span
        fitted.append([amap, best[1], max(best[2], half * (k - 1) + 12.0), k,
                       dl, 0.0, half * (k - 1), rigid,
                       len(groups) == 1 and len(groups[0]) >= 3])

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
    return mol


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
                stuck = rigid and sum(_clashes(_occupancy(xy, dl), o, cut)
                                      for o in others) > 0
                if rigid and not stuck:
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
    base = Chem.Mol(mol)
    # kekulize FIRST, while the metal bonds are still dative and therefore do not
    # count towards valence. Converting them to single bonds first makes pyridine
    # nitrogens unkekulizable, and RDKit then falls back to dashed aromatic lines
    try:
        Chem.Kekulize(base, clearAromaticFlags=True)
    except Exception:
        pass

    rw = Chem.RWMol(base)
    # every metal centre needs the same treatment, not just the first one:
    # a polynuclear complex otherwise gets centroids and plain bonds at one
    # centre and raw dative arrows through the ring at all the others
    for mi in metals:
      donors = [b.GetOtherAtomIdx(mi) for b in rw.GetAtomWithIdx(mi).GetBonds()]
      hapto = [g for g in _hapto_groups(rw, donors) if len(g) >= 3]

      for grp in hapto:
          conf = rw.GetConformer()
          xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                         for i in range(rw.GetNumAtoms())])
          P = _hapto_centre(rw, grp, xy)
          for i in grp:
              if rw.GetBondBetweenAtoms(mi, i) is not None:
                  rw.RemoveBond(mi, i)
          dummy = Chem.Atom(0)
          dummy.SetNoImplicit(True)
          di = rw.AddAtom(dummy)
          rw.GetAtomWithIdx(di).SetProp("atomLabel", "")
        # record which atoms this centroid stands for. The bond from the metal to
        # a ring centre has to cross that ring's perimeter to get there, so any
        # tool measuring bond crossings needs to know to forgive that one.
          rw.GetAtomWithIdx(di).SetProp("_hapticAtoms", ",".join(map(str, grp)))
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


def read_molecules(src, sanitize=True):
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

    if ext in (".smi", ".smiles", ".txt", ".csv", ".tsv"):
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
    yield "mol_0", Chem.MolFromSmiles(str(src), sanitize=sanitize)
