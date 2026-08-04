"""Layout for complexes with metal-metal bonds.

The model in core.py is one centre with its donors on a circle around it. A
metal-metal bonded cluster breaks that in two ways: a bridging ligand belongs
to two centres at once, and a metal is itself a donor to another metal. Cutting
the metal's bonds then severs the three-membered M-M-bridge rings, and no
amount of refitting puts them back.

So the core is built first, as a rigid template: the metals at fixed positions,
every donor site placed around them, bridges on the perpendicular bisector of
the M-M axis. Each ligand is then laid out on its own and fitted onto the donor
positions the template assigns it. A bridging ligand gets one target per metal,
so the least-squares fit lands it between the two centres without being told
that is what should happen.

Everything else is delegated to core.depict, including any metal that sits
inside a ligand (a pendant ferrocene, say) rather than in the cluster.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Geometry import Point3D

from . import core
from .core import METALS, ML, LB, HAPTO_R, _rot, _kabsch2d, _hapto_groups, _hapto_centre

# Metal-metal bonds are drawn at the same length as metal-donor bonds; making
# them longer just pushes the two halves apart and wastes the page.
D_MM = ML

# Keep terminal ligands this far, in degrees, from the M-M axis and from the
# bridges. Below about 25 the terminal bonds start to graze the bridge bonds.
CLEAR = 30.0


def metal_clusters(mol):
    """Groups of metals joined by metal-metal bonds. Singletons included."""
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS]
    parent = {i: i for i in metals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in parent and j in parent:
            parent[find(i)] = find(j)

    out = {}
    for i in metals:
        out.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in out.values()), key=len, reverse=True)


def has_cluster(mol):
    """True when two or more metals are bonded to each other."""
    c = metal_clusters(mol)
    return bool(c) and len(c[0]) > 1


def _free_arcs(occupied, clear=CLEAR):
    """Arcs around a metal that no bridge or M-M bond already uses."""
    occ = sorted(a % 360.0 for a in occupied)
    if not occ:
        return [(360.0, 0.0, 360.0)]
    arcs = []
    for i, a in enumerate(occ):
        b = occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0)
        lo, hi = a + clear, b - clear
        if hi > lo:
            arcs.append((hi - lo, lo, hi))
    if not arcs:
        arcs = [(occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0) - a,
                 a, occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0))
                for i, a in enumerate(occ)]
    return sorted(arcs, reverse=True)


def _share(occupied, weights, clear=CLEAR):
    """One direction per terminal ligand, widest ligand into the widest arc.

    Equal spacing is wrong here: a pendant ferrocene and a carbonyl are not
    interchangeable, and giving them the same wedge is what makes the big one
    collide with its neighbour.
    """
    k = len(weights)
    if k == 0:
        return []
    arcs = _free_arcs(occupied, clear)

    # deal ligands to arcs so that each arc's width per unit of demand is even
    load = [0.0] * len(arcs)
    hold = [[] for _ in arcs]
    for idx in np.argsort(-np.asarray(weights, dtype=float)):
        j = int(np.argmax([arcs[i][0] / (load[i] + weights[idx]) for i in range(len(arcs))]))
        load[j] += weights[idx]
        hold[j].append(int(idx))

    out = [0.0] * k
    for (w, lo, hi), members in zip(arcs, hold):
        if not members:
            continue
        tot = sum(weights[i] for i in members) or 1.0
        acc = lo
        for i in members:
            span = (hi - lo) * weights[i] / tot
            out[i] = acc + span / 2.0
            acc += span
    return out


def _fan(occupied, k, clear=CLEAR):
    """k directions spread through the widest free arcs, avoiding `occupied`."""
    if k == 0:
        return []
    if not occupied:
        return list(np.linspace(0.0, 360.0, k, endpoint=False))

    occ = sorted(a % 360.0 for a in occupied)
    gaps = []
    for i, a in enumerate(occ):
        b = occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0)
        lo, hi = a + clear, b - clear
        if hi > lo:
            gaps.append((hi - lo, lo, hi))
    if not gaps:
        # everything is blocked; ignore the clearance and share what is left
        gaps = [(occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0) - a,
                 a, occ[(i + 1) % len(occ)] + (360.0 if i + 1 == len(occ) else 0.0))
                for i, a in enumerate(occ)]

    # hand slots to the widest gaps first, then space them inside each gap
    gaps.sort(reverse=True)
    counts = [0] * len(gaps)
    for s in range(k):
        # width per slot if this gap took one more
        j = int(np.argmax([gaps[i][0] / (counts[i] + 1) for i in range(len(gaps))]))
        counts[j] += 1

    angles = []
    for (w, lo, hi), c in zip(gaps, counts):
        if c:
            step = (hi - lo) / c
            angles += [lo + step * (i + 0.5) for i in range(c)]
    return angles



def _mirror_share(first, weights, fallback, tol=0.25):
    """Reflect one metal's ligand directions onto its partner.

    Only when the two metals carry ligands of comparable bulk: reflecting a
    carbonyl onto the position worked out for a ferrocene would trade one kind
    of ugliness for another. Returns None when the halves do not match, so the
    caller keeps its own independent assignment.
    """
    if len(first) != len(weights):
        return None
    src = sorted(range(len(first)), key=lambda i: -first[i][0])
    dst = sorted(range(len(weights)), key=lambda i: -weights[i])
    for a, b in zip(src, dst):
        wa, wb = first[a][0], weights[b]
        if abs(wa - wb) > tol * max(wa, wb, 1.0):
            return None
    out = list(fallback)
    for a, b in zip(src, dst):
        out[b] = (180.0 - first[a][1]) % 360.0
    return out


def _sites(mol, cluster):
    """Donor sites of the cluster.

    Returns (positions, sites) where `sites` lists one entry per donor group as
    (piece-level donor atoms, [metals it binds]) and `positions` gives the metal
    coordinates. Only two-metal cores are templated; anything larger is refused.
    """
    if len(cluster) != 2:
        return None, None
    m0, m1 = cluster
    pos = {m0: np.array([0.0, 0.0]), m1: np.array([D_MM, 0.0])}

    nb = {m: [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()
              if n.GetIdx() not in cluster] for m in cluster}
    bridges = [d for d in nb[m0] if d in nb[m1]]
    term = {m: [d for d in nb[m] if d not in bridges] for m in cluster}
    return pos, (bridges, term)


def _template(mol, cluster):
    """Target position for every donor atom of the cluster."""
    pos, packed = _sites(mol, cluster)
    if pos is None:
        return None
    m0, m1 = cluster
    bridges, term = packed
    target = {}

    # --- bridges sit on the perpendicular bisector, alternating sides ------- #
    half = D_MM / 2.0
    h = float(np.sqrt(max(ML ** 2 - half ** 2, 0.25 * ML ** 2)))
    for n, d in enumerate(bridges):
        side = 1.0 if n % 2 == 0 else -1.0
        # a third or fourth bridge would land on top of the first, so step out
        grow = 1.0 + 0.45 * (n // 2)
        target[d] = np.array([half, side * h * grow])

    # --- terminals fan into whatever is left of each metal's circle --------- #
    for m, other in ((m0, m1), (m1, m0)):
        occupied = [np.degrees(np.arctan2(*(pos[other] - pos[m])[::-1]))]
        for d in bridges:
            v = target[d] - pos[m]
            occupied.append(np.degrees(np.arctan2(v[1], v[0])))
        for d, a in zip(term[m], _fan(occupied, len(term[m]))):
            r = np.radians(a)
            target[d] = pos[m] + ML * np.array([np.cos(r), np.sin(r)])

    return pos, target, bridges, term


def _piece_coords(piece, depth):
    """2D coordinates for one ligand fragment, at the library's bond length."""
    if depth < 4 and core.find_metal(piece) is not None:
        try:
            sub = core.depict(piece, _depth=depth + 1)
            c = sub.GetConformer()
            xy = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                           for i in range(piece.GetNumAtoms())])
            return xy
        except Exception:
            pass
    rdCoordGen.AddCoords(piece)
    c = piece.GetConformer()
    xy = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                   for i in range(piece.GetNumAtoms())])
    if piece.GetNumBonds():
        bl = np.median([np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()])
                        for b in piece.GetBonds()])
        if bl > 1e-6:
            xy = xy * (LB / bl)
    return xy


def _fit_single(piece, xy, d_local, target, anchor):
    """One donor: aim the free side of the donor atom back at its metal."""
    nbr = [n.GetIdx() for n in piece.GetAtomWithIdx(d_local).GetNeighbors()]
    if nbr:
        ang = np.sort([np.degrees(np.arctan2(*(xy[n] - xy[d_local])[::-1])) % 360.0
                       for n in nbr])
        gaps = np.diff(np.append(ang, ang[0] + 360.0))
        g = int(np.argmax(gaps))
        toward = ang[g] + gaps[g] / 2.0
    else:
        body = np.delete(xy, d_local, axis=0)
        ref = (body.mean(0) - xy[d_local]) if len(body) else np.array([-1.0, 0.0])
        toward = np.degrees(np.arctan2(ref[1], ref[0])) + 180.0
    want = np.degrees(np.arctan2(*(anchor - target)[::-1]))
    R = _rot(want - toward)
    return xy @ R.T + (target - R @ xy[d_local])


def _demand(piece, xy, groups, dl):
    """How wide a wedge this ligand needs, in degrees, seen from its metal."""
    try:
        probe = _fit_single(piece, xy, dl[0], np.array([ML, 0.0]), np.array([0.0, 0.0]))
        w = core._angular_width(probe)
    except Exception:
        w = 30.0
    return float(np.clip(2.0 * w, 20.0, 200.0))



def depict(mol, relax=True, _depth=0):
    """Lay out a metal-metal bonded cluster. Falls back to core.depict."""
    mol = Chem.Mol(mol)
    clusters = metal_clusters(mol)
    if not clusters or len(clusters[0]) < 2:
        return core.depict(mol, cluster=False)
    cluster = clusters[0]
    built = _template(mol, cluster)
    if built is None:
        return core.depict(mol, cluster=False)
    pos, target, bridges, term = built

    cut = [b.GetIdx() for m in cluster for b in mol.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in cluster]
    if not cut:
        return core.depict(mol, cluster=False)

    frag = Chem.FragmentOnBonds(mol, cut, addDummies=False)
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                              fragsMolAtomMapping=(maps := []))
    coords = np.zeros((mol.GetNumAtoms(), 2))
    for m in cluster:
        coords[m] = pos[m]

    # ---- 1. lay every ligand out on its own, and note what it binds -------- #
    items, loose = [], []
    for piece, amap in zip(pieces, maps):
        amap = list(amap)
        if set(amap) <= set(cluster):
            continue
        xy = _piece_coords(piece, _depth)
        dl = [amap.index(d) for d in target if d in amap]
        if not dl:
            loose.append((piece, amap, xy))
            continue
        groups = _hapto_groups(piece, dl)
        binds = {m for m in cluster for d in dl
                 if mol.GetBondBetweenAtoms(m, amap[d]) is not None}
        items.append(dict(piece=piece, amap=amap, xy=xy, dl=dl,
                          groups=groups, binds=binds,
                          bridging=len(binds) > 1))

    # ---- 2. terminals share each metal's free arcs, by how much they need -- #
    half = D_MM / 2.0
    first = None
    for m in cluster:
        other = [x for x in cluster if x != m][0]
        occupied = [np.degrees(np.arctan2(*(pos[other] - pos[m])[::-1]))]
        for d in bridges:
            v = target[d] - pos[m]
            occupied.append(np.degrees(np.arctan2(v[1], v[0])))
        mine = [it for it in items if not it["bridging"] and m in it["binds"]]
        if not mine:
            continue
        w = [_demand(it["piece"], it["xy"], it["groups"], it["dl"]) for it in mine]
        angles = _share(occupied, w)

        # The core of a two-metal cluster is symmetric about the perpendicular
        # bisector, and the bridges are already placed on it. Sharing each
        # metal's circle independently ignores that: the two halves come out
        # related by a rotation rather than a reflection, so a carbonyl on one
        # metal faces a cyclopentadienyl on the other. Reflect the first
        # metal's assignment onto the second whenever the two carry ligands of
        # matching demand, which is what makes the halves comparable in the
        # first place.
        if m == cluster[0]:
            first = [(mm, aa) for mm, aa in zip(w, angles)]
        elif first is not None:
            mirrored = _mirror_share(first, w, angles)
            if mirrored is not None:
                angles = mirrored

        for it, a in zip(mine, angles):
            r = np.radians(a)
            u = np.array([np.cos(r), np.sin(r)])
            # a ligand with several donors still gets one direction; its own
            # donors are spread around that direction by the least-squares fit
            for d in it["dl"]:
                target[it["amap"][d]] = pos[m] + ML * u
            it["dir"] = u
            it["metal"] = m

    # ---- 3. fit each ligand onto the targets the template gave it ---------- #
    for it in items:
        piece, amap, xy, groups = it["piece"], it["amap"], it["xy"], it["groups"]
        src, dst = [], []
        for g in groups:
            gd = [amap[i] for i in g if amap[i] in target]
            if not gd:
                continue
            t = np.mean([target[d] for d in gd], axis=0)
            if len(g) >= 3:
                metal = next(m for m in cluster
                             if any(mol.GetBondBetweenAtoms(m, d) is not None for d in gd))
                u = t - pos[metal]
                n = np.linalg.norm(u)
                if n > 1e-9:
                    t = pos[metal] + (u / n) * HAPTO_R
                src.append(_hapto_centre(piece, g, xy))
            else:
                src.append(xy[g[0]])
            dst.append(t)
        if not dst:
            continue

        if len(dst) >= 2:
            R, tr = _kabsch2d(np.array(src), np.array(dst))
            out = xy @ R.T + tr
        elif len(groups[0]) >= 3:
            out = xy + (dst[0] - src[0])
        else:
            d_local = groups[0][0]
            bound = [m for m in cluster
                     if mol.GetBondBetweenAtoms(m, amap[d_local]) is not None]
            # A bridge is anchored on the midpoint of the metals it spans, not
            # on either one of them: aiming its body away from a single metal
            # would tilt the substituent off the perpendicular bisector, and
            # the drawing loses the symmetry every paper draws it with.
            anchor = np.mean([pos[m] for m in bound], axis=0)
            out = _fit_single(piece, xy, d_local, dst[0], anchor)
        it["out"] = out

    # ---- 4. nudge terminals off each other -------------------------------- #
    if relax:
        _settle(items, cluster, pos)

    for it in items:
        for local, glob in enumerate(it["amap"]):
            coords[glob] = it["out"][local]

    x0 = max([it["out"][:, 0].max() for it in items] + [D_MM]) + 2.0
    for piece, amap, xy in loose:
        xy = xy - xy.min(0) + np.array([x0, 0.0])
        x0 = xy[:, 0].max() + 2.0
        for local, glob in enumerate(amap):
            coords[glob] = xy[local]

    if not np.isfinite(coords).all():
        return core.depict(mol, cluster=False)

    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    conf.Set3D(False)
    mol.AddConformer(conf)
    return mol


def _settle(items, cluster, pos, sweeps=3, span=24.0, step=4.0):
    """Rotate terminal ligands about their own metal to reduce clashes.

    Bridges are pinned: moving one would pull it off the M-M bisector, which is
    the whole point of the template. Only terminals have freedom left.
    """
    movable = [it for it in items if not it.get("bridging") and "metal" in it
               and "out" in it]
    if not movable:
        return

    def others(cur):
        pts = [pos[m] for m in cluster]
        for it in items:
            if it is not cur and "out" in it:
                pts.append(it["out"])
        return np.vstack([np.atleast_2d(p) for p in pts])

    for _ in range(sweeps):
        moved = False
        for it in movable:
            rest = others(it)
            c = pos[it["metal"]]
            best, keep = None, it["out"]
            for a in np.arange(-span, span + 1e-9, step):
                cand = (it["out"] - c) @ _rot(a).T + c
                d = np.linalg.norm(cand[:, None, :] - rest[None, :, :], axis=2)
                pen = float(np.sum(np.clip(1.0 * LB - d, 0, None) ** 2))
                # prefer standing still when nothing is gained
                pen += 1e-4 * abs(a)
                if best is None or pen < best:
                    best, keep = pen, cand
            if not np.allclose(keep, it["out"]):
                moved = True
            it["out"] = keep
        if not moved:
            break


# core.draw and core._drawing_mol work unchanged on the result
draw = core.draw
_drawing_mol = core._drawing_mol
find_metal = core.find_metal
