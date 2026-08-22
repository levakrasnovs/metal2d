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

# Metals with no M-M bond, held together only by bridging donor atoms, sit
# further apart: a M-X-M diamond with right angles at both metals keeps every
# metal-bridge bond at ML and stops the two centres crowding each other.
D_BRIDGED = ML * np.sqrt(2.0)

# Keep terminal ligands this far, in degrees, from the M-M axis and from the
# bridges. Below about 25 the terminal bonds start to graze the bridge bonds.
CLEAR = 30.0

# The direction of the partner metal needs more than that when the metals are
# not bonded to each other but merely bridged: they sit close, and what a
# terminal ligand has to clear is the partner's ligands, not the partner.
MM_CLEAR = 65.0

# _settle weights: pairwise clearance, and how hard the core region is guarded
CLASH = 1.1
CORE_W = 1.0
CORE_R = 1.6


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
    """Arcs around a metal that no bridge or M-M bond already uses.

    `occupied` is a list of angles, or of (angle, clearance) pairs when some
    directions need a wider berth than others - the partner metal of a bridged
    dimer does, because what has to be cleared is not its bond but its own
    ligands, standing a bond length away.
    """
    items = []
    for o in occupied:
        if isinstance(o, (tuple, list)):
            items.append((float(o[0]) % 360.0, float(o[1])))
        else:
            items.append((float(o) % 360.0, clear))
    items.sort()
    if not items:
        return [(360.0, 0.0, 360.0)]
    arcs = []
    for i, (a, ca) in enumerate(items):
        b, cb = items[(i + 1) % len(items)]
        if i + 1 == len(items):
            b += 360.0
        lo, hi = a + ca, b - cb
        if hi > lo:
            arcs.append((hi - lo, lo, hi))
    if not arcs:
        arcs = [(items[(i + 1) % len(items)][0] + (360.0 if i + 1 == len(items) else 0.0) - a,
                 a, items[(i + 1) % len(items)][0] + (360.0 if i + 1 == len(items) else 0.0))
                for i, (a, _c) in enumerate(items)]
    return sorted(arcs, reverse=True)


def _share(occupied, weights, clear=CLEAR, avoid=None):
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
        # Order inside the arc matters as much as the share of it. Filling from
        # one end puts the bulkiest ligand hard against whatever bounds that
        # end - the partner metal, usually - and on a bridged dimer that is
        # what makes two triarylphosphines meet over the core. The ends of an
        # arc are its cramped part; give them to the small ligands and keep the
        # middle for the big one.
        members = sorted(members, key=lambda i: -weights[i])
        left, right = [], []
        for n, i in enumerate(members[1:]):
            (left if n % 2 else right).append(i)
        members = left[::-1] + members[:1] + right
        if avoid is not None:
            def gap(a):
                return abs((a - avoid + 180.0) % 360.0 - 180.0)
            # with two ligands there is no middle to speak of; what decides the
            # picture is which end of the arc the big one gets, so hand it the
            # end further from the partner metal
            if gap(lo) < gap(hi):
                members = members[::-1]

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


def atom_bridged_pairs(mol):
    """Pairs of metals sharing bridging donor *atoms*, with no M-M bond.

    A halide- or alkoxide-bridged dimer such as Cu2Br2 has no metal-metal bond,
    so `has_cluster` is False, and the whole bridge is a single atom, so the
    macrocycle path treats it as a ring to be built by the classic depictor -
    which knows nothing about coordination geometry and stacks the terminal
    ligands on top of each other. Structurally these are clusters: the same
    rigid template applies, only the metal separation differs.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS]
    if len(metals) < 2:
        return []
    mset = set(metals)
    shared = {}
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in mset:
            continue
        ms = sorted(n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in mset)
        if len(ms) == 2:
            shared.setdefault(tuple(ms), []).append(i)
    out = [p for p, br in sorted(shared.items()) if br
           and mol.GetBondBetweenAtoms(*p) is None]
    return out


def _sites(mol, cluster, sep=D_MM):
    """Donor sites of the cluster.

    Returns (positions, sites) where `sites` lists one entry per donor group as
    (piece-level donor atoms, [metals it binds]) and `positions` gives the metal
    coordinates. Only two-metal cores are templated; anything larger is refused.
    """
    if len(cluster) != 2:
        return None, None
    m0, m1 = cluster
    pos = {m0: np.array([0.0, 0.0]), m1: np.array([sep, 0.0])}

    nb = {m: [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()
              if n.GetIdx() not in cluster] for m in cluster}
    bridges = [d for d in nb[m0] if d in nb[m1]]
    term = {m: [d for d in nb[m] if d not in bridges] for m in cluster}
    return pos, (bridges, term)


def _template(mol, cluster, sep=D_MM):
    """Target position for every donor atom of the cluster."""
    pos, packed = _sites(mol, cluster, sep)
    if pos is None:
        return None
    m0, m1 = cluster
    bridges, term = packed
    target = {}

    # --- bridges sit on the perpendicular bisector, alternating sides ------- #
    half = sep / 2.0
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


def _bite(piece, xy, groups):
    """Angle at the metal between neighbouring donors of one ligand."""
    reps = [(_hapto_centre(piece, g, xy) if len(g) >= 3 else xy[g[0]])
            for g in groups]
    if len(reps) < 2:
        return 0.0
    dd = float(np.mean([np.linalg.norm(reps[j + 1] - reps[j])
                        for j in range(len(reps) - 1)]))
    return float(np.degrees(np.arcsin(min(dd / (2 * ML), 0.95))) * 2.0)


def _demand(piece, xy, groups, dl):
    """How wide a wedge this ligand needs, in degrees, seen from its metal."""
    try:
        probe = _fit_single(piece, xy, dl[0], np.array([ML, 0.0]), np.array([0.0, 0.0]))
        w = core._angular_width(probe)
    except Exception:
        w = 30.0
    return float(np.clip(2.0 * w, 20.0, 200.0))



def depict(mol, relax=True, _depth=0, pair=None, sep=None):
    """Lay out a metal-metal bonded cluster. Falls back to core.depict.

    `pair` forces a two-metal core that is not M-M bonded (a halide-bridged
    dimer, say); `sep` overrides the metal separation used by the template.
    """
    mol = Chem.Mol(mol)
    if pair is not None:
        cluster = list(pair)
        if sep is None:
            sep = D_BRIDGED
    else:
        clusters = metal_clusters(mol)
        if not clusters or len(clusters[0]) < 2:
            return core.depict(mol, cluster=False)
        cluster = clusters[0]
        if sep is None:
            sep = D_MM
    built = _template(mol, cluster, sep)
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

    # ---- 2. share each metal's free arc between the ligands on it ---------- #
    # Every donor that is not a bridge needs a slot on its metal's circle, and
    # the donors of one ligand have to end up next to each other: a chelate
    # whose two donors are put on opposite sides of the metal cannot be drawn
    # without stretching a bond across the picture. So each ligand claims a
    # contiguous block of the free arc, sized by how much room it needs, and
    # spreads its own donors inside that block.
    #
    # A ligand that also owns a bridge is a special case worth handling
    # explicitly - a salen sharing one phenolate while keeping its imine and
    # its second phenolate on one metal is common - because its block has to
    # start at the bridge, not wherever the sharing happens to leave room.
    for m in cluster:
        other = [x for x in cluster if x != m][0]
        mm = np.degrees(np.arctan2(*(pos[other] - pos[m])[::-1]))
        occupied = [(mm, MM_CLEAR if mol.GetBondBetweenAtoms(m, other) is None
                     else CLEAR)]
        for d in bridges:
            v = target[d] - pos[m]
            occupied.append(np.degrees(np.arctan2(v[1], v[0])))

        claims = []
        for it in items:
            gs, bridge_ang = [], None
            for g in it["groups"]:
                bound = {x for x in cluster
                         if any(mol.GetBondBetweenAtoms(x, it["amap"][d]) is not None
                                for d in g)}
                if len(bound) > 1:
                    if m in bound:
                        v = np.mean([target[it["amap"][d]] for d in g], axis=0) - pos[m]
                        bridge_ang = np.degrees(np.arctan2(v[1], v[0]))
                elif bound == {m}:
                    gs.append(g)
            if gs:
                # A single rigid ligand donating at least twice to both metals
                # is the *object between the centres*, not two unrelated
                # terminal chelates. Seat its donor block towards the partner
                # metal. The ordinary free-arc allocator deliberately excludes
                # that direction and used to push all four donors around the
                # outside, folding one Ru/arene half back through the ligand.
                if it["bridging"] and len(gs) >= 2 and bridge_ang is None:
                    bite = max(_bite(it["piece"], it["xy"], gs), 35.0)
                    angles = np.linspace(mm - bite / 2.0,
                                         mm + bite / 2.0, len(gs))
                    for g, a in zip(gs, angles):
                        rr = HAPTO_R if len(g) >= 3 else ML
                        r = np.radians(a)
                        q = pos[m] + rr * np.array([np.cos(r), np.sin(r)])
                        for d in g:
                            target[it["amap"][d]] = q
                    it["dir"] = np.array([np.cos(np.radians(mm)),
                                           np.sin(np.radians(mm))])
                    continue
                claims.append(dict(it=it, groups=gs, bridge=bridge_ang,
                                   w=max(_demand(it["piece"], it["xy"], gs,
                                                 it["dl"]),
                                         _bite(it["piece"], it["xy"], gs)
                                         * (len(gs) - 1) + 20.0)))
        if not claims:
            continue

        def bite_of(c):
            return _bite(c["it"]["piece"], c["it"]["xy"], c["groups"])

        def _seat_rigid_unused(c, mid):
            """Three or more donors: put the metal at the centre of the circle
            through them, as core.depict does. An equal-angle ring of slots
            cannot match three real donor spacings at once - a salen's O-N-O is
            not two equal steps - and the fit pays for the mismatch by
            stretching a bond. Returns False when no usable circle exists.
            """
            piece, pxy = c["it"]["piece"], c["it"]["xy"]
            if len(c["groups"]) < 3 or any(len(g) >= 3 for g in c["groups"]):
                return False
            src = np.array([pxy[g[0]] for g in c["groups"]], dtype=float)
            cen, rad = core._donor_circle(src)
            if cen is None or not (0.7 * ML <= rad <= 2.2 * ML):
                return False
            d = (src - cen).mean(0)
            if np.linalg.norm(d) < 1e-9:
                return False
            R = _rot(mid - np.degrees(np.arctan2(d[1], d[0])))
            for g in c["groups"]:
                q = pos[m] + R @ (pxy[g[0]] - cen)
                for dd_ in g:
                    target[c["it"]["amap"][dd_]] = q
            r = np.radians(mid)
            c["it"]["dir"] = np.array([np.cos(r), np.sin(r)])
            if not c["it"]["bridging"]:
                c["it"]["metal"] = m
            return True

        def seat(c, angles):
            for g, a in zip(c["groups"], angles):
                rr = HAPTO_R if len(g) >= 3 else ML
                r = np.radians(a)
                q = pos[m] + rr * np.array([np.cos(r), np.sin(r)])
                for d in g:
                    target[c["it"]["amap"][d]] = q
            mid = np.radians(float(np.mean(angles)))
            c["it"]["dir"] = np.array([np.cos(mid), np.sin(mid)])
            if not c["it"]["bridging"]:
                c["it"]["metal"] = m

        # A ligand that owns a bridge is seated first and from the bridge
        # outwards: its donors are rungs of one chain, and the bridge end of
        # that chain is already pinned on the bisector. Anything else would ask
        # the fit to reconcile a pinned donor with a block placed elsewhere,
        # and the fit answers by stretching a metal-donor bond.
        anchored = [c for c in claims if c["bridge"] is not None]
        for c in anchored:
            b = bite_of(c) or 55.0
            sign = 1.0 if ((c["bridge"] - mm) % 360.0) < 180.0 else -1.0
            angles = [c["bridge"] + sign * b * (j + 1)
                      for j in range(len(c["groups"]))]
            seat(c, angles)
            for a in angles:
                occupied.append(a)

        free = sorted([c for c in claims if c["bridge"] is None],
                      key=lambda c: -c["w"])
        if not free:
            continue
        arcs = _free_arcs(occupied)
        w, lo, hi = arcs[0]
        tot = sum(c["w"] for c in free) or 1.0
        acc = lo
        for c in free:
            span = (hi - lo) * c["w"] / tot
            k = len(c["groups"])
            # the bite is the ligand's own and must not be squeezed to fit the
            # block: a clipped bite is a stretched metal-donor bond
            b = bite_of(c)
            seat(c, [acc + span / 2.0 + b * (j - (k - 1) / 2.0)
                     for j in range(k)])
            acc += span

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

    x0 = max([it["out"][:, 0].max() for it in items] + [sep]) + 2.0
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


def _settle(items, cluster, pos, sweeps=3, span=24.0, step=4.0,
            spin=40.0, spin_step=10.0):
    """Work terminal ligands off each other, and off the core.

    Three degrees of freedom per terminal, in order of how much they cost the
    drawing: rotation about its own metal, which moves the whole ligand round
    the circle; rotation about its own donor atom, which swings the body while
    the metal-donor bond stays put; and reflection of the body in that bond,
    which is free of charge because a 2D depiction has no handedness to lose.

    Rotation about the metal alone is not enough once the ligands are bulky and
    the core is small: four triarylphosphine-sized ligands on two metals a bond
    apart cannot be separated by sliding them round a circle they all share.
    The spin is what lets a ligand fold its bulk outward instead.

    Bridges are pinned: moving one would pull it off the M-M bisector, which is
    the whole point of the template.
    """
    movable = [it for it in items if not it.get("bridging") and "metal" in it
               and "out" in it]
    if not movable:
        return
    centre = np.mean([pos[m] for m in cluster], axis=0)

    for it in items:
        it["_bonds"] = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                        for b in it["piece"].GetBonds()]

    def others(cur):
        pts = [pos[m] for m in cluster]
        for it in items:
            if it is not cur and "out" in it:
                pts.append(it["out"])
        return np.vstack([np.atleast_2d(p) for p in pts])

    def other_segments(cur):
        segs = []
        for it in items:
            if it is not cur and "out" in it:
                xy = it["out"]
                for i, j in it["_bonds"]:
                    segs.append([xy[i], xy[j]])
        return np.array(segs) if segs else np.zeros((0, 2, 2))

    def _cross_count(seg_a, seg_b):
        """Number of crossings between two sets of segments, vectorised."""
        if not len(seg_a) or not len(seg_b):
            return 0.0
        p, q = seg_a[:, 0, :][:, None, :], seg_a[:, 1, :][:, None, :]
        u, v = seg_b[None, :, 0, :], seg_b[None, :, 1, :]
        d1, d2 = q - p, v - u
        den = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        w = u - p
        t = (w[..., 0] * d2[..., 1] - w[..., 1] * d2[..., 0]) / den
        r = (w[..., 0] * d1[..., 1] - w[..., 1] * d1[..., 0]) / den
        hit = ok & (t > 1e-6) & (t < 1 - 1e-6) & (r > 1e-6) & (r < 1 - 1e-6)
        return float(hit.sum())

    def penalty(cand, rest, keep_out):
        d = np.linalg.norm(cand[:, None, :] - rest[None, :, :], axis=2)
        pen = float(np.sum(np.clip(CLASH * LB - d, 0, None) ** 2))
        # a ligand folded back over the core is not caught by pairwise
        # clearances alone - the atoms it lands on top of are its neighbours'
        # empty middle - so charge for intruding on the core itself
        # Nothing but the metals, the bridges and each ligand's own donor
        # belongs in the middle of the core. Pairwise clearances do not defend
        # it: a phenyl can lie across the Cu2Br2 rhombus with every atom a
        # comfortable distance from every label, and the drawing is still
        # unreadable. Guard the core as a region.
        dc = np.linalg.norm(cand[keep_out] - centre, axis=1)
        pen += CORE_W * float(np.sum(np.clip(CORE_R * ML - dc, 0, None) ** 2))
        return pen

    def crossings(cand, bonds, segs):
        if not bonds:
            return 0.0
        mine = np.array([[cand[i], cand[j]] for i, j in bonds])
        return _cross_count(mine, segs)

    for _ in range(sweeps):
        moved = False
        for it in movable:
            rest = others(it)
            segs = other_segments(it)
            c = pos[it["metal"]]
            base = it["out"]
            d_local = it["dl"][0]
            keep_out = np.ones(len(base), bool)
            keep_out[it["dl"]] = False
            best, keep = None, base
            for a in np.arange(-span, span + 1e-9, step):
                rotated = (base - c) @ _rot(a).T + c
                anchor = rotated[d_local]
                axis = anchor - c
                nrm = np.linalg.norm(axis)
                for mirror in (False, True):
                    if mirror:
                        if nrm < 1e-9:
                            continue
                        u = axis / nrm
                        # reflect the body in the metal-donor bond
                        v = rotated - anchor
                        proj = np.outer(v @ u, u)
                        seed = anchor + 2.0 * proj - v
                    else:
                        seed = rotated
                    for b in np.arange(-spin, spin + 1e-9, spin_step):
                        cand = (seed - anchor) @ _rot(b).T + anchor
                        pen = penalty(cand, rest, keep_out)
                        # Clearances alone do not see a crossing: two aromatic
                        # rings can thread through each other with every atom
                        # a comfortable distance from every other. Count the
                        # crossings this ligand makes with the rest and charge
                        # for them directly - it is the thing being minimised.
                        pen += 6.0 * crossings(cand, it["_bonds"], segs)
                        # prefer standing still when nothing is gained
                        pen += 1e-4 * (abs(a) + abs(b) + 5.0 * mirror)
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



# --------------------------------------------------------------------------- #
#  metals tied together by whole bridging ligands (macrocyclic dimers)
# --------------------------------------------------------------------------- #
def ligand_bridged_pairs(mol):
    """Pairs of metals whose relative position is fixed by organic bridges.

    Two ligands across the same pair close a macrocycle through both centres.
    That pins their separation, and laying one centre out at a time cannot know
    it: the arms come out extended, the ring never closes and the bonds to the
    second metal are drawn several times too long. A single bridge normally
    leaves a free hinge and is handled fine by the ordinary path.  The important
    exception is one rigid multidentate ligand: if the same cyclic/conjugated
    fragment gives at least two donors to *each* metal, it fixes the pair just as
    surely as two separate bridges do.

    RingInfo is no use for this - SSSR does not report the macrocycle - so the
    ring is found by fragmenting on the metal bonds and asking which pieces
    touch two metals at once.
    """
    metals = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in METALS]
    if len(metals) < 2:
        return []
    mset = set(metals)
    cut = [b.GetIdx() for m in metals for b in mol.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in mset]
    if not cut:
        return []
    donor_of = {}
    for m in metals:
        for n in mol.GetAtomWithIdx(m).GetNeighbors():
            if n.GetIdx() not in mset:
                donor_of.setdefault(n.GetIdx(), set()).add(m)

    maps = []
    Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut, addDummies=False),
                     asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps)
    span = {}
    rigid_multidentate = set()
    for amap in maps:
        served = set()
        for a in amap:
            served |= donor_of.get(a, set())
        if len(served) == 2:
            key = tuple(sorted(served))
            span[key] = span.get(key, 0) + 1
            contacts = {
                m: [a for a in amap if m in donor_of.get(a, set())]
                for m in key
            }
            if all(len(contacts[m]) >= 2 for m in key):
                # A donor pair on opposite metals must be connected through a
                # rigid part of the ligand.  Ring bonds and multiple/aromatic
                # bonds cannot act as the free hinge that makes an ordinary
                # monodentate bridge harmless.  This deliberately tests graph
                # topology only; it is element- and complex-independent.
                rigid = False
                for a in contacts[key[0]]:
                    for b in contacts[key[1]]:
                        path = Chem.GetShortestPath(mol, a, b)
                        if path and all(
                            (bond := mol.GetBondBetweenAtoms(i, j)).IsInRing()
                            or bond.GetBondType() != Chem.BondType.SINGLE
                            for i, j in zip(path, path[1:])
                        ):
                            rigid = True
                            break
                    if rigid:
                        break
                if rigid:
                    rigid_multidentate.add(key)
    return sorted({p for p, n in span.items() if n >= 2} | rigid_multidentate)


def haptic_centroid_penalty(mol):
    """Penalty for an eta-bound ring collapsed onto its metal centre.

    The drawing uses one centroid bond for an eta group, so its length should
    be comparable to the other metal-ligand bonds.  Recursive and pinned-ring
    layouts can put the metal at the ring centroid instead; crossings alone do
    not notice that chemically misleading collapse.
    """
    if mol.GetNumConformers() == 0:
        return 0.0
    conf = mol.GetConformer()
    penalty = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in METALS:
            continue
        donors = [n.GetIdx() for n in atom.GetNeighbors()
                  if n.GetAtomicNum() not in METALS]
        for group in _hapto_groups(mol, donors):
            if len(group) < 3:
                continue
            centre = np.mean([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y] for i in group], axis=0)
            p = conf.GetAtomPosition(atom.GetIdx())
            distance = float(np.linalg.norm(centre - [p.x, p.y]))
            if distance < 0.55 * ML:
                penalty += 20.0 * (0.55 * ML - distance) / ML
    return penalty


def symmetrize_equivalent_metal_halves(mol, pair):
    """Copy the cleaner terminal half across a symmetric rigid M2 bridge.

    Recursive construction is rooted at one metal.  For two graph-equivalent
    centres this can leave one eta ligand outside the bridge and collapse the
    other one onto its metal.  The shared ligand is already a rigid scaffold;
    retain it unchanged and copy one complete terminal half (metal, eta ligand,
    halide, substituents) through the scaffold centre using a graph
    automorphism that exchanges the metals.
    """
    out = Chem.Mol(mol)
    if out.GetNumConformers() == 0:
        return out
    m0, m1 = pair
    mappings = [a for a in out.GetSubstructMatches(
        out, uniquify=False, maxMatches=256)
        if a[m0] == m1 and a[m1] == m0]
    if not mappings:
        return out

    mset = {m0, m1}
    cut = [b.GetIdx() for m in pair for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in mset]
    maps = []
    Chem.GetMolFrags(Chem.FragmentOnBonds(out, cut, addDummies=False),
                     asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    binds = []
    for amap in maps:
        binds.append({m for m in pair for a in amap
                      if out.GetBondBetweenAtoms(m, a) is not None})
    shared = [a for amap, bs in zip(maps, binds) if len(bs) == 2 for a in amap]
    if not shared:
        return out
    halves = {
        m: {m} | {a for amap, bs in zip(maps, binds) if bs == {m} for a in amap}
        for m in pair
    }
    conf = out.GetConformer()
    base = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                     for i in range(out.GetNumAtoms())])
    # Donor atoms define the coordination cavity more reliably than bulky
    # substituents do, so use them for the C2 centre.
    donors = [a for a in shared if any(
        out.GetBondBetweenAtoms(m, a) is not None for m in pair)]
    centre = base[donors if donors else shared].mean(axis=0)

    def with_xy(xy):
        cand = Chem.Mol(out)
        c = cand.GetConformer()
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        return cand

    candidates = [out]
    for amap in mappings:
        for source, target in ((m0, m1), (m1, m0)):
            if {amap[i] for i in halves[source]} != halves[target]:
                continue
            xy = base.copy()
            for i in halves[source]:
                xy[amap[i]] = 2.0 * centre - base[i]
            candidates.append(with_xy(xy))
    return min(candidates, key=lambda c: (
        haptic_centroid_penalty(c), _tangle(c, strain=False), _tangle(c)))


def _tangle(mol, strain=True):
    """Crossings plus near-coincident atoms, for ranking candidate layouts."""
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(mol.GetNumAtoms())])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not bonds:
        return 0.0
    unit = float(np.median([np.linalg.norm(xy[i] - xy[j]) for i, j in bonds]))
    if unit < 1e-9:
        return float("inf")
    n = 0
    for a in range(len(bonds)):
        p, q = xy[bonds[a][0]], xy[bonds[a][1]]
        for b in range(a + 1, len(bonds)):
            if len(set(bonds[a]) | set(bonds[b])) < 4:
                continue
            u, v = xy[bonds[b][0]], xy[bonds[b][1]]
            d1, d2 = q - p, v - u
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                continue
            t = ((u - p)[0] * d2[1] - (u - p)[1] * d2[0]) / den
            s = ((u - p)[0] * d1[1] - (u - p)[1] * d1[0]) / den
            if 1e-6 < t < 1 - 1e-6 and 1e-6 < s < 1 - 1e-6:
                n += 1
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    clashes = 2.0 * float(np.sum(d < 0.35 * unit) / 2)

    # Pinning the metals fixes where they are, not whether the ring can reach
    # them. At a separation the bridges cannot span, the depictor keeps the
    # metals put and stretches their bonds instead - three times over in the
    # worst case seen - which no amount of crossing-counting notices. Charge for
    # every bond that is not close to the molecule's own bond length.
    strain_cost = 0.0
    if strain:
        for i, j in bonds:
            L = float(np.linalg.norm(xy[i] - xy[j])) / unit
            if L > 1.3 or L < 0.7:
                strain_cost += 4.0 * abs(L - 1.0)
    return n + clashes + strain_cost


def _tight_bonds(mol, frac=0.30):
    """Count close non-crossing independent bond pairs for final acceptance.

    This intentionally mirrors the review metric, but is used only twice on a
    rare specialised candidate; candidate searches continue to use the much
    faster vectorised crossing score.
    """
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return 0
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(mol.GetNumAtoms())])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    lengths = [np.linalg.norm(xy[i] - xy[j]) for i, j in bonds]
    unit = float(np.median(lengths))
    if unit < 1e-9:
        return 0

    def point_segment(p, a, b):
        d = b - a
        t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * d)))

    count = 0
    for a in range(len(bonds)):
        i, j = bonds[a]
        p, q = xy[i], xy[j]
        for b in range(a + 1, len(bonds)):
            u, v = bonds[b]
            if len({i, j, u, v}) < 4:
                continue
            r, s = xy[u], xy[v]
            d1, d2 = q - p, s - r
            den = d1[0] * d2[1] - d1[1] * d2[0]
            crossed = False
            if abs(den) >= 1e-9:
                w = r - p
                t = (w[0] * d2[1] - w[1] * d2[0]) / den
                z = (w[0] * d1[1] - w[1] * d1[0]) / den
                crossed = 1e-6 < t < 1 - 1e-6 and 1e-6 < z < 1 - 1e-6
            if crossed:
                continue
            distance = min(point_segment(p, r, s), point_segment(q, r, s),
                           point_segment(r, p, q), point_segment(s, p, q))
            count += distance < frac * unit
    return int(count)


def _independent_bond_pairs(bonds):
    bonds = np.asarray(bonds, dtype=int)
    pairs = [(i, j) for i in range(len(bonds)) for j in range(i + 1, len(bonds))
             if len(set(bonds[i]) | set(bonds[j])) == 4]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _tangle_xy(xy, bonds, strain=True, pairs=None):
    """Coordinate-only equivalent of :func:`_tangle` for candidate searches."""
    if not bonds:
        return 0.0
    bonds = np.asarray(bonds, dtype=int)
    lengths = np.linalg.norm(xy[bonds[:, 0]] - xy[bonds[:, 1]], axis=1)
    unit = float(np.median(lengths))
    if unit < 1e-9:
        return float("inf")
    pairs = _independent_bond_pairs(bonds) if pairs is None else pairs
    n = 0
    if len(pairs):
        first, second = bonds[pairs[:, 0]], bonds[pairs[:, 1]]
        p, q = xy[first[:, 0]], xy[first[:, 1]]
        u, v = xy[second[:, 0]], xy[second[:, 1]]
        d1, d2 = q - p, v - u
        den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        ok = np.abs(den) >= 1e-9
        safe = np.where(ok, den, 1.0)
        w = u - p
        t = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / safe
        s = (w[:, 0] * d1[:, 1] - w[:, 1] * d1[:, 0]) / safe
        n = int(np.sum(ok & (t > 1e-6) & (t < 1 - 1e-6) &
                       (s > 1e-6) & (s < 1 - 1e-6)))
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    clashes = 2.0 * float(np.sum(d < 0.35 * unit) / 2)
    strain_cost = 0.0
    if strain:
        scaled = lengths / unit
        bad = (scaled > 1.3) | (scaled < 0.7)
        strain_cost = 4.0 * float(np.sum(np.abs(scaled[bad] - 1.0)))
    return n + clashes + strain_cost


def _coordinate_scorer(mol):
    """Return the exact drawing-topology score without rebuilding RDKit Mol.

    ``_drawing_mol`` is called once. Candidate coordinates for original atoms
    are copied into that topology, and eta-centroid dummy atoms are recomputed
    directly from their recorded ring groups.
    """
    drawn = core._drawing_mol(mol)
    n_original = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in drawn.GetBonds()]
    pairs = _independent_bond_pairs(bonds)
    dummy_groups = {}
    for a in drawn.GetAtoms():
        if a.HasProp("_hapticAtoms"):
            grp = [int(x) for x in a.GetProp("_hapticAtoms").split(",")]
            ring = [i for i in grp if drawn.GetAtomWithIdx(i).IsInRing()]
            dummy_groups[a.GetIdx()] = ring if ring else grp

    def score(xy, strain=True):
        if drawn.GetNumAtoms() == n_original:
            full = xy
        else:
            full = np.empty((drawn.GetNumAtoms(), 2), dtype=float)
            full[:n_original] = xy
            for i, grp in dummy_groups.items():
                full[i] = xy[grp].mean(axis=0)
        return _tangle_xy(full, bonds, strain=strain, pairs=pairs)
    return score


def _plain_coordinate_scorer(mol):
    """Coordinate scorer for code paths that intentionally rank raw bonds."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    pairs = _independent_bond_pairs(bonds)

    def score(xy, strain=True):
        return _tangle_xy(xy, bonds, strain=strain, pairs=pairs)
    return score


def polish_organic_branches(mol, sweeps=3, step=10.0, max_atoms=12):
    """Turn small organic substituents after a rigid cluster layout.

    Cluster templates intentionally treat a bridging ligand as a rigid object.
    That protects Fe/Cp/CO cores, but a benzyl or aryl branch attached through
    an ordinary acyclic single bond may then point straight through the core.
    Rotate only the smaller side of such bonds, never a metal bond or a ring
    bond, and accept a move only when global legibility improves.
    """
    out = Chem.Mol(mol)
    if out.GetNumConformers() == 0:
        return out
    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    axes = []
    for bond in out.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (out.GetAtomWithIdx(u).GetAtomicNum() in METALS or
                out.GetAtomWithIdx(v).GetAtomicNum() in METALS):
            continue
        edited = Chem.RWMol(out)
        edited.RemoveBond(u, v)
        frags = [set(f) for f in Chem.GetMolFrags(edited.GetMol())]
        side_v = next((f for f in frags if v in f), None)
        side_u = next((f for f in frags if u in f), None)
        if side_v is None or side_u is None:
            continue
        if len(side_v) > len(side_u):
            u, v, side_v = v, u, side_u
        if 2 <= len(side_v) <= min(max_atoms, out.GetNumAtoms() // 3):
            axes.append((u, np.asarray(sorted(side_v), dtype=int)))
    if not axes:
        return out

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def score(xy):
        return coordinate_score(xy, strain=False)

    angles = np.arange(-180.0, 180.0 + 1e-9, step)
    for _ in range(sweeps):
        changed = False
        for anchor, ids in axes:
            best = (score(P), 0.0, P)
            for angle in angles:
                cand = P.copy()
                cand[ids] = ((cand[ids] - cand[anchor]) @ _rot(angle).T
                             + cand[anchor])
                trial = (score(cand), abs(float(angle)), cand)
                if trial[:2] < best[:2]:
                    best = trial
            if best[0] < score(P):
                P = best[2]
                changed = True
        if not changed:
            break
    return with_coords(P)


def polish_bridge_ligands(mol, pair, sweeps=4, span=120.0, step=5.0):
    """Rigidly rotate O,N chelates about an O anchor after recursive layout.

    No internal coordinate of a ligand changes.  This is a final local polish
    for bridged dimers where recursion has produced a readable global layout,
    but a second donor can be brought closer to its metal by turning the whole
    ligand a few degrees around its already well-placed oxygen.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)

    pieces = []
    for amap0 in maps:
        amap = list(amap0)
        if len(amap) < 2:
            continue
        bound = [(d, m) for d in amap for m in metals
                 if out.GetBondBetweenAtoms(d, m) is not None]
        donors = sorted({d for d, _m in bound})
        if len(donors) < 2:
            continue
        oxy = [d for d in donors if out.GetAtomWithIdx(d).GetAtomicNum() == 8]
        if not oxy:
            continue
        # A shared oxygen is the strongest anchor; otherwise atom order is
        # deterministic and either phenoxide of an O,N,O ligand is acceptable.
        anchor = max(oxy, key=lambda d: (sum(x == d for x, _m in bound), -d))
        pieces.append((amap, anchor, bound))
    if not pieces or out.GetNumConformers() == 0:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _plain_coordinate_scorer(out)

    def quality(xy, focus=None):
        legibility = coordinate_score(xy, strain=False)
        total_bond_cost = 0.0
        for _amap, _anchor, bound in pieces:
            for d, m in bound:
                delta = np.linalg.norm(xy[d] - xy[m]) - ML
                total_bond_cost += delta * delta
        return legibility, total_bond_cost

    angles = np.arange(-span, span + 1e-9, step)
    for _ in range(sweeps):
        changed = False
        for amap, anchor, bound in pieces:
            focus = (amap, anchor, bound)
            best = (quality(P, focus), 0.0, P)
            ids = np.asarray(amap, dtype=int)
            for angle in angles:
                cand = P.copy()
                cand[ids] = ((cand[ids] - cand[anchor]) @ _rot(angle).T
                             + cand[anchor])
                trial = (quality(cand, focus), abs(float(angle)), cand)
                if trial[:2] < best[:2]:
                    best = trial
            if best[1] > 1e-9:
                changed = True
            P = best[2]
        if not changed:
            break
    return with_coords(P)


def polish_bridge_sides(mol, pair):
    """Reflect whole-ligand bridges across the M-M axis when that is clearer.

    This is a rigid operation, so the strict cluster core, internal ligand
    geometry, and all distances from a bridge to either metal are preserved.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    if not cut:
        return out
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    pieces = []
    for amap0 in maps:
        amap = list(amap0)
        if len(amap) < 2:
            continue
        binds = {m for m in metals for d in amap
                 if out.GetBondBetweenAtoms(m, d) is not None}
        if len(binds) == 2:
            pieces.append(amap)
    if not pieces or len(pieces) > 6:
        return out

    conf = out.GetConformer()
    base = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                     for i in range(out.GetNumAtoms())])
    origin = base[metals[0]]
    axis = base[metals[1]] - origin
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    u = axis / norm

    def reflected(xy, ids):
        cand = xy.copy()
        ii = np.asarray(ids, dtype=int)
        v = cand[ii] - origin
        cand[ii] = origin + 2.0 * np.outer(v @ u, u) - v
        return cand

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    best = (quality(base), 0, base)
    for mask in range(1, 1 << len(pieces)):
        xy = base
        for j, ids in enumerate(pieces):
            if mask & (1 << j):
                xy = reflected(xy, ids)
        trial = (quality(xy), mask.bit_count(), xy)
        if trial[:2] < best[:2]:
            best = trial
    return with_coords(best[2])


def polish_haptic_sectors(mol, pair, sweeps=2, step=10.0):
    """Turn an entire terminal eta-bound ligand around its own metal.

    Rotation about the metal preserves every M-C radius and the ligand's
    internal geometry.  The cluster fitter already searches a narrow local
    interval; this final pass supplies the missing full-circle sector choice
    when a bulky whole-ligand bridge occupies that initial sector.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    pieces = []
    for amap0 in maps:
        amap = list(amap0)
        if len(amap) < 3:
            continue
        bound = [(d, m) for d in amap for m in metals
                 if out.GetBondBetweenAtoms(d, m) is not None]
        donors = {d for d, _m in bound}
        bound_metals = {m for _d, m in bound}
        if len(donors) >= 3 and len(bound_metals) == 1:
            pieces.append((np.asarray(amap, dtype=int), next(iter(bound_metals))))
    if not pieces:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    angles = np.arange(-180.0, 180.0 + 1e-9, step)
    for _ in range(sweeps):
        changed = False
        for ids, metal in pieces:
            best = (quality(P), 0.0, P)
            for angle in angles:
                cand = P.copy()
                cand[ids] = ((cand[ids] - cand[metal]) @ _rot(angle).T
                             + cand[metal])
                trial = (quality(cand), abs(float(angle)), cand)
                if trial[:2] < best[:2]:
                    best = trial
            if best[1] > 1e-9:
                changed = True
            P = best[2]
        if not changed:
            break
    return with_coords(P)


def polish_anchored_bridges(mol, pair, step=5.0):
    """Swing a large chelating bridge about its shared M2 anchor.

    Some Fe2 C3/N bridges contain one atom bound to both metals plus additional
    donors bound to one centre.  A least-squares fit can fold the organic body
    through the core.  Rotation about the shared donor keeps that primary M2-C
    anchor and the complete ligand geometry fixed; only the closing
    coordination lines are allowed to lengthen when legibility improves.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    pieces = []
    for amap0 in maps:
        amap = list(amap0)
        if len(amap) < 6:
            continue
        donors = [d for d in amap
                  if any(out.GetBondBetweenAtoms(m, d) is not None
                         for m in metals)]
        shared = [d for d in donors
                  if all(out.GetBondBetweenAtoms(m, d) is not None
                         for m in metals)]
        binds = {m for m in metals for d in donors
                 if out.GetBondBetweenAtoms(m, d) is not None}
        if len(binds) == 2 and len(shared) == 1 and len(donors) >= 2:
            pieces.append((np.asarray(amap, dtype=int), shared[0]))
    if not pieces:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    for ids, anchor in pieces:
        best = (quality(P), 0.0, P)
        for angle in np.arange(-180.0, 180.0 + 1e-9, step):
            cand = P.copy()
            cand[ids] = ((cand[ids] - cand[anchor]) @ _rot(angle).T
                         + cand[anchor])
            trial = (quality(cand), abs(float(angle)), cand)
            # Closing coordination bonds may lengthen substantially.  Require
            # a strict legibility win; strain alone must never justify moving
            # an already readable anchored bridge.
            if trial[0][0] < best[0][0] or (
                    trial[0][0] == best[0][0] and
                    best[1] > 1e-9 and trial[:2] < best[:2]):
                best = trial
        P = best[2]
    return with_coords(P)


def orient_mu_carbonyl_opposite_bridge(mol, pair):
    """Put a mu-CO opposite a large shared organic bridge across M-M.

    Pure crossing scores can prefer both bridges in the same half-plane.  In
    Fe2 carbonyl chemistry that is chemically unreadable and wastes the empty
    sector below the core.  Reflecting the complete C-O fragment in the M-M
    axis preserves both M-C distances and its internal bond geometry.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                              fragsMolAtomMapping=maps)
    large_anchor = None
    mu_co = None
    for piece, amap0 in zip(pieces, maps):
        amap = list(amap0)
        shared = [d for d in amap
                  if all(out.GetBondBetweenAtoms(m, d) is not None
                         for m in metals)]
        if len(amap) >= 6 and len(shared) == 1:
            large_anchor = shared[0]
        if len(amap) == 2 and len(shared) == 1:
            nums = {out.GetAtomWithIdx(d).GetAtomicNum() for d in amap}
            if nums == {6, 8}:
                mu_co = (np.asarray(amap, dtype=int), shared[0])
    if large_anchor is None or mu_co is None:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    origin = P[metals[0]]
    axis = P[metals[1]] - origin
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    u = axis / norm
    cross = lambda v: u[0] * v[1] - u[1] * v[0]
    co_ids, co_anchor = mu_co
    if cross(P[large_anchor] - origin) * cross(P[co_anchor] - origin) <= 0:
        return out
    v = P[co_ids] - origin
    P[co_ids] = origin + 2.0 * np.outer(v @ u, u) - v

    out.RemoveAllConformers()
    c = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y) in enumerate(P):
        c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    c.Set3D(False)
    out.AddConformer(c)
    return out


def orient_metal_bound_carbonyls_outward(mol, pair):
    """Turn terminal C=O oxygen away from a bimetallic core.

    A large anchored organic bridge is moved as a rigid body by several cluster
    polishers.  Its metal atoms are not part of that body, so a perfectly good
    internal O=C--C angle can nevertheless finish with O pointing back along a
    metal--C bond.  Correct only the terminal oxygen, choosing between the two
    trigonal-planar directions around the carbonyl carbon.  Ordinary ketones,
    terminal metal carbonyls and mu-CO (which have no organic neighbour at the
    carbonyl carbon) are deliberately excluded.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) < 1 or out.GetNumConformers() == 0:
        return out
    mset = set(metals)
    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    changed = False

    for carbon in out.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        ci = carbon.GetIdx()
        oxygen = []
        metal_nbrs = []
        organic = []
        for bond in carbon.GetBonds():
            ni = bond.GetOtherAtomIdx(ci)
            atom = out.GetAtomWithIdx(ni)
            if (atom.GetAtomicNum() == 8 and
                    bond.GetBondType() == Chem.BondType.DOUBLE and
                    atom.GetDegree() == 1):
                oxygen.append(ni)
            elif ni in mset:
                metal_nbrs.append(ni)
            elif atom.GetAtomicNum() not in METALS:
                organic.append(ni)
        if len(oxygen) != 1 or not metal_nbrs or len(organic) != 1:
            continue

        oi, ai = oxygen[0], organic[0]
        axis = P[ai] - P[ci]
        norm = np.linalg.norm(axis)
        length = np.linalg.norm(P[oi] - P[ci])
        if norm < 1e-9 or length < 1e-9:
            continue
        axis /= norm
        candidates = [P[ci] + length * (_rot(turn) @ axis)
                      for turn in (-120.0, 120.0)]

        # Prefer the side with the greatest clearance from the complete metal
        # core.  Remaining atoms provide a deterministic tie-breaker and avoid
        # turning the oxygen into a neighbouring bond or label.
        others = [i for i in range(out.GetNumAtoms())
                  if i not in (ci, oi)]
        def clearance(q):
            metal_clear = min(np.linalg.norm(q - P[m]) for m in metals)
            atom_clear = min(np.linalg.norm(q - P[i]) for i in others)
            return (metal_clear, atom_clear)
        chosen = max(candidates, key=clearance)
        if np.linalg.norm(chosen - P[oi]) > 1e-8:
            P[oi] = chosen
            changed = True

    if not changed:
        return out
    out.RemoveAllConformers()
    c = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y) in enumerate(P):
        c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    c.Set3D(False)
    out.AddConformer(c)
    return out


def centre_shared_organic_bridge(mol, pair, step=5.0):
    """Seat the common carbon of a large Fe2 bridge above the Fe-Fe midpoint.

    In Cp2Fe2(mu-CO)(C3/N) motifs a least-squares multi-donor fit may pull the
    shared C2 atom far to one side.  The conventional and clearer projection
    keeps C2 on the perpendicular bisector, opposite mu-CO, then swings the
    intact organic bridge about it.  Apply only when two terminal eta ligands
    identify this topology, and never accept more crossings/overlaps.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    large = None
    mu_co_anchor = None
    haptic = 0
    for amap0 in maps:
        amap = list(amap0)
        donors = [d for d in amap
                  if any(out.GetBondBetweenAtoms(m, d) is not None
                         for m in metals)]
        shared = [d for d in donors
                  if all(out.GetBondBetweenAtoms(m, d) is not None
                         for m in metals)]
        bm = {m for m in metals for d in donors
              if out.GetBondBetweenAtoms(m, d) is not None}
        if len(amap) >= 6 and len(shared) == 1 and len(donors) >= 2:
            large = (np.asarray(amap, dtype=int), shared[0])
        if len(amap) == 2 and len(shared) == 1:
            nums = {out.GetAtomWithIdx(d).GetAtomicNum() for d in amap}
            if nums == {6, 8}:
                mu_co_anchor = shared[0]
        if len(donors) >= 3 and len(bm) == 1:
            haptic += 1
    if large is None or mu_co_anchor is None or haptic != 2:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    m0, m1 = metals
    midpoint = (P[m0] + P[m1]) / 2.0
    axis = P[m1] - P[m0]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    u = axis / norm
    normal = np.array([-u[1], u[0]])
    if (P[mu_co_anchor] - midpoint) @ normal > 0:
        normal = -normal
    target = midpoint + np.sqrt(max(ML * ML - (norm / 2.0) ** 2,
                                    0.25 * ML * ML)) * normal
    ids, anchor = large

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    base = quality(P)
    seed = P.copy()
    seed[ids] += target - seed[anchor]
    best = None
    for angle in np.arange(-180.0, 180.0 + 1e-9, step):
        cand = seed.copy()
        cand[ids] = ((seed[ids] - target) @ _rot(angle).T + target)
        trial = (quality(cand), abs(float(angle)), cand)
        if best is None or trial[:2] < best[:2]:
            best = trial
    if best is not None and best[0] <= base:
        return with_coords(best[2])
    return out


def orient_two_end_organic_bridge(mol, pair):
    """Put a rigid two-ended organic bridge above an M--M/Cp2 core.

    In ``Cp2M2(mu-CO)(C...C)`` motifs the large bridge has one terminal donor
    at each metal, rather than one atom shared by both centres.  The ordinary
    cluster fit can therefore lay the bridge diagonally through M--M.  Rotate
    the complete ligand so its donor chord is parallel to M--M, search only its
    height, and keep the shared CO in the opposite half-plane.  Internal ligand
    coordinates and both Cp fragments are never changed.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out

    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    bridge = None
    mu_co = None
    haptic_metals = []
    for amap0 in maps:
        amap = list(amap0)
        bound = [(i, m) for i in amap for m in metals
                 if out.GetBondBetweenAtoms(i, m) is not None]
        donor_atoms = {i for i, _m in bound}
        bound_metals = {m for _i, m in bound}
        if (len(amap) >= 6 and len(donor_atoms) == 2 and
                bound_metals == set(metals) and
                all(sum(1 for _i, m in bound if m == metal) == 1
                    for metal in metals)):
            bridge = (np.asarray(amap, dtype=int), bound)
        if (len(amap) == 2 and bound_metals == set(metals) and
                len(donor_atoms) == 1 and
                {out.GetAtomWithIdx(i).GetAtomicNum() for i in amap} == {6, 8}):
            mu_co = np.asarray(amap, dtype=int)
        if len(amap) >= 3 and len(donor_atoms) >= 3 and len(bound_metals) == 1:
            haptic_metals.append(next(iter(bound_metals)))
    if (bridge is None or mu_co is None or len(haptic_metals) != 2 or
            set(haptic_metals) != set(metals)):
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    m0, m1 = metals
    axis = P[m1] - P[m0]
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return out
    ex = axis / length
    ey = np.array([-ex[1], ex[0]])
    midpoint = (P[m0] + P[m1]) / 2.0
    ids, bound = bridge
    donor_for = {metal: atom for atom, metal in bound}
    d0, d1 = donor_for[m0], donor_for[m1]
    old = P[d1] - P[d0]
    separation = float(np.linalg.norm(old))
    if separation < 1e-9:
        return out
    rotation = _rot(np.degrees(np.arctan2(ex[1], ex[0]) -
                               np.arctan2(old[1], old[0])))
    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    base = quality(P)
    best = None
    # Fixed one-dimensional search; 44 candidates at the default bond scale.
    for side in (-1.0, 1.0):
        for height in np.arange(0.5 * core.LB, 2.61 * core.LB,
                                0.1 * core.LB):
            target0 = midpoint - 0.5 * separation * ex + side * height * ey
            cand = P.copy()
            cand[ids] = (P[ids] - P[d0]) @ rotation.T + target0
            # The mu-CO and organic bridge conventionally occupy opposite
            # half-planes. Reflecting the complete C-O pair preserves geometry.
            co_centre = cand[mu_co].mean(axis=0)
            if np.dot(co_centre - midpoint, side * ey) > 0:
                v = cand[mu_co] - midpoint
                cand[mu_co] = midpoint + 2.0 * np.outer(v @ ex, ex) - v
            trial = (quality(cand), abs(float(height - 1.2 * core.LB)), cand)
            if best is None or trial[:2] < best[:2]:
                best = trial
    if best is None or best[0] > base:
        return out

    out.RemoveAllConformers()
    c = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y) in enumerate(best[2]):
        c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    c.Set3D(False)
    out.AddConformer(c)
    # Crossing-only ranking can exchange a clean near miss for several bonds
    # squeezed into one narrow corridor.  Never accept that visual regression.
    if _tight_bonds(core._drawing_mol(out)) > _tight_bonds(core._drawing_mol(mol)):
        return Chem.Mol(mol)
    return out


def symmetrize_haptic_pair(mol, pair, step=5.0):
    """Place two equivalent terminal eta-bound rings as a mirrored pair.

    Independent relaxation can leave identical Cp ligands at different radii
    and angles.  Move each complete ring rigidly so their centroids are mirror
    images across the perpendicular bisector of M-M.  No ring bond is changed.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in metals]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    hp = []
    for amap0 in maps:
        amap = list(amap0)
        bound = [(d, m) for d in amap for m in metals
                 if out.GetBondBetweenAtoms(d, m) is not None]
        donors = {d for d, _m in bound}
        bm = {m for _d, m in bound}
        if len(amap) >= 3 and len(donors) >= 3 and len(bm) == 1:
            hp.append((np.asarray(amap, dtype=int), next(iter(bm))))
    if len(hp) != 2 or {m for _ids, m in hp} != set(metals):
        return out
    hp.sort(key=lambda x: metals.index(x[1]))
    # Only pair genuinely equivalent rings.
    sig = []
    for ids, _m in hp:
        sig.append(sorted(out.GetAtomWithIdx(int(i)).GetAtomicNum() for i in ids))
    if sig[0] != sig[1]:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    m0, m1 = metals
    axis = P[m1] - P[m0]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    ex = axis / norm
    ey = np.array([-ex[1], ex[0]])
    centres = [P[ids].mean(axis=0) for ids, _m in hp]
    radii = [np.linalg.norm(centres[i] - P[hp[i][1]]) for i in range(2)]
    radius = float(np.mean(radii))
    side = np.sign(np.mean([(c - (P[m0] + P[m1]) / 2.0) @ ey
                            for c in centres])) or -1.0

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    coordinate_score = _coordinate_scorer(out)

    def quality(xy):
        return (coordinate_score(xy, strain=False),
                coordinate_score(xy, strain=True))

    base_quality = quality(P)
    best = None
    for deg in np.arange(15.0, 166.0, step):
        a = np.radians(deg)
        # Outward x components are opposite; y components share one side.
        desired = [P[m0] + radius * (-np.cos(a) * ex + side * np.sin(a) * ey),
                   P[m1] + radius * ( np.cos(a) * ex + side * np.sin(a) * ey)]
        cand = P.copy()
        for i, (ids, metal) in enumerate(hp):
            old = centres[i] - P[metal]
            new = desired[i] - P[metal]
            rot = np.degrees(np.arctan2(new[1], new[0]) -
                             np.arctan2(old[1], old[0]))
            cand[ids] = ((P[ids] - centres[i]) @ _rot(rot).T + desired[i])
        trial = (quality(cand), abs(deg - 60.0), cand)
        if best is None or trial[:2] < best[:2]:
            best = trial
    if best is not None and best[0] <= base_quality:
        return with_coords(best[2])
    return out


def symmetrize_equivalent_terminal_pairs(mol, pair):
    """Mirror equivalent terminal Cp/eta rings and CO across an M--M core.

    Terminal fragments are initially assigned to each metal independently.
    Crossing minimisation alone therefore has no reason to choose a symmetric
    zero-crossing solution over an asymmetric zero-crossing solution.  Pair
    only graph-identical, single-metal fragments in the two safe rigid classes
    used here: eta-bound rings and two-atom carbonyls.  Both possible reference
    sides are tried and a mirrored candidate is accepted only when drawing
    quality does not become worse.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    if len(metals) != 2 or out.GetNumConformers() == 0:
        return out
    mset = set(metals)
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()
           if b.GetOtherAtomIdx(m) not in mset]
    if not cut:
        return out
    maps = []
    pieces = Chem.GetMolFrags(Chem.FragmentOnBonds(
        out, cut, addDummies=False), asMols=True, sanitizeFrags=False,
        fragsMolAtomMapping=maps)

    groups = {}
    for piece, amap0 in zip(pieces, maps):
        amap = list(amap0)
        bound = [(d, m) for d in amap for m in metals
                 if out.GetBondBetweenAtoms(d, m) is not None]
        bm = {m for _d, m in bound}
        if len(bm) != 1:
            continue
        donors = {d for d, _m in bound}
        nums = sorted(out.GetAtomWithIdx(i).GetAtomicNum() for i in amap)
        if len(amap) >= 3 and len(donors) >= 3:
            kind = "eta"
        elif len(amap) == 2 and nums == [6, 8] and any(
                b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                for b in piece.GetBonds()):
            kind = "co"
        else:
            continue
        try:
            identity = Chem.MolToSmiles(piece, canonical=True,
                                        isomericSmiles=True)
        except Exception:
            continue
        groups.setdefault((kind, identity), []).append(
            (piece, np.asarray(amap, dtype=int), next(iter(bm))))

    pairs = []
    for entries in groups.values():
        if len(entries) != 2 or {x[2] for x in entries} != mset:
            continue
        entries.sort(key=lambda x: metals.index(x[2]))
        pairs.append(entries)
    if not pairs:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])
    midpoint = (P[metals[0]] + P[metals[1]]) / 2.0
    axis = P[metals[1]] - P[metals[0]]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    ex = axis / norm

    def reflected(xy):
        v = xy - midpoint
        return xy - 2.0 * np.outer(v @ ex, ex)

    scorer = _coordinate_scorer(out)
    def quality(xy):
        return (scorer(xy, strain=False), scorer(xy, strain=True))

    for entries in pairs:
        base = quality(P)
        candidates = []
        for source_idx in (0, 1):
            target_idx = 1 - source_idx
            source_piece, source_ids, _ = entries[source_idx]
            target_piece, target_ids, _ = entries[target_idx]
            matches = target_piece.GetSubstructMatches(
                source_piece, uniquify=False, maxMatches=100)
            for match in matches:
                if len(match) != len(source_ids):
                    continue
                cand = P.copy()
                mirrored = reflected(P[source_ids])
                for local, target_local in enumerate(match):
                    cand[target_ids[target_local]] = mirrored[local]
                displacement = float(np.sum(
                    (cand[target_ids] - P[target_ids]) ** 2))
                candidates.append((quality(cand), displacement, cand))
        if candidates:
            # Rigid reflection preserves every internal bond and the terminal
            # metal distance.  The strain score also contains unrelated long
            # closing coordination lines, so use it only diagnostically here:
            # symmetry is preferred whenever actual drawing tangles do not
            # increase.
            best = min(candidates, key=lambda x: (x[0][0], x[1]))
            if best[0][0] <= base[0]:
                P = best[2]

    result = Chem.Mol(out)
    result.RemoveAllConformers()
    c = Chem.Conformer(result.GetNumAtoms())
    for i, (x, y) in enumerate(P):
        c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    c.Set3D(False)
    result.AddConformer(c)
    return result


def polish_terminal_sectors(mol, pair, sweeps=3, step=10.0):
    """Move bulky single-donor terminals between sectors around each metal.

    The fragment is rotated as one rigid body about its metal, so the
    metal-donor distance and every internal ligand coordinate are preserved.
    Bridges and multidentate chelates are deliberately excluded.
    """
    out = Chem.Mol(mol)
    metals = list(pair)
    cut = [b.GetIdx() for m in metals for b in out.GetAtomWithIdx(m).GetBonds()]
    maps = []
    frag = Chem.FragmentOnBonds(out, cut, addDummies=False)
    Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False,
                     fragsMolAtomMapping=maps)
    pieces = []
    for amap0 in maps:
        amap = list(amap0)
        if len(amap) < 3:
            continue
        bound = [(d, m) for d in amap for m in metals
                 if out.GetBondBetweenAtoms(d, m) is not None]
        if len({d for d, _m in bound}) != 1 or len({m for _d, m in bound}) != 1:
            continue
        pieces.append((np.asarray(amap, dtype=int), bound[0][1]))
    if not pieces or out.GetNumConformers() == 0:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    angles = np.arange(-180.0, 180.0 + 1e-9, step)
    coordinate_score = _plain_coordinate_scorer(out)
    for _ in range(sweeps):
        changed = False
        for ids, metal in pieces:
            base_score = coordinate_score(P, strain=False)
            best = (base_score, 0.0, P)
            for angle in angles:
                cand = P.copy()
                cand[ids] = ((cand[ids] - cand[metal]) @ _rot(angle).T
                             + cand[metal])
                trial = (coordinate_score(cand, strain=False),
                         abs(float(angle)), cand)
                if trial[:2] < best[:2]:
                    best = trial
            if best[1] > 1e-9:
                changed = True
            P = best[2]
        if not changed:
            break
    return with_coords(P)


def polish_terminal_atoms(mol, pair, sweeps=3, step=10.0):
    """Fan monoatomic terminal ligands into free sectors around each metal.

    Plain 2D generators may put e.g. terminal OH and Cl on the same point in a
    macrocyclic dimer.  Each atom is moved only on a circle about its own metal,
    preserving the M-X bond length; bridges and atoms with any other neighbour
    are excluded.
    """
    out = Chem.Mol(mol)
    if out.GetNumConformers() == 0:
        return out
    metals = list(pair)
    terminals = []
    for metal in metals:
        for atom in out.GetAtomWithIdx(metal).GetNeighbors():
            if atom.GetAtomicNum() != 1 and atom.GetDegree() == 1:
                terminals.append((atom.GetIdx(), metal))
    if not terminals:
        return out

    conf = out.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(out.GetNumAtoms())])

    def with_coords(xy):
        cand = Chem.Mol(out)
        cand.RemoveAllConformers()
        c = Chem.Conformer(cand.GetNumAtoms())
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        cand.AddConformer(c)
        return cand

    angles = np.arange(-180.0, 180.0 + 1e-9, step)
    coordinate_score = _plain_coordinate_scorer(out)
    for _ in range(sweeps):
        changed = False
        for atom, metal in terminals:
            radius = float(np.linalg.norm(P[atom] - P[metal]))
            if radius < 1e-9:
                continue
            fixed = [i for i in range(out.GetNumAtoms())
                     if i not in (atom, metal)]

            def quality(xy):
                legibility = coordinate_score(xy, strain=False)
                clearance = float(np.linalg.norm(
                    xy[atom] - xy[fixed], axis=1).min()) if fixed else 99.0
                return legibility, -clearance

            best = (quality(P), 0.0, P)
            base = P[atom] - P[metal]
            for angle in angles:
                cand = P.copy()
                cand[atom] = P[metal] + base @ _rot(angle).T
                trial = (quality(cand), abs(float(angle)), cand)
                if trial[:2] < best[:2]:
                    best = trial
            if best[1] > 1e-9:
                P = best[2]
                changed = True
        if not changed:
            break
    return with_coords(P)


def depict_macrocycle(mol, pair):
    """Lay out a macrocyclic dimer by pinning its two metals apart.

    Neither generator manages this molecule on its own - CoordGen puts the two
    metals almost on the same point - but the classic depictor will build the
    ring correctly once told where the metals go. The separation is not known in
    advance, so a few are tried and the least tangled one kept.
    """
    from rdkit.Chem import rdDepictor
    from rdkit.Geometry import Point2D

    m0, m1 = pair
    best, best_score = None, None
    # When a macrocycle also contains a two-atom M-X-M-X core, putting the metal
    # labels only one bond length apart makes an otherwise crossing-free drawing
    # unreadable: both bridge labels and the terminal donors have to share the
    # same tiny patch.  Such a core needs two bond lengths of horizontal room.
    # Plain two-bridge diamonds are laid out by the cluster template; this path
    # is reached when whole bridging ligands close an additional macrocycle.
    shared_atoms = [a.GetIdx() for a in mol.GetAtoms()
                    if a.GetIdx() not in pair
                    and mol.GetBondBetweenAtoms(a.GetIdx(), m0) is not None
                    and mol.GetBondBetweenAtoms(a.GetIdx(), m1) is not None]
    min_k = 2.0 if len(shared_atoms) >= 2 else 1.0
    # The span has to cover both kinds of dimer this path sees: a macrocycle
    # holds its metals many bond lengths apart, while metals sharing a bridging
    # donor atom sit almost on top of each other. Searching only the wide end
    # leaves the edge-sharing ones stretched.
    for k in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        if k < min_k:
            continue
        cand = Chem.Mol(mol)
        try:
            rdDepictor.Compute2DCoords(
                cand, coordMap={m0: Point2D(0.0, 0.0),
                                m1: Point2D(k * ML, 0.0)})
        except Exception:
            continue
        try:
            sc = _tangle(cand)
        except Exception:
            continue
        if best_score is None or sc < best_score:
            best, best_score = cand, sc
    if best is None:
        return None

    # Resolve local collisions between monoatomic terminal ligands before any
    # graph symmetry is imposed.  Symmetrisation below can then copy the chosen
    # sectors consistently to the equivalent half of the complex.
    best = polish_terminal_atoms(best, pair)

    # CoordGen is not symmetry-aware when a large ring is pinned at two atoms:
    # graph-equivalent halves can therefore acquire visibly different bends.
    # If the molecular graph has an automorphism exchanging the two metals,
    # copy the half on one side through the core centre by 180 degrees.  There
    # can be several atom mappings (not all preserve the useful ring
    # orientation), so retain a copy only when the global legibility score does
    # not get worse, and let bond strain break ties.
    conf = best.GetConformer()
    base_xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                        for i in range(best.GetNumAtoms())])
    centre = (base_xy[m0] + base_xy[m1]) / 2.0
    axis = base_xy[m1] - base_xy[m0]
    axis /= max(float(np.linalg.norm(axis)), 1e-12)

    def with_xy(xy):
        cand = Chem.Mol(best)
        c = cand.GetConformer()
        for i, (x, y) in enumerate(xy):
            c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
        c.Set3D(False)
        return cand

    # Components obtained after cutting coordination bonds are the rigid units
    # that may be exchanged by symmetry.  Choosing the source per atom is not
    # safe: a long substituent can cross the M-M bisector, causing alternate
    # atoms of one benzene ring to be copied from opposite halves.
    metal_bonds = sorted({b.GetIdx() for m in pair
                          for b in mol.GetAtomWithIdx(m).GetBonds()})
    if metal_bonds:
        cut = Chem.FragmentOnBonds(mol, metal_bonds, addDummies=False)
        components = [tuple(f) for f in Chem.GetMolFrags(cut)]
    else:
        components = [tuple(range(mol.GetNumAtoms()))]
    component_of = {i: k for k, comp in enumerate(components) for i in comp}

    symmetric = []
    for amap in mol.GetSubstructMatches(mol, uniquify=False, maxMatches=256):
        if amap[m0] != m1 or amap[m1] != m0:
            continue
        xy = base_xy.copy()
        done_components = set()
        for ci, comp in enumerate(components):
            if ci in done_components:
                continue
            mapped_components = {component_of[amap[i]] for i in comp}
            if len(mapped_components) != 1:
                continue
            cj = mapped_components.pop()
            done_components.update((ci, cj))
            if ci == cj:
                continue
            other = components[cj]
            pi = float(np.dot(base_xy[list(comp)].mean(axis=0) - centre, axis))
            pj = float(np.dot(base_xy[list(other)].mean(axis=0) - centre, axis))
            source = comp if pi <= pj else other
            # Copy the entire connected fragment through the C2 centre.  Atom
            # mapping is used only after the source fragment has been chosen.
            for i in source:
                xy[amap[i]] = 2.0 * centre - base_xy[i]
        cand = with_xy(xy)
        symmetric.append(cand)
    if symmetric:
        base_leg = _tangle(best, strain=False)
        admissible = [c for c in symmetric
                      if _tangle(c, strain=False) <= base_leg]
        if admissible:
            best = min(admissible, key=lambda c: _tangle(c))

    # A terminal monoatomic ligand has no shape for the organic packing code to
    # orient, so CoordGen often leaves it pointing into the bridge core.  Move
    # equivalent terminal atoms as a C2-related pair into the clearer of the
    # two side sectors.  The search is deliberately limited to directions
    # roughly perpendicular to the M-M axis: the inward sector belongs to the
    # bridges and the outward sector is normally enclosed by the chelate.
    terminals = {}
    for m in pair:
        terminals[m] = [n.GetIdx() for n in best.GetAtomWithIdx(m).GetNeighbors()
                        if n.GetDegree() == 1 and n.GetAtomicNum() != 1]
    if len(terminals[m0]) == len(terminals[m1]) == 1:
        t0, t1 = terminals[m0][0], terminals[m1][0]
        if best.GetAtomWithIdx(t0).GetAtomicNum() == best.GetAtomWithIdx(t1).GetAtomicNum():
            conf = best.GetConformer()
            xy = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y]
                           for i in range(best.GetNumAtoms())])
            centre = (xy[m0] + xy[m1]) / 2.0
            ex = xy[m1] - xy[m0]
            ex /= max(float(np.linalg.norm(ex)), 1e-12)
            ey = np.array([-ex[1], ex[0]])
            lengths = [np.linalg.norm(xy[b.GetBeginAtomIdx()] -
                                      xy[b.GetEndAtomIdx()])
                       for b in best.GetBonds()]
            bond_len = float(np.median(lengths))
            fixed = [i for i in range(best.GetNumAtoms())
                     if i not in (m0, m1, t0, t1)]
            base_leg = _tangle(best, strain=False)
            choices = []
            for angle in np.arange(-135.0, 135.0 + 1e-9, 5.0):
                if abs(angle) < 45.0:
                    continue
                v = bond_len * (np.cos(np.radians(angle)) * ex +
                                np.sin(np.radians(angle)) * ey)
                trial = xy.copy()
                trial[t0] = trial[m0] + v
                trial[t1] = trial[m1] - v
                cand = with_xy(trial)
                leg = _tangle(cand, strain=False)
                if leg > base_leg:
                    continue
                clearance = min(
                    float(np.linalg.norm(trial[t0] - trial[fixed], axis=1).min()),
                    float(np.linalg.norm(trial[t1] - trial[fixed], axis=1).min()))
                choices.append((leg, -clearance, abs(float(angle)), cand))
            if choices:
                best = min(choices, key=lambda x: x[:3])[3]

    # Two shared monoatomic bridges form the small M-X-M-X core.  A graph
    # automorphism is allowed to leave each X atom fixed while exchanging the
    # metals, but geometrically that produces a lopsided diamond.  Put the two
    # equivalent atoms on opposite sides of the perpendicular bisector.
    shared = [a.GetIdx() for a in best.GetAtoms()
              if a.GetIdx() not in pair
              and best.GetBondBetweenAtoms(a.GetIdx(), m0) is not None
              and best.GetBondBetweenAtoms(a.GetIdx(), m1) is not None]
    if (len(shared) == 2 and
            best.GetAtomWithIdx(shared[0]).GetAtomicNum() ==
            best.GetAtomWithIdx(shared[1]).GetAtomicNum()):
        conf = best.GetConformer()
        xy = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y]
                       for i in range(best.GetNumAtoms())])
        centre = (xy[m0] + xy[m1]) / 2.0
        ex = xy[m1] - xy[m0]
        ex /= max(float(np.linalg.norm(ex)), 1e-12)
        ey = np.array([-ex[1], ex[0]])
        lengths = [np.linalg.norm(xy[b.GetBeginAtomIdx()] -
                                  xy[b.GetEndAtomIdx()])
                   for b in best.GetBonds()]
        height = 0.70 * float(np.median(lengths))
        # Preserve which bridge was above the axis, only regularise its partner.
        first, second = sorted(shared,
                               key=lambda i: np.dot(xy[i] - centre, ey),
                               reverse=True)
        xy[first] = centre + height * ey
        xy[second] = centre - height * ey
        cand = with_xy(xy)
        if _tangle(cand, strain=False) <= _tangle(best, strain=False):
            best = cand

    conf = best.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(best.GetNumAtoms())])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in best.GetBonds()]
    unit = float(np.median([np.linalg.norm(xy[i] - xy[j]) for i, j in bonds]))
    if unit > 1e-9:
        xy = xy * (LB / unit)
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    conf.Set3D(False)
    return best
