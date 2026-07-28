"""
metrics.py - measure how readable a 2D depiction of a metal complex is.

Use it to compare depiction engines, to check for regressions after changing
metal2d, or to find the worst structures in your own dataset.

    python metrics.py complexes.csv                  # compare all engines
    python metrics.py complexes.csv --engine metal2d # just one
    python metrics.py file.sdf --step 10 --worst 20  # sample, list worst cases

As a library:

    from metrics import score, evaluate
    score(mol_with_2d_coords)      -> dict of metrics for one depiction
    evaluate('data.smi')           -> per-structure rows + printed summary

Everything is normalised by the median bond length, so the numbers do not
depend on the drawing scale and are comparable across engines.
"""
import sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdCoordGen, rdDepictor

RDLogger.DisableLog("rdApp.*")

try:
    import metal2d
except ImportError:                                   # metrics can run without it
    metal2d = None

try:                                    # keep one definition of what a metal is
    from metal2d import METALS
except ImportError:
    METALS = (set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))
              | {13, 31, 49, 50, 51, 81, 82, 83, 84} | set(range(89, 104)))


# --------------------------------------------------------------------------- #
#  geometry helpers
# --------------------------------------------------------------------------- #
def _find_metal(mol):
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in METALS:
            return a.GetIdx()
    return None


def _xy(mol):
    c = mol.GetConformer()
    return np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                     for i in range(mol.GetNumAtoms())])


def _seg_seg_distance(p, q, a, b):
    """Smallest distance between segments pq and ab (endpoint approximation,
    exact whenever the segments do not cross, which is the case we care about)."""
    best = np.inf
    for X, A, B in ((p, a, b), (q, a, b), (a, p, q), (b, p, q)):
        D = B - A
        t = np.clip(np.dot(X - A, D) / max(np.dot(D, D), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(X - (A + t * D))))
    return best


# --------------------------------------------------------------------------- #
#  the metrics
# --------------------------------------------------------------------------- #
def _haptic_pairs(mol):
    """Bond-index pairs whose crossing is an artefact of drawing an eta-bonded
    group as one line to the ring centre: that line must cross the ring it
    points at. Crossings with anything else are still genuine errors."""
    forgive = set()
    for a in mol.GetAtoms():
        if not a.HasProp("_hapticAtoms"):
            continue
        grp = set(int(x) for x in a.GetProp("_hapticAtoms").split(","))
        di = a.GetIdx()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if di in (i, j):                       # the metal-to-centre bond
                for o in mol.GetBonds():
                    oi, oj = o.GetBeginAtomIdx(), o.GetEndAtomIdx()
                    if oi in grp and oj in grp:    # a bond of that same ring
                        forgive.add((min(b.GetIdx(), o.GetIdx()),
                                     max(b.GetIdx(), o.GetIdx())))
    return forgive


def score(mol, close_frac=0.35, tight_frac=0.30):
    """Metrics for a single molecule that already carries a 2D conformer.

    crossings   bond pairs that intersect. The single most telling number:
                a drawing with zero crossings is almost always readable.
    overlaps    atom pairs sitting on top of each other. Worse than a crossing,
                because the drawing is then not just ugly but wrong.
    tight       pairs of unrelated bonds closer than `tight_frac` bond lengths.
                Catches near misses that crossings alone do not.
    on_metal    non-donor atoms drawn on top of the metal centre.
    flipped     donor atoms whose own ring lies between them and the metal,
                i.e. the nitrogen is turned away. Fraction of all ring donors.
    clean       True when there are no crossings and no overlaps.
    """
    out = dict(atoms=mol.GetNumAtoms(), crossings=0, overlaps=0, tight=0,
               on_metal=0, flipped=0, donors=0, clean=True, bond_len=0.0,
               stretch=0.0)
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return out

    P = _xy(mol)
    segs = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()])
    A, B = P[segs[:, 0]], P[segs[:, 1]]
    bl = float(np.median(np.linalg.norm(A - B, axis=1)))
    if bl < 1e-9:
        return out
    out["bond_len"] = bl

    # --- atom overlaps ----------------------------------------------------- #
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1) + np.eye(len(P)) * 1e9
    out["overlaps"] = int((D < close_frac * bl).sum() // 2)

    # --- bond crossings and near misses ------------------------------------ #
    d1 = B - A
    forgive = _haptic_pairs(mol)
    crossings = tight = 0
    for i in range(len(segs)):
        share = ((segs[i, 0] == segs[:, 0]) | (segs[i, 0] == segs[:, 1]) |
                 (segs[i, 1] == segs[:, 0]) | (segs[i, 1] == segs[:, 1]))
        share[:i + 1] = True
        idx = np.where(~share)[0]
        if len(idx) == 0:
            continue
        p, r = A[i], d1[i]
        q, s = A[idx], d1[idx]
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
            u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / rxs
        hit = np.isfinite(t) & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
        for j in idx[hit]:
            if (i, int(j)) not in forgive:
                crossings += 1
        for j in idx[~hit]:
            if (i, int(j)) in forgive:
                continue
            if _seg_seg_distance(A[i], B[i], A[j], B[j]) < tight_frac * bl:
                tight += 1
    out["crossings"], out["tight"] = crossings, tight

    # --- metal specific ----------------------------------------------------- #
    mi = _find_metal(mol)
    if mi is not None:
        don = set(b.GetOtherAtomIdx(mi) for b in mol.GetAtomWithIdx(mi).GetBonds())
        # eta-bonded ring carbons legitimately sit close to the metal
        hap = set()
        for a in list(don):
            if mol.GetAtomWithIdx(a).IsInRing():
                for r in mol.GetRingInfo().AtomRings():
                    if len(set(r) & don) >= 3:
                        hap |= set(r)
        r = np.linalg.norm(P - P[mi], axis=1)
        out["on_metal"] = int(sum(1 for i in range(mol.GetNumAtoms())
                                  if i != mi and i not in don and i not in hap
                                  and r[i] < 0.7 * bl))
        # longest metal-donor bond relative to a normal bond: catches ligands
        # shoved outward to resolve a clash, which no crossing count will show
        if don:
            out["stretch"] = float(max(np.linalg.norm(P[d] - P[mi]) for d in don) / bl)
        rings = mol.GetRingInfo().AtomRings()
        for d in don:
            own = [x for x in rings if d in x]
            if not own:
                continue
            out["donors"] += 1
            cen = P[list(own[0])].mean(0)
            if np.dot(P[mi] - P[d], cen - P[d]) > 0:
                out["flipped"] += 1

    out["clean"] = out["crossings"] == 0 and out["overlaps"] == 0
    return out


# --------------------------------------------------------------------------- #
#  depiction engines
# --------------------------------------------------------------------------- #
def _legacy(mol):
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(m)
    return m


def _coordgen(mol):
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    rdCoordGen.AddCoords(m)
    return m


def _metal2d(mol):
    g = metal2d.depict(mol)
    # measure what is actually drawn: haptic groups collapsed, dative bonds plain
    try:
        return metal2d._drawing_mol(g)
    except Exception:
        return g


ENGINES = {"compute2dcoords": _legacy, "coordgen": _coordgen, "metal2d": _metal2d}


# --------------------------------------------------------------------------- #
#  running over a file
# --------------------------------------------------------------------------- #
def read_molecules(src, column=None):
    """(name, Mol) from .sdf/.mol, .smi/.txt/.csv/.tsv, or a bare SMILES."""
    import os
    ext = os.path.splitext(str(src))[1].lower()

    if ext in (".sdf", ".mol"):
        for i, m in enumerate(Chem.SDMolSupplier(src, removeHs=False)):
            name = m.GetProp("_Name") if (m is not None and m.HasProp("_Name")) else ""
            yield name or "mol_%d" % i, m
        return

    if ext in (".smi", ".smiles", ".txt", ".csv", ".tsv"):
        import csv as _csv
        with open(src, newline="") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            if ext in (".csv", ".tsv") or "," in sample.split("\n")[0]:
                rdr = _csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
                cols = [c for c in (rdr.fieldnames or []) if "smiles" in c.lower()]
                if not cols:
                    raise SystemExit("no column with 'smiles' in its name found")
                if column:
                    col = column
                else:
                    # a file may hold several SMILES columns (ligands, complex,
                    # canonical...). The complex is the one we can score.
                    col = next((c for c in cols if "complex" in c.lower()),
                               next((c for c in cols if c.lower() == "smiles"),
                                    cols[0]))
                    if len(cols) > 1:
                        print("[using column %r of %r]" % (col, cols))
                namecol = next((c for c in rdr.fieldnames
                                if c.lower() in ("name", "id", "title")), None)
                for i, row in enumerate(rdr):
                    smi = (row.get(col) or "").strip()
                    if not smi:
                        continue
                    yield (row.get(namecol) or "mol_%d" % i), Chem.MolFromSmiles(smi)
                return
            for i, line in enumerate(fh):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace("\t", " ").split()
                if i == 0 and parts[0].lower() in ("smiles", "smi"):
                    continue
                yield (" ".join(parts[1:]) or "mol_%d" % i), Chem.MolFromSmiles(parts[0])
        return

    yield "mol_0", Chem.MolFromSmiles(str(src))


def count_molecules(src):
    """Cheap upfront count so the progress bar can show a total and an ETA.
    Returns None when the size cannot be established without a full parse."""
    import os
    ext = os.path.splitext(str(src))[1].lower()
    try:
        if ext in (".sdf", ".mol"):
            n = 0
            with open(src, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    n += chunk.count(b"$$$$")
            return n or None
        if ext in (".smi", ".smiles", ".txt", ".csv", ".tsv"):
            with open(src, "rb") as fh:
                n = sum(1 for line in fh if line.strip())
            return max(n - 1, 1)          # assume one header line
    except OSError:
        return None
    return 1                              # a bare SMILES string


class Progress:
    """Minimal stderr progress bar. Uses tqdm when it is installed, otherwise
    falls back to a plain bar so metrics.py keeps zero required dependencies."""

    def __init__(self, total=None, desc="scoring", enabled=True):
        self.enabled = enabled and sys.stderr.isatty()
        self.total, self.desc, self.n = total, desc, 0
        self.t0 = self.last = __import__("time").time()
        self.bar = None
        if self.enabled:
            try:
                from tqdm import tqdm
                self.bar = tqdm(total=total, desc=desc, unit="mol",
                                file=sys.stderr, leave=False)
            except ImportError:
                self.bar = None

    def update(self, k=1):
        self.n += k
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.update(k)
            return
        import time
        now = time.time()
        if now - self.last < 0.1 and (self.total is None or self.n != self.total):
            return
        self.last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0
        if self.total:
            frac = min(self.n / self.total, 1.0)
            width = 30
            done = int(width * frac)
            eta = (self.total - self.n) / rate if rate > 0 else 0
            msg = "\r%s |%s%s| %d/%d  %5.1f%%  %4.1f mol/s  eta %s" % (
                self.desc, "#" * done, "-" * (width - done), self.n, self.total,
                100 * frac, rate, _hms(eta))
        else:
            msg = "\r%s %d mol  %4.1f mol/s  elapsed %s" % (
                self.desc, self.n, rate, _hms(el))
        sys.stderr.write(msg[:120])
        sys.stderr.flush()

    def close(self):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.close()
        else:
            sys.stderr.write("\r" + " " * 120 + "\r")
            sys.stderr.flush()


def _hms(sec):
    sec = int(max(sec, 0))
    if sec < 60:
        return "%ds" % sec
    if sec < 3600:
        return "%dm%02ds" % (sec // 60, sec % 60)
    return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)


def evaluate(src, engines=("coordgen", "metal2d"), step=1, verbose=True,
             column=None, progress=True):
    """Score every structure with every engine. Returns a list of dict rows."""
    if "metal2d" in engines and metal2d is None:
        raise SystemExit("metal2d.py must be importable to score the metal2d engine")

    total = count_molecules(src)
    if total and step > 1:
        total = total // step
    bar = Progress(total, "scoring", enabled=progress)

    rows, skipped, failed = [], 0, 0
    for i, (name, m) in enumerate(read_molecules(src, column)):
        if i % step:
            continue
        bar.update()
        if m is None:
            failed += 1
            continue
        if _find_metal(m) is None:
            skipped += 1
            continue
        row = {"name": name, "atoms": m.GetNumAtoms(),
               "metal": m.GetAtomWithIdx(_find_metal(m)).GetSymbol()}
        ok = True
        for eng in engines:
            try:
                s = score(ENGINES[eng](m))
            except Exception:
                ok = False
                break
            for k in ("crossings", "overlaps", "tight", "on_metal",
                      "flipped", "donors", "clean", "stretch"):
                row["%s_%s" % (eng, k)] = s[k]
        if ok:
            rows.append(row)
        else:
            failed += 1

    bar.close()
    if verbose:
        summarise(rows, engines, skipped, failed)
    return rows


def summarise(rows, engines, skipped=0, failed=0):
    if not rows:
        print("nothing to report")
        return
    n = len(rows)
    print("structures scored : %d" % n)
    if skipped:
        print("no metal centre   : %d" % skipped)
    if failed:
        print("unparsable/errored: %d" % failed)
    print()
    print("%-16s %10s %9s %8s %9s %8s %8s %7s" %
          ("engine", "crossings", "overlaps", "tight", "on metal", "flipped",
           "stretch", "clean"))
    for eng in engines:
        g = lambda k: np.array([r["%s_%s" % (eng, k)] for r in rows], dtype=float)
        don = g("donors").sum()
        print("%-16s %10.2f %9.2f %8.2f %9.2f %7.1f%% %8.2f %6.1f%%" %
              (eng, g("crossings").mean(), g("overlaps").mean(), g("tight").mean(),
               g("on_metal").mean(),
               100 * g("flipped").sum() / max(don, 1),
               g("stretch").mean(),
               100 * g("clean").mean()))

    if len(engines) > 1:
        a, b = engines[0], engines[-1]
        xa = np.array([r["%s_crossings" % a] for r in rows], dtype=float)
        xb = np.array([r["%s_crossings" % b] for r in rows], dtype=float)
        print("\n%s beats %s on %.1f%% of structures, ties on %.1f%%, loses on %.1f%%"
              % (b, a, 100 * (xb < xa).mean(), 100 * (xb == xa).mean(),
                 100 * (xb > xa).mean()))

    metals = sorted({r["metal"] for r in rows})
    if len(metals) > 1:
        print("\nby metal (n >= 10):")
        for me in metals:
            sub = [r for r in rows if r["metal"] == me]
            if len(sub) < 10:
                continue
            line = "  %-3s n=%5d " % (me, len(sub))
            for eng in engines:
                v = np.array([r["%s_crossings" % eng] for r in sub], dtype=float)
                line += "  %-8s %5.2f" % (eng[:8], v.mean())
            print(line)


def worst(rows, engine="metal2d", k=15):
    key = "%s_crossings" % engine
    bad = sorted(rows, key=lambda r: (-r[key], -r["%s_overlaps" % engine]))[:k]
    print("\nworst %d structures for %s:" % (len(bad), engine))
    print("  %-34s %6s %10s %9s" % ("name", "atoms", "crossings", "overlaps"))
    for r in bad:
        print("  %-34s %6d %10d %9d"
              % (str(r["name"])[:34], r["atoms"], r[key], r["%s_overlaps" % engine]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(0)
    src = args[0]

    def opt(flag, default=None, cast=str):
        return cast(args[args.index(flag) + 1]) if flag in args else default

    engines = tuple(opt("--engine", "coordgen,metal2d").split(","))
    rows = evaluate(src, engines=engines, step=opt("--step", 1, int),
                    column=opt("--column"), progress="--quiet" not in args)
    if "--worst" in args:
        worst(rows, engines[-1], opt("--worst", 15, int))
    if "--csv" in args:
        import csv as _csv
        path = opt("--csv", "metrics.csv")
        with open(path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nper-structure metrics written to %s" % path)
