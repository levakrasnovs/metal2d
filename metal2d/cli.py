"""Command line interface: metal2d draw | metrics | compare."""
import argparse
import os
import sys

from rdkit import RDLogger

from . import __version__
from .core import depict, draw, find_metal, read_molecules
from .compare import compare as compare_figure
from .metrics import evaluate, worst


def _add_common(p):
    p.add_argument("input", help="SMILES string, .smi/.csv/.tsv list, or .sdf")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress RDKit warnings and the progress bar")
    p.add_argument("--column", help="which column holds the SMILES "
                                    "(default: the one mentioning 'complex')")


def cmd_draw(a):
    RDLogger.DisableLog("rdApp.*") if a.quiet else None
    os.makedirs(a.outdir, exist_ok=True)
    fmt = "png" if a.png else "svg"
    size = (a.size, a.size)
    ok = bad = nometal = 0
    for i, (name, m) in enumerate(read_molecules(a.input, a.column)):
        if a.index and i not in a.index:
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:60]
        path = os.path.join(a.outdir, "%03d_%s.%s" % (i, safe, fmt))
        if m is None:
            print("%-40s PARSE FAILED" % path)
            bad += 1
            continue
        if find_metal(m) is None:
            nometal += 1
            print("%-40s no metal centre - plain CoordGen depiction" % path)
        draw(depict(m), path, size=size)
        print(path)
        ok += 1
    print("\n%d written, %d unparsable, %d without a metal centre"
          % (ok, bad, nometal))
    return 0


def cmd_metrics(a):
    engines = tuple(a.engines.split(","))
    rows = evaluate(a.input, engines=engines, step=a.step, column=a.column,
                    progress=not a.quiet)
    if a.worst:
        worst(rows, engines[-1], a.worst)
    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nper-structure metrics written to %s" % a.csv)
    return 0


def cmd_compare(a):
    engines = tuple(a.engines.split(","))
    if os.path.exists(a.input):
        items = [(n, m) for i, (n, m) in enumerate(read_molecules(a.input, a.column))
                 if (not a.index or i in a.index)]
    else:
        items = [("mol", a.input)]
    os.makedirs(a.outdir, exist_ok=True)
    for name, m in items:
        if m is None:
            print("%-30s PARSE FAILED" % name)
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:40]
        path = a.out or os.path.join(a.outdir, "comparison_%s.svg" % safe)
        try:
            path, res = compare_figure(m, path, engines=engines, size=a.size,
                                       png=a.png)
        except Exception as exc:
            print("%-30s FAILED: %s" % (name, exc))
            continue
        print(path)
        for eng, s, (same, _diff) in res:
            print("   %-22s crossings %3d (%d inside a ligand)  overlaps %2d  "
                  "tight %3d  stretch %.1f"
                  % (eng, s["crossings"], same, s["overlaps"], s["tight"],
                     s["stretch"]))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="metal2d",
        description="Readable 2D depictions of coordination complexes.")
    p.add_argument("--version", action="version", version="metal2d " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw", help="write a depiction per structure")
    _add_common(d)
    d.add_argument("--index", type=int, nargs="*", help="only these positions")
    d.add_argument("--png", action="store_true", help="PNG instead of SVG")
    d.add_argument("--size", type=int, default=820)
    d.add_argument("--outdir", default=".")
    d.set_defaults(func=cmd_draw)

    m = sub.add_parser("metrics", help="score depictions and compare engines")
    _add_common(m)
    m.add_argument("--engines", default="compute2dcoords,coordgen,metal2d")
    m.add_argument("--step", type=int, default=1, help="score every Nth structure")
    m.add_argument("--worst", type=int, nargs="?", const=15,
                   help="list the worst structures")
    m.add_argument("--csv", help="write per-structure metrics here")
    m.set_defaults(func=cmd_metrics)

    c = sub.add_parser("compare", help="side-by-side figure of several engines")
    _add_common(c)
    c.add_argument("--index", type=int, nargs="*")
    c.add_argument("--engines", default="compute2dcoords,coordgen,metal2d")
    c.add_argument("-o", "--out", help="output path for a single structure")
    c.add_argument("--outdir", default=".")
    c.add_argument("--size", type=int, default=640)
    c.add_argument("--png", action="store_true")
    c.set_defaults(func=cmd_compare)

    a = p.parse_args(argv)
    if a.quiet:
        RDLogger.DisableLog("rdApp.*")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
