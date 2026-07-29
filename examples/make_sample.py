"""Pick a small, diverse, reproducible subset of complexes for smoke-testing.

Stratified by metal so rare ones survive, then spread within each metal across
size, coordination number, denticity and the presence of eta-bonded groups, so
the sample exercises the different code paths rather than 300 tris-bipyridines.
"""
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
import metal2d as M

TARGET = 300
SEED = 20260728


def features(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    mi = M.find_metal(m)
    if mi is None:
        return None
    don = [b.GetOtherAtomIdx(mi) for b in m.GetAtomWithIdx(mi).GetBonds()]
    if not don:
        return None
    groups = M._hapto_groups(m, don)
    hap = sum(1 for g in groups if len(g) >= 3)
    em = Chem.RWMol(m)
    for b in list(em.GetAtomWithIdx(mi).GetBonds()):
        em.RemoveBond(mi, b.GetOtherAtomIdx(mi))
    frags = [f for f in Chem.GetMolFrags(em.GetMol()) if mi not in f]
    dent = max([sum(1 for d in don if d in f) for f in frags] + [0])
    return dict(symbol=m.GetAtomWithIdx(mi).GetSymbol(), atoms=m.GetNumAtoms(),
                cn=len(don), maxdent=dent, hapto=hap,
                donors="".join(sorted({m.GetAtomWithIdx(d).GetSymbol()
                                       for d in don})))


def spread(rows, k, rng):
    """Greedy max-min selection on scaled features, so the picks are spread out
    rather than clustered on whatever is most common."""
    if len(rows) <= k:
        return list(range(len(rows)))
    X = np.array([[np.log1p(r["atoms"]), r["cn"], r["maxdent"], r["hapto"] * 2.0]
                  for r in rows], dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    chosen = [int(rng.integers(len(rows)))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return chosen


def main(src, out="sample.smi"):
    df = pd.read_csv(src)
    col = next(c for c in df.columns if "smiles" in c.lower() and "complex" in c.lower())
    smis = df[col].dropna().drop_duplicates().tolist()

    pool = {}
    for s in smis:
        f = features(s)
        if f is None:
            continue
        f["smiles"] = s
        pool.setdefault(f["symbol"], []).append(f)

    metals = sorted(pool, key=lambda m: -len(pool[m]))
    # a floor for every metal, the rest shared out by sqrt so Ru cannot swamp it
    quota = {m: min(len(pool[m]), 3) for m in metals}
    left = TARGET - sum(quota.values())
    w = np.array([np.sqrt(len(pool[m])) for m in metals])
    w = w / w.sum()
    for m, extra in zip(metals, np.floor(w * left).astype(int)):
        quota[m] = min(len(pool[m]), quota[m] + int(extra))
    # top up on the biggest groups until we hit the target exactly
    i = 0
    while sum(quota.values()) < TARGET and i < 10000:
        m = metals[i % len(metals)]
        if quota[m] < len(pool[m]):
            quota[m] += 1
        i += 1

    rng = np.random.default_rng(SEED)
    picked = []
    for m in metals:
        rows = sorted(pool[m], key=lambda r: r["smiles"])   # stable order
        for idx in spread(rows, quota[m], rng):
            picked.append(rows[idx])

    picked.sort(key=lambda r: (r["symbol"], r["atoms"], r["smiles"]))
    with open(out, "w") as fh:
        fh.write("SMILES\tname\n")
        counters = {}
        for r in picked:
            n = counters.get(r["symbol"], 0) + 1
            counters[r["symbol"]] = n
            fh.write("%s\t%s_%03d\n" % (r["smiles"], r["symbol"].lower(), n))

    print("wrote %d structures to %s" % (len(picked), out))
    return picked


if __name__ == "__main__":
    picked = main(sys.argv[1] if len(sys.argv) > 1 else
                  "/mnt/user-data/uploads/complexes_with_smiles.csv",
                  sys.argv[2] if len(sys.argv) > 2 else "sample.smi")
    p = pd.DataFrame(picked)
    print("\nmetals: %d" % p.symbol.nunique())
    print(p.symbol.value_counts().to_dict())
    print("\ncoordination number:", p.cn.value_counts().sort_index().to_dict())
    print("max denticity      :", p.maxdent.value_counts().sort_index().to_dict())
    print("with eta groups    : %d (%.0f%%)" % ((p.hapto > 0).sum(),
                                                100 * (p.hapto > 0).mean()))
    print("atoms: median %d, range %d-%d" % (p.atoms.median(), p.atoms.min(),
                                             p.atoms.max()))
    print("donor sets present :", p.donors.nunique())
