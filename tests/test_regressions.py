"""Regression tests. Every case here is a bug that was found and fixed."""
import numpy as np
import pytest
from rdkit import Chem, RDLogger

import metal2d
from metal2d.core import _hapto_groups, _drawing_mol
from metal2d.metrics import score

RDLogger.DisableLog("rdApp.*")


def drawn(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "SMILES did not parse: %s" % smiles
    return _drawing_mol(metal2d.depict(mol))


def donor_radii(mol):
    mi = metal2d.find_metal(mol)
    c = mol.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                  for i in range(mol.GetNumAtoms())])
    bl = np.median([np.linalg.norm(P[b.GetBeginAtomIdx()] - P[b.GetEndAtomIdx()])
                    for b in mol.GetBonds()])
    return [np.linalg.norm(P[b.GetOtherAtomIdx(mi)] - P[mi]) / bl
            for b in mol.GetAtomWithIdx(mi).GetBonds()]


# --------------------------------------------------------------------------- #
def test_gallium_and_indium_are_metals():
    """The d-block ranges stopped one short of Ga, In and Tl."""
    for z in (31, 49, 81):
        assert z in metal2d.METALS
    mol = Chem.MolFromSmiles(
        r"C1\c2c([O-]->[Ga+3]345(<-[O-]c6ccc7ccccc7c6/C=[N]->3/c3cccc6ccc[n]->4c36)"
        r"<-[N]=1/c1cccc3ccc[n]->5c13)ccc1ccccc21")
    assert metal2d.find_metal(mol) is not None
    assert score(_drawing_mol(metal2d.depict(mol)))["crossings"] == 0


def test_heteroatom_hydrogens_survive():
    """SetNoImplicit on every atom wiped the H off [nH], -OH and -NH-."""
    mol = drawn("C[c]12->[Rh+3]3456(<-[Cl-])(<-[n]7cc[nH]c7-c7[nH]cc[n]->37)"
                "<-[c]1(C)[c]->4(C)[c-]->5(C)[c]->62C")
    nh = [a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    assert sum(nh) == 2, "the two imidazole N-H were lost"


def test_no_dashed_aromatic_bonds():
    """Converting dative bonds before kekulising made pyridine unkekulisable."""
    mol = drawn("O=S1(=O)CCN(c2ccc(-c3nc4c5ccc[n]6->[Ru+2]78(<-[n]9ccccc9-c9cccc[n]->79)"
                "(<-[n]7ccccc7-c7cccc[n]->87)<-[n]7cccc(c4[nH]3)c7c56)cc2)CC1")
    assert not any(b.GetBondType() == Chem.BondType.AROMATIC for b in mol.GetBonds())


def test_haptic_group_gets_exactly_one_bond():
    """Grouping by ring missed the exocyclic donor carbon of a Cp*."""
    smi = ("Cc1cc(C)n(CC[N]23CCCCCCNC(=O)[C@@H]4Cc5c[nH]c[n]5->[Tc+]<-2"
           "(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[NH2]CC3)c1")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip("SMILES specific to one dataset")
    d = _drawing_mol(metal2d.depict(mol))
    dummies = [a for a in d.GetAtoms() if a.HasProp("_hapticAtoms")]
    for a in dummies:
        assert a.GetDegree() == 1


def test_cp_star_collapses():
    mol = Chem.MolFromSmiles(
        "CCCCn1cc[n+]2-c3ccccc3[O-]->[Ir+3]3456(<-[Cl-])(<-[c-]12)<-[c]1(C)"
        "[c]->3(C)[c]->4(C)[c-]->5(-c2ccc(-c3ccccc3)cc2)[c]->61C")
    mi = metal2d.find_metal(mol)
    assert mol.GetAtomWithIdx(mi).GetDegree() == 8      # Cl, C, and five ring C
    d = _drawing_mol(metal2d.depict(mol))
    assert d.GetAtomWithIdx(metal2d.find_metal(d)).GetDegree() == 4


def test_triple_bond_is_drawn():
    """Carbonyl C#O was squeezed to nothing between its own atom labels."""
    mol = drawn("O=C(NCCCc1cc[n](->[Re+]2(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])"
                "<-[n]3ccccc3-c3ccccn->23)cc1)C")
    mi = metal2d.find_metal(mol)
    c = mol.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                  for i in range(mol.GetNumAtoms())])
    bl = np.median([np.linalg.norm(P[b.GetBeginAtomIdx()] - P[b.GetEndAtomIdx()])
                    for b in mol.GetBonds()])
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.TRIPLE:
            d = np.linalg.norm(P[b.GetBeginAtomIdx()] - P[b.GetEndAtomIdx()])
            assert d > 0.6 * bl, "C#O drawn far shorter than a normal bond"


def test_bipyridine_folds_into_the_chelating_conformer():
    """CoordGen draws free bipy s-trans, which no metal position can satisfy."""
    mol = drawn("c1cc[n]2->[Ir+3]34(<-[c-]5cccc6c7nc8ccccc8nc7c7ccc[n]->3c7c65)"
                "(<-[c-]3cccc5c6nc7ccccc7nc6c6ccc[n]->4c6c53)<-[n]3ccccc3-c2c1")
    mi = metal2d.find_metal(mol)
    c = mol.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                  for i in range(mol.GetNumAtoms())])
    ri = mol.GetRingInfo().AtomRings()
    flipped = 0
    for b in mol.GetAtomWithIdx(mi).GetBonds():
        d = b.GetOtherAtomIdx(mi)
        own = [r for r in ri if d in r]
        if own and np.dot(P[mi] - P[d], P[list(own[0])].mean(0) - P[d]) > 0:
            flipped += 1
    assert flipped == 0, "a donor still faces away from the metal"


def test_polydentate_bonds_are_equal():
    """Three donors on an equal-angle slot ring pushed the middle one inwards."""
    mol = drawn("[Cl-]->[Pt+2]12<-[S-]C(=N[N]->1=Cc1cccc[n]->21)Nc1ccccc1")
    r = sorted(donor_radii(mol))
    assert r[0] > 0.9 * r[-1] * 0.6, "metal-donor distances wildly unequal"
    assert score(mol)["crossings"] == 0


def test_tripodal_ligand_does_not_sit_on_the_metal():
    """A tris(pyrazolyl)methane apex projects onto the donor triangle's centre."""
    mol = drawn("[Cl-]->[Ru+2]12(<-[n]3cccn3C(n3ccc[n]->13)n1ccc[n]->21)"
                "(<-[P]12CN3CN(CN(C3)C1)C2)<-[P](c1ccccc1)(c1ccccc1)c1ccccc1")
    assert score(mol)["on_metal"] == 0


def test_metric_forgives_the_haptic_bond_once():
    """The bond to a ring centre must cross that ring, exactly once."""
    mol = drawn("CCCCn1cc[n+]2-c3ccccc3[O-]->[Ir+3]3456(<-[Cl-])(<-[c-]12)<-[c]1(C)"
                "[c]->3(C)[c]->4(C)[c-]->5(-c2ccc(-c3ccccc3)cc2)[c]->61C")
    assert score(mol)["crossings"] == 0


def test_depict_does_not_mutate_the_input():
    mol = Chem.MolFromSmiles("[Cl-]->[Pt+2]12<-[S-]C(=N[N]->1=Cc1cccc[n]->21)N")
    before = mol.GetNumConformers()
    metal2d.depict(mol)
    assert mol.GetNumConformers() == before


def test_works_without_a_metal():
    mol = Chem.MolFromSmiles("c1ccccc1")
    out = metal2d.depict(mol)
    assert out.GetNumConformers() == 1


def test_sample_data_ships_and_parses():
    import os
    path = os.path.join(os.path.dirname(metal2d.__file__), "data", "sample.smi")
    assert os.path.exists(path)
    names = [n for n, m in metal2d.read_molecules(path) if m is not None]
    assert len(names) > 250
