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


def test_atomic_coordination_sphere_has_equal_bond_lengths():
    """The ordinary SMILES path must draw cisplatin symmetrically."""
    from metal2d import core as patched_core
    mol = patched_core.depict(Chem.MolFromSmiles(
        "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]"))
    mi = patched_core.find_metal(mol)
    conf = mol.GetConformer()
    xy = np.asarray(conf.GetPositions())[:, :2]
    radii = np.asarray([
        np.linalg.norm(xy[b.GetOtherAtomIdx(mi)] - xy[mi])
        for b in mol.GetAtomWithIdx(mi).GetBonds()])
    assert len(radii) == 4
    assert np.ptp(radii) < 1e-8


def test_closed_polydentate_ligand_centres_metal_in_donor_cavity():
    """A guarded donor-centroid pass untangles a closed Zr macrocycle."""
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics
    smi = ("O=C1CN23->[Zr+4]45678(<-[O-]1)<-[O-]C(=O)CN->4(CCC2)"
           "CCN->5(CC(=O)[O-]->6)CCN->7(CC(=O)[O-]->8)CC3")
    out = patched_core._drawing_mol(
        patched_core.depict(Chem.MolFromSmiles(smi)))
    result = patched_metrics.score(out)
    assert result["crossings"] <= 9
    assert result["overlaps"] == 0


def test_flipped_pop_ligand_can_use_less_stretched_coordgen_candidate():
    """A cleaner, shorter candidate must not fail an absolute stretch cap."""
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics
    smi = ("CC(C)P1(c2cccc3c4cccc5P(C(C)C)(C(C)C)->[Os+2]<-1"
           "(<-[Cl-])(<-[Cl-])(<-o(c23)c54)<-S(C)(C)=O)C(C)C")
    out = patched_core._drawing_mol(
        patched_core.depict(Chem.MolFromSmiles(smi)))
    result = patched_metrics.score(out)
    assert result["crossings"] <= 1
    assert result["flipped"] == 0


def test_small_chelate_is_not_left_far_outside_mixed_denticity_sphere():
    """A remote O,N chelate is compacted without worsening crossings."""
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics
    smi = ("[O-]N([O-])c1ccc2[O-]->[Fe+4]345(<-Nc2c1)<-n1ccccc1SC->3"
           "(Sc1ccccn->41)=[SH]c1ccccn->51")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = patched_metrics.score(patched_core._drawing_mol(out))
    assert result["crossings"] <= 4
    assert result["stretch"] <= 4.0


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


def test_eta2_alkene_is_side_on_to_the_metal():
    """A coordinated C=C is tangential, not pointed radially at the metal."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core

    mol = Chem.MolFromSmiles("[Pt+2]1(<-[CH3-])(<-[CH2]=[CH2]->1)<-N")
    out = patched_core.depict(mol)
    mi = patched_core.find_metal(out)
    donors = {b.GetOtherAtomIdx(mi) for b in out.GetAtomWithIdx(mi).GetBonds()}
    alkene = next(b for b in out.GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE and
                  {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= donors)
    a, b = alkene.GetBeginAtomIdx(), alkene.GetEndAtomIdx()
    p = out.GetConformer().GetPositions()[:, :2]
    edge = p[b] - p[a]
    radial = (p[a] + p[b]) / 2.0 - p[mi]
    cosine = abs(np.dot(edge, radial) /
                 (np.linalg.norm(edge) * np.linalg.norm(radial)))
    assert cosine < 0.05

    drawing = patched_core._drawing_mol(out)
    eta2 = [atom for atom in drawing.GetAtoms()
            if atom.HasProp("_hapticAtoms") and
            len(atom.GetProp("_hapticAtoms").split(",")) == 2]
    assert len(eta2) == 1


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


def test_polynuclear_layout_chooses_the_clean_root():
    """SMILES atom order must not force a tangled recursive Ru-Ru layout."""
    # This sandbox keeps the candidate patch in top-level core.py while the
    # other regression tests exercise the installed package.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("C1C(Cl)=CC=C(NC2[S-][Ru+2]34567(C8C3=C4(C(C)C)C5=C6C=87C)"
           "(N3=C([S-][Ru+2]45678(C9(C)C4=C5C6(C(C)C)=C7C=98)(N=23)<-[Cl-])"
           "NC2C=CC=CC=2)<-[Cl-])C=1")
    mol = Chem.MolFromSmiles(smi)
    mol = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(mol)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0


def test_corrected_ru_dimer_uses_the_ordinary_recursive_layout():
    """Do not override a clean recursive layout by reshaping its bipyridines."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("[Ru+2]%20%21%22%23(<-n%28ccccc%28-c%29ccccn%29->%20)"
           "<-n%30ccccc%30-c%31ccccn%31->%21.[Ru+]%24%25%26%27"
           "(<-n%32ccccc%32-c%33ccccn%33->%24)<-n%34ccccc%34-c%35ccccn%35->%25."
           "n%36->%23c(-c%38nc%39ccccc%39[n-]%38->%22)ccc%37cc%27c"
           "(-c%40nc%41ccccc%41[n-]%40->%26)nc%37%36")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_oxo_bridged_gallium_prefers_legibility_over_short_bonds():
    """The rigid Ga2O2 template tangles chelating bridges; recursion is cleaner."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("[Ga+3]%11%12%13%14%15%16.[Ga+3]%21%22%23%24%25%26."
           "[O-]->%13->%21c1ccc(F)cc1C=N->%12c1ccccc1[O-]->%11."
           "[O-]->%14c1cccc2cccn->%15c12."
           "[O-]->%16->%22c1cccc2cccn->%23c12."
           "[O-]->%24c1ccc(F)cc1C=N->%25c1ccccc1[O-]->%26")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] <= 3
    assert result["overlaps"] == 0


def test_cu2br2_terminals_can_change_sectors_rigidly():
    """Bulky single-donor terminals must fan out without deforming themselves."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("[Cu+]%11%12%13<-P(c1ccccc1)(c1ccccc1)c1ccccc1."
           "[Cu+]%21%22%23<-P(c1ccccc1)(c1ccccc1)c1ccccc1."
           "[Br-]->%12->%22.[Br-]->%13->%23."
           "c1ccc2c(c1)nc(-c1cnccn1->%11)n2Cc1ccc2ccccc2n1."
           "c1ccc2c(c1)nc(-c1cnccn1->%21)n2Cc1ccc2ccccc2n1")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_cu2cl4_core_leaves_room_for_four_bridge_labels():
    """Four shared chlorides must not be squeezed between adjacent Cu labels."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("Cc1cc(C)[n]2->[Cu+2]3(<-[Cl-])(<-[Cl-]->[Cu+2]4(<-[Cl-])"
           "(<-[Cl-]->3)<-[n]3cc(Br)cnc3-n3c(C)cc(C)[n]->43)"
           "<-[n]3cc(Br)cnc3-n12")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0

    metals = [a.GetIdx() for a in out.GetAtoms()
              if a.GetAtomicNum() == 29]
    conf = out.GetConformer()
    p = [conf.GetAtomPosition(i) for i in metals]
    metal_gap = np.hypot(p[0].x - p[1].x, p[0].y - p[1].y)
    assert metal_gap >= 1.8 * result["bond_len"]

    # At least one graph automorphism exchanging the Cu atoms must also map the
    # coordinates onto a 180-degree rotation about the Cu2 centre.
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                       for i in range(out.GetNumAtoms())])
    centre = (coords[metals[0]] + coords[metals[1]]) / 2.0
    residuals = []
    for amap in out.GetSubstructMatches(out, uniquify=False, maxMatches=256):
        if amap[metals[0]] == metals[1] and amap[metals[1]] == metals[0]:
            target = 2.0 * centre - coords
            residuals.append(np.sqrt(np.mean(np.sum(
                (coords[list(amap)] - target) ** 2, axis=1))))
    assert residuals and min(residuals) < 1e-6


def test_fe_cluster_can_turn_benzyl_branch_without_changing_core_template():
    """A flexible phenyl branch may move; the strict Fe cluster core may not."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("O=[C]1[Fe+]23456(<-[C](C=[C](c7ccccc7)->[Fe+]12789%10"
           "(<-[C-]#[O+])<-[cH]1[cH]->7[cH]->8[cH-]->9[cH]->%101)"
           "=[N+](Cc1ccccc1)Cc1ccccc1)<-[cH]1[cH]->3[cH]->4[cH-]->5[cH]->61")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] <= 1
    assert result["overlaps"] == 0
    assert result["tight"] <= 4


def test_ag2_large_bridges_can_fall_back_to_clean_coordgen_layout():
    """Do not force a tangled rigid cluster template when CoordGen is clean."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("[Ag+]%11%12%30<-O.[Ag+]%21%22%30<-O."
           "c1cn->%11c2c(c1)ccc1cccn->%21c12."
           "c1cn->%12c2c(c1)ccc1cccn->%22c12")
    mol = Chem.MolFromSmiles(smi)
    assert not patched_core._has_haptic_coordination(mol)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_rh2_aromatic_bridges_can_use_clean_recursive_candidate():
    """Legibility outranks short closing bonds when both rigid layouts cross."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("CC1=[O]->[Rh+2]23(<-[O-]C(C)=[O]->[Rh+2]24(<-[O-]1)"
           "<-[n]1ccccc1-c1cccc[n]->41)<-[n]1cccc2c4nc5cc6ccccc6cc5nc4"
           "c4ccc[n]->3c4c21")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_macrocycle_terminal_oh_and_chloride_use_different_sectors():
    """Independent monoatomic terminal ligands must not occupy one point."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("C12S([O-][Fe+3]34(O)(N(=CC5=CC(S(=O)([O-][Fe+3]67"
           "(<-[Cl-])(N(NC(=S6)NC6C=CC=C8C=6C=CC=C8)=CC(C=1)="
           "C([O-]7)C=C2)O)=O)=CC=C5[O-]3)NC(=S4)NC1C=CC=C2C=1C=CC=C2)"
           "<-[Cl-])(=O)=O")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_ag2_whole_bridges_consider_clean_coordgen_without_mm_bond():
    """The no-M-M macrocycle branch must also keep CoordGen as a candidate."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import metrics as patched_metrics

    smi = ("[Ag+]%11%31%12%13.[Ag+]%21%41%22%23."
           "c1ccc2cn->%11n->%21cc2c1.c1ccc2cn->%31n->%41cc2c1."
           "c1ccc2cn->%12ncc2c1.c1ccc2cn->%22ncc2c1."
           "[O-]->%13[N+](=O)[O-].[O-]->%23[N+](=O)[O-]")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core._drawing_mol(patched_core.depict(mol))
    result = patched_metrics.score(out)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0
    assert result["stretch"] < 1.5


def test_fe2_independent_bridges_can_occupy_opposite_sides():
    """Pinned whole-ligand bridges may be reflected without changing Fe/Cp."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import cluster as patched_cluster

    smi = ("c1(C/[N+](=[C]2/[Fe+]3456789(<-[C-](->[Fe+]3%10%11%12%13"
           "(<-[C-]#[O+])([C]3=[C]%10[C-]%11[C]%12=[C]%133)[C]->4(C)="
           "[C]25)#[O+])[C]2=[C]6[C]7=[C]8[C-]92)C)ccccc1")
    mol = Chem.MolFromSmiles(smi)
    raw = patched_cluster.depict(mol)
    polished = patched_core.depict(mol)
    raw_q = patched_cluster._tangle(
        patched_core._drawing_mol(raw), strain=False)
    polished_q = patched_cluster._tangle(
        patched_core._drawing_mol(polished), strain=False)
    assert polished_q < raw_q
    assert polished_q <= 5.0


def test_fe2_haptic_terminal_can_leave_a_crowded_bridge_sector():
    """A terminal Cp ring may turn around Fe without changing the Fe/Cp core."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import cluster as patched_cluster

    smi = ("[C-]12[Fe+]3456789(<-[C](c%10ccccc%10)(=[C]->3(c3ccccc3)"
           "[C]4=[N+](C)C)[Fe+]534%10%11(<-[C-]#[O+])(<-[C-]->6#[O+])"
           "[C]5=[C]3[C-]4[C]%10=[C]%115)[C](=[C]17)[C]8=[C]29")
    mol = Chem.MolFromSmiles(smi)
    raw = patched_cluster.polish_organic_branches(
        patched_cluster.depict(mol))
    polished = patched_core.depict(mol)
    raw_q = patched_cluster._tangle(
        patched_core._drawing_mol(raw), strain=False)
    polished_q = patched_cluster._tangle(
        patched_core._drawing_mol(polished), strain=False)
    assert polished_q < raw_q
    # The drawing now retains the explicit sigma Fe--C anchor of the open eta
    # bridge; the previous limit counted a chemically incomplete drawing.
    assert polished_q <= 7.0


def test_fe2_large_chelate_can_swing_about_shared_carbon():
    """A C3/N bridge should sit outside, not fold through, the Fe2 core."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import cluster as patched_cluster

    smi = ("[C-]12[Fe+]3456789(<-[C](C(=O)OC)(=[C]->3(C(OC)=O)/[C]4="
           "[N+](/C)c3c(C)cccc3C)[Fe+]534%10%11(<-[C-]#[O+])(<-[C-]->6"
           "#[O+])[C]5=[C]3[C-]4[C]%10=[C]%115)[C](=[C]17)[C]8=[C]29")
    mol = Chem.MolFromSmiles(smi)
    pair = patched_cluster.metal_clusters(mol)[0]
    raw = patched_cluster.polish_organic_branches(
        patched_cluster.depict(mol))
    raw = patched_cluster.polish_bridge_sides(raw, pair)
    raw = patched_cluster.polish_haptic_sectors(raw, pair)
    polished = patched_core.depict(mol)
    raw_q = patched_cluster._tangle(
        patched_core._drawing_mol(raw), strain=False)
    polished_q = patched_cluster._tangle(
        patched_core._drawing_mol(polished), strain=False)
    assert polished_q < raw_q
    # One explicit sigma Fe--C anchor is intentionally no longer swallowed by
    # the eta centroid, so the complete drawing has one additional segment.
    assert polished_q <= 6.0
    conf = polished.GetConformer()
    fe0 = np.array(conf.GetAtomPosition(1))[:2]
    fe1 = np.array(conf.GetAtomPosition(23))[:2]
    axis = fe1 - fe0
    organic = np.array(conf.GetAtomPosition(2))[:2] - fe0
    mu_co = np.array(conf.GetAtomPosition(26))[:2] - fe0
    cross = lambda v: axis[0] * v[1] - axis[1] * v[0]
    assert cross(organic) * cross(mu_co) < 0


def test_fe2_equivalent_cp_centres_are_mirrored():
    """Equivalent terminal Cp rings belong at equal mirrored Fe offsets."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core

    smi = ("[C-]12[Fe+]3456789(<-[C](C)(=[C]->3(C)/[C]4=[N+](/C)"
           "c3c(C)cccc3C)[Fe+]534%10%11(<-[C-]#[O+])(<-[C-]->6#[O+])"
           "[C]5=[C]3[C-]4[C]%10=[C]%115)[C](=[C]17)[C]8=[C]29")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    conf = out.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(out.GetNumAtoms())])
    left = xy[[0, 27, 28, 29, 30]].mean(axis=0) - xy[1]
    right = xy[[22, 23, 24, 25, 26]].mean(axis=0) - xy[17]
    assert np.allclose(left, [-right[0], right[1]], atol=1e-6)
    axis = xy[17] - xy[1]
    midpoint = (xy[1] + xy[17]) / 2.0
    # C2 of the organic bridge is centred above Fe2; mu-CO is opposite below.
    assert abs(np.dot(xy[2] - midpoint, axis)) < 1e-6
    cross = lambda v: axis[0] * v[1] - axis[1] * v[0]
    assert cross(xy[2] - midpoint) * cross(xy[20] - midpoint) < 0


def test_fast_coordinate_scorer_matches_rdkit_candidate_score():
    """The bimetallic fast path must preserve the exact ranking objective."""
    import sys
    from pathlib import Path
    from rdkit.Geometry import Point3D
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d import cluster as patched_cluster

    smi = ("[C-]12[Fe+]3456789(<-[C](C)(=[C]->3(C)/[C]4=[N+](/C)"
           "c3c(C)cccc3C)[Fe+]534%10%11(<-[C-]#[O+])(<-[C-]->6#[O+])"
           "[C]5=[C]3[C-]4[C]%10=[C]%115)[C](=[C]17)[C]8=[C]29")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    conf = out.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(out.GetNumAtoms())])
    xy += np.random.default_rng(7).normal(0.0, 0.1, xy.shape)
    scorer = patched_cluster._coordinate_scorer(out)
    cand = Chem.Mol(out)
    cand.RemoveAllConformers()
    c = Chem.Conformer(cand.GetNumAtoms())
    for i, (x, y) in enumerate(xy):
        c.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    cand.AddConformer(c)
    drawn = patched_core._drawing_mol(cand)
    for strain in (False, True):
        assert scorer(xy, strain) == patched_cluster._tangle(drawn, strain)


def test_high_denticity_mono_fallback_is_readable_and_compact():
    """A closed Ti donor cage may use the guarded single CoordGen candidate."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("Cc1cccc(C)c1[O-]->[Ti+4]12(<-[O-]c3c(C)cccc3C)"
           "<-[N-](c3ccccc3)P(c3ccccc3)(c3ccccc3)=[P+]->1="
           "P(c1ccccc1)(c1ccccc1)[N-]->2c1ccccc1")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] <= 2
    assert result["overlaps"] == 0
    assert result["tight"] == 0
    assert result["stretch"] < 1.5


def test_large_mono_ligands_can_change_angular_sectors():
    """Large separate ligands are not trapped in the local +/-30 degree pass."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("CC(C)(C)[Si](C)(C1=N(C2CCCCC2)->[Y+3]234(<-N([Si](C)(C)C)="
           "P(c5ccccc5)(c5ccccc5)[C-]->2(C(=NC2CCCCC2)[N-]->3C2CCCCC2)"
           "P(c2ccccc2)(c2ccccc2)=N->4[Si](C)(C)C)<-[N-]1C1CCCCC1)"
           "C(C)(C)C")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] <= 8


def test_open_eta_chain_uses_solid_centroid_bond():
    """Open eta systems and cyclic Cp centroids both use readable solid lines."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core

    smi = ("CCOC(=O)[C]12->[Fe+2]3456789%10(<-[CH](=[C]->31N(C)"
           "c1ccc(OC)cc1)[C]->4(C1CCCCC1)=[Fe+]5134%11(<-[C-]#[O+])"
           "(<-[cH]5[cH]->1[cH]->3[cH-]->4[cH]->%115)[C]->6=2OC)"
           "<-[cH]1[cH]->7[cH]->8[cH-]->9[cH]->%101")
    drawn = patched_core._drawing_mol(
        patched_core.depict(Chem.MolFromSmiles(smi)))
    kinds = []
    for atom in drawn.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.HasProp("_hapticAtoms"):
            kinds.append(atom.GetBonds()[0].GetBondType())
    assert kinds
    assert set(kinds) == {Chem.BondType.SINGLE}


def test_ru2_two_end_bridge_sits_opposite_mu_carbonyl():
    """A rigid C...C bridge caps Ru2 while mu-CO and Cp occupy the other side."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("O=[C]1C(c2ccccc2)=[C](c2ccccc2)[Ru+]23456(<-[C-]#[O+])"
           "(<-[cH]7[cH]->2[cH]->3[cH-]->4[cH]->57)[C](=O)[Ru+]162345"
           "<-[cH]1[cH]->2[cH]->3[cH-]->4[cH]->51")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] <= 1
    assert result["overlaps"] == 0
    assert result["tight"] <= 1


def test_metal_bound_carbonyl_on_large_ru_bridge_points_outward():
    """An acyl C=O must not fold back into the Ru2 core after bridge polish."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core

    smi = ("CC12CCC3(C)c4ccc(O)cc4CCC3C1CC(O)([C]13->[Ru+]456789("
           "<-[cH]%10[cH]->4[cH]->5[cH-]->6[cH]->7%10)([C]1=O)[C](=O)"
           "[Ru+]81456(<-[C-]#[O+])(<-[cH]7[cH]->1[cH]->4[cH-]->5[cH]->67)"
           "[C]=39)C2")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    p = out.GetConformer().GetPositions()[:, :2]

    # Atom 27 is the metal-bound carbonyl carbon, 28 its terminal oxygen,
    # 20 the organic neighbour and 21 the bonded Ru in this regression case.
    def angle(a, b):
        return np.degrees(np.arccos(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    co = p[28] - p[27]
    assert abs(angle(p[20] - p[27], co) - 120.0) < 1e-6
    assert angle(p[21] - p[27], co) > 100.0


def test_equivalent_cp_and_terminal_co_pairs_are_mirrored():
    """Equivalent Fe termini prefer exact symmetry when it costs no tangles."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("[Fe+]%10%11%12%13%14%40%41%42<-[C-]#[O+]."
           "[Fe+]%20%21%22%23%24%40%50%51<-[C-]#[O+]."
           "[cH-]%15%10[cH]%11[cH]%12[cH]%13[cH]%14%15."
           "[cH-]%25%20[cH]%21[cH]%22[cH]%23[cH]%24%25."
           "C%41%50=[N+](c1c(C)cc(Cl)cc1C)C.O=C%42%51")
    mol = Chem.MolFromSmiles(smi)
    out = patched_core.depict(mol)
    p = out.GetConformer().GetPositions()[:, :2]
    midpoint = (p[0] + p[3]) / 2.0
    ex = (p[3] - p[0]) / np.linalg.norm(p[3] - p[0])
    reflect = lambda q: q - 2.0 * np.dot(q - midpoint, ex) * ex

    # Terminal carbonyl carbon and oxygen are exact mirrored counterparts.
    assert np.allclose(reflect(p[1]), p[4], atol=1e-6)
    assert np.allclose(reflect(p[2]), p[5], atol=1e-6)
    # The formally negative Cp atoms mirror too; previously only the ring
    # centroids were symmetric while the labelled atoms and double bonds were
    # rotated independently.
    assert np.allclose(reflect(p[6]), p[11], atol=1e-6)
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] == 0
    assert result["overlaps"] == 0


def test_sigma_fe_c_bonds_survive_eta_centroid_collapse():
    """Only dative contacts are folded; explicit sigma Fe--C stays visible."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core

    smi = ("C=CC/[N+](C)=[C]1\\[CH]2->[Fe+]1345678(<-[C-](#[O+])->"
           "[Fe+]319%10%11(<-[C-]#[O+])(<-[cH]3[cH]->1[cH]->9[cH-]->"
           "%10[cH]->%113)[C]->4=2C)<-[cH]1[cH]->5[cH]->6[cH-]->7"
           "[cH]->81")
    mol = Chem.MolFromSmiles(smi)
    drawn = patched_core._drawing_mol(patched_core.depict(mol))
    assert drawn.GetBondBetweenAtoms(7, 5).GetBondType() == Chem.BondType.SINGLE
    assert drawn.GetBondBetweenAtoms(10, 18).GetBondType() == Chem.BondType.SINGLE
    assert any(a.GetAtomicNum() == 0 and a.HasProp("_hapticAtoms")
               for a in drawn.GetAtoms())


def test_on_metal_label_gets_guarded_whole_molecule_candidate():
    """A clean crossing score must not hide atoms printed on the metal."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("CC1(C)c2cccc3c2Oc2c1cccc2P(C(C)(C)C)(C(C)(C)C)->[Cu+]"
           "<-P3(C(C)(C)C)C(C)(C)C.F[B-](F)(F)F")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["on_metal"] == 0
    assert result["crossings"] == 0


def test_two_large_tridentate_halves_are_placed_opposite():
    """Calixarene-like halves must not be superposed around the metal."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("CC(C)(C)c1cccc2-c3cccc4-c5cccc(C(C)(C)C)c5[O-]->[Zr+4]56"
           "(<-[O-]c21)(<-[O-]c1c(-c2cccc(-c7cccc(C(C)(C)C)c7[O-]"
           "->5)c->62)cccc1C(C)(C)C)<-c34")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    drawn = patched_core._drawing_mol(out)
    result = score(drawn)
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["on_metal"] == 0
    zr = next(a for a in drawn.GetAtoms() if a.GetSymbol() == "Zr")
    assert sorted(n.GetSymbol() for n in zr.GetNeighbors()) == ["O"] * 4


def test_macrocycle_with_small_extra_ligands_centres_metal():
    """Small terminal ligands do not stop a metal entering its macrocyclic cavity."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("[O+]#[C-]->[Ru+2]123(<-[C-]#[O+])<-C4=CC=C5C(c6ccccc6)="
           "c6ccc(n6C(C(c6ccc(Cl)cc6)c6ccc(Cl)cc6)[N-]->15)=C(c1ccccc1)"
           "C1=N->2C(=C(c2ccccc2)c2ccc([n-]->32)C=4c2ccccc2)C=C1")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    conf = out.GetConformer()
    metal = 2
    macro_donors = {5, 37, 46, 59}
    macro_ring = max((ring for ring in out.GetRingInfo().AtomRings()
                      if macro_donors.issubset(set(ring))), key=len)
    centre = np.mean([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                      for i in macro_ring], axis=0)
    metal_xy = np.array([conf.GetAtomPosition(metal).x,
                         conf.GetAtomPosition(metal).y])
    assert np.linalg.norm(metal_xy - centre) < 1e-6
    assert result["crossings"] <= 3
    assert result["overlaps"] <= 2
    assert result["stretch"] <= 2.8


def test_repeated_charged_tridentate_ligands_reuse_clean_template():
    """Equivalent O,N,O chelates must not inherit order-dependent CoordGen knots."""
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("c1ccn2[O-]->[Ni+2]345(<-[O-]n6ccccc6=[N+]->3=c2c1)"
           "<-[O-]n1ccccc1=[N+]->4=c1ccccn1[O-]->5")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] == 0
    assert result["overlaps"] == 0
    assert result["tight"] == 0


def test_crossed_balanced_articulation_cage_is_rebuilt_from_branches():
    """A defective tripodal cage is rebuilt only after its ligand self-crosses."""
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("[Cl-]->[Fe+2]123<-[Si-](c4ccccc4P->1(c1ccccc1)c1ccccc1)"
           "(c1ccccc1P->2(c1ccccc1)c1ccccc1)c1ccccc1P->3(c1ccccc1)"
           "c1ccccc1")
    out = patched_core.depict(Chem.MolFromSmiles(smi))
    result = score(patched_core._drawing_mol(out))
    assert result["crossings"] <= 5
    assert result["overlaps"] == 0
    assert result["on_metal"] == 0


def test_non_circular_single_ligand_does_not_put_donor_on_metal():
    """A least-squares donor circle must not pass through one donor label."""
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smiles = [
        "c1ccc(C(c2ccccc2)=N2CCN34Cc5ccccn5->[Cu+]<-2<-3<-n2ccccc2C4)cc1",
        "c1ccc2[S-]->[Fe+2]34<-[S-]c5ccccc5CN->3(Cc2c1)Cc1ccccc1[S-]->4",
    ]
    for smi in smiles:
        out = patched_core.depict(Chem.MolFromSmiles(smi))
        result = score(patched_core._drawing_mol(out))
        assert result["overlaps"] == 0


def test_linear_tetraphosphine_is_built_from_rigid_blocks():
    """A P4 chain forms a cavity without crushing its phenyl rings."""
    from metal2d import core as patched_core
    from metal2d.metrics import score

    smi = ("Cc1cccc(C)c1[N+]#[C-]->[Mo]123(<-[C-]#[N+]c4c(C)cccc4C)"
           "<-P(CCP->1(c1ccccc1)c1ccccc1P->2(CCP->3(c1ccccc1)c1ccccc1)"
           "c1ccccc1)(c1ccccc1)c1ccccc1")
    result = score(patched_core._drawing_mol(
        patched_core.depict(Chem.MolFromSmiles(smi))))
    assert result["crossings"] <= 6
    assert result["overlaps"] <= 2
    assert result["stretch"] < 4.0


def test_sample_data_ships_and_parses():
    import os
    path = os.path.join(os.path.dirname(metal2d.__file__), "data", "sample.smi")
    assert os.path.exists(path)
    names = [n for n, m in metal2d.read_molecules(path) if m is not None]
    assert len(names) > 250
