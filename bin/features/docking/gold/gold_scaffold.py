import os
import random

from rdkit import Chem

from .scaffold import ScaffoldMatcher
from .smarts_generator import SmartsGenerator


def get_scaffold_indice(smiles, scaffold_smiles):
    if len(scaffold_smiles) == 0:
        return []
    scaffolds = [
        SmartsGenerator(s, add_hs=True).get_smarts().data for s in scaffold_smiles
    ]
    matcher = ScaffoldMatcher(
        smiles, scaffolds, check_distinct=True, exclude_dummy_for_distinct=True
    ).run()
    scaffold_indices = matcher.match_info["scaffold_indice"]
    if not scaffold_indices:
        raise Exception("Error: no scaffold matches for smiles")
    scaffold_indice = random.choice(scaffold_indices)
    return scaffold_indice


def get_gold_scaffold(ligand_sdf, output_sdf, scaffold_smiles_list):
    # Neutralize
    suppl = Chem.SDMolSupplier(ligand_sdf)
    mol = suppl[0]
    mol = neutralize_atoms(mol)
    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    print(smiles)
    print(scaffold_smiles_list)
    scaffold_indice = get_scaffold_indice(smiles, scaffold_smiles_list)
    non_scaffold_atoms = sorted(
        [at.GetIdx() for at in mol.GetAtoms() if at.GetIdx() not in scaffold_indice],
        reverse=True,
    )
    edmol = Chem.EditableMol(mol)
    for idx in non_scaffold_atoms:
        edmol.RemoveAtom(idx)
    scaffold_mol = edmol.GetMol()
    i = 0
    while True:
        i += 1
        if i == 3:
            break
        try:
            Chem.SanitizeMol(scaffold_mol)
        except Exception:
            continue
    # Aromacity sanitize
    [
        at.SetIsAromatic(False)
        for at in scaffold_mol.GetAtoms()
        if at.GetIsAromatic() and not at.IsInRing()
    ]
    # Hydrogen should be removed for Gold scaffold constraint docking.
    scaffold_mol = Chem.RemoveAllHs(scaffold_mol)
    Chem.SanitizeMol(scaffold_mol)
    if os.path.exists(output_sdf):
        os.remove(output_sdf)
        os.sync()
    with Chem.SDWriter(output_sdf) as w:
        w.write(scaffold_mol)
    os.sync()
    print(f"{output_sdf} is written.")
    return output_sdf


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol
    return mol
