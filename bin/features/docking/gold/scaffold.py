from itertools import product
from typing import Dict, List

from rdkit import Chem

from ...utils.utils import flatten


class ScaffoldMatcher:
    def __init__(
        self,
        smiles: str,
        scaffold_data: List[Dict],
        check_distinct: bool = True,
        exclude_dummy=True,
    ):
        self.smiles = Chem.CanonSmiles(smiles)
        if not isinstance(scaffold_data, list):
            scaffold_data = [scaffold_data]
        self.scaffold_data = scaffold_data
        self.check_distinct = check_distinct
        self.exclude_dummy = exclude_dummy

    def _set_scaffold_indice(self):
        self.match_info["scaffold_indices"] = []
        # Get lists from keys with non-empty lists
        lists_to_combine = [value for key, value in self.match_info.items() if value]
        if len(lists_to_combine) < len(self.scaffold_data):
            return
        combinations = list(product(*lists_to_combine))
        if self.check_distinct:
            combinations = [
                i for i in combinations if len(flatten(i)) == len(set(flatten(i)))
            ]
        flat_combi = [list(set(flatten(j))) for j in combinations]
        self.match_info["scaffold_indices"] = flat_combi

    def run(self):
        match_info = dict()
        self.mols = []
        for i, scaffold in enumerate(self.scaffold_data):
            smarts = scaffold["smarts"]
            add_hs = scaffold["add_hs"]
            set_chiral = scaffold["set_chiral"]
            mol, indices, dummy_atoms = substruct_match(
                self.smiles, smarts, check_hs=add_hs, use_chirality=set_chiral
            )
            indices = [list(i) for i in indices]
            if self.exclude_dummy:
                indices = [
                    list(set(i[0]) - set(i[1])) for i in zip(indices, dummy_atoms)
                ]
            match_info[i] = indices
            if mol:
                self.mols.append(mol)
        self.match_info = match_info
        self._set_scaffold_indice()
        self.scaffold_indices = self.match_info["scaffold_indices"]
        return self


def substruct_match(
    smiles: str,
    substruct: str,
    check_hs: bool = False,
    use_chirality: bool = True,
    substruct_is_smarts=True,
):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if check_hs:
        mol = Chem.AddHs(mol)
    # Chem.Kekulize(mol)
    if substruct_is_smarts:
        func = Chem.MolFromSmarts
    else:
        func = Chem.MolFromSmiles
    sub_mol = func(substruct)
    dummy_idx = [at.GetIdx() for at in sub_mol.GetAtoms() if at.GetAtomicNum() == 0]
    match_atoms = mol.GetSubstructMatches(sub_mol, useChirality=use_chirality)
    dummy_atoms = []
    for i in range(len(match_atoms)):
        dummy_atom = [match_atoms[i][j] for j in dummy_idx]
        dummy_atoms.append(dummy_atom)
    if check_hs:
        for at in mol.GetAtoms():
            at.SetIntProp("_H_ADDED_IDX", at.GetIdx())
        map = {at.GetIntProp("_H_ADDED_IDX"): at.GetIdx() for at in mol.GetAtoms()}
        mol = Chem.RemoveHs(mol)
        # for at in mol.GetAtoms():
        #     map.update({at.GetIntProp("_H_ADDED_IDX"): at.GetIdx()})
        mapped_match_atoms = []
        mapped_dummy_atoms = []
        for atom_list in match_atoms:
            mapped_match_atoms.append([map[at] for at in atom_list if at in map])
        for atom_list in dummy_atoms:
            mapped_dummy_atoms.append([map[at] for at in atom_list if at in map])
        return mol, mapped_match_atoms, mapped_dummy_atoms

    return mol, match_atoms, dummy_atoms
