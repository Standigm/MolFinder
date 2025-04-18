import pickle
from typing import Dict, Optional

from rdkit import Chem


class SmartsGenerator:
    """
    Example:
        add_dict = {
        "AnyAtomExcludeH":{
            "bond":"single",
            "indice": [0]
            }
        }
        chiral_dict = {
            "anticlockwise": [0]
        }
        smiles = 'NC(F)c1ccc2ccc(cc2c1)C(=O)c1ccnc2cc[nH]c12'
        obj = SmartsGenerator(smiles)
        obj.get_smarts(add_dict, chiral_dict)
        print(obj.smarts)

        >>> '[#7]-[#6](-[#9])-[#6]1:[#6]:[#6]:[#6]2:[#6]:[#6]:[#6](:[#6]:[#6]:2:[#6]:1)-[#6](=[#8])-[#6]1:[#6]:[#6]:[#7]:[#6]2:[#6]:[#6]:[#7H]:[#6]:1:2'

    """

    CHIRAL_TAG = {
        "anticlockwise": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "clockwise": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    }

    INHERENT_SMARTS = {
        "AnyAtom": "*",
        "AnyAtomOrNot": "?",
        "AnyAtomExcludeH": "[!H]",
        "AnyHalogen": "[Cl,Br,I,F]",
        "AnyHalogenOrNot": "[$(Cl),$(Br),$(I),$(F)]",
    }

    BOND_TYPE = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
    }

    def __init__(self, smiles, set_chiral=False, add_hs=False):
        self.mol = Chem.MolFromSmiles(smiles, sanitize=False)
        self.smiles = Chem.CanonSmiles(smiles)
        self.add_hs = add_hs
        self.set_chiral = set_chiral

        if not set_chiral:
            self.mol = self.remove_chirality(self.mol)

    def _add_chiral(self, input_dict):
        mol = self.mol
        # {1:"anticlockwise", 2:"clockwise"}
        transposed_dict = {
            value: key for key, values in input_dict.items() for value in values
        }
        for idx, chirality in transposed_dict.items():
            at = mol.GetAtomWithIdx(idx)
            at.SetChiralTag(self.CHIRAL_TAG[chirality])
        self.mol = mol

    @staticmethod
    def add_dummy_atom(mol, idx, bond_type):
        edit_mol = Chem.EditableMol(mol)
        added_atom_idx = edit_mol.AddAtom(Chem.Atom(0))
        edit_mol.AddBond(idx, added_atom_idx, bond_type)
        mol = edit_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol, added_atom_idx

    @staticmethod
    def dummy_smarts_to_star(smarts):
        marker = "[#0]"
        smarts = smarts.replace(marker, "*")
        return smarts

    @staticmethod
    def get_scaffold_fragment(
        smiles,
        add_hs,
        fragmentation_atom_sets: list,
        keep_atom_tag: int,
        remove_dummy=False,
    ):
        mol = Chem.MolFromSmiles(smiles)
        if add_hs:
            mol = Chem.AddHs(mol)
        bonds = [
            mol.GetBondBetweenAtoms(*atoms).GetIdx()
            for atoms in fragmentation_atom_sets
        ]
        frags = Chem.FragmentOnBonds(mol, bonds)

        frag_indice = Chem.GetMolFrags(frags)
        frag_mols = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)

        keep_frag_idx = [
            i for i, f in enumerate(frag_indice) if set([keep_atom_tag]) & set(f)
        ][0]
        frag_mol = frag_mols[keep_frag_idx]
        frag_mol = SmartsGenerator.remove_ring_dummy_duplicates(frag_mol)
        frag_mol = SmartsGenerator.reset_isotope(frag_mol)
        if remove_dummy:
            frag_mol = SmartsGenerator.dummy_to_hydrogen(frag_mol)
        for atom in frag_mol.GetAtoms():
            atom.SetIsAromatic(False)
        return Chem.MolToSmiles(frag_mol)

    @staticmethod
    def dummy_to_hydrogen(mol):
        atoms = mol.GetAtoms()
        dummy_indice = [at.GetIdx() for at in atoms if at.GetAtomicNum() == 0]
        edit_mol = Chem.EditableMol(mol)
        for idx in dummy_indice:
            edit_mol.ReplaceAtom(idx, Chem.Atom(1))
        mol = edit_mol.GetMol()
        # Chem.SanitizeMol(mol)
        return mol

    @staticmethod
    def reset_isotope(mol):
        [at.SetIsotope(0) for at in mol.GetAtoms()]
        return mol

    def _add_pattern(self, input_dict):
        mol = self.mol
        marker_info = {}
        for i, (patt, info) in enumerate(input_dict.items()):
            if "bond" in info:
                bond = info["bond"]
                bond_type = self.BOND_TYPE[bond]
            else:
                bond_type = Chem.BondType.UNSPECIFIED
            indice = info["indice"]
            for idx in indice:
                mol, added_atom_idx = self.add_dummy_atom(mol, idx, bond_type)
                # Set isotope marker
                at = mol.GetAtomWithIdx(added_atom_idx)
                at.SetIsotope(i + 1)
                marker = f"[{i + 1}#0]"
                if patt in self.INHERENT_SMARTS:
                    patt = self.INHERENT_SMARTS[patt]
                marker_info[marker] = patt
        self.mol = mol
        return marker_info

    @staticmethod
    def remove_chirality(mol):
        Chem.RemoveStereochemistry(mol)
        return mol

    def get_smarts(self, add_dict: Optional[Dict] = None, chiral_dict: Dict = {}):
        if self.set_chiral:
            self._add_chiral(chiral_dict)
        if add_dict is None:
            add_dict = {}
        marker_info = self._add_pattern(add_dict)
        smarts = preprocess_aromatic_rings(self.mol)
        if marker_info:
            for k, v in marker_info.items():
                smarts = smarts.replace(k, v)
        smarts = self.dummy_smarts_to_star(smarts)
        self.smarts = smarts
        self.data = {
            "smiles": self.smiles,
            "smarts": self.smarts,
            "add_hs": self.add_hs,
            "set_chiral": self.set_chiral,
        }
        return self

    @staticmethod
    def remove_ring_dummy_duplicates(mol):
        editable_mol = Chem.EditableMol(mol)
        isotope_count = {}
        atoms_to_remove = []
        atoms = [at for at in mol.GetAtoms() if at.GetAtomicNum() == 0]
        for atom in atoms:
            isotope = atom.GetIsotope()
            if isotope not in isotope_count:
                isotope_count[isotope] = [atom.GetIdx()]
            else:
                isotope_count[isotope].append(atom.GetIdx())
        for indices in isotope_count.values():
            if len(indices) > 1:
                atoms_to_remove.extend(indices[1:])
        for idx in sorted(atoms_to_remove, reverse=True):
            editable_mol.RemoveAtom(idx)
        return editable_mol.GetMol()

    def save(self, filename, save_mode="wb"):
        with open(filename, save_mode) as file:
            pickle.dump(self.data, file)
        return filename

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


def preprocess_aromatic_rings(mol: Chem.Mol) -> str:
    if mol is None:
        raise ValueError("Invalid molecule object")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    # Get ring information
    ring_info = mol.GetRingInfo()
    atom_rings = (
        ring_info.AtomRings()
    )  # List of rings (each ring is a tuple of atom indices)
    ring_atoms = {
        atom_idx for ring in atom_rings for atom_idx in ring
    }  # Set of all atoms in rings
    ring_bonds = set()  # Set to store ring bond indices
    # Clone the molecule for modifications
    emol = Chem.RWMol(mol)
    # Collect bonds that are part of rings
    for bond in emol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        # If both atoms of the bond are in rings, it's a ring bond
        if begin_idx in ring_atoms and end_idx in ring_atoms:
            ring_bonds.add((begin_idx, end_idx))

    # Record aromatic atoms and bonds before Kekulization
    aromatic_atoms = set()
    aromatic_bonds = set()
    for atom in emol.GetAtoms():
        if (
            atom.GetIsAromatic() and atom.GetIdx() in ring_atoms
        ):  # Only consider atoms in rings
            aromatic_atoms.add(atom.GetIdx())
    for bond in emol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if bond.GetIsAromatic() and (begin_idx, end_idx) in ring_bonds:
            aromatic_bonds.add((begin_idx, end_idx))

    # Kekulize the entire molecule
    Chem.Kekulize(emol, clearAromaticFlags=True)

    # Restore aromaticity for atoms and bonds inside rings
    for atom in emol.GetAtoms():
        if atom.GetIdx() in aromatic_atoms:
            atom.SetIsAromatic(True)  # Restore aromatic flag for atoms
    for bond in emol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if (begin_idx, end_idx) in aromatic_bonds or (
            end_idx,
            begin_idx,
        ) in aromatic_bonds:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)

    # Convert the modified molecule to SMARTS
    smarts = Chem.MolToSmarts(emol)
    return smarts
