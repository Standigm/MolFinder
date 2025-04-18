from openeye import oechem
from rdkit import Chem
from rdkit.Chem import rdRascalMCES

# from .utils import get_bonds_between_atoms


def get_bonds_between_atoms(mol, atoms):
    bonds = []
    for bond in mol.GetBonds():
        bgn_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if bgn_idx in atoms and end_idx in atoms:
            bonds.append(bond.GetIdx())


class MCS:
    def __init__(self, smi1, smi2, smiles_canonicalize=True):
        if smiles_canonicalize:
            smi1 = Chem.CanonSmiles(smi1)
            smi2 = Chem.CanonSmiles(smi2)
        self.smi1 = smi1
        self.smi2 = smi2
        self.mol1 = Chem.MolFromSmiles(smi1)
        self.mol2 = Chem.MolFromSmiles(smi2)
        self.mcs_results = self._get_rascal_mcs()
        atom_matches = list(zip(*self.mcs_results.atomMatches()))
        bond_matches = list(zip(*self.mcs_results.bondMatches()))
        self.mol1_mcs_atom = atom_matches[0]
        self.mol2_mcs_atom = atom_matches[1]
        self._atom_in_complete_ring()
        self.mol1_mcs_bond = bond_matches[0]
        self.mol2_mcs_bond = bond_matches[1]

    def _atom_in_complete_ring(self):
        mols = [self.mol1, self.mol2]
        mcs_atoms = [self.mol1_mcs_atom, self.mol2_mcs_atom]
        complete_indice_list = []
        for mol, mcs_atom in zip(mols, mcs_atoms):
            ring_info = mol.GetRingInfo()
            complete_required_rings = [
                r for r in ring_info.AtomRings() if set(r) & set(mcs_atom)
            ]
            remove_list = []
            for ring in complete_required_rings:
                if set(ring) - set(mcs_atom):
                    remove_list.extend(ring)
            complete_rings = list(set(mcs_atom) - set(remove_list))
            complete_indice = [mcs_atom.index(i) for i in complete_rings]
            complete_indice_list.append(complete_indice)
        both_complete_indice = list(
            set(complete_indice_list[0]) & set(complete_indice_list[1])
        )
        self.mol1_mcs_atom = [self.mol1_mcs_atom[i] for i in both_complete_indice]
        self.mol2_mcs_atom = [self.mol2_mcs_atom[i] for i in both_complete_indice]

    def _get_rascal_mcs(
        self,
    ):
        opts = rdRascalMCES.RascalOptions()
        opts.completeAromaticRings = True
        mcs_results = rdRascalMCES.FindMCES(self.mol1, self.mol2, opts)
        return mcs_results[0]

    # def _set_mcs_pattern(self):
    #     omcs = OpeneyeMCS(self.smi1, self.smi2)
    #     omcs.set_mcs_config()
    #     omcs.get_mcs()
    #     mcs = rdFMCS.FindMCS(
    #         [self.mol1, Chem.MolFromSmiles(Chem.CanonSmiles(omcs.mcs_smiles))],
    #         completeRingsOnly=True,
    #         # ringMatchesRingOnly=True,
    #         # ringCompare=Chem.rdFMCS.RingCompare.StrictRingFusion,
    #         timeout=15,
    #     )
    #     self.mcs_pattern = mcs.queryMol


#         oe_mols = [omcs.mol1,omcs.mol2]
#         oe_mcs_atoms = [omcs.mol1_atoms, omcs.mol2_atoms]
#         mols = []
#         mcs_atoms = []
#         for oe_mol, oe_mcs_atom in zip(oe_mols, oe_mcs_atoms):
#             for at in oe_mol.GetAtoms():
#                 if at.GetAtomicNum()==1:
#                     continue;
#                 if at.GetIdx() in oe_mcs_atom:
#                     at.SetIsotope(2)
#             smiles = Chem.CanonSmiles(oechem.OEMolToSmiles(oe_mol))
#             mol = Chem.MolFromSmiles(smiles)
#             mols.append(mol)
#             mcs_atom = []
#             for at in mol.GetAtoms():
#                 if at.GetIsotope()==2:
#                     at.SetIsotope(0)
#                     mcs_atom.append(at.GetIdx())
#             mcs_atoms.append(mcs_atom)
#         mcs_mols = []
#         for mol, mcs_atom in zip(mols, mcs_atoms):
#             complete_ring_mcs_atom = self._get_complete_ring_indice(mol, mcs_atom)
#             link_bonds = self._get_mcs_link_bonds(mol, complete_ring_mcs_atom)
#             Chem.FragmentOnBonds(mol, link_bonds)

#         complete_mcs_pattern = self._get_frag_part(Chem.CombineMols(*mcs_mols), False)
#         self.complete_mcs_pattern = complete_mcs_pattern

#     def _get_frag_part(self, combine_mol, larger=True):
#         frags = Chem.GetMolFrags(combine_mol, asMols=True)
#         if larger:
#             func = np.argmax
#         else:
#             func = np.argmin
#         return frags[func([m.GetNumAtoms() for m in frags])]


class OpeneyeMCS:
    def __init__(self, smi1, smi2):
        self.smi1 = Chem.CanonSmiles(smi1)
        self.smi2 = Chem.CanonSmiles(smi2)
        self.mol1 = self.get_mol(smi1)
        self.mol2 = self.get_mol(smi2)

    def get_mol(self, smi):
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        return mol

    def set_mcs_config(
        self,
        atomexpr: int = oechem.OEExprOpts_DefaultAtoms,
        bondexpr=oechem.OEExprOpts_ExactBonds,
        search_type=oechem.OEMCSType_Exhaustive,
        mcs_func=oechem.OEMCSMaxAtomsCompleteCycles(),
        min_atom_num=3,
        max_matches=1,
        unique_match=True,
    ):
        self.atomexpr = atomexpr
        self.bondexpr = bondexpr
        self.mcss = oechem.OEMCSSearch(
            self.mol1, self.atomexpr, self.bondexpr, search_type
        )
        self.mcss.SetMCSFunc(mcs_func)
        self.mcss.SetMinAtoms(min_atom_num)
        self.mcss.SetMaxMatches(max_matches)
        self.unique = unique_match

    def get_mcs(self):
        matches = self.mcss.Match(self.mol2, self.unique)
        if not matches:
            raise Exception("No Maximum Common Structure exists.")
        for match in matches:
            break
        atom_sets = [
            (atom.pattern.GetIdx(), atom.target.GetIdx()) for atom in match.GetAtoms()
        ]
        self.mol1_atoms, self.mol2_atoms = zip(*atom_sets)
        self.mcs_mol = oechem.OEGraphMol()
        oechem.OESubsetMol(self.mcs_mol, match, True)
        self.mcs_smiles = oechem.OEMolToSmiles(self.mcs_mol)
        self.match = match
        return self
