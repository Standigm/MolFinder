import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from openeye import oechem
from rdkit import Chem

from .gold_config import write_config
from .gold_scaffold import get_gold_scaffold
from .mcs import MCS


def set_gold_license():
    os.environ["CCDC_LICENSING_CONFIGURATION"] = (
        "lf-server;http://192.168.2.200:9876;num-retries=10"
    )


def oe_struct_converter(input_path, output_path):
    ifs = oechem.oemolistream(input_path)
    ofs = oechem.oemolostream(output_path)
    for mol in ifs.GetOEGraphMols():
        oechem.OEWriteMolecule(ofs, mol)


class GoldDock:
    GOLD_EXE = "/db2/CCDC/ccdc-software/gold/GOLD/bin/gold_auto"
    set_gold_license()

    def __init__(
        self,
        reference_ligand_path: str,
        ligand_path: str,
        pdb_path: str,
        output_path: str,
        config_path: str,
        autoscale: float = 1,
        binding_site_size: float = 15,
        n_pose: int = 50,
        fitness_function: Optional[List[str]] = None,
        rescore_mode: bool = False,
        temp_on_disk: bool = False,
    ):
        if fitness_function is None:
            fitness_function = ["plp", "chemscore"]
        self.reference_ligand_path = reference_ligand_path
        self.ligand_path = ligand_path
        self.pdb_path = pdb_path
        self.output_path = os.path.abspath(output_path)
        self.config_path = config_path
        self.autoscale = autoscale
        self.binding_site_size = binding_site_size
        self.binding_site_origin_coordinates = self.get_bs_coordinate(
            self.reference_ligand_path
        )
        self.n_pose = n_pose
        self.fitness_function = fitness_function
        self.rescore_mode = rescore_mode
        self.temp_on_disk = temp_on_disk

        self.constraint = None
        self.constraint_args = {}
        self.relative_ligand_energy = 1

    def set_similarity_constraint(self, constraint_weight=5):
        reference_path = self.reference_ligand_path
        reference_mol2 = os.path.splitext(reference_path)[0] + ".mol2"
        if not os.path.exists(reference_mol2):
            oe_struct_converter(reference_path, reference_mol2)
            os.sync()
        constraint_args = {
            "reference_mol2": reference_mol2,
            "constraint_weight": constraint_weight,
        }
        self.constraint = "similarity"
        self.constraint_args = constraint_args

    def set_scaffold_constraint(
        self, scaffold_ligand_path, scaffold_smiles_list, constraint_weight=5
    ):
        scaffold_path = os.path.splitext(scaffold_ligand_path)[0] + "-scaffold.sdf"
        get_gold_scaffold(scaffold_ligand_path, scaffold_path, scaffold_smiles_list)
        constraint_args = {
            "scaffold_path": scaffold_path,
            "constraint_weight": constraint_weight,
        }
        self.constraint = "scaffold"
        self.constraint_args = constraint_args

    def set_mcs_constraint(self, mcs_referece_path, constraint_weight=5):
        mcs_path = os.path.splitext(self.ligand_path)[0] + "-mcs.sdf"

        suppl = Chem.SDMolSupplier(mcs_referece_path)
        ref_mol = suppl[0]
        ref_smiles = Chem.MolToSmiles(ref_mol)

        suppl = Chem.SDMolSupplier(self.ligand_path)
        ligand_mol = suppl[0]
        ligand_smiles = Chem.MolToSmiles(ligand_mol)

        # MCS
        mcs = MCS(ref_smiles, ligand_smiles, smiles_canonicalize=False)
        mcs_indice = ref_mol.GetSubstructMatch(mcs.mcs_pattern, useChirality=False)
        remove_indice = sorted(
            [at.GetIdx() for at in ref_mol.GetAtoms() if at.GetIdx() not in mcs_indice],
            reverse=True,
        )
        edit_mol = Chem.EditableMol(ref_mol)
        [edit_mol.RemoveAtom(i) for i in remove_indice]
        mol = edit_mol.GetMol()
        # Aromacity sanitize
        [
            at.SetIsAromatic(False)
            for at in mol.GetAtoms()
            if at.GetIsAromatic() and not at.IsInRing()
        ]
        # Hydrogen should be removed for Gold scaffold constraint docking.
        mol = Chem.RemoveAllHs(mol)
        i = 0
        while True:
            i += 1
            if i == 3:
                break
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
        w = Chem.SDWriter(mcs_path)
        w.write(mol)
        w.close()
        constraint_args = {
            "scaffold_path": mcs_path,
            "constraint_weight": constraint_weight,
        }
        self.constraint = "scaffold"
        self.constraint_args = constraint_args

    def _write_rescore_config(self, temp_dir):
        if len(self.fitness_function) != 0:
            print(
                "Rescore mode needs one fitness function. Select one fitness function in force."
            )
            self.fitness_function = [self.fitness_function[0]]
        write_config(
            self.ligand_path,
            self.pdb_path,
            self.config_path,
            self.output_path,
            self.autoscale,
            self.binding_site_size,
            self.binding_site_origin_coordinates,
            self.n_pose,
            temp_dir,
            self.constraint,
            self.constraint_args,
            0,
            self.fitness_function,
            "RESCORE retrieve",
        )
        return self.config_path

    def _write_config(self, temp_dir):
        write_config(
            self.ligand_path,
            self.pdb_path,
            self.config_path,
            self.output_path,
            self.autoscale,
            self.binding_site_size,
            self.binding_site_origin_coordinates,
            self.n_pose,
            temp_dir,
            self.constraint,
            self.constraint_args,
            1,
            self.fitness_function,
        )
        return self.config_path

    def get_bs_coordinate(self, sdf):
        ifs = oechem.oemolistream()
        if not ifs.open(sdf):
            raise OSError(f"Cannot read {sdf}.")
        ref_mol = next(ifs.GetOEGraphMols())
        c = ref_mol.GetCoords()
        c = np.mean((list(c.values())), axis=0)
        x = c[0]
        y = c[1]
        z = c[2]
        return f"{x},{y},{z}"

    def run_gold(self):
        return subprocess.run([self.GOLD_EXE, self.config_path])

    def run(self):
        try:
            if not self.temp_on_disk:
                with tempfile.TemporaryDirectory() as temp_dir:
                    if self.rescore_mode:
                        self._write_rescore_config(temp_dir)
                    else:
                        self._write_config(temp_dir)
                    os.sync()
                    self.run_gold()
                    os.sync()
                return self.output_path
            else:
                temp_dir_path = (
                    Path(self.output_path).parent / Path(self.ligand_path).stem
                )
                temp_dir_path.mkdir(exist_ok=True, parents=True)
                if self.rescore_mode:
                    self._write_rescore_config(str(temp_dir_path))
                else:
                    self._write_config(str(temp_dir_path))
                os.sync()
                self.run_gold()
                os.sync()
                return self.output_path
        except Exception as e:
            if Path(self.output_path).exists():
                os.remove(self.output_path)
            raise e


# gd = GoldDock(
#     reference_ligand_path = "gold_sample/4twpA_lig.sdf",
#     ligand_path = "gold_sample/2v7aA_lig.sdf",
#     pdb_path = "gold_sample/4twpA.pdb",
#     output_path = "./",
#     config_path = "./sample.conf",
# )
# )
