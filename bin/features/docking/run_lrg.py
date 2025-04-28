from ligprep.oe_ligprep.oe_ligprep import LigPrep

from .gold.gold import GoldDock
from .similarity.rocs import Rocs

ligprep_configs = {
    "abl": {
        "enum_tautomer": True,
        "enum_protonation": False,
        "flip_chiral": True,
        "max_stereo": 32,
        "low_ph": 7.4,
        "high_ph": 7.4,
    },
    "p53": {
        "enum_tautomer": True,
        "enum_protonation": True,
        "flip_chiral": True,
        "max_stereo": 32,
        "low_ph": 7.3,
        "high_ph": 7.5,
    },
}

rocs_configs = {
    "abl": {
        "reference_ligand_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/abl_kinase/4twpA_lig.sdf",
        "maxconfs": 1,
    },
    "p53": {
        "reference_ligand_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/p53/4agqA_lig.sdf",
        "maxconfs": 1,
    },
}

gold_configs = {
    "abl": {
        "reference_ligand_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/abl_kinase/4twpA_lig.sdf",
        "pdb_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/abl_kinase/4twpA.pdb",
        "autoscale": 0.5,
        "binding_site_size": 15,
        "n_pose": 25,
        "rescore_mode": False,
        "fitness_function": ["plp", "chemscore"],
    },
    "p53": {
        "reference_ligand_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/p53/4agqA_lig.sdf",
        "pdb_path": "/db2/users/jingyulee/jobs/reinvent/stella_paper/inputs/p53/4agqA.pdb",
        "autoscale": 0.5,
        "binding_site_size": 15,
        "n_pose": 25,
        "rescore_mode": False,
        "fitness_function": ["plp", "chemscore"],
    },
}


def run_ligprep(input_path: str, output_path: str, target: str):
    ligprep_config = ligprep_configs[target]
    ligprep = LigPrep(
        low_ph=ligprep_config["low_ph"], high_ph=ligprep_config["high_ph"]
    )
    ligprep.run(
        input_path,
        output_file=output_path,
        enum_tautomer=ligprep_config["enum_tautomer"],
        enum_protonation=ligprep_config["enum_protonation"],
        flip_chiral=ligprep_config["flip_chiral"],
        max_stereo=ligprep_config["max_stereo"],
        title_col="inchikey",
    )


def run_rocs(input_path: str, output_path: str, target: str):
    rocs_config = rocs_configs[target]
    rocs = Rocs()
    rocs.run(
        dbase_path=input_path,
        query_path=rocs_config["reference_ligand_path"],
        output_path=output_path,
    )


def run_gold(input_path: str, output_path: str, target: str, config_path: str):
    gold_config = gold_configs[target]
    gold_dock = GoldDock(
        reference_ligand_path=gold_config["reference_ligand_path"],
        ligand_path=input_path,
        pdb_path=gold_config["pdb_path"],
        output_path=output_path,
        config_path=config_path,
        autoscale=gold_config["autoscale"],
        binding_site_size=gold_config["binding_site_size"],
        n_pose=gold_config["n_pose"],
        fitness_function=gold_config["fitness_function"],
        rescore_mode=gold_config["rescore_mode"],
    )
    gold_dock.run()
