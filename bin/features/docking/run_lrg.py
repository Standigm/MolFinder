import datamol
from ligprep.oe_ligprep.oe_ligprep import LigPrep

from features.docking.gold.gold import GoldDock
from features.docking.similarity.rocs import Rocs
from features.docking.similarity.sdf_handler import (
    concat_sdf,
    filter_sdf,
    get_best_scores_from_sdf,
)

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


def run_gold(
    input_path: str, output_path: str, target: str, config_path: str, where: int = 0
):
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
    gold_dock.run(where)


import pandas as pd
from loguru import logger

if __name__ == "__main__":
    # df = pd.read_csv("/db2/users/wonseokshin/sandbox/MolFinder/bin/init_bank.csv")
    # df["inchikey"] = [datamol.to_inchikey(mol) for mol in df["SMILES"]]
    # df.to_csv("/db2/users/wonseokshin/sandbox/MolFinder/bin/init_bank.csv", index=False)
    logger.info("Starting LRG Pipeline!")
    ligprep_output_path = "./ligprep_example.sdf"
    run_ligprep(
        input_path="/db2/users/wonseokshin/sandbox/MolFinder/bin/features/docking/lrg_sample.csv",
        output_path=ligprep_output_path,
        target="abl",
    )
    logger.info("Ligprep done")
    filtered_ligprep_output_path = filter_sdf(
        "./ligprep_example.sdf",
        rank_by="energy",
        keep_lowest_values=True,
        n_per_inchikey=5,
    )
    logger.info("Ligprep Filtering done")

    rocs_output_path = "./rocs_example.sdf"
    run_rocs(
        input_path=filtered_ligprep_output_path,
        output_path=rocs_output_path,
        target="abl",
    )
    logger.info("ROCS done")

    # # Run GoldDock
    # # Keep top N results
    filtered_rocs_output_path = filter_sdf(
        rocs_output_path,
        rank_by="ROCS_TanimotoCombo",
        keep_lowest_values=False,
        n_per_inchikey=1,
    )
    logger.info("ROCS Filtering done")
    logger.info("Running GoldDock")

    gold_output_path = "./gold_example.sdf"
    run_gold(
        input_path=filtered_rocs_output_path,
        output_path="./gold_example.sdf",
        target="abl",
        config_path="./gold_conf.conf",
        where=0,
    )
    logger.info("GoldDock Done")

    filtered_gold_output_path = filter_sdf(
        gold_output_path,
        rank_by="Gold.PLP.Fitness",
        keep_lowest_values=False,
        n_per_inchikey=1,
    )

    logger.info("Scoring:")
    print(
        get_best_scores_from_sdf(
            input_sdf_path=filtered_gold_output_path,
            select_by="Gold.PLP.Fitness",
            keep_lowest_scores=False,
        )
    )
    print()
