from tempfile import NamedTemporaryFile
from typing import List
import numpy as np
import datamol
from loguru import logger
import pandas as pd
from rdkit.Chem.Crippen import MolLogP  # type:ignore
from rdkit import Chem
from features.docking.run_lrg import (
    run_ligprep,
    run_rocs,
    run_gold,
    get_best_scores_from_sdf,
)
from features.docking.similarity.sdf_handler import filter_sdf
from features.qip_features.run_qip import run_qip
from scoring.sigmoid import sigmoid_funcs, obj_weights
from itertools import chain
from std_chem.parallel.backends import JoblibJobRunner, SlurmJobRunner
import os
from contextlib import redirect_stdout, redirect_stderr

N_JOBS = 2500
TARGET = "abl"
SCAFFOLD_SMILES = (
    "*c1ccccc1C(=O)NC" if TARGET == "abl" else "*c1cc(*)c(*)c(CN2CCC(*)CC2)c1"
)
logger.error(f"TARGET = {TARGET}, SCAFFOLD_SMILES = {SCAFFOLD_SMILES}")
# SCAFFOLD_SMILES = "c1ccccc1C=CC(Cl)=C"
SCAFFOLD_PATTERN = Chem.MolFromSmarts(SCAFFOLD_SMILES)


def _chunkify(data, n_chunks):
    """Split `data` into n_chunks lists of nearly equal size."""
    n_chunks = min(n_chunks, len(data))
    k, m = divmod(len(data), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        # distribute the “extra” m items one per chunk for the first m chunks
        size = k + (1 if i < m else 0)
        if size == 0:
            continue
        chunks.append(data[start : start + size])
        start += size
    return chunks


def feature_sascore(mol):
    """
    Calculate SAScore for a given molecule.
    """
    return datamol.descriptors.sas(mol)


def feature_qed(mol):
    """
    Calculate QED for a given molecule.
    """
    return datamol.descriptors.qed(mol)


def feature_alogp(x: Chem.Mol):
    return MolLogP(Chem.RemoveHs(x))


class FeatureCalculator:
    def __init__(self):
        self.cache = dict()
        self.feature_columns = []

    def __call__(self, smi):
        inchikey = datamol.to_inchikey(smi)
        feature_dict = {}
        if smi in self.cache:
            feature_dict = self.cache[smi]
        elif inchikey in self.cache:
            feature_dict = self.cache[inchikey]
        return feature_dict

    # Evaluate list and fill cache
    def evaluate(self, smiles_list):
        raise NotImplementedError(
            "evaluate function is not implemented in FeatureCalculator class."
        )


class QIPFeatureCalculator(FeatureCalculator):
    def __init__(self):
        super().__init__()
        self.feature_columns = [
            "caco2_wang",
            "solubility_aqsoldb",
            "ppbr_az",
            "cyp2d6_veith",
            "cyp3a4_veith",
            "cyp2c9_veith",
            "cyp3a4_substrate_carbonmangels",
            "cyp2c9_substrate_carbonmangels",
            "cyp2d6_substrate_carbonmangels",
            "clearance_hepatocyte_az",
            "herg",
        ]

    def __call__(self, smi):
        D = super().__call__(smi)
        return [D[col] for col in self.feature_columns]

    def evaluate(self, smiles_list: list[str]):
        for smiles in smiles_list:
            if smiles in self.cache:
                continue
            inchikey = datamol.to_inchikey(smiles)
            if inchikey in self.cache:
                continue
        df = pd.DataFrame(
            {
                "smiles": smiles_list,
            }
        )
        with NamedTemporaryFile(suffix=".csv", delete=False) as f:
            input_path = f.name
            df.to_csv(input_path, index=False)
            df_qip = run_qip(input_path)
            for row in df_qip.to_dict(orient="records"):
                try:
                    self.cache[row["smiles"]] = row
                except Exception as e:
                    pass
                try:
                    self.cache[row["inchikey"]] = row
                except Exception as e:
                    pass
                try:
                    kekule_smiles = Chem.MolToSmiles(
                        Chem.MolFromSmiles(row["smiles"]),
                        kekuleSmiles=True,
                        isomericSmiles=False,
                    )
                    self.cache[kekule_smiles] = row
                except Exception as e:
                    pass
        logger.info(
            f"QIP features for {len(smiles_list)} smiles are additionally cached."
        )
        return None


class RDKitFeatureCalculator(FeatureCalculator):
    def __init__(self):
        super().__init__()
        self.feature_columns = ["SAS", "QED", "ALogP", "Scaffold"]

    def __call__(self, smi):
        D = super().__call__(smi)
        return [D[col] for col in self.feature_columns]

    def evaluate(self, smiles_list: list[str]):
        for smiles in smiles_list:
            if smiles in self.cache:
                continue
            inchikey = datamol.to_inchikey(smiles)
            if inchikey in self.cache:
                continue
            mol = Chem.MolFromSmiles(smiles)
            self.cache[smiles] = {
                "SAS": feature_sascore(mol),
                "QED": feature_qed(mol),
                "ALogP": feature_alogp(mol),
                "Scaffold": float(mol.HasSubstructMatch(SCAFFOLD_PATTERN)),
            }
            self.cache[inchikey] = self.cache[smiles]
        logger.info(
            f"RDKit features for {len(smiles_list)} smiles are additionally cached."
        )
        return None


class DockingFeatureCalculator(FeatureCalculator):
    def __init__(self):
        super().__init__()
        self.feature_columns = ["rocs", "gold"]

    def __call__(self, smi):
        D = super().__call__(smi)
        return [D[col] for col in self.feature_columns]

    @staticmethod
    def save_ligand_list(smiles_list: list[str], file_path: str):
        inchikey_list = [datamol.to_inchikey(smi) for smi in smiles_list]
        df = pd.DataFrame(
            {
                "smiles": smiles_list,
                "inchikey": inchikey_list,
            }
        )
        df.to_csv(file_path, index=False)

    def lrg_score_eval(self, smiles):
        try:
            smiles_list = [smiles]
            with NamedTemporaryFile(suffix=".csv", delete=False) as ligand_input_file:
                ligand_input_path = ligand_input_file.name
                input_path_stem = ligand_input_path.split(".")[0]
                self.save_ligand_list(smiles_list, ligand_input_path)
                ligprep_out_path = input_path_stem + "_ligprep_out.sdf"
                logger.info(f"Running LigPrep: {smiles_list} -> {ligprep_out_path}")
                run_ligprep(ligand_input_path, ligprep_out_path, target=TARGET)

                filtered_ligand_out_path = filter_sdf(
                    ligprep_out_path,
                    rank_by="energy",
                    keep_lowest_values=True,
                    n_per_inchikey=5,
                )

                rocs_out_path = input_path_stem + "_rocs_out.sdf"
                logger.info(
                    f"Running ROCS: {filtered_ligand_out_path} -> {rocs_out_path}"
                )
                run_rocs(
                    input_path=filtered_ligand_out_path,
                    output_path=rocs_out_path,
                    target=TARGET,
                )

                filtered_rocs_out_path = filter_sdf(
                    rocs_out_path,
                    rank_by="ROCS_TanimotoCombo",
                    keep_lowest_values=False,
                    n_per_inchikey=1,
                )
                gold_out_path = input_path_stem + "_gold_out.sdf"
                logger.info(
                    f"Running GOLD: {filtered_rocs_out_path} -> {gold_out_path}"
                )
                gold_config_path = input_path_stem + "_gold_conf.conf"
                run_gold(
                    input_path=filtered_rocs_out_path,
                    output_path=gold_out_path,
                    target=TARGET,
                    config_path=gold_config_path,
                    where=0,
                )
                logger.info("Finished Running GOLD: Searching best scores")
                filtered_gold_out_path = filter_sdf(
                    gold_out_path,
                    rank_by="Gold.PLP.Fitness",
                    keep_lowest_values=False,
                    n_per_inchikey=1,
                )
                rocs_score_df = get_best_scores_from_sdf(
                    input_sdf_path=filtered_rocs_out_path,
                    select_by="ROCS_TanimotoCombo",
                    keep_lowest_scores=False,
                )
                gold_score_df = get_best_scores_from_sdf(
                    input_sdf_path=filtered_gold_out_path,
                    select_by="Gold.PLP.Fitness",
                    keep_lowest_scores=False,
                )
            return rocs_score_df, gold_score_df
        except Exception as e:
            return None

    def evaluate(self, smiles_list):
        logger.info(f"Running GOLD for {len(smiles_list)} ligands")
        # splitted_smiles_list = np.array_split(smiles_list, N_JOBS)
        # runner = JoblibJobRunner(
        #     n_jobs=N_JOBS, batch_size=1, show_progress=True, backend="multiprocessing"
        # )
        runner = SlurmJobRunner(
            n_jobs=min(N_JOBS, len(smiles_list)),
            batch_size=1,
            show_progress=True,
            slurm_partition="cpu256,cpu96",
            slurm_timeout_min=90,
            cpus_per_task=1,
            slurm_job_name=f"{TARGET}-MolFinder-GOLD",
            log_folder="/db2/users/wonseokshin/sandbox/tmp_dir_slurm",
        )
        chunked_results = runner.run(
            self.lrg_score_eval,
            data=smiles_list,
        )
        for chunked_result in chunked_results:
            try:
                if chunked_result is None:
                    continue
                rocs_score_df, gold_score_df = chunked_result
                joint_df = rocs_score_df.join(
                    gold_score_df.set_index("inchikey"), on="inchikey", how="inner"
                )
                joint_df.rename(
                    {
                        "ROCS_TanimotoCombo": "rocs",
                        "Gold.PLP.Fitness": "gold",
                        "inchikey": "inchikey",
                    },
                    axis=1,
                    inplace=True,
                )
                for row in joint_df.to_dict(orient="records"):
                    try:
                        self.cache[row["inchikey"]] = row
                    except Exception as e:
                        pass
            except Exception as e:
                logger.error(f"Error in evaluate: {e}")
                continue


FEATURE_CALCULATORS = [
    RDKitFeatureCalculator(),
    QIPFeatureCalculator(),
    DockingFeatureCalculator(),
]
FEATURE_COLUMNS = sum(
    [calculator.feature_columns for calculator in FEATURE_CALCULATORS], []
)
FEATURE_IDX = {name: idx + 3 for idx, name in enumerate(FEATURE_COLUMNS)}
logger.debug(f"FEATURE_COLUMNS = {FEATURE_COLUMNS}")
NUM_FEATURES = len(FEATURE_COLUMNS)
logger.info(f"NUM_FEATURES = {NUM_FEATURES}")


def cal_features(_smi, _mol, lazy_compute=True):  # set data_column
    if lazy_compute:
        return [_smi, _mol, True] + [None] * NUM_FEATURES
    else:
        L = [_smi, _mol, True]
        for calculator in FEATURE_CALCULATORS:
            try:
                L.extend(calculator(_smi))
            except Exception as e:
                return None
        return L


def compute_feature_of_bank(bank):
    logger.info(f"Computing features of bank... (size = {len(bank)})")
    smiles_list = bank[:, 0].tolist()
    for calculator in FEATURE_CALCULATORS:
        calculator.evaluate(smiles_list)
    bank_feature_calculated = []
    for i in range(len(bank)):
        bank_item = cal_features(bank[i, 0], bank[i, 1], lazy_compute=False)
        if bank_item is None:
            continue
        else:
            bank_feature_calculated.append(np.array(bank_item))
    logger.info(f"Feature computation ok (size = {len(bank_feature_calculated)})")
    return np.array(bank_feature_calculated)


def obj_fn(x):
    score = np.zeros(x.shape[0])
    for feature, idx in FEATURE_IDX.items():
        if feature == "Scaffold":
            score += (np.array(x[:, idx], np.float64) - 1) * 20
            continue
        s_i = np.asarray([sigmoid_funcs[feature](val) for val in x[:, idx]])
        w_i = obj_weights[feature]
        score += -s_i * w_i
    return score


# NUM_FEATURES = worst, 0 = best
def stella_obj_fn(x):
    score = np.zeros(x.shape[0])
    for feature, idx in FEATURE_IDX.items():
        if feature == "Scaffold":
            continue
        s_i = np.asarray([sigmoid_funcs[feature](val) for val in x[:, idx]])
        w_i = obj_weights[feature]
        score += s_i * w_i
    return score
