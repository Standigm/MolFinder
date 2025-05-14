from tempfile import NamedTemporaryFile
from typing import List
import numpy as np
import datamol
from loguru import logger
import pandas as pd
from rdkit.Chem.Crippen import MolLogP  # type:ignore
from rdkit import Chem
from features.qip_features.run_qip import run_qip
from scoring.sigmoid import sigmoid_funcs, obj_weights

# from ..scoring.components import docking_components, qip_components
# from .docking.run_pipeline import run_iteration
# from .qip_features.run_qip import run_qip

# score_columns = list({**qip_components, **docking_components}.keys())


# def calculate(smiles_list: List[str], target, output_dir, iteration: int):
#     with NamedTemporaryFile(suffix=".csv", delete=False) as f:
#         input_path = f.name
#         input_df = pd.DataFrame(smiles_list, columns=["smiles"])

#         # Add inchikey column
#         input_df["inchikey"] = [
#             Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) for smiles in smiles_list
#         ]

#         input_df.to_csv(input_path, index=False)

#         # Run QIP-ADMET
#         qip_df = run_qip(input_path)

#         # Run docking
#         best_rocs, best_gold = run_iteration(target, iteration, input_path, output_dir)

#         # Merge results onto input_df on inchikey
#         # if computation failed, score will be NaN
#         input_df = input_df.join(qip_df, on="inchikey", how="left")
#         input_df.join(best_rocs, on="inchikey", how="left")
#         input_df.join(best_gold, on="inchikey", how="left")

#         return input_df[score_columns].to_numpy()


GOLD_FEATURE_CACHE = dict()
QIP_FEATURE_CACHE = dict()
QIP_FEATURE_COLUMNS = [
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
FEATURE_COLUMNS = [
    "SAS",
    "QED",
    "ALogP",
] + QIP_FEATURE_COLUMNS
FEATURE_IDX = {name: idx + 3 for idx, name in enumerate(FEATURE_COLUMNS)}
NUM_FEATURES = len(FEATURE_COLUMNS)
logger.info(f"NUM_FEATURES = {NUM_FEATURES}")


def gold_features(smi):
    global GOLD_FEATURE_CACHE
    if smi in GOLD_FEATURE_CACHE:
        return GOLD_FEATURE_CACHE[smi]
    else:
        raise ValueError(f"??? GOLD feature for {smi} is not cached yet.")


def qip_features(smi):
    global QIP_FEATURE_CACHE
    inchikey = datamol.to_inchikey(smi)
    feature_dict = {}
    if smi in QIP_FEATURE_CACHE:
        feature_dict = QIP_FEATURE_CACHE[smi]
    elif inchikey in QIP_FEATURE_CACHE:
        feature_dict = QIP_FEATURE_CACHE[inchikey]
    else:
        raise ValueError(f"??? QIP feature for {smi} is not cached yet.")
    return [feature_dict[col] for col in QIP_FEATURE_COLUMNS]


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


def cal_features(_smi, _mol, lazy_compute=True):  # set data_column
    if lazy_compute:
        return [_smi, _mol, True] + [None] * NUM_FEATURES
    else:
        return (
            [
                _smi,
                _mol,
                True,
                feature_sascore(_mol),
                feature_qed(_mol),
                feature_alogp(_mol),
            ]
            # + gold_features(_smi)
            + qip_features(_smi)
        )


def compute_feature_of_bank(bank):
    logger.info(f"Computing features of bank... (size = {len(bank)})")
    smiles_list = bank[:, 0].tolist()
    evaluate_qip(smiles_list)
    for i in range(len(bank)):
        bank[i, 3:] = cal_features(bank[i, 0], bank[i, 1], lazy_compute=False)[3:]
    return bank


def obj_fn(x):
    score = np.zeros(x.shape[0])
    for feature, idx in FEATURE_IDX.items():
        s_i = np.asarray([sigmoid_funcs[feature](val) for val in x[:, idx]])
        w_i = obj_weights[feature]
        score += -np.log(s_i * w_i)
    return score


def evaluate_qip(smiles_list: str):
    for smiles in smiles_list:
        if smiles in QIP_FEATURE_CACHE:
            continue
        inchikey = datamol.to_inchikey(smiles)
        if inchikey in QIP_FEATURE_CACHE:
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
                QIP_FEATURE_CACHE[row["smiles"]] = row
            except Exception as e:
                pass
            try:
                QIP_FEATURE_CACHE[row["inchikey"]] = row
            except Exception as e:
                pass
            try:
                kekule_smiles = Chem.MolToSmiles(
                    Chem.MolFromSmiles(row["smiles"]),
                    kekuleSmiles=True,
                    isomericSmiles=False,
                )
                QIP_FEATURE_CACHE[kekule_smiles] = row
            except Exception as e:
                pass
    logger.info(f"QIP features for {len(smiles_list)} smiles are additionally cached.")
    return None
