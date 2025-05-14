import numpy as np
import pandas as pd
from rdkit import Chem
from features.calculate import (
    cal_features,
    NUM_FEATURES,
    compute_feature_of_bank,
    evaluate_qip,
    QIP_FEATURE_CACHE,
)
from loguru import logger


def init_bank(file_name, nbank=None, nsmiles=None, rseed=None):
    np.random.seed(rseed)

    _df = pd.read_csv(file_name)
    df = _df[:nsmiles].values

    shuffled_index = np.random.permutation(len(df))
    tmp_bank = df[shuffled_index][:nbank]
    df = pd.DataFrame(tmp_bank, columns=_df.columns.tolist())
    df.to_csv("init_bank.csv", index=False)

    del df, _df

    bank = np.empty([tmp_bank.shape[0], 3 + NUM_FEATURES], dtype=object)
    bank[:, 0] = tmp_bank[:, 0]  # SMILES
    bank[:, 2] = True  # usable label True

    for i, j in enumerate(bank[:, 0]):
        mol = Chem.MolFromSmiles(j)
        bank[i, 1] = mol
        Chem.Kekulize(mol)
        bank[i, 0] = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)

    bank = compute_feature_of_bank(bank)

    return bank
