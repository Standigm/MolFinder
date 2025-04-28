from tempfile import NamedTemporaryFile
from typing import List

import pandas as pd
from rdkit import Chem

from ..scoring.components import docking_components, qip_components
from .docking.run_pipeline import run_iteration
from .qip.run_qip import run_qip

score_columns = list({**qip_components, **docking_components}.keys())


def calculate(smiles_list: List[str], target, output_dir, iteration: int):
    with NamedTemporaryFile(suffix=".csv", delete=False) as f:
        input_path = f.name
        input_df = pd.DataFrame(smiles_list, columns=["smiles"])

        # Add inchikey column
        input_df["inchikey"] = [
            Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) for smiles in smiles_list
        ]

        input_df.to_csv(input_path, index=False)

        # Run QIP-ADMET
        qip_df = run_qip(input_path)

        # Run docking
        best_rocs, best_gold = run_iteration(target, iteration, input_path, output_dir)

        # Merge results onto input_df on inchikey
        # if computation failed, score will be NaN
        input_df = input_df.join(qip_df, on="inchikey", how="left")
        input_df.join(best_rocs, on="inchikey", how="left")
        input_df.join(best_gold, on="inchikey", how="left")

        return input_df[score_columns].to_numpy()
