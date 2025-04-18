import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

admet_list = [
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


def get_env_path():
    return "/db2/users/jingyulee/miniconda3/envs/qip/bin"


def run_qip(input_path):
    with NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name
        env_path = get_env_path()
        env = os.environ.copy()
        env["PATH"] = f"{env_path}:{env['PATH']}"
        current_file_path = Path(os.path.abspath(__file__)).parent / "qip_script.py"
        command = [
            "python",
            current_file_path,
            "--csv_path",
            input_path,
            "--smiles_column",
            "smiles",
            "--output_path",
            output_path,
        ]

        # Execute command and wait for it to finish
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        _, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Error in subprocess: {stderr.decode()}")

        # Load the output CSV into a DataFrame
        df_qip = pd.read_csv(output_path)
        return df_qip[admet_list]


# ['caco2_wang',
#  'hia_hou',
#  'pgp_broccatelli',
#  'bioavailability_ma',
#  'lipophilicity_astrazeneca',
#  'solubility_aqsoldb',
#  'bbb_martins',
#  'ppbr_az',
#  'vdss_lombardo',
#  'cyp2d6_veith',
#  'cyp3a4_veith',
#  'cyp2c9_veith',
#  'cyp2d6_substrate_carbonmangels',
#  'cyp3a4_substrate_carbonmangels',
#  'cyp2c9_substrate_carbonmangels',
#  'half_life_obach',
#  'clearance_microsome_az',
#  'clearance_hepatocyte_az',
#  'herg',
#  'ames',
#  'dili',
#  'ld50_zhu']
