import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
import pandas as pd


# TODO: Solve environment issue
def run_qip(input_path):
    with NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name
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
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Error in subprocess: {stderr.decode()}")

        return pd.read_csv(output_path)
