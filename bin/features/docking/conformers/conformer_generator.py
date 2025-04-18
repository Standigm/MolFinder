from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

from rdkit import Chem

from ...utils.utils import get_default_output_path


class ConformerGenerator(ABC):
    def __init__(self):
        super().__init__()
        self.job_name = "conformers"

    @abstractmethod
    def configure(self, **options):
        pass

    @abstractmethod
    def run(self, input_path: str, output_path: Optional[str] = None) -> str:
        pass

    @staticmethod
    def add_sddata_to_rdmol(
        mol, energy: float, variant_count: int, overwrite: bool = False
    ):
        properties = {
            "energy": str(energy),
            "smiles": Chem.MolToSmiles(mol),
            "variant": str(variant_count),
            "inchikey": Chem.MolToInchiKey(mol),
        }

        for prop, value in properties.items():
            if overwrite or not mol.HasProp(prop):
                mol.SetProp(prop, value)
        # ? Should the title be overwritten with the inchikey?
        return mol


class ConformerGeneratorCLI(ConformerGenerator):
    def __init__(self):
        super().__init__()
        self.temp_dir: Optional[str] = None

    def run(self, input_path: str, output_path: Optional[str] = None) -> str:
        # Resolve paths before calling subprocess.run with cwd
        input_path = Path(input_path).resolve().as_posix()
        if not output_path:
            output_path = get_default_output_path(input_path, self.job_name)
        else:
            output_path = Path(output_path).resolve().as_posix()

        if self.temp_dir is None:
            with TemporaryDirectory() as temp_dir:
                self.run_in_dir(input_path, output_path, Path(temp_dir))
        else:
            temp_dir = Path(self.temp_dir).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.run_in_dir(input_path, output_path, temp_dir)

        return output_path

    @abstractmethod
    def set_temp_dir(self, temp_dir: str):
        pass

    @abstractmethod
    def run_in_dir(
        self,
        input_path: str,
        output_path: str,
        temp_dir: Path,
        args: Optional[Dict[str, str]] = None,
    ):
        pass
