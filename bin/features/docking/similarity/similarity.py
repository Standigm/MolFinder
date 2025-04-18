from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional  # Add imports from typing

from ...utils import sdf_handler
from ...utils.utils import get_default_output_path
from ..conformers.conformer_generator import ConformerGenerator
from ..conformers.oeomega import OEOmegaCLI

STR_TO_CONF = {
    "omega": OEOmegaCLI,
}

STR_TO_SDF_HANDLER = {
    "omega": sdf_handler,
}


class SimilarityScorer(ABC):
    def __init__(self):
        super().__init__()
        self.job_name = "similarity"
        self.get_confs_from = ""

    @abstractmethod
    def configure(self, **options):
        pass

    @abstractmethod
    def run(
        self,
        dbase_path: str,
        query_path: str,
        output_path: Optional[str] = None,
        conformer_config: Optional[Dict] = None,
    ) -> str:
        pass

    # ? Consider letting get_confs_from be an instance of ConformerGenerator
    # ? Pro: easier to configure the conformer generator if required
    # ? Con: might make it harder to parallelize via Slurm
    # Probably make this be a separate method so that it can be used in ligprep too
    def get_conformers(
        self, input_path: str, temp_dir: Path, conformer_config: Optional[Dict] = None
    ) -> str:
        conf_gen: ConformerGenerator = STR_TO_CONF[self.get_confs_from]()

        if conformer_config:
            conf_gen.configure(**conformer_config)

        conf_output = (
            temp_dir / f"{conf_gen.job_name}_output-{Path(input_path).stem}.sdf"
        ).as_posix()
        conf_gen.run(input_path, conf_output)

        # Prepare proper sdf_handler to parse conformer generator output
        # Relevant for similarity scorers that aren't dependent on openeye
        self.sdf_handler = STR_TO_SDF_HANDLER[self.get_confs_from]

        return conf_output


class SimilarityScorerCLI(SimilarityScorer):
    def __init__(self):
        super().__init__()
        self.temp_dir: Optional[str] = None

    def run(
        self,
        dbase_path: str,
        query_path: str,
        output_path: Optional[str] = None,
        conformer_config: Optional[Dict] = None,
    ) -> str:
        # Resolve paths before calling subprocess.run with cwd
        dbase_path = Path(dbase_path).resolve().as_posix()
        query_path = Path(query_path).resolve().as_posix()
        if not output_path:
            output_path = get_default_output_path(dbase_path, self.job_name)
        else:
            output_path = Path(output_path).resolve().as_posix()

        if self.temp_dir is None:
            with TemporaryDirectory() as temp_dir:
                self.run_in_dir(
                    dbase_path,
                    query_path,
                    output_path,
                    temp_dir=Path(temp_dir),
                    conformer_config=conformer_config,
                )
        else:
            temp_dir = Path(self.temp_dir).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.run_in_dir(
                dbase_path,
                query_path,
                output_path,
                temp_dir=temp_dir,
                conformer_config=conformer_config,
            )

        return output_path

    @abstractmethod
    def set_temp_dir(self, temp_dir: Optional[str]):
        pass

    @abstractmethod
    def run_in_dir(
        self,
        dbase_path: str,
        query_path: str,
        output_path: str,
        temp_dir: Path,
        args: Optional[Dict[str, str]] = None,
        conformer_config: Optional[Dict] = None,
    ):
        pass
