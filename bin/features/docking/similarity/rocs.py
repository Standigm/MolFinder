import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from ...utils.utils import set_oe_env
from .similarity import SimilarityScorerCLI

# from stella_tasks.utils.utils import check_file_suffix, get_logger

ROCS_COLS = [
    "ROCS_ShapeQuery",
    "ROCS_TanimotoCombo",
    "ROCS_ShapeTanimoto",
    "ROCS_ColorTanimoto",
    "ROCS_RefTversky",
    "ROCS_RefColorTversky",
    "ROCS_RefTverskyCombo",
    "ROCS_FitTversky",
    "ROCS_FitColorTversky",
    "ROCS_FitTverskyCombo",
    "ROCS_ColorScore",
]


@dataclass
class RocsConfig:
    maxconfs: int = 1
    cutoff: float = -1.0


class Rocs(SimilarityScorerCLI):
    """
    Example usage:

    Rocs(get_confs_from=get_confs_from)
    .set_temp_dir(temp_dir=temp_dir)  # Can be skipped if Rocs should be run in a TemporaryDirectory
    .configure(**options)  # Can be skipped if using default settings
    .run(
        dbase_path=dbase_path,
        query_path=query_path,
        output_path=output_path,
    )

    """

    base_extensions = [".oeb", ".sdf", ".mol", ".mol2", ".pdb", ".ent"]
    ROCS_SUPPORTED_INPUT_SUFFIX = base_extensions + [
        f"{ext}.gz" for ext in base_extensions
    ]

    def __init__(
        self,
        get_confs_from: Literal["omega", "confgen", ""] = "omega",
    ):
        super().__init__()
        set_oe_env()
        self.job_name = "rocs"
        self.get_confs_from = get_confs_from

        # self.logger = get_logger(job_name=self.job_name)
        self.config: RocsConfig = None

    def configure(self, **options):
        self.config = RocsConfig(**options)
        return self

    def set_temp_dir(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir
        return self

    # Will be executed via SimilarityScorerCLI.run
    def run_in_dir(
        self,
        dbase_path: str,
        query_path: str,
        output_path: str,
        temp_dir: Path,
        args: Optional[Dict[str, str]] = None,
        conformer_config: Optional[dict] = None,
    ):
        if self.config is None:
            # self.logger.info("Configuring ROCS with default settings")
            self.configure()

        # * The list of valid suffixes should be an intersection of
        # * supported suffixes from Rocs and get_confs_from
        # check_file_suffix(dbase_path, self.ROCS_SUPPORTED_INPUT_SUFFIX)

        if self.get_confs_from:
            dbase_path = self.get_conformers(dbase_path, temp_dir, conformer_config)

        self.run_command(
            dbase_path,
            query_path,
            output_path,
            self.config,
            cwd=temp_dir,
            args=args,
        )

    @staticmethod
    def run_command(
        dbase_path,
        query_path,
        output_path,
        config: RocsConfig,
        cwd: Union[Path, str] = Path("."),
        args: Optional[Dict[str, str]] = None,
    ):
        command = [
            "rocs",
            "-dbase",
            dbase_path,
            "-query",
            query_path,
            "-outputdir",
            cwd,
            "-maxhits",
            "999999999",
            "-maxconfs",
            str(config.maxconfs),
            "-report",
            "none",
            "-oformat",
            "sdf",
            "-outputQuery",
            "false",
            "-hitsfile",
            output_path,
            "-progress",
            "none",
            "-cutoff",
            str(config.cutoff),
        ]

        if args:
            command.extend([item for pair in args.items() for item in pair])

        try:
            subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"oeomega failed with error: {e.stderr}")
