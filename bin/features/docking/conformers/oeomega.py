import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Literal, Optional, Union

import pandas as pd
from openeye import oechem

from ...utils.utils import set_oe_env
from .conformer_generator import ConformerGeneratorCLI

# from stella_tasks.utils.utils import check_file_suffix, get_logger

OEOMEGA_MODES = ["classic", "macrocycle", "rocs", "pose", "dense", "fastrocs"]


@dataclass
class OEOmegaConfig:
    mode: str = "rocs"
    maxconfs: int = 200
    rms: float = 0.5
    use_gpu: bool = False
    flipper: bool = False
    strictstereo: bool = False
    strictatomtyping: bool = False

    # Write strain energy to SD data, and use that instead of mol.GetEnergy()
    # Required if energy should be written to an output file whose extension is not .oeb / .oeb.gz
    sd_energy: bool = False

    # SD data tag for energy if sd_energy is True; should be the value of the searchff option
    search_ff: str = "mmff94smod_NoEstat"

    sd_data: Literal["keep", "fill", "overwrite"] = "keep"

    def __post_init__(self):
        if self.mode not in OEOMEGA_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {OEOMEGA_MODES}"
            )

        if self.sd_data == "keep":
            self.sd_energy = False


class OEOmegaCLI(ConformerGeneratorCLI):
    """
    Example usage:

    OEOmegaCLI()
    .set_temp_dir(temp_dir=temp_dir)  # Can be skipped if Omega should be run in a TemporaryDirectory
    .configure(**options)  # Can be skipped if using default settings
    .run(
        input_path=input_path,
        output_path=output_path,
    )

    """

    OMEGA_SUPPORTED_INPUT_SUFFIX = [
        "can",
        "cdx",
        "cif",
        "csv",
        "cxsmiles",
        "dat",
        "ent",
        "fasta",
        "inchi",
        "ism",
        "isosmi",
        "json",
        "mdl",
        "mmd",
        "mmod",
        "mol",
        "mol2",
        "mol2h",
        "oeb",
        "oez",
        "pdb",
        "sd",
        "sdf",
        "seq",
        "skc",
        "smi",
        "syb",
        "usm",
        "xyz",
    ]
    OMEGA_SUPPORTED_INPUT_SUFFIX += [f"{i}.gz" for i in OMEGA_SUPPORTED_INPUT_SUFFIX]
    OMEGA_SUPPORTED_OUTPUT_SUFFIX = [
        "cdx",
        "cif",
        "dat",
        "ent",
        "json",
        "mdl",
        "mmd",
        "mmod",
        "mol",
        "mol2",
        "mol2h",
        "mopac",
        "oeb",
        "oez",
        "pac",
        "pdb",
        "sd",
        "sdf",
        "syb",
        "xyz",
    ]
    OMEGA_SUPPORTED_OUTPUT_SUFFIX += [f"{i}.gz" for i in OMEGA_SUPPORTED_OUTPUT_SUFFIX]

    def __init__(self):
        super().__init__()
        set_oe_env()
        self.job_name = "oeomega"

        # self.logger = get_logger(job_name=self.job_name)
        self.config: OEOmegaConfig = None

    def configure(self, **options):
        self.config = OEOmegaConfig(**options)
        return self

    def set_temp_dir(self, temp_dir: str):
        self.temp_dir = temp_dir
        return self

    # Will be executed via ConformerGeneratorCLI.run
    def run_in_dir(
        self,
        input_path: str,
        output_path: str,
        temp_dir: Path,
        args: Optional[Dict[str, str]] = None,
    ):
        if self.config is None:
            # self.logger.info("Configuring conformer generator with default settings")
            self.configure()

        # check_file_suffix(input_path, self.OMEGA_SUPPORTED_INPUT_SUFFIX)
        # check_file_suffix(output_path, self.OMEGA_SUPPORTED_OUTPUT_SUFFIX)

        if "".join(Path(input_path).suffixes) in [".csv", ".csv.gz"]:
            input_path = self.prep_input_csv(input_path, temp_dir)

        if self.config.sd_data == "keep":
            self.run_command(
                input_path, output_path, self.config, cwd=temp_dir, args=args
            )
        else:
            output_suffix = "".join(Path(output_path).suffixes)
            if output_suffix not in [".oeb", ".oeb.gz"]:
                self.config.sd_energy = True

            with NamedTemporaryFile(
                "w", suffix="".join(Path(output_path).suffixes)
            ) as temp_sdf:
                self.run_command(
                    input_path, temp_sdf.name, self.config, cwd=temp_dir, args=args
                )

                # Add sd data if self.config_sd_data == "fill" or "overwrite"
                variant_counts = defaultdict(int)
                overwrite = True if self.config.sd_data == "overwrite" else False
                ifs = oechem.oemolistream(temp_sdf.name)
                ofs = oechem.oemolostream(output_path)
                for mol in ifs.GetOEGraphMols():
                    title = mol.GetTitle() or oechem.OEMolToSTDInChIKey(mol)
                    variant_counts[title] += 1

                    if self.config.sd_energy:
                        energy = oechem.OEGetSDData(mol, self.config.search_ff)
                        oechem.OEDeleteSDData(mol, self.config.search_ff)
                    else:
                        energy = mol.GetEnergy()

                    oechem.OEWriteMolecule(
                        ofs,
                        self.add_sddata_to_oemol(
                            mol,
                            energy,
                            variant_count=variant_counts[title],
                            overwrite=overwrite,
                        ),
                    )
                ifs.close()
                ofs.close()

    def run_command(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        config: OEOmegaConfig,
        cwd: Union[Path, str] = Path("."),
        args: Optional["dict[str, str]"] = None,
    ):
        # flipper has "false", "true", and "force" options
        # "true" is not currently supported
        flipper = "force" if config.flipper else "false"

        command = [
            "oeomega",
            config.mode,
            "-in",
            input_path,
            "-out",
            output_path,
            "-maxconfs",
            str(config.maxconfs),
            "-rms",
            str(config.rms),
            "-useGPU",
            str(config.use_gpu).lower(),
            "-verbose",
            "false",
            "-flipper",
            flipper,
            "-strictstereo",
            str(config.strictstereo).lower(),
            "-strictatomtyping",
            str(config.strictatomtyping).lower(),
            "-sdEnergy",
            str(config.sd_energy).lower(),
            "-searchff",
            config.search_ff,
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

    @staticmethod
    def prep_input_csv(input_path: str, temp_dir: Path) -> str:
        input_df = pd.read_csv(input_path)
        if "smiles" not in input_df.columns:
            raise ValueError("smiles column not found in input csv")
        # Reorder columns so that smiles comes first
        input_df = input_df[
            ["smiles"] + [col for col in input_df.columns if col != "smiles"]
        ]

        input_path = (temp_dir / Path(input_path).name).as_posix()
        input_df.to_csv(input_path, index=False)

        return input_path

    @staticmethod
    def add_sddata_to_oemol(
        mol, energy: float, variant_count: int, overwrite: bool = False
    ):
        properties = {
            "energy": str(energy),
            "smiles": oechem.OEMolToSmiles(mol),
            "variant": str(variant_count),
            "inchikey": oechem.OEMolToSTDInChIKey(mol),
        }

        for prop, value in properties.items():
            if overwrite or not oechem.OEHasSDData(mol, prop):
                oechem.OESetSDData(mol, prop, value)

        return mol
        return mol
