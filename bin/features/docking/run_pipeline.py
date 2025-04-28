import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import submitit

from ...scoring.components import docking_components
from ..utils.sdf_handler import concat_sdf, filter_sdf, get_best_scores_from_sdf
from .run_lrg import run_gold, run_ligprep, run_rocs


def split_csv_into_shards(
    input_csv: Union[str, Path],
    output_dir: Union[str, Path],
    n_shards: int = 100,
) -> List[Path]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    shard_paths = []

    # Shard by row slicing
    shards = [df.iloc[i::n_shards] for i in range(n_shards)]
    for i, shard_df in enumerate(shards):
        shard_id = f"{i:02d}"  # e.g., 00, 01, ..., 15
        shard_path = output_dir / f"chunk_{shard_id}.csv"
        shard_df.to_csv(shard_path, index=False)
        shard_paths.append(shard_path)

    return shard_paths


def get_shard_id(shard_path: Union[str, Path]) -> str:
    return Path(shard_path).stem.split("_")[-1]


def run_lrg_pipeline(target: str, shard_path, iter_dir: str):
    """
    Run the ligprep-rocs-gold pipeline on a given shard.
    """
    shard_id = get_shard_id(shard_path)

    # Paths
    rocs_output_path = os.path.join(iter_dir, f"rocs_{shard_id}.sdf")
    config_path = os.path.join(shard_path, f"gold_{shard_id}.conf")
    output_path = os.path.join(iter_dir, f"output_{shard_id}.sdf")

    # Run LigPrep
    ligprep_output_path = os.path.join(iter_dir, f"ligprep_{shard_id}.sdf")
    run_ligprep(shard_path, ligprep_output_path, target)

    # Keep top N results
    filtered_ligprep_output_path = filter_sdf(
        ligprep_output_path, rank_by="energy", keep_lowest_values=True, n_per_inchikey=5
    )

    # Run ROCS
    run_rocs(
        input_path=filtered_ligprep_output_path,
        output_path=rocs_output_path,
        target=target,
    )

    # Keep top N results
    filtered_rocs_output_path = filter_sdf(
        rocs_output_path,
        rank_by=docking_components["rocs"]["prop"],
        keep_lowest_values=False,
        n_per_inchikey=1,
    )

    # Run GoldDock
    run_gold(
        input_path=filtered_rocs_output_path,
        output_path=output_path,
        target=target,
        config_path=config_path,
    )

    # Keep top N results
    filtered_gold_output_path = filter_sdf(
        output_path,
        rank_by=docking_components["gold"]["prop"],
        keep_lowest_values=False,
        n_per_inchikey=1,
    )
    return filtered_rocs_output_path, filtered_gold_output_path


def run_iteration(
    target: str,
    iteration: int,
    input_csv: Union[str, Path],
    base_output_dir: Union[str, Path],
):
    shard_paths = split_csv_into_shards(
        input_csv=input_csv,
        output_dir=base_output_dir,
        n_shards=100,
    )
    base_output_dir = Path(base_output_dir)
    iter_dir = Path(base_output_dir) / f"iter{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base_output_dir / f"logs_iter{iteration}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rocs_final_path = iter_dir / f"_rocs_final_iter{iteration}.sdf"
    gold_final_path = iter_dir / f"_gold_final_iter{iteration}.sdf"
    executor = submitit.AutoExecutor(folder=str(log_dir))

    executor.update_parameters(
        timeout_min=720,
        cpus_per_task=4,
        slurm_partition="cpu256,cpu96",
        name=f"molf_iter{iteration}",
    )

    jobs = []
    with executor.batch():
        for shard in shard_paths:
            jobs.append(executor.submit(run_lrg_pipeline, target, shard, str(iter_dir)))

    output_paths = [job.result() for job in jobs]
    rocs_paths = [Path(rpath[0]).resolve().as_posix() for rpath in output_paths]
    gold_paths = [Path(gpath[1]).resolve().as_posix() for gpath in output_paths]

    concat_sdf(rocs_paths, rocs_final_path)
    concat_sdf(gold_paths, gold_final_path)

    # Just get the inchikeys and the best scores
    best_rocs = get_best_scores_from_sdf(
        rocs_final_path,
        select_by=docking_components["rocs"]["prop"],
        keep_lowest_scores=False,
    )
    best_gold = get_best_scores_from_sdf(
        gold_final_path,
        select_by=docking_components["gold"]["prop"],
        keep_lowest_scores=False,
    )

    return best_rocs, best_gold
