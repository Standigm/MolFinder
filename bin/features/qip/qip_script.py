import os

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from qip.datamodules.collaters import DefaultCollater
from qip.datamodules.featurizers import QIPFeaturizer
from qip.datamodules.transforms import RandomWalkGenerator
from rdkit import Chem
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    print("CUDA is available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# inference
class InfernceModule:
    def __init__(self, encoder, task_heads, task_head_post_processes):
        self.encoder = encoder
        self.task_heads = task_heads
        self.task_head_post_processes = task_head_post_processes

    def __call__(self, batch):
        with torch.no_grad():
            batch = batch.to(device)
            encoder_out = self.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rwse
            )
            task_outputs = {}
            for task_name, task_head in self.task_heads.items():
                task_output = task_head(encoder_out[0], encoder_out[1])
                task_output_post_process = (
                    self.task_head_post_processes[task_name](task_output).cpu().numpy()
                )
                task_outputs.update({task_name: task_output_post_process})
        return task_outputs


def get_data(smiles):
    featurizer = QIPFeaturizer()
    rw = RandomWalkGenerator(ksteps=[1, 17])
    data = featurizer(smiles)
    data = rw(data)
    return data


def main(smiles_list):
    data_list = [get_data(smiles) for smiles in smiles_list]
    collate_fn = DefaultCollater(follow_batch=None, exclude_keys=None)
    batch = collate_fn(data_list)

    # Load model
    ckpt = torch.load("/db2/users/hokyun/multitask_weight_HAD.ckpt")
    encoder_config = OmegaConf.load(
        "/db2/users/hokyun/git/QIP/configs/system/encoder_config/gps/medium.yaml"
    )
    encoder = hydra.utils.instantiate(encoder_config)["module"]
    encoder_dict = {}
    for key in ckpt["state_dict"].keys():
        if "encoder" in key:
            encoder_dict[".".join(key.split(".")[1:])] = ckpt["state_dict"][key]

    encoder.load_state_dict(encoder_dict)
    encoder.eval()
    encoder.to(device)

    multitask_config_path = (
        "/db2/users/hokyun/git/QIP/configs/system/task_head_configs/gps/MT0.yaml"
    )
    task_head_path = "/".join(multitask_config_path.split("/")[:-2])
    task_head_configs = OmegaConf.load(
        "/db2/users/hokyun/git/QIP/configs/system/task_head_configs/gps/MT0.yaml"
    )

    task_heads = {}
    task_head_post_processes = {}
    for task_head_config in task_head_configs["defaults"]:
        task_head_config = OmegaConf.load(
            os.path.join(task_head_path, task_head_config)
        )
        task_name = list(task_head_config.keys())[0]
        task_head_config[task_name]["module"]["in_features"] = encoder_config["module"][
            "d_model"
        ]

        task_head_instance = hydra.utils.instantiate(task_head_config)[task_name]

        task_head = task_head_instance["module"]
        task_head_dict = {}
        for key in ckpt["state_dict"].keys():
            if task_name in key:
                task_head_dict[".".join(key.split(".")[2:])] = ckpt["state_dict"][key]
        task_head.load_state_dict(task_head_dict)
        task_head.eval()
        task_head.to(device)
        task_heads.update({task_name: task_head})

        task_head_post_process = task_head_instance["post_process"]
        task_head_post_processes.update({task_name: task_head_post_process})

    data_list = [get_data(smiles) for smiles in smiles_list]
    inference = InfernceModule(encoder, task_heads, task_head_post_processes)
    dataloader = DataLoader(data_list, batch_size=2, collate_fn=collate_fn)
    df_list = []
    for batch in dataloader:
        res = inference(batch)
        df = pd.DataFrame({key: value.flatten() for key, value in res.items()})
        df_list.append(df)
    df_qip = pd.concat(df_list)
    df_qip.insert(0, "smiles", smiles_list)
    df_qip.insert(
        1,
        "inchikey",
        [Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) for smiles in smiles_list],
    )
    return df_qip


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to csv file")
    parser.add_argument(
        "--smiles_column", type=str, help="Column name of smiles", default="smiles"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save result", default="qip_result.csv"
    )
    args = parser.parse_args()

    df_qip = main(pd.read_csv(args.csv_path)[args.smiles_column].tolist())
    df_qip.to_csv(args.output_path, index=False)
