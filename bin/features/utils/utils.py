import csv
import os
from pathlib import Path


def set_oe_env():
    """get openeye binary path"""
    with open("/etc/os-release", "r", encoding="utf-8") as f:
        linux_info = dict(csv.reader(f, delimiter="="))
    linux_name = linux_info.get("NAME", "ubuntu").lower()
    if linux_name.startswith("oracle"):
        linux_name = "redhat"
    linux_ver = linux_info.get("VERSION_ID", "22.04")
    oe_path = f"/db2/OpenEye/{linux_name}-{linux_ver}/bin"
    if oe_path not in os.environ["PATH"]:
        os.environ["PATH"] = f"{oe_path}:{os.environ['PATH']}"

    os.environ["OE_LICENSE"] = "/db2/OpenEye/oe_license.txt"


def get_default_output_path(input_path: str, job_name: str) -> str:
    in_path = Path(input_path).resolve()
    return Path(in_path.parent / f"{job_name}_output-{in_path.stem}.sdf").as_posix()


def flatten(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def add_prefix_to_filename(filepath, prefix):
    # Split the filepath into the directory and the filename
    directory, filename = os.path.split(filepath)

    # Add the prefix to the filename
    new_filename = f"{prefix}_{filename}"

    # Join the directory and the new filename to create the new filepath
    new_filepath = os.path.join(directory, new_filename)

    return new_filepath
