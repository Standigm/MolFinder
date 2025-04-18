import gzip
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from openeye import oechem
from rdkit import Chem

from .utils import add_prefix_to_filename

# from stella_tasks.openeye_script.sdf2csv import SDF2CSV


# def sdf_to_csv(sdf_path: Path | str, csv_path: Path | str):
#     csv = oechem.oeofstream(str(csv_path))
#     ifs = oechem.oemolistream(str(sdf_path))
#     SDF2CSV(ifs, csv)
#     return csv_path


def sdf_to_df(sdf_path: Union[Path, str]):
    taglist = []

    ifs = oechem.oemolistream(str(sdf_path))

    # Read through once to find all unique tags and handle duplicates
    for mol in ifs.GetOEGraphMols():
        tag_tracker = defaultdict(int)
        for dp in oechem.OEGetSDDataPairs(mol):
            tag = dp.GetTag()
            tag_tracker[tag] += 1
            if tag_tracker[tag] > 1:
                tag = f"{tag}_{tag_tracker[tag]}"
            if tag not in taglist:
                taglist.append(tag)

    ifs.rewind()

    # Build DataFrame
    data = []
    for mol in ifs.GetOEGraphMols():
        row = {"Title": mol.GetTitle()}
        tag_tracker = defaultdict(int)
        for tag in taglist:
            base_tag = (
                tag.rsplit("_", 1)[0]
                if "_" in tag and tag.rsplit("_", 1)[1].isdigit()
                else tag
            )
            tag_tracker[base_tag] += 1

            actual_tag = (
                f"{base_tag}_{tag_tracker[base_tag]}"
                if tag_tracker[base_tag] > 1
                else base_tag
            )

            if oechem.OEHasSDData(mol, base_tag):
                value = oechem.OEGetSDData(mol, base_tag)
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep value as string if it can't be converted to float
            else:
                value = ""
            row[actual_tag] = value
        data.append(row)

    df = pd.DataFrame(data)
    return df


def subset_sdf(
    input_sdf: Union[Path, str],
    output_sdf: Union[Path, str],
    indices: Union[List[int], pd.Index],
    remove_origin: bool = False,
    verbose: bool = True,
):
    # Helper function for printing logs only if verbose is set to True
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Create a molecule supplier for reading the input SDF
    ifs = oechem.oemolistream()
    if ifs.open(str(input_sdf)):
        vprint(f"Reading molecules from {input_sdf}")

        # Create a dictionary to store molecules by their index
        mol_dict = {}
        for i, mol in enumerate(ifs.GetOEMols()):
            mol_dict[i] = oechem.OEMol(mol)  # Copy the molecule

        # Create a molecule stream for writing the output SDF
        ofs = oechem.oemolostream()
        if ofs.open(str(output_sdf)):
            vprint(f"Writing subset to {output_sdf}")

            # Write the selected molecules to the output SDF in the order of indices
            for i in indices:
                if i in mol_dict:
                    oechem.OEWriteMolecule(ofs, mol_dict[i])

            ofs.close()
        else:
            oechem.OEThrow.Fatal("Unable to open output SDF file")

        ifs.close()
    else:
        oechem.OEThrow.Fatal("Unable to open input SDF file")

    # Check if the origin file path is provided and remove the file
    if remove_origin:
        try:
            os.remove(input_sdf)
            vprint(f"Origin file {input_sdf} removed successfully.")
        except OSError as e:
            vprint(f"Error removing origin file {input_sdf}: {e}")

    return output_sdf


def subset_mol(
    input_sdf: Union[Path, str], indices: Union[List[int], pd.Index]
) -> List[oechem.OEMol]:
    # Create a molecule supplier for reading the input SDF
    ifs = oechem.oemolistream()
    if ifs.open(str(input_sdf)):
        # Create a dictionary to store molecules by their index
        mol_dict = {}

        # Iterate through molecules and store them in the dictionary with their index
        for i, mol in enumerate(ifs.GetOEMols()):
            mol_dict[i] = oechem.OEMol(mol)  # Copy the molecule

        # Create a list to store selected molecules in the order of indices
        selected_mols = [mol_dict[i] for i in indices if i in mol_dict]

        ifs.close()

        return selected_mols
    else:
        oechem.OEThrow.Fatal("Unable to open input SDF file")


def write_mols_to_sdf(output_sdf: Union[Path, str], mol_list: List[oechem.OEMol]):
    if os.path.exists(output_sdf):
        os.remove(output_sdf)
    # Create a molecule stream for writing the output SDF
    ofs = oechem.oemolostream()
    if ofs.open(str(output_sdf)):
        print(f"Writing molecules to {output_sdf}")

        # Iterate through molecules and write them to the output SDF
        for mol in mol_list:
            oechem.OEWriteMolecule(ofs, mol)

        ofs.close()
    else:
        oechem.OEThrow.Fatal("Unable to open output SDF file")
    os.sync()


def read_sdf_rdkit(sdf_path: Union[Path, str]):
    mol_list = []
    with Chem.SDMolSupplier(str(sdf_path)) as suppl:
        for mol in suppl:
            if mol is None:
                continue
            else:
                mol_list.append(mol)
    return mol_list


def write_mols_to_sdf_rdkit(mols: List[Chem.Mol], output_path: Union[Path, str]):
    output_path = Path(str(output_path)).resolve()
    # Write the molecules to the SDF file
    with Chem.SDWriter(str(output_path)) as writer:
        for mol in mols:
            if mol is not None:  # Check if the molecule is valid
                writer.write(mol)
    print(f"{output_path} was written.")


def sdf_gz_to_mol(sdf_path: Union[Path, str]):
    with gzip.open(sdf_path, "rt") as sdf_file:
        # Read the contents of the Structure Data File
        sdf_contents = sdf_file.read()
    # Use RDKit to work with the contents of the Structure Data File
    # For example, you can loop through molecules and perform operations
    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdf_contents, sanitize=False, removeHs=False)
    mols = [mol for mol in suppl]
    return mols


def get_mols_from_output_paths(df, path_col, key="inchikey"):
    """
    Input:
        pd.DataFrame:
            columns: inchikey | sdf_path
    """
    total_mols = []
    paths = set(df[path_col].to_list())
    for p in paths:
        df_sdf = sdf_to_df(p)
        indices = df_sdf[df_sdf[key].isin(df[key].to_list())].index
        mols = subset_mol(p, indices)
        total_mols.extend(mols)
    return total_mols


def subset_sdf_top(
    sdf_path: Union[Path, str],
    output_sdf: Union[Path, str],
    key="inchikey",
    sort_by=None,
    ascending=False,
    n_top=1,
):
    """
    Given an input SDF file, this function extracts the top N entries for each group defined by 'key',
    sorted by the 'sort_by' column, and writes them to an output SDF file.

    Parameters:
    sdf_path (str): Path to the input SDF file.
    output_sdf (str): Path to the output SDF file where the subset will be saved.
    key (str): Column name to group by (default is 'inchikey').
    sort_by (str): Column name to sort by (default is None).
    ascending (bool): Sort order (default is True for ascending order).
    n_top (int): Number of top entries to extract per group (default is 1).
    """

    # Convert SDF to DataFrame
    df = sdf_to_df(sdf_path)

    # Sort DataFrame based on 'sort_by' column
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Get the indices of the top N entries per group defined by 'key'
    top_indices = df.groupby(key).head(n_top).index

    # Create the subset SDF file based on the top indices
    subset_sdf(sdf_path, output_sdf, top_indices)


def random_sample_sdf(
    input_sdf: Union[Path, str],
    output_sdf: Union[Path, str],
    sample_size: int = 3,
    group_col: str = "inchikey",
):
    sample_indices = (
        sdf_to_df(input_sdf)
        .groupby(group_col)
        .apply(lambda x: x.sample(sample_size) if len(x) >= sample_size else x)
        .index
    )
    if sample_indices.nlevels == 1:
        sample_indices = sample_indices.get_level_values(0)
    elif sample_indices.nlevels == 2:
        sample_indices = sample_indices.get_level_values(1)
    subset_sdf(input_sdf, output_sdf, sample_indices, False)


def reorder_sdf(
    sdf_path: Union[Path, str],
    ordered_inchikeys: List[str],
    output_sdf_path: Union[Path, str],
):
    # Convert the SDF file to a DataFrame
    df_sdf = sdf_to_df(sdf_path)

    # Set the 'inchikey' column as a categorical type with the specified order
    df_sdf["inchikey"] = pd.Categorical(
        df_sdf["inchikey"], categories=ordered_inchikeys, ordered=True
    )

    # Sort the DataFrame by the 'inchikey' column
    df_sdf = df_sdf.sort_values("inchikey")

    # Get the indices of the sorted DataFrame
    sorted_indices = list(df_sdf.index)

    # Create the subset SDF file with the sorted indices
    subset_sdf(sdf_path, output_sdf_path, sorted_indices)

    # Return the path to the output SDF file
    return output_sdf_path


def add_title_to_tag_molecules(input_sdf, output_sdf, tag_name="inchikey"):
    try:
        ifs = oechem.oemolistream()
        if not ifs.open(input_sdf):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % input_sdf)

        ofs = oechem.oemolostream()
        if not ofs.open(output_sdf):
            oechem.OEThrow.Fatal("Unable to open %s for writing" % output_sdf)

        for mol in ifs.GetOEGraphMols():
            title = mol.GetTitle()
            oechem.OEAddSDData(mol, tag_name, title)
            oechem.OEWriteMolecule(ofs, mol)
    finally:
        ifs.close()
        ofs.close()


def filter_sdf(
    sdf_path: Union[Path, str],
    rank_by: str,
    keep_lowest_values: bool = False,
    n_per_inchikey: int = 1,
    prefix: Union[str, None] = "filtered",
) -> str:
    sdf_path = str(sdf_path)
    df = sdf_to_df(sdf_path)
    if rank_by == "random":
        indices = df.groupby("inchikey").sample(n=n_per_inchikey).index.to_list()
    else:
        indices = (
            df.sort_values(rank_by, ascending=keep_lowest_values)
            .groupby("inchikey")
            .head(n_per_inchikey)
            .index.to_list()
        )
    if prefix:
        subset_sdf_path = add_prefix_to_filename(sdf_path, prefix)
    else:
        # If not prefix, overwrite the input sdf
        subset_sdf_path = sdf_path
    subset_sdf(sdf_path, subset_sdf_path, indices, verbose=False)
    return subset_sdf_path


def get_best_scores_from_sdf(
    input_sdf_path: Union[Path, str],
    select_by: str,
    keep_lowest_scores=False,
    output_sdf_path: Optional[Union[Path, str]] = None,
    smiles_inchi_df: Optional[pd.DataFrame] = None,
    backfill_value: float = 0,
) -> List[float]:
    """_summary_

    Args:
        input_sdf_path (str): _description_
        output_sdf_path (str): _description_
        select_by (str): _description_
        smiles_inchi_df (pd.DataFrame | None, optional): _description_. Defaults to None.
            If not None, assign to missing inchikeys a score of np.nan.
            If None, return existing scores without imputation.
        backfill_value: _description_ Should probably be np.nan or 0.

    Returns:
        list[float]: _description_
    """
    input_sdf_path = str(input_sdf_path)
    best_rows = (
        sdf_to_df(input_sdf_path)[["inchikey", select_by]]
        .sort_values(select_by, ascending=keep_lowest_scores)
        .groupby("inchikey")
        .head(1)
        .sort_index()
    )
    best_indices = best_rows.index.to_list()
    if output_sdf_path:
        subset_sdf(input_sdf_path, str(output_sdf_path), best_indices, verbose=False)

    if not isinstance(smiles_inchi_df, pd.DataFrame):
        return best_rows[select_by].values.tolist()

    scores = []
    missing = []
    inchikeys = smiles_inchi_df["inchikey"].values
    for inchi in inchikeys:
        if inchi in best_rows["inchikey"].values:
            score = best_rows.loc[best_rows["inchikey"] == inchi, select_by].values[0]  # type: ignore
        else:
            missing.append(inchi)
            score = backfill_value
        scores.append(score)

    if missing:
        print(
            "Inchikeys missing from %s were assigned scores of %s",
            Path(input_sdf_path).name,
            backfill_value,
        )

        # If output_sdf_path is provided, write missing inchikeys to a file in the output dir
        if output_sdf_path:
            output_sdf_path = Path(output_sdf_path)
            missing_inchikeys_file = (
                output_sdf_path.parent / f"{output_sdf_path.stem}_missing.csv"
            )
            smiles_inchi_df[smiles_inchi_df["inchikey"].isin(missing)].to_csv(
                missing_inchikeys_file, index=False
            )
            print("Missing inchikeys written to %s", missing_inchikeys_file)
        else:
            print("Missing inchikeys: %s", missing)

    return scores
