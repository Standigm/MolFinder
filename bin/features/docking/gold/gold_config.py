import os


def write_config(
    ligand_path,
    pdb_path,
    config_path,
    output_path,
    autoscale=1,
    binding_site_size=15,
    binding_site_origin_coordinates="0,0,0",
    n_pose=50,
    gold_temp_dir="./",
    constraint=None,
    constraint_args={},
    relative_ligand_energy=1,
    fitness_function=["plp", "chemscore"],
    run_flag="CONSENSUS",
):
    # origin
    coords = binding_site_origin_coordinates.split(",")
    bs_x = coords[0]
    bs_y = coords[1]
    bs_z = coords[2]
    force_constraints = 0
    # constraint_line
    if constraint == "scaffold":
        scaffold_path = constraint_args["scaffold_path"]
        constraint_weight = constraint_args["constraint_weight"]
        constraint_line = f"constraint scaffold {scaffold_path} {constraint_weight}"
    elif constraint == "similarity":
        reference_mol2 = constraint_args["reference_mol2"]
        constraint_weight = constraint_args["constraint_weight"]
        constraint_line = (
            f"constraint similarity all {reference_mol2} {constraint_weight}"
        )
        force_constraints = 1
    elif constraint is None:
        constraint_line = ""
    # gold_fitfunc_path
    if len(fitness_function) == 1:
        gold_fitfunc_path = fitness_function[0]
        additional_func_setting = ""
    elif len(fitness_function) == 2:
        gold_fitfunc_path = "consensus_score"
        additional_func_setting = f"""docking_fitfunc_path = {fitness_function[0]}
docking_param_file = DEFAULT
rescore_fitfunc_path = {fitness_function[1]}
rescore_param_file = DEFAULT
"""
    # Configs
    configs = f"""  GOLD CONFIGURATION FILE
  AUTOMATIC SETTINGS
autoscale = {autoscale}

  POPULATION
popsiz = auto
select_pressure = auto
n_islands = auto
maxops = auto
niche_siz = auto

  GENETIC OPERATORS
pt_crosswt = auto
allele_mutatewt = auto
migratewt = auto

  FLOOD FILL
radius = {binding_site_size}
origin = {bs_x} {bs_y} {bs_z}
do_cavity = 1
floodfill_atom_no = 0
floodfill_center = point

  DATA FILES
ligand_data_file {ligand_path} {n_pose}
param_file = DEFAULT
set_ligand_atom_types = 1
set_protein_atom_types = 0
directory = {gold_temp_dir}
tordist_file = DEFAULT
make_subdirs = 0
save_lone_pairs = 1
fit_points_file = fit_pts.mol2
read_fitpts = 0

  FLAGS
internal_ligand_h_bonds = 0
flip_free_corners = 1
match_ring_templates = 0
flip_amide_bonds = 0
flip_planar_n = 1 flip_ring_NRR flip_ring_NHR
flip_pyramidal_n = 0
rotate_carboxylic_oh = flip
use_tordist = 1
postprocess_bonds = 1
rotatable_bond_override_file = DEFAULT
diverse_solutions = 1
divsol_rmsd = 1.5
divsol_cluster_size = 3
solvate_all = 1

  TERMINATION
early_termination = 1
n_top_solutions = 5
rms_tolerance = 1.5

  CONSTRAINTS
force_constraints = {force_constraints}
{constraint_line}

  COVALENT BONDING
covalent = 0

  SAVE OPTIONS
save_score_in_file = 1 comments
save_protein_torsions = 1
concatenated_output = {output_path}
output_file_format = MACCS

  FITNESS FUNCTION SETTINGS
initial_virtual_pt_match_max = 3
relative_ligand_energy = {relative_ligand_energy}
gold_fitfunc_path = {gold_fitfunc_path}
score_param_file = DEFAULT
{additional_func_setting}
  
  WRITE OPTIONS
write_options = NO_LOG_FILES NO_LINK_FILES NO_RNK_FILES NO_BESTRANKING_LST_FILE NO_GOLD_LIGAND_MOL2_FILE NO_GOLD_PROTEIN_MOL2_FILE NO_LGFNAME_FILE NO_PID_FILE NO_FIT_PTS_FILES NO_ASP_MOL2_FILES
  
  RUN TYPE
run_flag = {run_flag}

  PROTEIN DATA
protein_datafile = {pdb_path}"""
    if os.path.exists(config_path):
        os.remove(config_path)
        os.sync()
    with open(config_path, "w") as f:
        f.write(configs)
        print(f"Configuration file written: {config_path}")
    os.sync()
    return config_path
