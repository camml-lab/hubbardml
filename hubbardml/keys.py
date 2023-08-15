# Define constants for the various columns of the dataframe
MATERIAL = "material"
FORMULA = "formula"
UV_ITER = "uv_iter"

# Atom elements
ATOM_1_ELEMENT = "atom_1_element"
ATOM_2_ELEMENT = "atom_2_element"
# Atom occupations
ATOM_1_OCCS_1 = "atom_1_occs_1"
ATOM_2_OCCS_1 = "atom_2_occs_1"
ATOM_1_OCCS_2 = "atom_1_occs_2"
ATOM_2_OCCS_2 = "atom_2_occs_2"
# Distances
DIST_IN = "dist_bohr_in"
DIST_OUT = "dist_bohr_out"
# Parameters
PARAM_IN = "param_in"  # The input parameter to hp (i.e. either the value of U or V)
PARAM_TYPE = "param_type"  # The parameter type of that particular row
PARAM_OUT = "param_out"  # The output parameter as calculated by hp
PARAM_DELTA = "param_delta"  # PARAM_OUT - PARAM_IN
PARAM_OUT_FINAL = "param_final"  # The output parameter value of the final self-consistent iteration
# Misc
IS_VDW = "is_vdw"
DIR = "dir"
PERSON = "person"

U_IN = "U_in"
U_OUT = "U_out"
V_IN = "V_in"
V_OUT = "V_out"

PARAM_U = "U"
PARAM_V = "V"

ATOM_1_IN_NAME = "atom_1_in_name"
ATOM_1_OUT_NAME = "atom_1_out_name"
ATOM_2_IN_NAME = "atom_2_in_name"
ATOM_2_OUT_NAME = "atom_2_out_name"

ATOM_1_IDX = "atom_1_idx"
ATOM_1_IDX_UC = "atom_1_idx_uc"
ATOM_2_IDX = "atom_2_idx"
ATOM_2_IDX_UC = "atom_2_idx_uc"

HP_TIME_UNIX = "hp_time_unix"

# Calculated properties
N_ATOM_UC = "n_atoms_uc"

IRREPS_MEAN_1 = "irreps_mean_1"  # The occupation matrix in irrep form for atom 1
IRREPS_MEAN_2 = "irreps_mean_2"  # The occupation matrix in irrep form for atom 2

TRAINING_LABEL = "training_label"
TRAIN = "train"
TEST = "test"
VALIDATE = "validate"
DUPLICATE = "duplicate"
REFERENCE = "reference"

ATOM_SPECIES_1_HOT = "atom_sepcies_1_tensor"
ATOM_SPECIES_2_HOT = "atom_sepcies_2_tensor"

PARAM_OUT_PREDICTED = "param_out_predicted"

COLOUR = "colour"

EPOCH = "epoch"
TRAIN_LOSS = "train_loss"
VALIDATE_LOSS = "validate_loss"
TEST_LOSS = "test_loss"
LABEL = "label"

# Self-consistent paths
SC_PATHS = "sc_paths"

SPECIES = "species"
