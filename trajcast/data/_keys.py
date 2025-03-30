"""Keys for dictionaries employed to handle different datasets and dataobjects.
This part of code is heavily inspired by and partially copied from the implementation in NequiP: https://github.com/mir-group/nequip.
"""

from typing import Dict, Final, Set

# Define allowed keys as constants
POSITIONS_KEY: Final[str] = "positions"
CELL_KEY: Final[str] = "cell"
PBC_KEY: Final[str] = "pbc"
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
TOTAL_MASS_KEY: Final[str] = "total_mass"
FORCES_KEY: Final[str] = "forces"
VELOCITIES_KEY: Final[str] = "velocities"
DISPLACEMENTS_KEY: Final[str] = "displacements"
TIMESTEP_KEY: Final[str] = "timestep"
TIME_KEY: Final[str] = "time_run"
FRAME_KEY: Final[str] = "frame"
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
EDGE_LENGTHS_KEY: Final[str] = "edge_lengths"
NODE_FEATURES_KEY: Final[str] = "node_features"
EDGE_FEATURES_KEY: Final[str] = "edge_features"
EDGE_ATTRIBUTES_KEY: Final[str] = "edge_attributes"
NODE_ATTRIBUTES_KEY: Final[str] = "node_attributes"
EDGE_LENGTHS_EMBEDDING_KEY: Final[str] = "edge_lengths_embedding"
ATOM_TYPE_EMBEDDING_KEY: Final[str] = "atom_type_embedding"
ATOM_TYPES_KEY: Final[str] = "atom_types"
ATOMIC_MASSES_KEY: Final[str] = "atomic_masses"
SPHERICAL_HARMONIC_KEY: Final[str] = "sh_embedding"
UPDATE_KEY: Final[str] = "update"
ADDITION_KEY: Final[str] = "add"
UPDATE_VELOCITIES_KEY: Final[str] = f"{UPDATE_KEY}_{VELOCITIES_KEY}"
TEMPERATURE_KEY: Final[str] = "temperature"
MODEL_KEY: Final[str] = "model"
RUN_KEY: Final[str] = "run"
CONFIG_KEY: Final[str] = "configuration"
SCORE_KEY: Final[str] = "score"
UPDATE_SCORE_KEY: Final[str] = f"{UPDATE_KEY}_{SCORE_KEY}"
ARCHITECTURE_KEY: Final[str] = "architecture"
CUTOFF_KEY: Final[str] = "cutoff"
TYPE_MAPPER_KEY: Final[str] = "type_mapper"
TIMESTEP_ENCODING_KEY: Final[str] = "timestep_encoding"
TENSORBOARD_LOG_ROOT_KEY: Final[str] = "log_dir"
TENSORBOARD_LOSS_KEY: Final[str] = "loss"
TENSORBOARD_WEIGHT_STATS_KEY: Final[str] = "weight_stats"
TENSORBOARD_WEIGHTS_KEY: Final[str] = "weights"
TENSORBOARD_GRADIENTS_KEY: Final[str] = "gradients"
TENSORBOARD_VALIDATION_LOSS_KEY: Final[str] = "loss_validation"
TENSORBOARD_LR_KEY: Final[str] = "lr"
WRITE_TRAJECTORY_KEY: Final[str] = "write"
ZERO_MOMENTUM_KEY: Final[str] = "zero_momentum"
SET_MOMENTA_KEY: Final[str] = "set_momenta"
FILENAME_KEY: Final[str] = "filename"
MODEL_TYPE_KEY: Final[str] = "model_type"
CELL_SHIFTS_KEY: Final[str] = "shifts"
THERMOSTAT_KEY: Final[str] = "thermostat"
UNITS_KEY: Final[str] = "units"
EXTRA_DOF_KEY: Final[str] = "extra_dof"


# Define Mapping from ASE to keys used throughout this code
INPUT_KEY_MAPPING: Dict[str, str] = {
    POSITIONS_KEY: ["positions", "coords", "pos"],
    ATOMIC_NUMBERS_KEY: ["numbers", "nuclear_charges"],
    PBC_KEY: ["pbc"],
    CELL_KEY: ["cell", "lattice"],
    FORCES_KEY: ["force", "forces"],
    VELOCITIES_KEY: ["velocities", "vel"],
    TOTAL_ENERGY_KEY: ["energy", "Energy", "energies", "E"],
    DISPLACEMENTS_KEY: ["displacement", "displacements"],
    FRAME_KEY: ["i", "frame"],
    TIME_KEY: ["time", "t"],
}
# ASE FIELDS
ASE_INFO_FIELDS: Final[Set[str]] = {
    TOTAL_ENERGY_KEY,
    TIMESTEP_KEY,
    FRAME_KEY,
    TIME_KEY,
}
ASE_ARRAY_FIELDS: Final[Set[str]] = {
    FORCES_KEY,
    VELOCITIES_KEY,
    DISPLACEMENTS_KEY,
    UPDATE_VELOCITIES_KEY,
}

# define on which dimensions different attributes live
GRAPH_FIELDS: Set[str] = {
    PBC_KEY,
    CELL_KEY,
    TOTAL_ENERGY_KEY,
    TIMESTEP_KEY,
}
NODE_FIELDS: Set[str] = {
    POSITIONS_KEY,
    ATOMIC_NUMBERS_KEY,
    FORCES_KEY,
    VELOCITIES_KEY,
    DISPLACEMENTS_KEY,
    UPDATE_VELOCITIES_KEY,
}
EDGE_FIELDS: Set[str] = {EDGE_VECTORS_KEY, EDGE_LENGTHS_KEY}
