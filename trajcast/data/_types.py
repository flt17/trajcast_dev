import torch
from e3nn.o3 import Irrep
from numpy import float32, float64

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    ATOMIC_NUMBERS_KEY,
    CELL_KEY,
    CELL_SHIFTS_KEY,
    DISPLACEMENTS_KEY,
    EDGE_VECTORS_KEY,
    FORCES_KEY,
    FRAME_KEY,
    PBC_KEY,
    POSITIONS_KEY,
    SCORE_KEY,
    TIME_KEY,
    TIMESTEP_KEY,
    TOTAL_ENERGY_KEY,
    TOTAL_MASS_KEY,
    UPDATE_SCORE_KEY,
    UPDATE_VELOCITIES_KEY,
    VELOCITIES_KEY,
)


class DynamicPrecisionDict(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if value == "DEFAULT_PRECISION":
            return torch.get_default_dtype()
        return value


FIELD_TYPES = DynamicPrecisionDict(
    {
        POSITIONS_KEY: "DEFAULT_PRECISION",
        ATOMIC_NUMBERS_KEY: torch.long,
        PBC_KEY: torch.bool,
        CELL_KEY: "DEFAULT_PRECISION",
        FORCES_KEY: "DEFAULT_PRECISION",
        VELOCITIES_KEY: "DEFAULT_PRECISION",
        TOTAL_ENERGY_KEY: "DEFAULT_PRECISION",
        DISPLACEMENTS_KEY: "DEFAULT_PRECISION",
        TIMESTEP_KEY: "DEFAULT_PRECISION",
        EDGE_VECTORS_KEY: "DEFAULT_PRECISION",
        FRAME_KEY: torch.long,
        CELL_SHIFTS_KEY: torch.long,
        TIME_KEY: "DEFAULT_PRECISION",
        UPDATE_VELOCITIES_KEY: "DEFAULT_PRECISION",
        SCORE_KEY: "DEFAULT_PRECISION",
        UPDATE_SCORE_KEY: "DEFAULT_PRECISION",
        ATOMIC_MASSES_KEY: "DEFAULT_PRECISION",
        TOTAL_MASS_KEY: "DEFAULT_PRECISION",
    }
)

FIELD_IRREPS = {
    DISPLACEMENTS_KEY: Irrep("1o"),
    UPDATE_VELOCITIES_KEY: Irrep("1o"),
}

DTYPE_MAPPING = {torch.float32: float32, torch.float64: float64}
