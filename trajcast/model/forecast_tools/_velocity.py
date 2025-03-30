from torch import Tensor, get_default_dtype, rand, randn

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    TOTAL_MASS_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.forecast_tools import Temperature
from trajcast.utils.atomic_computes import (
    remove_angular_momentum_for_individual_state,
    remove_linear_momentum_for_individual_state,
)
from trajcast.utils.misc import GLOBAL_DEVICE


def init_velocity(
    target_temperature: float,
    graph: AtomicGraph,
    zero_linear: bool,
    zero_angular: bool,
    distribution: str,
    temperature_handler: Temperature,
) -> Tensor:

    if distribution not in ["gaussian", "uniform"]:
        raise KeyError(
            "Velocities can only be drawn from a Gaussian or uniform distribution."
        )

    # get number of atoms
    n_atoms = graph.num_nodes

    # draw initial velocities
    if distribution == "gaussian":
        velocities = randn(
            n_atoms, 3, dtype=get_default_dtype(), device=GLOBAL_DEVICE.device
        )
    elif distribution == "uniform":
        velocities = rand(
            n_atoms, 3, dtype=get_default_dtype(), device=GLOBAL_DEVICE.device
        )

    positions = graph.pos
    masses = graph[ATOMIC_MASSES_KEY]
    total_mass = graph[TOTAL_MASS_KEY]

    # adjust these velocities to account for zero linear and zero angular momentum if required
    if zero_linear:
        velocities = remove_linear_momentum_for_individual_state(
            velocities=velocities, masses=masses, total_mass=total_mass
        )

    if zero_angular:
        velocities = remove_angular_momentum_for_individual_state(
            positions=positions,
            velocities=velocities,
            masses=masses,
            total_mass=total_mass,
        )

    # rescale now to correct temperature
    temp_current = temperature_handler.from_velocities_masses(
        velocities=velocities, masses=masses
    )

    rescale_factor = (target_temperature / temp_current) ** 0.5

    velocities *= rescale_factor

    return velocities
