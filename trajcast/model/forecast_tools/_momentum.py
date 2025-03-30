from typing import Dict, Union

import torch

from trajcast.data.atomic_graph import AtomicGraph
from trajcast.utils.atomic_computes import (
    compute_angular_momentum_for_individual_state,
    compute_com_velocity_for_individual_state,
    compute_inertia_tensor_for_individual_state,
)


class ZeroMomentum(torch.nn.Module):
    """This class handles the removal of the linear momentum (center of mass motion).
    It takes either a boolen (True) or a dictionary with settings on how often to remove it.
    """

    def __init__(self, settings: Union[bool, Dict]):
        super().__init__()

        if isinstance(settings, bool):
            if not settings:
                raise ValueError(
                    "Momentum is not zeroed by default. Please do not set this argument if you do want to adjust the velocities."
                )
            # if ZeroMomentum is set to True we set the following defaults
            else:
                # adjust_freq is how the frequency on how often we zero the momentum by adjusting the velocities
                # adjust_freq = 1 means in every step.
                self.adjust_freq = 1
                # whether to zero the linear momentum, defaults to True.
                self.zero_linear = True
                # whether to zero the angular momentum, defaults to False.
                self.zero_angular = False

        else:
            self.adjust_freq = settings.get("every", 1)
            self.zero_linear = settings.get("linear", True)
            self.zero_angular = settings.get("angular", False)

        match (self.zero_linear, self.zero_angular):
            case (False, False):
                raise ValueError(
                    "Why adding the keyword momentum if you do not want to change anything? Adjust please!"
                )
            case (True, False):
                self.adjust = self._remove_linear_momentum

            case (False, True):
                self.adjust = self._remove_angular_momentum

            case _:
                self.adjust = self._remove_all_momentum

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        return self.adjust(data)

    def _remove_all_momentum(self, graph: AtomicGraph) -> AtomicGraph:
        """Passed an AtomicGraph this function removes linear and angelur momentum
        returns the updates AtomicGraph.

        Args:
            graph (AtomicGraph): Frame from which we'd like to remove the linear and angular momentum.

        Returns:
            graph:  Cleaned frame.
        """

        graph = self._remove_linear_momentum(graph)
        graph = self._remove_angular_momentum(graph)

        return graph

    def _remove_linear_momentum(self, graph: AtomicGraph) -> AtomicGraph:
        """Passed an AtomicGraph this function removes linear momentum and
        returns the updates AtomicGraph.

        Args:
            graph (AtomicGraph): Frame from which we'd like to remove the linear momentum.

        Returns:
            AtomicGraph:  Cleaned frame.
        """

        com_velocity = compute_com_velocity_for_individual_state(
            velocities=graph.velocities,
            masses=graph.atomic_masses,
            total_mass=graph.total_mass,
        )

        graph.velocities -= com_velocity

        return graph

    def _remove_angular_momentum(self, graph: AtomicGraph) -> AtomicGraph:
        """Passed an AtomicGraph this function removes the angular momentum and
        returns the updates AtomicGraph.
        The angular momentum is defined by L = r x p, where r is the vectors to the center of mass
        and p is the momentum vector. To obtain the angular velocities we need to compute
        omega = I^-1 \times L, where I is the inertia tensor of the system.

        Args:
            ASEAtomsObject (Atoms): Frame from which we'd like to remove the angular momentum.

        Returns:
            Atoms:  Cleaned frame.
        """
        # compute angular momentum
        momenta = graph.atomic_masses * graph.velocities

        angular_momentum, dist_com = compute_angular_momentum_for_individual_state(
            positions=graph.pos,
            momenta=momenta,
            masses=graph.atomic_masses,
            total_mass=graph.total_mass,
        )
        # compute inertia of the system
        inertia_tensor = compute_inertia_tensor_for_individual_state(
            masses=graph.atomic_masses, dist_com=dist_com
        )

        angular_velocity = torch.matmul(torch.inverse(inertia_tensor), angular_momentum)

        graph.velocities -= torch.linalg.cross(angular_velocity.unsqueeze(0), dist_com)

        return graph
