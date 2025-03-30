"""
This is a collection of tests validating the structures and trajectories predicted
by the model obey the laws of physics. Generally, these functions do not need a reference
trajectory to compare to.

Author: Fabian Thiemann [add name when contributing]
"""

from typing import Optional, Tuple

import numpy as np
from ase.units import J, kB, kg

from ..utils.atomic_computes import (
    align_vectors_with_periodicity,
    cell_parameters_to_lattice_vectors,
)
from ..utils.misc import convert_units

try:
    from MDAnalysis import Universe

    def track_min_max_distances_in_molecule(
        trajectory: Universe,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the time evolution of the minimum and maximum distance in a molecule.

        Args:
            trajectory (Universe): Trajectory given as MDAnalysis.Universe.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of min and max distances.
        """
        # based on the first frame we need identify the two atom pairs
        # one for the min distance and one for the max distance
        # start by computing vectors between all atoms
        vectors = trajectory.atoms.positions[:, np.newaxis] - trajectory.atoms.positions

        # make sure everything is in line with nearest image convention and pbc
        vectors = align_vectors_with_periodicity(
            vectors=vectors,
            lattice_vectors=cell_parameters_to_lattice_vectors(
                params=trajectory.dimensions
            ),
        )

        # now compute distances
        distances = np.linalg.norm(vectors, axis=2)

        # now find min and max indices
        min_indices = np.argwhere(distances == np.min(distances[distances > 0]))[0]
        max_indices = np.argwhere(distances == np.max(distances))[0]

        # instantiate arrays
        min_distances = []
        max_distances = []

        # now let's loop over the trajectory
        for frame in trajectory.trajectory:
            # save minimum distance
            min_distances.append(
                np.linalg.norm(
                    align_vectors_with_periodicity(
                        np.diff(frame.positions[min_indices], axis=0),
                        lattice_vectors=cell_parameters_to_lattice_vectors(
                            params=trajectory.dimensions
                        ),
                    )
                )
            )
            # save maximum distance
            max_distances.append(
                np.linalg.norm(
                    align_vectors_with_periodicity(
                        np.diff(frame.positions[max_indices], axis=0),
                        lattice_vectors=cell_parameters_to_lattice_vectors(
                            params=trajectory.dimensions
                        ),
                    )
                )
            )

        return np.asarray(min_distances), np.asarray(max_distances)

    def track_average_distance_from_molecule_center_of_mass(
        trajectory: Universe,
    ) -> np.ndarray:
        """Computes the time evolution of average distance over all atoms in a molecule from its center of mass.
            Note: We assume an unwrapped trajectory here for now.
        Args:
            trajectory (Universe): Trajectory given as MDAnalysis.Universe.

        Returns:
            (np.ndarray): Array of distances from center of mass.
        """

        # instantiate arrays
        mean_distances = []

        # now let's loop over the trajectory
        for frame in trajectory.trajectory:
            # get center of mass
            COM = trajectory.atoms.center_of_mass()

            # next compute distances
            vectors = frame.positions - COM

            # get distances in line with pbcs
            distances = np.linalg.norm(
                align_vectors_with_periodicity(
                    vectors,
                    lattice_vectors=cell_parameters_to_lattice_vectors(
                        params=trajectory.dimensions
                    ),
                ),
                axis=1,
            )

            # take average and save
            mean_distances.append(np.mean(distances))

        return np.asarray(mean_distances)

    def track_instantaneous_temperature(
        trajectory: Universe,
        linear_momentum_fixed: Optional[bool] = False,
        angular_momentum_fixed: Optional[bool] = False,
        unit_velocity: Optional[str] = "angstroms/femtoseconds",
    ) -> np.ndarray:
        """Computes the time evolution of the instantanteous temperature based on the atomic velocities.

        Args:
            trajectory (Universe): Trajectory given as MDAnalysis.Universe.
            linear_momentum_fixed (Optional[bool], optional): True if the center of mass motion has been removed from the trajectory.
                Defaults to False to not remove the translational degrees of freedom of a molecule.
            angular_momentum_fixed (Optional[bool], optional): True if angular momentum is removed in trajectory generation.
                Defaults to False.
            unit_velocity (Optional[str], optional): Unit of the velocities in the Universe. Defaults to "angstroms/femtoseconds".

        Raises:
            AttributeError: If a Universe is given without velocities.

        Returns:
            np.ndarray: Array of distances from center of mass.
        """

        # instantiate array
        temperatures = []

        # get masses for each atom in kg per particle, we assume (g/mol in universe)
        masses = trajectory.atoms.masses / kg

        # get the conversion factor for the velocities
        # we want the velocities to be in m/s after all
        # we convert distance and time units seperately
        # to do this, let's split at the slash after getting rid of all spaces
        units_distance_time = unit_velocity.replace(" ", "").split("/")

        # now we can get the conversion factors for both
        conv_factor_velocity = convert_units(
            origin=units_distance_time[0], target="meter"
        ) / convert_units(origin=units_distance_time[1], target="seconds")

        # check the universe has velocities
        # doing it this way allows to also work with slices
        if not hasattr(trajectory.trajectory[0], "velocities"):
            raise AttributeError(
                "The trajectory does not have velocities, therefore we cannot compute the temperature."
            )

        # compute the degrees of freedom
        degrees_of_freedom = 3 * len(masses)

        if linear_momentum_fixed:
            degrees_of_freedom -= 3
        if angular_momentum_fixed:
            degrees_of_freedom -= 3

        # now let's loop over the trajectory
        for frame in trajectory.trajectory:
            # compute momenta in kg*m/s
            momenta = (
                masses * np.linalg.norm(frame.velocities, axis=1) * conv_factor_velocity
            )

            # now we can easily compute the instantaneous temperature
            # details on how the temperature is computed can be found in the following reference
            # D. Frenkel and B Smit; "Understanding Molecular Simulation: Algorithms to Applications", Academic Press, 2nd Edition, 2002, page 64, equation 4.1.2
            temp_inst = (
                np.sum(np.square(momenta) / masses) / (kB / J) / degrees_of_freedom
            )

            # append to list
            temperatures.append(temp_inst)

        return np.asarray(temperatures)

except ImportError:
    print("All validation function built on MDAnalysis, please install MDAnalysis.")
