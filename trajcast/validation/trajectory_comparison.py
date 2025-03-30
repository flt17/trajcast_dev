import copy
from typing import Optional, Tuple

import numpy as np

from ..utils.misc import get_least_common_multiple
from .physical_behaviour import track_instantaneous_temperature

try:
    from MDAnalysis import Universe

    def compute_particle_position_error(
        predicted_trajectory: Universe,
        reference_trajectory: Universe,
        mode: Optional[str] = "mae",
        tolerance: Optional[float] = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        # do a copy of the the universes
        pred_traj_modify = copy.copy(predicted_trajectory)
        ref_traj_modify = copy.copy(reference_trajectory)

        # First we check whether both trajectories have the same starting point
        if not np.allclose(
            pred_traj_modify.trajectory[0].positions,
            ref_traj_modify.trajectory[0].positions,
            tolerance,
        ):
            raise ValueError(
                "For this test, trajectories need to start from same frame!"
            )

        # we also need to be sure the velocities are identical, if given
        if pred_traj_modify.trajectory.ts.has_velocities and not np.allclose(
            pred_traj_modify.trajectory[0].velocities,
            ref_traj_modify.trajectory[0].velocities,
            tolerance,
        ):
            raise ValueError(
                "Check that velocities are also identical of initial frame!"
            )

        # next we need to make sure the compare the correct timesteps with each other
        # for this, we compute frame frequency (fq) and common length
        pred_fq, ref_fq, length, time_between_frames = _align_trajectories(
            pred_traj_modify, ref_traj_modify
        )
        # slice trajectories accordingly
        predicted_traj_truncated = pred_traj_modify.trajectory[::pred_fq][:length]
        reference_traj_truncated = ref_traj_modify.trajectory[::ref_fq][:length]

        # based on input, compute the required statistics
        error_function = {
            "mae": np.mean,
            "mse": lambda x: np.mean(np.square(x)),
            "rmse": lambda x: np.sqrt(np.mean(np.square(x))),
        }[mode]
        # now we loop over them
        errors = []
        for frame_pred, frame_ref in zip(
            predicted_traj_truncated, reference_traj_truncated
        ):
            errors.append(
                error_function(
                    np.linalg.norm(frame_pred.positions - frame_ref.positions, axis=1)
                )
            )

        # we also return the time between frames (in fs instead of ps)
        return (np.asarray(errors), time_between_frames * 1000)

    def compare_instantaneous_temperature(
        predicted_trajectory: Universe,
        reference_trajectory: Universe,
        linear_momentum_fixed: Optional[bool] = False,
        unit_velocity_predicted: Optional[str] = "angstroms/femtoseconds",
        unit_velocity_reference: Optional[str] = "angstroms/femtoseconds",
        tolerance: Optional[float] = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        # do a copy of the the universes
        pred_traj_modify = copy.copy(predicted_trajectory)
        ref_traj_modify = copy.copy(reference_trajectory)

        # First we check whether both trajectories have the same starting point
        if not np.allclose(
            pred_traj_modify.trajectory[0].positions,
            ref_traj_modify.trajectory[0].positions,
            tolerance,
        ):
            raise ValueError(
                "For this test, trajectories need to start from same frame!"
            )

        # we also need to be sure the velocities are identical, if given
        if pred_traj_modify.trajectory.ts.has_velocities and not np.allclose(
            pred_traj_modify.trajectory[0].velocities,
            ref_traj_modify.trajectory[0].velocities,
            tolerance,
        ):
            raise ValueError(
                "Check that velocities are also identical of initial frame!"
            )

        # next we need to make sure the compare the correct timesteps with each other
        # for this, we compute frame frequency (fq) and common length
        pred_fq, ref_fq, length, time_between_frames = _align_trajectories(
            pred_traj_modify, ref_traj_modify
        )
        # slice trajectories accordingly
        pred_traj_modify.trajectory = pred_traj_modify.trajectory[::pred_fq][:length]
        ref_traj_modify.trajectory = ref_traj_modify.trajectory[::ref_fq][:length]

        # compute temperatures
        pred_temperature = track_instantaneous_temperature(
            pred_traj_modify, linear_momentum_fixed, unit_velocity_predicted
        )

        ref_temperature = track_instantaneous_temperature(
            ref_traj_modify, linear_momentum_fixed, unit_velocity_reference
        )

        return (np.abs(pred_temperature - ref_temperature), time_between_frames)

    def _align_trajectories(
        traj1: Universe,
        traj2: Universe,
    ) -> Tuple[int, int, int, float]:
        """For two trajectories (expressed as Universes in MDAnalysis) compute their
        aligned frame frequencies and the max lengh to compare them.

        Args:
            traj1 (Universe): Trajectory 1.
            traj2 (Universe): Trajectory 2.
        Returns:
            fq1, fq2, length common, dt_common (Tuple[int, int, int, float]): Frame fequencies of each trajectory
                required to align them and the common length as well as the time between frames (dt_common).
        """
        # get timesteps
        dt1 = traj1.trajectory.dt
        dt2 = traj2.trajectory.dt

        # get least common multiple of the timestep
        dt_common = get_least_common_multiple(dt1, dt2)

        # frome dt to frame frequency we need
        fq1 = int(dt_common / dt1)
        fq2 = int(dt_common / dt2)

        # next get the correct length
        length_common = min(
            (
                traj1.trajectory.n_frames // fq1
                + (traj1.trajectory.n_frames % fq1 != 0),
                traj2.trajectory.n_frames // fq2
                + (traj2.trajectory.n_frames % fq2 != 0),
            )
        )

        return fq1, fq2, length_common, dt_common

except ImportError:
    print("All validation function built on MDAnalysis, please install MDAnalysis.")
