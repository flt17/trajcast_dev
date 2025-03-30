import os
import unittest

import numpy as np
from ase import Atoms

from trajcast.data._keys import (
    ATOMIC_NUMBERS_KEY,
    CELL_KEY,
    DISPLACEMENTS_KEY,
    PBC_KEY,
    POSITIONS_KEY,
    TIMESTEP_KEY,
    UPDATE_VELOCITIES_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.trajectory import ASETrajectory


class TestASETrajectoryReadFromFile(unittest.TestCase):
    def test_raises_file_not_found_error(self):
        with self.assertRaises(
            FileNotFoundError,
            msg="Path: ./this_is_a_pseudo_filename.abc does not exist.",
        ):
            ASETrajectory.read_from_file(
                root="./", filename="this_is_a_pseudo_filename.abc"
            )

    def test_raises_type_error(self):
        with self.assertRaises(
            TypeError,
            msg="Expected a list but received ase.Atoms",
        ):
            ase_kwargs = {"format": "extxyz", "index": 0}
            ASETrajectory.read_from_file(
                root="./tests/unit/data/data",
                filename="md22_Ac-Ala3-NHMe_5frames.xyz",
                **ase_kwargs,
            )

    def test_returns_ase_object(self):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        trajec = ASETrajectory.read_from_file(
            root="./tests/unit/data/data",
            filename="md22_Ac-Ala3-NHMe_5frames.xyz",
            **ase_kwargs,
        )
        self.assertTrue(isinstance(trajec.data[0], Atoms))

        self.assertTrue(len(trajec.available_fields) == 7)

    def test_returns_correct_time_between_frames_when_only_every_second_frame(self):
        ase_kwargs = {"format": "extxyz", "index": "::2"}
        trajec = ASETrajectory.read_from_file(
            root="./tests/unit/data/data",
            filename="md22_Ac-Ala3-NHMe_5frames.xyz",
            frame_interval=5.0,
            **ase_kwargs,
        )

        self.assertEqual(trajec.time_between_frames, 10.0)

    def test_returns_ase_object_from_lammps_dump(self):
        ase_kwargs = {"index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajec = ASETrajectory.read_from_file(
            root="./tests/unit/data/data",
            filename="benzene_lammps_short.xyz",
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajec.data[0].positions[3][0], 2.608 + 10, delta=8
        )  # the +10 is because we recenter the cell (lammps is -10,10 and we go 0,20)
        self.assertAlmostEqual(
            trajec.data[1].arrays["velocities"][5][1], -0.00287309, delta=8
        )
        self.assertAlmostEqual(trajec.data[2].arrays["forces"][8][2], 0.671044, delta=8)
        self.assertEqual(trajec.data[-1].get_atomic_numbers()[0], 6)


class TestTrajectoryComputeAdditionalFields(unittest.TestCase):
    def test_returns_correct_displacements_for_ase_trajectory_frame_interval1_time_step1(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][0][1], 0.00401447, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY}, time_step=1, truncate=True
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_3 = (
            trajectory.data[3].positions - trajectory.data[2].positions
        )
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_3, displacements_function).all()
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 1.0)

    def test_returns_correct_displacements_for_ase_trajectory_frame_interval1_time_step2(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][0][1], 0.00401447, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY}, time_step=2, truncate=True
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_4 = (
            trajectory.data[4].positions - trajectory.data[2].positions
        )
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_4, displacements_function).all()
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 2.0)

    def test_returns_correct_displacements_for_ase_trajectory_frame_interval1_every_second_frame_time_step1(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": "::2"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[1].arrays["velocities"][1][0], -0.00475873, delta=8
        )
        self.assertEqual(2.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY}, time_step=1, truncate=True
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_4 = (
            trajectory.data[3].positions - trajectory.data[2].positions
        )
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_4, displacements_function).all()
        )
        self.assertEqual(2.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 2.0)

    def test_returns_correct_disps_and_vels_for_ase_trajectory_frame_interval1_time_step1(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][0][1], 0.00401447, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step=1,
            truncate=True,
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_3 = (
            trajectory.data[3].positions - trajectory.data[2].positions
        )
        vel_frame2_to_3 = trajectory.data[3].arrays["velocities"]
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        vel_function = trajectory.data[2].arrays[UPDATE_VELOCITIES_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_3, displacements_function).all()
        )
        self.assertTrue(np.isclose(vel_frame2_to_3, vel_function).all())
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 1.0)

    def test_returns_correct_disps_and_vels_for_ase_trajectory_frame_interval1_time_step2(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][0][1], 0.00401447, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step=2,
            truncate=True,
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_4 = (
            trajectory.data[4].positions - trajectory.data[2].positions
        )
        vel_frame2_to_4 = trajectory.data[4].arrays["velocities"]
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        vel_function = trajectory.data[2].arrays[UPDATE_VELOCITIES_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_4, displacements_function).all()
        )
        self.assertTrue(np.isclose(vel_frame2_to_4, vel_function).all())
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 2.0)

    def test_returns_correct_disps_and_vels_for_ase_trajectory_frame_interval1_time_step2_float(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][0][1], 0.00401447, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step_in_fs=2.0,
            truncate=True,
        )

        # compute by hand (no pbc crossing)
        displacements_frame2_to_4 = (
            trajectory.data[4].positions - trajectory.data[2].positions
        )
        vel_frame2_to_4 = trajectory.data[4].arrays["velocities"]
        # computed by function
        displacements_function = trajectory.data[2].arrays[DISPLACEMENTS_KEY]
        vel_function = trajectory.data[2].arrays[UPDATE_VELOCITIES_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_4, displacements_function).all()
        )
        self.assertTrue(np.isclose(vel_frame2_to_4, vel_function).all())
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 2.0)

    def test_returns_correct_disps_and_vels_for_ase_trajectory_frame_interval1_every_second_time_step4fs(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": "::2"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )

        self.assertAlmostEqual(
            trajectory.data[2].arrays["velocities"][3][2], -0.0036703, delta=8
        )
        self.assertEqual(2.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        # compute by hand (no pbc crossing)
        displacements_frame2_to_6 = (
            trajectory.data[3].positions - trajectory.data[1].positions
        )
        vel_frame2_to_6 = trajectory.data[3].arrays["velocities"]

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step_in_fs=4.0,
            truncate=True,
        )

        # computed by function
        displacements_function = trajectory.data[1].arrays[DISPLACEMENTS_KEY]
        vel_function = trajectory.data[1].arrays[UPDATE_VELOCITIES_KEY]
        self.assertTrue(
            np.isclose(displacements_frame2_to_6, displacements_function).all()
        )
        self.assertTrue(np.isclose(vel_frame2_to_6, vel_function).all())
        self.assertEqual(2.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 4.0)

    def test_returns_correct_disps_and_vels_for_boundary_crossing_molecule_frame_interval1_time_step1(
        self,
    ):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_cross_pbc_wrapped.extxyz",
            frame_interval=1.0,
            **ase_kwargs,
        )

        self.assertTrue(trajectory.is_wrapped)

        self.assertAlmostEqual(
            trajectory.data[0].arrays["velocities"][1][2], -0.00500151, delta=8
        )
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertNotIn(TIMESTEP_KEY, trajectory.data[0].info.keys())

        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step=1,
            truncate=True,
        )

        # # compute by hand
        displacements_frame4_to_5 = (
            trajectory.data[5].positions - trajectory.data[4].positions
        )
        displacements_frame4_to_5[displacements_frame4_to_5 > 10] -= 20
        vel_frame4_to_5 = trajectory.data[5].arrays["velocities"]
        # computed by function
        displacements_function = trajectory.data[4].arrays[DISPLACEMENTS_KEY]
        vel_function = trajectory.data[4].arrays[UPDATE_VELOCITIES_KEY]
        self.assertTrue(
            np.isclose(displacements_frame4_to_5, displacements_function).all()
        )
        self.assertTrue(np.isclose(vel_frame4_to_5, vel_function).all())
        self.assertEqual(1.0, trajectory.time_between_frames)
        self.assertIn(TIMESTEP_KEY, trajectory.data[0].info.keys())
        self.assertTrue(trajectory.data[0].info[TIMESTEP_KEY] == 1.0)


class TestASETrajectoryWriteToFile(unittest.TestCase):
    def test_raises_file_not_found_error(self):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="md22_Ac-Ala3-NHMe_5frames.xyz",
            **ase_kwargs,
        )
        with self.assertRaises(FileNotFoundError, msg="Directory does not exist!"):
            trajectory.write_to_file(root="./fake_directory/")

    def test_raises_key_error_due_to_not_existent_field(self):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="md22_Ac-Ala3-NHMe_5frames.xyz",
            **ase_kwargs,
        )

        with self.assertRaises(
            KeyError,
            msg="Some elements of chosen fields are not available in the trajectory or minimum requirements (atomic numbers and positions) not satisfied.",
        ):
            trajectory.write_to_file(
                root="tests/unit/data/data", chosen_fields={"fake_field"}
            )

    def test_raises_key_error_due_to_missing_field(self):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="md22_Ac-Ala3-NHMe_5frames.xyz",
            **ase_kwargs,
        )

        with self.assertRaises(
            KeyError,
            msg="Some elements of chosen fields are not available in the trajectory or minimum requirements (atomic numbers and positions) not satisfied.",
        ):
            trajectory.write_to_file(
                root="tests/unit/data/data", chosen_fields={ATOMIC_NUMBERS_KEY}
            )

    def test_writes_ase_trajectory_to_exyz_file(self):
        ase_kwargs = {"format": "extxyz", "index": ":"}
        wrapper_dict = {"lammps_units": "real", "type_mapping": {1: 6, 2: 1}}
        trajectory = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="benzene_lammps_example.xyz",
            frame_interval=1.0,
            wrapper="lammps",
            wrapper_kwargs=wrapper_dict,
            **ase_kwargs,
        )
        trajectory.compute_additional_fields(
            add_fields={DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
            time_step=1,
            truncate=True,
        )
        trajectory.write_to_file(
            root="tests/unit/data/data",
            filename_prefix="test_writer",
            chosen_fields=set(
                (
                    DISPLACEMENTS_KEY,
                    UPDATE_VELOCITIES_KEY,
                    ATOMIC_NUMBERS_KEY,
                    POSITIONS_KEY,
                    VELOCITIES_KEY,
                    TIMESTEP_KEY,
                    CELL_KEY,
                    PBC_KEY,
                )
            ),
        )

        self.assertTrue(os.path.exists("tests/unit/data/data/test_writer.extxyz"))

        traj_writ = ASETrajectory.read_from_file(
            root="tests/unit/data/data",
            filename="test_writer.extxyz",
            frame_interval=1.0,
            **ase_kwargs,
        )

        self.assertTrue(
            np.isclose(
                trajectory.data[3].arrays[DISPLACEMENTS_KEY],
                traj_writ.data[3].arrays[DISPLACEMENTS_KEY],
            ).all()
        )

        self.assertTrue(
            np.isclose(
                trajectory.data[2].arrays[UPDATE_VELOCITIES_KEY],
                traj_writ.data[2].arrays[UPDATE_VELOCITIES_KEY],
            ).all()
        )

        self.assertTrue(
            np.isclose(
                trajectory.data[1].arrays[POSITIONS_KEY],
                traj_writ.data[1].arrays[POSITIONS_KEY],
            ).all()
        )

        os.remove("tests/unit/data/data/test_writer.extxyz")

    def test_writes_to_npz_file(self):
        pass


if __name__ == "__main__":
    unittest.main()
