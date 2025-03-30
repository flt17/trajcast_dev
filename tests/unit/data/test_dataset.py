import os
import unittest
from typing import Optional

import torch

# from trajcast.utils.misc import
from trajcast.data._keys import (
    ATOMIC_NUMBERS_KEY,
    CELL_KEY,
    DISPLACEMENTS_KEY,
    PBC_KEY,
    POSITIONS_KEY,
    TIMESTEP_KEY,
    TOTAL_ENERGY_KEY,
    UPDATE_VELOCITIES_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.data.trajectory import ASETrajectory


class TestAtomicGraphDataset(unittest.TestCase):
    def test_returns_correct_graph_dataset_for_benzene_trajectory_without_pbc_crossing(
        self,
    ):
        prepare_trajectory_for_dataset(
            filename="benzene_lammps_example.xyz", wrapper="lammps"
        )

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="preprocessed_trajectory",
            files="preprocessed_trajectory.extxyz",
            cutoff_radius=5.0,
            atom_type_mapper={1: 0, 6: 1},
            rename=False,
            time_reversibility=True,
        )

        os.remove("tests/unit/data/data/preprocessed_trajectory.extxyz")
        [
            os.remove(os.path.join("tests/unit/data/data/", file_path))
            for file_path in [
                f for f in os.listdir("tests/unit/data/data/") if f.endswith(".pt")
            ]
        ]

        self.assertIsInstance(dataset[0], AtomicGraph)
        self.assertTrue(
            (torch.isclose(dataset[1].update_velocities, dataset[3].velocities))
            .all()
            .item()
        )

        self.assertAlmostEqual(dataset[7].update_velocities[5][2].item(), -0.00596801)
        self.assertAlmostEqual(dataset[10].pos[9][1], 14.00144)
        self.assertAlmostEqual(dataset[5].displacements[2][2].item(), 0.00054)
        # no pbc crossing
        # self.assertTrue(
        #     (torch.isclose(dataset[9].displacements, dataset[11].pos - dataset[9].pos)) # TODO: Fix this test
        #     .all()
        #     .item
        # )

    def test_returns_correct_graph_dataset_for_benzene_trajectory_with_pbc_crossing(
        self,
    ):
        prepare_trajectory_for_dataset(filename="benzene_cross_pbc_wrapped.extxyz")

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="preprocessed_trajectory",
            files="preprocessed_trajectory.extxyz",
            cutoff_radius=5.0,
            atom_type_mapper={1: 0, 6: 1},
            rename=False,
            time_reversibility=True,
        )

        self.assertIsInstance(dataset[0], AtomicGraph)

        self.assertTrue(
            (torch.isclose(dataset[1].update_velocities, dataset[3].velocities))
            .all()
            .item()
        )

        self.assertAlmostEqual(dataset[7].update_velocities[5][2].item(), -0.00794175)
        self.assertAlmostEqual(dataset[10].pos[9][1].item(), 19.99462699, delta=1e-6)
        self.assertAlmostEqual(dataset[5].displacements[2][2].item(), -0.00008)

        # compute displacements by hand
        # # compute by hand
        displacements_frame4_to_5 = dataset[9].pos - dataset[8].pos
        displacements_frame4_to_5[displacements_frame4_to_5 > 10] -= 20

        self.assertTrue(
            (torch.isclose(dataset[8].displacements, displacements_frame4_to_5))
            .all()
            .item
        )

        os.remove("tests/unit/data/data/preprocessed_trajectory.extxyz")
        [
            os.remove(os.path.join("tests/unit/data/data/", file_path))
            for file_path in [
                f for f in os.listdir("tests/unit/data/data/") if f.endswith(".pt")
            ]
        ]

    def test_returns_time_reversed_data(
        self,
    ):
        prepare_trajectory_for_dataset(filename="benzene_cross_pbc_wrapped.extxyz")

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="preprocessed_trajectory",
            files="preprocessed_trajectory.extxyz",
            cutoff_radius=5.0,
            atom_type_mapper={1: 0, 6: 1},
            rename=False,
            time_reversibility=True,
        )

        d1 = dataset[0]
        d2 = dataset[1]

        self.assertTrue((d1.displacements == -d2.displacements).all().item())
        self.assertTrue((d1.velocities == -d2.update_velocities).all().item())
        self.assertTrue((d2.velocities == -d1.update_velocities).all().item())
        self.assertTrue((d2.pos == d1.pos + d1.displacements).all().item())

    def test_returs_correct_dataset_with_multiple_files(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="benzene2x",
            files=[
                "benzene_cross_pbc_wrapped.extxyz",
                "benzene_cross_pbc_wrapped.extxyz",
            ],
            cutoff_radius=5.0,
            atom_type_mapper={1: 0, 6: 1},
            rename=False,
            time_reversibility=True,
        )

        self.assertEqual(len(dataset), 40)
        [
            os.remove(os.path.join("tests/unit/data/data/", file_path))
            for file_path in [
                f for f in os.listdir("tests/unit/data/data/") if f.endswith(".pt")
            ]
        ]

    def test_returns_correct_dataset_with_energy_included(self):
        prepare_trajectory_for_dataset(
            filename="aspirin_energy_array.xyz",
            wrapper="lammps",
            wrapper_dict={"lammps_units": "real", "desired_units": "real"},
            add_fields={
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
            },
            output_fields={
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
                ATOMIC_NUMBERS_KEY,
                POSITIONS_KEY,
                VELOCITIES_KEY,
                TIMESTEP_KEY,
                CELL_KEY,
                PBC_KEY,
                TOTAL_ENERGY_KEY,
            },
        )

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="preprocessed_trajectory",
            files="preprocessed_trajectory.extxyz",
            cutoff_radius=5.0,
            rename=False,
        )

        self.assertTrue(isinstance(dataset[0][TOTAL_ENERGY_KEY], torch.Tensor))

        os.remove(os.path.join("tests/unit/data/data/preprocessed_trajectory.extxyz"))
        os.remove(os.path.join("tests/unit/data/data/preprocessed_trajectory.pt"))


def prepare_trajectory_for_dataset(
    filename: str,
    root: Optional[str] = "tests/unit/data/data",
    wrapper: Optional[str] = None,
    wrapper_dict: Optional[dict] = {
        "lammps_units": "real",
        "type_mapping": {1: 6, 2: 1},
    },
    add_fields: Optional[set] = {DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY},
    output_fields: Optional[set] = {
        DISPLACEMENTS_KEY,
        UPDATE_VELOCITIES_KEY,
        ATOMIC_NUMBERS_KEY,
        POSITIONS_KEY,
        VELOCITIES_KEY,
        TIMESTEP_KEY,
        CELL_KEY,
        PBC_KEY,
    },
) -> None:
    ase_kwargs = {"format": "extxyz", "index": ":"}
    trajectory = ASETrajectory.read_from_file(
        root=root,
        filename=filename,
        frame_interval=1.0,
        wrapper=wrapper,
        wrapper_kwargs=wrapper_dict,
        **ase_kwargs,
    )
    trajectory.compute_additional_fields(
        add_fields=add_fields,
        time_step=1,
        truncate=True,
    )

    # write to file
    trajectory.write_to_file(
        root="tests/unit/data/data",
        filename_prefix="preprocessed_trajectory",
        chosen_fields=output_fields,
    )


if __name__ == "__main__":
    unittest.main()
