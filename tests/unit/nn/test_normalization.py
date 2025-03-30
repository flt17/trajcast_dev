import unittest

import torch

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.nn._normalization import NormalizationLayer


class TestNormalizationLayer(unittest.TestCase):
    def test_init_returns_error_when_different_types_for_input_output_fields(self):
        with self.assertRaises(
            TypeError, msg="Input and output fields should be of the same type!"
        ):
            NormalizationLayer(input_fields="Test", output_fields=["Test"])

    def test_init_returns_error_when_different_lengths_for_input_output_fields(self):
        with self.assertRaises(
            IndexError, msg="We need as many output_fields as input_fields."
        ):
            NormalizationLayer(input_fields=["Test", "Test2"], output_fields=["Test"])

    def test_init_returns_error_when_mean_std_keys_dont_match_input_fields(self):
        with self.assertRaises(
            KeyError,
            msg="Keys in means and stds should correspond to those in input_fields!",
        ):
            NormalizationLayer(
                input_fields=["Test"], output_fields=["Test"], means={"Test2": 2.0}
            )

    def test_forward_returns_transformation_works_correct_for_one_input(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/",
            name="Peptide",
            cutoff_radius=4.0,
            files="md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
            rename=False,
        )

        all_displacements = getattr(dataset, DISPLACEMENTS_KEY)

        mean = all_displacements.mean()

        std = torch.sqrt(torch.mean(all_displacements**2))

        transform = NormalizationLayer(
            input_fields=DISPLACEMENTS_KEY,
            output_fields=DISPLACEMENTS_KEY,
            means={DISPLACEMENTS_KEY: mean},
            stds={DISPLACEMENTS_KEY: std},
        )

        data = transform(dataset[0])
        sec_moment = data[DISPLACEMENTS_KEY].pow(2).mean()
        self.assertTrue(
            torch.isclose(
                sec_moment,
                torch.tensor(1.0),
                atol=1e-1,
            )
        )

    def test_forward_returns_transformation_works_correct_for_one_input_if_multiple_given(
        self,
    ):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/",
            name="Peptide",
            cutoff_radius=4.0,
            files="md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
            rename=False,
        )

        all_displacements = getattr(dataset, DISPLACEMENTS_KEY)

        mean = all_displacements.mean()

        std = torch.sqrt(torch.mean(all_displacements**2))

        transform = NormalizationLayer(
            input_fields=[DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY],
            output_fields=[DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY],
            means={DISPLACEMENTS_KEY: mean, UPDATE_VELOCITIES_KEY: 0},
            stds={DISPLACEMENTS_KEY: std, UPDATE_VELOCITIES_KEY: 0.001},
        )

        data = transform(dataset[0])
        sec_moment = data[DISPLACEMENTS_KEY].pow(2).mean()
        self.assertTrue(
            torch.isclose(
                sec_moment,
                torch.tensor(1.0),
                atol=1e-1,
            )
        )

    def test_forward_returns_retransformation_works_correct_for_one_input(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/",
            name="Peptide",
            cutoff_radius=4.0,
            files="md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
            rename=False,
        )

        all_displacements = getattr(dataset, DISPLACEMENTS_KEY)
        mean = all_displacements.mean()
        std = torch.sqrt(torch.mean(all_displacements**2))
        transform = NormalizationLayer(
            input_fields=DISPLACEMENTS_KEY,
            output_fields=DISPLACEMENTS_KEY,
            means={DISPLACEMENTS_KEY: mean},
            stds={DISPLACEMENTS_KEY: std},
        )
        retransform = NormalizationLayer(
            input_fields=DISPLACEMENTS_KEY,
            output_fields=DISPLACEMENTS_KEY,
            means={DISPLACEMENTS_KEY: mean},
            stds={DISPLACEMENTS_KEY: std},
            inverse=True,
        )
        disp0 = dataset[0][DISPLACEMENTS_KEY]
        data = transform(dataset[0])
        data = retransform(data)
        assert torch.allclose(disp0, data[DISPLACEMENTS_KEY])


if __name__ == "__main__":
    unittest.main()
