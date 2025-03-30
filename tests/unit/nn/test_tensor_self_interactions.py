import unittest
from typing import Union

import ase
import ase.build
import numpy as np
import torch
from e3nn.o3 import Irreps

from trajcast.data._keys import ATOM_TYPE_EMBEDDING_KEY, NODE_FEATURES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._encoding import OneHotAtomTypeEncoding
from trajcast.nn._tensor_self_interactions import LinearTensorMixer
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)


class TestLinearTensorMixer(unittest.TestCase):
    def test_raises_key_error_when_irreps_for_input_not_given(self):
        with self.assertRaises(KeyError, msg="Irreps for input_field Test not given"):
            LinearTensorMixer(input_field="Test")

    def test_returns_init_successfully_for_even_scalars(self):
        one_hot = OneHotAtomTypeEncoding(
            number_of_species=4, output_field=NODE_FEATURES_KEY
        )
        feature_mixer = LinearTensorMixer(
            input_field=NODE_FEATURES_KEY,
            irreps_in=one_hot.irreps_out,
            irreps_out="4x0e+20x1o",
        )

        # checking whether the module has in total 20 learnable parameters
        # 16 weights and 4 biases
        self.assertEqual(
            np.sum(
                [len(params) for _, params in feature_mixer.linear.named_parameters()]
            ),
            16,
        )

    def test_returns_init_for_pseudo_scalars(self):
        feature_mixer = LinearTensorMixer(
            input_field="input",
            irreps_in={"input": Irreps("4x0o")},
            irreps_out="4x0o+20x1o",
        )

        # biases are only added for real scalars (0e) not pseudo scalars (0o)
        self.assertEqual(feature_mixer.linear.bias.numel(), 0)

    def test_returns_init_successfully_for_even_scalars_with_mixed_scalars(
        self,
    ):
        feature_mixer = LinearTensorMixer(
            input_field="input",
            irreps_in={"input": Irreps("4x0o+5x0e")},
            irreps_out="4x0o+5x0e+20x1o",
        )

        # no biases
        self.assertEqual(feature_mixer.linear.bias.numel(), 0)

        self.assertEqual(
            np.sum(
                [len(params) for _, params in feature_mixer.linear.named_parameters()]
            ),
            41,
        )

    def test_returns_properly_mixed_features_for_different_fields(self):
        _, graph = CH3ONO()
        one_hot = OneHotAtomTypeEncoding(number_of_species=4)
        feature_mixer = LinearTensorMixer(
            input_field=ATOM_TYPE_EMBEDDING_KEY,
            output_field=NODE_FEATURES_KEY,
            irreps_in=one_hot.irreps_out,
        )
        updated_graph = one_hot(graph)
        final_graph = feature_mixer(updated_graph)

        self.assertTrue(
            torch.unique(final_graph[NODE_FEATURES_KEY], dim=0).size(0) == 4
        )

    def test_returns_properly_mixed_features_for_same_field_more_irreps(self):
        _, graph = CH3ONO()
        one_hot = OneHotAtomTypeEncoding(
            number_of_species=4, output_field=NODE_FEATURES_KEY
        )
        feature_mixer = LinearTensorMixer(
            input_field=NODE_FEATURES_KEY,
            irreps_in=one_hot.irreps_out,
            irreps_out="10x0e+20x1o",
        )

        updated_graph = one_hot(graph)
        final_graph = feature_mixer(updated_graph)
        self.assertTrue((final_graph[NODE_FEATURES_KEY][:, -60:] == 0).all())

    def test_returns_linear_output_is_normalised_irrespective_of_missing_irreps(
        self,
    ):
        _, graph = CH3ONO()

        mom2_with_0e = 0
        mom2_with_1o = 0
        for i in range(100):
            torch.manual_seed(i)
            node_features = torch.randn(7, 4)
            # module which only mixes the scalars without initialising 1o with 0s
            feature_mixer_only_0e = LinearTensorMixer(
                input_field=NODE_FEATURES_KEY,
                irreps_in={NODE_FEATURES_KEY: "4x0e"},
                irreps_out="10x0e",
            )
            # module which does the mixing and initialises 1o with zeros
            feature_mixer_with_1o = LinearTensorMixer(
                input_field=NODE_FEATURES_KEY,
                irreps_in={NODE_FEATURES_KEY: "4x0e"},
                irreps_out="10x0e+20x1o",
            )

            graph.node_features = node_features
            mom2_with_0e += feature_mixer_only_0e(graph).node_features.pow(2).mean()

            graph.node_features = node_features
            mom2_with_1o += (
                feature_mixer_with_1o((graph)).node_features[:, 0:10].pow(2).mean()
            )

        self.assertTrue(
            torch.isclose(
                torch.tensor(mom2_with_0e) / 100, torch.tensor(1.0), atol=1e-1
            )
        )
        # not properly normalised (component) if we account for 1o but only have scalar in the input
        self.assertTrue(
            torch.isclose(
                torch.tensor(mom2_with_1o) / 100, torch.tensor(1.0), atol=1e-1
            )
        )


def CH3ONO() -> Union[ase.Atoms, AtomicGraph]:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    ase_atoms = ase.build.molecule("CH3ONO")
    graph = AtomicGraph.from_atoms_dict(
        atoms_dict=format_values_in_dictionary(
            convert_ase_atoms_to_dictionary(ase_atoms)
        ),
        r_cut=5.0,
    )

    return ase_atoms, graph


if __name__ == "__main__":
    unittest.main()
