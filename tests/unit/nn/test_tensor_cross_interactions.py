import unittest

import torch
from e3nn.o3 import Irreps

from tests.unit.nn.test_modules import Si
from trajcast.data._keys import ADDITION_KEY, EDGE_VECTORS_KEY, SPHERICAL_HARMONIC_KEY
from trajcast.nn._encoding import SphericalHarmonicProjection
from trajcast.nn._tensor_cross_interactions import (
    DepthwiseTensorProduct,
    FieldsAddition,
)


class TestDepthwiseTensorProduct(unittest.TestCase):
    def test_returns_not_implemented_error_for_unknown_modes(self):
        irreps1 = Irreps("2x0e+2x1e+2e")
        irreps2 = Irreps("0e+1e")

        with self.assertRaises(NotImplementedError):
            DepthwiseTensorProduct(
                max_rotation_order=2,
                irreps_input1=irreps1,
                irreps_input2=irreps2,
                trainable=False,
                multiplicity_mode="new mode",
            )

    def test_returns_depthwise_tensor_product_without_weights(self):
        irreps1 = Irreps("2x0e+2x1e+2e")
        irreps2 = Irreps("0e+1e")

        dtp = DepthwiseTensorProduct(
            max_rotation_order=2,
            irreps_input1=irreps1,
            irreps_input2=irreps2,
            trainable=False,
        )

        self.assertEqual(dtp.weight_numel, 0)

    def test_returns_depthwise_tensor_product_for_uvu_mode(self):
        irreps1 = Irreps("2x0e+2x1e")
        irreps2 = Irreps("0e+5x1e")

        dtp = DepthwiseTensorProduct(
            max_rotation_order=2,
            irreps_input1=irreps1,
            irreps_input2=irreps2,
            multiplicity_mode="uvu",
        )
        self.assertTrue(all([mul == 2 for mul, _ in dtp.irreps_out]))

    def test_returns_depthwise_tensor_product_for_uvv_mode(self):
        irreps1 = Irreps("2x0e+2x1e")
        irreps2 = Irreps("10x1e")

        dtp = DepthwiseTensorProduct(
            max_rotation_order=2,
            irreps_input1=irreps1,
            irreps_input2=irreps2,
            multiplicity_mode="uvv",
        )
        self.assertTrue(all([mul == 10 for mul, _ in dtp.irreps_out]))

    def test_returns_depthwise_tensor_product_for_uvuv_mode(self):
        irreps1 = Irreps("2x0e+2x1e")
        irreps2 = Irreps("3x0e+3x1e")

        dtp = DepthwiseTensorProduct(
            max_rotation_order=2,
            irreps_input1=irreps1,
            irreps_input2=irreps2,
            multiplicity_mode="uvuv",
        )

        self.assertTrue(all([mul == 6 for mul, _ in dtp.irreps_out]))


class TestFieldsAddition(unittest.TestCase):
    def test_returns_key_error_if_not_all_fields_have_an_irrep_given(self):
        irreps_in = {"input1": Irreps("12x0e+5x1o")}
        input_fields = ["input1", "input2"]
        with self.assertRaises(KeyError):
            FieldsAddition(
                input_fields=input_fields,
                output_field="output_field",
                irreps_in=irreps_in,
            )

    def test_returns_key_error_if_not_all_fields_have_same_irrep(self):
        irreps_in = {"input1": Irreps("12x0e+5x1o"), "input2": Irreps("12x0e+8x1o")}
        input_fields = ["input1", "input2"]
        with self.assertRaises(KeyError):
            FieldsAddition(
                input_fields=input_fields,
                output_field="output_field",
                irreps_in=irreps_in,
            )

    def test_returns_correctly_initiliazed_module(self):
        irreps_in = {"input1": Irreps("12x0e+5x1o"), "input2": Irreps("12x0e+5x1o")}
        input_fields = ["input1", "input2"]
        mod = FieldsAddition(
            input_fields=input_fields,
            irreps_in=irreps_in,
        )

        self.assertTrue(
            mod.irreps_out[f"{ADDITION_KEY}_input1_input2"] == Irreps("12x0e+5x1o")
        )

    def test_returns_correct_module_output(self):
        _, graph = Si()
        graph.compute_edge_vectors()
        sh_encoding = SphericalHarmonicProjection(
            max_rotation_order=3,
        )
        add_fields = FieldsAddition(
            input_fields=[f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}"] * 2,
            output_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}_times_2",
            irreps_in=sh_encoding.irreps_out,
        )

        graph = sh_encoding(graph)
        graph = add_fields(graph)
        self.assertTrue(
            torch.allclose(
                graph[f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}"],
                graph[f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}_times_2"],
            )
        )


if __name__ == "__main__":
    unittest.main()
