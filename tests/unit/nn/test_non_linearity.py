import unittest
from typing import Union

import ase
import torch
from e3nn.o3 import Irreps

from trajcast.data._keys import NODE_FEATURES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._non_linearity import GatedNonLinearity
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)

# This dictionary entails the average square for each activation function.
# For instance, for tanh this is computed via: (torch.tanh(torch.randn(1_000_000))).pow(2).mean()
# We do this as e3nn normalises the activation functions with the sqrt of this value to
# obey the 'component' normalisation.
MOM2_DICT = {
    "tanh": torch.tensor([0.3943]),
    "sigmoid": torch.tensor([0.2933]),
    "silu": torch.tensor([0.3557]),
    "relu": torch.tensor([0.500]),
}


class TestGatedNonLinearity(unittest.TestCase):
    def test_init_returns_error_if_no_irreps_for_input_given(self):
        with self.assertRaises(
            KeyError,
            msg=f"Could not infer the irreps for the field given. Please add the irrpes for {NODE_FEATURES_KEY}",
        ):
            GatedNonLinearity()

    def test_init_returns_error_if_all_irreps_given_do_not_add_up_to_irreps_input_field(
        self,
    ):
        irreps_in = {"input": Irreps("12x0e+5x1o")}
        irreps_scalar = Irreps("6x0e")
        irreps_gates = Irreps("7x0e")
        irreps_gated = Irreps("5x1o")
        with self.assertRaises(
            ValueError, msg="Irreps provided do not end up to perform a correct gate."
        ):
            GatedNonLinearity(
                input_field="input",
                irreps_scalars=irreps_scalar,
                irreps_gates=irreps_gates,
                irreps_gated=irreps_gated,
                irreps_in=irreps_in,
            )

    def test_init_returns_error_if_scalars_and_gates_given_do_not_add_up_to_irreps_input_field(
        self,
    ):
        irreps_in = {"input": Irreps("12x0e+5x1o")}
        irreps_scalar = Irreps("6x0e")
        irreps_gates = Irreps("7x0e")
        with self.assertRaises(
            ValueError,
            msg="The scalars and gates provided do not match up with the input irreps.",
        ):
            GatedNonLinearity(
                input_field="input",
                irreps_scalars=irreps_scalar,
                irreps_gates=irreps_gates,
                irreps_in=irreps_in,
            )

    def test_init_raises_error_when_scalars_given_leave_not_enough_scalars_as_gates(
        self,
    ):
        irreps_in = {"input": Irreps("12x0e+5x1o")}
        irreps_scalar = Irreps("8x0e")
        with self.assertRaises(
            ValueError,
            msg="If you use that many scalars you won't have enough left for gating.",
        ):
            GatedNonLinearity(
                input_field="input",
                irreps_scalars=irreps_scalar,
                irreps_in=irreps_in,
            )

    def test_init_raises_error_when_input_given_does_not_provide_sufficient_scalars_for_gating(
        self,
    ):
        irreps_in = {"input": Irreps("4x0e+5x1o")}

        with self.assertRaises(
            ValueError, msg="Not enough scalars to gate all the non-scalar irreps."
        ):
            GatedNonLinearity(
                input_field="input",
                irreps_in=irreps_in,
            )

    def test_init_returns_correct_layer_based_on_input_only_even_scalars_only(self):
        irreps_in = {"input": Irreps("12x0e+5x1o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
        )
        self.assertEqual(gate.gate.irreps_scalars, Irreps("7x0e"))
        self.assertEqual(gate.gate.irreps_gates, Irreps("5x0e"))
        self.assertEqual(gate.irreps_out["output"], Irreps("7x0e+5x1o"))

    def test_init_returns_correct_layer_based_on_input_only_mixed_scalars(self):
        irreps_in = {"input": Irreps("3x0o+3x0e+5x1o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
        )
        self.assertEqual(gate.gate.irreps_scalars, Irreps("1x0o"))
        self.assertEqual(gate.gate.irreps_gates, Irreps("2x0o+3x0e"))
        self.assertEqual(
            gate.irreps_out["output"],
            Irreps("1x0o+2x1e+3x1o").sort().irreps.simplify(),
        )

    def test_init_returns_correct_layer_based_on_input_only_odd_scalars_only(self):
        irreps_in = {"input": Irreps("12x0o+5x1o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
        )
        self.assertEqual(gate.gate.irreps_scalars, Irreps("7x0o"))
        self.assertEqual(gate.gate.irreps_gates, Irreps("5x0o"))
        self.assertEqual(gate.irreps_out["output"], Irreps("7x0o+5x1e"))

    def test_init_returns_correct_layer_if_scalars_are_passed_even_only(self):
        irreps_in = {"input": Irreps("12x0e+6x1o")}
        irreps_scalars = Irreps("6x0e")
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            irreps_scalars=irreps_scalars,
        )
        self.assertEqual(gate.gate.irreps_scalars, Irreps("6x0e"))
        self.assertEqual(gate.gate.irreps_gates, Irreps("6x0e"))
        self.assertEqual(gate.irreps_out["output"], Irreps("6x0e+6x1o"))

    def test_init_returns_correct_when_mixed_gates_and_scalars(self):
        irreps_in = {"input": Irreps("12x0e+12x0o+6x1o")}
        irreps_gates = Irreps("3x0e+3x0o")
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            irreps_gates=irreps_gates,
        )
        self.assertEqual(
            gate.gate.irreps_scalars.sort().irreps.simplify(), Irreps("9x0o+9x0e")
        )
        self.assertEqual(
            gate.gate.irreps_gates.sort().irreps.simplify(), Irreps("3x0o+3x0e")
        )
        self.assertEqual(gate.irreps_out["output"], Irreps("9x0o+9x0e+3x1o+3x1e"))

    def test_init_returns_correct_when_mixed_gates_and_scalars_with_different_activation_functions(
        self,
    ):
        irreps_in = {"input": Irreps("12x0e+12x0o+6x1o")}
        irreps_gates = Irreps("3x0e+3x0o")
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            irreps_gates=irreps_gates,
            activation_gates=["tanh", "silu"],
            activation_scalars=["tanh", "relu"],
        )
        self.assertEqual(
            gate.gate.irreps_scalars.sort().irreps.simplify(), Irreps("9x0o+9x0e")
        )
        self.assertEqual(
            gate.gate.irreps_gates.sort().irreps.simplify(), Irreps("3x0o+3x0e")
        )
        self.assertEqual(gate.irreps_out["output"], Irreps("9x0o+9x0e+3x1o+3x1e"))

    def test_init_returns_correct_when_mixed_gates_and_scalars_with_different_activation_functions_via_dictionary(
        self,
    ):
        irreps_in = {"input": Irreps("12x0e+12x0o+6x1o")}
        irreps_gates = Irreps("3x0e+3x0o")
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            irreps_gates=irreps_gates,
            activation_gates={"o": "tanh", "e": "silu"},
            activation_scalars={"o": "tanh", "e": "relu"},
        )
        self.assertEqual(
            gate.gate.irreps_scalars.sort().irreps.simplify(), Irreps("9x0o+9x0e")
        )
        self.assertEqual(
            gate.gate.irreps_gates.sort().irreps.simplify(), Irreps("3x0o+3x0e")
        )
        self.assertEqual(gate.irreps_out["output"], Irreps("9x0o+9x0e+3x1o+3x1e"))

    def test_init_returns_correct_when_all_scalars_as_gates(
        self,
    ):
        irreps_in = {"input": Irreps("4x0e+1x0o+3x1o+2x2e")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates={"o": "tanh", "e": "silu"},
            activation_scalars={"o": "tanh", "e": "silu"},
        )
        self.assertEqual(gate.gate.irreps_scalars.sort().irreps.simplify(), Irreps(""))
        self.assertEqual(
            gate.gate.irreps_gates.sort().irreps.simplify(), Irreps("1x0o+4x0e")
        )
        self.assertEqual(gate.irreps_out["output"], Irreps("2x1o+1x1e+2x2e"))

    def test_init_returns_correct_when_all_scalars_as_gates_with_dict_for_activations(
        self,
    ):
        irreps_in = {"input": Irreps("4x0e+1x0o+3x1o+2x2e")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh", "silu"],
            activation_scalars=[None],
        )
        self.assertEqual(gate.gate.irreps_scalars.sort().irreps.simplify(), Irreps(""))
        self.assertEqual(
            gate.gate.irreps_gates.sort().irreps.simplify(), Irreps("1x0o+4x0e")
        )
        self.assertEqual(gate.irreps_out["output"], Irreps("2x1o+1x1e+2x2e"))

    def test_forward_returns_correct_evaluation_with_all_even_scalars(self):
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = Irreps("11x0e+6x1o").randn(1, -1)

        # do gating manually
        scalars = graph.input[0][0:5]
        gates = graph.input[0][5:11]
        gated = graph.input[0][11:]

        scalars_manual = torch.sigmoid(scalars) * MOM2_DICT["sigmoid"].pow(-0.5)
        gates_manual = torch.tanh(gates) * MOM2_DICT["tanh"].pow(-0.5)
        gated_manual = torch.mul(gated.view(-1, 3), gates_manual.view(-1, 1)).view(-1)

        # init gate
        irreps_in = {"input": Irreps("11x0e+6x1o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh"],
            activation_scalars=["sigmoid"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(gate.irreps_out["output"], Irreps("5x0e+6x1o"))
        # compare with handwritten
        self.assertTrue(
            all(torch.isclose(scalars_manual, graph.output[0][0:5], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_manual, graph.output[0][5:], atol=5e-3))
        )

    def test_forward_correct_evaluation_with_all_odd_scalars(self):
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = Irreps("11x0o+7x1o").randn(1, -1)

        # do gating manually
        scalars = graph.input[0][0:4]
        gates = graph.input[0][4:11]
        gated = graph.input[0][11:]

        scalars_manual = torch.tanh(scalars) * MOM2_DICT["tanh"].pow(-0.5)
        gates_manual = torch.tanh(gates) * MOM2_DICT["tanh"].pow(-0.5)
        gated_manual = torch.mul(gated.view(-1, 3), gates_manual.view(-1, 1)).view(-1)

        # init gate
        irreps_in = {"input": Irreps("11x0o+7x1o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh"],
            activation_scalars=["tanh"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(gate.irreps_out["output"], Irreps("4x0o+7x1e"))
        # compare with handwritten
        self.assertTrue(
            all(torch.isclose(scalars_manual, graph.output[0][0:4], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_manual, graph.output[0][4:], atol=5e-3))
        )

    def test_forward_returns_correct_evaluation_with_mixed_scalars_and_mixed_acts(self):
        irreps_input = Irreps("11x0o+5x0e+7x1o+4x1e")
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = irreps_input.randn(1, -1)

        # do gating manually
        scalars_odd = graph.input[0][0:3]
        scalars_even = graph.input[0][11:13]
        gates_odd = graph.input[0][3:11]
        gates_even = graph.input[0][13:16]
        gated = graph.input[0][16:]

        scalars_manual = torch.cat(
            (
                torch.tanh(scalars_odd) * MOM2_DICT["tanh"].pow(-0.5),
                torch.nn.SiLU()(scalars_even) * MOM2_DICT["silu"].pow(-0.5),
            )
        )
        gates_manual = torch.cat(
            (
                torch.tanh(gates_odd) * MOM2_DICT["tanh"].pow(-0.5),
                torch.sigmoid(gates_even) * MOM2_DICT["sigmoid"].pow(-0.5),
            )
        )
        gated_manual = torch.mul(gated.view(-1, 3), gates_manual.view(-1, 1)).view(-1)

        # init gate
        irreps_in = {"input": irreps_input}
        irreps_gates = Irreps("8x0o+3x0e")
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            irreps_gates=irreps_gates,
            activation_gates=["tanh", "sigmoid"],
            activation_scalars=["tanh", "silu"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(gate.irreps_out["output"], Irreps("3x0o+2x0e+1x1o+10x1e"))
        # compare with handwritten
        self.assertTrue(
            all(torch.isclose(scalars_manual, graph.output[0][0:5], atol=1e-2))
        )
        self.assertTrue(
            all(torch.isclose(gated_manual, graph.output[0][5:], atol=1e-2))
        )

    def test_proves_normalisation_by_e3nn_for_all_acts(self):
        data = torch.randn(1_000_000)

        # tanh
        y_tanh = torch.tanh(data) * MOM2_DICT["tanh"].pow(-0.5)
        tanh_mom2 = y_tanh.pow(2).mean().item()
        self.assertAlmostEqual(tanh_mom2, 1.0, places=1)
        # relu
        y_relu = torch.relu(data) * MOM2_DICT["relu"].pow(-0.5)
        relu_mom2 = y_relu.pow(2).mean().item()
        self.assertAlmostEqual(relu_mom2, 1.0, places=1)
        # sigmoid
        y_sigmoid = torch.sigmoid(data) * MOM2_DICT["sigmoid"].pow(-0.5)
        sigmoid_mom2 = y_sigmoid.pow(2).mean().item()
        self.assertAlmostEqual(sigmoid_mom2, 1.0, places=1)
        # sigmoid
        y_silu = torch.nn.SiLU()(data) * MOM2_DICT["silu"].pow(-0.5)
        silu_mom2 = y_silu.pow(2).mean().item()
        self.assertAlmostEqual(silu_mom2, 1.0, places=1)

    def test_forward_returns_correct_evaluation_for_higher_order_vectors_with_all_even_scalars(
        self,
    ):
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = Irreps("20x0e+6x1o+2x2o+1x2e+4x3o").randn(1, -1)

        # do gating manually
        scalars = graph.input[0][0:7]
        gates = graph.input[0][7:20]
        gated_l1 = graph.input[0][20:38]
        gated_l2 = graph.input[0][38:53]
        gated_l3 = graph.input[0][53:]

        scalars_manual = torch.sigmoid(scalars) * MOM2_DICT["sigmoid"].pow(-0.5)
        gates_manual = torch.tanh(gates) * MOM2_DICT["tanh"].pow(-0.5)
        # now gate each l separately
        # l = 1
        gated_l1_manual = torch.mul(
            gated_l1.view(-1, 3), gates_manual[0:6].view(-1, 1)
        ).view(-1)
        # l = 2
        gated_l2_manual = torch.mul(
            gated_l2.view(-1, 5), gates_manual[6:9].view(-1, 1)
        ).view(-1)
        # l = 3
        gated_l3_manual = torch.mul(
            gated_l3.view(-1, 7), gates_manual[9:].view(-1, 1)
        ).view(-1)

        # init gate
        irreps_in = {"input": Irreps("20x0e+6x1o+2x2o+1x2e+4x3o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh"],
            activation_scalars=["sigmoid"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(gate.irreps_out["output"], Irreps("7x0e+6x1o+2x2o+1x2e+4x3o"))
        # compare with handwritten
        self.assertTrue(
            all(torch.isclose(scalars_manual, graph.output[0][0:7], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l1_manual, graph.output[0][7:25], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l2_manual, graph.output[0][25:40], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l3_manual, graph.output[0][40:], atol=5e-3))
        )

    def test_forward_returns_correct_evaluation_for_higher_order_vectors_with_all_odd_scalars(
        self,
    ):
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = Irreps("10x0o+3x1o+1x2e+5x3o").randn(1, -1)

        # do gating manually
        scalars = graph.input[0][0]
        gates = graph.input[0][1:10]
        gated_l1 = graph.input[0][10:19]
        gated_l2 = graph.input[0][19:24]
        gated_l3 = graph.input[0][24:]

        scalars_manual = torch.tanh(scalars) * MOM2_DICT["tanh"].pow(-0.5)
        gates_manual = torch.tanh(gates) * MOM2_DICT["tanh"].pow(-0.5)
        # now gate each l separately
        # l = 1
        gated_l1_manual = torch.mul(
            gated_l1.view(-1, 3), gates_manual[0:3].view(-1, 1)
        ).view(-1)
        # l = 2
        gated_l2_manual = torch.mul(
            gated_l2.view(-1, 5), gates_manual[3].view(-1, 1)
        ).view(-1)
        # l = 3
        gated_l3_manual = torch.mul(
            gated_l3.view(-1, 7), gates_manual[4:].view(-1, 1)
        ).view(-1)

        # init gate
        irreps_in = {"input": Irreps("10x0o+3x1o+1x2e+5x3o")}
        gate = GatedNonLinearity(
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh"],
            activation_scalars=["tanh"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(gate.irreps_out["output"], Irreps("1x0o+3x1e+1x2o+5x3e"))
        # compare with handwritten
        self.assertTrue(
            all(torch.isclose(scalars_manual, graph.output[0][0], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l1_manual, graph.output[0][1:10], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l2_manual, graph.output[0][10:15], atol=5e-3))
        )
        self.assertTrue(
            all(torch.isclose(gated_l3_manual, graph.output[0][15:], atol=5e-3))
        )

    def test_forward_returns_correct_evaluation_for_higher_order_vectors_with_mixed_scalars(
        self,
    ):
        # generate graph
        _, graph = SulphurAtom()
        # add the input irreps values to the graph
        graph["input"] = Irreps("5x0o+3x0e+3x1o+2x2o+1x3e").randn(1, -1)

        # do gating manually
        scalars_odd = graph.input[0][0]
        scalars_even = graph.input[0][5]
        gates_odd = graph.input[0][1:5]
        gates_even = graph.input[0][6:8]
        gated_l1 = graph.input[0][8:17]
        gated_l2 = graph.input[0][17:27]
        gated_l3 = graph.input[0][27:]

        scalars_manual = torch.cat(
            (
                torch.tanh(scalars_odd) * MOM2_DICT["tanh"].pow(-0.5),
                torch.relu(scalars_even) * MOM2_DICT["relu"].pow(-0.5),
            )
        )

        gates_manual = torch.cat(
            (
                torch.tanh(gates_odd) * MOM2_DICT["tanh"].pow(-0.5),
                torch.nn.SiLU()(gates_even) * MOM2_DICT["silu"].pow(-0.5),
            )
        )

        # now gate each l separately
        # l = 1
        gated_l1_manual = torch.mul(
            gated_l1.view(-1, 3), gates_manual[0:3].view(-1, 1)
        ).view(-1)
        # l = 2
        gated_l2_manual = torch.mul(
            gated_l2.view(-1, 5), gates_manual[3:5].view(-1, 1)
        ).view(-1)
        # l = 3
        gated_l3_manual = torch.mul(
            gated_l3.view(-1, 7), gates_manual[5:].view(-1, 1)
        ).view(-1)

        # init gate
        irreps_in = {"input": Irreps("5x0o+3x0e+3x1o+2x2o+1x3e")}
        gate = GatedNonLinearity(
            irreps_gates=Irreps("4x0o+2x0e"),
            input_field="input",
            output_field="output",
            irreps_in=irreps_in,
            activation_gates=["tanh", "silu"],
            activation_scalars=["tanh", "relu"],
        )
        # update
        graph = gate(graph)

        self.assertEqual(
            gate.irreps_out["output"], Irreps("1x0o+1x0e+3x1e+1x2o+1x2e+1x3e")
        )
        # compare with handwritten
        self.assertTrue(
            all(
                torch.isclose(
                    scalars_manual,
                    graph.output[0][0:2],
                    atol=1e-2,
                )
            )
        )
        self.assertTrue(
            all(torch.isclose(gated_l1_manual, graph.output[0][2:11], atol=1e-2))
        )
        self.assertTrue(
            all(torch.isclose(gated_l2_manual, graph.output[0][11:21], atol=1e-2))
        )
        self.assertTrue(
            all(torch.isclose(gated_l3_manual, graph.output[0][21:], atol=1e-2))
        )


def SulphurAtom() -> Union[ase.Atoms, AtomicGraph]:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    ase_atoms = ase.Atoms("S")
    graph = AtomicGraph.from_atoms_dict(
        atoms_dict=format_values_in_dictionary(
            convert_ase_atoms_to_dictionary(ase_atoms)
        ),
        r_cut=5.0,
    )

    return ase_atoms, graph


if __name__ == "__main__":
    unittest.main()
