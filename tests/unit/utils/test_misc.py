import unittest

import ase
import ase.build
import torch
from e3nn.o3 import Irreps

from trajcast.data._keys import TOTAL_ENERGY_KEY
from trajcast.utils.misc import (
    check_irreps_is_subset,
    convert_ase_atoms_to_dictionary,
    convert_units,
    determine_irreps_for_gate,
    get_activation_functions_from_dict,
    get_activation_functions_from_list,
    GLOBAL_DEVICE,
    Device,
)


class TestDevice(unittest.TestCase):
    def test_initialization(self):
        d = Device()
        self.assertEqual(d.device, torch.device("cpu"))

    def test_set_cpu(self):
        d = Device()
        d.device = "cpu"
        self.assertEqual(d.device, torch.device("cpu"))

    def test_set_cuda(self):
        d = Device()
        d.device = "cuda"
        self.assertEqual(d.device, torch.device("cuda"))

    def test_global_device_initialization(self):
        self.assertEqual(GLOBAL_DEVICE.device, torch.device("cpu"))

    def test_global_device_set(self):
        GLOBAL_DEVICE.device = "cuda"
        self.assertEqual(GLOBAL_DEVICE.device, torch.device("cuda"))
        GLOBAL_DEVICE.device = "cpu"
        self.assertEqual(GLOBAL_DEVICE.device, torch.device("cpu"))


class TestDetermineIrrepsForGate(unittest.TestCase):
    def test_returns_irreps_when_only_input_given_with_only_even_scalars(self):
        irreps_input = Irreps("30x0e+10x1o+15x2e")
        irreps_scalars, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input
        )

        self.assertEqual(irreps_scalars, Irreps("5x0e"))
        self.assertEqual(irreps_gates, Irreps("25x0e"))
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_returns_irreps_when_only_input_given_with_only_odd_scalars(self):
        irreps_input = Irreps("30x0o+10x1o+15x2e")
        irreps_scalars, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input
        )

        self.assertEqual(irreps_scalars, Irreps("5x0o"))
        self.assertEqual(irreps_gates, Irreps("25x0o"))
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_returns_irreps_when_only_input_given_with_mixed_scalars(self):
        irreps_input = Irreps("15x0o+15x0e+10x1o+15x2e")
        irreps_scalars, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input
        )
        # by default we use only/all even scalars
        self.assertEqual(irreps_scalars, Irreps("5x0o"))
        self.assertEqual(
            irreps_gates.sort().irreps, Irreps("15x0e+10x0o").sort().irreps
        )
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_raises_error_when_only_input_given_but_not_sufficient_scalars_for_gating(
        self,
    ):
        irreps_input1 = Irreps("15x0o+10x1o+15x2e")
        with self.assertRaises(
            ValueError,
            msg="You don't have sufficient scalars in the input to gate all tensors.",
        ):
            determine_irreps_for_gate(irreps_input=irreps_input1)

        irreps_input2 = Irreps("15x0e+10x1o+15x2e")
        with self.assertRaises(
            ValueError,
            msg="You don't have sufficient scalars in the input to gate all tensors.",
        ):
            determine_irreps_for_gate(irreps_input=irreps_input2)

        irreps_input3 = Irreps("15x0e+5x0o+10x1o+15x2e")
        with self.assertRaises(
            ValueError,
            msg="You don't have sufficient scalars in the input to gate all tensors.",
        ):
            determine_irreps_for_gate(irreps_input=irreps_input3)

    def test_returns_irreps_when_input_and_scalars_are_given_with_even_scalars_only(
        self,
    ):
        irreps_input = Irreps("30x0e+10x1o+15x2e")
        irreps_scalars = Irreps("5x0e")
        _, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_scalars=irreps_scalars
        )
        self.assertEqual(irreps_gates, Irreps("25x0e"))
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_returns_irreps_when_input_and_scalars_are_given_with_odd_scalars_only(
        self,
    ):
        irreps_input = Irreps("30x0o+10x1o+15x2e")
        irreps_scalars = Irreps("5x0o")
        _, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_scalars=irreps_scalars
        )
        self.assertEqual(irreps_gates, Irreps("25x0o"))
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_returns_irreps_when_input_and_scalars_are_given_with_mixed_scalars(
        self,
    ):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_scalars = Irreps("30x0e+5x0o")
        _, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_scalars=irreps_scalars
        )
        self.assertEqual(irreps_gates, Irreps("25x0o"))
        self.assertEqual(irreps_gated, Irreps("10x1o+15x2e"))

    def test_raises_error_when_scalars_given_leave_not_enough_scalars_as_gates(
        self,
    ):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_scalars = Irreps("30x0e+10x0o")
        with self.assertRaises(
            ValueError,
            msg="If you use that many scalars you won't have enough or too many left for gating.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input, irreps_scalars=irreps_scalars
            )

    def test_raises_error_when_scalars_given_leave_to_many_scalars_as_gates(self):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_scalars = Irreps("10x0e+10x0o")
        with self.assertRaises(
            ValueError,
            msg="If you use that many scalars you won't have enough or too many left for gating.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input, irreps_scalars=irreps_scalars
            )

    def test_raises_error_when_scalars_given_are_not_part_of_input_irreps(self):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_scalars = Irreps("40x0o")
        with self.assertRaises(
            ValueError,
            msg="Given scalars are not even contained in the input irreps.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input, irreps_scalars=irreps_scalars
            )

    def test_raises_error_when_gates_given_are_not_part_of_input_irreps(self):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_gates = Irreps("40x0e")
        with self.assertRaises(
            ValueError,
            msg="Given gates are not even contained in the input irreps.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input, irreps_gates=irreps_gates
            )

    def test_raises_error_when_n_gates_do_not_correspond_to_non_scalar_number(self):
        irreps_input = Irreps("30x0o+30x0e+10x1o+15x2e")
        irreps_gates = Irreps("30x0e")
        with self.assertRaises(
            ValueError,
            msg="Number of gates does not correspond to non-scalar irreps.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input, irreps_gates=irreps_gates
            )

    def test_returns_irreps_for_gates_given_even_only(self):
        irreps_input = Irreps("6x0o+5x0e+3x1o+2x2e")
        irreps_gates = Irreps("5x0e")
        irreps_scalars, _, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_gates=irreps_gates
        )
        self.assertEqual(irreps_scalars, Irreps("6x0o"))
        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))

    def test_returns_irreps_for_gates_given_odd_only(self):
        irreps_input = Irreps("6x0o+5x0e+3x1o+2x2e")
        irreps_gates = Irreps("5x0o")
        irreps_scalars, _, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_gates=irreps_gates
        )
        self.assertEqual(irreps_scalars, Irreps("1x0o+5x0e"))
        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))

    def test_returns_irreps_for_gates_given_mixed_scalars(self):
        irreps_input = Irreps("6x0o+5x0e+3x1o+2x2e")
        irreps_gates = Irreps("3x0o+2x0e")
        irreps_scalars, _, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_gates=irreps_gates
        )
        self.assertEqual(irreps_scalars, Irreps("3x0o+3x0e"))
        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))

    def test_returns_irreps_for_gates_given_mixed_scalars_reversed(self):
        irreps_input = Irreps("6x0o+5x0e+3x1o+2x2e")
        irreps_gates = Irreps("2x0e+3x0o")
        irreps_scalars, _, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input, irreps_gates=irreps_gates
        )
        self.assertEqual(irreps_scalars, Irreps("3x0o+3x0e"))
        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))

    def test_raises_error_if_for_all_given_irreps_do_not_end_up_correctly(self):
        irreps_input = Irreps("8x0o+4x0e+3x1o+2x2e")
        irreps_gates = Irreps("3x0o+2x0e")
        irreps_scalars = Irreps("5x0o+1x0e")
        with self.assertRaises(
            ValueError,
            msg="The scalars and gates provided do not match up with the input irreps.",
        ):
            determine_irreps_for_gate(
                irreps_input=irreps_input,
                irreps_scalars=irreps_scalars,
                irreps_gates=irreps_gates,
            )

    def test_returns_correct_gated_without_errors_when_everything_is_given(self):
        irreps_input = Irreps("8x0o+4x0e+3x1o+2x2e")
        irreps_gates = Irreps("3x0o+2x0e")
        irreps_scalars = Irreps("5x0o+2x0e")
        _, _, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input,
            irreps_scalars=irreps_scalars,
            irreps_gates=irreps_gates,
        )
        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))

    def test_returns_scalars_none_when_all_scalars_are_used_as_gates(self):
        irreps_input = Irreps("4x0e+1x0o+3x1o+2x2e")
        irreps_scalars, irreps_gates, irreps_gated = determine_irreps_for_gate(
            irreps_input=irreps_input
        )

        self.assertEqual(irreps_gated, Irreps("3x1o+2x2e"))
        self.assertEqual(irreps_gates, Irreps("1x0o+4x0e"))
        self.assertEqual(irreps_scalars, Irreps(None))


class TestCheckIrrepsIsSubset(unittest.TestCase):
    def test_returns_true_for_single_irrep(self):
        irreps_in = Irreps("0e")
        irreps_all = Irreps("0e+1o")

        self.assertTrue(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )

    def test_returns_false_for_single_irrep(self):
        irreps_in = Irreps("0e")
        irreps_all = Irreps("1o")

        self.assertFalse(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )

    def test_returns_true_for_single_irrep_with_higher_multiplicity(self):
        irreps_in = Irreps("2x0e")
        irreps_all = Irreps("10x0e+1o")

        self.assertTrue(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )

    def test_returns_false_for_single_irrep_with_higher_multiplicty(self):
        irreps_in = Irreps("2x0e")
        irreps_all = Irreps("1o+0e")

        self.assertFalse(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )

    def test_returns_true_for_multiple_irreps(self):
        irreps_in = Irreps("2x0e+3x0o")
        irreps_all = Irreps("2x1o+10x0e+3x0o")

        self.assertTrue(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )

    def test_returns_true_for_unsorted_irreps(self):
        irreps_in = Irreps("2x0e+2x1o+8x0e")
        irreps_all = Irreps("10x0e+2x1o")

        self.assertTrue(
            check_irreps_is_subset(irreps_in=irreps_in, irreps_all=irreps_all)
        )


class TestConvertUnits(unittest.TestCase):
    def test_converts_distance_unit_correctly(self):
        conv_factor = convert_units("angstrom", "meter")
        self.assertEqual(conv_factor, 1e-10)

    def test_converts_ps_to_fs_correctly(self):
        conv_factor = convert_units("ps", "fs")
        self.assertAlmostEqual(conv_factor, 1e3)

    def test_converts_fs_to_ps_correctly(self):
        conv_factor = convert_units("fs", "ps")
        self.assertAlmostEqual(conv_factor, 1e-3)

    def test_returns_error_when_first_unit_unknown(self):
        with self.assertRaises(
            KeyError,
            msg="Unit fake_unit not known, don't know whether this is a distance or time or whatever.",
        ):
            convert_units("fake_unit", "fs")

    def test_returns_error_when_second_unit_unknown(self):
        with self.assertRaises(
            KeyError,
            msg="Unit fake_unit2 not known, don't know whether this is a distance or time or whatever.",
        ):
            convert_units("ps", "fake_unit2")


class TestGetActivationFunctionFromDict(unittest.TestCase):
    def test_raises_error_for_wrong_keys_in_dict(self):
        scalar_irreps = Irreps("1x0e")
        dict_act = {"p": "silu"}
        with self.assertRaises(
            KeyError, msg="Only odd (key 'o') and even (key 'e') parity allowed."
        ):
            get_activation_functions_from_dict(
                scalar_irreps=scalar_irreps, dict_act=dict_act
            )

    def test_raises_error_when_scalar_irreps_not_scalar(self):
        scalar_irreps = Irreps("1x0e+1x1o")
        dict_act = {"e": "silu", "o": "tanh"}
        with self.assertRaises(ValueError, msg="Scalar irreps are not scalar only."):
            get_activation_functions_from_dict(
                scalar_irreps=scalar_irreps, dict_act=dict_act
            )

    def test_raises_error_when_dict_does_not_contain_all_parities_in_scalar_irreps(
        self,
    ):
        scalar_irreps = Irreps("1x0o+1x0e")
        dict_act = {"e": "silu"}
        with self.assertRaises(
            KeyError,
            msg="Activation functions not for both parities given in dict_act.",
        ):
            get_activation_functions_from_dict(
                scalar_irreps=scalar_irreps, dict_act=dict_act
            )

    def test_raises_error_when_scalar_irreps_not_simplified(
        self,
    ):
        scalar_irreps = Irreps("1x0o+1x0e+1x0o")
        dict_act = {"e": "silu"}
        with self.assertRaises(
            ValueError,
            msg="Simplify irreps first.",
        ):
            get_activation_functions_from_dict(
                scalar_irreps=scalar_irreps, dict_act=dict_act
            )

    def test_returns_correct_activation_function_for_even_scalar_only(self):
        scalar_irreps = Irreps("8x0e")
        dict_act = {"e": "silu", "o": "tanh"}
        acts = get_activation_functions_from_dict(
            scalar_irreps=scalar_irreps, dict_act=dict_act
        )

        self.assertIsInstance(acts, list)
        self.assertEqual(len(acts), 1)
        self.assertIsInstance(acts[0], torch.nn.SiLU)

    def test_returns_correct_activation_function_for_odd_scalar_only(self):
        scalar_irreps = Irreps("8x0o")
        dict_act = {"e": "silu", "o": "tanh"}
        acts = get_activation_functions_from_dict(
            scalar_irreps=scalar_irreps, dict_act=dict_act
        )

        self.assertIsInstance(acts, list)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0], torch.tanh)

    def test_returns_correct_activation_function_for_odd_and_even_scalar(self):
        scalar_irreps = Irreps("8x0o+2x0e")
        dict_act = {"e": "silu", "o": "tanh"}
        acts = get_activation_functions_from_dict(
            scalar_irreps=scalar_irreps, dict_act=dict_act
        )
        self.assertIsInstance(acts, list)
        self.assertEqual(len(acts), 2)
        self.assertIsInstance(acts[1], torch.nn.SiLU)
        self.assertEqual(acts[0], torch.tanh)

    def test_returns_correct_activation_function_for_even_and_odd_scalar(self):
        scalar_irreps = Irreps("1x0e+3x0o")
        dict_act = {"e": "silu", "o": "tanh"}
        acts = get_activation_functions_from_dict(
            scalar_irreps=scalar_irreps, dict_act=dict_act
        )

        self.assertIsInstance(acts, list)
        self.assertEqual(len(acts), 2)
        self.assertIsInstance(acts[0], torch.nn.SiLU)
        self.assertEqual(acts[1], torch.tanh)

    def test_returns_correct_empty_list_when_no_scalars_given(self):
        scalar_irreps = Irreps("")
        dict_act = {"e": "silu", "o": "tanh"}
        acts = get_activation_functions_from_dict(
            scalar_irreps=scalar_irreps, dict_act=dict_act
        )
        self.assertIsInstance(acts, list)
        self.assertEqual(len(acts), 0)


class TestGetActivationFunctionFromList(unittest.TestCase):
    def test_returns_one_tanh(self):
        n_act = 1
        list_act = ["tanh"]
        acts = get_activation_functions_from_list(
            n_functions_needed=n_act, list_act=list_act
        )

        self.assertEqual(acts, [torch.tanh])

    def test_returns_two_tanh_with_only_list_act_len_1(self):
        n_act = 2
        list_act = ["tanh"]
        acts = get_activation_functions_from_list(
            n_functions_needed=n_act, list_act=list_act
        )

        self.assertEqual(acts, [torch.tanh, torch.tanh])

    def test_raises_error_if_elements_in_list_act_larger_than_functions_needed(self):
        n_act = 1
        list_act = ["tanh", "silu"]
        with self.assertRaises(
            ValueError,
            msg="List contains more functions then needed, how should I pick?",
        ):
            get_activation_functions_from_list(
                n_functions_needed=n_act, list_act=list_act
            )

    def test_returns_different_activation_functions(self):
        n_act = 2
        list_act = ["tanh", "relu"]
        acts = get_activation_functions_from_list(
            n_functions_needed=n_act, list_act=list_act
        )
        self.assertEqual(acts, [torch.tanh, torch.relu])

    def test_returns_single_silu_correctly(self):
        # for silu we need a test case
        test_tensor = torch.Tensor([0.5])
        n_act = 1
        list_act = ["silu"]
        acts = get_activation_functions_from_list(n_act, list_act)
        self.assertEqual(acts[0](test_tensor), torch.nn.SiLU()(test_tensor))

    def test_returns_three_sigmoids_correctly(self):
        n_act = 3
        list_act = ["sigmoid"]
        acts = get_activation_functions_from_list(n_act, list_act)
        self.assertEqual(acts, [torch.sigmoid, torch.sigmoid, torch.sigmoid])

    def test_returns_activations_correctly_case_sensitive_for_silu_relu(self):
        list_act = ["relu", "ReLU"]
        acts = get_activation_functions_from_list(
            n_functions_needed=2, list_act=list_act
        )
        self.assertEqual(acts, [torch.relu, torch.relu])

        list_act = ["silu", "SiLU"]
        acts = get_activation_functions_from_list(
            n_functions_needed=2, list_act=list_act
        )
        test_tensor = torch.Tensor([0.5])
        self.assertEqual(acts[0](test_tensor), torch.nn.SiLU()(test_tensor))
        self.assertEqual(acts[1](test_tensor), torch.nn.SiLU()(test_tensor))


class TestConvertAseAtomsToDictionary(unittest.TestCase):
    def test_returns_correct_dictionary_with_renaming(self):
        ase_atoms = H2O()
        dictionary = convert_ase_atoms_to_dictionary(ase_atoms_object=ase_atoms)

        self.assertNotIn("info", dictionary.keys())
        self.assertIn(TOTAL_ENERGY_KEY, dictionary.keys())

    def test_returns_correct_dictionary_without_renaming(self):
        ase_atoms = H2O()
        dictionary = convert_ase_atoms_to_dictionary(
            ase_atoms_object=ase_atoms,
            rename=False,
        )
        self.assertNotIn("info", dictionary.keys())
        self.assertIn("energy", dictionary.keys())


def H2O() -> ase.Atoms:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    atoms = ase.build.molecule("H2O")
    atoms.info["energy"] = -100.0
    return atoms


if __name__ == "__main__":
    unittest.main()
