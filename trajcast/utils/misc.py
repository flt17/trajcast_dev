import math
from typing import Dict, List, Optional, Union

import ase
import numpy as np
import torch
from ase.units import _Nav
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irrep, Irreps

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    ATOMIC_NUMBERS_KEY,
    INPUT_KEY_MAPPING,
)
from trajcast.data._types import FIELD_TYPES

ACTIVATION_FUNCTIONS = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "silu": torch.nn.SiLU(),
    "SiLU": torch.nn.SiLU(),
    "ReLU": torch.relu,
    "relu": torch.relu,
    "gelu": torch.nn.GELU(),
    "GeLU": torch.nn.GELU(),
    "Identity": torch.nn.Identity(),
    "identity": torch.nn.Identity(),
}

# conversion factors can be found here https://en.wikipedia.org/wiki/Metric_prefix
SI_UNIT_DICT = {
    "m": ["angstroms", "angstrom", "Angstrom", "Angstroms", "m", "meter", "meters"],
    "s": [
        "ns",
        "ps",
        "fs",
        "s",
        "nanoseconds",
        "nanosecond",
        "picoseconds",
        "picosecond",
        "femtoseconds",
        "femtosecond",
        "seconds",
        "second",
    ],
    "kg": [
        "grams/mole",
        "g/mol",
        "kg",
    ],
}

CONV_FACTORS_TO_SI_UNIT = {
    1e-10: ["angstroms", "angstrom", "Angstrom", "Angstroms"],
    1e-15: ["fs", "femtoseconds", "femtosecond"],
    1e-12: ["ps", "picoseconds", "picosecond"],
    1e-9: ["ns", "nanoseconds", "nanosceond"],
    1: ["s", "seconds", "m", "meter", "meters", "kg"],
    1e-3 / _Nav: ["grams/mole", "g/mol"],
}


class Device:
    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str = None):
        self._device = torch.device(device)


global GLOBAL_DEVICE
GLOBAL_DEVICE = Device()


def invert_dictionary(dictionary: Dict[str, str]):
    """Inverse dictionary such that the original value becomes the key, and the original key is the value.

    Args:
        dictionary (Dict[str, str]): _description_

    Returns:
        _type_: _description_
    """
    return {value: key for key, values in dictionary.items() for value in values}


def string2index(string: str) -> Union[int, slice, str]:
    """Convert index string to either int or slice.
    This part of the code has been taken from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/io/formats.html#read.
    """
    if ":" not in string:
        # may contain database accessor
        try:
            return int(string)
        except ValueError:
            return string
    i: List[Optional[int]] = []
    for s in string.split(":"):
        if s == "":
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)


def truncate_dictionary(
    dictionary: Dict,
    n_values: int,
    keys_to_truncate: Optional[List[str]] = None,
) -> Dict:
    """Truncate a dictionary based on index

    Args:
        dictionary (Dict): Original dictionary
        n_values (int): Take first n_values.

    Returns:
        Dict: Truncated dictionary
    """
    if keys_to_truncate is None:
        keys_to_truncate = dictionary.keys()

    return {
        keys: values[:n_values]
        for keys, values in dictionary.items()
        if keys in keys_to_truncate
    }


def guess_filetype(filename: str) -> str:
    file_suffix = filename.split(".")[-1]
    if "extxyz" in file_suffix or "exyz" in file_suffix:
        return "extxyz"
    if "npz" in file_suffix:
        return "npz"
    if "pt" in file_suffix:
        return "pytorch"
    if "yaml" in file_suffix:
        return "yaml"

    return file_suffix


def convert_ase_atoms_to_dictionary(
    ase_atoms_object: ase.Atoms,
    rename: Optional[bool] = True,
    key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
) -> Dict:
    """Converts ase.Atoms into dictionary

    Args:
        ase_atoms_object (ase.Atoms): _description_
        rename (Optional[bool], optional): _description_. Defaults to True.
        key_mapping (Optional[Dict[str, str]], optional): _description_. Defaults to invert_dictionary(INPUT_KEY_MAPPING).

    Returns:
        Dict: _description_
    """

    # convert ase_atoms_object to dictionary
    dictionary = ase_atoms_object.todict()

    # a lot of information is given in info
    # check whether available
    if "info" in dictionary.keys():
        additional_information = dictionary["info"]
        for key, value in additional_information.items():
            dictionary[key] = value

        dictionary.pop("info")

    # once this is done we can rename the keys based on a key_mapping
    if rename:
        dictionary = {
            key_mapping.get(key, key): value for key, value in dictionary.items()
        }

    # add masses
    dictionary[ATOMIC_MASSES_KEY] = ase_atoms_object.get_masses().reshape(-1, 1)
    dictionary[ATOMIC_NUMBERS_KEY] = ase_atoms_object.get_atomic_numbers().reshape(
        -1, 1
    )
    return dictionary


def convert_npz_to_dictionary(
    npz: np.lib.npyio.NpzFile,
    rename: Optional[bool] = True,
    key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
) -> Dict:
    """Converts ase.Atoms into dictionary

    Args:
        ase_atoms_object (ase.Atoms): _description_
        rename (Optional[bool], optional): _description_. Defaults to True.
        key_mapping (Optional[Dict[str, str]], optional): _description_. Defaults to invert_dictionary(INPUT_KEY_MAPPING).

    Returns:
        Dict: _description_
    """
    dictionary = {key: value for key, value in npz.items()}

    # once this is done we can rename the keys based on a key_mapping
    if rename:
        dictionary = {
            key_mapping.get(key, key): value for key, value in dictionary.items()
        }

    return dictionary


def format_values_in_dictionary(
    dictionary: Dict, types_of_fields: Optional[Dict] = FIELD_TYPES
) -> Dict[str, torch.Tensor]:
    return {
        key: (
            torch.from_numpy(value).type(types_of_fields[key])
            if isinstance(value, np.ndarray)
            else (
                value.clone().detach().type(types_of_fields[key])
                if isinstance(value, torch.Tensor)
                else torch.tensor(value).type(types_of_fields[key])
            )
        )
        for key, value in dictionary.items()
    }


def mlp_config_from_dictionary(layer_config: Dict) -> Union[list[int], str]:
    # let's get the hidden neurons and activation functions from the config
    n_neurons = layer_config["n_neurons"]
    activation_string = layer_config.get("activation", "silu")

    # let us start with the neurons
    neurons = (
        n_neurons
        if isinstance(n_neurons, list)
        else layer_config.get("n_layers", 1) * [n_neurons]
    )
    # then activation functions
    if activation_string not in ACTIVATION_FUNCTIONS:
        raise KeyError(
            f"Don't know this activation function {activation_string}. Please define in utils/misc.py."
        )
    activation_function = ACTIVATION_FUNCTIONS[activation_string]

    return neurons, activation_function


def determine_irreps_for_gate(
    irreps_input: Union[Irreps, str],
    irreps_scalars: Optional[Union[Irreps, str]] = None,
    irreps_gates: Optional[Union[Irreps, str]] = None,
):
    irreps_input = Irreps(irreps_input).sort().irreps.simplify()

    # get any irreps l>0
    n_irreps_scalar = irreps_input.count("0e") + irreps_input.count("0o")
    n_irreps_non_scalar = irreps_input.num_irreps - n_irreps_scalar

    if n_irreps_scalar < n_irreps_non_scalar:
        raise ValueError("Not enough scalars to gate all the non-scalar irreps.")
    if not irreps_scalars and not irreps_gates:
        # per default, even scalars are primarirly used as gates
        # therefore we check first how many gates are created on even scalars
        # of course the maximum is the number of even scalars in the input
        # Note: If not specified otherwise by the user, we use even gates by default.
        # The reasoning behind this is that even gates merely act as a scaling factor
        # on the higher order tensors without changing their parity (this is what odd gates do).
        # To minimise potential sources of errors for the user by accidentally changing the parity
        # of their desired higher order tensors, we proceed with even gates by default.
        n_even_gates = min(irreps_input.count("0e"), n_irreps_non_scalar)
        # if we have even scalars left we can take them as scalars
        n_even_scalars = irreps_input.count("0e") - n_even_gates

        # based on these numbers we can compute the odd scalars for gates and scalars
        n_odd_gates = n_irreps_non_scalar - n_even_gates
        # check if we have sufficient scalars
        if (irreps_input.count("0o") - n_odd_gates) < 0:
            raise ValueError("Not enough scalars to gate all the non-scalar irreps.")
        n_odd_scalars = irreps_input.count("0o") - n_odd_gates

        # now we can build the scalars
        irreps_scalars = (
            Irreps(f"{n_odd_scalars}x0o+{n_even_scalars}x0e")
        ).remove_zero_multiplicities()

        # ... and gates
        irreps_gates = Irreps(
            f"{n_odd_gates}x0o+{n_even_gates}x0e"
        ).remove_zero_multiplicities()

    # now get the remaining scalars to be irreps_scalar
    if irreps_scalars and not irreps_gates:
        irreps_scalars = Irreps(irreps_scalars).sort().irreps.simplify()

        # first let's check whether the given scalars are a subset of the input
        if not check_irreps_is_subset(irreps_scalars, irreps_input):
            raise ValueError(
                "Given scalars are not even contained in the input irreps."
            )

        if n_irreps_non_scalar - (n_irreps_scalar - irreps_scalars.num_irreps) != 0:
            raise ValueError(
                "If you use that many scalars you won't have enough or too many left for gating."
            )

        irreps_gates = (
            Irreps(
                f"{irreps_input.count(Irrep(0,-1))-irreps_scalars.count(Irrep(0,-1))}x0o"
            )
            + Irreps(
                f"{irreps_input.count(Irrep(0,1))-irreps_scalars.count(Irrep(0,1))}x0e"
            )
        ).remove_zero_multiplicities()

    if not irreps_scalars and irreps_gates:
        irreps_gates = Irreps(irreps_gates).sort().irreps.simplify()
        # first let's check whether the given gates are a subset of the input
        if not check_irreps_is_subset(irreps_gates, irreps_input):
            raise ValueError("Given gates are not even contained in the input irreps.")

        # next check that the number of gates correspond to the number of non-scalars
        if not irreps_gates.num_irreps == n_irreps_non_scalar:
            raise ValueError(
                "Number of gates does not correspond to non-scalar irreps."
            )
        irreps_scalars = (
            Irreps(
                f"{irreps_input.count(Irrep(0,-1))-irreps_gates.count(Irrep(0,-1))}x0o"
            )
            + Irreps(
                f"{irreps_input.count(Irrep(0,1))-irreps_gates.count(Irrep(0,1))}x0e"
            )
        ).remove_zero_multiplicities()

    # assemble the irreps
    irreps_gated = (
        Irreps([(mul, (ir.l, ir.p)) for (mul, ir) in irreps_input if ir.l > 0])
        .sort()
        .irreps.simplify()
    )

    # do final sanity check (mainly in case everything was given by the user)
    if (
        irreps_scalars + irreps_gates + irreps_gated
    ).sort().irreps.simplify() != irreps_input:
        raise ValueError(
            "The scalars and gates provided do not match up with the input irreps."
        )

    return irreps_scalars, irreps_gates, irreps_gated


def check_irreps_is_subset(irreps_in: Irreps, irreps_all: Irreps) -> bool:
    """Compares two irreps and checks whether irreps_in is a subset of irreps_all.
    In other words, this function checks whether irreps_in are contained in irreps_all.


    Args:
        irreps_in (Irreps): Irreps which are checked to be a subset of irreps_all.
        irreps_all (Irreps): Irreps which are checked to contain irreps_in.

    Returns:
        bool : Returns whether irreps_in is a subset of irreps_all.
    """
    # let's start by sorting and simplifying both irreps
    ir_in_sorted = irreps_in.sort().irreps.simplify()
    ir_all_sorted = irreps_all.sort().irreps.simplify()

    # now we can loop over all irreps in irreps in and check if the same or a higher multiplicty exist in the reference
    check_array = [
        True if i.mul <= ir_all_sorted.count(i.ir) else False for i in ir_in_sorted
    ]

    return all(check_array)


def get_activation_functions_from_list(
    n_functions_needed: int,
    list_act: List[str],
):
    if len(list_act) > n_functions_needed:
        raise ValueError("List contains more functions then needed, how should I pick?")
    list_strings = (
        list_act * n_functions_needed
        if len(list_act) < n_functions_needed
        else list_act
    )

    acts = [ACTIVATION_FUNCTIONS[act] for act in list_strings]

    return acts


def get_activation_functions_from_dict(
    scalar_irreps: Irreps,
    dict_act: Dict[str, str],
) -> List:
    """Returns activation function based on dictionary. This allows for extra flexibility in the code and
    is particularly helpful when we do not know whether both odd and even scalars are created within a MP Layer.

    Args:
        scalar_irreps (Irreps): Irreps of the scalars to be gated or passed.
        dict_act (Dict[str, str]): Dictionary stating how to deal with odd ('o') and even ('even') scalars.

    Raises:
        KeyError: Raised if dict_act has other keys than 'o' and 'e'.
        ValueError: Raised if scalar_irreps are not scalar but higher order.
        ValueError: Raised if len(scalar_irreps) > 2. Then simplify first.
        KeyError: Raised if not all parities in irreps are found in dict_act.

    Returns:
        list: List of activation functions. Analogoues to function get_activation_functions_from_list
    """

    if not set(dict_act.keys()).issubset(set(["o", "e"])):
        raise KeyError("Only odd (key 'o') and even (key 'e') parity allowed.")

    # if empty: return
    if not scalar_irreps:
        return []

    if scalar_irreps.lmax > 0:
        raise ValueError("Scalar irreps are not scalar only.")

    if len(scalar_irreps) > 2:
        raise ValueError("Simplify irreps first.")

    # see which parity is in the l=0 irreps
    parities = [{-1: "o", 1: "e"}.get(ir.p) for _, ir in scalar_irreps]

    # raise error if more parieties in irreps then in dictionary
    if not set(parities).issubset(dict_act.keys()):
        raise KeyError("Activation functions not for both parities given in dict_act.")

    # now let's get the activation functions
    acts = [ACTIVATION_FUNCTIONS[dict_act.get(p)] for p in parities]
    return acts


def build_mlp_for_tp_weights(
    input_field: str, irreps_in: Irreps, mlp_kwargs: Dict, output_dim: int
) -> FullyConnectedNet:
    # extract information from kwargs
    neurons_per_layer, activation = (
        mlp_config_from_dictionary(mlp_kwargs)
        if mlp_kwargs
        else ([64, 64, 64], torch.nn.SiLU())
    )

    # build mlp
    n_input_neurons = irreps_in[input_field][0].mul

    # add potentially the raise of an error if more than scalars
    return FullyConnectedNet(
        [n_input_neurons] + neurons_per_layer + [output_dim],
        activation,
    )


def convert_irreps_to_string(d: Dict):
    for key, value in d.items():
        if isinstance(value, dict):
            if "irreps" in key:
                for inner_key, inner_value in value.items():
                    value[inner_key] = str(inner_value)
            else:
                convert_irreps_to_string(value)


def get_least_common_multiple(
    a: Union[int, float], b: Union[int, float], precision: Optional[float] = 1e-10
) -> float:
    """Given two real numbers, compute their least common multiple.

    Args:
        a (Union[int, float]): Real number 1.
        b (Union[int, float]): Real number 2.
        precision (Optional[float], optional): To go from float to int. Defaults to 1e-10.

    Returns:
        _type_: _description_
    """
    int_a = int(a / precision)
    int_b = int(b / precision)
    return (a * b) / (math.gcd(int_a, int_b) * precision)


def convert_units(origin: str, target: str) -> float:
    """Compute version factor between original and desired units.

    Args:
        origin (str): Original unit.
        target (str): Desired unit.

    Returns:
        float: Conversion factor.

    """

    # as a first step we map the original unit to its SI unit
    # e.g. any distance unit will be mapped to meters, any time unit to seconds, etc.
    # to do this, we first need to check if we know the SI unit of the original unit
    if origin not in invert_dictionary(SI_UNIT_DICT).keys():
        raise KeyError(
            f"Unit {origin} not known, don't know whether this is a distance or time or whatever."
        )

    # same for the target unit
    if target not in invert_dictionary(SI_UNIT_DICT).keys():
        raise KeyError(
            f"Unit {target} not known, don't know whether this is a distance or time or whatever."
        )

    # now we have established we know the mapping to both the origin and the target
    # we can compute the conversion factor and return it
    # first we invert our factor dictionary
    conv_factors = invert_dictionary(CONV_FACTORS_TO_SI_UNIT)

    return conv_factors[origin] / conv_factors[target]
