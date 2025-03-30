from typing import Dict, List, Optional, Union

import torch
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from trajcast.data._keys import NODE_FEATURES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.utils.misc import (
    determine_irreps_for_gate,
    get_activation_functions_from_dict,
    get_activation_functions_from_list,
)

from trajcast.nn._wrapper_ops import CuEquivarianceConfig, Gate


@compile_mode("script")
class GatedNonLinearity(torch.nn.Module, GraphModuleIrreps):
    """The objective of this module is to introduce non-linearity into the model.
    While scalars (l=0) can be passed directly through a non-linear function, higherd order
    tensors (l>0) need to be "gated" to obey equivariance. Gating can be seen as scaling the scaling
    higherd order vectors by a scalar which has been previously activated. For more details we refer to
    the original publication:
    M. Weiler et al., "3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data", NeurIPS 32, 2018.
    """

    def __init__(
        self,
        irreps_scalars: Optional[Irreps] = None,
        irreps_gates: Optional[Irreps] = None,
        irreps_gated: Optional[Irreps] = None,
        activation_scalars: Optional[Union[List, Dict]] = ["tanh"],
        activation_gates: Optional[Union[List, Dict]] = ["tanh"],
        input_field: Optional[str] = NODE_FEATURES_KEY,
        output_field: Optional[str] = None,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        """To initialize the gate will need some information about the irreps of our input field. Additionally,
        we can also pass information about which scalars of the input irreps are used as gates (scaling factors)
        and which are passed through a non-linear function but preserved in the output.
        Note that odd gates will change the parity of the gated higher order tensors.

        Args:
            irreps_scalars (Optional[Irreps], optional): Irreps of the scalars which will not be used for gating but preserved
                in the output function after being activated. The number of scalars must be identical to the number of
                gated higher order tensors. Only l=0 irreps can be used.  Defaults to None.
            irreps_gates (Optional[Irreps], optional): Irreps of the gates which will be used to scale and potentially
                invert the higher order tensors.  Only l=0 irreps can be used. Defaults to None.
            irreps_gated (Optional[Irreps], optional): Irreps of the higher order tensors (l>0) which will be scaled
                by the gates.  All l>0 irreps must be used. Defaults to None.
            activation_scalars (Optional[Union[List,Dict]], optional): List of activation functions given as strings to be used to activate the scalars to be passed
                through to the output layer. For odd parity only tanh can be used. If using a dictionary, keys will be 'o' and 'e' for odd and even parity,
                respectively. Defaults to ["tanh"].
            activation_gates (Optional[Union[List, Dict]], optional): List or dictionary of activation functions given as strings to be used to activate
                the gates. For odd parity only tanh can be used. If using a dictionary, keys will be 'o' and 'e' for odd and even parity, resepctively.
                Defaults to ["tanh"].
            input_field (Optional[str], optional): Name of the input field used in an AtomicGraph. Will be used to infere the irreps. Defaults to NODE_FEATURES_KEY.
            output_field (Optional[str], optional): Name of the output field we would like to save the gated features in. Defaults to None
                which will then be the same as the input field.
            irreps_in (Optional[Dict[str, Irreps]], optional): Dictionary with the irreps of all relevant fields. Defaults to {}.

        Raises:
            KeyError: Will be returned if irreps of input_field not given in irreps_in
            ValueError: Will be returned when irreps of scalars, gates, and gated to not end up to input_field irreps.
        """
        super().__init__()

        if input_field not in irreps_in:
            raise KeyError(
                f"Could not infer the irreps for the field given. Please add the irrpes for {input_field}"
            )

        self.input_field = input_field
        self.output_field = output_field if output_field else self.input_field

        # first make sure irreps are given/determined and ok
        if irreps_scalars and irreps_gates and irreps_gated:
            self.irreps_scalars = irreps_scalars.sort().irreps.simplify()
            self.irreps_gates = irreps_gates.sort().irreps.simplify()
            self.irreps_gated = irreps_gated.sort().irreps.simplify()

            if (
                self.irreps_scalars + self.irreps_gates + self.irreps_gated
            ).sort().irreps.simplify() != irreps_in[
                input_field
            ].sort().irreps.simplify():
                raise ValueError(
                    "Irreps provided do not end up to perform a correct gate."
                )
        else:
            (
                self.irreps_scalars,
                self.irreps_gates,
                self.irreps_gated,
            ) = determine_irreps_for_gate(
                irreps_input=irreps_in[input_field],
                irreps_scalars=irreps_scalars,
                irreps_gates=irreps_gates,
            )

        # make sure activation function is working:
        # scalars in case given as a list
        if isinstance(activation_scalars, list):
            act_scalars = (
                get_activation_functions_from_list(
                    len(self.irreps_scalars), activation_scalars
                )
                if len(self.irreps_scalars)
                else []
            )

        # scalars when given as dictionary
        if isinstance(activation_scalars, dict):
            act_scalars = get_activation_functions_from_dict(
                self.irreps_scalars, activation_scalars
            )

        # gates
        if isinstance(activation_gates, list):
            act_gates = get_activation_functions_from_list(
                len(self.irreps_gates), activation_gates
            )

        # gates when given as dictionary
        if isinstance(activation_gates, dict):
            act_gates = get_activation_functions_from_dict(
                self.irreps_gates, activation_gates
            )

        # now setup gate
        self.gate = Gate(
            self.irreps_scalars,
            act_scalars,
            self.irreps_gates,
            act_gates,
            self.irreps_gated,
            cueq_config=cueq_config,
        )

        # update irreps
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.output_field: self.gate.irreps_out.sort().irreps.simplify()
            },
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data[self.output_field] = self.gate(data[self.input_field])
        return data
