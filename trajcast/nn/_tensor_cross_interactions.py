"""
Collection of individual modules to be used for tensors to interact with other tensors.

Authors: Fabian Thiemann
"""

from typing import Dict, List, Optional, Union

import torch
from e3nn.o3 import Irreps

from trajcast.data._keys import ADDITION_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._wrapper_ops import CuEquivarianceConfig, TensorProduct


class DepthwiseTensorProduct(torch.nn.Module):
    def __init__(
        self,
        max_rotation_order: int,
        irreps_input1: Union[str, list, Irreps],
        irreps_input2: Union[str, list, Irreps],
        trainable: Optional[bool] = True,
        multiplicity_mode: Optional[str] = "uvuv",
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.l_max = max_rotation_order
        self.irreps_input1 = (
            irreps_input1 if not isinstance(irreps_input1, Irreps) else irreps_input1
        )
        self.irreps_input2 = (
            irreps_input2 if not isinstance(irreps_input1, Irreps) else irreps_input2
        )
        if multiplicity_mode not in ["uvu", "uvuv", "uvv"]:
            raise NotImplementedError
        self.mode = multiplicity_mode
        self.trainable = trainable

        # get the irreps dimensoins of the output based on max_rotation order and
        # the mode on how multiplicities are treated
        self.irreps_out, instructions = self._get_irreps_out_and_instructions()

        # define now corresponding tensor product
        self.tp = TensorProduct(
            self.irreps_input1,
            self.irreps_input2,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
            cueq_config=cueq_config,
        )

        self.weight_numel = self.tp.weight_numel

    def forward(
        self,
        input1: torch.tensor,
        input2: torch.tensor,
        weights: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        return self.tp(input1, input2, weights)

    # This code has been taken from NequIP published under MIT license:
    # https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
    def _get_irreps_out_and_instructions(self):
        """This computes the output irreps based on the allowed rotation order and on how multiplicities are handled.
        For "uvuv" this returns the FullTensorProduct from e3nn.o3.FullTensorProduct. With the exception of this subtlety
        this code is taken from NequiP nn._interaction_block, can be updated later.
        """

        irreps_out = []
        instructions = []
        for i, (mul1, ir_in1) in enumerate(self.irreps_input1):
            for j, (mul2, ir_in2) in enumerate(self.irreps_input2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out.l <= self.l_max:
                        k = len(irreps_out)
                        prefactor = (
                            mul1 * mul2
                            if self.mode == "uvuv"
                            else mul1 if self.mode == "uvu" else mul2
                        )
                        irreps_out.append((prefactor, ir_out))
                        instructions.append((i, j, k, self.mode, self.trainable))

        ###### this is a comment from Nequip: https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
        ###### Published under MIT License.
        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = Irreps(irreps_out)
        irreps_out, p, _ = irreps_out.sort()

        ###### this is a comment from Nequip: https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
        ###### Published under MIT License.
        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions


class FieldsAddition(torch.nn.Module, GraphModuleIrreps):
    def __init__(
        self,
        input_fields: Optional[List[str]] = None,
        output_field: Optional[str] = None,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        normalization: Optional[bool] = True,
    ) -> None:
        super().__init__()
        # first check that all have fields have the same irrep given
        # return error if that's not the case or if any of the fields has no irreps given
        if not all(
            irreps_in.get(field) == irreps_in[input_fields[0]] for field in input_fields
        ):
            raise KeyError("Irreps not given for all input fields.")

        # initialise input and output fields
        self.input_fields = input_fields
        self.output_field = (
            output_field
            if output_field
            else f"{ADDITION_KEY}_{'_'.join(self.input_fields)}"
        )

        # define prefactor based on normalization
        self.prefactor = 1 / len(self.input_fields) if normalization else 1

        # initialise irreps
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.output_field: irreps_in[self.input_fields[0]]},
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data[self.output_field] = (
            torch.sum(torch.stack([data[field] for field in self.input_fields]), dim=0)
            * self.prefactor
        )
        return data
