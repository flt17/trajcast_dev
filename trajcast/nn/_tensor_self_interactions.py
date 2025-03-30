"""
Collection of modules used to work on the node features itself
without interaction with other notes. In literature these are, therefore,
often referred to as "self-interaction" or "atom-wise layers". Here, they
can be just a linear layer mixing features of identical irrep or some
other operation acting exclusively on the current state of the respective node.


Authors: Fabian Thiemann
"""

from typing import Dict, Optional, Union

import torch
from e3nn.o3 import Irreps

from trajcast.data._keys import NODE_FEATURES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._wrapper_ops import CuEquivarianceConfig, Linear


class LinearTensorMixer(GraphModuleIrreps, torch.nn.Module):
    """This module is used to linearly combine the tensors of identical irreps.
    This is very handy for combining/reducing the feature dimensions.

    Args:
        GraphModuleIrreps (_type_): _description_
        torch (_type_): _description_
    """

    def __init__(
        self,
        input_field: Optional[str] = NODE_FEATURES_KEY,
        output_field: Optional[str] = None,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        irreps_out: Optional[Union[str, list, Irreps]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        """Initialises the module.

        Args:
            input_field (Optional[str], optional): Input field for the linear layer. Defaults to NODE_FEATURES_KEY.
            output_field (Optional[str], optional): Output field where the result of the linear layer is saved. Defaults to None.
            irreps_in (Optional[Dict[str, Irreps]], optional): Dictionary with irreps before linear layer is applied. Defaults to {}.
            irreps_out (Optional[Union[str, Irreps]], optional): Dictionary with irreps after linear layer is applied. Defaults to None.
        Raises:
            KeyError: If the input_field is not found in the irreps_in dictionary.
        """
        super().__init__()
        if input_field not in irreps_in.keys():
            raise KeyError(f"Irreps for input_field '{input_field}' not given")

        self.input_field = input_field
        self.output_field = input_field if not output_field else output_field

        irreps_out = irreps_in[input_field] if not irreps_out else irreps_out

        if isinstance(irreps_out, str) or isinstance(irreps_out, list):
            irreps_out = Irreps(irreps_out)
        irreps_out = {self.output_field: irreps_out.sort().irreps.simplify()}

        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        self.linear = Linear(
            irreps_in=self.irreps_in[self.input_field],
            irreps_out=self.irreps_out[self.output_field],
            cueq_config=cueq_config,
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data[self.output_field] = self.linear(
            data[self.input_field].to(torch.get_default_dtype())
        )
        return data
