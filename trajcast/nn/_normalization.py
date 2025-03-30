from typing import Dict, List, Optional, Union

import torch
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from trajcast.data._keys import NODE_FEATURES_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps


@compile_mode("script")
class NormalizationLayer(GraphModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        input_fields: Optional[Union[str, List[str]]] = NODE_FEATURES_KEY,
        output_fields: Optional[Union[str, List[str]]] = NODE_FEATURES_KEY,
        means: Optional[Union[float, Dict[str, float]]] = 1.0,
        stds: Optional[Union[float, Dict[str, float]]] = 0.0,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        inverse: Optional[bool] = False,
    ):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_in)
        output_fields = input_fields if not output_fields else output_fields
        if isinstance(input_fields, List):
            if not isinstance(output_fields, List):
                raise TypeError("Input and output fields should be of the same type!")
            if not len(input_fields) == len(output_fields):
                raise IndexError("We need as many output_fields as input_fields.")

            self.input_fields = input_fields
            self.output_fields = output_fields
        else:
            if not isinstance(output_fields, str):
                raise TypeError("Input and output fields should be of the same type!")
            self.input_fields = [input_fields]
            self.output_fields = [output_fields]
        stds = stds if not isinstance(stds, float) else {self.input_fields[0]: stds}
        means = means if not isinstance(means, float) else {self.input_fields[0]: means}

        if not (
            set(means.keys()).issubset(self.input_fields)
            and set(stds.keys()).issubset(self.input_fields)
        ):
            raise KeyError(
                "Keys in means and stds should correspond to those in input_fields!"
            )
        self.means = means
        self.stds = stds
        self.inverse = inverse

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        operation = self._normalize if not self.inverse else self._unnormalize
        for count, field in enumerate(self.input_fields):
            if {field}.issubset(set(data.to_dict().keys())):

                mean = self.means[field]

                data[self.output_fields[count]] = operation(
                    tensor=data[field], mean=mean, std=self.stds[field]
                )
        return data

    def _normalize(self, tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        return (tensor - mean) / std

    def _unnormalize(
        self, tensor: torch.Tensor, mean: float, std: float
    ) -> torch.Tensor:
        return tensor * std + mean
