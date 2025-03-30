"""
This file is taken from the original MACE code distributed under the MIT License.
The link to the exact file can be found here: https://github.com/ACEsuit/mace/blob/main/mace/modules/wrapper_ops.py

Own contributions include the Gate and wrapper around it.

Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import itertools
import types
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

if CUET_AVAILABLE:

    class O3_e3nn(cue.O3):
        def __mul__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l_rot in itertools.count(0):
                yield O3_e3nn(l=l_rot, p=1 * (-1) ** l_rot)
                yield O3_e3nn(l=l_rot, p=-1 * (-1) ** l_rot)

    @compile_mode("script")
    class CueqGate(torch.nn.Module):
        def __init__(
            self,
            irreps_scalars: o3.Irreps,
            act_scalars: List,
            irreps_gates: o3.Irreps,
            act_gates: List,
            irreps_gated: o3.Irreps,
            group: Union[cue.O3, O3_e3nn] = O3_e3nn,
            layout: cue.IrrepsLayout = None,
            layout_in: cue.IrrepsLayout = None,
            layout_out: cue.IrrepsLayout = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            math_dtype: Optional[torch.dtype] = None,
            optimize_fallback: Optional[bool] = None,
        ):
            super().__init__()
            # we start with o3 irreps
            # so we can use same activation and sortcut as the e3n.gate
            irreps_scalars = o3.Irreps(irreps_scalars)
            irreps_gates = o3.Irreps(irreps_gates)
            irreps_gated = o3.Irreps(irreps_gated)

            if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
                raise ValueError(
                    f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
                )
            if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
                raise ValueError(
                    f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
                )
            if irreps_gates.num_irreps != irreps_gated.num_irreps:
                raise ValueError(
                    f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number "
                    f"({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
                )

            # define the layers from e3nn which are merely operating wihtin each block of irreps
            self.sc = nn._gate._Sortcut(irreps_scalars, irreps_gates, irreps_gated)
            self._irreps_in = cue.Irreps(group, self.sc.irreps_in)
            irreps_scalars_e3nn, irreps_gates_e3nn, irreps_gated_e3nn = (
                self.sc.irreps_outs
            )

            self.act_scalars = nn.Activation(irreps_scalars_e3nn, act_scalars)
            self.act_gates = nn.Activation(irreps_gates_e3nn, act_gates)

            # now transfer to cue
            self.irreps_scalars = cue.Irreps(group, self.act_scalars.irreps_out)
            self.irreps_gates = cue.Irreps(group, self.act_gates.irreps_out)
            self.irreps_gated = cue.Irreps(group, irreps_gated_e3nn)

            # now comes the cueq part -> replace elementwise tensor product
            math_dtype = math_dtype or dtype
            e = cue.descriptors.elementwise_tensor_product(
                self.irreps_gated, self.irreps_gates
            )

            self.mul = cuet.EquivariantTensorProduct(
                e,
                layout=layout,
                layout_in=(layout_in, cue.ir_mul),
                layout_out=layout_out,
                device=device,
                math_dtype=math_dtype,
                optimize_fallback=optimize_fallback,
            )

            self._irreps_out = self.irreps_scalars + self.irreps_gated

        def forward(
            self,
            features: torch.Tensor,
            use_fallback: Optional[bool] = None,
        ) -> torch.Tensor:
            # this is independent of e3nn or cueq, just splits the tensor into irrep blocks
            scalars, gates, gated = self.sc(features)

            # scalars are layout insensitive
            scalars = self.act_scalars(scalars)

            if gates.shape[-1]:
                # insensitive to layout
                gates = self.act_gates(gates)
                gated = self.mul(gated, gates, use_fallback=use_fallback)
                features = torch.cat([scalars, gated], dim=-1)
            else:
                features = scalars
            return features

        @property
        def irreps_in(self):
            """Input representations."""
            return self._irreps_in

        @property
        def irreps_out(self):
            """Output representations."""
            return self._irreps_out

else:
    print(
        "cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled."
    )


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "ir_mul"  # One of: mul_ir, ir_mul
    layout_str: str = "ir_mul"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_fctp: bool = False

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
            )


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            instance = cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                optimize_fallback=True,
            )

            instance.original_forward = instance.forward

            def cuet_forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.original_forward(x, use_fallback=True)

            instance.forward = types.MethodType(cuet_forward, instance)
            return instance

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            biases=False,
            internal_weights=internal_weights,
        )


class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            instance = cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                layout_in1=cueq_config.layout,
                layout_in2=cueq_config.layout,
                layout_out=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )
            instance.original_forward = instance.forward

            def cuet_forward(
                self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
            ) -> torch.Tensor:
                return self.original_forward(x, y, z, use_fallback=None)

            instance.forward = types.MethodType(cuet_forward, instance)
            return instance

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            instance = cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                optimize_fallback=True,
            )
            instance.original_forward = instance.forward

            def cuet_forward(
                self, x: torch.Tensor, attrs: torch.Tensor
            ) -> torch.Tensor:
                return self.original_forward(x, attrs, use_fallback=True)

            instance.forward = types.MethodType(cuet_forward, instance)
            return instance

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class Gate:
    """Wrapper around nn.Gate/CueqGate"""

    def __new__(
        cls,
        irreps_scalars: o3.Irreps,
        act_scalars: List,
        irreps_gates: o3.Irreps,
        act_gates: List,
        irreps_gated: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            instance = CueqGate(
                irreps_scalars,
                act_scalars,
                irreps_gates,
                act_gates,
                irreps_gated,
                layout=cueq_config.layout,
                optimize_fallback=True,
            )

            instance.original_forward = instance.forward

            def cuet_forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.original_forward(x, use_fallback=True)

            instance.forward = types.MethodType(cuet_forward, instance)
            return instance

        else:
            return nn.Gate(
                irreps_scalars,
                act_scalars,
                irreps_gates,
                act_gates,
                irreps_gated,
            )
