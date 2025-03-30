"""
Collection of Radial basis functions. Use this to call e3nn.math.soft_one_hot_linspace or bessel function as implemented in NequiP with trainable parameters.
Later: Potentially also other radial functions with learnable weights.

Authors: Fabian Thiemann
"""

from typing import Optional

import torch
import torch.nn
import math

from e3nn.math import soft_one_hot_linspace


class FixedBasis(torch.nn.Module):
    """Radial basis without adjustable/trainable weights. This merely calls the basis functions via e3nn.
    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        rmax: float,
        rmin: Optional[float] = 0,
        basis_function: Optional[str] = "gaussian",
        basis_size: Optional[int] = 10,
        normalization: Optional[bool] = True,
    ):
        super().__init__()
        self.basis_function = basis_function
        self.basis_size = basis_size
        self.rmin = rmin
        self.rmax = rmax

        self.norm_const = 1 if not normalization else basis_size**0.5

    def forward(self, radial_distance: torch.Tensor) -> torch.Tensor:
        return (
            soft_one_hot_linspace(
                radial_distance,
                start=self.rmin,
                end=self.rmax,
                number=self.basis_size,
                basis=self.basis_function,
                cutoff=True,
            )
            * self.norm_const
        )


class BesselBasisTrainable(torch.nn.Module):
    """This is the basis function used in NequiP:
    Note, compared to the Bessel implementation in e3nn, this function does not take the square root of the prefactor

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        rmax: float,
        basis_size: Optional[int] = 8,
        epsilon: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.basis_size = basis_size
        self.rmax = rmax
        self.epsilon = epsilon

        # define bessel roots
        bessel_roots = torch.arange(1, basis_size + 1) * math.pi
        self.bessel_roots = torch.nn.Parameter(bessel_roots)

        # define prefactor
        self.prefactor = 2.0 / self.rmax

    def forward(self, radial_distances: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(
            self.bessel_roots * radial_distances.unsqueeze(-1) / self.rmax
        )
        return self.prefactor * (
            numerator / (radial_distances.unsqueeze(-1) + self.epsilon)
        )
