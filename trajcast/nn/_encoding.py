from typing import Any, Dict, Optional

import torch
import torch.nn.functional
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.util.jit import compile_mode
from torch_geometric.data import Batch

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    ATOM_TYPES_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    SPHERICAL_HARMONIC_KEY,
    TIMESTEP_ENCODING_KEY,
    TIMESTEP_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._cutoff_functions import PolynomialCutoff
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._radial_basis import BesselBasisTrainable, FixedBasis
from trajcast.nn._wrapper_ops import CuEquivarianceConfig, FullyConnectedTensorProduct


@compile_mode("script")
class OneHotAtomTypeEncoding(GraphModuleIrreps, torch.nn.Module):
    """One hot encoding, identical to NequiP.

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        number_of_species: int,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        output_field: Optional[str] = ATOM_TYPE_EMBEDDING_KEY,
    ):
        """_summary_

        Args:
            number_of_species (int): _description_
            irreps_in (Optional[Dict[str, Irreps]], optional): _description_. Defaults to {}.
        """
        super().__init__()
        self.number_of_species = number_of_species
        self.output_field = output_field
        # update/instantiate the irreps dictionary which will be important to further handle the data
        # the input irreps dictionary will given by the user or can be left blank
        # the output will be defined here
        irreps_out = {self.output_field: Irreps([(self.number_of_species, (0, 1))])}

        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        type_numbers = data[ATOM_TYPES_KEY].squeeze(1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.number_of_species
        )
        data[self.output_field] = one_hot.to(torch.float)

        return data


@compile_mode("script")
class EdgeLengthEncoding(GraphModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        radial_basis: Optional[str] = "FixedBasis",
        cutoff_function: Optional[str] = {},
        basis_kwargs: Optional[Dict[str, Any]] = {},
        cutoff_kwargs: Optional[Dict[str, Any]] = {},
        irreps_in: Optional[Dict[str, Irreps]] = {},
        output_field: Optional[str] = EDGE_LENGTHS_EMBEDDING_KEY,
    ):
        super().__init__()
        self.radial_basis = {
            "FixedBasis": FixedBasis,
            "BesselBasisTrainable": BesselBasisTrainable,
        }[radial_basis](**basis_kwargs)
        if cutoff_function:
            self.cutoff_function = {
                "PolynomialCutoff": PolynomialCutoff,
            }[
                cutoff_function
            ](**cutoff_kwargs)
        else:
            # if no cutoff function is wanted
            self.cutoff_function = torch.ones_like

        self.output_field = output_field
        irreps_out = {
            self.output_field: Irreps([(self.radial_basis.basis_size, (0, 1))])
        }
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        edge_lengths = data.edge_lengths.squeeze(1)

        edge_lengths_encoding = (
            self.radial_basis(edge_lengths)
            * self.cutoff_function(edge_lengths)[..., None]
        )
        data[self.output_field] = edge_lengths_encoding

        return data


@compile_mode("script")
class ElementBasedNormEncoding(GraphModuleIrreps, torch.nn.Module):
    """This module embeds the norm of a tensor in a set of basis functions which
    are then weighted in a tensor product with the one-hot encoding of the
    chemical species.The obtained embedding can then be used in an MLP to
    weight the paths in a tensor product in a message passing layer.
    """

    def __init__(
        self,
        input_field: str,
        atom_type_embedding_field: Optional[str] = ATOM_TYPE_EMBEDDING_KEY,
        basis: Optional[str] = "FixedBasis",
        cutoff_function: Optional[Any] = {},
        basis_kwargs: Optional[Dict[str, Any]] = {},
        cutoff_kwargs: Optional[Dict[str, Any]] = {},
        irreps_in: Optional[Dict[str, Irreps]] = {},
        output_field: Optional[str] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()

        self.input_field = input_field
        self.atom_type_embedding_field = atom_type_embedding_field
        # check output field and give it name
        self.output_field = (
            f"norm_embedding_{input_field}" if not output_field else output_field
        )

        # information about the basis functions
        self.basis = {
            "FixedBasis": FixedBasis,
            "BesselBasisTrainable": BesselBasisTrainable,
        }[basis](**basis_kwargs)

        # cutoff function
        if cutoff_function:
            self.cutoff_function = cutoff_function(**cutoff_kwargs)
        else:
            # if no cutoff function is wanted
            self.cutoff_function = torch.ones_like

        # initialise irreps
        irreps_out = {self.output_field: Irreps([(self.basis.basis_size, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        # define the tensor product
        self.element_tp = FullyConnectedTensorProduct(
            self.irreps_out[self.output_field],
            self.irreps_in[self.atom_type_embedding_field],
            self.irreps_out[self.output_field],
            shared_weights=True,
            internal_weights=True,
            cueq_config=cueq_config,
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # first compute the norm of the tensor
        norm_tensor = torch.norm(data[self.input_field], dim=1)

        # now encode this norm in basis functions
        norm_encoding = (
            self.basis(norm_tensor) * self.cutoff_function(norm_tensor)[..., None]
        ).to(torch.get_default_dtype())

        # perform tensorproduct
        element_based_norm_encoding = self.element_tp(
            norm_encoding, data[self.atom_type_embedding_field]
        )

        data[self.output_field] = element_based_norm_encoding
        return data


class TensorNormEncoding(GraphModuleIrreps, torch.nn.Module):
    """This module embeds the norm of any tensor in a set of basis functions.
    The obtained embedding can then be used in an MLP to weight the paths
    in a tensor product in a message passing layer.
    Note: This is the general case of the edge length encoding.
    However, here computing the norm of the tensor is part of the module.
    """

    def __init__(
        self,
        input_field: str,
        basis: Optional[str] = "FixedBasis",
        cutoff_function: Optional[Any] = {},
        basis_kwargs: Optional[Dict[str, Any]] = {},
        cutoff_kwargs: Optional[Dict[str, Any]] = {},
        irreps_in: Optional[Dict[str, Irreps]] = {},
        output_field: Optional[str] = None,
    ):
        super().__init__()

        self.input_field = input_field
        # check output field and give it name
        self.output_field = (
            f"norm_embedding_{input_field}" if not output_field else output_field
        )

        # information about the basis functions
        self.basis = {
            "FixedBasis": FixedBasis,
            "BesselBasisTrainable": BesselBasisTrainable,
        }[basis](**basis_kwargs)

        # cutoff function
        if cutoff_function:
            self.cutoff_function = cutoff_function(**cutoff_kwargs)
        else:
            # if no cutoff function is wanted
            self.cutoff_function = torch.ones_like

        # initialise irreps
        irreps_out = {self.output_field: Irreps([(self.basis.basis_size, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # first compute the norm of the tensor
        norm_tensor = torch.norm(data[self.input_field], dim=1)

        # now encode this norm in basis functions
        norm_encoding = (
            self.basis(norm_tensor) * self.cutoff_function(norm_tensor)[..., None]
        )
        data[self.output_field] = norm_encoding
        return data


@compile_mode("script")
class SphericalHarmonicProjection(GraphModuleIrreps, torch.nn.Module):
    """This class produces the spherical harmoinc projection of a tensor.

    Args:
        GraphModuleIrreps (_type_): _description_
        torch (_type_): _description_
    """

    def __init__(
        self,
        max_rotation_order: int,
        project_on_unit_sphere: Optional[bool] = True,
        normalization_projection: Optional[str] = "component",
        irreps_in: Optional[Dict[str, Irreps]] = {},
        input_field: Optional[str] = EDGE_VECTORS_KEY,
        output_field: Optional[str] = None,
    ):
        """_summary_

        Args:
            max_rotation_order (int): Corresponds to l_max in e3nn and determines when to truncate the spherical harmonic expansion
            project_on_unit_sphere (Optional[bool], optional): Whether to project all input tensors onto the unit sphere (radius==1). Required for our case but can be changed.
            normalization_projection (Optional[str], optional): Corresponds to the "normalization argument in e3nn. Here, we highly recommend the component normalisation (||Y^l (x)|| = 2l +1, for x in S^2) as recommend by the original e3nn paper.
                The reasons for this are: "(i) all the preactivation have mean 0 and variance 1 (ii) all the post activations have the second moment 1 (iii) all the layers learn when the width (i.e. the multiplicities) is sent to infinity."
            irreps_in (Optional[Dict[str, Irreps]], optional): _description_. Defaults to {}.
            input_field (Optional[str], optional): The field in the graph we would like to encode with spherical harmonics. Defaults to EDGE_VECTORS_KEY as we usually do this as part of the convolution in TFNNs.
            output_field (Optional[str], optional): The field where to store the encoding. Defaults to None.
        """
        super().__init__()

        self.input_field = input_field
        # check output field and give it name
        self.output_field = (
            f"{SPHERICAL_HARMONIC_KEY}_{input_field}"
            if not output_field
            else output_field
        )

        irreps_out = {
            self.output_field: Irreps.spherical_harmonics(lmax=max_rotation_order)
        }
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        self.project_on_unit_sphere = project_on_unit_sphere
        self.normalization_projection = normalization_projection

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        sh_encoding = spherical_harmonics(
            l=self.irreps_out[self.output_field],
            x=data[self.input_field],
            normalize=self.project_on_unit_sphere,
            normalization=self.normalization_projection,
        )
        data[self.output_field] = sh_encoding
        return data


class TimestepEncoding(GraphModuleIrreps, torch.nn.Module):
    """Create embedding of the timestep similar to position encoding in Transformer.

    Note, compared to the TemporalEncoding this module uses the timestep as continuous float
    rather than mapping it to an int a priori and fixing the available timesteps.
    """

    def __init__(
        self,
        embedding_dimension: int,
        max_timestep: float,
        input_field: Optional[str] = TIMESTEP_KEY,
        output_field: Optional[str] = TIMESTEP_ENCODING_KEY,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        device: Optional[torch.device] = None,
    ):
        """Initialise the timestep encoding module.

        Args:
            embedding_dimension (int): Desired dimension of the encoding.
            max_timestep (float): Maximum timestep in fs. This is important for scaling the encoding vectors.
            input_field (Optional[str], optional): Defaults to TIMESTEP_KEY.
            output_field (Optional[str], optional): Defaults to TIMESTEP_ENCODING_KEY.
            irreps_in (Optional[Dict[str, Irreps]], optional): Irreps entering this module. Defaults to {}.
            device (Optional[torch.device], optional): Device the module is run on. Defaults to None.
        """
        super().__init__()
        self.input_field = input_field
        self.output_field = output_field
        self.embedding_dimension = embedding_dimension
        self.register_buffer("max_timestep", torch.tensor([max_timestep]))

        # Please Double Check Irreps Out
        irreps_out = {self.output_field: Irreps([(embedding_dimension, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        self.to(device)

    def forward(self, data: AtomicGraph):
        timestep = data[self.input_field]

        if isinstance(data, Batch):
            batch_size = data.num_graphs

        else:
            batch_size = 1

        embedding = torch.zeros(
            self.embedding_dimension, batch_size, device=self.device
        )
        for i in range(0, self.embedding_dimension, 2):
            embedding[i] = torch.sin(
                timestep / (self.max_timestep ** (i / self.embedding_dimension))
            )
            if i + 1 < self.embedding_dimension:
                embedding[i + 1] = torch.cos(
                    timestep
                    / (self.max_timestep ** ((i + 1) / self.embedding_dimension))
                )

        data[self.output_field] = embedding.T

        return data
