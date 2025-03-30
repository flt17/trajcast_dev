import unittest
from typing import Dict, Optional, Union

import ase
import ase.build
import torch
from e3nn.o3 import Irreps
from torch.nn.functional import cosine_similarity

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    SPHERICAL_HARMONIC_KEY,
    TIMESTEP_ENCODING_KEY,
    TIMESTEP_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._encoding import (
    EdgeLengthEncoding,
    ElementBasedNormEncoding,
    OneHotAtomTypeEncoding,
    SphericalHarmonicProjection,
    TensorNormEncoding,
    TimestepEncoding,
)
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)


def CH3SCH3(type_mapper: Optional[Dict] = {}) -> Union[ase.Atoms, AtomicGraph]:
    """Create ase.Atoms object for CH3SCH3 molecule in vacuum.
    Returns:
        ase.Atoms: _description_
    """
    ase_atoms = ase.build.molecule("CH3SCH3")
    ase_atoms.arrays[VELOCITIES_KEY] = torch.randn(len(ase_atoms), 3)
    graph = AtomicGraph.from_atoms_dict(
        atoms_dict=format_values_in_dictionary(
            convert_ase_atoms_to_dictionary(ase_atoms)
        ),
        r_cut=3.0,
        atom_type_mapper=type_mapper,
    )
    return ase_atoms, graph


class TestOneHotAtomTypeEncoding(unittest.TestCase):
    def test_returns_correctly_encoded_vector(self):
        _, graph = HCN()
        one_hot = OneHotAtomTypeEncoding(number_of_species=3)
        updated_graph = one_hot(graph)

        self.assertTrue(updated_graph[ATOM_TYPE_EMBEDDING_KEY].size()[-1] == 3)
        self.assertTrue(one_hot.irreps_out[ATOM_TYPE_EMBEDDING_KEY] == Irreps("3x0e"))


class TestEdgeLengthEncoding(unittest.TestCase):
    def test_returns_correctly_encoded_edge_vector_with_20_gaussians(self):
        _, graph = HCN()
        basis_kwargs = {"rmax": graph.cutoff_radius, "basis_size": 20}
        edge_encoding = EdgeLengthEncoding(basis_kwargs=basis_kwargs)
        graph.compute_edge_vectors()
        updated_graph = edge_encoding(graph)
        num_edges = graph.num_edges
        output_field = edge_encoding.output_field

        self.assertTrue(updated_graph[output_field].size()[0] == num_edges)

        self.assertTrue(updated_graph[output_field].size()[1] == 20)
        self.assertTrue(
            edge_encoding.irreps_out[output_field] == Irreps([(20, (0, 1))])
        )


class TestTensorNormEncoding(unittest.TestCase):
    def test_returns_init_works_correctly(self):
        encoding = TensorNormEncoding(
            input_field=EDGE_VECTORS_KEY,
            basis_kwargs={"rmax": 5.0, "basis_size": 20, "basis_function": "gaussian"},
        )
        self.assertIsInstance(encoding, TensorNormEncoding)

    def test_returns_forward_can_reproduce_edge_length_encoding(self):
        _, graph = HCN()
        basis_kwargs = {"rmax": graph.cutoff_radius, "basis_size": 20}

        # create the embedding with EdgeLengthEncoding
        ref_layer = EdgeLengthEncoding(
            output_field="output_ref", basis_kwargs=basis_kwargs
        )
        graph.compute_edge_vectors()
        graph = ref_layer(graph)
        output_ref = graph["output_ref"]

        # now with TensorNormEncoding
        tn_encoder = TensorNormEncoding(
            input_field=EDGE_VECTORS_KEY,
            basis_kwargs={"rmax": graph.cutoff_radius, "basis_size": 20},
        )
        graph = tn_encoder(graph)
        output_tn = graph[tn_encoder.output_field]

        self.assertTrue(torch.all(output_tn == output_ref))

    def test_returns_forward_truncates_values_larger_rmax_for_most_functions_except_gaussian(
        self,
    ):
        _, graph = HCN()
        graph.compute_edge_vectors()

        bases = ["cosine", "smooth_finite", "fourier", "bessel"]

        for base in bases:
            basis_kwargs = {"rmax": 2.24, "basis_size": 5, "basis_function": base}

            # now with TensorNormEncoding
            tn_encoder = TensorNormEncoding(
                input_field=EDGE_VECTORS_KEY, basis_kwargs=basis_kwargs
            )
            graph = tn_encoder(graph)
            output_tn = graph[tn_encoder.output_field]

            # last value should be zero for all these
            self.assertTrue(torch.all(output_tn[-1] == 0))

        # check for Gaussian
        basis_kwargs = {"rmax": 2.24, "basis_size": 5, "basis_function": "gaussian"}

        # now with TensorNormEncoding
        tn_encoder = TensorNormEncoding(
            input_field=EDGE_VECTORS_KEY, basis_kwargs=basis_kwargs
        )
        graph = tn_encoder(graph)
        output_tn = graph[tn_encoder.output_field]

        # last value should be zero for all these
        self.assertFalse(torch.all(output_tn[-1] == 0))


class TestElementBasedNormEncoding(unittest.TestCase):
    def test_returns_init_works_correctly(self):
        encoding = ElementBasedNormEncoding(
            input_field=EDGE_VECTORS_KEY,
            atom_type_embedding_field=ATOM_TYPE_EMBEDDING_KEY,
            basis_kwargs={"rmax": 5.0, "basis_size": 20, "basis_function": "gaussian"},
            irreps_in={ATOM_TYPE_EMBEDDING_KEY: Irreps("3x0e")},
        )
        self.assertIsInstance(encoding, ElementBasedNormEncoding)
        self.assertTrue(
            encoding.element_tp.irreps_out
            == encoding.irreps_out["norm_embedding_edge_vectors"]
        )

    def test_returns_forward_works_with_OneHotEncoding(self):
        _, graph = HCN()
        graph[VELOCITIES_KEY] = torch.randn(3, 3)
        onehot = OneHotAtomTypeEncoding(number_of_species=3)
        encoding = ElementBasedNormEncoding(
            input_field=VELOCITIES_KEY,
            atom_type_embedding_field=ATOM_TYPE_EMBEDDING_KEY,
            basis_kwargs={"rmax": 5.0, "basis_size": 10, "basis_function": "gaussian"},
            irreps_in=onehot.irreps_out,
        )
        graph = onehot(graph)
        graph = encoding(graph)
        self.assertTrue(graph[encoding.output_field].size() == torch.Size([3, 10]))


class TestSphericalHarmonicProjection(unittest.TestCase):
    def test_returns_correctly_projected_vector_edge_vector_with_l2(self):
        _, graph = HCN()
        graph.compute_edge_vectors()
        max_rotation_order = 2
        sh_encoding = SphericalHarmonicProjection(
            max_rotation_order=max_rotation_order,
        )
        updated_graph = sh_encoding(graph)
        self.assertTrue(
            updated_graph[f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}"].size()[-1]
            == (max_rotation_order + 1) ** 2
        )


def HCN() -> Union[ase.Atoms, AtomicGraph]:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    ase_atoms = ase.build.molecule("HCN")
    graph = AtomicGraph.from_atoms_dict(
        atoms_dict=format_values_in_dictionary(
            convert_ase_atoms_to_dictionary(ase_atoms)
        ),
        r_cut=5.0,
    )

    return ase_atoms, graph


class TestTimestepEncoding(unittest.TestCase):
    def test_module_initialises_correct(self):
        module = TimestepEncoding(embedding_dimension=16, max_timestep=5.0)

        self.assertTrue(isinstance(module, TimestepEncoding))
        self.assertTrue(module.max_timestep.dtype == torch.get_default_dtype())
        self.assertTrue(module.irreps_out[TIMESTEP_ENCODING_KEY] == Irreps("16x0e"))

    def test_forward_works_correctly(self):
        module = TimestepEncoding(embedding_dimension=16, max_timestep=5.0)

        _, graph = CH3SCH3()
        graph[TIMESTEP_KEY] = 1.0

        graph = module(graph)

        self.assertTrue(graph[TIMESTEP_ENCODING_KEY].shape == torch.Size([1, 16]))

    def test_forward_produces_distinct_encodings(self):
        module = TimestepEncoding(embedding_dimension=16, max_timestep=5.0)

        _, graph = CH3SCH3()

        dts = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        encodings = torch.zeros(len(dts), 16)
        for i, dt in enumerate(dts):
            graph[TIMESTEP_KEY] = dt

            encodings[i] = module(graph)[TIMESTEP_ENCODING_KEY].squeeze()

        # compute cosine similarities
        cs = torch.zeros(len(dts), len(dts))
        for i in range(len(dts)):
            for j in range(len(dts)):
                cs[i, j] = cosine_similarity(encodings[i], encodings[j], dim=0)

        # check that the cosine similarity decreases going from small to large timesteps
        self.assertTrue(torch.all(cs[0] == torch.sort(cs[0], descending=True)[0]))


if __name__ == "__main__":
    unittest.main()
