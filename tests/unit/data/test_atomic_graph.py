import unittest

import ase
import ase.build
import ase.build.supercells
import torch
from numpy import identity

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    ATOMIC_NUMBERS_KEY,
    EDGE_LENGTHS_KEY,
    EDGE_VECTORS_KEY,
    TOTAL_MASS_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)


def Si() -> ase.Atoms:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    lattice = torch.tensor(
        [
            [3.34939851, 0, 1.93377613],
            [1.11646617, 3.1578432, 1.93377613],
            [0, 0, 3.86755226],
        ]
    )
    coords = torch.tensor([[0, 0, 0], [1.11646617, 0.7894608, 1.93377613]])

    temp = ase.Atoms("Si2", cell=lattice, positions=coords, pbc=True)

    # make supercell
    return ase.build.make_supercell(temp, identity(3) * 2)


def H2() -> ase.Atoms:
    """Create ase.Atoms object for H2O molecule in vacuum.

    Returns:
        ase.Atoms: _description_
    """
    return ase.build.molecule("H2")


class TestAtomicGraphInit(unittest.TestCase):
    def test_returns_correct_graph_for_periodic_system_with_formated_dict(self):
        ase_atoms = Si()
        atoms_dict = convert_ase_atoms_to_dictionary(ase_atoms)
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=4.0)

        self.assertTrue(
            graph.__getattr__(ATOMIC_NUMBERS_KEY).shape
            == torch.Size([len(ase_atoms), 1])
        )
        self.assertTrue(
            graph.__getattr__(ATOMIC_MASSES_KEY).shape
            == torch.Size([len(ase_atoms), 1])
        )

        self.assertTrue(isinstance(graph.__getattr__(TOTAL_MASS_KEY), torch.Tensor))

        self.assertTrue(graph.__getattr__(TOTAL_MASS_KEY).size() == torch.Size([]))

    def test_returns_correct_graph_for_isolated_molecule_with_formated_dict(self):
        atoms_dict = convert_ase_atoms_to_dictionary(H2())
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=2.0)
        self.assertTrue(
            graph.__getattr__(ATOMIC_NUMBERS_KEY).shape == torch.Size([2, 1])
        )
        self.assertTrue(
            graph.__getattr__(ATOMIC_MASSES_KEY).shape == torch.Size([2, 1])
        )
        self.assertTrue(graph.__getattr__(TOTAL_MASS_KEY).size() == torch.Size([]))


class TestAtomicGraphComputeEdgeVectors(unittest.TestCase):
    def test_returns_edge_vectors_for_molecular_system(self):
        atoms_dict = convert_ase_atoms_to_dictionary(H2())
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=2.0)
        graph.compute_edge_vectors()

        self.assertTrue(hasattr(graph, EDGE_VECTORS_KEY))
        self.assertTrue(hasattr(graph, EDGE_LENGTHS_KEY))

        self.assertTrue(
            torch.all(
                graph[EDGE_LENGTHS_KEY]
                == torch.linalg.norm(graph[EDGE_VECTORS_KEY], dim=1).view(-1, 1)
            )
        )

    def test_returns_edge_vectors_for_periodic_system(self):
        ase_atoms = Si()
        atoms_dict = convert_ase_atoms_to_dictionary(ase_atoms)
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=4.0)

        graph.compute_edge_vectors()

        self.assertTrue(torch.all(graph[EDGE_LENGTHS_KEY] <= 4.0))

    def test_returns_edge_attributes_are_added_to_autograd_graph(self):
        atoms_dict = convert_ase_atoms_to_dictionary(H2())
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=2.0)
        graph.pos.requires_grad_(True)

        graph.compute_edge_vectors()

        self.assertTrue(graph[EDGE_LENGTHS_KEY].requires_grad)
        self.assertTrue(graph[EDGE_VECTORS_KEY].requires_grad)

    def test_returns_torch_nl_edge_indices_equivalent_to_ase(self):
        # molecule
        atoms_dict = convert_ase_atoms_to_dictionary(H2())
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph_molecule = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=2.0)
        graph_molecule.compute_edge_vectors()
        ase_edge_index = graph_molecule.edge_index
        ase_edge_vectors = graph_molecule.edge_vectors

        graph_molecule.batch = torch.zeros(graph_molecule.num_nodes, dtype=torch.long)
        graph_molecule.update_edge_index()
        graph_molecule.compute_edge_vectors()
        tnl_edge_index = graph_molecule.edge_index
        tnl_edge_vectors = graph_molecule.edge_vectors

        self.assertTrue(torch.all(tnl_edge_index == ase_edge_index))
        self.assertTrue(torch.all(tnl_edge_vectors == ase_edge_vectors))
        # material (solid)
        atoms_dict = convert_ase_atoms_to_dictionary(Si())
        atoms_dict = format_values_in_dictionary(atoms_dict)
        graph_mat = AtomicGraph.from_atoms_dict(atoms_dict=atoms_dict, r_cut=4.0)
        graph_mat.compute_edge_vectors()
        ase_edge_index = graph_mat.edge_index

        graph_mat.batch = torch.zeros(graph_mat.num_nodes, dtype=torch.long)
        graph_mat.update_edge_index()
        graph_mat.compute_edge_vectors()
        tnl_edge_index = graph_mat.edge_index

        for node_index in range(graph_mat.num_nodes):
            ase_pair_index = ase_edge_index[1][ase_edge_index[0] == node_index]
            tnl_pair_index = tnl_edge_index[1][tnl_edge_index[0] == node_index]

            self.assertTrue(
                torch.all(
                    torch.bincount(tnl_pair_index) == torch.bincount(ase_pair_index)
                )
            )


if __name__ == "__main__":
    unittest.main()
