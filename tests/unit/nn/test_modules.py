import unittest
from typing import Dict, Optional, Union

import ase
import ase.build
import numpy as np
import torch
from e3nn.nn import Gate
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

from tests.unit.model.test_training import delete_all_torch_files
from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    NODE_FEATURES_KEY,
    SPHERICAL_HARMONIC_KEY,
    TIMESTEP_ENCODING_KEY,
    TIMESTEP_KEY,
    TOTAL_MASS_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.nn._encoding import (
    EdgeLengthEncoding,
    OneHotAtomTypeEncoding,
    SphericalHarmonicProjection,
    TensorNormEncoding,
    TimestepEncoding,
)
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._message_passing import MessagePassingLayer
from trajcast.nn._tensor_self_interactions import LinearTensorMixer
from trajcast.nn.modules import (
    ConservationLayer,
    ForecastHorizonConditioning,
)
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    format_values_in_dictionary,
)


class TestConservationLayer(unittest.TestCase):
    def test_layer_removes_total_linear_momentum_properly_for_one_molecule(self):
        atoms, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        # generate fake target vectors
        graph["target"] = torch.randn(graph.num_nodes, 6)
        # get layer
        layer = ConservationLayer(index_vel_target=3)

        # forward
        graph = layer(graph)

        # check whether momentum is conserved
        prior_pred = (
            (atoms.arrays["velocities"] * atoms.get_masses().reshape(-1, 1))
            .to(dtype=torch.float32)
            .sum(0)
        )

        post_pred = (
            (graph.target[:, 3:] * atoms.get_masses().reshape(-1, 1))
            .to(dtype=torch.float32)
            .sum(0)
        )

        self.assertTrue(torch.allclose(prior_pred, post_pred, atol=1e-4))

    def test_layer_removes_total_linear_momentum_properly_for_one_molecule_when_target_is_set(
        self,
    ):
        atoms, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        # generate fake target vectors
        graph["target"] = torch.randn(graph.num_nodes, 6)
        # get layer
        layer = ConservationLayer(index_vel_target=3)

        # set target to zero:
        target_net_lin_mom = torch.zeros(3)
        layer.net_lin_mom = target_net_lin_mom

        # forward
        graph = layer(graph)

        # get the new net linear momentum which should be zero irrespective of the initial velocities
        post_pred = (
            (graph.target[:, 3:] * atoms.get_masses().reshape(-1, 1))
            .to(dtype=torch.float32)
            .sum(0)
        )

        # check displacements
        com_motion = (
            (graph.target[:, 0:3] * atoms.get_masses().reshape(-1, 1)).sum(0)
            / atoms.get_masses().sum()
        ).to(dtype=torch.float32)

        self.assertTrue(torch.allclose(post_pred, target_net_lin_mom, atol=1e-4))
        self.assertTrue(torch.allclose(com_motion, torch.zeros(3), atol=1e-6))

    def test_layer_removes_total_linear_momentum_properly_for_batch(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        layer = ConservationLayer(index_vel_target=3)

        for batch in dataloader:
            batch["target"] = torch.randn(batch.num_nodes, 6)

            batch = layer(batch)

            # check whether momentum is conserved
            prior_pred = scatter(
                batch.velocities * batch.atomic_masses,
                batch.batch,
                dim=0,
                reduce="sum",
            )

            post_pred = scatter(
                batch.target[:, 3:] * batch.atomic_masses,
                batch.batch,
                dim=0,
                reduce="sum",
            )
            self.assertTrue(torch.allclose(prior_pred, post_pred, atol=1e-4))
        delete_all_torch_files()

    def test_layer_removes_total_linear_momentum_properly_for_batch_when_target_is_set(
        self,
    ):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        layer = ConservationLayer(index_vel_target=3, disp_norm_const=0.05)

        # set target to zero:
        target_net_lin_mom = torch.zeros(3)
        layer.net_lin_mom = target_net_lin_mom

        for batch in dataloader:
            batch["target"] = torch.randn(batch.num_nodes, 6)

            batch = layer(batch)

            post_pred = scatter(
                batch.target[:, 3:] * batch.atomic_masses,
                batch.batch,
                dim=0,
                reduce="sum",
            )

            # check displacements
            com_motion = (
                scatter(
                    batch.target[:, 0:3] * batch.atomic_masses,
                    batch.batch,
                    dim=0,
                    reduce="sum",
                )
                / batch[TOTAL_MASS_KEY].view(-1, 1)
                * layer.prefactor_disps
            )
            self.assertTrue(torch.allclose(target_net_lin_mom, post_pred, atol=1e-4))
            self.assertTrue(torch.allclose(com_motion, torch.zeros(3), atol=1e-7))

        delete_all_torch_files()

    def test_layer_removes_total_linear_momentum_properly_with_normalization(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        norm_constant_velocities = 0.02
        layer = ConservationLayer(
            index_vel_target=3, vel_norm_const=norm_constant_velocities
        )

        for batch in dataloader:
            batch["target"] = torch.randn(batch.num_nodes, 6)

            batch = layer(batch)

            # check whether momentum is conserved
            prior_pred = scatter(
                batch.velocities * batch.atomic_masses / norm_constant_velocities,
                batch.batch,
                dim=0,
                reduce="sum",
            )

            post_pred = scatter(
                batch.target[:, 3:] * batch.atomic_masses,
                batch.batch,
                dim=0,
                reduce="sum",
            )
            self.assertTrue(torch.allclose(prior_pred, post_pred, atol=1e-4))
        delete_all_torch_files()

    def test_layer_removes_angular_momentum_for_molecules_in_batch(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        layer = ConservationLayer(
            index_disp_target=0, index_vel_target=3, conserve_angular=True
        )

        for batch in dataloader:
            batch["target"] = torch.randn(batch.num_nodes, 6)

            batch = layer(batch)

            # compute by hand whether angular momentum is conserved
            masses = batch.atomic_masses
            total_mass = scatter(masses, index=batch.batch, dim=0, reduce="sum")
            # center of mass
            coms_in = (
                scatter(masses * batch.pos, index=batch.batch, dim=0, reduce="sum")
                / total_mass
            )
            # distances (no pbc)
            dist_com_in = batch.pos - coms_in[batch.batch]

            momenta_in = masses * batch.velocities
            momenta_out = masses * batch.target[:, 3:]

            ang_mom_in = scatter(
                torch.linalg.cross(dist_com_in, momenta_in),
                index=batch.batch,
                dim=0,
                reduce="sum",
            )

            # center of mass output
            new_pos = batch.pos + batch.target[:, 0:3]
            coms_out = (
                scatter(masses * new_pos, index=batch.batch, dim=0, reduce="sum")
                / total_mass
            )
            # distances (no pbc)
            dist_com_out = new_pos - coms_out[batch.batch]

            ang_mom_out = scatter(
                torch.linalg.cross(dist_com_out, momenta_out),
                index=batch.batch,
                dim=0,
                reduce="sum",
            )
            self.assertTrue(torch.allclose(ang_mom_in, ang_mom_out, atol=1e-4))
        delete_all_torch_files()

    def test_layer_removes_angular_momentum_for_molecules_in_batch_with_set_target(
        self,
    ):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        layer = ConservationLayer(
            index_disp_target=0, index_vel_target=3, conserve_angular=True
        )
        target_ang_mom = torch.zeros(3)
        layer.net_ang_mom = target_ang_mom

        for batch in dataloader:
            batch["target"] = torch.randn(batch.num_nodes, 6)

            batch = layer(batch)

            # compute by hand whether angular momentum is conserved
            masses = batch.atomic_masses
            total_mass = batch.total_mass

            momenta_out = masses * batch.target[:, 3:]

            # center of mass output
            new_pos = batch.pos + batch.target[:, 0:3]
            coms_out = scatter(
                masses * new_pos, index=batch.batch, dim=0, reduce="sum"
            ) / total_mass.view(-1, 1)
            # distances (no pbc)
            dist_com_out = new_pos - coms_out[batch.batch]

            ang_mom_out = scatter(
                torch.linalg.cross(dist_com_out, momenta_out),
                index=batch.batch,
                dim=0,
                reduce="sum",
            )
            self.assertTrue(
                torch.allclose(target_ang_mom.repeat(2, 1), ang_mom_out, atol=1e-4)
            )

    def test_layer_removes_angular_and_linear_momentum_for_one_molecule(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        # generate fake target vectors
        graph["target"] = torch.randn(graph.num_nodes, 6)
        # get layer
        layer = ConservationLayer(
            index_disp_target=0, index_vel_target=3, conserve_angular=True
        )

        # forward
        graph = layer(graph)

        # check whether momenta are conserved
        masses = graph.atomic_masses
        momenta_in = graph.velocities * masses
        momenta_out = graph.target[:, 3:] * masses
        com_in = (graph.pos * masses / masses.sum()).sum(0)
        dist_com_in = graph.pos - com_in
        new_pos = graph.pos + graph.target[:, 0:3]
        com_out = (new_pos * masses / masses.sum()).sum(0)
        dist_com_out = new_pos - com_out

        # linear
        prior_pred_lin = (momenta_in).sum(0)
        post_pred_lin = (momenta_out).sum(0)

        self.assertTrue(torch.allclose(prior_pred_lin, post_pred_lin, atol=1e-4))

        # angular
        prior_pred_ang = torch.linalg.cross(dist_com_in, momenta_in).sum(0)
        post_pred_ang = torch.linalg.cross(dist_com_out, momenta_out).sum(0)

        self.assertTrue(torch.allclose(prior_pred_ang, post_pred_ang, atol=1e-4))

    def test_layer_removes_angular_and_linear_momentum_for_one_molecule_with_set_targets(
        self,
    ):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        # generate fake target vectors
        graph["target"] = torch.randn(graph.num_nodes, 6)
        # get layer
        layer = ConservationLayer(
            index_disp_target=0, index_vel_target=3, conserve_angular=True
        )
        layer.net_lin_mom = torch.zeros(3)
        layer.net_ang_mom = torch.zeros(3)
        # forward
        graph = layer(graph)

        # check whether momenta are conserved
        masses = graph.atomic_masses
        momenta_out = graph.target[:, 3:] * masses
        new_pos = graph.pos + graph.target[:, 0:3]
        com_out = (new_pos * masses / masses.sum()).sum(0)
        dist_com_out = new_pos - com_out

        # linear
        post_pred_lin = (momenta_out).sum(0)

        # check displacements
        com_motion = (graph.target[:, 0:3] * graph.atomic_masses).sum(0) / graph[
            TOTAL_MASS_KEY
        ]

        self.assertTrue(torch.allclose(torch.zeros(3), post_pred_lin, atol=1e-4))
        self.assertTrue(torch.allclose(com_motion, torch.zeros(3), atol=1e-6))

        # angular
        post_pred_ang = torch.linalg.cross(dist_com_out, momenta_out).sum(0)

        self.assertTrue(torch.allclose(torch.zeros(3), post_pred_ang, atol=1e-4))


class TestHorizonConditioningLayer(unittest.TestCase):
    def test_returns_module_initialises_correctly(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 16: 2})
        graph[TIMESTEP_KEY] = 0.5
        number_of_species = 3
        embedding = EncodingLayer(
            n_species=number_of_species,
            irreps_node_features_0e_only=True,
            lmax=2,
            timestep_encoding=True,
        )
        output_field = "node_features"
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=True,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            irreps_out="8x0e+8x1o+8x1e+8x2o+8x2e",
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            non_linearity_after_resnet=True,
            resnet_self_interaction=True,
            resnet_sc_element=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": {"e": "tanh", "o": "tanh"},
                "activation_gates": {"e": "tanh", "o": "tanh"},
            },
        )

        timestep_conditioning = ForecastHorizonConditioning(
            node_features_field=NODE_FEATURES_KEY,
            timestep_embedding_field=TIMESTEP_ENCODING_KEY,
            irreps_in=mp_layer.irreps_out,
        )
        self.assertTrue(timestep_conditioning.scalar_slice == slice(0, 8, None))
        self.assertTrue(timestep_conditioning.vector_slice == slice(8, 32, None))
        self.assertIsInstance(timestep_conditioning.gate, Gate)
        self.assertTrue(timestep_conditioning.irreps_out[NODE_FEATURES_KEY] == "8x1o")

    def test_returns_module_performs_forward_correctly(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 16: 2})
        graph[TIMESTEP_KEY] = 0.5
        number_of_species = 3
        embedding = EncodingLayer(
            n_species=number_of_species,
            irreps_node_features_0e_only=True,
            lmax=2,
            timestep_encoding=True,
        )
        output_field = "node_features"
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=True,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            irreps_out="8x0e+8x1o+8x1e+8x2o+8x2e",
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            non_linearity_after_resnet=True,
            resnet_self_interaction=True,
            resnet_sc_element=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": {"e": "tanh", "o": "tanh"},
                "activation_gates": {"e": "tanh", "o": "tanh"},
            },
        )

        timestep_conditioning = ForecastHorizonConditioning(
            node_features_field=NODE_FEATURES_KEY,
            timestep_embedding_field=TIMESTEP_ENCODING_KEY,
            irreps_in=mp_layer.irreps_out,
            activation_function_gate="identity",
        )

        graph = embedding(graph)
        graph = mp_layer(graph)
        node_features = graph[NODE_FEATURES_KEY]
        graph.timestep_encoding = graph.timestep_encoding.to(graph.pos.device)
        graph = timestep_conditioning(graph)

        self.assertTrue(graph[NODE_FEATURES_KEY].shape[1] == 24)

        # calculate by hand
        scalar_prior = node_features[:, 0:8]

        scalar_w_timestep = torch.cat(
            [scalar_prior, graph[TIMESTEP_ENCODING_KEY].expand(graph.num_nodes, -1)],
            dim=1,
        )
        scalar_w_timestep = timestep_conditioning.fc(scalar_w_timestep)
        vectors_scaled = scalar_w_timestep.reshape(-1, 8, 1) * node_features[
            :, 8:32
        ].reshape(-1, 8, 3)
        vectors_scaled = vectors_scaled.reshape(-1, 24)

        self.assertTrue(torch.allclose(vectors_scaled, graph.node_features, atol=5e-3))

    def test_returns_module_performs_forward_correctly_for_batch(self):
        dataset = AtomicGraphDataset(
            root="tests/unit/model/data/forecast_benzene/",
            name="benzene",
            cutoff_radius=5.0,
            files="benzene_validation_traj.extxyz",
            rename=False,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        number_of_species = 3
        embedding = EncodingLayer(
            n_species=number_of_species,
            irreps_node_features_0e_only=True,
            lmax=2,
            timestep_encoding=True,
        )
        output_field = "node_features"
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=True,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            irreps_out="8x0e+8x1o+8x1e+8x2o+8x2e",
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            non_linearity_after_resnet=True,
            resnet_self_interaction=True,
            resnet_sc_element=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": {"e": "tanh", "o": "tanh"},
                "activation_gates": {"e": "tanh", "o": "tanh"},
            },
        )

        timestep_conditioning = ForecastHorizonConditioning(
            node_features_field=NODE_FEATURES_KEY,
            timestep_embedding_field=TIMESTEP_ENCODING_KEY,
            irreps_in=mp_layer.irreps_out,
            activation_function_gate="identity",
        )

        for batch in dataloader:
            batch["timestep"] = torch.tensor([1.0, 2.0])
            batch = embedding(batch)
            batch = mp_layer(batch)
            batch.timestep_encoding = batch.timestep_encoding.to(batch.pos.device)
            batch = timestep_conditioning(batch)

            self.assertTrue(batch[NODE_FEATURES_KEY].size() == torch.Size([24, 24]))
        delete_all_torch_files()


def Si() -> Union[ase.Atoms, AtomicGraph]:
    """Create ase.Atoms object for periodic silicon.

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

    ase_atoms = ase.Atoms("Si2", cell=lattice, positions=coords, pbc=True)
    ase_atoms = ase.build.make_supercell(ase_atoms, P=2 * np.identity(3))

    ase_atoms.rattle()
    graph = AtomicGraph.from_atoms_dict(
        atoms_dict=format_values_in_dictionary(
            convert_ase_atoms_to_dictionary(ase_atoms)
        ),
        r_cut=3.0,
    )

    return ase_atoms, graph


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


class EncodingLayer(torch.nn.Module, GraphModuleIrreps):
    def __init__(
        self,
        lmax: int = 1,
        n_features=8,
        n_species=1,
        irreps_node_features_0e_only=False,
        timestep_encoding=False,
    ):
        super().__init__()
        self.one_hot_encoding_atom_types = OneHotAtomTypeEncoding(
            number_of_species=n_species
        )
        self.edge_length_encoding = EdgeLengthEncoding(
            basis_kwargs={"rmax": 3.0},
            irreps_in=self.one_hot_encoding_atom_types.irreps_out,
        )
        self.sh_encoding_edge_vecs = SphericalHarmonicProjection(
            max_rotation_order=lmax,
            input_field=EDGE_VECTORS_KEY,
            irreps_in=self.edge_length_encoding.irreps_out,
        )

        self.sh_encoding_vel_vecs = SphericalHarmonicProjection(
            max_rotation_order=lmax,
            input_field=VELOCITIES_KEY,
            irreps_in=self.sh_encoding_edge_vecs.irreps_out,
        )

        self.vel_norm_encoding = TensorNormEncoding(
            basis_kwargs={"rmax": 3.0},
            input_field=VELOCITIES_KEY,
            irreps_in=self.sh_encoding_vel_vecs.irreps_out,
        )
        if not irreps_node_features_0e_only:
            irreps_out_linear = [(n_features, (i, 1)) for i in range(lmax + 1)] + [
                (n_features, (i, -1)) for i in range(lmax + 1)
            ]
        else:
            irreps_out_linear = [(n_features, (0, 1))]

        self.linear_init = LinearTensorMixer(
            input_field=ATOM_TYPE_EMBEDDING_KEY,
            output_field=NODE_FEATURES_KEY,
            irreps_in=self.vel_norm_encoding.irreps_out,
            irreps_out=irreps_out_linear,
        )

        self.irreps_out = self.linear_init.irreps_out

        self.timestep_encoding = timestep_encoding
        if timestep_encoding:
            self.dt_encoder = TimestepEncoding(
                embedding_dimension=n_features,
                max_timestep=5.0,
                irreps_in=self.linear_init.irreps_out,
            )

            self.irreps_out = self.dt_encoder.irreps_out

        self.lmax = lmax
        self.n_features = n_features

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data.compute_edge_vectors()
        data = self.one_hot_encoding_atom_types(data)
        data = self.edge_length_encoding(data)
        data = self.sh_encoding_edge_vecs(data)
        data = self.sh_encoding_vel_vecs(data)
        data = self.vel_norm_encoding(data)
        data = self.linear_init(data)
        if self.timestep_encoding:
            data = self.dt_encoder(data)
        return data


if __name__ == "__main__":
    unittest.main()
