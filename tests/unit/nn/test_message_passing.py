import unittest
from copy import deepcopy
from typing import Optional

import torch
from e3nn.o3 import Irreps
from torch_scatter import scatter

from tests.unit.nn.test_modules import CH3SCH3, EncodingLayer, Si
from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    ATOM_TYPES_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    NODE_FEATURES_KEY,
    SPHERICAL_HARMONIC_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._message_passing import (
    ConditionedMessagePassingLayer,
    MessagePassingLayer,
    ResidualConditionedMessagePassingLayer,
)


class TestResidualConditionedMessagePassingLayer(unittest.TestCase):
    def test_returns_module_is_correctly_initialised(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})
        embedding = EncodingLayer(
            lmax=2, n_features=64, n_species=4, irreps_node_features_0e_only=True
        )

        layer = ResidualConditionedMessagePassingLayer(
            max_rotation_order=embedding.lmax,
            node_features_field=NODE_FEATURES_KEY,
            edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            edge_length_embedding_field=EDGE_LENGTHS_EMBEDDING_KEY,
            output_field={},
            irreps_in=embedding.irreps_out,
            irreps_out="64x0o+64x0e+64x1o+64x1e+64x2o+64x2e",
            vel_len_emb_field="norm_embedding_velocities",
            nl_gate_kwargs={
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            species_emb_field=ATOM_TYPE_EMBEDDING_KEY,
        )

        # do some testing on the dimension
        self.assertTrue(
            layer.irreps_out[layer.output_field]
            == Irreps("64x0e+64x1o+64x1e+64x2o+64x2e")
        )
        self.assertTrue(layer.dtp_edge.irreps_out == Irreps("64x0e+64x1o+64x2e"))
        self.assertTrue(
            layer.linear_contraction.irreps_out == Irreps("64x0e+64x1o+64x2e")
        )
        self.assertTrue(len(layer.edge_mlp.hs) == 5)
        self.assertTrue(len(layer.vel_mlp.hs) == 5)

        self.assertTrue(
            layer.dtp_vel.irreps_out.simplify()
            == Irreps("192x0e+256x1o+128x1e+128x2o+256x2e")
        )
        self.assertTrue(
            layer.linear_updates.irreps_out == Irreps("320x0e+64x1o+64x1e+64x2o+64x2e")
        )

        self.assertTrue(
            layer.gate.irreps_out[layer.output_field]
            == Irreps("64x0e+64x1o+64x1e+64x2o+64x2e")
        )
        self.assertTrue(
            layer.tp_resnet.irreps_out == Irreps("64x0e+64x1o+64x1e+64x2o+64x2e")
        )
        self.assertTrue(layer.tp_resnet.irreps_in1 == Irreps("64x0e"))
        self.assertTrue(layer.tp_resnet.irreps_in2 == Irreps("4x0e"))

        # compare to old layer
        old_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field="norm_embedding_velocities",
            conditioning_weights_shared=True,
            linear_between_tps=True,
            irreps_in=embedding.irreps_out,
            irreps_out="64x0e+64x1o+64x1e+64x2o+64x2e",
            output_field="output",
            avg_num_neighbors=10.0,
            non_linearity=True,
            non_linearity_after_resnet=False,
            resnet=True,
            resnet_self_interaction=True,
            resnet_sc_element=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            fc_conditioning_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            non_linearity_kwargs={
                "irrep_gates": "256x0e",
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )

        # now check whether they are identical
        self.assertTrue(
            layer.irreps_out[layer.output_field]
            == old_layer.irreps_out[old_layer.output_field]
        )

        self.assertTrue(layer.dtp_edge.irreps_out == old_layer.dtp_message.irreps_out)
        self.assertTrue(
            layer.linear_contraction.irreps_out
            == old_layer.linear_contraction.irreps_out
        )
        self.assertTrue(layer.edge_mlp.hs == old_layer.fc.hs)
        self.assertTrue(layer.vel_mlp.hs == old_layer.fc_conditioning.hs)

        self.assertTrue(
            layer.dtp_vel.irreps_out.simplify()
            == old_layer.dtp_update.irreps_out.simplify()
        )
        self.assertTrue(
            layer.linear_updates.irreps_out == old_layer.linear_updates.irreps_out
        )

        self.assertTrue(
            layer.gate.irreps_out[layer.output_field]
            == old_layer.gate.irreps_out[old_layer.output_field]
        )
        self.assertTrue(layer.tp_resnet.irreps_out == old_layer.tp_resnet.irreps_out)

    def test_returns_forward_works_correctly(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})
        embedding = EncodingLayer(
            lmax=2, n_features=64, n_species=4, irreps_node_features_0e_only=True
        )

        torch.manual_seed(42)
        layer = ResidualConditionedMessagePassingLayer(
            max_rotation_order=embedding.lmax,
            node_features_field=NODE_FEATURES_KEY,
            edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            edge_length_embedding_field=EDGE_LENGTHS_EMBEDDING_KEY,
            output_field={},
            irreps_in=embedding.irreps_out,
            irreps_out="64x0o+64x0e+64x1o+64x1e+64x2o+64x2e",
            vel_len_emb_field="norm_embedding_velocities",
            nl_gate_kwargs={
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            species_emb_field=ATOM_TYPE_EMBEDDING_KEY,
        )

        # compare to old layer
        torch.manual_seed(42)
        old_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field="norm_embedding_velocities",
            conditioning_weights_shared=True,
            linear_between_tps=True,
            irreps_in=embedding.irreps_out,
            irreps_out="64x0e+64x1o+64x1e+64x2o+64x2e",
            output_field="output",
            avg_num_neighbors=10.0,
            non_linearity=True,
            non_linearity_after_resnet=False,
            resnet=True,
            resnet_self_interaction=True,
            resnet_sc_element=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            fc_conditioning_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            non_linearity_kwargs={
                "irrep_gates": "256x0e",
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )
        graph_old = deepcopy(graph)
        graph = embedding(graph)
        graph_old = embedding(graph_old)

        graph = layer(graph)
        graph_old = old_layer(graph_old)
        self.assertTrue(torch.all(graph.node_features == graph_old.output))


class TestConditionedMessagePassingLayer(unittest.TestCase):
    def test_returns_module_is_correctly_initialised(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})
        embedding = EncodingLayer(
            lmax=2, n_features=64, n_species=4, irreps_node_features_0e_only=True
        )

        layer = ConditionedMessagePassingLayer(
            max_rotation_order=embedding.lmax,
            node_features_field=NODE_FEATURES_KEY,
            edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            edge_length_embedding_field=EDGE_LENGTHS_EMBEDDING_KEY,
            output_field={},
            irreps_in=embedding.irreps_out,
            irreps_out="64x0o+64x0e+64x1o+64x1e+64x2o+64x2e",
            vel_len_emb_field="norm_embedding_velocities",
            nl_gate_kwargs={
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )

        # do some testing on the dimension
        self.assertTrue(
            layer.irreps_out[layer.output_field]
            == Irreps("64x0e+64x1o+64x1e+64x2o+64x2e")
        )
        self.assertTrue(layer.dtp_edge.irreps_out == Irreps("64x0e+64x1o+64x2e"))

        self.assertTrue(len(layer.edge_mlp.hs) == 5)
        self.assertTrue(len(layer.vel_mlp.hs) == 5)

        self.assertTrue(
            layer.dtp_vel.irreps_out.simplify()
            == Irreps("192x0e+256x1o+128x1e+128x2o+256x2e")
        )
        self.assertTrue(
            layer.linear_updates.irreps_out.simplify()
            == Irreps("320x0e+64x1o+64x1e+64x2o+64x2e")
        )

        self.assertTrue(
            layer.gate.irreps_out[layer.output_field]
            == Irreps("64x0e+64x1o+64x1e+64x2o+64x2e")
        )

        # compare to old layer
        old_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field="norm_embedding_velocities",
            conditioning_weights_shared=True,
            linear_between_tps=False,
            irreps_in=embedding.irreps_out,
            irreps_out="64x0e+64x1o+64x1e+64x2o+64x2e",
            output_field="output",
            avg_num_neighbors=10.0,
            non_linearity=True,
            non_linearity_after_resnet=False,
            resnet=False,
            resnet_self_interaction=False,
            resnet_sc_element=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            fc_conditioning_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            non_linearity_kwargs={
                "irrep_gates": "256x0e",
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )

        # now check whether they are identical
        self.assertTrue(
            layer.irreps_out[layer.output_field]
            == old_layer.irreps_out[old_layer.output_field]
        )

        self.assertTrue(layer.dtp_edge.irreps_out == old_layer.dtp_message.irreps_out)

        self.assertTrue(layer.edge_mlp.hs == old_layer.fc.hs)
        self.assertTrue(layer.vel_mlp.hs == old_layer.fc_conditioning.hs)

        self.assertTrue(
            layer.dtp_vel.irreps_out.simplify()
            == old_layer.dtp_update.irreps_out.simplify()
        )
        self.assertTrue(
            layer.linear_updates.irreps_out.simplify()
            == old_layer.linear_updates.irreps_out.simplify()
        )

        self.assertTrue(
            layer.gate.irreps_out[layer.output_field]
            == old_layer.gate.irreps_out[old_layer.output_field]
        )

    def test_returns_forward_works_correctly(self):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})
        embedding = EncodingLayer(
            lmax=2, n_features=64, n_species=4, irreps_node_features_0e_only=True
        )

        torch.manual_seed(42)
        layer = ConditionedMessagePassingLayer(
            max_rotation_order=embedding.lmax,
            node_features_field=NODE_FEATURES_KEY,
            edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            edge_length_embedding_field=EDGE_LENGTHS_EMBEDDING_KEY,
            output_field={},
            irreps_in=embedding.irreps_out,
            irreps_out="64x0e+64x1o+64x1e+64x2o+64x2e",
            vel_len_emb_field="norm_embedding_velocities",
            nl_gate_kwargs={
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )

        # compare to old layer
        torch.manual_seed(42)
        old_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field="norm_embedding_velocities",
            conditioning_weights_shared=True,
            linear_between_tps=False,
            irreps_in=embedding.irreps_out,
            irreps_out="64x0e+64x1o+64x1e+64x2o+64x2e",
            output_field="output",
            avg_num_neighbors=10.0,
            non_linearity=True,
            non_linearity_after_resnet=False,
            resnet=False,
            resnet_self_interaction=False,
            resnet_sc_element=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            fc_conditioning_kwargs={"n_neurons": [64, 64, 64], "activation": "silu"},
            non_linearity_kwargs={
                "irrep_gates": "256x0e",
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
        )
        graph_old = deepcopy(graph)
        graph = embedding(graph)
        graph_old = embedding(graph_old)

        graph = layer(graph)
        graph_old = old_layer(graph_old)

        self.assertTrue(torch.all(graph.node_features == graph_old.output))


class TestMessagePassingLayer(unittest.TestCase):
    def test_raises_error_if_no_irreps_given_for_inputs(self):
        with self.assertRaises(
            KeyError,
            msg=f"Specification of the irreps of {NODE_FEATURES_KEY} and {SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY} needed!",
        ):
            MessagePassingLayer()

    def test_returns_init_works_correctly_no_conditioning_no_resnet_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 3)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )

    def test_returns_init_works_correctly_no_conditioning_no_resnet_no_nonlinearity_node_features_0e_only(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer(irreps_node_features_0e_only=True)

        # define irreps out:
        irreps_out = [
            (embedding.n_features, (i, 1)) for i in range(embedding.lmax + 1)
        ] + [(embedding.n_features, (i, -1)) for i in range(embedding.lmax + 1)]

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=embedding.irreps_out,
            irreps_out=irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 3)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps.simplify(),
            mp_layer.irreps_out["output"].sort().irreps,
        )

    def test_returns_init_works_correctly_no_conditioning_no_resnet_with_nonlinearity_tanh(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=True,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            non_linearity_kwargs={
                "activation_scalars": ["tanh", "tanh"],
                "activation_gates": ["tanh"],
            },
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 4)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertEqual(
            mp_layer.linear_updates.irreps_out, mp_layer.gate.gate.irreps_in
        )
        self.assertEqual(len(mp_layer.gate.gate.irreps_gates), 1)
        self.assertEqual(mp_layer.gate.gate.irreps_gates[0][1].p, 1)
        self.assertEqual(
            mp_layer.gate.gate.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )

    def test_returns_init_works_correctly_no_conditioning_with_resnet_no_self_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=True,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 3)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )

    def test_returns_init_works_correctly_no_conditioning_with_resnet_with_self_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 4)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.linear_resnet.linear.irreps_out.sort().irreps,
        )

    def test_returns_init_works_correctly_with_conditioning_no_resnet_no_self_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 5)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

    def test_returns_init_works_correctly_with_conditioning_no_resnet_no_self_no_nonlinearity_species_weights(
        self,
    ):
        number_of_species = 4
        embedding = EncodingLayer(n_species=number_of_species)

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=False,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        # overall structure of the MP
        self.assertIsInstance(mp_layer, MessagePassingLayer)
        self.assertEqual(len(mp_layer._modules.keys()), 5)

        # first TP (same as above)
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])

        # second TP, now a a ModuleList
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning[0].hs[-1]
        )
        # make sure there is the right number of mlps
        self.assertEqual(len(mp_layer.fc_conditioning), number_of_species)
        # initialised with different weights
        weights_mlp1 = torch.cat(
            [param.data.view(-1) for param in mp_layer.fc_conditioning[0].parameters()]
        )
        weights_mlp2 = torch.cat(
            [param.data.view(-1) for param in mp_layer.fc_conditioning[1].parameters()]
        )
        self.assertFalse(torch.equal(weights_mlp1, weights_mlp2))
        # check that the last layer does not have an activation function for both conditioning and message MLP
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning[0]
            .__getattr__(f"layer{len(mp_layer.fc_conditioning[0].hs)-2}")
            .act
        )
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

    def test_returns_init_works_correctly_with_conditioning_with_resnet_no_self_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=True,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        # check number of modules
        self.assertEqual(len(mp_layer._modules.keys()), 5)
        # weights for dtps
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )

        # last weight layer without activation
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )

        # check irreps are correct
        # linear_updates out == node_features irreps
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )

        # linear updates in == dtp update out
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        # dtp message out = dtp update in
        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

    def test_returns_init_works_correctly_with_conditioning_with_resnet_with_self_no_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        # check number of modules
        self.assertEqual(len(mp_layer._modules.keys()), 6)
        # weights for dtps
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )

        # last weight layer without activation
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )

        # check irreps are correct
        # linear_updates out == node_features irreps
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps,
        )

        # linear updates in == dtp update out
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        # dtp message out = dtp update in
        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

        # resnet style
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps.simplify(),
            mp_layer.linear_resnet.linear.irreps_out.sort().irreps.simplify(),
        )

    def test_returns_init_works_correctly_with_conditioning_with_resnet_with_self_with_nonlinearity(
        self,
    ):
        _, graph = Si()
        embedding = EncodingLayer()

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": ["tanh", "tanh"],
                "activation_gates": ["tanh"],
            },
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        # check number of modules
        self.assertEqual(len(mp_layer._modules.keys()), 7)
        # weights for dtps
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )

        # last weight layer without activation
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )

        # check irreps are correct

        # linear updates in == dtp update out
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        # dtp message out = dtp update in
        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

        # resnet style
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps.simplify(),
            mp_layer.linear_resnet.linear.irreps_out.sort().irreps.simplify(),
        )

        # gate
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps.simplify(),
            mp_layer.gate.gate.irreps_in.sort().irreps.simplify(),
        )
        self.assertEqual(len(mp_layer.gate.gate.irreps_gates), 1)
        self.assertEqual(mp_layer.gate.gate.irreps_gates[0][1].p, 1)
        self.assertEqual(
            mp_layer.gate.gate.irreps_out.sort().irreps.simplify(),
            mp_layer.irreps_in[NODE_FEATURES_KEY].sort().irreps.simplify(),
        )

    def test_returns_init_works_correctly_with_conditioning_with_resnet_with_self_with_nonlinearity_with_linear_scalar_node_features_0e_only(
        self,
    ):
        _, graph = CH3SCH3()

        embedding = EncodingLayer(n_species=3, irreps_node_features_0e_only=True)

        # define irreps out:
        irreps_out = [
            (embedding.n_features, (i, 1)) for i in range(embedding.lmax + 1)
        ] + [(embedding.n_features, (i, -1)) for i in range(embedding.lmax + 1)]

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            irreps_out=irreps_out,
            output_field="output",
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": {"o": "tanh", "e": "tanh"},
                "activation_gates": {"o": "tanh", "e": "tanh"},
            },
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        # check number of modules
        self.assertEqual(len(mp_layer._modules.keys()), 7)
        # weights for dtps
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )

        # last weight layer without activation
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )

        # check irreps are correct

        # linear updates in == dtp update out
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        # dtp message out = dtp update in
        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps.simplify(),
        )

        # resnet style
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.linear_resnet.linear.irreps_out.sort().irreps,
        )

        # gate
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.gate.gate.irreps_in.sort().irreps,
        )
        self.assertEqual(len(mp_layer.gate.gate.irreps_gates), 1)
        self.assertEqual(mp_layer.gate.gate.irreps_gates[0][1].p, 1)
        self.assertEqual(
            mp_layer.gate.gate.irreps_out.sort().irreps,
            mp_layer.irreps_out["output"].sort().irreps,
        )

        # check equivariance
        self.assertTrue(
            test_layer_is_equivariant(
                graph=graph,
                encoding=embedding,
                layer=mp_layer,
                output_field="output",
            )
        )

    def test_returns_init_works_correctly_with_all_gadgets_and_linear_scalar_node_features_0e_only_up_to_rotation_order_2(
        self,
    ):
        _, graph = CH3SCH3()

        embedding = EncodingLayer(
            lmax=2, n_species=3, irreps_node_features_0e_only=True
        )

        # define irreps out:
        irreps_out = [
            (embedding.n_features, (i, 1)) for i in range(embedding.lmax + 1)
        ] + [(embedding.n_features, (i, -1)) for i in range(embedding.lmax + 1)]

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            irreps_in=embedding.irreps_out,
            irreps_out=irreps_out,
            output_field="output",
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": {"o": "tanh", "e": "tanh"},
                "activation_gates": {"o": "tanh", "e": "tanh"},
            },
        )

        self.assertIsInstance(mp_layer, MessagePassingLayer)
        # check number of modules
        self.assertEqual(len(mp_layer._modules.keys()), 7)
        # weights for dtps
        self.assertEqual(mp_layer.dtp_message.tp.weight_numel, mp_layer.fc.hs[-1])
        self.assertEqual(
            mp_layer.dtp_update.tp.weight_numel, mp_layer.fc_conditioning.hs[-1]
        )

        # last weight layer without activation
        self.assertIsNone(mp_layer.fc.__getattr__(f"layer{len(mp_layer.fc.hs)-2}").act)
        self.assertIsNone(
            mp_layer.fc_conditioning.__getattr__(
                f"layer{len(mp_layer.fc_conditioning.hs)-2}"
            ).act
        )

        # check irreps are correct

        # linear updates in == dtp update out
        self.assertEqual(
            mp_layer.linear_updates.irreps_in.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_out.sort().irreps.simplify(),
        )

        # dtp message out = dtp update in
        self.assertEqual(
            mp_layer.dtp_message.tp.irreps_out.sort().irreps.simplify(),
            mp_layer.dtp_update.tp.irreps_in1.sort().irreps,
        )

        # resnet style
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.linear_resnet.linear.irreps_out.sort().irreps,
        )

        # gate
        self.assertEqual(
            mp_layer.linear_updates.irreps_out.sort().irreps,
            mp_layer.gate.gate.irreps_in.sort().irreps,
        )
        self.assertEqual(len(mp_layer.gate.gate.irreps_gates), 1)
        self.assertEqual(mp_layer.gate.gate.irreps_gates[0][1].p, 1)
        self.assertEqual(
            mp_layer.gate.gate.irreps_out.sort().irreps,
            mp_layer.irreps_out["output"].sort().irreps,
        )

        # check equivariance
        self.assertTrue(
            test_layer_is_equivariant(
                graph=graph,
                encoding=embedding,
                layer=mp_layer,
                output_field="output",
            )
        )

    def test_returns_forward_works_correctly_with_conditioning_no_resnet_no_self_no_nonlinearity_species_weights(
        self,
    ):
        # we are passing a dictionary with 4 species on purpose as we might have trained a model which can also predicted
        # systems containing oxygen. For CH3SCH3 only 3 species/mlps would be used.
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        # we want to make sure that the graph still only has 3 different atom types
        # first assert is to check that we indeed have sulfur (atom type 3.)
        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        # this one is to check this graph does not contain oxygen (atom type 2.)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(n_species=number_of_species)

        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=False,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            output_field="output",
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
        )

        self.assertTrue(len(mp_layer.fc_conditioning) == number_of_species)

        graph = embedding(graph)
        graph = mp_layer(graph)

        self.assertIsInstance(graph, AtomicGraph)
        # node features should only have 0e features not zero
        # check for one atom
        # 0e
        node_features_0e = [
            graph.node_features[0][mp_layer.irreps_out["node_features"].slices()[i]]
            for i, (m, ir) in enumerate(mp_layer.irreps_out["node_features"])
            if ir == (0, 1)
        ][0]
        self.assertFalse(
            torch.equal(node_features_0e, torch.zeros(len(node_features_0e)))
        )
        # rest
        node_features_rest = torch.cat(
            [
                graph.node_features[0][mp_layer.irreps_out["node_features"].slices()[i]]
                for i, (m, ir) in enumerate(mp_layer.irreps_out["node_features"])
                if ir != (0, 1)
            ]
        )
        self.assertTrue(
            torch.equal(node_features_rest, torch.zeros(len(node_features_rest)))
        )

        # now check for the output field
        # we should find only 0o with zeros, rest non zero
        output_0o = [
            graph.output[0][mp_layer.irreps_out["output"].slices()[i]]
            for i, (m, ir) in enumerate(mp_layer.irreps_out["output"])
            if ir == (0, -1)
        ][0]
        self.assertTrue(torch.equal(output_0o, torch.zeros(len(output_0o))))

        # rest
        output_rest = torch.cat(
            [
                graph.output[0][mp_layer.irreps_out["output"].slices()[i]]
                for i, (m, ir) in enumerate(mp_layer.irreps_out["output"])
                if ir != (0, -1)
            ]
        )
        self.assertFalse(torch.equal(output_rest, torch.zeros(len(output_rest))))

    def test_returns_forward_works_correctly_with_conditioning_with_resnet_with_self_with_nonlinearity_species_weights_output_name_not_input(
        self,
    ):
        """This test is performed to check whether a change in the output field name will affect the results and whether all irreps are handled correctly."""
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(n_species=number_of_species)
        output_field = "output"
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=False,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": ["tanh", "tanh"],
                "activation_gates": ["tanh"],
            },
        )

        self.assertTrue(len(mp_layer.fc_conditioning) == number_of_species)

        graph = embedding(graph)
        graph = mp_layer(graph)

        self.assertIsInstance(graph, AtomicGraph)

        # node features should only have 0e features not zero
        # check for one atom
        # 0e
        node_features_0e = [
            graph.node_features[0][mp_layer.irreps_out["node_features"].slices()[i]]
            for i, (m, ir) in enumerate(mp_layer.irreps_out["node_features"])
            if ir == (0, 1)
        ][0]
        self.assertFalse(
            torch.equal(node_features_0e, torch.zeros(len(node_features_0e)))
        )
        # rest
        node_features_rest = torch.cat(
            [
                graph.node_features[0][mp_layer.irreps_out["node_features"].slices()[i]]
                for i, (m, ir) in enumerate(mp_layer.irreps_out["node_features"])
                if ir != (0, 1)
            ]
        )
        self.assertTrue(
            torch.equal(node_features_rest, torch.zeros(len(node_features_rest)))
        )

        # now check for the output field
        # we should find only 0o with zeros, rest non zero
        output_0o = [
            graph.output[0][mp_layer.irreps_out[output_field].slices()[i]]
            for i, (m, ir) in enumerate(mp_layer.irreps_out[output_field])
            if ir == (0, -1)
        ][0]
        self.assertTrue(torch.equal(output_0o, torch.zeros(len(output_0o))))

        # rest
        output_rest = torch.cat(
            [
                graph.output[0][mp_layer.irreps_out[output_field].slices()[i]]
                for i, (m, ir) in enumerate(mp_layer.irreps_out[output_field])
                if ir != (0, -1)
            ]
        )
        self.assertFalse(torch.equal(output_rest, torch.zeros(len(output_rest))))

        # check equivariance
        self.assertTrue(
            test_layer_is_equivariant(
                graph=graph,
                encoding=embedding,
                layer=mp_layer,
                output_field=output_field,
            )
        )

    def test_returns_forward_works_correctly_with_conditioning_with_resnet_with_self_with_nonlinearity_species_weights_same_name(
        self,
    ):
        """This test is performed to check whether the features are updated correctly in the same field."""
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(n_species=number_of_species)
        output_field = "node_features"
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=False,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
            non_linearity_kwargs={
                "activation_scalars": ["tanh", "tanh"],
                "activation_gates": ["tanh"],
            },
        )

        self.assertTrue(len(mp_layer.fc_conditioning) == number_of_species)

        graph = embedding(graph)

        # node features should only have 0e features not zero
        # check for one atom
        # 0e
        node_features_0e = [
            graph.node_features[0][embedding.irreps_out["node_features"].slices()[i]]
            for i, (m, ir) in enumerate(embedding.irreps_out["node_features"])
            if ir == (0, 1)
        ][0]
        self.assertFalse(
            torch.equal(node_features_0e, torch.zeros(len(node_features_0e)))
        )
        # rest
        node_features_rest = torch.cat(
            [
                graph.node_features[0][
                    embedding.irreps_out["node_features"].slices()[i]
                ]
                for i, (m, ir) in enumerate(embedding.irreps_out["node_features"])
                if ir != (0, 1)
            ]
        )
        self.assertTrue(
            torch.equal(node_features_rest, torch.zeros(len(node_features_rest)))
        )

        graph = mp_layer(graph)

        # now check for the output field
        # we should find only 0o with zeros, rest non zero
        output_0o = [
            graph[output_field][0][mp_layer.irreps_out[output_field].slices()[i]]
            for i, (m, ir) in enumerate(mp_layer.irreps_out[output_field])
            if ir == (0, -1)
        ][0]
        self.assertTrue(torch.equal(output_0o, torch.zeros(len(output_0o))))

        # rest
        output_rest = torch.cat(
            [
                graph[output_field][0][mp_layer.irreps_out[output_field].slices()[i]]
                for i, (m, ir) in enumerate(mp_layer.irreps_out[output_field])
                if ir != (0, -1)
            ]
        )
        self.assertFalse(torch.equal(output_rest, torch.zeros(len(output_rest))))

        # check equivariance
        self.assertTrue(
            test_layer_is_equivariant(
                graph=graph,
                encoding=embedding,
                layer=mp_layer,
                output_field=output_field,
            )
        )

    def test_forward_shows_element_specific_mlps_are_correctly_called(self):
        """Here, we check that element mlps are called correctly"""
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        number_of_species = 4
        embedding = EncodingLayer(n_species=number_of_species)
        output_field = "node_features"
        torch.manual_seed(45)
        mp_layer = MessagePassingLayer(
            max_rotation_order=embedding.lmax,
            input1_field=NODE_FEATURES_KEY,
            input2_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            weight_field=EDGE_LENGTHS_EMBEDDING_KEY,
            conditioning_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            conditioning_weight_field=f"norm_embedding_{VELOCITIES_KEY}",
            conditioning_weights_shared=False,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            output_field=output_field,
            non_linearity=False,
            resnet=False,
            resnet_self_interaction=False,
            tp_message_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            tp_update_kwargs={"multiplicity_mode": "uvu", "trainable": True},
            fc_kwargs={"n_neurons": [64, 64]},
            fc_conditioning_kwargs={
                "n_neurons": [64, 128, 64],
                "activation": "sigmoid",
            },
        )

        # embedding
        graph = embedding(graph)

        # do manual prediction
        # input to message passing layer
        node_features = graph.node_features
        # weights for the first tp
        weights_tp1 = mp_layer.fc(graph.edge_lengths_embedding)
        sender_indices = graph.edge_index[1]
        receiver_indices = graph.edge_index[0]
        messages = mp_layer.dtp_message(
            node_features[sender_indices], graph.sh_embedding_edge_vectors, weights_tp1
        )
        pooled_messages = (
            scatter(
                messages,
                receiver_indices,
                dim=0,
                dim_size=graph.num_nodes,
                reduce="sum",
            )
            / graph.ave_n_neighbors
        )

        pooling_no_torch_atom_0 = (
            messages[torch.where(receiver_indices == 0)[0]]
            .sum(dim=0)
            .div(graph.ave_n_neighbors)
        )

        self.assertTrue(torch.equal(pooled_messages[0], pooling_no_torch_atom_0))

        # compute tp for conditioning
        weights_tp2 = torch.stack(
            [
                mp_layer.fc_conditioning[graph[ATOM_TYPES_KEY][node]](
                    graph["norm_embedding_velocities"][node]
                )
                for node in range(graph.num_nodes)
            ]
        )

        # check the correct mlp is called for two random atoms 3 and 8
        weights_tp2_atom2 = mp_layer.fc_conditioning[graph.atom_types[2]](
            graph["norm_embedding_velocities"][2]
        )
        weights_tp2_atom8 = mp_layer.fc_conditioning[graph.atom_types[8]](
            graph["norm_embedding_velocities"][8]
        )
        # we do use different elements and thus different mlps
        self.assertFalse(torch.equal(graph.atom_types[2], graph.atom_types[8]))
        self.assertFalse(
            torch.equal(
                mp_layer.fc_conditioning[graph.atom_types[8]][2].weight,
                mp_layer.fc_conditioning[graph.atom_types[2]][2].weight,
            )
        )
        self.assertTrue(torch.equal(weights_tp2_atom2, weights_tp2[2]))
        self.assertTrue(torch.equal(weights_tp2_atom8, weights_tp2[8]))

        # perform tp
        pooled_messages = mp_layer.dtp_update(
            pooled_messages, graph.sh_embedding_velocities, weights_tp2
        )

        update_manual = mp_layer.linear_updates(pooled_messages)

        # compare with prediction of the module
        update_code = mp_layer(graph)[output_field]

        self.assertTrue(torch.equal(update_code, update_manual))

    def test_returns_full_forward_works_correctly_with_linear_between_tensor_products(
        self,
    ):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(
            n_species=number_of_species, irreps_node_features_0e_only=True, lmax=2
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
            linear_between_tps=True,
            number_of_species=number_of_species,
            irreps_in=embedding.irreps_out,
            irreps_out="8x0e+8x1o+8x1e+8x2o+8x2e",
            output_field=output_field,
            non_linearity=True,
            resnet=True,
            resnet_self_interaction=True,
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

        # check linear layer is there with correct input and output irreps
        self.assertEqual(
            mp_layer.linear_contraction.irreps_in, mp_layer.dtp_message.irreps_out
        )
        self.assertEqual(
            mp_layer.linear_contraction.irreps_out, mp_layer.dtp_update.tp.irreps_in1
        )

        # forward without breaking
        graph = embedding(graph)
        graph = mp_layer(graph)

    def test_returns_full_forward_works_correctly_with_non_linearity_on_messages_only(
        self,
    ):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(
            n_species=number_of_species, irreps_node_features_0e_only=True, lmax=2
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
            non_linearity_after_resnet=False,
            resnet_self_interaction=True,
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

        # check whether resnet sc is correct irreps (same as output irreps)
        # this is reassurring as it would break if we passed these irreps to the gate
        # as they do not add up with the messages
        self.assertTrue(
            mp_layer.linear_resnet.irreps_out["node_features"]
            == mp_layer.irreps_out["node_features"]
        )

        # forward without breaking
        graph = embedding(graph)
        graph = mp_layer(graph)

    def test_returns_full_forward_works_correctly_with_non_linearity_on_messages_only_and_sc_with_elements(
        self,
    ):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(
            n_species=number_of_species, irreps_node_features_0e_only=True, lmax=2
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
            non_linearity_after_resnet=False,
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

        # check tensor product in resnet rather than linear
        self.assertTrue(hasattr(mp_layer, "tp_resnet"))

        # check whether resnet sc (here a tensorproduct) is correct irreps (same as output irreps)
        # this is reassurring as it would break if we passed these irreps to the gate
        # as they do not add up with the messages
        self.assertTrue(
            mp_layer.tp_resnet.irreps_out == mp_layer.irreps_out["node_features"]
        )

        # forward without breaking
        graph = embedding(graph)
        graph = mp_layer(graph)

    def test_returns_full_forward_works_correctly_with_non_linearity_after_resnet_and_sc_with_elements(
        self,
    ):
        _, graph = CH3SCH3({1: 0, 6: 1, 8: 2, 16: 3})

        self.assertTrue(graph[ATOM_TYPES_KEY].max() == 3)
        self.assertFalse(torch.any(graph[ATOM_TYPES_KEY] == 2))

        number_of_species = 4
        embedding = EncodingLayer(
            n_species=number_of_species, irreps_node_features_0e_only=True, lmax=2
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

        # check tensor product in resnet rather than linear
        self.assertTrue(hasattr(mp_layer, "tp_resnet"))

        # check whether resnet sc (here a tensorproduct) is correct irreps (same as input for gate)
        # this is reassurring as it would break if we passed these irreps to the gate
        # as they do not add up with the messages
        self.assertTrue(
            mp_layer.tp_resnet.irreps_out == mp_layer.gate.irreps_in["node_features"]
        )

        # forward without breaking
        graph = embedding(graph)
        graph = mp_layer(graph)


def test_layer_is_equivariant(
    graph: AtomicGraph,
    encoding: torch.nn.Module,
    layer: MessagePassingLayer,
    output_field: Optional[str] = "output",
    seed: Optional[int] = torch.random.seed(),
    atol: Optional[float] = 1e-5,
    rtol: Optional[float] = 1e-3,
) -> bool:
    """This method checks whether the mp layer (in combination with the encoding) is equivariant.
    We do this to make sure the irreps are not screwed up and our message passing layer ends up not being equivariant.

    Args:
        graph (AtomicGraph): The graph the test will be performed with.
        encoding (torch.nn.Module): The encoding layer (already initialised)
        layer (MessagePassingLayer): The message passing layer (already initialised)
        output_field (Optional[str], optional): Name of the output field. Defaults to "output".
        seed (Optional[int], optional): Seed to initial the rotation matrix. Defaults to torch.random.seed().

    Returns:
        bool: True if module is equivariant, False if not.
    """
    torch.manual_seed(seed)
    # define rotation matrix
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    graph_before = graph
    # apply encoding before rotation
    graph_before = encoding(graph)
    graph_before = layer(graph)
    output_before = graph_before[output_field]

    # apply rotation to all relevant properties
    graph.pos = torch.mm(R, graph.pos.T).T
    graph.edge_vectors = torch.mm(R, graph.edge_vectors.T).T
    graph[VELOCITIES_KEY] = torch.mm(R, graph[VELOCITIES_KEY].T).T
    # pass then to
    graph_after = encoding(graph)
    graph_after = layer(graph)
    output_after = graph_after[output_field]

    # apply rotation to output space for output_before
    output_before = torch.mm(
        layer.irreps_out[output_field].D_from_matrix(R), output_before.T
    ).T

    return torch.allclose(output_before, output_after, atol=atol, rtol=rtol)
