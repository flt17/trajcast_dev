import os
import unittest

import torch
from e3nn.o3 import Irreps, Linear
import cuequivariance_torch as cuet

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    DISPLACEMENTS_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.model.models import EfficientTrajCastModel, FlexibleModel, TrajCastModel
from trajcast.cli.convert_o3_backend import convert_state_dicts


class TestFlexibleModel(unittest.TestCase):
    def test_returns_correct_model_from_dict_and_dumps_yaml(self):
        layer_dict = {
            "OneHotAtomTypeEncoding": {
                "OneHotAtomTypeEncoding": {
                    "number_of_species": 4,
                    "irreps_in": {},
                    "output_field": ATOM_TYPE_EMBEDDING_KEY,
                }
            },
            "EdgeLengthEncoding": {
                "EdgeLengthEncoding": {
                    "radial_basis": "FixedBasis",
                    "cutoff_function": {},
                    "basis_kwargs": {
                        "rmax": 3.0,
                        "rmin": 0,
                        "basis_function": "gaussian",
                        "basis_size": 10,
                        "normalization": True,
                    },
                    "cutoff_kwargs": {},
                    "irreps_in": {},
                    "output_field": EDGE_LENGTHS_EMBEDDING_KEY,
                }
            },
        }

        model = FlexibleModel(config=layer_dict)

        model.dump_config_to_yaml("tests/unit/model/data/config_dump.yaml")
        self.assertTrue(os.path.exists("tests/unit/model/data/config_dump.yaml"))
        os.remove("tests/unit/model/data/config_dump.yaml")

    def test_returns_correct_model_from_yaml(self):
        model = FlexibleModel.build_from_yaml("tests/unit/model/data/config.yaml")

        self.assertTrue(
            model.layers.EdgeLengthEncoding.irreps_out[EDGE_LENGTHS_EMBEDDING_KEY]
            == Irreps("10x0e")
        )

    def test_raises_error_if_yaml_file_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            FlexibleModel.build_from_yaml("tests/unit/model/data/config_fake.yaml")


class TestEfficientTrajCastModel(unittest.TestCase):
    def test_returns_model_is_build_correctly_and_forward_works(self):
        config = {
            "num_chem_elements": 3,
            "edge_cutoff": 5.0,
            "num_edge_rbf": 8,
            "num_edge_poly_cutoff": 6,
            "num_vel_rbf": 8,
            "vel_max": 0.20,
            "max_rotation_order": 2,
            "num_hidden_channels": 64,
            "num_mp_layers": 5,
            "avg_num_neighbors": 10.0,
            "edge_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "vel_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "nl_gate_kwargs": {
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            "conserve_ang_mom": True,
            "net_lin_mom": [0.0, 0.0, 0.0],
            "net_ang_mom": [0.0, 0.0, 0.0],
        }

        model = EfficientTrajCastModel(
            config=config,
            predicted_fields=[
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
            ],
        )

        self.assertTrue(hasattr(model, "_encoding"))
        self.assertTrue(hasattr(model, "_int_out"))
        self.assertTrue(hasattr(model, "_conservation"))

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="aspirin",
            files="aspirin_energy.extxyz",
            cutoff_radius=5.0,
            time_reversibility=False,
            rename=False,
        )

        graph = dataset[0]

        # test forward
        graph = model(graph)

        self.assertTrue(graph.target.size() == torch.Size([21, 6]))

    def test_returns_forward_results_independent_of_o3_backend(self):

        config = {
            "num_chem_elements": 3,
            "edge_cutoff": 5.0,
            "num_edge_rbf": 8,
            "num_edge_poly_cutoff": 6,
            "num_vel_rbf": 8,
            "vel_max": 0.20,
            "max_rotation_order": 2,
            "num_hidden_channels": 16,
            "num_mp_layers": 4,
            "avg_num_neighbors": 10.0,
            "edge_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "vel_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "nl_gate_kwargs": {
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            "conserve_ang_mom": True,
            "net_lin_mom": [0.0, 0.0, 0.0],
            "net_ang_mom": [0.0, 0.0, 0.0],
            "o3_backend": "e3nn",
        }
        model_e3nn = EfficientTrajCastModel(
            config=config,
        )

        model_cueq = EfficientTrajCastModel(config=config)
        model_cueq.o3_backend = "cueq"

        cueq_state_dict = convert_state_dicts(model_e3nn, device="cpu", write=False)

        model_cueq.load_state_dict(cueq_state_dict)

        self.assertTrue(isinstance(model_e3nn._int_out.LinearReadOut.linear, Linear))
        self.assertTrue(
            isinstance(model_cueq._int_out.LinearReadOut.linear, cuet.Linear)
        )

        dataset = AtomicGraphDataset(
            root="tests/unit/data/data",
            name="aspirin",
            files="aspirin_energy.extxyz",
            cutoff_radius=5.0,
            time_reversibility=False,
            rename=False,
        )

        graph_e3nn = dataset[0]
        graph_cueq = graph_e3nn.clone()

        graph_e3nn = model_e3nn(graph_e3nn)

        graph_cueq = model_cueq(graph_cueq)

        self.assertTrue(torch.all(torch.isclose(graph_e3nn.target, graph_cueq.target)))

        self.assertTrue(
            torch.all(
                torch.isclose(
                    graph_e3nn.node_features[:, :32],
                    graph_cueq.node_features[:, :32],
                    atol=1e-6,
                )
            )
        )


class TestTrajCastModel(unittest.TestCase):
    def test_returns_model_is_build_correctly_with_default_norms(self):
        config = {
            "num_chem_elements": 3,
            "edge_cutoff": 5.0,
            "num_edge_rbf": 8,
            "num_edge_poly_cutoff": 6,
            "num_vel_rbf": 8,
            "vel_max": 0.20,
            "max_rotation_order": 2,
            "num_hidden_channels": 64,
            "num_mp_layers": 5,
            "avg_num_neighbors": 10.0,
            "edge_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "vel_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "nl_gate_kwargs": {
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            "conserve_ang_mom": True,
            "net_lin_mom": [0.0, 0.0, 0.0],
            "net_ang_mom": [0.0, 0.0, 0.0],
        }

        model = TrajCastModel(
            config=config,
            predicted_fields=[
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
            ],
        )

        # check whether dump works as expected
        model.dump_config_to_yaml(filename="tests/unit/model/data/config_dump.yaml")
        self.assertTrue(os.path.exists("tests/unit/model/data/config_dump.yaml"))

        # check if we can load this
        model_from_yaml = TrajCastModel.build_from_yaml(
            "tests/unit/model/data/config_dump.yaml",
            rms_targets=[1.5, 2.0],
            mean_targets=[-1.0, 0.485],
            predicted_fields=[
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
            ],
        )

        self.assertTrue(
            model_from_yaml.layers.Conservation.prefactor_disps == torch.tensor(1.5)
        )
        os.remove("tests/unit/model/data/config_dump.yaml")

    def test_returns_model_is_build_correctly_with_user_norms(self):
        config = {
            "num_chem_elements": 3,
            "edge_cutoff": 5.0,
            "num_edge_rbf": 8,
            "num_edge_poly_cutoff": 6,
            "num_vel_rbf": 8,
            "vel_max": 0.20,
            "max_rotation_order": 2,
            "num_hidden_channels": 64,
            "num_mp_layers": 5,
            "avg_num_neighbors": 10.0,
            "edge_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "vel_mlp_kwargs": {"n_neurons": [64, 64, 64], "activation": "silu"},
            "nl_gate_kwargs": {
                "activation_scalars": {"o": "tanh", "e": "silu"},
                "activation_gates": {"e": "silu"},
            },
            "conserve_ang_mom": True,
            "net_lin_mom": [0.0, 0.0, 0.0],
            "net_ang_mom": [0.0, 0.0, 0.0],
        }

        model = TrajCastModel(
            config=config,
            predicted_fields=[
                DISPLACEMENTS_KEY,
                UPDATE_VELOCITIES_KEY,
            ],
            rms_targets=torch.tensor([1.5, 2.0]),
        )

        self.assertTrue(
            torch.all(model.state_dict()["rms_targets"] == torch.tensor([1.5, 2.0]))
        )
