import glob
import os
import shutil
import unittest
from typing import Dict

import torch

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    DISPLACEMENTS_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    NODE_FEATURES_KEY,
    SPHERICAL_HARMONIC_KEY,
)
from trajcast.model.training import Trainer


def delete_all_torch_files():
    files = glob.glob("tests/unit/data/data/*pt")
    for file in files:
        os.remove(file)
    files = glob.glob("tests/unit/model/data/*pt")
    for file in files:
        os.remove(file)
    files = glob.glob("tests/unit/model/data/forecast_benzene/benz*pt")
    for file in files:
        os.remove(file)


def config() -> Dict:
    return {
        "model": {
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
                        "rmax": 4.0,
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
            "SH_EdgeVectors": {
                "SphericalHarmonicProjection": {
                    "max_rotation_order": 2,
                    "project_on_unit_sphere": True,
                    "normalization_projection": "component",
                    "irreps_in": {},
                    "input_field": EDGE_VECTORS_KEY,
                    "output_field": f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
                }
            },
            "LinearTypeEncoding": {
                "LinearTensorMixer": {
                    "input_field": ATOM_TYPE_EMBEDDING_KEY,
                    "output_field": NODE_FEATURES_KEY,
                    "irreps_in": {},
                    "irreps_out": "8x0e+8x0o+8x1e+8x1o",
                }
            },
            "MessagePassingLayer1": {
                "MessagePassingLayer": {
                    "max_rotation_order": 2,
                    "input1_field": NODE_FEATURES_KEY,
                    "input2_field": f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
                    "weight_field": EDGE_LENGTHS_EMBEDDING_KEY,
                    "conditioning_field": {},
                    "output_field": NODE_FEATURES_KEY,
                    "irreps_in": {},
                    "resnet": True,
                    "fc_kwargs": {"n_neurons": [64, 64]},
                    "tp_message_kwargs": {
                        "multiplicity_mode": "uvu",
                        "trainable": True,
                    },
                    "tp_update_kwargs": {},
                }
            },
            "MessagePassingLayer2": {
                "MessagePassingLayer": {
                    "max_rotation_order": 2,
                    "input1_field": NODE_FEATURES_KEY,
                    "input2_field": f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
                    "weight_field": EDGE_LENGTHS_EMBEDDING_KEY,
                    "conditioning_field": {},
                    "output_field": NODE_FEATURES_KEY,
                    "irreps_in": {},
                    "resnet": True,
                    "fc_kwargs": {"n_neurons": [64, 64]},
                    "tp_message_kwargs": {
                        "multiplicity_mode": "uvu",
                        "trainable": True,
                    },
                    "tp_update_kwargs": {},
                }
            },
            "LinearTarget": {
                "LinearTensorMixer": {
                    "input_field": NODE_FEATURES_KEY,
                    "output_field": "target",
                    "irreps_in": {},
                    "irreps_out": "1o",
                }
            },
        },
        "data": {
            "root": "tests/unit/model/data/",
            "name": "Ac-Ala3-NHMe",
            "cutoff_radius": 4.0,
            "files": "md22_Ac-Ala3-NHMe_disp_100frames.extxyz",
            "atom_type_mapper": {1: 0, 6: 1, 7: 2, 8: 3},
        },
        "training": {
            "target_field": "target",
            "reference_fields": [DISPLACEMENTS_KEY],
            "batch_size": 5,
            "num_epochs": 10,
            "criterion": {
                "loss_type": {"main_loss": "mse"},
                "learnable_weights": False,
                "weights": [1],
            },
            "optimizer": "Adam",
            "optimizer_settings": {
                "lr": 0.01,
                "amsgrad": True,
            },
            "scheduler": ["LinearLR", "CosineAnnealingLR"],
            "scheduler_settings": {
                "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
                "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
            },
            "chained_scheduler_hp": [5000, 80000],
            "tensorboard_settings": {
                "log_dir": "tests/unit/model/data/tb_log",
                "loss": True,
            },
            "checkpoint_settings": {"root": "tests/unit/model/data/checkpoints"},
        },
    }


class TestTrainerInit(unittest.TestCase):
    def test_raises_error_if_file_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            Trainer.build_from_yaml("tests/unit/model/data/config_fake.yaml")

    def test_builds_experiment_from_config_dict_and_dumps_to_yaml_flexible(self):
        conf = config()
        experiment = Trainer(conf)
        experiment.dump_config_to_yaml("tests/unit/model/data/config.yaml")
        self.assertTrue(os.path.exists("tests/unit/model/data/config.yaml"))

        delete_all_torch_files()

    def test_returns_experiment_with_flexible_model(self):
        experiment = Trainer.build_from_yaml("tests/unit/model/data/config.yaml")
        assert hasattr(experiment, "model")

        delete_all_torch_files()

    def test_returns_experiment_with_trajcast_model(self):
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_train_trajcast.yaml"
        )
        assert hasattr(experiment, "model")

        delete_all_torch_files()


class TestTrainerTrain(unittest.TestCase):
    def test_perform_experiment_with_complete_tensorboard(self):
        # test tensorboard
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_tb_complete.yaml"
        )
        experiment.train()

        # check whether log file exists
        log_files = glob.glob(os.path.join("tests/unit/model/data/logs/", "log*"))

        self.assertTrue(log_files)

        # check tensorboard files exist
        self.assertTrue(os.path.exists("tests/unit/model/data/tb_log/loss_training"))

        delete_all_torch_files()

        for direc in [
            "tests/unit/model/data/checkpoints",
            "tests/unit/model/data/logs",
            "tests/unit/model/data/tb_log",
        ]:
            shutil.rmtree(direc)

    def test_performs_experiment_correctly_with_and_saves_and_restarts_from_checkpoint_no_scheduler(
        self,
    ):
        # Running for 10 and then other 10
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_nosched.yaml"
        )
        experiment.scheduler = None
        experiment.scheduler_settings = None
        experiment.train()

        # check whether log file exists
        log_files = glob.glob(os.path.join("tests/unit/model/data/logs/", "log*"))
        self.assertTrue(log_files)

        # check checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-9.pt")
        )

        delete_all_torch_files()

        # let us now try to restart
        # we checked manually in the log file it is restarting from the correct checkpoint
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_nosched.yaml"
        )
        experiment.restart_latest = True
        experiment.num_epochs = 15

        experiment.train()

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-14.pt")
        )

        # delete all directories
        for direc in [
            "tests/unit/model/data/checkpoints",
            "tests/unit/model/data/logs",
            "tests/unit/model/data/tb_log",
        ]:
            shutil.rmtree(direc)

    def test_performs_experiment_correctly_with_and_saves_and_restarts_from_checkpoint_chained_scheduler(
        self,
    ):
        # Starting the experiment for 10 epochs and continuing it for 10
        experiment = Trainer.build_from_yaml("tests/unit/model/data/config_chain.yaml")
        experiment.train()

        # check whether log file exists
        log_files = glob.glob(os.path.join("tests/unit/model/data/logs/", "log*"))
        self.assertTrue(log_files)

        # check checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        delete_all_torch_files()

        # let us now try to restart
        # we checked manually in the log file it is restarting from the correct checkpoint
        experiment = Trainer.build_from_yaml("tests/unit/model/data/config_chain.yaml")
        experiment.restart_latest = True
        experiment.num_epochs = 8

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt"
        )
        list_of_lrs_1 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt")
        )

        # Running the experiment for 8 epochs
        experiment = Trainer.build_from_yaml("tests/unit/model/data/config_chain.yaml")
        experiment.num_epochs = 8

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt"
        )
        list_of_lrs_2 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]

        for i in range(len(list_of_lrs_2)):
            self.assertEqual(list_of_lrs_2[i], list_of_lrs_1[i])

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt")
        )

        # delete all directories
        for direc in [
            "tests/unit/model/data/checkpoints",
            "tests/unit/model/data/logs",
            "tests/unit/model/data/tb_log",
        ]:
            shutil.rmtree(direc)

    def test_performs_experiment_correctly_with_and_saves_and_restarts_from_checkpoint_chained_scheduler_per_epoch(
        self,
    ):
        # Starting the experiment for 10 epochs and continuing it for 10
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_per_epoch.yaml"
        )
        experiment.train()

        # check whether log file exists
        log_files = glob.glob(os.path.join("tests/unit/model/data/logs/", "log*"))
        self.assertTrue(log_files)

        # check checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        delete_all_torch_files()

        # let us now try to restart
        # we checked manually in the log file it is restarting from the correct checkpoint
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_per_epoch.yaml"
        )
        experiment.restart_latest = True
        experiment.num_epochs = 5

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt"
        )
        list_of_lrs_1 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        # Running the experiment for 20 epochs
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_per_epoch.yaml"
        )
        experiment.num_epochs = 5

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt"
        )
        list_of_lrs_2 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]

        for i in range(len(list_of_lrs_2)):
            self.assertEqual(list_of_lrs_2[i], list_of_lrs_1[i])

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        # delete all directories
        for direc in [
            "tests/unit/model/data/checkpoints",
            "tests/unit/model/data/logs",
            "tests/unit/model/data/tb_log",
        ]:
            shutil.rmtree(direc)

    def test_performs_experiment_correctly_with_and_saves_and_restarts_from_checkpoint_chained_scheduler_one_scheduler(
        self,
    ):
        # Starting the experiment for 11 epochs and continuing it for 5
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_one_scheduler.yaml"
        )
        experiment.train()

        # check whether log file exists
        log_files = glob.glob(os.path.join("tests/unit/model/data/logs/", "log*"))
        self.assertTrue(log_files)

        # check checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        delete_all_torch_files()

        # let us now try to restart
        # we checked manually in the log file it is restarting from the correct checkpoint
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_one_scheduler.yaml"
        )
        experiment.restart_latest = True
        experiment.num_epochs = 8

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt"
        )

        list_of_lrs_1 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]
        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-7.pt")
        )

        # Running the experiment for 20 epochs
        experiment = Trainer.build_from_yaml(
            "tests/unit/model/data/config_chain_one_scheduler.yaml"
        )
        experiment.num_epochs = 5

        experiment.train()

        # Loading the list_of_lrs
        dictionary_model_opt_lrs = torch.load(
            "tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt"
        )
        list_of_lrs_2 = dictionary_model_opt_lrs["lr_scheduler"]["list_of_lrs"]

        for i in range(len(list_of_lrs_2)):
            self.assertEqual(list_of_lrs_2[i], list_of_lrs_1[i])

        delete_all_torch_files()

        # check new checkpoint exists
        self.assertTrue(
            os.path.exists("tests/unit/model/data/checkpoints/checkpoint_epoch-4.pt")
        )

        # delete all directories
        for direc in [
            "tests/unit/model/data/checkpoints",
            "tests/unit/model/data/logs",
            "tests/unit/model/data/tb_log",
        ]:
            shutil.rmtree(direc)


if __name__ == "__main__":
    unittest.main()
