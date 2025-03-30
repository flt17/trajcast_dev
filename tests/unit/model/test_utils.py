import math
import os
import shutil
import subprocess
import unittest

import numpy as np
import torch

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    TENSORBOARD_LOG_ROOT_KEY,
    TENSORBOARD_LOSS_KEY,
    TENSORBOARD_VALIDATION_LOSS_KEY,
    TENSORBOARD_WEIGHT_STATS_KEY,
    TENSORBOARD_WEIGHTS_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.model.losses import MultiobjectiveLoss
from trajcast.model.models import FlexibleModel, TrajCastModel
from trajcast.model.utils import CustomChainedScheduler, TensorBoard

LR_STRATEGIES = {
    "MultiStep": torch.optim.lr_scheduler.MultiStepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "Exponential": torch.optim.lr_scheduler.ExponentialLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "Linear": torch.optim.lr_scheduler.LinearLR,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "Constant": torch.optim.lr_scheduler.LinearLR,
    "ConstantLR": torch.optim.lr_scheduler.LinearLR,
}


class TestTensorBoardInit(unittest.TestCase):
    def test_returns_warning_if_no_log_dir_given(self):
        settings = {TENSORBOARD_LOSS_KEY: True}

        with self.assertWarns(UserWarning):
            TensorBoard(settings=settings)

    def test_returns_error_if_unknown_metric(self):
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir",
            "metric1": {},
        }
        with self.assertRaises(
            KeyError, msg="Unknown metrics in Tensorboard settings!"
        ):
            TensorBoard(settings=settings)
        shutil.rmtree("tests/unit/model/data/log_dir")


class TestTensorBoardUpdate(unittest.TestCase):
    def test_returns_sets_up_tb_correctly_with_loss_only(self):
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir",
            TENSORBOARD_LOSS_KEY: True,
        }
        tb = TensorBoard(settings=settings)
        for i in range(100):
            tb.update(
                epoch=i,
                loss=torch.rand(1).item(),
                maes={"prop1": torch.randn(1).item(), "prop2": torch.randn(1).item()},
            )

        self.assertTrue(os.path.exists("tests/unit/model/data/log_dir/loss_training"))

        subprocess.run(["rm", "-r", "tests/unit/model/data/log_dir"])

    def test_computes_loss_for_validation_set_correctly(self):
        atom_type_mapper = {1: 0, 6: 1}
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir",
            TENSORBOARD_LOSS_KEY: True,
            TENSORBOARD_VALIDATION_LOSS_KEY: {
                "data": {
                    "root": "tests/unit/model/data/forecast_benzene",
                    "name": "validation_set",
                    "files": "benzene_validation_traj.extxyz",
                    "cutoff_radius": 5.0,
                    "atom_type_mapper": atom_type_mapper,
                    "rename": False,
                }
            },
        }
        tb = TensorBoard(settings=settings)
        model = FlexibleModel.build_from_yaml(
            "tests/unit/model/data/forecast_benzene/architecture.yaml"
        )
        criterion_dict = {
            "loss_type": {"main_loss": "mse"},
            "learnable_weights": False,
            "weights": [1, 1],
            "dimensions": [3, 3],
        }
        tb.loss_function = MultiobjectiveLoss(**criterion_dict)
        tb.target_field = "target"
        tb.reference_fields = [DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY]
        for i in range(10):
            tb.update(
                epoch=i,
                model=model,
                loss=torch.rand(1).item(),
                maes={"prop1": torch.randn(1).item(), "prop2": torch.randn(1).item()},
            )

        self.assertTrue(os.path.exists("tests/unit/model/data/log_dir"))
        self.assertTrue(os.path.exists("tests/unit/model/data/log_dir/loss_validation"))
        self.assertTrue(os.path.exists("tests/unit/model/data/log_dir/loss_training"))
        # delte everything again

        subprocess.run(["rm", "-r", "tests/unit/model/data/log_dir"])

        os.remove(
            os.path.join("tests/unit/model/data/forecast_benzene", "pre_filter.pt")
        )
        os.remove(
            os.path.join("tests/unit/model/data/forecast_benzene", "pre_transform.pt")
        )
        os.remove(
            os.path.join(
                "tests/unit/model/data/forecast_benzene",
                "validation_set.pt",
            )
        )

    def test_track_weight_stats(self):
        # Testing access to weights and plots works correctly
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir/weights",
            TENSORBOARD_WEIGHT_STATS_KEY: {
                "every": 5,
                "stats": ["MAX_NORM", "MIN_NORM", "MAX", "MIN", "AVERAGE_NORM"],
            },
        }
        tb = TensorBoard(settings=settings)
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        for epoch in range(1000):
            tb.update(epoch=epoch, model=model)

        # Testing catching error in case the in case one of the stats is mispelled
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir/weights",
            TENSORBOARD_WEIGHT_STATS_KEY: {
                "every": 5,
                "stats": ["MAX_NORM", "MIN_NORM", "AVERA_NORM"],
            },
        }

        tb = TensorBoard(settings=settings)

        with self.assertRaises(ValueError) as context:
            tb.update(epoch=1, model=model)
            self.assertTrue(
                'Stats should be "MAX_NORM", "MIN_NORM", "MAX", "MIN", "AVERAGE_NORM"'
                in str(context.exception)
            )

        subprocess.run(["rm", "-r", "tests/unit/model/data/log_dir"])

    def test_tracking_weights_in_histogram(self):
        # Testing access to weights and plots works correctly
        settings = {
            TENSORBOARD_LOG_ROOT_KEY: "tests/unit/model/data/log_dir",
            TENSORBOARD_WEIGHTS_KEY: {"every": 10},
        }

        tb = TensorBoard(settings=settings)
        file_path = "tests/unit/model/data/config_train_trajcast.yaml"

        model = TrajCastModel.build_from_yaml(file_path)

        for step in range(100):
            tb.update(
                epoch=step,
                model=model,
            )

        subprocess.run(["rm", "-r", "tests/unit/model/data/log_dir"])


class TestScheduler(unittest.TestCase):
    def test_init(self):
        # First Test Check if init works correctly

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [1000, 2001]
        num_epochs = 2
        num_training_steps = 1000

        chained_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        self.assertIsInstance(chained_scheduler, CustomChainedScheduler)

        # Second Test Checks if errors are raised if milestones are uncompatible with num_epochs * num_training_steps
        per_epoch = False
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [100, 200]
        num_epochs = 2
        num_training_steps = 1000

        with self.assertRaises(ValueError) as context:
            CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                num_epochs,
                num_training_steps,
                monitor_lr_scheduler,
                schedulers_list_str,
            )
            self.assertTrue(
                "The last milestones must have a value which is smaller than 2000"
                in str(context.exception)
            )

        # Third Test Checks if errors are raised if milestones are not in ascending order
        per_epoch = False
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
        }
        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [15000, 2000]
        num_epochs = 2
        num_training_steps = 1000

        with self.assertRaises(ValueError) as context:
            CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                num_epochs,
                num_training_steps,
                monitor_lr_scheduler,
                schedulers_list_str,
            )
            self.assertTrue(
                "Milestones should be in ascending order" in str(context.exception)
            )

    def test_init_per_epoch(self):
        # First Test Check if init works correctly with per_epoch = True

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 2},
            "CosineAnnealingLR": {"T_max": 1, "eta_min": 1e-6},
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [2, 4]
        num_epochs = 3
        num_training_steps = 1000
        monitor_lr_scheduler = False

        chained_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        self.assertIsInstance(chained_scheduler, CustomChainedScheduler)

        # Second Test Checks if errors are raised if milestones are uncompatible with num_epochs * num_training_steps
        per_epoch = True
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 2},
            "CosineAnnealingLR": {"T_max": 1, "eta_min": 1e-6},
        }
        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [2, 3]
        num_epochs = 4
        num_training_steps = 1000
        monitor_lr_scheduler = False

        with self.assertRaises(ValueError) as context:
            CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                num_epochs,
                num_training_steps,
                monitor_lr_scheduler,
                schedulers_list_str,
            )
            self.assertTrue(
                "The last milestones must have a value which is smaller than 3000"
                in str(context.exception)
            )

        # Third Test Checks if errors are raised if milestones are not in ascending order
        per_epoch = True
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1},
            "CosineAnnealingLR": {"T_max": 2, "eta_min": 1e-6},
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [3, 1]
        num_epochs = 2
        num_training_steps = 1000
        monitor_lr_scheduler = False

        with self.assertRaises(ValueError) as context:
            CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                num_epochs,
                num_training_steps,
                monitor_lr_scheduler,
                schedulers_list_str,
            )
            self.assertTrue(
                "Milestones should be in ascending order" in str(context.exception)
            )

    def test_exponential_lr(self):
        # Test Exponential LR

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        gamma = 0.95
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["ExponentialLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "ExponentialLR": {
                "gamma": gamma,
            }
        }
        monitor_lr_scheduler = False
        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [2001]
        num_epochs = 2
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = [initial_lr * gamma**i for i in range(2000)]
        fake_val_loss = 0

        for i in range(2000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        for i in range(len(list_lr)):
            self.assertAlmostEqual(list_lr[i], reference_lr[i], places=6)

    def test_exponential_lr_per_epoch(self):
        # Test Exponential LR per epoch

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        gamma = 0.95
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["ExponentialLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "ExponentialLR": {
                "gamma": gamma,
            }
        }
        monitor_lr_scheduler = False
        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [21]
        num_epochs = 20
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = [initial_lr * gamma**i for i in range(20)]
        fake_val_loss = 0

        for i in range(20000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        count_for_reference_lr = 0
        for i in range(len(list_lr)):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_lr[count_for_reference_lr]
                )
                count_for_reference_lr += 1

    def test_linear_lr(self):
        # Test Linear Lr

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        start_factor = 1
        end_factor = 0.01
        total_iters = 2000
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["LinearLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {
                "start_factor": start_factor,
                "end_factor": end_factor,
                "total_iters": total_iters,
            }
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [2001]
        num_epochs = 2
        num_training_steps = 1000

        chained_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = np.linspace(1e-3, 1e-5, 2000)
        fake_val_loss = 0

        for i in range(2000):
            list_lr.append(chained_scheduler.return_lr(optimizer))
            chained_scheduler.step(fake_val_loss)

        for i in range(2000):
            self.assertAlmostEqual(list_lr[i], reference_lr[i], places=6)

    def test_linear_lr_per_epoch(self):
        # Test Linear Lr per_epoch

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        start_factor = 1
        end_factor = 0.01
        total_iters = 20
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["LinearLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {
                "start_factor": start_factor,
                "end_factor": end_factor,
                "total_iters": total_iters,
            }
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [21]
        num_epochs = 20
        num_training_steps = 1000

        chained_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = np.linspace(1e-3, 1e-5, 20)
        fake_val_loss = 0

        for i in range(2000):
            list_lr.append(chained_scheduler.return_lr(optimizer))
            chained_scheduler.step(fake_val_loss)

        count_for_reference_lr = 0
        for i in range(len(list_lr)):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_lr[count_for_reference_lr]
                )
                count_for_reference_lr += 1

    def test_reduce_on_plateau_per_epoch(self):
        # Test reduce_on_plateau

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

        factor = 0.1
        patience = 10
        cooldown = 0
        eps = 1e-8
        schedulers_list_str = ["ReduceLROnPlateau"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "ReduceLROnPlateau": {
                "factor": factor,
                "patience": patience,
                "cooldown": cooldown,
                "eps": eps,
            }
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [21]
        num_epochs = 20
        num_training_steps = 10

        chained_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        x_values_1 = np.linspace(0.0001, 1, 100)[::-1]
        x_values_2 = np.full(101, 1e-4)
        fake_val_loss = np.concatenate((x_values_1, x_values_2))

        for i in range(201):
            chained_scheduler.step(fake_val_loss[i])

        chained_scheduler.return_lr(optimizer)

    def test_cosine_annealing_lr(self):
        # Test Cosine Annealing LR

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        T_max = 2000
        eta_min = 1e-5
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "CosineAnnealingLR": {
                "T_max": T_max,
                "eta_min": eta_min,
            }
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [2001]
        num_epochs = 2
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = [
            eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]
        fake_val_loss = 0

        for i in range(2000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        for i in range(2000):
            self.assertAlmostEqual(list_lr[i], reference_lr[i])

    def test_cosine_annealing_lr_per_epoch(self):
        # Test Cosine Annealing LR per epoch

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        T_max = 20
        eta_min = 1e-5
        initial_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        schedulers_list_str = ["CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "CosineAnnealingLR": {
                "T_max": T_max,
                "eta_min": eta_min,
            }
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [21]
        num_epochs = 20
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )

        list_lr = []
        reference_lr = [
            eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]
        fake_val_loss = 0

        for i in range(20000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        count_for_reference_lr = 0
        for i in range(len(list_lr)):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_lr[count_for_reference_lr]
                )
                count_for_reference_lr += 1

    def test_linear_cosine(self):
        # Test Linear and Cosine

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        eta_min = 1e-6
        T_max = 1000
        eta_max = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
        }
        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [1000, 2001]
        num_epochs = 2
        num_training_steps = 1000

        reference_linear = np.linspace(1e-4, 1e-3, 1001)
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        fake_val_loss = 0

        list_lr = []
        for i in range(2000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        for i in range(1001):
            self.assertAlmostEqual(list_lr[i], reference_linear[i], places=5)

        for i in range(1000):
            self.assertAlmostEqual(list_lr[1000 + i], reference_cosine[i], places=6)

    def test_linear_cosine_per_epoch(self):
        # Test Linear and Cosine per_epoch

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers

        per_epoch = True
        eta_min = 1e-6
        T_max = 15
        eta_max = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 5},
            "CosineAnnealingLR": {"T_max": 15, "eta_min": 1e-6},
        }
        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        monitor_lr_scheduler = False
        milestones = [5, 21]
        num_epochs = 20
        num_training_steps = 1000

        reference_linear = np.linspace(1e-4, 1e-3, 6)
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        fake_val_loss = 0

        list_lr = []
        for i in range(20000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        count_for_reference_lr = 0
        for i in range(6000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_linear[count_for_reference_lr]
                )
                count_for_reference_lr += 1

        count_for_reference_lr = 0
        for i in range(6000, 20000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_cosine[count_for_reference_lr]
                )
                count_for_reference_lr += 1

    def test_linear_cosine_constant(self):
        # Test Linear Cosine Constant

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        eta_min = 1e-6
        T_max = 1000
        eta_max = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": 1000, "eta_min": 1e-6},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 1000},
        }

        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [1000, 2000, 3001]
        num_epochs = 3
        num_training_steps = 1000

        reference_linear = np.linspace(1e-4, 1e-3, 1001)
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]
        reference_constant = [1e-6 for i in range(1000)]

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        fake_val_loss = 0

        list_lr = []
        for i in range(3000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        for i in range(1001):
            self.assertAlmostEqual(list_lr[i], reference_linear[i], places=5)

        for i in range(1000):
            self.assertAlmostEqual(list_lr[1000 + i], reference_cosine[i], places=6)

        for i in range(1000):
            self.assertAlmostEqual(list_lr[2000 + i], reference_constant[i])

    def test_linear_cosine_constant_per_epoch(self):
        # Test Linear Cosine Constant per_epoch

        # load fake model
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        eta_min = 1e-6
        T_max = 10
        eta_max = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 5},
            "CosineAnnealingLR": {"T_max": 10, "eta_min": 1e-6},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 10},
        }

        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [5, 15, 27]
        num_epochs = 26
        num_training_steps = 1000

        reference_linear = np.linspace(1e-4, 1e-3, 6)
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max)
        ]
        reference_constant = [1e-6 for i in range(10)]
        monitor_lr_scheduler = False

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        fake_val_loss = 0

        list_lr = []
        for i in range(26000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        count_for_reference_lr = 0
        for i in range(5000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_linear[count_for_reference_lr]
                )
                count_for_reference_lr += 1

        count_for_reference_lr = 0
        for i in range(5001, 15000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_cosine[count_for_reference_lr]
                )
                count_for_reference_lr += 1

        count_for_reference_lr = 0
        for i in range(15001, 25000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    list_lr[i - 1], reference_constant[count_for_reference_lr]
                )
                count_for_reference_lr += 1

    def test_save_and_load(self):
        # test_save_and_load

        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = False
        eta_min = 1e-6
        eta_max = 1e-3
        T_max = 1000
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": T_max, "eta_min": eta_min},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 1000},
        }

        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [1000, 2000, 3001]
        num_epochs = 3
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max + 1)
        ]
        reference_constant = [1e-6 for i in range(1000)]
        fake_val_loss = 0

        list_lr = []
        for i in range(1500):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        last_lr = chain_scheduler.return_lr(optimizer)
        dictionary_saving = chain_scheduler.save()

        # Re-instantiate and load
        per_epoch = False
        eta_min = 1e-6
        eta_max = 1e-3
        T_max = 1000
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 1000},
            "CosineAnnealingLR": {"T_max": T_max, "eta_min": eta_min},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 1000},
        }

        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [1000, 2000, 3001]
        num_epochs = 3
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        chain_scheduler.load(dictionary_saving)
        first_lr = chain_scheduler.return_lr(optimizer)

        self.assertEqual(last_lr, first_lr)
        self.assertEqual(chain_scheduler.step_count, 1500)
        self.assertEqual(chain_scheduler.current_milestone, 2000)
        self.assertEqual(chain_scheduler.milestones, [3001])
        self.assertEqual(chain_scheduler.current_scheduler_index, 1)
        fake_val_loss = 0

        for i in range(1500):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        for i in range(500):
            self.assertAlmostEqual(
                reference_cosine[500 + i], list_lr[1500 + i], places=3
            )
        for i in range(1000):
            self.assertAlmostEqual(
                reference_constant[i], chain_scheduler.return_lr(optimizer)
            )

    def test_save_and_load_per_epoch(self):
        # test_save_and_load_per_epoch
        file_path = "tests/unit/model/data/config.yaml"

        model = FlexibleModel.build_from_yaml(file_path)

        # create fake optimizers
        per_epoch = True
        eta_min = 1e-6
        eta_max = 1e-3
        T_max = 10
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 10},
            "CosineAnnealingLR": {"T_max": T_max, "eta_min": eta_min},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 5},
        }

        schedulers = []
        monitor_lr_scheduler = False
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [10, 20, 26]
        num_epochs = 25
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        reference_cosine = [
            eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max))
            for i in range(T_max + 1)
        ]
        reference_constant = [1e-6 for i in range(5)]
        fake_val_loss = 0

        list_lr = []
        for i in range(15000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        last_lr = chain_scheduler.return_lr(optimizer)
        dictionary_saving = chain_scheduler.save()

        # Re-instantiate and load
        eta_min = 1e-6
        eta_max = 1e-3
        T_max = 10
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        schedulers_list_str = ["LinearLR", "CosineAnnealingLR", "Constant"]
        schedulers_list = [LR_STRATEGIES[sched] for sched in schedulers_list_str]
        schedulers_settings = {
            "LinearLR": {"start_factor": 0.1, "end_factor": 1, "total_iters": 10},
            "CosineAnnealingLR": {"T_max": T_max, "eta_min": eta_min},
            "Constant": {"start_factor": 1, "end_factor": 1, "total_iters": 5},
        }

        schedulers = []
        for sched_type, sched in zip(schedulers_list_str, schedulers_list):
            schedulers.append(sched(optimizer, **schedulers_settings[sched_type]))

        milestones = [10, 20, 26]
        num_epochs = 25
        num_training_steps = 1000

        chain_scheduler = CustomChainedScheduler(
            per_epoch,
            schedulers,
            milestones,
            num_epochs,
            num_training_steps,
            monitor_lr_scheduler,
            schedulers_list_str,
        )
        chain_scheduler.load(dictionary_saving)
        first_lr = chain_scheduler.return_lr(optimizer)

        self.assertEqual(last_lr, first_lr)
        self.assertEqual(chain_scheduler.step_count, 15000)
        self.assertEqual(chain_scheduler.current_milestone, 20000)
        self.assertEqual(chain_scheduler.milestones, [26000])
        self.assertEqual(chain_scheduler.current_scheduler_index, 1)
        fake_val_loss = 0

        for i in range(10000):
            list_lr.append(chain_scheduler.return_lr(optimizer))
            chain_scheduler.step(fake_val_loss)

        count_for_reference = 5
        for i in range(5000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    reference_cosine[count_for_reference], list_lr[15000 + i], places=3
                )
                count_for_reference += 1
        count_for_reference = 0
        for i in range(5000):
            if i % 1000 == 0 and i != 0:
                self.assertAlmostEqual(
                    reference_constant[count_for_reference], list_lr[20000 + i]
                )
                count_for_reference += 1


if __name__ == "__main__":
    unittest.main()
