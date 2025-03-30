import glob
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    MODEL_TYPE_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.model.checkpoint import CheckpointHandler, CheckpointState
from trajcast.model.losses import MultiobjectiveLoss
from trajcast.model.models import EfficientTrajCastModel, FlexibleModel, TrajCastModel
from trajcast.model.utils import (
    CustomChainedScheduler,
    TensorBoard,
)
from trajcast.utils.misc import GLOBAL_DEVICE, convert_irreps_to_string

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


class Trainer:
    _allowed_train_attrs = [
        "seed",
        "device",
        "target_field",
        "restart_latest",
        "reference_fields",
        "batch_size",
        "max_grad_norm",
        "num_epochs",
        "criterion",
        "optimizer",
        "optimizer_settings",
        "scheduler",
        "scheduler_settings",
        "chained_scheduler_hp",
        "checkpoint_settings",
        "tensorboard_settings",
        "model_type",
    ]

    def __init__(self, config: Dict):
        self.config = config
        train_config = config.get("training")
        # based on config build:
        # - the variables related to training the model
        # we start with this to get the seed for initialising the model
        for key, value in train_config.items():
            if key not in self._allowed_train_attrs:
                raise ValueError(f"Key '{key}' is not allowed.")

        # Reference and target, batchsize, num_epochs
        self.reference_fields = train_config.get("reference_fields")
        self.target_field = train_config.get("target_field")
        self.batch_size = train_config.get("batch_size")
        self.num_epochs = train_config.get("num_epochs")

        if "precision" in self.config["model"]:
            assert self.config["model"]["precision"] in [64, 32]
            if self.config["model"]["precision"] == 64:
                torch.set_default_dtype(torch.float64)
            elif self.config["model"]["precision"] == 32:
                torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float32)

        if train_config.get("device"):
            GLOBAL_DEVICE.device = train_config["device"]

        # set the seed:
        self.device = GLOBAL_DEVICE.device
        self.seed = train_config.get("seed", 42)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        # we also want to make sure we have the argument for restarting
        self.restart_latest = train_config.get("restart_latest", False)

        # and deal with the required checkpoints
        checkpoint_settings = train_config.get("checkpoint_settings")
        if not checkpoint_settings:
            self.checkpoint_settings = {
                "root": os.path.join(os.getcwd(), "checkpoints"),
            }
        else:
            self.checkpoint_settings = checkpoint_settings
            if "root" not in self.checkpoint_settings.keys():
                self.checkpoint_settings["root"] = os.path.join(
                    os.getcwd(), "checkpoints"
                )

        # - the dataset
        self.dataset = AtomicGraphDataset(**self.config["data"])
        # get output_dimensions for reference_fields
        if isinstance(self.reference_fields, list):
            self.output_dimensions = [
                self.dataset[0][prop].size(-1) for prop in self.reference_fields
            ]

        # - the model
        model_type = train_config.get(MODEL_TYPE_KEY, "Flexible")

        if model_type == "Flexible":
            self.model = FlexibleModel(
                config=self.config["model"], predicted_fields=self.reference_fields
            ).to(self.device)

        elif model_type == "TrajCast":
            # compute RMS for normalising
            rms = []
            means = []
            for field in self.reference_fields:
                data = getattr(self.dataset, field)
                rms.append(data.pow(2).mean().sqrt().item())
                means.append(0.0)

            # avg number of neighbors
            if not self.config["model"].get("avg_num_neighbors"):
                self.config["model"]["avg_num_neighbors"] = (
                    torch.tensor(
                        [data.num_edges / data.num_nodes for data in self.dataset],
                        dtype=torch.float32,
                    )
                    .mean()
                    .item()
                )

            self.model = TrajCastModel(
                config=self.config["model"],
                predicted_fields=self.reference_fields,
                rms_targets=rms,
                mean_targets=means,
            ).to(self.device)

        elif model_type == "EfficientTrajCastModel":
            # compute RMS for normalising
            rms = []
            means = []
            for field in self.reference_fields:
                data = getattr(self.dataset, field)

                rms.append(data.pow(2).mean().sqrt().item())
                means.append(0.0)

            # avg number of neighbors
            if not self.config["model"].get("avg_num_neighbors"):
                self.config["model"]["avg_num_neighbors"] = (
                    torch.tensor(
                        [data.num_edges / data.num_nodes for data in self.dataset],
                        dtype=torch.float32,
                    )
                    .mean()
                    .item()
                )

            self.model = EfficientTrajCastModel(
                config=self.config["model"],
                predicted_fields=self.reference_fields,
                rms_targets=rms,
                mean_targets=means,
            ).to(self.device)

        else:
            raise KeyError(f"The chosen model type: {model_type} is not allowed.")

        # Define Loss Function
        criterion = train_config.get("criterion")
        if isinstance(criterion, str):
            self.loss_function = {
                "mse": torch.nn.MSELoss(),
                "mae": torch.nn.L1Loss(),
            }[criterion]
        elif isinstance(criterion, dict):
            criterion["dimensions"] = self.output_dimensions
            self.loss_function = MultiobjectiveLoss(**criterion)

        # Choose optimiser
        optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[train_config.get("optimizer").lower()]
        params = list(self.model.parameters())

        # Setup optimiser
        self.optimizer = optimizer(
            params=params, **train_config.get("optimizer_settings")
        )

        # initialise gradient clipping
        self.max_grad_norm = train_config.get("max_grad_norm", float("inf"))

        # scheduler

        # Check whether tensorboard is available
        if train_config.get("tensorboard_settings"):
            self.tensorboard = TensorBoard(
                settings=train_config.get("tensorboard_settings")
            )
            self.tensorboard.loss_function = self.loss_function
            self.tensorboard.target_field = self.target_field
            self.tensorboard.reference_fields = self.reference_fields
            self.tensorboard.dimensions = self.output_dimensions

        else:
            print("Please provide tensorboard settings with path to validation set.")

    @classmethod
    def build_from_yaml(cls, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Could not find the file under the path {filename}"
            )
        with open(filename, "r") as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return cls(config=dictionary)

    def dump_config_to_yaml(self, filename: Optional[str] = "config.yaml"):
        convert_irreps_to_string(self.config)

        with open(filename, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)

    def create_logger(self, directory: Optional[str] = os.getcwd()):
        """_summary_"""

        logger = logging.getLogger()

        # we are not interested in things below info level
        logger.setLevel(logging.INFO)

        # same format as in MACE: https://github.com/ACEsuit/mace/blob/main/mace/tools/utils.py
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # here we set where to save the log
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(
            directory, f"log_{str(int(datetime.timestamp(datetime.now())))}.txt"
        )

        # now we add this to the logger
        fiha = logging.FileHandler(path)
        fiha.setFormatter(formatter)

        logger.addHandler(fiha)

    def _train_epoch(self, train_loader: DataLoader):
        running_loss = 0.0
        mae_disp = 0.0
        mae_vel = 0.0
        for data_batch in train_loader:
            # Forward pass

            data_batch = self.model(data_batch.to(self.device))

            predictions = data_batch[self.target_field]
            reference = (
                data_batch[self.reference_fields]
                if isinstance(self.reference_fields, str)
                else torch.hstack(
                    [data_batch[field] for field in self.reference_fields]
                )
            )

            # compute loss
            loss = self.loss_function(predictions, reference)

            # compute mae
            err_disp, err_vel = torch.split(
                (predictions - reference).abs(), self.output_dimensions, dim=1
            )

            mae_disp += err_disp.mean() * data_batch.size(0)
            mae_vel += err_vel.mean() * data_batch.size(0)

            # Backward pass and update weights
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2,
                error_if_nonfinite=False,
            )

            self.optimizer.step()

            # Compute and accumluate loss
            running_loss += loss.detach() * data_batch.size(0)

        epoch_loss = running_loss / train_loader.dataset.num_nodes
        # store maes in dictionary
        maes = {}
        maes[DISPLACEMENTS_KEY] = mae_disp.item() / train_loader.dataset.num_nodes
        maes[UPDATE_VELOCITIES_KEY] = mae_vel.item() / train_loader.dataset.num_nodes
        return epoch_loss, maes

    def train(self):
        # setup the logger
        log_dir = os.path.join(os.path.dirname(self.tensorboard.log_dir), "logs")

        self.create_logger(directory=log_dir)
        logging.info("You are using TrajCast.")

        # Create DataLoader
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        logging.info(f"Using {len(self.dataset)} configurations for training.")
        logging.info(
            f"Training files are stored under root: {self.config['data']['root']} with the filenames: {self.config['data']['files']}."
        )

        if hasattr(self.model, "layers"):
            logging.info(f"Model architecture: {self.model.layers}")
        else:
            logging.info(f"Model architecture: {self.model.config}")
        logging.info(f"Precision {torch.get_default_dtype()}")
        if self.model.o3_backend == "cueq":
            logging.info("Running with cuequivariance as o3 backend.")

        # Check whether scheduler is available
        lr_scheduler = None
        scheduler = self.config["training"].get("scheduler")
        scheduler_settings = self.config["training"].get("scheduler_settings")
        chained_scheduler_hp = self.config["training"].get("chained_scheduler_hp")

        if scheduler:
            # Create list of the schedulers
            scheduler_list = [scheduler] if isinstance(scheduler, str) else scheduler
            # Check scheduler settings correpond to
            if scheduler_settings:
                if not set(scheduler_list).issubset(set(scheduler_settings.keys())):
                    raise KeyError(
                        "The keys in the scheduler_settings dictionary should correspond with the declared schedulers."
                    )
            else:
                raise TypeError(
                    "Scheduler attribute is present but scheduler_settings attribute is absent."
                )

            if chained_scheduler_hp:
                milestones = chained_scheduler_hp["milestones"]
                per_epoch = chained_scheduler_hp.get("per_epoch", True)
                monitor_lr_scheduler = chained_scheduler_hp.get(
                    "monitor_lr_scheduler", False
                )
            else:
                raise KeyError(
                    "chaned_scheduler_hp is not specified. Please specify it."
                )

            schedulers = []
            for sched in scheduler_list:
                scheduler = LR_STRATEGIES[sched]
                scheduler_params = scheduler_settings.get(sched, {})
                schedulers.append(scheduler(self.optimizer, **scheduler_params))

            lr_scheduler = CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                self.num_epochs,
                len(data_loader),
                monitor_lr_scheduler,
                scheduler_list,
            )

        logging.info(
            f"We are using the following training parameters: {self.config['training']}"
        )

        # Training loop
        # get params (from MACE: https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py)
        n_params = int(sum(np.prod(p.shape) for p in self.model.parameters()))
        logging.info(f"Number of model parameters: {n_params}")
        logging.info("Started training.")

        # initialise the checkpoint handler
        checkpoint_handler = CheckpointHandler(
            directory=self.checkpoint_settings["root"],
            keep_latest=self.checkpoint_settings.get("keep_latest", False),
        )
        checkpoint_interval = self.checkpoint_settings.get("interval", 1)

        # init the starting epoch
        start_epoch = 0
        best_loss = None

        # restart from latest checkpoint in case this is desired
        if self.restart_latest and os.path.exists(self.checkpoint_settings["root"]):

            restart_epoch, best_loss = checkpoint_handler.load_latest(
                state=CheckpointState(
                    self.model, self.optimizer, lr_scheduler, best_loss
                )
            )

            if restart_epoch is not None:
                start_epoch = restart_epoch + 1
                logging.info(
                    f"Restarting from latest checkpoint in epoch: {start_epoch}. Current best loss is {best_loss}"
                )

        epoch = start_epoch
        loss_val = 0.0

        os.makedirs(self.checkpoint_settings["root"], exist_ok=True)

        while epoch < self.num_epochs:

            loss_train, maes_train = self._train_epoch(data_loader)

            if lr_scheduler is not None:
                lr_rate = lr_scheduler.return_lr(self.optimizer)
            else:
                lr_rate = self.config["training"].get("optimizer_settings")["lr"]

            loss_val = self.tensorboard.update(
                epoch=epoch,
                loss=loss_train.item(),
                model=self.model,
                lr=lr_rate,
                maes=maes_train,
            )

            # if scheduler is set update learning rate
            if lr_scheduler:
                lr_scheduler.step(loss_val)

            # report loss
            logging.info(
                f"Epoch {epoch}: train_loss={loss_train}; \t val_loss={loss_val}."
            )

            # In the first epoch and checkpoint has no best_loss
            if epoch == start_epoch and not best_loss:
                best_loss = loss_val

            if loss_val and loss_val <= best_loss:
                best_loss = loss_val

                logging.info("Saving new best model!")

                # Delete old one if exists
                old_best_path = glob.glob(
                    os.path.join(self.checkpoint_settings["root"], "best*")
                )
                if old_best_path:
                    os.remove(old_best_path[0])

                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.checkpoint_settings["root"], f"best_model_epoch-{epoch}.pt"
                    ),
                )

            # generate checkpoints and save loss to logging
            if epoch % checkpoint_interval == 0:

                best_loss = 0 if loss_val is None else best_loss

                # save checkpoints
                checkpoint_handler.save(
                    state=CheckpointState(
                        self.model, self.optimizer, lr_scheduler, best_loss
                    ),
                    epoch=epoch,
                )

            epoch += 1

        logging.info("Training done.")
        path_to_model = os.path.join(os.path.dirname(log_dir), "model_params.pt")
        torch.save(self.model.state_dict(), path_to_model)
        logging.info(f"Final model saved to {path_to_model}.")
