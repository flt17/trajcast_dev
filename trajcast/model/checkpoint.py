import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from trajcast.utils.misc import GLOBAL_DEVICE

# This part of the code is heavily influenced by the checkpoint.py file of MACE distributed under the MIT License.
# https://github.com/ACEsuit/mace/blob/main/mace/tools/checkpoint.py


Checkpoint = Dict[str, Dict[str, torch.Tensor]]


@dataclass
class CheckpointState:
    """State of model, optimizer, and scheduler after a given number of epochs."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None  # Optional
    best_loss: Union[torch.Tensor, None] = torch.tensor([])


@dataclass
class CheckpointPathInfo:
    """Object with all information on where the checkpoint should be saved."""

    path: str
    epoch: int


class CheckpointBuilder:
    """Subclass to save and load checkpoints."""

    @staticmethod
    def create_checkpoint(state: CheckpointState) -> Checkpoint:
        if state.lr_scheduler is not None:
            lrs = state.lr_scheduler.save()
        else:
            lrs = {}

        return {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": lrs,
            "best_loss": state.best_loss,
        }

    @staticmethod
    def load_checkpoint(state: CheckpointState, checkpoint: Checkpoint) -> None:
        state.model.load_state_dict(checkpoint["model"])
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("lr_scheduler"):
            state.lr_scheduler.load(checkpoint["lr_scheduler"])
        if checkpoint.get("best_loss"):
            state.best_loss = checkpoint["best_loss"]


class CheckpointIO:
    def __init__(self, directory: str, keep_latest: Optional[bool] = False) -> None:
        """Handles input and output related to checkpoints.

        Args:
            directory (str): Path to directory were checkpoints will be saved.
            keep_latest (Optional[bool], optional): Whether to keep previous checkpoints or to delete them. Defaults to False.
        """
        self.directory = directory
        self.keep_latest = keep_latest

        # define variable for old_path
        self.old_path = None

        # conventions on how to save things
        self._epochs_string = "_epoch-"
        self._filename_extension = "pt"

    def _get_checkpoint_filename(self, epoch: int) -> str:
        return f"checkpoint{self._epochs_string}{epoch}.{self._filename_extension}"

    def _get_checkpoint_path_info(self, path: str) -> CheckpointPathInfo:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file under {path}")

        filename = os.path.basename(path)
        # get epoch
        epoch = int(filename.split(self._epochs_string)[-1].split(".")[0])

        return CheckpointPathInfo(path=path, epoch=epoch)

    def _get_path_to_latest_checkpoint(self) -> str:
        # get all files in the directory
        all_checkpoint_files = glob.glob(os.path.join(self.directory, "checkpoint*"))

        checkpoint_infos = [
            self._get_checkpoint_path_info(file) for file in all_checkpoint_files
        ]

        # find the file with the latest epoch
        epochs = [checkpoint.epoch for checkpoint in checkpoint_infos]

        # get index with largest epoch
        index_max = np.argmax(np.asarray(epochs))

        return checkpoint_infos[index_max].path

    def save(self, checkpoint: Checkpoint, epoch: int) -> None:
        """Saves a checkpoint to a file and handles previous checkpoints if specified.

        Args:
            checkpoint (Checkpoint): The checkpoint to be saved.
            epoch (int): Epoch we are currently in.
        """

        if self.old_path and not self.keep_latest:
            logging.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epoch)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)

        # now update old_path
        self.old_path = path

    def load(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._get_checkpoint_path_info(path)

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epoch,
        )

    def load_latest(
        self, device: Optional[torch.device] = None
    ) -> Optional[Tuple[Checkpoint, int]]:
        path = self._get_path_to_latest_checkpoint()

        if not path:
            return None

        return self.load(path, device=device)


class CheckpointHandler:
    """Object to manage saving and loading of checkpoints."""

    def __init__(self, *args, **kwargs) -> None:
        self.builder = CheckpointBuilder()
        self.io = CheckpointIO(*args, **kwargs)
        self.device = GLOBAL_DEVICE.device

    def save(self, state: CheckpointState, epoch: int) -> None:
        checkpoint = self.builder.create_checkpoint(state=state)
        self.io.save(checkpoint, epoch)

    def load_latest(self, state: CheckpointState) -> Optional[int]:
        """Load most recent produced checkpoint.

        Args:
            state (CheckpointState): State to load the data into.

        Returns:
            int: Returns epoch we are loading from.
        """
        result = self.io.load_latest(device=self.device)
        if result is None:
            return None
        checkpoint, epoch = result
        best_loss = checkpoint.get("best_loss")
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint)
        return epoch, best_loss

    def load(
        self,
        state: CheckpointState,
        path: str,
    ) -> int:
        """Load a checkpoint state under the given path into state.

        Args:
            state (CheckpointState): State to load the data into.
            path (str): Path to checkpoint.

        Returns:
            int: Returns epoch we are loading from.
        """
        checkpoint, epoch = self.io.load(path, device=self.device)
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint)
        return epoch
