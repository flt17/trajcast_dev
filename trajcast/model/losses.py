from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from torch.optim import Adam
from trajcast.utils.misc import GLOBAL_DEVICE


class MultiobjectiveLoss(torch.nn.Module):
    cross_angle_loss_atom_vector_custom_dict = {
        "cosine_eps": 1e-6,
        "power_value": 2.0,
        "margin_value": 0.0,
    }

    def __init__(
        self,
        loss_type: dict,
        learnable_weights: Optional[bool] = False,
        dimensions: Optional[list] = [],
        weights: Optional[Union[torch.Tensor, list]] = [],
        fields: Optional[list] = [],
        return_individual_losses: Optional[bool] = False,
    ):
        """For correct usage of the class, check the unittest_utils.py file.

        The loss_type is a dictionary that specifies the different types of losses that can be applied.
        There is a main loss, which can be mse or mae, an angle_loss, which can be of different types (angle_difference, cosine_similarity, cosine_distance),
        and a diagonal_cosine_similarity_loss, which measures the angles between the predicted and the reference vector.

        The other parameters should be self-explanatory."""
        super().__init__()
        if dimensions:
            self.dimensions = dimensions
        elif fields:
            raise NotImplementedError(
                "Will be implemented later. For now pass the dimensions."
            )
        else:
            raise ValueError(
                "Please either specifcy the dimensions of the different fields or pass their name."
            )

        # In the following four lines, we record the types of losses we want to track.
        # Only the main loss is required; cross_angle_loss and diagonal_cosine_similarity_loss are optional.
        self.device = GLOBAL_DEVICE.device
        self.loss_type = loss_type
        self.main_loss = self.loss_type.get("main_loss")
        self.cross_angle_loss = self.loss_type.get("cross_angle_loss")
        self.atom_vector_cosine_similarity_loss = self.loss_type.get(
            "atom_vector_cosine_similarity_loss"
        )
        self.return_individual_losses = return_individual_losses

        if self.main_loss is None:
            raise KeyError("main_loss is needed. Please provide it.")

        # We set the num_weights of the associated losses to one. If cross_angle_loss or diagonal_cosine_similarity_loss are also required,
        # we increase num_weights. Subsequently, we check if we have provided the necessary number of weights for the specified losses.
        # In other words, we verify that num_weights == len(weights).
        num_weights = 1
        if self.cross_angle_loss:
            num_weights += 1
            if (
                set(self.cross_angle_loss.keys()).issubset(
                    set(
                        [
                            "cosine_eps",
                            "power_value",
                            "margin_value",
                            "cross_angle_loss_type",
                        ]
                    )
                )
                is False
            ):
                raise KeyError(
                    "The loss' keys of cross_angle_loss should be any of 'cross_angle_loss_type', 'margin_value', 'cosine_eps', 'power_value'."
                )
            if (
                self.cross_angle_loss.get(
                    "power_value",
                    self.cross_angle_loss_atom_vector_custom_dict["power_value"],
                )
                < 1
            ):
                raise ValueError("power_value should be bigger than one")
            if (
                self.cross_angle_loss.get(
                    "cosine_eps",
                    self.cross_angle_loss_atom_vector_custom_dict["cosine_eps"],
                )
                < 0
            ):
                raise ValueError("cosine_eps should be bigger equal than zero")
        if self.atom_vector_cosine_similarity_loss:
            num_weights += 1
            if (
                set(self.atom_vector_cosine_similarity_loss.keys()).issubset(
                    set(["cosine_eps"])
                )
                is False
            ):
                raise KeyError(
                    "The loss' keys of diagonal_cosine_similarity_loss should be 'cosine_eps'."
                )
            if (
                self.atom_vector_cosine_similarity_loss.get(
                    "cosine_eps",
                    self.cross_angle_loss_atom_vector_custom_dict["cosine_eps"],
                )
                < 0
            ):
                raise ValueError("cosine_eps should be bigger equal than zero")
        if learnable_weights:
            self.raw_weights = torch.nn.Parameter(torch.ones(len(self.dimensions)))
        else:
            if weights and not len(weights) == len(self.dimensions) * num_weights:
                raise TypeError("Number of weights is not correct.")
            else:
                weights = [1.0] * len(self.dimensions) * num_weights
            self.raw_weights = torch.Tensor(weights)

    def forward(
        self,
        predictions: Optional[torch.Tensor] = torch.tensor([]),
        reference: Optional[torch.Tensor] = torch.tensor([]),
        predictions_split: Optional[tuple] = (),
        reference_split: Optional[tuple] = (),
    ) -> torch.Tensor:

        if reference.numel() > 0:
            # split torch tensors into the different parts
            reference_split = torch.split(reference, self.dimensions, dim=1)

        if predictions.numel() > 0:
            predictions_split = torch.split(predictions, self.dimensions, dim=1)

        if not reference_split or not predictions_split:
            raise ValueError(
                "If reference and predictions are not specified please provide the splits directly!"
            )
        # We decide if we want main_loss equal to mae or mse
        if self.main_loss == "mae":
            func = torch.abs
        elif self.main_loss == "mse":
            func = torch.square

        main_loss = torch.stack(
            [
                torch.mean(func(predictions_split[prop] - reference_split[prop]))
                for prop in range(len(predictions_split))
            ]
        ).to(self.device)

        # We decide if angle_loss should be equal to cosine_similarity_ref_vs_pred, angle_difference, cosine_distance_angle,
        # and then we compute the angle loss. For more details check the respective methods.
        cross_angle_loss = 0

        if self.cross_angle_loss:
            if (
                self.cross_angle_loss["cross_angle_loss_type"]
                == "cross_vector_cosine_similarity"
            ):
                cross_angle_loss = self.cross_vector_cosine_similarity(
                    predictions_split, reference_split
                ).to(self.device)
            elif (
                self.cross_angle_loss["cross_angle_loss_type"] == "cross_angle_lp_error"
            ):
                cross_angle_loss = self.cross_angle_lp_error(
                    predictions_split, reference_split
                ).to(self.device)
            elif (
                self.cross_angle_loss["cross_angle_loss_type"]
                == "cross_angle_cosine_similarity"
            ):
                cross_angle_loss = self.cross_angle_cosine_similarity(
                    predictions_split, reference_split
                ).to(self.device)

        # If self.diagonal_cosine_similarity_loss is not None we compute it. For more details check the respective method.
        atom_vector_cos_similarity_loss = 0

        if self.atom_vector_cosine_similarity_loss:
            atom_vector_cos_similarity_loss = self.atom_vector_cosine_similarity(
                predictions_split, reference_split
            ).to(self.device)

        # We concatenate the loss computed so that we can weight them again real_weights.
        if self.cross_angle_loss and self.atom_vector_cosine_similarity_loss:
            losses = torch.concat(
                [main_loss, cross_angle_loss, atom_vector_cos_similarity_loss]
            )
        elif self.cross_angle_loss and not self.atom_vector_cosine_similarity_loss:
            losses = torch.concat([main_loss, cross_angle_loss])
        elif not self.cross_angle_loss and self.atom_vector_cosine_similarity_loss:
            losses = torch.concat([main_loss, atom_vector_cos_similarity_loss])

        else:
            losses = main_loss

        if self.return_individual_losses:
            return losses

        else:
            # make weights positive
            real_weights = torch.exp(self.raw_weights)

            # Normalize the positive weights to sum up to 1
            real_weights = (real_weights / real_weights.sum()).to(self.device)

            self.weights = real_weights

            return torch.sum(losses * real_weights)

    def cross_vector_cosine_similarity(self, predictions_split, reference_split):
        """This method takes prediction split and reference_split and compute the scalar product between the i_th prediction_split element
        and all the references vector in reference_split. In this way all the points should have a similar global behaviour.
        """

        # For all the elments in predictions_split we compute the transpose. (elements in predictions_split can be displacements or velocities)
        transposed_predictions_split = [
            torch.transpose(predictions_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]

        # For each pair of reference and transposed predictions elements we compute the scalar product and place the outcome in a list.
        # E.g. if we have the reference displacement and the prediction displacement transposed we compute it and place in a list.
        dot_product_predictions_split_reference_split = []
        for a, b in zip(reference_split, transposed_predictions_split):
            dot_product_predictions_split_reference_split.append(a @ b)

        # We compute the norms of each prediction and reference elements so that we can rescale the scalar product.
        norm_reference_split = [
            torch.norm(reference_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]
        norm_predictions_split = [
            torch.norm(predictions_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]

        # Here we first compute the cos of the angle between to vectors. Then we compare this consine of the angle between two vectors with 1.
        # We then compute the mean to get the final loss.
        cross_vector_cosine_similarity_loss = torch.stack(
            [
                torch.mean(
                    1.0
                    - dot_product_predictions_split_reference_split[prop]
                    / (
                        torch.mm(
                            norm_reference_split[prop].unsqueeze(1),
                            norm_predictions_split[prop].unsqueeze(0),
                        )
                        + self.cross_angle_loss["cosine_eps"]
                    )
                )
                for prop in range(len(self.dimensions))
            ]
        )
        return cross_vector_cosine_similarity_loss

    def atom_vector_cosine_similarity(self, predictions_split, reference_split):
        """This methods computes the cosine of the angle between the i_th predicted displacement (velocity) and the i_th reference displacement (velocity)."""

        # For all the elments in predictions_split we compute the transpose. (elements in predictions_split can be displacements or velocities)
        transposed_predictions_split = [
            torch.transpose(predictions_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]

        # For each pair of reference and transposed predictions elements we compute the scalar product, we extract the diagonal of this matrix
        # and place the outcome in a list.
        # E.g. if we have the reference displacement and the prediction displacement transposed we compute the scalar product of these two matrices,
        # extract the diagonal, and place it in a list.
        dot_product_predictions_split_reference_split = []
        for a, b in zip(reference_split, transposed_predictions_split):
            dot_product_predictions_split_reference_split.append(torch.diag(a @ b))

        # We compute the norms of each prediction and reference elements so that we can rescale the scalar product.
        norm_reference_split = [
            torch.norm(reference_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]
        norm_predictions_split = [
            torch.norm(predictions_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]

        # Here we first compute the cos of the angle between to vectors. Then we compare this consine of the angle between two vectors with 1.
        # We then compute the mean to get the final loss.
        atom_vector_cosine_similarity_loss = torch.stack(
            [
                torch.sum(
                    1.0
                    - dot_product_predictions_split_reference_split[prop]
                    / (
                        norm_reference_split[prop] * norm_predictions_split[prop]
                        + self.atom_vector_cosine_similarity_loss["cosine_eps"]
                    )
                )
                / (dot_product_predictions_split_reference_split[prop].shape[0])
                for prop in range(len(self.dimensions))
            ]
        )
        return atom_vector_cosine_similarity_loss

    def cross_angle_lp_error(self, predictions_split, reference_split):
        """In this method we compute the angles between all of the predictions elements and all of the reference elements and then we compare them.
        For example we compute the angle between the i-th prediction element and the j-th prediction element, let's call this angle_theta_pred_ij,
        then we compute the angle between the i-th reference element and the j-th reference element, let's call this angle_theta_ref_ij.
        Afterwards we compare via L^p, p>1, loss theta_ref_ij and theta_pred_ij. We repeat this for all ij.
        """

        # For all the elments in predictions_split and reference_split we compute the transpose.
        transposed_predictions_split = [
            torch.transpose(predictions_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]
        transposed_reference_split = [
            torch.transpose(reference_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]

        # We compute the scalar product between predictions_spli and transposed_predictions_split and add the output into a list.
        # We compute the scalar product between reference_split and transposed_reference_split and add the output into a list.
        self_dot_product_prediction_split = []
        self_dot_product_reference_split = []

        for a, b in zip(predictions_split, transposed_predictions_split):
            self_dot_product_prediction_split.append(a @ b)
        for a, b in zip(reference_split, transposed_reference_split):
            self_dot_product_reference_split.append(a @ b)

        # We compute the norms of each prediction and reference elements so that we can rescale the scalar product.
        norm_reference_split = [
            torch.norm(reference_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]
        norm_predictions_split = [
            torch.norm(predictions_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]

        # We rescale the scalar product by the relative normed tensor to obtain the cosine angle.
        cos_angle_difference_predictions = []
        cos_angle_difference_reference = []
        for a, b in zip(self_dot_product_prediction_split, norm_predictions_split):
            cos_angle_difference_predictions.append(
                (
                    a
                    / (
                        torch.mm(b.unsqueeze(1), b.unsqueeze(0))
                        + self.cross_angle_loss["cosine_eps"]
                    )
                )
            )
            # print('max no arccos', torch.max(a/(torch.mm(b.unsqueeze(1), b.unsqueeze(0)) + self.angle_loss['cosine_eps'])))
            # print('min', torch.min(a/(torch.mm(b.unsqueeze(1), b.unsqueeze(0)) + self.angle_loss['cosine_eps'])))
            # print('all', a/(torch.mm(b.unsqueeze(1), b.unsqueeze(0))+ 1e-6))
        for a, b in zip(self_dot_product_reference_split, norm_reference_split):
            cos_angle_difference_reference.append(
                a
                / (
                    torch.mm(b.unsqueeze(1), b.unsqueeze(0))
                    + self.cross_angle_loss["cosine_eps"]
                )
            )

        # print('max', torch.max(angle_difference_predictions[0]))
        # print('min', torch.min(angle_difference_predictions[0]))
        # print('shape', angle_difference_predictions[0].shape)

        # We compute a margin loss or a simple L^p loss between the angle angle_theta_ref_ij and angle_theta_pred_ij
        if (
            self.cross_angle_loss.get(
                "margin_value",
                self.cross_angle_loss_atom_vector_custom_dict["margin_value"],
            )
            > 0
        ):
            cross_angle_loss = [
                torch.sum(
                    torch.max(
                        torch.zeros(cos_angle_difference_predictions[prop].shape),
                        torch.pow(
                            torch.abs(
                                (
                                    cos_angle_difference_predictions[prop]
                                    - cos_angle_difference_reference[prop]
                                )
                            ),
                            self.cross_angle_loss["power_value"],
                        )
                        - torch.FloatTensor([self.cross_angle_loss["margin_value"]]),
                    )
                )
                / (
                    cos_angle_difference_predictions[prop].shape[0] ** 2
                    - cos_angle_difference_predictions[prop].shape[0]
                )
                for prop in range(len(self.dimensions))
            ]
        else:
            cross_angle_loss = [
                torch.sum(
                    torch.pow(
                        torch.abs(
                            (
                                cos_angle_difference_predictions[prop]
                                - cos_angle_difference_reference[prop]
                            )
                        ),
                        self.cross_angle_loss["power_value"],
                    )
                )
                / (
                    cos_angle_difference_predictions[prop].shape[0] ** 2
                    - cos_angle_difference_predictions[prop].shape[0]
                )
                for prop in range(len(self.dimensions))
            ]
        cross_angle_loss = torch.stack(cross_angle_loss)
        return cross_angle_loss

    def cross_angle_cosine_similarity(self, predictions_split, reference_split):
        """In this method we just compute the cosine distance angle between the ith prediction element and the ith reference element"""
        """NOTICE THAT AT THE MOMENT THIS IS NUMERICALLY UNSTABLE"""

        # For all the elments in predictions_split and reference_split we compute the transpose.
        transposed_predictions_split = [
            torch.transpose(predictions_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]
        transposed_reference_split = [
            torch.transpose(reference_split[prop], 1, 0)
            for prop in range(len(self.dimensions))
        ]

        self_dot_product_prediction_split = []
        self_dot_product_reference_split = []

        # We compute the scalar product between predictions_spli and transposed_predictions_split and add the output into a list.
        # We compute the scalar product between reference_split and transposed_reference_split and add the output into a list.
        for a, b in zip(predictions_split, transposed_predictions_split):
            self_dot_product_prediction_split.append(a @ b)
        for a, b in zip(reference_split, transposed_reference_split):
            self_dot_product_reference_split.append(a @ b)

        # We compute the norms of each prediction and reference elements so that we can rescale the scalar product.
        norm_reference_split = [
            torch.norm(reference_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]
        norm_predictions_split = [
            torch.norm(predictions_split[prop], dim=1)
            for prop in range(len(self.dimensions))
        ]

        # We rescale the scalar product by the relative normed tensor to obtain the cosine angle.
        # Afterwards we compute a the cosine of the difference between angle_ref_ij and angle_pred_ij, and compare this value against 1.
        # In this way if the angles are similar the loss is zero. Otherwise it is a positive value.
        angle_difference_predictions = []
        angle_difference_reference = []
        for a, b in zip(self_dot_product_prediction_split, norm_predictions_split):
            angle_difference_predictions.append(
                torch.arccos(
                    torch.clip(
                        a
                        / (
                            torch.mm(b.unsqueeze(1), b.unsqueeze(0))
                            + self.cross_angle_loss["cosine_eps"]
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        for a, b in zip(self_dot_product_reference_split, norm_reference_split):
            angle_difference_reference.append(
                torch.arccos(
                    torch.clip(
                        a
                        / (
                            torch.mm(b.unsqueeze(1), b.unsqueeze(0))
                            + self.cross_angle_loss["cosine_eps"]
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        cross_angle_cosine_distance_loss = [
            torch.sum(
                1
                - torch.cos(
                    torch.abs(
                        angle_difference_predictions[prop]
                        - angle_difference_reference[prop]
                    )
                )
            )
            / (
                angle_difference_predictions[prop].shape[0] ** 2
                - angle_difference_predictions[prop].shape[0]
            )
            for prop in range(len(self.dimensions))
        ]
        cross_angle_cosine_distance_loss = torch.stack(cross_angle_cosine_distance_loss)
        return cross_angle_cosine_distance_loss


class FAMOLoss(torch.nn.Module):
    """This is an implementation of the Fast Adaptive Multitask Optimisation (FAMO) algorithm proposed in:
    B. Liu et al., FAMO: Fast Adaptive Multitask Optimization, 2023, arXiv:2306.03792.
    This class can be instantiated in the training procedure to allow for adaptive weights within the loss function.
    For instance, when aiming to minimize the loss for two vectors, v1 and v2, corresponding displacements and velocities,
    FAMO will adjust the weights w1 and w2 in the loss function over the training run such that the loss in both vectors
    is reduced in a balanced way.
    """

    def __init__(
        self,
        n_tasks: int,
        loss_func: torch.nn.Module,
        device: Optional[torch.device] = torch.device("cpu"),
        min_loss: Optional[Union[torch.Tensor, float]] = 0.0,
        alpha: Optional[float] = 0.025,
        gamma: Optional[float] = 0.001,
        tol: Optional[float] = 1e-8,
        **kwargs,
    ) -> None:
        """Initialises the FAMO Loss.

        Args:
            n_tasks (int): Number of tasks to be tackled in multitask optimisation. In TrajCast this is currently set to 2 as we
                aim to predict displacements and velocities. Note, however, we can also extend it to use losses regarding to the angles
                of the predicted vectors as outlined in MultiObjectiveLoss.
            loss_func (torch.nn.Module): The loss function we use compute the loss per task. We can pass an instance of MultiObjectiveLoss.
            device (Optional[torch.device], optional): On which we are running. Defaults to torch.device("cpu").
            min_loss (Optional[Union[torch.Tensor, float]], optional): Minimum losses achievable for each task. If float is given this is applied to all tasks.
                Alternatively a tensor with lower boundaries for each task can be given. Defaults to 0.0.
            alpha (Optional[float], optional): Corresponds to the learning rate of the Adam optimizer used to update the weights within the
                    loss function. Defaults to 0.025.
            gamma (Optional[float], optional): Corresponds to the weight decay of the Adam optimizer used to update the weights within the
                    loss function. Defaults to 0.001.
            tol (Optional[float], optional): Tolerance to avoid taking the log of zero. Defaults to 1e-8.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.gamma = gamma
        self._tol = tol
        self._device = device

        self._xi = torch.nn.Parameter(
            torch.zeros(n_tasks, requires_grad=True, device=device)
        )
        self._xi_opt = Adam([self._xi], lr=alpha, weight_decay=gamma)

        if isinstance(min_loss, float):
            self._min_loss = min_loss * torch.ones(n_tasks, device=device)
        else:
            self._min_loss = min_loss

        # define how we compute loss
        self._lf = loss_func

        # initialise previous and current loses losses
        self.register_buffer("_prev_losses", None)
        self.register_buffer("_curr_losses", torch.zeros(n_tasks, device=device))

        self._count = 0
        self._mapping = torch.nn.Softmax(-1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Computes the reweighted loss based on individual task losses.

        Args and kwargs correspond to the inputs passed to the loss function. If simply passed as positional arguments,
        we first pass the predictions and then the reference. Alternatively, we can pass the keyword arguments such as
        predictions = predictions, reference = reference. This will compute the loss based on the loss function defined
        in self._lf.

        Returns:
            torch.Tensor: Reweighted loss
        """

        losses = self._lf(*args, **kwargs)

        if self.training:
            self._acc_losses(losses)
        z = self._mapping(self._xi)
        d = losses - self._min_loss + self._tol
        c = 1 / (z / d).sum().detach()
        loss = (c * d.log() * z).sum()
        return loss

    def _acc_losses(self, losses: torch.Tensor) -> None:
        """This function tracks the running average of the loss.

        Args:
            losses (torch.Tensor): Losses of a given iteration (epoch/batch).
        """
        n = self._count
        self._curr_losses = (n * self._curr_losses.detach() + losses) / (n + 1)
        self._count += 1

    def mtl_update(self) -> None:
        """Updates the weights (xi) via a step of the optimizer."""
        if self._prev_losses is not None:
            delta = (self._prev_losses - self._min_loss + self._tol).log() - (
                self._curr_losses - self._min_loss + self._tol
            ).log()
            with torch.enable_grad():
                d = torch.autograd.grad(
                    self._mapping(self._xi), self._xi, grad_outputs=delta.detach()
                )

                self._xi_opt.zero_grad()
                self._xi.grad = d[0]
                self._xi_opt.step()
        self._prev_losses = deepcopy(self._curr_losses.detach())
        self._curr_losses = torch.zeros_like(self._prev_losses)
        self._count = 0

    def save_state(self) -> Dict:
        """Save current state of losses and optimizer.

        Returns:
            Dict: State dictionary.
        """
        return {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": self._xi_opt.state_dict(),
        }

    def load_state(self, state: Dict) -> None:
        """Load previous state or checkpoint and adjust current losses and optimizer state dict accordingly.

        Args:
            state (Dict): State dict.
        """
        self._prev_losses = torch.zeros_like(self._curr_losses, device=self._device)
        self.load_state_dict(state["state_dict"])
        self._xi_opt.load_state_dict(state["optimizer_state_dict"])


class SmoothTchebycheffLoss(torch.nn.Module):
    def __init__(
        self,
        n_tasks: int,
        loss_func: torch.nn.Module,
        device: Optional[torch.device] = torch.device("cpu"),
        min_loss: Optional[Union[torch.Tensor, float]] = 0.0,
        mu: Optional[float] = 0.2,
        weights: Union[Optional[List], None] = None,
        **kwargs,
    ) -> None:
        """Initialises the Smooth Tchebytcheff Loss.

        Args:
            n_tasks (int): Number of tasks to be tackled in multitask optimisation. In TrajCast this is currently set to 2 as we
                aim to predict displacements and velocities. Note, however, we can also extend it to use losses regarding to the angles
                of the predicted vectors as outlined in MultiObjectiveLoss.
            loss_func (torch.nn.Module): The loss function we use compute the loss per task. We can pass an instance of MultiObjectiveLoss.
            device (Optional[torch.device], optional): On which we are running. Defaults to torch.device("cpu").
            min_loss (Optional[Union[torch.Tensor, float]], optional): Minimum losses achievable for each task. If float is given this is applied to all tasks.
                Alternatively a tensor with lower boundaries for each task can be given. Defaults to 0.0.
            mu (Optional[float], optional): Smoothing parameter. 0 matches original Tchebycheff scalarization. Defaults to 0.2.
            weights (Optional[List], optional): Preferences/Weight vector. Should sum up to 1. Defaults to 1/n_tasks to all tasks.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.mu = mu
        self._device = device

        if isinstance(min_loss, float):
            self._min_loss = min_loss * torch.ones(n_tasks, device=device)
        else:
            self._min_loss = min_loss

        if weights is None:
            self.weights = (torch.ones(n_tasks) / n_tasks).to(self._device)
        else:
            self.weights = torch.tensor(weights, device=self._device)
            assert self.weights.sum() == 1, "Weights should sum up to 1"
            assert (
                len(self.weights) == self.n_tasks
            ), "Weights should have the same length of n_tasks"

        self._lf = loss_func

    def forward(self, *args, **kwargs) -> torch.Tensor:
        losses = self._lf(*args, **kwargs)

        u = self.weights * (losses - self._min_loss)
        u_max = u.max()
        v = u - u_max

        loss = u_max + self.mu * torch.exp(v / self.mu).sum().log()

        return loss
