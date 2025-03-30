import unittest

import numpy as np
import torch.nn
from torch_scatter import scatter

from trajcast.model.losses import (
    FAMOLoss,
    MultiobjectiveLoss,
    SmoothTchebycheffLoss,
)


class TestMultiObjectiveLoss(unittest.TestCase):
    def test_init(self):
        # This test is used to check if we are able to instantiate the multiloss class with some given values, and without atom_vector_cosine_similarity_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        self.assertIsInstance(mol, MultiobjectiveLoss)

        # This test is used to check if we are able to instantiate the multiloss class with some given values.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        self.assertIsInstance(mol, MultiobjectiveLoss)

        # This test is used to check if we are able to instantiate the multiloss class using also default values.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        self.assertIsInstance(mol, MultiobjectiveLoss)

        # This test is used to check if we are able to instantiate the multiloss class using just atom_vector_cosine_similarity_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": None,
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        self.assertIsInstance(mol, MultiobjectiveLoss)

        # This test is used to check if we are able to instantiate the multiloss class using just main_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": None,
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        self.assertIsInstance(mol, MultiobjectiveLoss)

    def test_raise_errors(self):
        # In this test we check if we rise a error in case we do not specify the right amount of weights.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        with self.assertRaises(TypeError, msg="Number of weights is not correct."):
            MultiobjectiveLoss(**dictionary_loss)

        # In this test we check if we rise a error in case we do not specify a valid epsilon value for cross_angle_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": -0.1,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        with self.assertRaises(
            ValueError, msg="cosine_eps should be bigger equal than zero"
        ):
            MultiobjectiveLoss(**dictionary_loss)

        # In this test we check if we rise a error in case we do not specify the right p power value.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 0.2,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        with self.assertRaises(ValueError, msg="power_value should be bigger than one"):
            MultiobjectiveLoss(**dictionary_loss)

        # In this test we check if we rise a error in case we do not specify the epsilon value for atom_vector_cosine_similairty_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": -0.1},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        with self.assertRaises(
            ValueError, msg="cosine_eps should be bigger equal than zero."
        ):
            MultiobjectiveLoss(**dictionary_loss)

    def test_compute_loss(self):
        # In this test we check that cross_angle_lp_error loss works correctly.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor
                        - mock_dot_product_reference_tensor
                    ),
                    2,
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2),
            np.asarray(mock_cross_angle_lp_error),
            5,
        )

        # In this test we check that cross_angle_lp_error loss works correctly when we use as main_loss mae
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mae",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        loss_mae = torch.mean(torch.abs(mock_prediction_tensor - mock_reference_tensor))
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor
                        - mock_dot_product_reference_tensor
                    ),
                    2,
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mae * 0.5) * 2),
            np.asarray(mock_cross_angle_lp_error),
            5,
        )

        # In this test we check that cross_angle_lp_error loss works correctly when we also use atom_vector_cosine_similarity_loss.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        diagonal_scalar_product_loss = torch.diag(
            mock_prediction_tensor @ torch.transpose(mock_reference_tensor, 1, 0)
        )
        diagonal_scalar_product_loss = (
            torch.sum(
                1
                - diagonal_scalar_product_loss
                / (mock_prediction_norm_tensor * mock_reference_norm_tensor + 1e-6)
            )
            / 2
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor
                        - mock_dot_product_reference_tensor
                    ),
                    2,
                )
            )
            / 2
        )
        mock_loss = torch.sum(
            0.3333
            * torch.concat(
                [
                    torch.stack([loss_mse]),
                    torch.stack([mock_cross_angle_lp_error]),
                    torch.stack([diagonal_scalar_product_loss]),
                ]
            )
        )
        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(np.asarray(loss), np.asarray(mock_loss), 3)

        # In this test we check that cross_angle_lp_error loss works correctly when we also margin_loss
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.01,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.max(
                    torch.zeros(2, 2),
                    torch.pow(
                        (
                            mock_dot_product_prediction_tensor
                            - mock_dot_product_reference_tensor
                        )
                        / 1.0,
                        2,
                    )
                    - 0.01,
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray(loss - loss_mse * 0.5) * 2,
            np.asarray(mock_cross_angle_lp_error),
            3,
        )

        # In this test we check that cross_vector_cosine_similarity_loss
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_vector_cosine_similarity",
                    "cosine_eps": 1e-6,
                    "power_value": 1.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        mock_dot_product = mock_reference_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        cosine_mock_loss = torch.mean(
            1
            - mock_dot_product
            / (
                torch.mm(
                    mock_reference_norm_tensor.unsqueeze(1),
                    mock_prediction_norm_tensor.unsqueeze(0),
                )
                + 1e-6
            )
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2), np.asarray(cosine_mock_loss), 5
        )

        # In this test we check that cross_angle_cosine_similarity_loss. Notice that right now this runs but on the long run it is numerically instable.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_cosine_similarity",
                    "cosine_eps": 1e-6,
                    "power_value": 2,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [2],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor([[2, 4], [4, 8]])
        mock_reference_tensor = torch.FloatTensor(([[1, 3], [5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(torch.FloatTensor([20, 80]))
        mock_reference_norm_tensor = torch.sqrt(torch.FloatTensor([10, 50]))
        mock_dot_product_prediction_tensor = torch.arccos(
            mock_dot_product_prediction_tensor
            / (
                torch.mm(
                    mock_prediction_norm_tensor.unsqueeze(1),
                    mock_prediction_norm_tensor.unsqueeze(0),
                )
                + 1e-6
            )
        )
        mock_dot_product_reference_tensor = torch.arccos(
            mock_dot_product_reference_tensor
            / (
                torch.mm(
                    mock_reference_norm_tensor.unsqueeze(1),
                    mock_reference_norm_tensor.unsqueeze(0),
                )
                + 1e-6
            )
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                1
                - torch.cos(
                    torch.abs(
                        (
                            mock_dot_product_prediction_tensor
                            - mock_dot_product_reference_tensor
                        )
                        / 1.0
                    )
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2),
            np.asarray(mock_cross_angle_lp_error),
            5,
        )

    def test_compute_loss_3d(self):
        # In this test we check corr_angle_lp_error for 3d data.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [3],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor(
            [
                [2, 4, 5],
                [4, 8, 9],
            ]
        )
        mock_reference_tensor = torch.FloatTensor(([[1, 3, 2], [5, 5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(
            torch.FloatTensor([2**2 + 4**2 + 5**2, 4**2 + 8**2 + 9**2])
        )
        mock_reference_norm_tensor = torch.sqrt(
            torch.FloatTensor([1**2 + 3**2 + 2**2, 3 * 5**2])
        )
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor
                        - mock_dot_product_reference_tensor
                    ),
                    2,
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2),
            np.asarray(mock_cross_angle_lp_error),
            5,
        )

        # In this test we check cross_angle_cosine_similarity for 3d data.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_cosine_similarity",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [3],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor(
            [
                [2, 4, 5],
                [4, 8, 9],
            ]
        )
        mock_reference_tensor = torch.FloatTensor(([[1, 3, 2], [5, 5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(
            torch.FloatTensor([2**2 + 4**2 + 5**2, 4**2 + 8**2 + 9**2])
        )
        mock_reference_norm_tensor = torch.sqrt(
            torch.FloatTensor([1**2 + 3**2 + 2**2, 3 * 5**2])
        )
        mock_dot_product_prediction_tensor = torch.arccos(
            mock_dot_product_prediction_tensor
            / (
                torch.mm(
                    mock_prediction_norm_tensor.unsqueeze(1),
                    mock_prediction_norm_tensor.unsqueeze(0),
                )
                + 1e-6
            )
        )
        mock_dot_product_reference_tensor = torch.arccos(mock_dot_product_reference_tensor / (torch.mm(mock_reference_norm_tensor.unsqueeze(1), mock_reference_norm_tensor.unsqueeze(0),)+ 1e-5))

        mock_cross_angle_lp_error = (
            torch.sum(
                1
                - torch.cos(
                    torch.abs(
                        (
                            mock_dot_product_prediction_tensor
                            - mock_dot_product_reference_tensor
                        )
                        / 1.0
                    )
                )
            )
            / 2
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2),
            np.asarray(mock_cross_angle_lp_error),
            5,
        )

        # In this test we check cross_vector_cosine_similarity for 3d data.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_vector_cosine_similarity",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [3],
            "weights": [1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor(
            [
                [2, 4, 5],
                [4, 8, 9],
            ]
        )
        mock_reference_tensor = torch.FloatTensor(([[1, 3, 2], [5, 5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product = mock_reference_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(
            torch.FloatTensor([2**2 + 4**2 + 5**2, 4**2 + 8**2 + 9**2])
        )
        mock_reference_norm_tensor = torch.sqrt(
            torch.FloatTensor([1**2 + 3**2 + 2**2, 3 * 5**2])
        )
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        cosine_mock_loss = torch.mean(
            1
            - mock_dot_product
            / (
                torch.mm(
                    mock_reference_norm_tensor.unsqueeze(1),
                    mock_prediction_norm_tensor.unsqueeze(0),
                )
                + 1e-6
            )
        )

        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(
            np.asarray((loss - loss_mse * 0.5) * 2), np.asarray(cosine_mock_loss), 5
        )

        # In this test we check atom_vector_cosine_similarity_loss with cross_angle_lp_error for 3d data.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": {"cosine_eps": 1e-6},
            },
            "learnable_weights": False,
            "dimensions": [3],
            "weights": [1, 1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor = torch.FloatTensor(
            [
                [2, 4, 5],
                [4, 8, 9],
            ]
        )
        mock_reference_tensor = torch.FloatTensor(([[1, 3, 2], [5, 5, 5]]))
        loss_mse = torch.mean(
            torch.square(mock_prediction_tensor - mock_reference_tensor)
        )
        mock_dot_product_prediction_tensor = mock_prediction_tensor @ torch.transpose(
            mock_prediction_tensor, 1, 0
        )
        mock_dot_product_reference_tensor = mock_reference_tensor @ torch.transpose(
            mock_reference_tensor, 1, 0
        )
        mock_prediction_norm_tensor = torch.sqrt(
            torch.FloatTensor([2**2 + 4**2 + 5**2, 4**2 + 8**2 + 9**2])
        )
        mock_reference_norm_tensor = torch.sqrt(
            torch.FloatTensor([1**2 + 3**2 + 2**2, 3 * 5**2])
        )
        mock_dot_product_prediction_tensor = mock_dot_product_prediction_tensor / (
            torch.mm(
                mock_prediction_norm_tensor.unsqueeze(1),
                mock_prediction_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )
        mock_dot_product_reference_tensor = mock_dot_product_reference_tensor / (
            torch.mm(
                mock_reference_norm_tensor.unsqueeze(1),
                mock_reference_norm_tensor.unsqueeze(0),
            )
            + 1e-6
        )

        diagonal_scalar_product_loss = torch.diag(
            mock_prediction_tensor @ torch.transpose(mock_reference_tensor, 1, 0)
        )
        diagonal_scalar_product_loss = (
            torch.sum(
                1
                - diagonal_scalar_product_loss
                / (mock_prediction_norm_tensor * mock_reference_norm_tensor + 1e-6)
            )
            / 2
        )

        mock_cross_angle_lp_error = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor
                        - mock_dot_product_reference_tensor
                    ),
                    2,
                )
            )
            / 2
        )
        mock_loss = torch.sum(
            0.3333
            * torch.concat(
                [
                    torch.stack([loss_mse]),
                    torch.stack([mock_cross_angle_lp_error]),
                    torch.stack([diagonal_scalar_product_loss]),
                ]
            )
        )
        loss = mol(mock_prediction_tensor, mock_reference_tensor)

        self.assertAlmostEqual(np.asarray(loss), np.asarray(mock_loss), 3)

    def test_compute_loss_displacements_and_velocities(self):
        # In this test we check multiobjective loss for displacements and velocities.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
                "cross_angle_loss": {
                    "cross_angle_loss_type": "cross_angle_lp_error",
                    "cosine_eps": 1e-6,
                    "power_value": 2.0,
                    "margin_value": 0.0,
                },
                "atom_vector_cosine_similarity_loss": None,
            },
            "learnable_weights": False,
            "dimensions": [3, 3],
            "weights": [1, 1, 1, 1],
        }

        mol = MultiobjectiveLoss(**dictionary_loss)

        mock_prediction_tensor_displacements = torch.FloatTensor(
            [
                [2, 4, 5],
                [4, 8, 9],
            ]
        )
        mock_prediction_tensor_velocities = torch.FloatTensor(
            [[0.5, 0.2, 0.3], [4, 2, 3]]
        )
        mock_prediction = torch.concat(
            [mock_prediction_tensor_displacements, mock_prediction_tensor_velocities],
            axis=1,
        )
        mock_reference_tensor_displacements = torch.FloatTensor(
            ([[1, 3, 2], [5, 5, 5]])
        )
        mock_reference_tensor_velocities = torch.FloatTensor(
            [[0.7, 0.1, 0.3], [8, 2, 3]]
        )
        mock_reference = torch.concat(
            [mock_reference_tensor_displacements, mock_reference_tensor_velocities],
            axis=1,
        )
        loss_mse_displacements = torch.mean(
            torch.square(
                mock_prediction_tensor_displacements
                - mock_reference_tensor_displacements
            )
        )
        loss_mse_velocities = torch.mean(
            torch.square(
                mock_prediction_tensor_velocities - mock_reference_tensor_velocities
            )
        )
        mock_dot_product_prediction_tensor_displacements = (
            mock_prediction_tensor_displacements
            @ torch.transpose(mock_prediction_tensor_displacements, 1, 0)
        )
        mock_dot_product_reference_tensor_displacements = (
            mock_reference_tensor_displacements
            @ torch.transpose(mock_reference_tensor_displacements, 1, 0)
        )
        mock_dot_product_prediction_tensor_velocities = (
            mock_prediction_tensor_velocities
            @ torch.transpose(mock_prediction_tensor_velocities, 1, 0)
        )
        mock_dot_product_reference_tensor_velocities = (
            mock_reference_tensor_velocities
            @ torch.transpose(mock_reference_tensor_velocities, 1, 0)
        )
        mock_prediction_norm_tensor_displacements = torch.sqrt(
            torch.FloatTensor([2**2 + 4**2 + 5**2, 4**2 + 8**2 + 9**2])
        )
        mock_reference_norm_tensor_displacements = torch.sqrt(
            torch.FloatTensor([1**2 + 3**2 + 2**2, 3 * 5**2])
        )
        mock_prediction_norm_tensor_velocities = torch.sqrt(
            torch.FloatTensor([0.5**2 + 0.2**2 + 0.3**2, 4**2 + 2**2 + 3**2])
        )
        mock_reference_norm_tensor_velocities = torch.sqrt(
            torch.FloatTensor([0.7**2 + 0.1**2 + 0.3**2, 8**2 + 2**2 + 3**2])
        )

        mock_dot_product_prediction_tensor_displacements = (
            mock_dot_product_prediction_tensor_displacements
            / (
                torch.mm(
                    mock_prediction_norm_tensor_displacements.unsqueeze(1),
                    mock_prediction_norm_tensor_displacements.unsqueeze(0),
                )
                + 1e-6
            )
        )
        mock_dot_product_reference_tensor_displacements = (
            mock_dot_product_reference_tensor_displacements
            / (
                torch.mm(
                    mock_reference_norm_tensor_displacements.unsqueeze(1),
                    mock_reference_norm_tensor_displacements.unsqueeze(0),
                )
                + 1e-6
            )
        )
        mock_dot_product_prediction_tensor_velocities = (
            mock_dot_product_prediction_tensor_velocities
            / (
                torch.mm(
                    mock_prediction_norm_tensor_velocities.unsqueeze(1),
                    mock_prediction_norm_tensor_velocities.unsqueeze(0),
                )
                + 1e-6
            )
        )
        mock_dot_product_reference_tensor_velocities = (
            mock_dot_product_reference_tensor_velocities
            / (
                torch.mm(
                    mock_reference_norm_tensor_velocities.unsqueeze(1),
                    mock_reference_norm_tensor_velocities.unsqueeze(0),
                )
                + 1e-6
            )
        )

        mock_cross_angle_lp_error_displacements = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_prediction_tensor_displacements
                        - mock_dot_product_reference_tensor_displacements
                    ),
                    2,
                )
            )
            / 2
        )
        mock_cross_angle_lp_error_velocities = (
            torch.sum(
                torch.pow(
                    (
                        mock_dot_product_reference_tensor_velocities
                        - mock_dot_product_reference_tensor_velocities
                    ),
                    2,
                )
            )
            / 2
        )

        mock_loss = (
            loss_mse_displacements * 0.25
            + loss_mse_velocities * 0.25
            + 0.25 * mock_cross_angle_lp_error_displacements
            + 0.25 * mock_cross_angle_lp_error_velocities
        )

        loss = mol(mock_prediction, mock_reference)
        self.assertAlmostEqual(np.asarray(loss), np.asarray(mock_loss), 5)

    def test_returns_correct_loss_for_energies_with_node_properties(self):
        # In this test we check multiobjective loss for displacements and velocities.
        dictionary_loss = {
            "loss_type": {
                "main_loss": "mse",
            },
            "learnable_weights": False,
            "dimensions": [1, 3, 3],
            "weights": [1, 1, 1],
            "return_individual_losses": True,
        }
        l_fn = MultiobjectiveLoss(**dictionary_loss)

        # generate pseudo data
        # this is saved in target
        predictions = torch.randn(10, 7)
        batch = torch.ones(10).to(torch.long)
        batch[0:6] = 0

        # for reference we have to take into account that energies are global/graph properties
        global_ref = torch.randn(2)  # batchsize of 2
        node_ref = torch.randn(10, 6)
        reference_split = tuple([global_ref, node_ref[:, 0:3], node_ref[:, 3:]])

        # split into local and global
        node_predictions = predictions[:, 1:]
        global_predictions = scatter(predictions[:, 0], index=batch, reduce="sum")

        # generate splits
        node_splits = torch.split(node_predictions, [3] * 2, dim=1)

        # generate splits
        predictions_split = (global_predictions,) + node_splits

        loss = l_fn(
            predictions_split=predictions_split, reference_split=reference_split
        )
        self.assertTrue(loss.size(0) == 3)

        # by hand
        mse_global = (global_predictions - global_ref).pow(2).mean()
        self.assertTrue(mse_global == loss[0])

        mse_node1 = (node_splits[0] - reference_split[1]).pow(2).mean()
        self.assertTrue(mse_node1 == loss[1])

        mse_node2 = (node_splits[1] - reference_split[2]).pow(2).mean()
        self.assertTrue(mse_node2 == loss[2])


class TestFamoInit(unittest.TestCase):
    def test_returns_famo_initialises_correctly_with_torch_mse_loss(self):
        famo_loss = FAMOLoss(n_tasks=2, loss_func=torch.nn.MSELoss())

        # check instances are okay
        self.assertTrue(isinstance(famo_loss, torch.nn.Module))
        self.assertTrue(isinstance(famo_loss._lf, torch.nn.MSELoss))

        # check initial losses are okay
        self.assertTrue(torch.all(famo_loss._curr_losses == 0))
        self.assertIsNone(famo_loss._prev_losses)

    def test_returns_famo_initialises_correctly_with_mse_from_MultiObjectiveLoss(self):
        famo_loss = FAMOLoss(
            n_tasks=2,
            loss_func=MultiobjectiveLoss(
                loss_type={"main_loss": "mse"}, dimensions=[3, 3]
            ),
        )

        # check instances are okay
        self.assertTrue(isinstance(famo_loss, torch.nn.Module))
        self.assertTrue(isinstance(famo_loss._lf, MultiobjectiveLoss))

        # check initial losses are okay
        self.assertTrue(torch.all(famo_loss._curr_losses == 0))
        self.assertIsNone(famo_loss._prev_losses)


class TestFamoForward(unittest.TestCase):
    def test_returns_famo_correctly_weights_loss_with_MultiObjectiveLoss(self):
        loss_func = MultiobjectiveLoss(
            loss_type={"main_loss": "mse"},
            dimensions=[3, 3],
            return_individual_losses=True,
        )
        famo_loss = FAMOLoss(n_tasks=2, loss_func=loss_func)

        # get some random targets and predictions
        reference = torch.randn(10, 6)
        predictions = torch.randn(10, 6)

        # compute original loss function
        losses_org = loss_func(predictions, reference)
        self.assertTrue(losses_org.numel() == 2)

        # reweighting now
        losses_new = famo_loss(predictions, reference)
        self.assertTrue(losses_new.numel() == 1)
        self.assertIsNotNone(losses_new.grad_fn)


class TestFamoUpdate(unittest.TestCase):
    def test_returns_famo_correctly_updates_weights_of_tasks_linear_model(self):
        loss_func = MultiobjectiveLoss(
            loss_type={"main_loss": "mse"},
            dimensions=[3, 3],
            return_individual_losses=True,
        )
        famo_loss = FAMOLoss(n_tasks=2, loss_func=loss_func)

        # define toy model
        model = torch.nn.Linear(10, 6)

        # optimiser
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params=params, lr=0.01)

        # toy training data
        input = torch.randn(100, 10)
        reference = torch.randn(100, 6)

        # let's train for 10 epochs
        n_epochs = 10
        epoch = 0

        # track losses
        losses_unb = []
        losses_famo = []

        while epoch < n_epochs:
            predictions = model(input)
            # loss_equal_weighting
            loss_equal = loss_func(predictions, reference).mean()
            # famo loss
            loss = famo_loss(predictions, reference)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            famo_loss.mtl_update()

            losses_unb.append(loss_equal.detach().item())
            losses_famo.append(loss.detach().item())
            epoch += 1

        # let us make sure the last previous losses are stored correctly
        self.assertTrue(torch.all(loss_equal == famo_loss._prev_losses.mean()))


class TestSmoothTchebycheffInit(unittest.TestCase):
    def test_returns_smooth_tchebycheff_initialises_correctly_with_torch_mse_loss(self):
        loss = SmoothTchebycheffLoss(n_tasks=2, loss_func=torch.nn.MSELoss())

        # check instances are okay
        self.assertTrue(isinstance(loss, torch.nn.Module))
        self.assertTrue(isinstance(loss._lf, torch.nn.MSELoss))

    def test_returns_famo_initialises_correctly_with_mse_from_MultiObjectiveLoss(self):
        loss = SmoothTchebycheffLoss(
            n_tasks=2,
            loss_func=MultiobjectiveLoss(
                loss_type={"main_loss": "mse"}, dimensions=[3, 3]
            ),
        )

        # check instances are okay
        self.assertTrue(isinstance(loss, torch.nn.Module))
        self.assertTrue(isinstance(loss._lf, MultiobjectiveLoss))


class TestSmoothTchebycheffForward(unittest.TestCase):
    def test_returns_smooth_tchebycheff_correctly_weights_loss_with_MultiObjectiveLoss(
        self,
    ):
        loss_func = MultiobjectiveLoss(
            loss_type={"main_loss": "mse"},
            dimensions=[3, 3],
            return_individual_losses=True,
        )
        loss = SmoothTchebycheffLoss(n_tasks=2, loss_func=loss_func)

        # get some random targets and predictions
        reference = torch.randn(10, 6)
        predictions = torch.randn(10, 6)
        predictions.requires_grad_(True)

        # compute original loss function
        losses_org = loss_func(predictions, reference)
        self.assertTrue(losses_org.numel() == 2)

        # reweighting now
        losses_new = loss(predictions, reference)
        self.assertTrue(losses_new.numel() == 1)
        self.assertIsNotNone(losses_new.grad_fn)


if __name__ == "__main__":
    unittest.main()
