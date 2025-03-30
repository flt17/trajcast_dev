import unittest

import torch

from tests.unit.nn.test_modules import CH3SCH3
from trajcast.utils.atomic_computes import (
    compute_angular_momentum_for_individual_state,
    compute_com_velocity_for_individual_state,
    remove_angular_momentum_for_individual_state,
    remove_linear_momentum_for_individual_state,
)


class TestRemoveLinearMomentumForIndividualState(unittest.TestCase):
    def test_returns_linear_momentum_is_correctly_zeroed_with_masses_N_1(self):
        _, graph = CH3SCH3()

        velocities = torch.randn(graph.num_nodes, 3)
        masses = graph.atomic_masses
        total_mass = graph.total_mass

        velocities = remove_linear_momentum_for_individual_state(
            velocities=velocities, masses=masses, total_mass=total_mass
        )

        com_vel = compute_com_velocity_for_individual_state(
            velocities, masses, total_mass
        )

        self.assertTrue(torch.all(torch.isclose(com_vel, torch.zeros(3), atol=1e-6)))

    def test_returns_linear_momentum_is_correctly_zeroed_with_masses_N(self):
        _, graph = CH3SCH3()

        velocities = torch.randn(graph.num_nodes, 3)
        masses = graph.atomic_masses.view(-1)
        total_mass = graph.total_mass

        velocities = remove_linear_momentum_for_individual_state(
            velocities=velocities, masses=masses, total_mass=total_mass
        )

        masses = masses.view(-1, 1)
        com_vel = compute_com_velocity_for_individual_state(
            velocities, masses, total_mass
        )

        self.assertTrue(torch.all(torch.isclose(com_vel, torch.zeros(3), atol=1e-6)))


class TestRemoveAngularMomentumForIndividualState(unittest.TestCase):
    def test_returns_angular_momentum_is_correctly_zeroed_with_masses_N_1(self):
        _, graph = CH3SCH3()

        positions = graph.pos
        velocities = torch.randn(graph.num_nodes, 3)
        masses = graph.atomic_masses
        total_mass = graph.total_mass

        ang_mom_before, _ = compute_angular_momentum_for_individual_state(
            positions=positions,
            momenta=masses * velocities,
            masses=masses,
            total_mass=total_mass,
        )

        self.assertTrue(torch.linalg.norm(ang_mom_before) > 1e-3)

        velocities = remove_angular_momentum_for_individual_state(
            positions=positions,
            velocities=velocities,
            masses=masses,
            total_mass=total_mass,
        )

        ang_mom, _ = compute_angular_momentum_for_individual_state(
            positions=positions,
            momenta=masses * velocities,
            masses=masses,
            total_mass=total_mass,
        )

        self.assertTrue(torch.all(torch.isclose(ang_mom, torch.zeros(3), atol=1e-5)))

    def test_returns_angular_momentum_is_correctly_zeroed_with_masses_N(self):
        _, graph = CH3SCH3()

        positions = graph.pos
        velocities = torch.randn(graph.num_nodes, 3)
        masses = graph.atomic_masses
        total_mass = graph.total_mass

        ang_mom_before, _ = compute_angular_momentum_for_individual_state(
            positions=positions,
            momenta=masses * velocities,
            masses=masses,
            total_mass=total_mass,
        )

        self.assertTrue(torch.linalg.norm(ang_mom_before) > 1e-3)

        velocities = remove_angular_momentum_for_individual_state(
            positions=positions,
            velocities=velocities,
            masses=masses.view(-1),
            total_mass=total_mass,
        )

        ang_mom, _ = compute_angular_momentum_for_individual_state(
            positions=positions,
            momenta=masses * velocities,
            masses=masses,
            total_mass=total_mass,
        )

        self.assertTrue(torch.all(torch.isclose(ang_mom, torch.zeros(3), atol=1e-5)))


if __name__ == "__main__":
    unittest.main()
