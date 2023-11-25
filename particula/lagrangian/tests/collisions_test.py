"""Contains tests for the collisions module."""
import torch
from particula.lagrangian import collisions


def test_find_collisions():
    """Test finding collisions between particles."""
    distance_matrix = torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0],
                                    [7.0, 8.0, 9.0]])
    indices = torch.tensor([0, 1, 2])
    mass = torch.tensor([1.0, 2.0, 3.0])
    k = 1
    expected_result = torch.empty(0, 2)
    result = collisions.find_collisions(distance_matrix, indices, mass, k)

    assert torch.allclose(result, expected_result)


def test_coalescence():
    """Test coalescence of particles."""
    velocity = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
    mass = torch.tensor([1.0, 2.0, 3.0])
    collision_indices_pairs = torch.empty(0, 2)
    expected_velocity = velocity.clone()
    expected_mass = mass.clone()
    result_velocity, result_mass = collisions.coalescence(
        velocity, mass, collision_indices_pairs)
    assert torch.allclose(result_velocity, expected_velocity)
    assert torch.allclose(result_mass, expected_mass)
