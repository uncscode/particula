"""Contains tests for the collisions module."""
import torch
from particula.lagrangian import collisions


def test_find_collisions():
    """Test finding collisions between particles."""
    distance_matrix = torch.tensor([[-1, 2.0, 3.0],
                                    [1.0, -1, 3.0],
                                    [1.0, -0.1, -1]])
    indices = torch.tensor([0, 1, 2])
    mass = torch.tensor([1.0, 2.0, 3.0])
    k = 1
    expected_result = torch.tensor([[1, 2]], dtype=torch.int32)
    result = collisions.find_collisions(distance_matrix, indices, mass, k)
    assert torch.allclose(result, expected_result)


def test_coalescence():
    """Test coalescence of particles."""
    velocity = torch.tensor([[1.0, 2.0, 3.0],
                             [1.0, 2.0, 6.0],
                             [7.0, 8.0, 9.0],
                             [0.0, 0.0, 0.0]]).T
    mass = torch.tensor([1.0, 2.0, 3.0, 1.0])
    collision_indices_pairs = torch.tensor([[0, 1]], dtype=torch.int32)
    expected_velocity = torch.tensor([[1.0, 2.0, 5.0],
                                      [0.0, 0.0, 0.0],
                                      [7.0, 8.0, 9.0],
                                      [0.0, 0.0, 0.0]]).T
    expected_mass = torch.tensor([3.0, 0.0, 3.0, 1.0])
    result_velocity, result_mass = collisions.coalescence(
        velocity, mass, collision_indices_pairs)
    assert torch.allclose(result_mass, expected_mass, atol=1e-4)
    assert torch.allclose(result_velocity, expected_velocity, atol=1e-4)
