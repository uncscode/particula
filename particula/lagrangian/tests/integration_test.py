"""Test the integration module."""
import torch
from particula.lagrangian import integration


def test_integration_leapfrog():
    """Test leapfrog integration."""
    position = torch.tensor([0.0, 0.0, 0.0])
    velocity = torch.tensor([1.0, 0.0, 0.0])
    force = torch.tensor([0.0, 1.0, 0.0])
    mass = torch.tensor([1.0, 1.0, 1.0])
    time_step = 1.0
    expected_position = torch.tensor([1.0, 0.5, 0.0])
    expected_velocity = torch.tensor([1.0, 1.0, 0.0])
    result_position, result_velocity = integration.leapfrog(
        position, velocity, force, mass, time_step)
    assert torch.allclose(result_position, expected_position)
    assert torch.allclose(result_velocity, expected_velocity)
