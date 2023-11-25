"""test module for particle_property.py"""
import torch
from particula.lagrangian import particle_property


def test_radius_vector():
    """Test calculating the radius vector."""
    mass = torch.tensor([10.0, 20.0, 30.0])
    density = torch.tensor([2.0, 3.0, 4.0])
    expected_radius = torch.tensor(
        [1.0608, 1.1675, 1.2143])
    result = particle_property.radius(mass, density)
    assert torch.allclose(result, expected_radius, atol=1e-4)
