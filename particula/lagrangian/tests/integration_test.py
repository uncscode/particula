import torch
from typing import Tuple

def test_integration_leapfrog():
    # Test case 1: Zero force, zero velocity
    position = torch.tensor([0.0, 0.0, 0.0])
    velocity = torch.tensor([0.0, 0.0, 0.0])
    force = torch.tensor([0.0, 0.0, 0.0])
    mass = 1.0
    time_step = 0.1
    expected_position = torch.tensor([0.0, 0.0, 0.0])
    expected_velocity = torch.tensor([0.0, 0.0, 0.0])
    assert integration_leapfrog(position, velocity, force, mass, time_step) == (expected_position, expected_velocity)

    # Test case 2: Non-zero force, non-zero velocity
    position = torch.tensor([1.0, 2.0, 3.0])
    velocity = torch.tensor([4.0, 5.0, 6.0])
    force = torch.tensor([7.0, 8.0, 9.0])
    mass = 2.0
    time_step = 0.5
    expected_position = torch.tensor([3.0, 4.5, 6.0])
    expected_velocity = torch.tensor([8.5, 10.0, 11.5])
    assert integration_leapfrog(position, velocity, force, mass, time_step) == (expected_position, expected_velocity)

    # Test case 3: Negative force, negative velocity
    position = torch.tensor([-1.0, -2.0, -3.0])
    velocity = torch.tensor([-4.0, -5.0, -6.0])
    force = torch.tensor([-7.0, -8.0, -9.0])
    mass = 3.0
    time_step = 0.2
    expected_position = torch.tensor([-0.2, -0.4, -0.6])
    expected_velocity = torch.tensor([-3.4, -4.6, -5.8])
    assert integration_leapfrog(position, velocity, force, mass, time_step) == (expected_position, expected_velocity)

    # Add more test cases as needed
