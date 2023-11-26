"""Test module for boundary.py"""

import torch
from particula.lagrangian import boundary


def test_wrapped_cube():
    """Test wrapping a position vector to a cube."""
    # Test case 1: position within the cube
    position = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 4.0]])
    cube_side = 10.0
    expected_result = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 4.0]])
    result = boundary.wrapped_cube(
        position=position,
        cube_side=cube_side)

    assert torch.allclose(
        result,
        expected_result)

    # Test case 2: position exceeding the positive boundary
    position = torch.tensor([[6.0, 7.0, 8.0], [5.0, 4.0, 8.0]])
    cube_side = 10.0
    expected_result = torch.tensor([[-4.0, -3.0, -2.0], [5.0, 4.0, -2.0]])
    result = boundary.wrapped_cube(
        position,
        cube_side)
    assert torch.allclose(
        result,
        expected_result)

    # Test case 3: position exceeding the negative boundary
    position = torch.tensor([[-6.0, -7.0, -8.0], [-5.0, -4.0, -8.0]])
    cube_side = 10.0
    expected_result = torch.tensor([[4.0, 3.0, 2.0], [-5.0, -4.0, 2.0]])
    assert torch.allclose(
        boundary.wrapped_cube(
            position,
            cube_side),
        expected_result)
