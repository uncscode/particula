"""Tests for particle_pairs.py."""
import torch
from particula.lagrangian import particle_pairs


def test_remove_duplicates():
    """Test removing duplicate entries from a tensor of index pairs."""
    index_pairs = torch.tensor([[1, 2], [3, 4], [1, 2]])
    index_to_remove = 0
    expected_output = torch.tensor([[1, 2], [3, 4]])
    result = particle_pairs.remove_duplicates(index_pairs, index_to_remove)
    assert torch.all(result == expected_output)


def test_calculate_pairwise_distance():
    """Test calculating pairwise distances between points
    in a position tensor."""
    position = torch.tensor([[1, 1, 1], [1, 1, 2], [-1, 1, 1]])
    expected_output = torch.tensor([[0.0, 2.0, 2.2361],
                                    [2.0, 0.0, 1.0],
                                    [2.2361, 1.0, 0.0]])
    result = particle_pairs.calculate_pairwise_distance(position)
    assert torch.allclose(result, expected_output, atol=1e-4)
