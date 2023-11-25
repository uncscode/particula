import torch
import pytest
from particle_pairs import remove_duplicates, calculate_pairwise_distance

def test_remove_duplicates():
    index_pairs = torch.tensor([[1, 2], [3, 4], [1, 2]])
    index_to_remove = 0
    expected_output = torch.tensor([[1, 2], [3, 4]])
    assert torch.all(remove_duplicates(index_pairs, index_to_remove) == expected_output)

    index_pairs = torch.tensor([[1, 2], [3, 4], [1, 2]])
    index_to_remove = 1
    expected_output = torch.tensor([[1, 2], [3, 4]])
    assert torch.all(remove_duplicates(index_pairs, index_to_remove) == expected_output)

def test_calculate_pairwise_distance():
    position = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_output = torch.tensor([[0.0000, 5.1962, 10.3923],
                                    [5.1962, 0.0000, 5.1962],
                                    [10.3923, 5.1962, 0.0000]])
    assert torch.allclose(calculate_pairwise_distance(position), expected_output)

    position = torch.tensor([[0, 0], [3, 4]])
    expected_output = torch.tensor([[0.0000, 5.0000],
                                    [5.0000, 0.0000]])
    assert torch.allclose(calculate_pairwise_distance(position), expected_output)
