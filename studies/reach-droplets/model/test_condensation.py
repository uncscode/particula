"""Physical and gradient checks for the study-specific sectional remap."""

import pytest

torch = pytest.importorskip("torch")
from condensation import WATER_MASS, budget_error, remap


def test_growth_and_evaporation_conserve_cell_water():
    diameter = torch.tensor([11.0, 13.0, 15.0], dtype=torch.float64)
    vol = diameter**3
    # Both cells end at 12 µm, one by growing and the other by evaporating.
    pop = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    target = torch.full_like(pop, 144.0, requires_grad=True)
    wet, uptake = remap(pop, target, vol)
    torch.testing.assert_close(wet.sum(1), pop.sum(1))
    assert torch.all(wet >= 0)
    torch.testing.assert_close(wet @ vol, pop.sum(1) * 12**3)
    expected = WATER_MASS * pop.sum(1) * (12**3 - torch.tensor([11**3, 13**3]))
    torch.testing.assert_close(uptake, expected, rtol=1e-12, atol=1e-20)
    cell = torch.tensor([0.2, 0.8], dtype=torch.float64)
    gas = torch.full_like(cell, 0.01)
    assert budget_error(pop, wet, vol, gas, gas - uptake / cell, cell) < 1e-17
    # A compensating error between two cells would vanish in a chamber sum.
    wrong_gas = gas - uptake / cell + torch.tensor([1e-7, -1e-7], dtype=gas.dtype) / cell
    assert budget_error(pop, wet, vol, gas, wrong_gas, cell) > 4e-7
    (wet @ vol).sum().backward()
    torch.testing.assert_close(target.grad, 1.5 * 12 * pop)


def test_evaporation_and_upper_grid_boundary():
    vol = torch.tensor([1.0, 27.0], dtype=torch.float64)
    pop = torch.ones(3, 2, dtype=torch.float64)
    new_d2 = torch.tensor([[-1.0, 0.0], [0.25, 9.0], [16.0, 9.0]], dtype=torch.float64)
    wet, uptake = remap(pop, new_d2, vol)
    expected = torch.tensor([[0.0, 0.0], [0.125, 1.0], [0.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(wet, expected)
    torch.testing.assert_close(uptake, WATER_MASS * ((expected - pop) @ vol))
