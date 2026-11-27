"""Check the reversible exchange step against independent dense solves."""

import numpy as np
import torch
from process_scale import SPECIES, ReversibleBox
from stage3_one_box import Box


def test_signed_exchange_matches_dense_implicit_solution():
    """Recover all gas and particle masses for the three vapor regimes."""
    hosts = 7
    box = ReversibleBox.__new__(ReversibleBox)
    Box.__init__(
        box, hosts, SPECIES, 1e12, "neutral", "cpu", 901003, 0, "physical"
    )
    box.uptake = None
    box.initialize_reservoirs()
    radius, mass = box.geometry()
    dt = 0.0625
    k = box.conductance(radius, slice(None)).numpy()
    start = np.vstack((box.gas.numpy(), box.mass[:, 1:].numpy() / box.volume))
    expected = np.empty_like(start)
    for j in range(SPECIES):
        # Unknowns are gas and per-particle mass concentrations (kg/m3).
        gain = dt * k[:, j] / box.volume
        loss = dt * k[:, j] * box.cstar[j].item() / mass.numpy()
        matrix = np.diag(np.r_[1 + gain.sum(), 1 + loss])
        matrix[0, 1:] = -loss
        matrix[1:, 0] = -gain
        expected[:, j] = np.linalg.solve(matrix, start[:, j])
    box.vapor(dt, 0)
    actual = np.vstack((box.gas.numpy(), box.mass[:, 1:].numpy() / box.volume))
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=0)
    np.testing.assert_allclose(actual.sum(0), start.sum(0), rtol=5e-14)
    assert torch.all(box.mass >= 0)
    assert np.all(actual[0, :16] < start[0, :16])
    assert np.all(actual[0, 16:24] > start[0, 16:24])
    np.testing.assert_allclose(actual[0, 24:], start[0, 24:], rtol=5e-14)
