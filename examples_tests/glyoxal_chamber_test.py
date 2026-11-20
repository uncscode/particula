"""Regression checks for the Galloway dilution example and its source units."""

import numpy as np
import pytest
from docs.examples.processes.customization import glyoxal_chamber as case


def test_vector_measurements_and_held_out_rate():
    observed = case.observations()
    assert [len(series) for series in observed.values()] == [2214, 780, 773]
    for series in observed.values():
        assert np.all(np.diff(series[:, 0]) > 0)
    assert case.decline(observed["organic_per_sulfate"]) == pytest.approx(
        10.20, abs=0.02
    )
    checks = case.independent_checks()
    assert checks["IC_M_per_s"] == pytest.approx(9.74e-8, abs=0.90e-8)


@pytest.mark.parametrize("device", ["numpy", "cuda:0"])
def test_vapor_forcing_and_nitrogen_control(device):
    if device == "cuda:0":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("CUDA device unavailable")
    gas = case.observations()["gas_ppbv"]
    for nitrogen in (True, False):
        rows, balance, _ = case.replay(
            gas, device, 150, end=10, nitrogen=nitrogen
        )
        assert balance < 1e-11
        assert rows[-1, 3] > 0
        assert (rows[-1, 2] > 0) == nitrogen
        np.testing.assert_allclose(
            rows[:, 5], rows[:, 6] - rows[:, 7], atol=1e-21
        )
