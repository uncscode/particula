"""Regression checks for the Galloway dilution example and its source units."""

import numpy as np
import pytest
from docs.examples.processes.customization import glyoxal_chamber as case


def test_vector_measurements_and_held_out_rate():
    """Preserve digitized observations and the independent laboratory rate."""
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
    """Verify the forced budget and finite nitrogen requirement."""
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


@pytest.mark.parametrize("device", ["numpy", "cuda:0"])
def test_evaporation_from_equilibrated_seed(device):
    """A halved gas pressure releases previously dissolved glyoxal."""
    if device == "cuda:0":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("CUDA device unavailable")
    state = case.initial_state(100.0)
    mass = state.particles.masses[0, 0]
    # Check both experimental salting regimes with an independently specified
    # monomer concentration for an imposed partial pressure of 100 ppbv.
    for molal, ratio in [(14.0, 0.5), (8.0, 1.0)]:
        mass[2] = molal * case.MW[2] * mass[4]
        h = 4.19e5 * 10 ** (0.24 * min(molal, 12.0)) / (1.0 + ratio)
        mass[0] = h * 1e-7 * mass[4] * case.MW[0]
        state.gas.concentration[0, 0] = case.gas_kg(50.0)
        before = mass[0]
        baseline = case.ledger(state)
        transfer = case.par.dynamics.MassCondensation(
            case.condensation(state, case.exchange_coefficient(state))
        )
        if device == "numpy":
            transfer.execute(state, 10.0)
            result = state
        else:
            with transfer.prepare(state, device=device, time_step=10.0) as sim:
                sim.step()
                result = sim.snapshot()
        assert result.particles.masses[0, 0, 0] < before
        assert result.gas.concentration[0, 0] > case.gas_kg(50.0)
        np.testing.assert_allclose(case.ledger(result), baseline, rtol=2e-13)
