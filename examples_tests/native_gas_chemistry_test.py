"""Qualification of the complete native chamber pulse and closed conversion."""

import numpy as np
import pytest
from particula import ProcessRegistry
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import GasChemistry
from particula.execution.cpu_simulation import CPUSimulation
from scripts.docs_examples import import_study

example = import_study("native-gas-chemistry")


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_closed_first_order_case(device):
    """Recover the closed exponential with fixed inert gas and non-unit volumes."""
    import warp as wp

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    state = example.chamber()
    process = example.chemistry(state)
    initial = state.gas.concentration.copy()
    if device == "numpy":
        owner = CPUSimulation(state, process, time_step=1.0)
        owner.step(1.0)
        result = owner.snapshot()
    else:
        registry = ProcessRegistry()
        registry.register(GasChemistry, GasChemistryAdapter())
        with process.prepare(
            state, device=device, time_step=1.0, process_registry=registry
        ) as sim:
            sim.step()
            result = sim.snapshot()
    a = initial[:, 1] * np.exp(-0.3)
    expected = np.column_stack((a, initial[:, 1] - a))
    np.testing.assert_allclose(
        result.gas.concentration[:, 1:], expected, rtol=2e-9, atol=1e-20
    )
    volume = state.particles.volume
    np.testing.assert_allclose(
        result.gas.concentration[:, 1:].sum(axis=1) * volume,
        initial[:, 1] * volume,
        rtol=1e-12,
        atol=1e-22,
    )
    np.testing.assert_array_equal(result.gas.concentration[:, 0], initial[:, 0])


@pytest.mark.cuda
def test_41_observations_and_restart():
    """Exercise both pulse histories and both same-backend restart profiles."""
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    report = example.simulate()
    assert len(report["time_s"]) == 41
    assert set(report["backends"]) == {"numpy", "cuda:0"}


def test_published_conversion_pair(tmp_path):
    """Reproduce the native file supplied in the worked example download."""
    from pathlib import Path

    from particula.dynamics.chemistry.ncar import convert_ncar

    data = Path(__file__).resolve().parents[1] / "docs/data/gas-chemistry"
    destination = tmp_path / "combine.json"
    convert_ncar(data / "combine-ncar.yaml", destination)
    assert (
        destination.read_bytes() == (data / "combine-native.json").read_bytes()
    )
