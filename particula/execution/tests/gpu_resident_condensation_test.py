"""Qualify native conditional-graph adaptation independently of host control."""

import numpy as np
import pytest
import warp as wp

from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.gpu_resident_condensation import (
    GPUResidentCondensation,
)
from particula.execution.tests.gpu_condensation_ode_test import advance
from particula.execution.tests.gpu_multimaterial_condensation_test import (
    material_case,
)


@pytest.fixture
def cuda():
    """Require real conditional CUDA graphs; never emulate control on CPU."""
    if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
        pytest.skip("native CUDA conditional graphs unavailable")
    return "cuda:0"


@pytest.mark.parametrize("method", ["rk23", "ros3"])
@pytest.mark.parametrize("materials", [3, 12])
def test_device_control_matches_host_reference_without_trial_readbacks(
    cuda, monkeypatch, method, materials
):
    """All trial decisions stay in a captured device loop, including rejections."""
    initial, strategy = material_case(materials)
    controls = dict(
        method=method,
        first_step=0.05,
        max_step=0.05,
        rtol=1e-8,
        particle_atol=1e-27,
        gas_atol=1e-16,
    )
    owner = GPUResidentCondensation(initial, strategy, device=cuda, **controls)
    before = owner.snapshot()
    np.testing.assert_array_equal(
        before.particles.masses, initial.particles.masses
    )
    reference = GPUCondensationODE(initial, strategy, device=cuda, **controls)
    expected = advance(reference, 0.05)
    original_numpy = wp.array.numpy
    reads = []

    def boundary_only(array, *args, **kwargs):
        assert array is owner._control or array is owner._counters
        reads.append(array)
        return original_numpy(array, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "trial loop must not upload or call host adaptation"
        )

    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", boundary_only)
        patch.setattr(wp.array, "assign", forbidden)
        patch.setattr(GPUCondensationODE, "_step", forbidden)
        owner.advance(0.05)
    assert len(reads) == 2
    assert owner.statistics.accepted > 1
    assert owner.statistics.rejected > 0
    actual = owner.snapshot()
    np.testing.assert_allclose(
        actual.particles.masses,
        expected.particles.masses,
        rtol=2e-8,
        atol=1e-27,
    )
    np.testing.assert_allclose(
        actual.gas.concentration,
        expected.gas.concentration,
        rtol=2e-8,
        atol=1e-17,
    )


def test_no_cpu_or_host_forcing_fallback():
    """Unsupported device and callback configurations reject explicitly."""
    initial, strategy = material_case(3)
    with pytest.raises(ValueError, match="native CUDA"):
        GPUResidentCondensation(initial, strategy, device="cpu")
    with pytest.raises(ValueError, match="environment segments"):
        GPUResidentCondensation(
            initial, strategy, device="cpu", forcing=lambda t: (298.15, 101325)
        )


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_reset_controller_matches_fresh_flow_without_physical_transfer(
    cuda, method, monkeypatch
):
    """Fixed split boundaries restart numerical history using CUDA state."""
    initial, strategy = material_case(3)
    controls = dict(method=method, first_step=0.001, max_step=0.001)
    owner = GPUResidentCondensation(initial, strategy, device=cuda, **controls)
    owner.advance(0.001)
    reference = GPUResidentCondensation(
        owner.snapshot(), strategy, device=cuda, time=0.001, **controls
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            wp.array,
            "numpy",
            lambda *args: pytest.fail("reset downloaded state"),
        )
        patch.setattr(
            wp.array,
            "assign",
            lambda *args: pytest.fail("reset uploaded state"),
        )
        owner.restart_controller(0.001)
    assert owner.statistics.accepted == 0
    owner.advance(0.002)
    reference.advance(0.002)
    np.testing.assert_array_equal(
        owner.snapshot().particles.masses, reference.snapshot().particles.masses
    )
    np.testing.assert_array_equal(
        owner.snapshot().gas.concentration,
        reference.snapshot().gas.concentration,
    )


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_empty_resident_population_retains_gas(cuda, method):
    """An exhausted cloud can continue to a later physical boundary."""
    initial, strategy = material_case(3)
    initial.particles.masses[:] = 0
    initial.particles.concentration[:] = 0
    initial.particles.charge[:] = 0
    owner = GPUResidentCondensation(
        initial, strategy, device=cuda, method=method
    )
    owner.advance(0.1)
    actual = owner.snapshot()
    np.testing.assert_array_equal(
        actual.gas.concentration, initial.gas.concentration
    )
    np.testing.assert_array_equal(
        actual.particles.masses, initial.particles.masses
    )


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_binary_bat_resident_against_direct_cpu(cuda, method):
    """Tabulated material pressures drive a conservative kinetic trajectory."""
    from particula.execution.condensation_ode import CondensationODE
    from particula.particles import ActivityBAT

    initial, strategy = material_case(2)
    strategy.activity_strategy = ActivityBAT(0.2, 0.5, 1400.0)
    controls = dict(rtol=1e-7, particle_atol=1e-27, gas_atol=1e-16)
    gpu = GPUResidentCondensation(
        initial, strategy, method=method, device=cuda, **controls
    )
    cpu = CondensationODE(initial, strategy, method="rk23", **controls)
    for end in (0.001, 0.01):
        gpu.advance(end)
        got = gpu.snapshot()
        ref = advance(cpu, end)
        np.testing.assert_allclose(
            got.particles.radii, ref.particles.radii, rtol=1e-3
        )
        np.testing.assert_array_equal(
            got.gas.concentration[:, 1], initial.gas.concentration[:, 1]
        )
        for source, target in strategy.species_map.gas_to_particle:
            np.testing.assert_allclose(
                got.gas.concentration[:, source]
                + got.particles.species_mass_density[:, target],
                initial.gas.concentration[:, source]
                + initial.particles.species_mass_density[:, target],
                rtol=3e-13,
            )
            np.testing.assert_allclose(
                got.particles.masses[..., target],
                ref.particles.masses[..., target],
                rtol=3e-4,
                atol=1e-27,
            )
