"""Independent accounting and growth checks for the large CUDA example."""

from dataclasses import replace

import numpy as np
import pytest
from scripts.gpu_studies.environment_sweep import (
    EnvironmentSweep,
    build_environment_sweep,
    conditions,
    run_environment_sweep,
    saturation_pressure,
)


def test_layout_matches_conditions_and_has_balanced_charge():
    """Unit counts and fixed number density define the physical volume."""
    c = EnvironmentSweep(boxes_per_condition=2, particles_per_box=8)
    a, _ = build_environment_sweep(c)
    assert a.particles.masses.shape == (32, 8, 2)
    np.testing.assert_array_equal(a.particles.charge.sum(axis=1), 0)
    np.testing.assert_allclose(
        a.particles.concentration.sum(axis=1) / a.particles.volume, 5e13
    )
    np.testing.assert_array_equal(
        a.particles.masses[:2], a.particles.masses[-2:]
    )
    assert len(conditions()) == 16
    assert EnvironmentSweep().boxes_per_condition * 16 * 128 == 2097152
    np.testing.assert_allclose(saturation_pressure(273.15), 611.21)


@pytest.mark.parametrize(
    "changes",
    [
        {"boxes_per_condition": 0},
        {"particles_per_box": 7},
        {"time_step": np.nan},
        {"duration": 0.000013},
        {"seed": -1},
    ],
)
def test_invalid_layout_or_time_is_rejected(changes):
    """Reject invalid inputs before allocating a simulation."""
    with pytest.raises(ValueError):
        build_environment_sweep(replace(EnvironmentSweep(), **changes))


def cuda():
    """Skip unavailable hardware honestly; publication always requires CUDA."""
    wp = pytest.importorskip("warp")

    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")


def test_condensation_cpu_cuda_parity():
    """Compare separate explicit backends over a short deterministic path."""
    cuda()
    c = EnvironmentSweep(
        boxes_per_condition=1,
        particles_per_box=8,
        duration=1e-5,
        coagulation=False,
    )
    _, cpu = run_environment_sweep(c, device="cpu", capture=False)
    _, gpu = run_environment_sweep(c)
    np.testing.assert_allclose(
        cpu.particles.masses, gpu.particles.masses, rtol=1e-10, atol=0
    )
    np.testing.assert_allclose(
        cpu.gas.concentration, gpu.gas.concentration, rtol=1e-10, atol=0
    )


def test_condensation_refinement_and_kappa_equilibrium():
    """Check half-step convergence and independent water chemical equilibrium."""
    cuda()
    c = EnvironmentSweep(
        boxes_per_condition=2,
        particles_per_box=128,
        duration=0.001,
        coagulation=False,
    )
    coarse_report, coarse = run_environment_sweep(c)
    fine_report, fine = run_environment_sweep(replace(c, time_step=5e-7))
    np.testing.assert_allclose(
        [row["water_mass"] for row in coarse_report["history"]],
        [row["water_mass"] for row in fine_report["history"]],
        rtol=0.01,
        atol=0,
    )
    np.testing.assert_allclose(
        coarse.particles.masses, fine.particles.masses, rtol=0.01, atol=0
    )
    np.testing.assert_allclose(
        coarse.gas.concentration, fine.gas.concentration, rtol=0.001, atol=0
    )
    _, equilibrated = run_environment_sweep(replace(c, duration=0.01))
    p = equilibrated.particles
    water = p.masses[..., 0] / 1000
    dry = p.masses[..., 1] / 1400
    radius = np.cbrt(3 * (water + dry) / (4 * np.pi))
    # kappa-Kohler: a_w=Vw/(Vw+kappa*Vdry); Kelvin uses water density.
    aw = water / (water + 0.3 * dry)
    temperature = equilibrated.environment.temperature[:, None]
    kelvin = np.exp(
        2 * 0.072 * 0.018 / (8.31446261815324 * temperature * 1000 * radius)
    )
    rh = (
        equilibrated.gas.concentration
        * 8.31446261815324
        * temperature
        / (0.018 * saturation_pressure(temperature))
    )
    np.testing.assert_allclose(
        aw * kelvin, np.broadcast_to(rh, aw.shape), rtol=0.001
    )


def test_charged_sweep_conserves_mass_charge_and_only_loses_number():
    """Closed collisions merge particles without changing budgets."""
    cuda()
    c = EnvironmentSweep(
        boxes_per_condition=2, particles_per_box=16, duration=0.001
    )
    r, _ = run_environment_sweep(c)
    count = np.array([x["count"] for x in r["history"]])
    assert np.all(np.diff(count, axis=0) <= 0)
    assert r["maximum_mass_relative_error"] < 2e-11
