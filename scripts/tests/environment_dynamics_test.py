"""Independent atmosphere and resolved-population diagnostics checks."""

from dataclasses import replace

import numpy as np
import pytest
from scripts.gpu_studies.dry_environment_sweep import (
    DryEnvironmentSweep,
    build_dry_sweep,
    dry_conditions,
    run_dry_sweep,
    standard_atmosphere,
)
from scripts.gpu_studies.population_diagnostics import (
    DIAMETER_EDGES,
    population_summary,
)


def test_atmosphere_matches_published_layer_boundaries():
    """Compare NASA Appendix A pressure/temperature anchors independently."""
    h = np.array([0, 11, 20, 32, 47, 51, 71.0])
    z = 6356.766 * h / (6356.766 - h)
    t, p = standard_atmosphere(z)
    np.testing.assert_allclose(
        t, [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65], atol=1e-10
    )
    np.testing.assert_allclose(
        p, [101325, 22632.06, 5474.89, 868.02, 110.91, 66.94, 3.96], rtol=0.001
    )
    t, p = standard_atmosphere([86])
    np.testing.assert_allclose(t, [186.946], atol=0.001)
    np.testing.assert_allclose(p, [0.3733836], rtol=2e-5)


@pytest.mark.parametrize("altitude", [-1, 86.1, np.nan, np.inf])
def test_atmosphere_rejects_extrapolation(altitude):
    """Do not silently extrapolate the lower-atmosphere equations."""
    with pytest.raises(ValueError):
        standard_atmosphere(altitude)


def test_dry_cases_preserve_density_and_separate_high_pressure():
    """Pressure controls hold temperature fixed and reuse a ground control."""
    config = DryEnvironmentSweep(boxes_per_condition=2, particles_per_box=4)
    a, _ = build_dry_sweep(config)
    np.testing.assert_allclose(
        a.particles.concentration.sum(axis=1) / a.particles.volume, 1e11
    )
    np.testing.assert_array_equal(
        a.particles.masses[:2], a.particles.masses[-2:]
    )
    t, p = dry_conditions()
    np.testing.assert_array_equal(t[[0, 16, 17, 18]], 288.15)
    np.testing.assert_array_equal(p[[0, 16, 17, 18]] / 101325, [1, 2, 5, 10])
    assert np.all(np.diff(p[:16]) < 0)


def test_distributions_reproduce_known_sizes_counts_and_charge():
    """Check bin integrals and charge states without using process diagnostics."""
    a, _ = build_dry_sweep(
        DryEnvironmentSweep(boxes_per_condition=1, particles_per_box=4)
    )
    diameters = np.array([20.0, 40.0, 80.0, 160.0]) * 1e-9
    a.particles.masses[:] = (1400 * np.pi / 6 * diameters**3)[None, :, None]
    a.particles.charge[:] = [-2, -1, 1, 2]
    r = population_summary(a, 19)
    np.testing.assert_array_equal(r["number"], 4)
    np.testing.assert_array_equal(r["net_charge"], 0)
    np.testing.assert_allclose(
        r["diameter_quantiles"], np.tile([26.0, 60.0, 136.0], (19, 1)) * 1e-9
    )
    expected = np.zeros(len(DIAMETER_EDGES) - 1)
    expected[np.searchsorted(DIAMETER_EDGES, diameters) - 1] = 1
    np.testing.assert_array_equal(
        r["diameter_histogram"], np.tile(expected, (19, 1))
    )
    for q in (-2, -1, 1, 2):
        np.testing.assert_array_equal(r["charge_counts"][str(q)], 1)
    a.particles.concentration[:, 0] = 0
    a.particles.masses[:, 0] = 0
    a.particles.charge[:, 0] = 0
    np.testing.assert_array_equal(population_summary(a, 19)["number"], 3)


def test_histogram_refuses_to_hide_out_of_range_particles():
    """A plotting range must never silently lose represented particles."""
    a, _ = build_dry_sweep(
        DryEnvironmentSweep(boxes_per_condition=1, particles_per_box=4)
    )
    a.particles.masses[:, 0] = 1.0
    with pytest.raises(AssertionError):
        population_summary(a, 19)


def test_extreme_dry_conditions_are_finite_and_conservative_on_cuda():
    """Exercise the top-of-profile and ten-atmosphere cases in one pipeline."""
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    c = DryEnvironmentSweep(
        boxes_per_condition=2, particles_per_box=16, duration=10, time_step=1
    )
    r, _ = run_dry_sweep(c)
    number = np.array([row["number"] for row in r["history"]])
    assert np.all(np.diff(number, axis=0) <= 0)
    assert r["maximum_mass_relative_error"] < 2e-11


@pytest.mark.parametrize(
    "changes",
    [{"number_concentration": 0}, {"time_step": 3}, {"particles_per_box": 3}],
)
def test_invalid_dry_configuration_rejected(changes):
    """Reject invalid density, geometry and time controls before preparation."""
    with pytest.raises(ValueError):
        build_dry_sweep(replace(DryEnvironmentSweep(), **changes))


def test_ion_humidity_balances_and_aerosol_spectra():
    """Source/export accounting distinguishes ion losses from aerosol mergers."""
    from scripts.gpu_studies.environment_sweep import (
        EnvironmentSweep,
        run_environment_sweep,
    )

    config = EnvironmentSweep(boxes_per_condition=2, ions=True, duration=1e-5)
    report, final = run_environment_sweep(config, device="cpu", capture=False)
    assert report["initial_particles"] == 16 * 2 * (128 + 32)
    assert sum(report["history"][0]["number"]) == 16 * 2 * 128
    assert report["maximum_mass_relative_error"] < 2e-11
    last = report["history"][-1]
    for key in [
        "attachment_positive",
        "attachment_negative",
        "recombination",
        "injected_pairs",
    ]:
        assert np.all(np.array(last[key]) >= 0)
    np.testing.assert_array_equal(final.particles.charge.sum(axis=1), 0)
