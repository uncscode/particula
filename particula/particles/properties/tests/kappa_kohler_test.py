"""Physical checks for the equilibrium critical dry diameter (all lengths m)."""

import numpy as np
import pytest
from particula.particles import (
    get_kappa_kohler_critical_dry_diameter,
    get_kelvin_radius,
)
from particula.particles.properties.tests.kohler_reference import (
    KELVIN_DIAMETER,
    forward_peak,
    forward_threshold,
)


@pytest.mark.parametrize("kappa", [0, 0.01, 0.1, 0.3, 0.6, 1])
@pytest.mark.parametrize("ratio", [1.001, 1.002, 1.005, 1.01])
def test_forward_peak_and_threshold(kappa, ratio):
    """Both the dry length and actual forward supersaturation must close."""
    log_target = np.log(ratio)
    coarse = np.linspace(-40, 40, 321)
    fine = np.linspace(-50, 50, 801)
    reference = forward_threshold(kappa, log_target, fine)
    np.testing.assert_allclose(
        forward_threshold(kappa, log_target, coarse),
        reference,
        rtol=1e-10,
        atol=0,
    )
    diameter = get_kappa_kohler_critical_dry_diameter(
        kappa, ratio, KELVIN_DIAMETER
    )
    np.testing.assert_allclose(diameter, reference, rtol=1e-9, atol=0)
    np.testing.assert_allclose(
        forward_peak(diameter, kappa, fine), log_target, rtol=1e-9, atol=0
    )
    assert forward_peak(0.999 * diameter, kappa, fine) > log_target
    assert forward_peak(1.001 * diameter, kappa, fine) < log_target


@pytest.mark.parametrize(
    ("kappa", "ratio", "expected_m"),
    [
        (0.3, np.nextafter(1.0, 2.0), 45.26725244609933),
        (1000.0, np.exp(0.5), 0.22672742696165227e-9),
    ],
)
def test_global_barrier_outside_study_grid(kappa, ratio, expected_m):
    """Near saturation and competing local extrema have independent checks."""
    log_target = np.log(ratio)
    fine = np.linspace(-50, 50, 801)
    ref = forward_threshold(kappa, log_target, fine)
    np.testing.assert_allclose(ref, expected_m, atol=0, rtol=1e-10)
    diameter = get_kappa_kohler_critical_dry_diameter(
        kappa, ratio, KELVIN_DIAMETER
    )
    np.testing.assert_allclose(diameter, ref, rtol=1e-9, atol=0)
    np.testing.assert_allclose(
        forward_peak(diameter, kappa, fine), log_target, rtol=1e-9, atol=0
    )


def test_kelvin_limit_and_scale():
    """Recover the Kelvin diameter, approach it continuously, and scale in A."""
    radius = get_kelvin_radius(0.072, 997, 0.01801528, 298.15)
    kappas = np.array([0.0, 1e-24, 1e-10, 0.01, 0.3, 1.0])[:, None]
    ratios = np.array([1.001, 1.002, 1.01])
    sizes = get_kappa_kohler_critical_dry_diameter(kappas, ratios, 2 * radius)
    assert sizes.shape == (6, 3)
    assert sizes.dtype == np.float64
    np.testing.assert_allclose(
        sizes[0], 2 * radius / np.log(ratios), rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(sizes[1], sizes[0], rtol=1e-9, atol=0)
    assert np.all(np.diff(sizes, axis=1) < 0)
    assert np.all(np.diff(sizes, axis=0) <= 0)
    np.testing.assert_allclose(
        get_kappa_kohler_critical_dry_diameter(kappas, ratios, radius),
        sizes / 2,
        rtol=1e-12,
        atol=0,
    )


def test_scalar_and_empty_result():
    """Scalar arguments return metres as float and empty axes broadcast."""
    value = get_kappa_kohler_critical_dry_diameter(0.3, 1.002, KELVIN_DIAMETER)
    assert isinstance(value, float)
    assert value == pytest.approx(104.53745311998e-9, rel=1e-11, abs=0)
    empty = get_kappa_kohler_critical_dry_diameter(
        np.empty((0, 1)), np.array([1.001, 1.002]), KELVIN_DIAMETER
    )
    assert empty.shape == (0, 2)


@pytest.mark.parametrize(
    ("kappa", "ratio", "length"),
    [
        ([-0.1, 0.3], 1.002, 2e-9),
        (np.inf, 1.002, 2e-9),
        (np.nan, 1.002, 2e-9),
        (0.3, 1.0, 2e-9),
        (0.3, 0.5, 2e-9),
        (0.3, np.nan, 2e-9),
        (0.3, np.inf, 2e-9),
        (0.3, 1.002, -1e-9),
        (0.3, 1.002, 0),
        (0.3, 1.002, np.inf),
        (0.3, 1.002, np.nan),
        ([0.1, 0.3], [1.002, 1.003, 1.004], 2e-9),
    ],
)
def test_invalid_domain_and_shapes(kappa, ratio, length):
    """Reject invalid elements anywhere in the call and incompatible axes."""
    with pytest.raises(ValueError):
        get_kappa_kohler_critical_dry_diameter(kappa, ratio, length)


@pytest.mark.parametrize(
    ("ratio", "length"),
    [
        (np.nextafter(1.0, 2.0), np.finfo(np.float64).max),
        (np.finfo(np.float64).max, np.nextafter(0.0, 1.0)),
    ],
)
def test_unrepresentable_threshold(ratio, length):
    """A finite input combination may have no positive finite float result."""
    with pytest.raises(FloatingPointError):
        get_kappa_kohler_critical_dry_diameter(0.0, ratio, length)
