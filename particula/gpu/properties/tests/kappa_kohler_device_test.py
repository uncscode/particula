"""Independent checks of the binary64 device κ-Köhler property."""

import itertools
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.warp

wp = pytest.importorskip("warp")

from particula.gpu.properties.kappa_kohler import (  # noqa: E402
    kappa_kohler_critical_dry_diameter,
)
from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402
from particula.particles.properties.tests.kohler_reference import (  # noqa: E402
    KELVIN_DIAMETER,
    forward_peak,
    forward_threshold,
)


@wp.kernel
def _evaluate(kappa: Any, ratio: Any, kelvin_length: Any, result: Any):
    i = wp.tid()
    result[i] = kappa_kohler_critical_dry_diameter(
        kappa[i], ratio[i], kelvin_length[i]
    )


def _launch(device, cases):
    kappa, ratio, length = np.asarray(cases, dtype=np.float64).T
    inputs = [
        wp.array(x, dtype=wp.float64, device=device)
        for x in (kappa, ratio, length)
    ]
    result = wp.empty(len(cases), dtype=wp.float64, device=device)
    wp.launch(
        _evaluate,
        dim=len(cases),
        inputs=inputs,
        outputs=[result],
        device=device,
    )
    return result.numpy()


@pytest.mark.parametrize("device", warp_devices(wp))
def test_device_inverse_closes_independent_forward_equation(device):
    """Both wet-barrier height and dry diameter close on the actual device."""
    pairs = list(
        itertools.product(
            [0, 0.01, 0.1, 0.3, 0.6, 1.0], [1.001, 1.002, 1.005, 1.01]
        )
    )
    # Test both competing-maximum orderings, the weakly supersaturated
    # limit, and extreme mathematical cases outside atmospheric conditions.
    pairs.extend(
        [
            (0.3, np.nextafter(1.0, 2.0)),
            (1000.0, np.exp(0.002)),
            (1000.0, np.exp(0.5)),
            (35.0, 3.0),
            (1e300, np.exp(0.5)),
            (0.3, np.finfo(np.float64).max),
        ]
    )
    cases = [(k, ratio, KELVIN_DIAMETER) for k, ratio in pairs]
    for (k, ratio), actual in zip(pairs, _launch(device, cases), strict=True):
        # The near-dry local peak and the dilute peak may be hundreds of
        # natural-log units apart when κ is extreme.
        grid = np.linspace(-60, 430, 1961)
        log_target = np.log(ratio)
        expected = forward_threshold(k, log_target, grid)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
        np.testing.assert_allclose(
            forward_peak(actual, k, grid), log_target, rtol=1e-9, atol=0
        )


@pytest.mark.parametrize("device", warp_devices(wp))
def test_device_insoluble_limit_and_invalid_sentinels(device):
    """Subnormal κ approaches the Kelvin limit; invalid lanes signal failure."""
    smallest = np.nextafter(0.0, 1.0)
    largest = np.finfo(np.float64).max
    cases = [
        (smallest, 1.002, KELVIN_DIAMETER),
        (-0.1, 1.002, KELVIN_DIAMETER),
        (np.nan, 1.002, KELVIN_DIAMETER),
        (np.inf, 1.002, KELVIN_DIAMETER),
        (0.3, 1.0, KELVIN_DIAMETER),
        (0.3, np.nan, KELVIN_DIAMETER),
        (0.3, np.inf, KELVIN_DIAMETER),
        (0.3, 1.002, 0.0),
        (0.3, 1.002, np.nan),
        (0.3, 1.002, np.inf),
        (0.0, np.nextafter(1.0, 2.0), largest),
        (0.0, largest, smallest),
    ]
    actual = _launch(device, cases)
    np.testing.assert_allclose(
        actual[0], KELVIN_DIAMETER / np.log(1.002), rtol=2e-14, atol=0
    )
    assert np.all(actual[1:] < 0)
