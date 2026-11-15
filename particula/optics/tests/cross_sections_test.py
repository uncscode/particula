"""Dimensional cross-section contracts and device particle absorption."""

import numpy as np
import pytest
from particula.optics import (
    get_differential_cross_sections,
    get_mie_efficiencies,
)


def test_polarized_differential_cross_sections():
    """Check absolute units, medium scaling and the angle-axis contract."""
    s1 = np.array([[1 + 2j, 3 - 4j]])
    s2 = np.array([[2 - 1j, 1 + 1j]])
    wave = np.array([500e-9, 700e-9])
    medium = np.array([1, 1.33])
    result = get_differential_cross_sections(s1, s2, wave, medium)
    factor = (wave / (2 * np.pi * medium))[:, None] ** 2
    np.testing.assert_allclose(result.perpendicular, [[5, 25]] * factor)
    np.testing.assert_allclose(result.parallel, [[5, 2]] * factor)
    np.testing.assert_allclose(result.unpolarized, [[5, 13.5]] * factor)
    # Common length scaling changes area by the square of that scale.
    scaled = get_differential_cross_sections(s1, s2, 2 * wave, medium)
    np.testing.assert_allclose(scaled.unpolarized, 4 * result.unpolarized)
    # Scaling before squaring avoids overflowing dimensionless amplitudes.
    tiny = get_differential_cross_sections([1e200], [1e200], 1e-200)
    np.testing.assert_allclose(tiny.unpolarized, [1 / (2 * np.pi) ** 2])


@pytest.mark.parametrize(
    "s1,s2,wave,medium",
    [
        ([1], [1, 2], 1, 1),
        (1, 1, 1, 1),
        ([np.nan], [1], 1, 1),
        ([1], [1], 0, 1),
        ([1], [1], 1, 0),
        ([1], [1], 1, 1 + 0j),
    ],
)
def test_differential_rejection(s1, s2, wave, medium):
    """Reject malformed angular fields and unsupported absorbing media."""
    with pytest.raises(ValueError):
        get_differential_cross_sections(s1, s2, wave, medium)


def test_differential_overflow():
    """Signal unusable dimensional values explicitly without runtime warnings."""
    with pytest.raises(RuntimeError, match="Nonfinite"):
        get_differential_cross_sections([1e200], [1], 1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_particle_cross_sections(device):
    """Retain all particle areas, including absorption, and propagate failures."""
    wp = pytest.importorskip("warp")
    from particula.optics.gpu.cross_sections import cross_sections_kernel

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    radius = np.array([0, 10e-9, 200e-9, 1e-6])
    q = get_mie_efficiencies(radius, 550e-9, 1.5 + 0.1j)
    values = np.stack(q, -1)
    inputs = wp.array(values, dtype=wp.float64, device=device)
    r = wp.array(radius, dtype=wp.float64, device=device)
    upstream = wp.zeros(4, dtype=wp.int32, device=device)
    output = wp.empty((4, 5), dtype=wp.float64, device=device)
    status = wp.empty(4, dtype=wp.int32, device=device)

    def launch():
        wp.launch(
            cross_sections_kernel,
            dim=4,
            inputs=[inputs, r, upstream, output, status],
            device=device,
        )

    expected = values[:, [0, 1, 2, 4, 5]] * (np.pi * radius**2)[:, None]
    pointer = output.ptr
    for _ in range(2):
        launch()
        np.testing.assert_array_equal(status.numpy(), 0)
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-14)
        assert output.ptr == pointer
    upstream.assign(np.array([1, 0, 0, 0], dtype=np.int32))
    r.assign(np.array([0, -1, 1e300, 1e-6]))
    values[3, 2] = -1
    inputs.assign(values)
    launch()
    np.testing.assert_array_equal(status.numpy(), [2, 1, 3, 1])
