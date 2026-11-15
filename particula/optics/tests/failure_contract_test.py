"""Explicit physical-domain and unresolved-arithmetic failure contracts."""

import numpy as np
import pytest
from particula.optics import (
    MieCoefficients,
    get_mie_coefficients,
    get_mie_efficiencies,
)


@pytest.mark.parametrize("argument", ["radius", "wavelength", "medium_index"])
def test_complex_physical_coordinates_rejected(argument):
    """Never silently discard an imaginary coordinate or medium absorption."""
    kwargs = dict(
        radius=1e-7, wavelength=5e-7, refractive_index=1.5, medium_index=1.0
    )
    kwargs[argument] = np.array([kwargs[argument] + 1j])
    with pytest.raises(ValueError, match="real"):
        get_mie_efficiencies(**kwargs)
    with pytest.raises(ValueError, match="real"):
        get_mie_coefficients(1.5, np.array([1 + 1j]))


def test_extreme_zero_and_unresolved_cases():
    """Index matching remains exact; unresolved nonzero scattering fails."""
    result = get_mie_efficiencies(1e-200, 1.0, 1.0)
    assert all(value == 0 for value in result)
    with pytest.raises(RuntimeError):
        get_mie_efficiencies(1e-200, 1.0, 1.5 + 0.1j)


@pytest.mark.parametrize("coefficient", [2 + 0j, 1e200 + 0j])
def test_cpu_rejects_unphysical_or_nonfinite_efficiencies(
    monkeypatch, coefficient
):
    """A failed coefficient calculation cannot escape as valid efficiency."""
    import particula.optics.mie as mie

    def coefficients(*_):
        return MieCoefficients(
            np.array([coefficient]), np.array([0j]), np.array(1)
        )

    monkeypatch.setattr(mie, "get_mie_coefficients", coefficients)
    with pytest.raises(RuntimeError):
        get_mie_efficiencies(1 / (2 * np.pi), 1.0, 1.5 + 0.1j)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_efficiency_failure_status(device):
    """Reject negative absorption and overflow while retaining upstream status."""
    wp = pytest.importorskip("warp")
    from particula.optics.gpu.mie import mie_efficiencies_kernel

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    m = wp.array(np.tile([1.5, 0.1], (3, 1)), dtype=wp.vec2d, device=device)
    x = wp.ones(3, dtype=wp.float64, device=device)
    a = wp.array(
        np.array([[[2.0, 0]], [[1e200, 0]], [[0.0, 0]]]),
        dtype=wp.vec2d,
        device=device,
    )
    b = wp.zeros_like(a)
    count = wp.ones(3, dtype=wp.int32, device=device)
    status = wp.array(np.array([0, 0, 2]), dtype=wp.int32, device=device)
    output = wp.empty((3, 7), dtype=wp.float64, device=device)
    wp.launch(
        mie_efficiencies_kernel,
        dim=3,
        inputs=[m, x, a, b, count, status, output],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), [4, 3, 2])
    np.testing.assert_array_equal(output.numpy()[2], 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_order_boundary_and_weak_absorption(device):
    """Double constants must not inflate series length at capacity boundaries."""
    wp = pytest.importorskip("warp")
    from particula.optics.gpu import prepare_mie
    from scipy.optimize import brentq

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Native CUDA is unavailable")
    boundary = brentq(lambda x: x + 4.05 * np.cbrt(x) + 2 - 100, 50, 100)
    x = np.array([boundary - 1e-8, 1.0, 10.0])
    m = 1.33 + 1e-20j
    assert get_mie_coefficients(m, x).term_count[0] == 100
    work = prepare_mie(
        wp.array(
            np.tile([m.real, m.imag], (3, 1)), dtype=wp.vec2d, device=device
        ),
        wp.array(x, dtype=wp.float64, device=device),
        coefficient_capacity=100,
    )
    work.enqueue()
    work.check()
    actual = work.efficiencies.numpy()
    expected = get_mie_efficiencies(x / (2 * np.pi), 1.0, m)
    assert np.all(actual[:, 2] >= 0)
    assert np.all(expected.absorption >= 0)
    np.testing.assert_allclose(
        actual[:, 0], actual[:, 1] + actual[:, 2], rtol=1e-14
    )
    np.testing.assert_allclose(
        actual, np.stack(expected, -1), rtol=1e-8, atol=1e-12
    )
