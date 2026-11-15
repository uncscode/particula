"""Independent large-sphere angular oracle using AMOS and Jacobi polynomials.

Scaled cylindrical Bessel functions avoid overflow at large imaginary m*x.
P_n'(mu) = (n+1)/2 P_(n-1)^(1,1)(mu) supplies the angular function without
using the production pi/tau recurrence. Both are SciPy special functions:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jve.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from particula.optics import get_mie_coefficients, get_scattering_amplitudes
from scipy.special import (
    eval_jacobi,
    eval_legendre,
    jve,
    spherical_jn,
    spherical_yn,
)

ANGLES = np.array(
    [0, 1e-6, 1e-4, 1e-3, 0.01, 0.5, np.pi / 2, np.pi - 1e-4, np.pi]
)


@lru_cache(maxsize=4)
def _angular_basis(x):
    """Compute independent polynomial values once per series length."""
    # Extra terms independently check the production truncation as well.
    n = np.arange(1, int(np.ceil(x + 4.05 * np.cbrt(x) + 2)) + 11)
    mu = np.cos(ANGLES)[None, :]
    order = n[:, None]
    pi = (order + 1) / 2 * eval_jacobi(order - 1, 1, 1, mu)
    tau = order * (order + 1) * eval_legendre(order, mu) - mu * pi
    return n, pi, tau


@lru_cache(maxsize=12)
def _amplitudes(m, x):
    """Use scaled AMOS ratios and external spherical Bessel values."""
    if m == 0.2 + 3j and x == 10000.0:
        fixture = json.loads(
            (
                Path(__file__).parent / "data/high_precision_angular.json"
            ).read_text()
        )
        np.testing.assert_array_equal(fixture["angles_radians"], ANGLES)
        parts = np.asarray(fixture["amplitudes"])
        return parts[..., 0] + 1j * parts[..., 1]
    n, pi, tau = _angular_basis(x)
    z = m * x
    # psi_n'(z)/psi_n(z); the shared exponential scaling cancels.
    derivative = jve(n - 0.5, z) / jve(n + 0.5, z) - n / z
    j, jp = spherical_jn(n, x), spherical_jn(n, x, True)
    y, yp = spherical_yn(n, x), spherical_yn(n, x, True)
    psi, psip = x * j, j + x * jp
    xi, xip = x * (j + 1j * y), j + 1j * y + x * (jp + 1j * yp)
    a = (m * psip - derivative * psi) / (m * xip - derivative * xi)
    b = (psip - m * derivative * psi) / (xip - m * derivative * xi)
    factor = (2 * n + 1) / (n * (n + 1))
    s1 = np.sum(factor[:, None] * (a[:, None] * pi + b[:, None] * tau), 0)
    s2 = np.sum(factor[:, None] * (a[:, None] * tau + b[:, None] * pi), 0)
    return np.stack((s1, s2), axis=-1)


@pytest.mark.parametrize("x", [100.0, 1000.0, 3000.0, 10000.0])
@pytest.mark.parametrize("m", [1.33 + 1e-8j, 1.5 + 0.1j, 0.2 + 3j])
@pytest.mark.parametrize("backend", ["numpy", "cpu", "cuda:0"])
def test_large_angular_independent(m, x, backend):
    """Qualify both amplitudes against a different numerical construction."""
    if backend == "numpy":
        actual = np.stack(
            get_scattering_amplitudes(get_mie_coefficients(m, x), ANGLES), -1
        )
    else:
        wp = pytest.importorskip("warp")
        from particula.optics.gpu import prepare_mie

        if backend.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("Native CUDA is unavailable")
        work = prepare_mie(
            wp.array([[m.real, m.imag]], dtype=wp.vec2d, device=backend),
            wp.array([x], dtype=wp.float64, device=backend),
            angles=wp.array(ANGLES, dtype=wp.float64, device=backend),
            angle_tile=3,
        )
        work.enqueue()
        work.check()
        amplitudes = []
        for value in (work.perpendicular, work.parallel):
            parts = value.numpy()[0]
            amplitudes.append(parts[:, 0] + 1j * parts[:, 1])
        actual = np.stack(amplitudes, -1)
    expected = _amplitudes(m, x)
    assert np.all(np.isfinite(expected))
    # Per-case peak scaling is the planned angular criterion near zeros.
    scale = np.max(abs(expected))
    np.testing.assert_allclose(
        actual / scale, expected / scale, rtol=0, atol=1e-8
    )
    # Also constrain each non-forward direction, so its weaker intensity is
    # not hidden by the forward peak. These cases avoid exact amplitude zeros.
    np.testing.assert_allclose(actual[4:], expected[4:], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("x", [1000.0, 10000.0])
@pytest.mark.parametrize("size_per_panel", [20, 40])
def test_large_sphere_angular_integral(x, size_per_panel):
    """Gauss quadrature resolves the forward peak and recovers Csca and g."""
    from particula.optics import get_mie_efficiencies
    from scipy.special import roots_legendre

    m = 1.33 + 1e-8j
    coefficients = get_mie_coefficients(m, x)
    # Composite quadrature in theta avoids the inaccurate endpoint weights
    # of a single >10000-node Legendre rule in cos(theta).
    nodes, weights = roots_legendre(128)
    panels = int(np.ceil(x / size_per_panel))
    width = np.pi / panels
    theta = (np.arange(panels)[:, None] + (nodes + 1) / 2) * width
    weights = np.broadcast_to(weights * width / 2, theta.shape).ravel()
    theta = theta.ravel()
    weights *= np.sin(theta)
    s1, s2 = get_scattering_amplitudes(coefficients, theta)
    intensity = (abs(s1) ** 2 + abs(s2) ** 2) / 2
    integral = np.sum(weights * intensity)
    efficiencies = get_mie_efficiencies(x / (2 * np.pi), 1.0, m)
    np.testing.assert_allclose(
        2 * integral / x**2, efficiencies.scattering, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.sum(weights * np.cos(theta) * intensity) / integral,
        efficiencies.asymmetry,
        rtol=1e-8,
    )
