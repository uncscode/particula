"""Independent limits and pinned reference checks for Mie calculations."""

import json
from pathlib import Path

import numpy as np
import pytest
from particula.optics import get_mie_coefficients, get_mie_efficiencies
from scipy.special import spherical_jn, spherical_yn

REFERENCE = json.loads(
    (Path(__file__).parent / "data/pymiescatt_spheres.json").read_text()
)


@pytest.mark.parametrize(
    "case", [c for c in REFERENCE["cases"] if c["size_parameter"] >= 0.051]
)
def test_reference_efficiencies(case):
    """Compare each physical efficiency independently through x=10000."""
    m = complex(*case["index"])
    result = get_mie_efficiencies(case["diameter_nm"] * 0.5e-9, 500e-9, m)
    expected = case["efficiencies"]
    known_reference_error = (
        case["index"] in ([1.33, 0.0], [1.33, 1e-8])
        and case["size_parameter"] in (100.0, 1000.0, 3000.0, 10000.0)
    ) or (case["index"] == [1.5, 0.01] and case["size_parameter"] == 100.0)
    if known_reference_error:
        independent = _bessel_oracle(m, case["size_parameter"])
        assert not np.allclose(
            expected[:3], independent[:3], rtol=1e-6, atol=1e-12
        )
        expected = independent
    for actual, index in zip(result[:6], [0, 1, 2, 3, 5, 4], strict=True):
        np.testing.assert_allclose(
            actual, expected[index], rtol=1e-6, atol=1e-12
        )


def test_rayleigh_and_zero_limits():
    """Verify independent leading-order dipole results and exact zeros."""
    x = 1e-5
    m = 1.5 + 0.1j
    ll = (m * m - 1) / (m * m + 2)
    result = get_mie_efficiencies(x / (2 * np.pi), 1.0, m)
    np.testing.assert_allclose(
        result.scattering, 8 / 3 * x**4 * abs(ll) ** 2, rtol=1e-7
    )
    np.testing.assert_allclose(result.absorption, 4 * x * ll.imag, rtol=1e-7)
    result = get_mie_efficiencies([0.0, 1.0], 1.0, [1.5, 1.0])
    assert np.all(result.extinction == 0)
    assert not np.any(result.scattering_valid)


def test_medium_equivalence_and_broadcasting():
    """Medium normalization is applied once, with explicit outer axes."""
    radii = np.array([1e-8, 1e-7, 1e-6])[:, None]
    wavelengths = np.array([400e-9, 600e-9])[None, :]
    actual = get_mie_efficiencies(radii, wavelengths, 1.5 + 0.1j, 1.33)
    expected = get_mie_efficiencies(
        radii, wavelengths / 1.33, (1.5 + 0.1j) / 1.33
    )
    assert actual.extinction.shape == (3, 2)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "index,x",
    [
        (1.5, -1),
        (1.5, 10001),
        (1.5 - 0.1j, 1),
        (0, 1),
        (np.nan, 1),
        (1.5, np.inf),
    ],
)
def test_invalid_inputs(index, x):
    """Reject nonphysical or unsupported coefficient inputs."""
    with pytest.raises(ValueError):
        get_mie_coefficients(index, x)


def _bessel_oracle(m, x):
    """Evaluate B&H coefficients directly with independent SciPy functions."""
    n = np.arange(1, int(np.ceil(x + 4.05 * np.cbrt(x) + 2)) + 1)
    j = spherical_jn(n, x)
    jp = spherical_jn(n, x, derivative=True)
    h = j + 1j * spherical_yn(n, x)
    hp = jp + 1j * spherical_yn(n, x, derivative=True)
    jm = spherical_jn(n, m * x)
    jmp = spherical_jn(n, m * x, derivative=True)
    psi, psip = x * j, j + x * jp
    xi, xip = x * h, h + x * hp
    pm, pmp = m * x * jm, jm + m * x * jmp
    a = (m * pm * psip - psi * pmp) / (m * pm * xip - xi * pmp)
    b = (pm * psip - m * psi * pmp) / (pm * xip - m * xi * pmp)
    ext = 2 / x**2 * np.sum((2 * n + 1) * (a + b).real)
    sca = 2 / x**2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    adjacent = np.sum(
        n[:-1]
        * (n[:-1] + 2)
        / (n[:-1] + 1)
        * (a[:-1] * a[1:].conj() + b[:-1] * b[1:].conj()).real
    )
    cross = np.sum((2 * n + 1) / (n * (n + 1)) * (a * b.conj()).real)
    g = 4 / (x * x * sca) * (adjacent + cross)
    back = abs(np.sum((2 * n + 1) * (-1.0) ** n * (a - b))) ** 2 / x**2
    return ext, sca, ext - sca, g, ext - g * sca, back
