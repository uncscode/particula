"""Angular reference checks and independent polarization/energy identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre, leggauss
from particula.optics.angular import get_scattering_amplitudes
from particula.optics.cross_sections import (
    get_cross_sections,
    get_differential_mueller_matrix,
    get_phase_matrix,
)
from particula.optics.mie import get_mie_coefficients, get_mie_efficiencies
from particula.optics.polarization import (
    apply_mueller_matrix,
    get_mueller_matrix,
    rotate_stokes,
)
from scipy.special import spherical_jn, spherical_yn

DATA = json.loads(
    (Path(__file__).parent / "data/pymiescatt_spheres.json").read_text()
)


@pytest.mark.parametrize(
    "case", [c for c in DATA["cases"] if 0.5 <= c["size_parameter"] <= 50]
)
def test_raw_angular_reference(case):
    """Match complex amplitudes and all independent Mueller elements."""
    coeff = get_mie_coefficients(
        complex(*case["index"]), case["size_parameter"]
    )
    s1, s2 = get_scattering_amplitudes(coeff, DATA["angles_radians"])
    reference = np.asarray(case["amplitudes"])
    reference = reference[..., 0] + 1j * reference[..., 1]
    np.testing.assert_allclose(
        np.stack([s1, s2], axis=-1) / np.max(abs(reference)),
        reference / np.max(abs(reference)),
        rtol=0,
        atol=1e-6,
    )
    matrix = get_mueller_matrix(s1, s2)
    actual = np.stack(
        [
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 2, 2],
            matrix[..., 2, 3],
        ],
        axis=-1,
    )
    reference = np.asarray(case["matrix_elements"])[..., 0]
    np.testing.assert_allclose(
        actual / np.max(abs(reference)),
        reference / np.max(abs(reference)),
        rtol=0,
        atol=1e-6,
    )
    if case["size_parameter"] == 50:
        independent = _independent_amplitudes(
            complex(*case["index"]), 50.0, np.asarray(DATA["angles_radians"])
        )
        np.testing.assert_allclose(
            np.stack([s1, s2], axis=-1), independent, rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("x", [1e-5, 0.05, 0.5, 1.0, 10.0, 100.0])
def test_angular_energy_and_asymmetry(x):
    """Independent solid-angle quadrature recovers cross section and g."""
    radius, wavelength, medium = x / (2 * np.pi * 1.33), 1.0, 1.33
    m = 1.5 + 0.01j
    eff = get_mie_efficiencies(radius, wavelength, m, medium)
    cross = get_cross_sections(eff, radius)
    mu, weight = leggauss(max(40, int(2 * x + 40)))
    coeff = get_mie_coefficients(m / medium, x)
    s1, s2 = get_scattering_amplitudes(coeff, np.arccos(mu))
    differential = get_differential_mueller_matrix(
        get_mueller_matrix(s1, s2), wavelength, medium
    )
    intensity = differential[..., 0, 0]
    integrated = 2 * np.pi * np.sum(weight * intensity)
    np.testing.assert_allclose(integrated, cross.scattering, rtol=1e-9)
    np.testing.assert_allclose(
        2 * np.pi * np.sum(weight * mu * intensity) / integrated,
        eff.asymmetry,
        rtol=1e-7,
        atol=1e-12,
    )
    phase, valid = get_phase_matrix(differential, cross.scattering)
    assert valid
    np.testing.assert_allclose(
        2 * np.pi * np.sum(weight * phase[..., 0, 0]), 1.0, rtol=1e-9
    )


def test_jones_stokes_equivalence():
    """Independent complex-field propagation fixes circular sign convention."""
    field = np.array([0.7 + 0.2j, -0.3 + 0.5j])

    def stokes(e):
        z = e[0] * e[1].conjugate()
        return np.array(
            [
                abs(e[0]) ** 2 + abs(e[1]) ** 2,
                abs(e[0]) ** 2 - abs(e[1]) ** 2,
                2 * z.real,
                -2 * z.imag,
            ]
        )

    s1, s2 = 0.4 + 0.8j, -0.2 + 0.3j
    matrix = get_mueller_matrix(s1, s2)
    actual = apply_mueller_matrix(matrix, stokes(field))
    expected = stokes(field * np.array([s2, s1]))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(
        actual[0] ** 2, np.sum(actual[1:] ** 2), atol=1e-15
    )
    angles = np.array([0.0, 0.3, np.pi / 2, np.pi])
    rotated = rotate_stokes(stokes(field), angles)
    np.testing.assert_allclose(
        rotate_stokes(rotated, -angles),
        np.broadcast_to(stokes(field), (4, 4)),
        atol=1e-15,
    )


def test_zero_and_broadcast_angular_cases():
    """Zero scattering is marked undefined without NaNs or squeezed angles."""
    coeff = get_mie_coefficients([[1.5], [1.0]], [0.0, 1.0, 2.0])
    s1, s2 = get_scattering_amplitudes(coeff, 0.0)
    assert s1.shape == (2, 3, 1)
    assert np.all(s1[1] == 0)
    np.testing.assert_allclose(s1, s2)
    phase, valid = get_phase_matrix(
        get_mueller_matrix(s1, s2), [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    assert not np.any(valid[1])
    assert np.all(phase[1] == 0)


@pytest.mark.parametrize("angle", [-0.1, np.pi + 0.1, np.nan, [[0.0]]])
def test_invalid_angles(angle):
    """Reject invalid angular coordinates before accumulation."""
    with pytest.raises(ValueError):
        get_scattering_amplitudes(get_mie_coefficients(1.5, 1.0), angle)


def _independent_amplitudes(m, x, theta):
    """Direct Bessel functions and Legendre derivatives avoid both recurrences."""
    n = np.arange(1, int(np.ceil(x + 4.05 * np.cbrt(x) + 2)) + 1)
    j, jp = spherical_jn(n, x), spherical_jn(n, x, True)
    h = j + 1j * spherical_yn(n, x)
    hp = jp + 1j * spherical_yn(n, x, True)
    jm, jmp = spherical_jn(n, m * x), spherical_jn(n, m * x, True)
    psi, psip, xi, xip = x * j, j + x * jp, x * h, h + x * hp
    pm, pmp = m * x * jm, jm + m * x * jmp
    a = (m * pm * psip - psi * pmp) / (m * pm * xip - xi * pmp)
    b = (pm * psip - m * psi * pmp) / (pm * xip - m * xi * pmp)
    mu = np.cos(theta)
    s1, s2 = np.zeros_like(mu, dtype=complex), np.zeros_like(mu, dtype=complex)
    for i, order in enumerate(n):
        polynomial = Legendre.basis(int(order))
        pi = polynomial.deriv()(mu)
        tau = order * (order + 1) * polynomial(mu) - mu * pi
        scale = (2 * order + 1) / (order * (order + 1))
        s1 += scale * (a[i] * pi + b[i] * tau)
        s2 += scale * (a[i] * tau + b[i] * pi)
    return np.stack([s1, s2], axis=-1)
