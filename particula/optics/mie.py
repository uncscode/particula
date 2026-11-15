"""Independent homogeneous-sphere Mie calculations in SI units.

The coefficient equations follow Bohren and Huffman (1983), chapter 4,
https://doi.org/10.1002/9783527618156. Logarithmic derivatives use a bounded
continued fraction followed by downward recurrence; external regular Bessel
functions use a convergent power series for small arguments.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MieCoefficients(NamedTuple):
    """Complex coefficients, zero padded along the final order axis.

    Attributes:
        electric: Electric coefficients a_n, starting at order one.
        magnetic: Magnetic coefficients b_n, starting at order one.
        term_count: Valid number of coefficients for each broadcast case.
    """

    electric: NDArray[np.complex128]
    magnetic: NDArray[np.complex128]
    term_count: NDArray[np.int64]


class MieEfficiencies(NamedTuple):
    """Dimensionless efficiencies and asymmetry for broadcast input cases.

    Attributes:
        extinction: Total extinction efficiency.
        scattering: Scattering efficiency.
        absorption: Absorption efficiency.
        asymmetry: Scattering-weighted cosine, zero when undefined.
        backscatter: Radar-convention backscatter efficiency.
        radiation_pressure: Extinction minus asymmetry times scattering.
        scattering_valid: Whether asymmetry is defined.
    """

    extinction: NDArray[np.float64]
    scattering: NDArray[np.float64]
    absorption: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    backscatter: NDArray[np.float64]
    radiation_pressure: NDArray[np.float64]
    scattering_valid: NDArray[np.bool_]


def _log_derivative(z: complex, order: int) -> complex:
    """Evaluate D_order(z) using the modified Lentz continued fraction."""
    tiny = 1e-300
    value = (2 * order + 1) / z
    c_value = value
    d_value = 0j
    for index in range(1, 10001):
        denominator = (2 * (order + index) + 1) / z
        d_value = denominator - d_value
        c_value = denominator - 1 / c_value
        if abs(d_value) < tiny:
            d_value = complex(tiny)
        if abs(c_value) < tiny:
            c_value = complex(tiny)
        d_value = 1 / d_value
        delta = c_value * d_value
        value *= delta
        if abs(delta - 1) < 2e-14:
            return value - order / z
    raise RuntimeError("Mie logarithmic derivative did not converge")


def _regular_small(x: float, order: int) -> float:
    """Evaluate x*j_order(x) by its convergent small-argument series."""
    leading = x
    for n in range(1, order + 1):
        leading *= x / (2 * n + 1)
    term = leading
    value = term
    for k in range(1, 100):
        term *= -x * x / (2 * k * (2 * order + 2 * k + 1))
        value += term
        if abs(term) <= abs(value) * 2e-16:
            return value
    raise RuntimeError("Regular Bessel series did not converge")


def _coefficients(m: complex, x: float, count: int):
    """Calculate one admitted case without allocating broadcast outputs."""
    electric = np.zeros(count, dtype=np.complex128)
    magnetic = np.zeros(count, dtype=np.complex128)
    if x == 0 or m == 1:
        return electric, magnetic
    derivatives = np.empty(count + 1, dtype=np.complex128)
    derivatives[count] = _log_derivative(m * x, count)
    for n in range(count, 0, -1):
        ratio = n / (m * x)
        derivatives[n - 1] = ratio - 1 / (derivatives[n] + ratio)
    previous_psi = np.sin(x)
    psi = np.sin(x) / x - np.cos(x)
    previous_chi = np.cos(x)
    chi = np.cos(x) / x + np.sin(x)
    for n in range(1, count + 1):
        if x < 1:
            previous_psi = _regular_small(x, n - 1)
            psi = _regular_small(x, n)
        previous_xi = previous_psi - 1j * previous_chi
        xi = psi - 1j * chi
        da = derivatives[n] / m + n / x
        db = derivatives[n] * m + n / x
        electric[n - 1] = (da * psi - previous_psi) / (da * xi - previous_xi)
        magnetic[n - 1] = (db * psi - previous_psi) / (db * xi - previous_xi)
        previous_psi, psi = psi, (2 * n + 1) * psi / x - previous_psi
        previous_chi, chi = chi, (2 * n + 1) * chi / x - previous_chi
    if not (np.all(np.isfinite(electric)) and np.all(np.isfinite(magnetic))):
        raise RuntimeError("Nonfinite Mie coefficients")
    return electric, magnetic


def get_mie_coefficients(
    relative_refractive_index: ArrayLike, size_parameter: ArrayLike
) -> MieCoefficients:
    """Calculate external coefficients for nonmagnetic homogeneous spheres.

    Args:
        relative_refractive_index: Particle/medium index n+ik, n>0, k>=0.
        size_parameter: Dimensionless 2*pi*r*n_medium/lambda_vacuum, 0..10000.

    Returns:
        Broadcast coefficients with a final, zero-padded order axis and
        per-case valid term counts. Inputs are not mutated.

    Raises:
        ValueError: If physical inputs are nonfinite or outside the domain.
        RuntimeError: If the numerical recurrence fails.
    """
    m, x = np.broadcast_arrays(
        np.asarray(relative_refractive_index, dtype=np.complex128),
        np.asarray(size_parameter, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(m))
        or np.any(~np.isfinite(x))
        or np.any(m.real <= 0)
        or np.any(m.imag < 0)
        or np.any(x < 0)
        or np.any(x > 10000)
    ):
        raise ValueError("Require finite n>0, k>=0 and 0<=x<=10000")
    counts = np.ceil(x + 4.05 * np.cbrt(x) + 2).astype(np.int64)
    counts = np.where((x == 0) | (m == 1), 0, counts)
    width = int(np.max(counts, initial=0))
    a = np.zeros(m.shape + (width,), dtype=np.complex128)
    b = np.zeros_like(a)
    for index in np.ndindex(m.shape):
        count = int(counts[index])
        if count:
            a[index][:count], b[index][:count] = _coefficients(
                complex(m[index]), float(x[index]), count
            )
    return MieCoefficients(a, b, counts)


def get_mie_efficiencies(
    radius: ArrayLike,
    wavelength: ArrayLike,
    refractive_index: ArrayLike,
    medium_index: ArrayLike = 1.0,
) -> MieEfficiencies:
    """Calculate efficiencies from SI radius and vacuum wavelength.

    Args:
        radius: Nonnegative sphere radius [m].
        wavelength: Positive vacuum wavelength [m].
        refractive_index: Absolute particle refractive index n+ik.
        medium_index: Positive real nonabsorbing medium index, default 1.

    Returns:
        Named dimensionless arrays with the broadcast input shape. Zero
        scattering gives asymmetry zero and scattering_valid False.

    Raises:
        ValueError: If inputs violate the physical domain or x exceeds 10000.
        RuntimeError: If coefficient calculation fails.
    """
    r, wave, m, medium = np.broadcast_arrays(
        np.asarray(radius, dtype=np.float64),
        np.asarray(wavelength, dtype=np.float64),
        np.asarray(refractive_index, dtype=np.complex128),
        np.asarray(medium_index, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(medium))
        or np.any(r < 0)
        or np.any(wave <= 0)
        or np.any(medium <= 0)
    ):
        raise ValueError("Require finite radius>=0, wavelength>0, medium>0")
    x = 2 * np.pi * r * medium / wave
    coefficients = get_mie_coefficients(m / medium, x)
    a, b = coefficients.electric, coefficients.magnetic
    n = np.arange(1, a.shape[-1] + 1, dtype=np.float64)
    factor = np.divide(2.0, x * x, out=np.zeros_like(x), where=x > 0)
    ext = factor * np.sum((2 * n + 1) * (a + b).real, axis=-1)
    sca = factor * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2), axis=-1)
    adjacent = np.sum(
        (n[:-1] * (n[:-1] + 2) / (n[:-1] + 1))
        * (
            a[..., :-1] * a[..., 1:].conj() + b[..., :-1] * b[..., 1:].conj()
        ).real,
        axis=-1,
    )
    cross = np.sum((2 * n + 1) / (n * (n + 1)) * (a * b.conj()).real, axis=-1)
    g = np.divide(
        2 * factor * (adjacent + cross),
        sca,
        out=np.zeros_like(sca),
        where=sca > 0,
    )
    back = (
        factor
        / 2
        * abs(np.sum((2 * n + 1) * (-1.0) ** n * (a - b), axis=-1)) ** 2
    )
    absorption = ext - sca
    # Lossless spheres have identically zero absorption; avoid cancellation.
    absorption = np.where(m.imag == 0, 0.0, absorption)
    ext = np.where(m.imag == 0, sca, ext)
    return MieEfficiencies(
        ext, sca, absorption, g, back, ext - g * sca, sca > 0
    )
