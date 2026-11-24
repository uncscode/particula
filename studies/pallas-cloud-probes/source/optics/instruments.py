# %% Synthetic response for a liquid-water sphere in air
import numpy as np
from scipy.special import j0
from scipy.stats import ncx2, qmc
from scipy.optimize import isotonic_regression
from scipy.ndimage import gaussian_filter1d
from particula.optics import get_mie_coefficients, get_scattering_amplitudes

COLORS = (633, 660)
WATER = (
    1.332  # Common index deliberately isolates wavelength at fixed material.
)
BEADS = 1.50  # Representative glass/silica surrogate, varied in interpretation.
MARKERS = np.array([5.0, 9.18, 25.60])


def mie(diameter, nm, index):
    return get_mie_coefficients(index, np.pi * diameter / (nm / 1000))


def water_index(nm, temperature=278.15, density=999.97):
    # IAPWS R9-97 Eq. 1. Vacuum wavelengths; ignore small air-index correction.
    l2 = (nm / 589.0) ** 2
    t = temperature / 273.15
    r = density / 1000
    a = [
        0.244257733,
        0.00974634476,
        -0.00373234996,
        0.000268678472,
        0.00158920570,
        0.00245934259,
        0.900704920,
        -0.0166626219,
    ]
    ll = r * (
        a[0]
        + a[1] * r
        + a[2] * t
        + a[3] * l2 * t
        + a[4] / l2
        + a[5] / (l2 - 0.2292020**2)
        + a[6] / (l2 - 5.432937**2)
        + a[7] * r * r
    )
    return np.sqrt((1 + 2 * ll) / (1 - ll))


def annular_signal(d, nm, order, index=WATER):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    low, high = np.cos(np.deg2rad([12.0, 4.0]))
    mu = low + (nodes + 1) * (high - low) / 2
    a, b = get_scattering_amplitudes(mie(d, nm, index), np.arccos(mu))
    return (
        (abs(a) ** 2 + abs(b) ** 2)
        @ weights
        * ((high - low) / 2 * np.pi / (2 * np.pi / (nm / 1000)) ** 2)
    )


def pulse_lookup(nm, order, index=WATER):
    # Smooth the oscillatory response over an assumed finite calibration width
    # then use its nearest nondecreasing curve to assign increasing channels.
    grid = np.arange(0.5, 43.01, 0.025)
    signal = annular_signal(grid, nm, order, index)
    smooth = gaussian_filter1d(signal, 0.35 / 0.025, mode="nearest")
    curve = isotonic_regression(smooth).x
    return grid, signal, curve


def fssp(d, nm, order, draws, cal_color, spread, cache):
    grid, signal, _ = cache[nm]
    _, _, curve = cache[cal_color]
    amplitude = np.interp(d, grid, signal)[:, None]
    # Unit mean pulse gains; fixed Sobol quantiles shared across both colors.
    from scipy.special import ndtri

    amplitude = amplitude * np.exp(
        spread * ndtri(draws[:, 3]) - 0.5 * spread**2
    )
    uniq, start, num = np.unique(curve, return_index=True, return_counts=True)
    inferred = np.interp(
        amplitude, uniq, 0.5 * (grid[start] + grid[start + num - 1])
    )
    return np.where(amplitude >= np.interp(2.0, grid, curve), inferred, 0.0)


def geometry(nz, draws):
    # Draw uniformly in frustum volume, with z the particle-to-camera distance.
    L = 56.4
    boundaries = np.linspace(13.3, 33.6, nz + 1)
    w = -np.diff((1 - boundaries / L) ** 3)
    z = (boundaries[1:] + boundaries[:-1]) / 2
    iz = np.searchsorted(np.cumsum(w / w.sum()), draws[:, 2]).clip(0, nz - 1)
    return z, 3.45 * (1 - z / L), iz.astype(np.int32)


def profiles(d, nm, index, z, r, order, kind):
    # Radius, wavelength and diameters below are micrometres.
    lam = nm / 1000
    rayleigh = 5.1 * (z / 33.6) * (lam / 0.660)
    if kind == "shadow":
        # Equivalent in-focus dark disk blurred by a Gaussian approximating the
        # Airy intensity-core FWHM, then the kernel integrates over each pixel.
        sigma = (1.028 / 1.22) * rayleigh / 2.35482
        arr = np.empty((len(d), len(z), len(r)), np.float32)
        for j, s in enumerate(sigma):
            arr[:, j] = ncx2.cdf(
                (d[:, None] / (2 * s)) ** 2, 2, (r[None, :] / s) ** 2
            )
        return arr
    # Ideal complex-field reconstruction with circular pupil, single scattering.
    # Low-angle scalar approximation, azimuthal average of the two Mie amplitudes.
    # Far-field convention gives t-1 -> -2*pi*S/k^2 in object-plane frequency.
    # The in-focus inverse transform contributes -S sin(theta)cos(theta) dtheta.
    k = 2 * np.pi / lam
    nodes, weights = np.polynomial.legendre.leggauss(order)
    na = 0.61 * lam / rayleigh
    sine = na[:, None] * (nodes + 1) / 2
    pupil = []
    for j, v in enumerate(sine):
        pupil.append(
            (v * weights * na[j] / 2)[:, None] * j0(k * v[:, None] * r)
        )
    out = np.empty((len(d), len(z), len(r)), np.float32)
    for first in range(0, len(d), 64):
        block = d[first : first + 64]
        amps = get_scattering_amplitudes(
            mie(block, nm, index), np.arcsin(sine.ravel())
        )
        amp = (amps.perpendicular + amps.parallel).reshape(
            len(block), len(z), order
        ) / 2
        for j, hankel in enumerate(pupil):
            field = 1 - amp[:, j] @ hankel
            out[first : first + len(block), j] = 1 - abs(field) ** 2
    return out


def sobol(n, seed=816):
    return qmc.Sobol(4, scramble=True, seed=seed).random_base2(int(np.log2(n)))
