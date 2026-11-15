"""Angular Mie amplitudes using reusable external field coefficients.

Angular recurrences follow Bohren and Huffman (1983), chapter 4,
https://doi.org/10.1002/9783527618156. Angles are radians; amplitudes use
unnormalized S1/S2 so integral S11 dOmega = pi*x²*Qsca.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.optics.mie import MieCoefficients


class ScatteringAmplitudes(NamedTuple):
    """Dimensionless complex amplitudes with a final angle axis.

    Attributes:
        perpendicular: S1 amplitude perpendicular to the scattering plane.
        parallel: S2 amplitude parallel to the scattering plane.
    """

    perpendicular: NDArray[np.complex128]
    parallel: NDArray[np.complex128]


def get_scattering_amplitudes(
    coefficients: MieCoefficients, angle: ArrayLike
) -> ScatteringAmplitudes:
    """Calculate S1 and S2 without repeating the coefficient calculation.

    Args:
        coefficients: Zero-padded results from get_mie_coefficients.
        angle: Scalar or 1D scattering angles [rad] within [0, pi].

    Returns:
        Complex arrays shaped coefficient batch shape plus (n_angles,).
        Scalar angle retains a length-one axis. Inputs are unchanged.

    Raises:
        ValueError: If angles, coefficient shapes or values are invalid.
    """
    theta = np.atleast_1d(np.asarray(angle, dtype=np.float64))
    if (
        theta.ndim != 1
        or np.any(~np.isfinite(theta))
        or np.any(theta < 0)
        or np.any(theta > np.pi)
    ):
        raise ValueError("Angles must be scalar or 1D within [0, pi]")
    a, b, counts = coefficients
    if (
        a.ndim < 1
        or a.shape != b.shape
        or a.shape[:-1] != counts.shape
        or np.any(~np.isfinite(a))
        or np.any(~np.isfinite(b))
        or np.any(counts < 0)
        or np.any(counts > a.shape[-1])
    ):
        raise ValueError("Invalid Mie coefficient arrays")
    mu = np.cos(theta)
    shape = a.shape[:-1] + theta.shape
    s1 = np.zeros(shape, dtype=np.complex128)
    s2 = np.zeros_like(s1)
    previous = np.zeros_like(mu)
    current = np.ones_like(mu)
    for n in range(1, a.shape[-1] + 1):
        tau = n * mu * current - (n + 1) * previous
        weight = (2 * n + 1) / (n * (n + 1))
        an = np.where(counts >= n, a[..., n - 1], 0)[..., None]
        bn = np.where(counts >= n, b[..., n - 1], 0)[..., None]
        s1 += weight * (an * current + bn * tau)
        s2 += weight * (an * tau + bn * current)
        previous, current = (
            current,
            ((2 * n + 1) * mu * current - (n + 1) * previous) / n,
        )
    return ScatteringAmplitudes(s1, s2)
