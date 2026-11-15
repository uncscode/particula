"""Dimensional and normalized optical quantities for individual spheres."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.optics.mie import MieEfficiencies


class OpticalCrossSections(NamedTuple):
    """Per-particle optical areas in m², with the broadcast input shape.

    Attributes:
        extinction: Extinction cross section [m²].
        scattering: Scattering cross section [m²].
        absorption: Absorption cross section [m²].
        backscatter: Radar backscatter cross section [m²], not m²/sr.
        radiation_pressure: Radiation-pressure cross section [m²].
    """

    extinction: NDArray[np.float64]
    scattering: NDArray[np.float64]
    absorption: NDArray[np.float64]
    backscatter: NDArray[np.float64]
    radiation_pressure: NDArray[np.float64]


def get_cross_sections(
    efficiencies: MieEfficiencies, radius: ArrayLike
) -> OpticalCrossSections:
    """Multiply efficiencies by geometric area pi*r² without recalculating.

    Args:
        efficiencies: Output of get_mie_efficiencies for the same radius.
        radius: Nonnegative finite radius [m], broadcastable with efficiencies.

    Returns:
        Named cross sections [m²]. Inputs are not mutated.

    Raises:
        ValueError: If the radius is negative or nonfinite.
    """
    r = np.asarray(radius, dtype=np.float64)
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("Radius must be finite and nonnegative")
    area = np.pi * r * r
    return OpticalCrossSections(
        *(
            area * value
            for value in (
                efficiencies.extinction,
                efficiencies.scattering,
                efficiencies.absorption,
                efficiencies.backscatter,
                efficiencies.radiation_pressure,
            )
        )
    )


def get_differential_mueller_matrix(
    matrix: ArrayLike,
    wavelength: ArrayLike,
    medium_index: ArrayLike = 1.0,
) -> NDArray[np.float64]:
    """Convert unnormalized angular Mueller values to m²/sr.

    Args:
        matrix: Raw Mueller matrix shaped batch+(angles,4,4).
        wavelength: Positive vacuum wavelength [m], broadcastable with batch.
        medium_index: Positive real medium index, broadcastable with batch.

    Returns:
        Matrix divided by (2*pi*n_medium/lambda_vacuum)² [m²/sr].

    Raises:
        ValueError: If shapes or physical values are invalid.
    """
    array = np.asarray(matrix, dtype=np.float64)
    wave, medium = np.broadcast_arrays(
        np.asarray(wavelength, dtype=float),
        np.asarray(medium_index, dtype=float),
    )
    if (
        array.ndim < 3
        or array.shape[-2:] != (4, 4)
        or np.any(~np.isfinite(array))
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(medium))
        or np.any(wave <= 0)
        or np.any(medium <= 0)
    ):
        raise ValueError(
            "Require angular matrices and finite wavelength/medium>0"
        )
    return array * (wave / (2 * np.pi * medium))[..., None, None, None] ** 2


def get_phase_matrix(
    differential_matrix: ArrayLike, scattering_cross_section: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize a differential Mueller matrix by total scattering.

    Args:
        differential_matrix: Angular matrix [m²/sr], batch+(angles,4,4).
        scattering_cross_section: Nonnegative total scattering [m²],
            broadcastable with batch, evaluated for the same particle.

    Returns:
        Phase matrix [sr⁻¹] and batch validity mask. Element 11 integrates
        to one over solid angle for valid cases. Zero scattering returns a
        zero matrix and False; this is not a uniform-scattering assumption.

    Raises:
        ValueError: If shapes or physical values are invalid.
    """
    array = np.asarray(differential_matrix, dtype=np.float64)
    cross = np.asarray(scattering_cross_section, dtype=np.float64)
    if (
        array.ndim < 3
        or array.shape[-2:] != (4, 4)
        or np.any(~np.isfinite(array))
        or np.any(~np.isfinite(cross))
        or np.any(cross < 0)
    ):
        raise ValueError("Require finite angular matrices and scattering>=0")
    shape = np.broadcast_shapes(array.shape[:-3], cross.shape)
    valid = np.broadcast_to(cross > 0, shape)
    denominator = np.broadcast_to(cross, shape)[..., None, None, None]
    result = np.zeros(shape + array.shape[-3:], dtype=np.float64)
    np.divide(array, denominator, out=result, where=denominator > 0)
    return result, valid


class DifferentialCrossSections(NamedTuple):
    """Angular scattering areas per solid angle in m²/sr.

    Attributes:
        perpendicular: Perpendicular incident polarization, |S1|²/k².
        parallel: Parallel incident polarization, |S2|²/k².
        unpolarized: Mean of the two polarized cross sections.
    """

    perpendicular: NDArray[np.float64]
    parallel: NDArray[np.float64]
    unpolarized: NDArray[np.float64]


def get_differential_cross_sections(
    perpendicular: ArrayLike,
    parallel: ArrayLike,
    wavelength: ArrayLike,
    medium_index: ArrayLike = 1.0,
) -> DifferentialCrossSections:
    """Convert S1/S2 amplitudes to polarized differential cross sections.

    Uses the nonabsorbing-medium wave number k=2*pi*n_medium/lambda_vacuum,
    following Bohren and Huffman (1983), chapter 4,
    https://doi.org/10.1002/9783527618156.

    Args:
        perpendicular: Dimensionless S1, shaped batch+(angles,).
        parallel: Dimensionless S2, same shape as perpendicular.
        wavelength: Positive finite vacuum wavelength [m]; broadcasts with
            batch axes, not the angle axis.
        medium_index: Positive finite real medium index, default one;
            broadcasts with batch axes.

    Returns:
        Perpendicular, parallel and unpolarized differential scattering
        cross sections [m²/sr], shaped broadcast batch+(angles,).
        Inputs are unchanged; scalar angles must retain a length-one axis.

    Raises:
        ValueError: If amplitudes, shapes or physical inputs are invalid.
        RuntimeError: If dimensional cross sections overflow.
    """
    s1 = np.asarray(perpendicular, dtype=np.complex128)
    s2 = np.asarray(parallel, dtype=np.complex128)
    if np.iscomplexobj(wavelength) or np.iscomplexobj(medium_index):
        raise ValueError("Wavelength and medium index must be real")
    wave = np.asarray(wavelength, dtype=np.float64)
    medium = np.asarray(medium_index, dtype=np.float64)
    if (
        s1.ndim < 1
        or s1.shape != s2.shape
        or np.any(~np.isfinite(s1))
        or np.any(~np.isfinite(s2))
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(medium))
        or np.any(wave <= 0)
        or np.any(medium <= 0)
    ):
        raise ValueError("Require finite angular amplitudes and wave/medium>0")
    shape = np.broadcast_shapes(s1.shape[:-1], wave.shape, medium.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        length = np.broadcast_to(wave / medium / (2 * np.pi), shape)
        # Scale amplitudes before squaring to avoid avoidable overflow.
        per = np.asarray(abs(s1 * length[..., None]) ** 2, dtype=np.float64)
        par = np.asarray(abs(s2 * length[..., None]) ** 2, dtype=np.float64)
        unpolarized = per / 2 + par / 2
    if np.any(~np.isfinite(per)) or np.any(~np.isfinite(par)):
        raise RuntimeError("Nonfinite differential cross sections")
    return DifferentialCrossSections(per, par, unpolarized)
