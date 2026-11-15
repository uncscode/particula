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
