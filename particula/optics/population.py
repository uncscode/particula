"""Number-weighted optical properties of independent spherical particles.

Population sums assume incoherent, independent scattering. Number weights
are densities [m⁻³], including quadrature weights for continuous radius PDFs.
Bulk asymmetry is weighted by scattering, not by particle number alone.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.optics.cross_sections import OpticalCrossSections


class PopulationOptics(NamedTuple):
    """Bulk optical coefficients and intensive properties.

    Attributes:
        extinction: Extinction coefficient [m⁻¹].
        scattering: Scattering coefficient [m⁻¹].
        absorption: Absorption coefficient [m⁻¹].
        backscatter: Radar backscatter coefficient [m⁻¹].
        radiation_pressure: Radiation-pressure coefficient [m⁻¹].
        single_scattering_albedo: Scattering/extinction, zero if undefined.
        asymmetry: Scattering-weighted cosine, zero if undefined.
        extinction_valid: Whether single-scattering albedo is defined.
        scattering_valid: Whether asymmetry is defined.
    """

    extinction: NDArray[np.float64]
    scattering: NDArray[np.float64]
    absorption: NDArray[np.float64]
    backscatter: NDArray[np.float64]
    radiation_pressure: NDArray[np.float64]
    single_scattering_albedo: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    extinction_valid: NDArray[np.bool_]
    scattering_valid: NDArray[np.bool_]


def _nonnegative(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Coerce a finite nonnegative physical array without modifying it."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def get_population_weights(
    concentration: ArrayLike,
    distribution_type: str,
    *,
    volume: ArrayLike | None = None,
    radius_grid: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Convert native population measures to number-density weights.

    Args:
        concentration: (..., particles) number density [m⁻³] for discrete,
            represented count for particle_resolved, or dn/dr [m⁻⁴] for
            continuous_pdf. The final axis is always the particle axis.
        distribution_type: discrete, particle_resolved, or continuous_pdf.
        volume: Required positive physical volume [m³] for resolved counts,
            broadcastable with concentration's leading axes. Omit otherwise.
        radius_grid: Required positive increasing 1D radius nodes [m] for
            continuous_pdf. At least two nodes; omit for other measures.

    Returns:
        New number-density weights [m⁻³]. PDF weights implement trapezoidal
        integration of the complete optical-cross-section times PDF product.

    Raises:
        ValueError: If the measure, shape, values or supplied metadata conflict.
    """
    values = _nonnegative(concentration, "Concentration")
    if values.ndim < 1:
        raise ValueError("Concentration requires a final particle axis")
    if distribution_type == "discrete":
        if volume is not None or radius_grid is not None:
            raise ValueError("Discrete density needs no volume or radius grid")
        return values.copy()
    if distribution_type == "particle_resolved":
        if volume is None or radius_grid is not None:
            raise ValueError("Resolved counts require only physical volume")
        physical_volume = _nonnegative(volume, "Volume")
        if np.any(physical_volume == 0):
            raise ValueError("Volume must be positive")
        physical_volume = np.broadcast_to(physical_volume, values.shape[:-1])
        with np.errstate(over="ignore", invalid="ignore"):
            result = values / physical_volume[..., None]
        return _nonnegative(result, "Resolved number density")
    if distribution_type == "continuous_pdf":
        if radius_grid is None or volume is not None:
            raise ValueError("Radius PDF requires only its radius grid")
        grid = _nonnegative(radius_grid, "Radius grid")
        if (
            grid.ndim != 1
            or grid.size < 2
            or grid.size != values.shape[-1]
            or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError("PDF requires matching positive increasing nodes")
        spacing = np.diff(grid)
        weights = np.zeros_like(grid)
        weights[:-1] += spacing / 2
        weights[1:] += spacing / 2
        with np.errstate(over="ignore", invalid="ignore"):
            result = values * weights
        return _nonnegative(result, "PDF number weights")
    raise ValueError("Unknown optical population measure")


def get_population_optics(
    cross_sections: OpticalCrossSections,
    asymmetry: ArrayLike,
    number_density_weights: ArrayLike,
    *,
    particle_axis: int = -1,
) -> PopulationOptics:
    """Reduce per-particle optical areas to bulk optical coefficients.

    Args:
        cross_sections: Nonnegative per-particle areas [m²].
        asymmetry: Per-particle scattering-weighted cosine in [-1,1].
        number_density_weights: Number weights [m⁻³]; use explicit singleton
            axes for wavelength broadcasting, e.g. (boxes,1,particles).
        particle_axis: Particle axis in the broadcast shape, default last.

    Returns:
        Bulk coefficients [m⁻¹], albedo, asymmetry and validity masks with
        the particle axis removed. Empty/zero populations produce zeros.

    Raises:
        ValueError: If inputs, reduction axis or resulting values are invalid.
    """
    weights = _nonnegative(number_density_weights, "Number weights")
    g = np.asarray(asymmetry, dtype=np.float64)
    if np.any(~np.isfinite(g)) or np.any(abs(g) > 1):
        raise ValueError("Asymmetry must be finite within [-1,1]")
    arrays = np.broadcast_arrays(
        *[_nonnegative(v, "Cross section") for v in cross_sections],
        g,
        weights,
    )
    if (
        not arrays[0].ndim
        or not -arrays[0].ndim <= particle_axis < arrays[0].ndim
    ):
        raise ValueError("Invalid particle axis")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        totals = [
            np.sum(v * arrays[-1], axis=particle_axis) for v in arrays[:5]
        ]
        extinction, scattering = totals[:2]
        albedo = np.divide(
            scattering,
            extinction,
            out=np.zeros_like(scattering),
            where=extinction > 0,
        )
        weighted_g = np.sum(
            arrays[1] * arrays[-2] * arrays[-1], axis=particle_axis
        )
        bulk_g = np.divide(
            weighted_g,
            scattering,
            out=np.zeros_like(scattering),
            where=scattering > 0,
        )
    if any(np.any(~np.isfinite(v)) for v in (*totals, albedo, bulk_g)):
        raise ValueError("Population optical reduction overflow")
    return PopulationOptics(
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        albedo,
        bulk_g,
        extinction > 0,
        scattering > 0,
    )


def get_population_mueller_matrix(
    differential_matrix: ArrayLike,
    number_density_weights: ArrayLike,
    *,
    particle_axis: int = -1,
) -> NDArray[np.float64]:
    """Sum independent-particle differential Mueller matrices by number.

    Args:
        differential_matrix: Per-particle matrix [m²/sr] with shape
            batch+(angles,4,4), where batch includes the particle axis.
            All matrices must use the same incident/outgoing reference frames.
        number_density_weights: Nonnegative weights [m⁻³] broadcastable with
            batch. Negative polarized matrix entries are valid.
        particle_axis: Particle axis within the broadcast batch shape.

    Returns:
        Bulk differential matrix [m⁻¹/sr]. Normalize by bulk scattering
        coefficient to obtain the population phase matrix.

    Raises:
        ValueError: If inputs or resulting matrix values are invalid.
    """
    matrix = np.asarray(differential_matrix, dtype=np.float64)
    weights = _nonnegative(number_density_weights, "Number weights")
    if (
        matrix.ndim < 4
        or matrix.shape[-2:] != (4, 4)
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("Require finite particle angular Mueller matrices")
    shape = np.broadcast_shapes(matrix.shape[:-3], weights.shape)
    if not -len(shape) <= particle_axis < len(shape):
        raise ValueError("Invalid particle axis")
    axis = particle_axis % len(shape)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = (
            matrix * np.broadcast_to(weights, shape)[..., None, None, None]
        )
        result = np.sum(weighted, axis=axis)
    if np.any(~np.isfinite(result)):
        raise ValueError("Population Mueller reduction overflow")
    return result
