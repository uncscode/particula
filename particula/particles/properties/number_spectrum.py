"""Number-density spectra on caller-supplied diameter edges."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.particles.particle_data import ParticleData
from particula.particles.slot_management import get_slot_diagnostics


class NumberSpectrum(NamedTuple):
    """Integrated densities [m⁻³] on the last axis and two per-row tails."""

    bin_number_density: NDArray[np.float64]
    below_range: NDArray[np.float64]
    above_range: NDArray[np.float64]


def _real(values: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError("Spectrum inputs must be real")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Spectrum inputs must be finite")
    return array


def _edges(diameter_edges: ArrayLike) -> NDArray[np.float64]:
    edges = _real(diameter_edges)
    if (
        edges.ndim != 1
        or edges.size < 2
        or edges[0] <= 0
        or np.any(edges[1:] <= edges[:-1])
    ):
        raise ValueError("Require positive, strictly increasing diameter edges")
    return edges


def _log_width(low, high):
    # log1p resolves adjacent floats even when their logarithms coincide.
    # Subtract logs for wide ratios that would overflow before taking a log.
    with np.errstate(over="ignore"):
        relative = (high - low) / low
    return np.where(
        relative < 1,
        np.log1p(relative) / np.log(10.0),
        np.log10(high) - np.log10(low),
    )


def _bin_values(values, edges):
    array = _real(values)
    if array.ndim == 0 or array.shape[-1] != edges.size - 1:
        raise ValueError("Last axis must contain one value per diameter bin")
    if np.any(array < 0):
        raise ValueError("Number density must be nonnegative")
    return array


def get_number_per_decade(
    bin_number_density: ArrayLike, diameter_edges: ArrayLike
) -> NDArray[np.float64]:
    """Divide integrated bin densities [m⁻³] by their log10 diameter widths.

    The shared 1-D edges are in metres. The last input axis enumerates bins.
    Returns dN/dlog10D [m⁻³ per decade] with the input density shape.
    """
    edges = _edges(diameter_edges)
    return _bin_values(bin_number_density, edges) / _log_width(
        edges[:-1], edges[1:]
    )


def get_bin_number_density(
    number_per_decade: ArrayLike, diameter_edges: ArrayLike
) -> NDArray[np.float64]:
    """Integrate dN/dlog10D [m⁻³ per decade] over each diameter interval.

    The shared 1-D edges are in metres. Returns number density per bin [m⁻³]
    with the bin axis last and all leading dimensions preserved.
    """
    edges = _edges(diameter_edges)
    return _bin_values(number_per_decade, edges) * _log_width(
        edges[:-1], edges[1:]
    )


def get_binned_number_spectrum(
    diameter: ArrayLike,
    number_density: ArrayLike,
    diameter_edges: ArrayLike,
    *,
    include_upper_edge: bool = True,
) -> NumberSpectrum:
    """Bin weighted point diameters on explicit diameter edges.

    Args:
        diameter: Particle diameters [m] with shape (..., slots). Zero diameter
            is allowed for a zero-weight slot. Occupied diameters are positive.
        number_density: Nonnegative weights [m⁻³] matching diameter.
        diameter_edges: Increasing positive bin boundaries [m], shape (bins+1,).
        include_upper_edge: Include the last edge in the last bin (default).
            Set False when a study counts that edge in an overflow channel.

    Returns:
        Integrated densities [m⁻³] with shape (..., bins) and densities below
        and above the covered range with shape (...,). Interior edges belong
        to the bin on their right.
    """
    edges = _edges(diameter_edges)
    sizes = _real(diameter)
    weights = _real(number_density)
    if sizes.ndim == 0 or sizes.shape != weights.shape:
        raise ValueError("Diameter and number density must share a slot axis")
    if (
        np.any(weights < 0)
        or np.any(sizes < 0)
        or np.any((sizes == 0) & (weights > 0))
    ):
        raise ValueError(
            "Require nonnegative weights and positive occupied diameters"
        )

    shape = sizes.shape[:-1]
    rows = int(np.prod(shape, dtype=np.intp))
    # Accumulate each bin independently so a large lower tail cannot erase
    # a small count in a subsequent bin by cumulative subtraction.
    totals = np.zeros((rows, edges.size + 1))
    for i, (d, w) in enumerate(
        zip(
            sizes.reshape(rows, sizes.shape[-1]),
            weights.reshape(rows, sizes.shape[-1]),
            strict=True,
        )
    ):
        indices = np.searchsorted(edges, d, side="right")
        if include_upper_edge:
            indices[d == edges[-1]] -= 1
        totals[i] = np.bincount(indices, weights=w, minlength=edges.size + 1)
    return NumberSpectrum(
        totals[:, 1:-1].reshape((*shape, edges.size - 1)),
        totals[:, 0].reshape(shape),
        totals[:, -1].reshape(shape),
    )


def get_particle_number_spectrum(
    particles: ParticleData, diameter_edges: ArrayLike
) -> NumberSpectrum:
    """Bin volume-equivalent diameters of discrete or resolved CPU particles.

    Supply diameter edges in metres. Returns one row of integrated number
    densities per box [m⁻³]. Resolved counts use the owning box's volume.
    Continuous radius PDFs require integration over the radius grid.
    """
    if particles.distribution_type not in ("discrete", "particle_resolved"):
        raise ValueError("Require tagged discrete or particle_resolved data")
    weights = particles.slot_concentration_density
    if particles.distribution_type == "particle_resolved":
        get_slot_diagnostics(particles)
    return get_binned_number_spectrum(
        2 * particles.radii, weights, diameter_edges
    )


def get_rebinned_number_spectrum(
    bin_number_density: ArrayLike,
    source_edges: ArrayLike,
    target_edges: ArrayLike,
) -> NumberSpectrum:
    """Rebin densities assuming constant dN/dlog10D within each source bin.

    Supply nonnegative densities [m⁻³] with the source bin axis last and two
    positive diameter edge vectors [m]. Returns integrated densities [m⁻³]
    on the target partition and partial or whole bins outside either end.
    """
    source = _edges(source_edges)
    target = _edges(target_edges)
    counts = _bin_values(bin_number_density, source)
    # Split only where either partition has a boundary. Every subinterval
    # belongs to exactly one source bin and one target bin (or tail).
    cuts = np.union1d(
        source, target[(target > source[0]) & (target < source[-1])]
    )
    source_bin = np.searchsorted(source, cuts[:-1], side="right") - 1
    destination = np.searchsorted(target, cuts[:-1], side="right")
    fraction = _log_width(cuts[:-1], cuts[1:]) / _log_width(
        source[source_bin], source[source_bin + 1]
    )
    out = np.zeros((*counts.shape[:-1], target.size + 1))
    for j, k, share in zip(source_bin, destination, fraction, strict=True):
        out[..., k] += counts[..., j] * share
    return NumberSpectrum(out[..., 1:-1], out[..., 0], out[..., -1])
