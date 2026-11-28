"""Internal conversions between raw storage and physical population measures."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from particula.particles.particle_data import ParticleData


def slot_density(particles: "ParticleData") -> NDArray[np.float64]:
    """Convert admitted extensive bins, slots or radius ordinates to density."""
    return particles.population / particles.volume[:, None]


def write_binned_density(
    particles: "ParticleData", density: NDArray[np.float64]
) -> None:
    """Commit admitted bin densities or PDF ordinates to existing storage."""
    if particles.distribution_type not in ("discrete", "continuous_pdf"):
        raise ValueError("writing bin densities requires a binned population")
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = density * particles.volume[:, None]
    if not np.all(np.isfinite(candidate)) or np.any(candidate < 0):
        raise ValueError("binned population must remain finite and nonnegative")
    particles.population[:] = candidate


def radius_node_widths(radius: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoidal integration widths (m) for an admitted 1-D radius grid."""
    intervals = np.diff(radius)
    widths = np.empty_like(radius)
    widths[0] = intervals[0] / 2
    widths[-1] = intervals[-1] / 2
    widths[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    return widths


def number_density_weights(particles: "ParticleData") -> NDArray[np.float64]:
    """Return detached (box, slot) contributions to number density in m⁻³."""
    weights = particles.slot_concentration_density
    if particles.radius_grid is not None:  # Only PDFs admit a radius grid.
        weights *= radius_node_widths(particles.radius_grid)
    return weights
