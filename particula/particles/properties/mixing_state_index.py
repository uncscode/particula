"""Dry aerosol mixing-state index from particle-by-species masses."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import entr

from particula.util.validate_inputs import validate_inputs


@validate_inputs({"species_masses": "nonnegative"})
def get_mixing_state_index(
    species_masses: NDArray[np.float64],
) -> float:
    """Calculate the mass-weighted Riemer–West mixing-state index χ.

    For each particle, let H_i be the Shannon entropy of its species mass
    fractions, using natural logarithms and the convention 0 ln 0 = 0.
    H_alpha is the particle-mass-weighted mean of H_i. H_gamma is the
    entropy of bulk species mass fractions. The index is
    expm1(H_alpha) / expm1(H_gamma).

    Args:
        species_masses: Nonnegative dry species masses [kg], shape
            (particles, species). Rows represent individual particles.
            For representative particles, multiply each row by its
            represented count before calling. Zero-mass rows are ignored.

    Returns:
        Dimensionless χ in [0, 1]. Pure particles of different species
        give zero. Identical multispecies compositions give one. Returns
        NaN if fewer than two species carry mass, including an empty
        population, because the bulk diversity then gives no contrast.

    Examples:
        Two equal-mass particles with opposite 90/10 compositions:

        ```py
        import numpy as np
        import particula as par

        masses = 1e-18 * np.array([[9., 1.], [1., 9.]])
        chi = par.particles.get_mixing_state_index(masses)  # 0.38414549
        ```

    References:
        Riemer and West (2013), Quantifying aerosol mixing state with
        entropy and diversity measures. https://doi.org/10.5194/acp-13-11423-2013
    """
    masses = np.asarray(species_masses, dtype=np.float64)
    if masses.ndim != 2:
        raise ValueError("species_masses must be a particle-by-species matrix")
    scale = masses.max(initial=0.0)
    if scale == 0:
        return np.nan
    masses = masses / scale
    species_totals = masses.sum(axis=0)
    if np.count_nonzero(species_totals) < 2:
        return np.nan

    row_mass = masses.sum(axis=1)
    active = row_mass > 0
    particle_fractions = masses[active] / row_mass[active, None]
    particle_entropy = entr(particle_fractions).sum(axis=1)
    mean_entropy = np.average(particle_entropy, weights=row_mass[active])
    bulk_entropy = entr(species_totals / species_totals.sum()).sum()
    # Concavity bounds the ratio by one; sums can differ by roundoff.
    return float(min(1.0, np.expm1(mean_entropy) / np.expm1(bulk_entropy)))
