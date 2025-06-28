"""Compute the aerosol mixing state index χ from an NxS matrix of per-particle
species masses.
"""

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_exp, get_safe_log10
from particula.util.validate_inputs import validate_inputs


@validate_inputs({"species_masses": "nonnegative"})
def get_mixing_state_index(
    species_masses: NDArray[np.float64],
) -> float:
    """Calculate the aerosol mixing-state index (χ).

    The index quantifies how internally or externally mixed an aerosol
    population is. Fully internally mixed aerosols
    have χ = 1, while fully externally mixed aerosols have χ = 0. The mixing
    state index is a measure of the heterogeneity of the aerosol population,
    and is defined as the ratio of the mass-weighted mean diversity of the
    aerosol population to the bulk diversity of the aerosol population.
    It is defined as:

    - χ = (D̄ᵅ - 1) / (Dᵞ - 1)
        - D̄ᵅ = Σₙ (Mₙ · Dₙ) / Σₙ Mₙ
        - Dᵞ  = exp(−Σₛ Fₛ log Fₛ)
        - Dₙ  = exp(−Σₛ fₙₛ log fₙₛ)
        - fₙₛ = Mₙₛ / Mₙ
        - Fₛ  = Mₛ  / Σₛ Mₛ
        - D̄ᵅ is the mass-weighted mean diversity of the aerosol population
        - Dᵞ is the bulk diversity of the aerosol population
        - Dₙ is the diversity of particle n
        - fₙₛ is the mass fraction of species s in particle n
        - Fₛ is the mass fraction of species s in the aerosol population
        - Mₙₛ is mass of species s in particle n
        - Mₙ is total mass of particle n (Σₛ Mₙₛ)
        - Mₛ is total mass of species s (Σₙ Mₙₛ)

    Arguments:
        - species_masses : Per-particle species masses (NxS matrix).
          [kg] where N is the number of particles and S the number of species.

    Returns:
        - mixing_state_index : Mixing-state index χ (float, 0 ≤ χ ≤ 1).
          Returns NaN when the aerosol has no mass.

    Examples:
        ``` py title="Mixing State Index Calculation"
        import numpy as np
        import particula as par

        # two particles, two species
        masses = np.array([[1.0e-15, 0.0],
                           [5.0e-16, 5.0e-16]])
        chi = par.particles.get_mixing_state_index(masses)
        print(chi)  # 0.5
        ```

    References:
    - Riemer, N., West, M., Zaveri, R. A., & Easter, R. C. (2009).
      Simulating the evolution of soot mixing state with a particle-resolved
      aerosol model. Journal of Geophysical Research Atmospheres, 114(9).
      [DOI](https://doi.org/10.1029/2008JD011073)
    - Riemer, N., Ault, A. P., West, M., Craig, R. L., & Curtis, J. H.
      (2019). Aerosol Mixing State: Measurements, Modeling, and Impacts.
      Reviews of Geophysics, 57(2), 187–249.
      [DOI](https://doi.org/10.1029/2018RG000615)
    """
    species_masses_array = np.asarray(species_masses, dtype=float)

    # only keep particles with non‑zero total mass
    species_masses_array = species_masses_array[
        species_masses_array.sum(axis=1) > 0
    ]
    if species_masses_array.size == 0:
        return np.nan

    # total mass of each particle
    mass_per_particle = species_masses_array.sum(axis=1)

    # per‑particle mass fractions
    mass_fraction = species_masses_array / (mass_per_particle[:, None])

    # per‑particle diversity
    per_particle_diversity = get_safe_exp(
        -(mass_fraction * get_safe_log10(mass_fraction)).sum(axis=1)
    )

    # total aerosol mass
    total_mass = mass_per_particle.sum()
    if total_mass <= 0:
        return np.nan

    # mass‑weighted mean diversity (D̄ᵅ)
    mass_weighted_diversity = (
        np.sum(mass_per_particle * per_particle_diversity) / total_mass
    )

    # bulk diversity (Dᵞ)
    total_species_mass = species_masses_array.sum(axis=0)
    bulk_mass_fraction = total_species_mass / (total_mass)
    bulk_diversity = get_safe_exp(
        -(bulk_mass_fraction * get_safe_log10(bulk_mass_fraction)).sum()
    )

    return (mass_weighted_diversity - 1.0) / (bulk_diversity - 1.0)
