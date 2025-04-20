"""
Compute the aerosol mixing state index χ from an N×S matrix of per-particle
"""

from numpy.typing import NDArray
import numpy as np


def get_mixing_state_index(
    species_masses: NDArray[np.float64],
) -> float:
    """
    Compute the aerosol mixing state index (χ) from an N×S matrix of per-particle
    species masses. Here, N is the number of particles (rows) and S is the number
    of species (columns).

    --------
    Overview
    --------
    - Let Mₙₛ be the mass of species s in particle n.
    - The per-particle total mass Mₙ = Σₛ Mₙₛ (sum over species).
    - The per-particle mass fraction fₙₛ = Mₙₛ / Mₙ.
    - Per-particle diversity:

        Dₙ = exp( -Σₛ [ fₙₛ ln(fₙₛ) ] ).

    - Mass-weighted average per-particle diversity (D̄ᵅ):

        D̄ᵅ = ( Σₙ [ Mₙ Dₙ ] ) / ( Σₙ Mₙ ).

    - Bulk (overall) diversity (Dᵞ):
      1) First, compute total species masses: Mₛ = Σₙ Mₙₛ (sum over particles).
      2) Let Fₛ = Mₛ / (Σₛ Mₛ) be the bulk mass fraction of species s.
      3) Then,

         Dᵞ = exp( -Σₛ [ Fₛ ln(Fₛ) ] ).

    - Mixing State Index (χ):

        χ = (D̄ᵅ - 1) / (Dᵞ - 1).

    -----------
    Parameters
    -----------
    M : ndarray, shape (N, S)
        A 2D NumPy array where each row corresponds to a particle and each column
        to a species. Entries are the mass of that species in that particle.

    -------
    Returns
    -------
    Xi : float
        The mixing state index, χ (0 ≤ χ ≤ 1). χ = 0 represents a fully external
        mixture (each particle is composed of one species). χ = 1 represents a
        fully internal mixture (all particles have the same composition as the bulk).

    ----------
    References
    ----------
    1) Riemer, N., West, M., Zaveri, R. A., & Barnard, J. C. (2009).
       "Simulating the evolution of soot mixing state with a particle-resolved
       aerosol model." Journal of Geophysical Research: Atmospheres, 114(D9).
    2) Riemer, N., Ault, A. P., West, M., Craig, R. L., & Curtis, J. H. (2019).
       "Aerosol Mixing State: Measurements, Modeling, and Impacts."
       Reviews of Geophysics, 57(2), 187–249.
    """
    # Small number to avoid log(0)
    small = 1e-30

    # 0. Remove particles with no mass (all species are zero)
    M = np.array(M)  # Ensure M is a NumPy array
    M = M[M.sum(axis=1) > 0]  # shape (N, S)
    if M.size == 0:
        # If no particles remain, return NaN
        return np.nan

    # 1. Total mass per particle (M_n)
    M_n = M.sum(axis=1)  # shape (N,)

    # 2. Per-particle mass fractions (f_{n,s})
    f = M / (M_n[:, None] + small)  # shape (N, S)

    # 3. Per-particle diversity: D_n = exp( -sum(f log f) )
    D_n = np.exp(-(f * np.log(f + small)).sum(axis=1))  # shape (N,)

    # 4. Mass-weighted average per-particle diversity (D_alpha)
    M_tot = M_n.sum()
    if M_tot < small:
        # If total mass is extremely small, mixing state is undefined
        return np.nan

    D_alpha = np.sum(M_n * D_n) / M_tot

    # 5. Bulk diversity (D_gamma)
    M_s = M.sum(axis=0)  # total mass of each species, shape (S,)
    F_s = M_s / (M_tot + small)
    D_gamma = np.exp(-(F_s * np.log(F_s + small)).sum())

    # 6. Mixing State Index (Xi)
    Xi = (D_alpha - 1.0) / (D_gamma - 1.0)

    return Xi
