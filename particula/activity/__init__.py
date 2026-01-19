"""Particula activity sub-package.

Provides Binary Activity Thermodynamics (BAT) utilities for organic–water
systems following Gorkowski, Preston, and Zuend (2019).

References:
    Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    Relative-humidity-dependent organic aerosol thermodynamics via an
    efficient reduced-complexity model. Atmospheric Chemistry and Physics.
    https://doi.org/10.5194/acp-19-13383-2019
"""

from particula.activity.activity_coefficients import bat_activity_coefficients
from particula.activity.bat_blending import bat_blending_weights
from particula.activity.bat_coefficients import (
    FitValues,
    G19_FIT_HIGH,
    G19_FIT_LOW,
    G19_FIT_MID,
    coefficients_c,
)
from particula.activity.convert_functional_group import convert_to_oh_equivalent
from particula.activity.gibbs import gibbs_free_energy
from particula.activity.gibbs_mixing import gibbs_mix_weight, gibbs_of_mixing
from particula.activity.phase_separation import (
    MIN_SPREAD_IN_AW,
    Q_ALPHA_AT_1PHASE_AW,
    find_phase_sep_index,
    find_phase_separation,
    organic_water_single_phase,
    q_alpha,
)
from particula.activity.ratio import from_molar_mass_ratio, to_molar_mass_ratio

__all__ = [
    # Core activity calculation
    "bat_activity_coefficients",
    # Gibbs mixing
    "gibbs_mix_weight",
    "gibbs_of_mixing",
    "gibbs_free_energy",
    # BAT model components
    "bat_blending_weights",
    "coefficients_c",
    "FitValues",
    "G19_FIT_LOW",
    "G19_FIT_MID",
    "G19_FIT_HIGH",
    # Phase separation
    "find_phase_separation",
    "find_phase_sep_index",
    "organic_water_single_phase",
    "q_alpha",
    "MIN_SPREAD_IN_AW",
    "Q_ALPHA_AT_1PHASE_AW",
    # Utilities
    "convert_to_oh_equivalent",
    "to_molar_mass_ratio",
    "from_molar_mass_ratio",
]
