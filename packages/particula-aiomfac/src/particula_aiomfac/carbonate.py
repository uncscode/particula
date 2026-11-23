"""A molal, Gibbs-consistent extension of AIOMFAC for aqueous carbon dioxide."""

import numpy as np
from scipy.special import xlogy

from ._terms import electrolyte_terms, short_range


def solute_gibbs(model, neutral, ions):
    """Dimensionless extensive CO₂ contribution G/(RT), with amounts in mol.

    Solvent mass excludes CO₂. The molal standard state is 1 mol/kg.
    A zero solute inventory has zero contribution.
    """
    n = np.asarray(neutral)
    solvent = n.copy()
    solvent[..., model.co2] = 0
    kg = solvent @ model.neutral_mw
    co2 = n[..., model.co2]
    return xlogy(co2, co2 / kg) - co2 + 2 * co2 * (ions @ model.co2_lambda) / kg


def activity_terms(model, neutral, ions, temperature):
    """Return activity coefficients from analytical partial derivatives of G.

    Evaluate the solvent/ion AIOMFAC terms with CO₂ excluded. Neutral log
    coefficients are then expressed on the actual species mole-fraction
    basis, including the conversion to unit molality for aqueous CO₂.
    """
    n = neutral.shape[1]
    co2 = neutral[:, model.co2]
    solvent = neutral.copy()
    solvent[:, model.co2] = 0.0
    kg = solvent @ model.neutral_mw
    solvent_total = solvent.sum(axis=1)
    base_total = solvent_total + ions.sum(axis=1)
    actual_total = base_total + co2
    x = np.column_stack((solvent, ions)) / base_total[:, None]
    molal = ions / kg[:, None]
    sr = np.empty_like(x)
    for t in np.unique(temperature):
        rows = temperature == t
        sr[rows] = short_range(model, x[rows], t)
    mr, lr, strength, scale = electrolyte_terms(
        model, solvent / solvent_total[:, None], molal, temperature
    )
    concentration = co2 / kg
    salting = 2 * (molal @ model.co2_lambda)
    sr[:, :n] += np.log(actual_total / base_total)[:, None]
    mr[:, :n] -= (concentration * (1.0 + salting))[:, None] * model.neutral_mw
    mr[:, n:] += 2 * concentration[:, None] * model.co2_lambda
    sr[:, model.co2] = np.log(actual_total / kg)
    mr[:, model.co2] = salting
    lr[:, model.co2] = 0.0
    return sr, mr, lr, strength, scale, molal, kg
