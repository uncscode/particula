"""Check ionic transfers and conserved inventories in two liquids."""

import numpy as np
import pytest
from particula_aiomfac import AIOMFACModel, MixtureSpec


def system(salts):
    """Prepare a water–alcohol mixture containing the analytical salts."""
    groups = {"water": {16: 1}, "alcohol": {141: 2, 151: 1, 153: 1}}
    return AIOMFACModel.prepare(MixtureSpec({**groups, **salts}))


def phase_ions(result, mass):
    """Express true equilibrium species amounts in moles per original feed."""
    return (
        result.species_moles_per_kg_input[len(result.neutral_ids) :]
        * mass.sum()
    )


def test_reciprocal_salt_masses_do_not_constrain_ionic_ratios():
    model = system({"NaCl": {202: 1, 242: 1}, "K2SO4": {203: 2, 261: 1}})
    feed = np.array([0.4, 0.4, 0.1, 0.1])
    split = model.phase_equilibrium(feed, 303.15, 0)
    alpha, beta = split.alpha_masses, split.beta_masses
    np.testing.assert_allclose(alpha + beta, feed, rtol=1e-13)
    inventory = model.evaluate(feed, 303.15)
    ia = phase_ions(split.alpha, alpha)
    ib = phase_ions(split.beta, beta)
    np.testing.assert_allclose(ia + ib, phase_ions(inventory, feed), rtol=1e-10)
    for ions in (ia, ib):
        np.testing.assert_allclose(ions @ model.charge, 0.0, atol=1e-12)
    # NaCl contributes equimolar Na and Cl to the feed. The organic-rich
    # liquid takes up more chloride than sodium and remains electroneutral.
    assert ib[model.ions.index(242)] > 1.5 * ib[model.ions.index(202)]
    for field in ("neutral_activities", "ion_activity_products"):
        np.testing.assert_allclose(
            getattr(split.alpha, field), getattr(split.beta, field), rtol=1e-7
        )
    assert split.minimum_tpd >= -1e-7


def test_identical_ions_from_different_precursor_salts():
    a = system({"NaCl": {202: 1, 242: 1}, "K2SO4": {203: 2, 261: 1}})
    b = system({"KCl": {203: 1, 242: 1}, "Na2SO4": {202: 2, 261: 1}})
    phases = []
    for model in (a, b):
        mass = np.r_[0.4, 0.4, model.salt_mw * [2.0, 1.0]]
        phases.append(model.phase_equilibrium(mass, 303.15, 0))
    for side in ("alpha", "beta"):
        chemistry = [getattr(p, side) for p in phases]
        weights = [getattr(p, side + "_masses").sum() for p in phases]
        assert weights[0] == pytest.approx(weights[1], rel=1e-7)
        np.testing.assert_allclose(
            chemistry[0].species_moles_per_kg_input,
            chemistry[1].species_moles_per_kg_input,
            rtol=1e-7,
        )


def test_bisulfate_speciates_in_each_separated_liquid():
    model = system({"AHS": {204: 1, 248: 1}, "AS": {204: 2, 261: 1}})
    mass = np.array([0.4, 0.4, 0.02, 0.18])
    split = model.phase_equilibrium(mass, 303.15, 0)
    assert split.beta is not None
    totals = np.zeros(3)
    for phase, weight in (
        (split.alpha, split.alpha_masses),
        (split.beta, split.beta_masses),
    ):
        assert phase.solver_status == 1
        assert np.max(phase.diagnostics["equilibrium_residuals"]) < 1e-8
        ammonium, proton, acid, sulfate = phase_ions(phase, weight)
        totals += [ammonium, proton + acid, acid + sulfate]
    nh4hso4, ammonium_sulfate = mass[2:] / model.salt_mw[:2]
    np.testing.assert_allclose(
        totals,
        [nh4hso4 + 2 * ammonium_sulfate, nh4hso4, nh4hso4 + ammonium_sulfate],
        rtol=1e-9,
    )
