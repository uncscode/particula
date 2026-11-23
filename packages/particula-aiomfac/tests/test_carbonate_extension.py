"""Free-energy gradients, closed reaction budgets and carbonate phase stability."""

import unittest

import numpy as np
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula_aiomfac._equilibrium import log_constants
from particula_aiomfac.phases import component_molar_masses

WATER = {16: 1}
ALCOHOL = {141: 2, 151: 1, 153: 1}
ACID_SALT = {204: 1, 248: 1}
BICARBONATE = {203: 1, 250: 1}
CO2 = {173: 1}


def _model(recipe):
    return AIOMFACModel.prepare(
        MixtureSpec(recipe), carbonate_model="gibbs_consistent"
    )


def _standard_potentials(model):
    # Zero references for H+, H₂O, sulfate, carbonate and inert species.
    # Products minus reactants for each standard reaction equals −ln(K).
    k1, k2, kw, ks = log_constants(298.15)
    neutral = np.zeros(len(model.neutrals))
    neutral[model.co2] = k1 + k2
    ions = np.array(
        [{248: ks, 250: k2, 247: -kw}.get(j, 0) for j in model.ions]
    )
    return neutral, ions


def _gibbs_and_grad(model, amounts):
    mw = component_molar_masses(model)
    r = model.evaluate(amounts * mw, 298.15)
    neutral_ref, ion_ref = _standard_potentials(model)
    neutral = np.log(r.neutral_activities) + neutral_ref
    ions = np.log(r.ion_molalities) + r.ion_log_coefficients + ion_ref
    # Euler sum over *equilibrated species*. Derivatives act on the supplied
    # analytical amounts; each perturbation repeats the reaction solve.
    g = np.r_[neutral, ions] @ r.species_moles_per_kg_input * (amounts @ mw)
    mu = np.zeros(model.ncomponents + model.virtual_count)
    mu[model.neutral_indices] = neutral
    mu[model.salt_indices] = model.stoich @ ions
    return g, mu[: model.ncomponents]


class TestCarbonateGibbs(unittest.TestCase):
    """Independent equilibrium, integrability and phase-split oracles."""

    def test_minimized_energy_gradient_and_maxwell_symmetry(self):
        model = _model(
            {
                "water": WATER,
                "IPA": ALCOHOL,
                "NH4HSO4": ACID_SALT,
                "KHCO3": BICARBONATE,
                "CO2": CO2,
            }
        )
        amounts = np.array([50.0, 1.0, 0.1, 0.15, 0.01])
        gibbs, grad = _gibbs_and_grad(model, amounts)
        self.assertAlmostEqual(gibbs, amounts @ grad, places=10)
        derivative = np.empty_like(amounts)
        jac = np.empty((len(amounts), len(amounts)))
        for j, dx in enumerate(1e-5 * amounts):
            shift = np.zeros_like(amounts)
            shift[j] = dx
            gp, mup = _gibbs_and_grad(model, amounts + shift)
            gm, mum = _gibbs_and_grad(model, amounts - shift)
            derivative[j] = (gp - gm) / (2 * dx)
            jac[:, j] = (mup - mum) / (2 * dx)
        np.testing.assert_allclose(derivative, grad, rtol=0, atol=2e-7)
        np.testing.assert_allclose(jac, jac.T, rtol=0, atol=2e-7)

    def test_co2_dilution_water_autoprotolysis_and_mass_action(self):
        model = _model({"water": WATER, "CO2": CO2})
        mw = component_molar_masses(model)
        amounts = np.array([[1 / mw[0], d] for d in [0, 1e-7, 1e-5, 0.0344]])
        results = model.evaluate(amounts * mw, 298.15, diagnostics=True)
        ions = results.species_moles_per_kg_input[:, 2:]
        n = results.species_moles_per_kg_input[:, :2]
        scale = amounts @ mw
        c = (
            n[:, 1]
            + ions[:, model.ions.index(250)]
            + ions[:, model.ions.index(262)]
        )
        np.testing.assert_allclose(c * scale, amounts[:, 1], rtol=1e-10)
        np.testing.assert_allclose(ions @ model.charge, 0, atol=1e-13)
        # Both carbonic-acid dissociations and water ionization from outputs.
        a = np.exp(results.ion_log_coefficients) * results.ion_molalities
        hydrogen, hydroxide, bicarb, carbonate = a.T
        aw, ac = results.neutral_activities.T
        k1, k2, kw, _ = log_constants(298.15)
        np.testing.assert_allclose(
            np.log(hydrogen * hydroxide / aw), kw, atol=1e-8
        )
        np.testing.assert_allclose(
            np.log((hydrogen * bicarb)[1:] / (ac * aw)[1:]), k1, atol=1e-8
        )
        np.testing.assert_allclose(
            np.log((hydrogen * carbonate)[1:] / bicarb[1:]), k2, atol=1e-8
        )
        self.assertEqual(results.model_identity[-1], "gibbs_consistent")
        pure_co2_molality = n[:, 1] / (n[:, 0] * mw[0])
        np.testing.assert_allclose(ac[1:], pure_co2_molality[1:], rtol=3e-4)
        # In this isolated water/CO₂ system total H and O are independent
        # of the reaction extents and the reported speciation.
        htot = 2 * n[:, 0] + ions[:, 0] + ions[:, 1] + ions[:, 2]
        otot = (
            n[:, 0] + 2 * n[:, 1] + ions[:, 1] + 3 * (ions[:, 2] + ions[:, 3])
        )
        np.testing.assert_allclose(htot * scale, 2 * amounts[:, 0], rtol=1e-12)
        np.testing.assert_allclose(
            otot * scale, amounts[:, 0] + 2 * amounts[:, 1], rtol=1e-12
        )

    def test_reactive_carbonate_sulfate_liquid_split(self):
        model = _model(
            {
                "water": WATER,
                "IPA": ALCOHOL,
                "NH4HSO4": ACID_SALT,
                "KHCO3": BICARBONATE,
            }
        )
        amounts = np.array([0.4, 0.4, 0.18, 0.02])
        feed = model.evaluate(amounts, 298.15)
        result = model.phase_equilibrium(amounts, 298.15, 0)
        self.assertIsNotNone(result.beta)
        self.assertGreater(result.minimum_tpd, -1e-7)
        self.assertLess(result.chemical_potential_error, 1e-7)
        np.testing.assert_allclose(
            result.alpha.neutral_activities,
            result.beta.neutral_activities,
            rtol=2e-7,
        )

        # Actual inventories include all reaction products in both liquids.
        def conserved(phase):
            n = phase.species_moles_per_kg_input[: len(model.neutrals)]
            ions = phase.species_moles_per_kg_input[len(model.neutrals) :]
            i = dict(zip(phase.ion_ids, ions, strict=True))
            self.assertLess(
                abs(ions @ model.charge) / (ions @ abs(model.charge)), 1e-10
            )
            return np.array(
                [
                    n[1],
                    i[203],
                    i[204],
                    i[248] + i[261],
                    n[model.co2] + i[250] + i[262],
                    n[model.water]
                    + 2 * n[model.co2]
                    + 3 * (i[250] + i[262])
                    + i[247],
                    2 * n[model.water] + i[205] + i[248] + i[250] + i[247],
                ]
            )

        np.testing.assert_allclose(
            conserved(result.alpha) * result.alpha_masses.sum()
            + conserved(result.beta) * result.beta_masses.sum(),
            conserved(feed),
            rtol=2e-10,
        )

    def test_alkaline_and_molecular_carbon_sources(self):
        for recipe, mass in [
            (
                {"water": WATER, "IPA": ALCOHOL, "K2CO3": {203: 2, 262: 1}},
                [0.4, 0.4, 0.2],
            ),
            (
                {
                    "water": WATER,
                    "IPA": ALCOHOL,
                    "CO2": CO2,
                    "NH4HSO4": ACID_SALT,
                    "NaCl": {202: 1, 242: 1},
                },
                [0.4, 0.4, 0.002, 0.188, 0.01],
            ),
        ]:
            with self.subTest(recipe=recipe):
                model = _model(recipe)
                result = model.phase_equilibrium(mass, 298.15, 0)
                self.assertIsNotNone(result.beta)
                self.assertLess(result.chemical_potential_error, 1e-7)
                self.assertGreater(result.minimum_tpd, -1e-7)
                np.testing.assert_allclose(
                    result.alpha_masses + result.beta_masses, mass
                )
                np.testing.assert_allclose(
                    result.alpha.neutral_activities,
                    result.beta.neutral_activities,
                    rtol=1e-7,
                )
