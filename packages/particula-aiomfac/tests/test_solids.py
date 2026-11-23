"""Mass, charge and chemical equilibrium with anhydrous minerals."""

# ruff: noqa: S101

import numpy as np
import pytest
from particula_aiomfac import (
    MixtureSpec,
    SolidSalt,
    solve_ionic_solid_liquid,
    solve_solid_liquid,
)
from particula_aiomfac.solids import solid_parameters

pytest.importorskip("particula")
from particula_aiomfac.adapter import AIOMFACActivity  # noqa: E402


def _provider(extra=None):
    recipe = {"water": {16: 1}, **(extra or {"NaCl": {202: 1, 242: 1}})}
    return AIOMFACActivity(MixtureSpec(recipe), dict(enumerate(recipe)))


def test_halite_saturation_precipitation_and_redissolution():
    """Recover saturation, then dissolve the entire crystal on dilution."""
    model = _provider()
    solids = {"NaCl": "halite"}
    saturated = solve_solid_liquid(model, [1.0, 1.0], 298.15, solids)
    assert saturated.solid_masses[0] > 0
    assert saturated.log_saturation[0] == pytest.approx(0, abs=1e-9)
    assert saturated.charge_error < 1e-12
    # Independent saturated solution RH from Greenspan (1977), Table 2:
    # 75.29 ± 0.12 % RH; model tolerance is distinct from measured uncertainty.
    assert saturated.liquids.alpha.neutral_activities[0] == pytest.approx(
        0.7529, abs=0.004
    )
    np.testing.assert_allclose(
        saturated.ion_masses.sum() + saturated.solid_masses.sum(), 1.0
    )
    diluted = solve_solid_liquid(model, [10.0, 1.0], 298.15, solids)
    assert diluted.log_saturation[0] < 0
    assert diluted.solid_masses[0] == 0
    assert diluted.ion_masses.sum() == pytest.approx(1.0)
    water = solve_solid_liquid(model, [1.0, 0.0], 298.15, solids)
    assert water.log_saturation[0] == -np.inf
    assert water.complementarity_error == 0


@pytest.mark.parametrize("feed", [1.0, 1e-24])
def test_dry_excess_net_charge_retained_as_named_ions(feed):
    """Unpaired charge survives even when the whole feed is a trace."""
    model = _provider()
    salts = {"NaCl": "halite"}
    chloride, sodium = 35.453e6 * feed, 22.9898e6 * feed
    result = solve_ionic_solid_liquid(
        model, {}, {202: 2 * feed, 242: feed}, 298.15, salts
    )
    assert result.solid_masses[0] == pytest.approx(
        chloride + sodium, rel=5e-5, abs=0
    )
    np.testing.assert_allclose(
        result.residual_ion_masses, [sodium, 0], rtol=5e-5, atol=0
    )
    assert result.liquids is None
    assert result.ion_masses.sum() == result.neutral_masses.sum() == 0
    assert np.isnan(result.log_saturation[0])


def test_wet_charge_temperature_and_mineral_formula():
    """Reject unsupported charge, reference temperature and stoichiometry."""
    model = _provider()
    with pytest.raises(ValueError, match="electroneutral"):
        solve_ionic_solid_liquid(
            model, {"water": 1}, {202: 2, 242: 1}, 298.15, {"NaCl": "halite"}
        )
    with pytest.raises(ValueError, match="temperature"):
        solve_solid_liquid(model, [1, 1], 273.15, {"NaCl": "halite"})
    with pytest.raises(ValueError, match="formula"):
        solid_parameters(model, {"NaCl": "sylvite"})
    user = SolidSalt(1.57, 297, 299)
    specs, nu = solid_parameters(model, {"NaCl": user})
    assert specs == (user,)
    np.testing.assert_array_equal(nu, [[1, 1]])


def test_common_ion_and_cross_precipitation_conserve_each_element():
    """Both initial salts can feed halite without an input NaCl lane."""
    model = _provider({"NaNO3": {202: 1, 245: 1}, "KCl": {203: 1, 242: 1}})
    salts = {"NaCl": "halite", "KCl": "sylvite"}
    amounts = np.array([1.0, 2.0, 1.5])
    result = solve_solid_liquid(model, amounts, 298.15, salts)
    assert (result.solid_masses > 0).all()
    assert result.complementarity_error < 1e-7
    assert result.charge_error < 1e-12
    ion_mw = np.array([22.98977, 39.0983, 35.453, 62.0049])
    salt_mw = np.array([ion_mw[0] + ion_mw[2], ion_mw[1] + ion_mw[2]])
    initial = np.array([2.0 / (ion_mw[0] + ion_mw[3]), 1.5 / salt_mw[1]])
    final_solid = result.solid_masses / salt_mw
    _, nu = solid_parameters(model, salts)
    np.testing.assert_allclose(
        result.ion_masses.sum(axis=0) / ion_mw + final_solid @ nu,
        initial[[0, 1, 1, 0]],
        rtol=5e-5,
    )
    np.testing.assert_allclose(
        result.solid_masses.sum()
        + result.ion_masses.sum()
        + result.neutral_masses.sum(),
        amounts.sum(),
        rtol=1e-12,
    )


def test_two_liquids_and_crystal_coexist_at_the_same_ion_activity_product():
    """A synthetic Ksp exercises simultaneous liquid and crystal coexistence."""
    model = _provider(
        {
            "2-propanol": {141: 2, 151: 1, 153: 1},
            "ammonium sulfate": {204: 2, 261: 1},
        }
    )
    masses = [0.3992, 0.39545, 0.20535]
    # Numerical fixture only. This is not a measured ammonium sulfate Ksp.
    result = solve_solid_liquid(
        model,
        masses,
        303.15,
        {"ammonium sulfate": SolidSalt(0, 303.15, 303.15)},
    )
    assert result.liquids.beta is not None
    assert result.solid_masses[0] > 0
    assert result.complementarity_error < 2e-7
    assert result.charge_error < 1e-12
    np.testing.assert_allclose(
        result.liquids.alpha.neutral_activities,
        result.liquids.beta.neutral_activities,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        result.neutral_masses.sum()
        + result.ion_masses.sum()
        + result.solid_masses.sum(),
        sum(masses),
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "name, formula, model_name",
    [
        ("NaHCO3", {202: 1, 250: 1}, "gibbs_consistent"),
        ("NH4HSO4", {204: 1, 248: 1}, "pinned"),
    ],
)
def test_reacted_species_restart_through_physical_and_independent_inputs(
    name, formula, model_name
):
    """Reconstruct the original solution from its actual reacted species."""
    from particula_aiomfac.solids import _ion_mw, conserved_species_matrix

    mixture = MixtureSpec({"water": {16: 1}, name: formula})
    provider = AIOMFACActivity(
        mixture, {0: "water", 1: name}, carbonate_model=model_name
    )
    p = provider.prepared
    init = solve_solid_liquid(provider, [10.0, 0.1], 298.15, {})
    neutrals = init.neutral_masses.sum(axis=0) * 1e-9 / p.neutral_mw
    ions = init.ion_masses.sum(axis=0) * 1e-9 / _ion_mw(p)
    args = (
        provider,
        dict(zip(p.neutrals, neutrals, strict=True)),
        dict(zip(p.ions, ions, strict=True)),
        298.15,
        {},
    )
    bulk = conserved_species_matrix(p) @ np.r_[neutrals, ions]
    for result in (
        solve_ionic_solid_liquid(*args),
        solve_ionic_solid_liquid(
            provider, {}, {}, 298.15, {}, conserved_moles=bulk
        ),
    ):
        np.testing.assert_allclose(
            result.neutral_masses, init.neutral_masses, rtol=1e-9
        )
        np.testing.assert_allclose(
            result.ion_masses, init.ion_masses, rtol=1e-9
        )


@pytest.mark.parametrize("execution", ["cpu", "cuda"])
def test_reactive_carbonate_crystal_from_bicarbonate(execution):
    """Hydration and proton exchange feed carbonate precipitation."""
    if (
        execution == "cuda"
        and not pytest.importorskip("warp").is_cuda_available()
    ):
        pytest.skip("requires NVIDIA CUDA")
    mixture = MixtureSpec(
        {
            "water": {16: 1},
            "CaCl2": {221: 1, 242: 2},
            "NaHCO3": {202: 1, 250: 1},
            "CaCO3": {221: 1, 262: 1},
        }
    )
    provider = AIOMFACActivity(
        mixture,
        dict(enumerate(mixture.components)),
        carbonate_model="gibbs_consistent",
    )
    mass = [10.0, 0.1, 0.12, 0.0]
    result = solve_solid_liquid(
        provider,
        mass,
        298.15,
        {"CaCO3": SolidSalt(-8.48)},
        phase_mode="single_liquid",
        execution=execution,
    )
    assert result.solid_masses[0] > 0.05
    assert abs(result.log_saturation[0]) < 1e-8
    assert result.charge_error < 1e-10
    final_mass = (
        result.solid_masses.sum()
        + result.neutral_masses.sum()
        + result.ion_masses.sum()
    )
    np.testing.assert_allclose(final_mass, sum(mass), rtol=1e-12)
