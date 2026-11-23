"""Verify exclusive reservoirs, individual ion budgets and native publication."""

from copy import deepcopy
from dataclasses import replace
import os

import numpy as np
import pytest
from scipy.constants import Avogadro, R
from particula_aiomfac import (
    GasReaction,
    IonSurface,
    MixtureSpec,
    SpeciesInventory,
    SpeciesLayout,
)
from particula_aiomfac.reservoirs import HCL_298

pytest.importorskip("particula")
from particula.equilibria import Equilibria, EquilibriaFactory
from particula.particles import ActivityFactory


def provider():
    """Bind acidic sodium chloride and its references at 298.15 K."""
    groups = {
        "water": {16: 1},
        "NaCl": {202: 1, 242: 1},
        "HCl": {205: 1, 242: 1},
    }
    return ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(groups),
            "material_mapping": dict(enumerate(groups)),
            "phase_mode": "stable",
            "solids": {"NaCl": "halite"},
        },
    )


def test_salt_changes_mass_owner_and_dry_residue_redissolves():
    """Salt label, dissolved ions and dry residuals never duplicate mass."""
    model = provider()
    assert deepcopy(model).prepared_identity() == model.prepared_identity()
    solve = (
        EquilibriaFactory()
        .get_strategy("liquid_vapor", {"activity_model": model})
        .solve
    )
    initial = SpeciesInventory(
        solids={"NaCl": 1.0}, liquid_neutrals={"water": 1000.0}
    )
    wet = solve(species_inventory=initial, temperature=298.15).inventory
    assert wet.solids["NaCl"] == 0
    np.testing.assert_allclose(
        [wet.liquid_ions[202], wet.liquid_ions[242]],
        [0.3933747412, 0.6066252588],
        rtol=1e-8,
    )
    dry = solve(
        species_inventory=replace(wet, liquid_neutrals={}), temperature=298.15
    ).inventory
    assert dry.solids["NaCl"] == pytest.approx(1.0)
    assert sum(dry.liquid_ions.values()) == 0.0
    charged = solve(
        species_inventory=replace(dry, residual_ions={202: 0.1}),
        temperature=298.15,
    )
    assert charged.inventory.residual_ions[202] == pytest.approx(0.1)
    back = solve(
        species_inventory=replace(dry, liquid_neutrals={"water": 1000.0}),
        temperature=298.15,
    )
    assert back.inventory.liquid_ions == pytest.approx(wet.liquid_ions)
    assert back.mass_balance_error < 1e-12


def test_neutral_acid_gas_uses_two_aqueous_activities():
    """Compare modeled HCl pressure to the published activity-product law."""
    inv = SpeciesInventory(
        liquid_neutrals={"water": 1000.0},
        solids={"NaCl": 50.0},
        gases={"HCl": 1.0},
    )
    out = provider().equilibrate_species(
        species_inventory=inv,
        temperature=298.15,
        gas_reactions={"HCl": HCL_298},
    )
    state = out.condensed.liquids.alpha
    product = state.ion_activity_products[
        state.electrolyte_ids.index((205, 242))
    ]
    pressure = (
        out.inventory.gases["HCl"] * 1e-9 / (0.001008 + 0.035453) * R * 298.15
    )
    assert pressure == pytest.approx(101325 * product / 1.971e6, rel=2e-4)
    assert sum(out.inventory.liquid_ions.values()) + out.inventory.gases[
        "HCl"
    ] == pytest.approx(51.0)
    assert out.charge_balance_mol_m3 == pytest.approx(0.0, abs=1e-15)


def charged_options():
    """Use explicit test parameters to put measurable amounts in each reservoir."""
    return dict(
        gas_reactions={"Na+": GasReaction({202: 1}, np.log(1e10), 298.15, 1)},
        surface=IonSurface(2e-7, 1.0, {242: -2.0}),
    )


@pytest.mark.parametrize(
    "execution",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.getenv("AIOMFAC_TEST_CUDA") != "1",
                reason="requires real CUDA",
            ),
        ),
    ],
)
def test_individual_ion_transfer_keeps_charge_and_each_element(execution):
    """A sodium ion can enter the gas while chloride stays on the particle."""
    model = provider()
    initial = SpeciesInventory(
        liquid_neutrals={"water": 1000.0}, solids={"NaCl": 50.0}
    )
    reference = model.equilibrate_species(
        species_inventory=initial, temperature=298.15
    ).inventory
    out = model.equilibrate_species(
        species_inventory=initial,
        temperature=298.15,
        execution=execution,
        **charged_options(),
    )
    final = out.inventory
    assert final.gases["Na+"] > 0.1
    assert final.surface_ions[242] > 0.0
    assert final.liquid_ions[202] + final.gases["Na+"] == pytest.approx(
        reference.liquid_ions[202], rel=1e-8
    )
    assert final.liquid_ions[242] + final.surface_ions[242] == pytest.approx(
        reference.liquid_ions[242], rel=1e-8
    )
    assert final.total_mass == pytest.approx(initial.total_mass, rel=1e-10)
    assert out.surface_potential_v < 0
    again = model.equilibrate_species(
        species_inventory=final,
        temperature=298.15,
        execution=execution,
        **charged_options(),
    )
    assert again.inventory.gases == pytest.approx(final.gases, rel=1e-8)


def test_native_runnable_transfers_actual_lanes_and_rolls_back_on_capacity():
    """Write solid, ion, and gas mass while keeping integer particle charges."""
    from particula import Aerosol
    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    count = 1e14
    particles = ParticleData(
        masses=np.array([[[1000.0, 0, 0, 50.0, 0], [0.0, 0, 0, 0, 0]]])
        * 1e-9
        / count,
        concentration=np.array([[count, 0.0]]),
        charge=np.zeros((1, 2)),
        density=np.full(5, 1000.0),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    aerosol = Aerosol(
        particles=particles,
        gas=GasData(
            ["Na+"],
            np.array([0.02299]),
            np.zeros((1, 1)),
            np.ones(1, dtype=bool),
        ),
        environment=EnvironmentData([298.15], [101325.0], [[0.0]]),
    )
    strategy = EquilibriaFactory().get_strategy(
        "liquid_vapor", {"activity_model": provider()}
    )
    layout = SpeciesLayout(
        {"water": 0},
        {202: 1, 242: 2},
        {"NaCl": 3},
        surface_ions={242: 4},
        slots=(0,),
    )
    original = particles.masses.copy()
    with pytest.raises(ValueError, match="second slot"):
        Equilibria(
            strategy, species_layout=layout, species_options=charged_options()
        ).execute(aerosol, 1.0)
    np.testing.assert_array_equal(particles.masses, original)
    run = Equilibria(
        strategy,
        species_layout=replace(layout, slots=(0, 1)),
        species_options=charged_options(),
    )
    run.execute(aerosol, 1.0)
    assert particles.concentration.sum() == pytest.approx(count)
    np.testing.assert_array_equal(particles.charge, np.round(particles.charge))
    material = particles.concentration[0] @ particles.masses[0]
    assert material[3] == 0.0
    assert np.all(material[[1, 2, 4]] > 0.0)
    assert 1e9 * (
        material.sum() + aerosol.gas.concentration.sum()
    ) == pytest.approx(1050.0)
    net_charge = particles.concentration[0] @ particles.charge[0] / Avogadro
    assert net_charge == pytest.approx(
        -aerosol.gas.concentration.sum() / 0.02299, rel=1e-4
    )
    run.execute(aerosol, 1.0)
    np.testing.assert_allclose(
        particles.concentration[0] @ particles.masses[0], material, rtol=1e-8
    )


@pytest.mark.parametrize("inventory_water, dry", [(5.0, True), (9.0, False)])
def test_closed_water_gas_over_halite(inventory_water, dry):
    """Start with dry salt and water vapor below and above deliquescence."""
    model = provider()
    molar_mass = model.prepared.neutral_mw[
        model.prepared.neutrals.index("water")
    ]
    saturation = 10.0
    k = 101325 * molar_mass * 1e9 / (R * 298.15 * saturation)
    options = {
        "temperature": 298.15,
        "gas_reactions": {
            "water vapor": GasReaction({"water": 1}, np.log(k), 298.15)
        },
    }
    out = model.equilibrate_species(
        species_inventory=SpeciesInventory(
            solids={"NaCl": 1.0}, gases={"water vapor": inventory_water}
        ),
        **options,
    )
    assert (out.condensed.liquids is None) == dry
    assert out.inventory.total_mass == pytest.approx(1 + inventory_water)
    assert out.inventory.liquid_neutrals["water"] + out.inventory.gases[
        "water vapor"
    ] == pytest.approx(inventory_water)
    repeat = model.equilibrate_species(
        species_inventory=out.inventory, **options
    )
    assert repeat.inventory.solids == pytest.approx(out.inventory.solids)
    assert repeat.inventory.gases == pytest.approx(out.inventory.gases)


def test_bicarbonate_supplies_co2_gas_and_restarts_from_actual_species():
    """Carbon can leave bicarbonate even when no molecular CO2 was supplied."""
    groups = {"water": {16: 1}, "NaHCO3": {202: 1, 250: 1}}
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(groups),
            "material_mapping": dict(enumerate(groups)),
            "carbonate_model": "gibbs_consistent",
            "phase_mode": "stable",
        },
    )
    model = activity.prepared
    co2 = model.neutrals[model.co2]
    sodium = 2.3
    # kg/mol from the pinned ion parameter table, scaled to µg/m³.
    from particula_aiomfac.reservoirs import _ion_mw

    ion_masses = _ion_mw([202, 250])
    feed = SpeciesInventory(
        liquid_neutrals={"water": 1000.0},
        liquid_ions={202: sodium, 250: sodium * ion_masses[1] / ion_masses[0]},
    )
    options = {
        "temperature": 298.15,
        "gas_reactions": {
            "CO2 gas": GasReaction({co2: 1}, np.log(0.034), 298.15)
        },
    }
    out = activity.equilibrate_species(species_inventory=feed, **options)
    assert out.inventory.gases["CO2 gas"] > 0
    assert out.inventory.total_mass == pytest.approx(feed.total_mass, rel=1e-8)
    assert out.inventory.liquid_ions[202] == pytest.approx(sodium)
    ag = (
        out.inventory.gases["CO2 gas"]
        * 1e-9
        / model.neutral_mw[model.co2]
        * R
        * 298.15
        / 101325
    )
    assert out.condensed.liquids.alpha.neutral_activities[
        model.co2
    ] == pytest.approx(0.034 * ag, rel=1e-8)
    repeat = activity.equilibrate_species(
        species_inventory=out.inventory, **options
    )
    assert repeat.inventory.gases == pytest.approx(
        out.inventory.gases, rel=1e-8
    )
    assert repeat.inventory.liquid_neutrals == pytest.approx(
        out.inventory.liquid_neutrals, rel=1e-8
    )


def test_dry_halite_seed_uses_deliquescence_activity():
    """The exactly dry seed approaches the same water activity as wet salt."""
    strategy = provider()
    activities = strategy.activities_at_temperature(
        np.array([[0.0, 1.0, 0.0], [0.01, 1.0, 0.0]]), 298.15, ["water"]
    )
    np.testing.assert_allclose(activities[:, 0], 0.755334857, rtol=1e-8)


def test_charged_gas_from_reacting_brine():
    """Conserve Na, carbon and charge across bicarbonate surface adsorption."""
    from particula_aiomfac.reservoirs import _ion_mw

    groups = {"water": {16: 1}, "NaHCO3": {202: 1, 250: 1}, "CO2": {173: 1}}
    model = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
            carbonate_model="gibbs_consistent",
        ),
    )
    masses = _ion_mw([202, 250]) * 1000
    initial = SpeciesInventory(
        liquid_neutrals={"water": 1000.0},
        liquid_ions=dict(zip([202, 250], masses, strict=True)),
    )
    out = model.equilibrate_species(
        species_inventory=initial,
        temperature=298.15,
        phase_mode="single_liquid",
        gas_reactions=charged_options()["gas_reactions"],
        surface=IonSurface(2e-7, 1.0, {250: -2.0}),
    )
    assert out.inventory.gases["Na+"] > 0
    assert out.inventory.surface_ions[250] > 0
    assert out.mass_balance_error < 1e-8
    assert abs(out.charge_balance_mol_m3) < 1e-14
    activities = model.activities_at_temperature(
        [1000, masses.sum(), 0.1], 298.15, ["CO2"]
    )
    pressure = model.partial_pressures_at_temperature(
        [1000, masses.sum(), 0.1], 298.15, ["CO2"], [101325 / 0.034]
    )
    np.testing.assert_allclose(pressure, activities * 101325 / 0.034)
    assert deepcopy(model).prepared_identity() == model.prepared_identity()
    with pytest.raises(ValueError, match="direct"):
        ActivityFactory().get_strategy(
            "aiomfac",
            dict(
                mixture=MixtureSpec(groups),
                material_mapping=dict(enumerate(groups)),
                carbonate_model="gibbs_consistent",
                evaluation="hybrid",
                table="unused",
            ),
        )


@pytest.mark.parametrize(
    "initial_water, expected_gas", [(3.0, 3.0), (15.0, 10.0)]
)
def test_physical_pure_water_vapor_limit(initial_water, expected_gas):
    model = provider()
    mw = model.prepared.neutral_mw[model.prepared.neutrals.index("water")]
    ln_k = np.log(101325 * mw * 1e9 / (R * 298.15 * 10))
    out = model.equilibrate_species(
        species_inventory=SpeciesInventory(gases={"vapor": initial_water}),
        temperature=298.15,
        solids={},
        gas_reactions={"vapor": GasReaction({"water": 1}, ln_k, 298.15)},
    )
    assert out.inventory.gases["vapor"] == pytest.approx(expected_gas)
    assert out.inventory.liquid_neutrals["water"] == pytest.approx(
        initial_water - expected_gas
    )
