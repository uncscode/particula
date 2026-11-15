# %% [markdown]
# # Create particles by transferring a bounded precursor source
#
# How can new particles appear without creating material from nothing? This
# example uses the supported CPU nucleation runnable to transfer a mapped gas
# species into initially inactive particle slots. It demonstrates the bookkeeping
# needed to connect a formation-rate model to a finite gas reservoir and finite
# storage capacity.
#
# Read [native state and counts](../../fundamentals/native_state_access.py) first.
# This is a deterministic API and inventory example, not a calibration of
# atmospheric nucleation kinetics.
#
# ## Separate the source law from committing new particles
#
# The one-box volume is 0.25 m³ with three inactive particle slots and one material
# of density 1000 kg m⁻³. Gas contains 10⁻¹² kg m⁻³ precursor plus an inert
# species. Temperature and pressure are 298.15 K and 101325 Pa.
#
# The activation strategy specifies a 0.01 s⁻¹ coefficient, a 1 nm formation
# diameter, and explicit temperature and precursor bounds. The extremely broad
# bounds are demonstration settings, not literature-supported validity limits.
# The source configuration maps precursor gas lane 0 to material lane 0. The
# commit configuration sets slot-weight and exhaustion controls, with requested
# and minimum scale both one so the inventory comparison uses the unscaled state.
#
# `_total_mass` converts count-weighted particle mass to kg m⁻³ before adding
# gas. After a 1 s advance in two substeps, the assertions require an active
# particle population and unchanged mapped particle-plus-gas mass within tolerance.
# Only a success message is printed; the checks are the evidence behind it.

# %%
"""Run the supported bounded CPU-only nucleation example.

The native ``Nucleation`` runnable reads a unified gas container and the current
aerosol environment. It transfers mapped gas into fixed resolved particle slots;
zero duration or zero rate paths are no-ops.
"""

import numpy as np
from particula import Aerosol
from particula.dynamics import (
    ActivationNucleationBuilder,
    Nucleation,
    NucleationCommitConfig,
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from particula.particles.exhaustion import ExhaustionControls


def _total_mass(particles: ParticleData, gas: GasData) -> np.ndarray:
    """Calculate concentration-weighted particle and gas mass by species.

    Args:
        particles: One-box particle data with masses in kg and concentrations
            in represented counts.
        gas: One-box gas data with concentrations in kg/m³.

    Returns:
        Total mass concentration in kg/m³ for each species.
    """
    return (
        np.sum(particles.masses * particles.concentration[..., None], axis=1)
        / particles.volume[:, None]
        + gas.concentration[:, :1]
    )


def run_example() -> Aerosol:
    """Run a supported one-box CPU nucleation step through public APIs.

    The example uses fixed particle capacity and transfers partitioning gas to
    particles in place. It asserts the runnable preserves the input
    ``Aerosol`` identity and conserves the unscaled particle-plus-gas mass.

    Returns:
        The same ``Aerosol`` instance after two equal nucleation substeps.
    """
    particles = ParticleData(
        masses=np.zeros((1, 3, 1), dtype=np.float64),
        concentration=np.zeros((1, 3), dtype=np.float64),
        charge=np.zeros((1, 3), dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([0.25], dtype=np.float64),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        name=["precursor", "inert"],
        molar_mass=np.array([0.1, 0.04], dtype=np.float64),
        concentration=np.array([[1.0e-12, 2.0e-6]], dtype=np.float64),
        partitioning=np.array([True, False]),
    )
    aerosol = Aerosol(
        particles=particles,
        gas=gas,
        environment=EnvironmentData(
            temperature=np.array([298.15], dtype=np.float64),
            pressure=np.array([101325.0], dtype=np.float64),
            saturation_ratio=np.array([[1.0, 1.0]], dtype=np.float64),
        ),
    )
    strategy = (
        ActivationNucleationBuilder()
        .set_parameters(
            {
                "coefficient": 1.0e-2,
                "coefficient_units": "s^-1",
                "coefficient_provenance": (
                    "deterministic documentation example"
                ),
                "precursor_number_concentration_lower": 1.0,
                "precursor_number_concentration_lower_units": "1/m^3",
                "precursor_number_concentration_upper": 1.0e30,
                "precursor_number_concentration_upper_units": "1/m^3",
                "temperature_lower": 200.0,
                "temperature_lower_units": "K",
                "temperature_upper": 400.0,
                "temperature_upper_units": "K",
                "injection_composition": [1, 0],
                "formation_diameter": 1.0,
                "formation_diameter_units": "nm",
            }
        )
        .build()
    )
    nucleation = Nucleation(
        NucleationSourceConfig(
            strategy=strategy,
            precursor_index=0,
            species_map=NucleationSpeciesMap(tuple(gas.name), ((0, 0),)),
        ),
        NucleationCommitConfig(
            maximum_slot_weight=1.0e20,
            source_charge=0.0,
            exhaustion_controls=ExhaustionControls(),
            requested_scale=np.array([1.0], dtype=np.float64),
            minimum_scale=np.array([1.0], dtype=np.float64),
            minimum_volume=np.array([0.01], dtype=np.float64),
        ),
    )
    before = _total_mass(particles, gas)
    result = nucleation.execute(aerosol, time_step=1.0, sub_steps=2)
    assert result is aerosol
    np.testing.assert_allclose(
        _total_mass(particles, gas), before, rtol=1e-12, atol=1e-30
    )
    assert np.any(particles.concentration > 0.0)
    return aerosol


if __name__ == "__main__":
    run_example()
    print("CPU nucleation example completed.")

# %% [markdown]
# ## Use conservation as the first gate
#
# The run demonstrates that an initially empty active population can receive a
# bounded source while retaining the mapped material inventory and aerosol
# identity. Three storage slots do not limit the physical population to three
# particles: slot weights represent counts. A successful inventory check does
# not establish the physical correctness of the chosen formation coefficient.
#
# For a research application, supply a sourced rate law with its units and domain,
# then vary precursor abundance and formation size while tracking number produced,
# gas depletion, material balance, and capacity pressure. Test exhaustion behavior
# before coupling growth or collisions. The [custom-source example](custom_nucleation_single_species.py)
# shows an illustrative alternative law and external vapor supply; it should not
# be treated as a validated replacement for the supported bounded interface.
