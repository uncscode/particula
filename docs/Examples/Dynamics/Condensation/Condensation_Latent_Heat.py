# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula_dev312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CPU latent-heat condensation bookkeeping
#
# This runnable example demonstrates a small CPU-only condensation workflow that
# uses public builder and factory APIs, wraps the strategy in
# `par.dynamics.MassCondensation`, and reports latent-heat bookkeeping from real
# mass transfer. The reported energy is diagnostic only: it does not feed back
# into the temperature state in this example.

# %%
import copy

import numpy as np
import particula as par

CPU_ONLY_NOTE = (
    "CPU-only example: this workflow uses public CPU condensation APIs only."
)
BOOKKEEPING_ONLY_NOTE = (
    "Bookkeeping only: latent-heat energy is diagnostic only, not thermal "
    "feedback. Positive values indicate condensation and negative values "
    "indicate evaporation."
)


def _as_float(value: float | np.ndarray) -> float:
    """Return a scalar float from scalar-like numeric values."""
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _build_aerosol() -> par.Aerosol:
    """Build a small supersaturated single-species aerosol state."""
    molar_mass_water = 18.015e-3  # kg/mol
    temperature = 298.15  # K
    vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy(
        "water_buck"
    )
    saturation_concentration = vapor_pressure_water.saturation_concentration(
        molar_mass=molar_mass_water,
        temperature=temperature,
    )
    gas_species = (
        par.gas.GasSpeciesBuilder()
        .set_molar_mass(molar_mass_water, "kg/mol")
        .set_vapor_pressure_strategy(vapor_pressure_water)
        .set_concentration(saturation_concentration * 1.03, "kg/m^3")
        .set_name("H2O")
        .set_partitioning(True)
        .build()
    )
    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_more_partitioning_species(gas_species)
        .set_temperature(25.0, temperature_units="degC")
        .set_pressure(1.0, pressure_units="atm")
        .build()
    )

    particle_radii = np.array([30e-9, 45e-9, 60e-9, 90e-9], dtype=np.float64)
    density = 1000.0  # kg/m^3
    particle_mass = (4.0 / 3.0) * np.pi * particle_radii**3 * density
    particles = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(par.particles.ActivityIdealMass())
        .set_surface_strategy(par.particles.SurfaceStrategyVolume())
        .set_mass(particle_mass.reshape(-1, 1), "kg")
        .set_density(np.array([density], dtype=np.float64), "kg/m^3")
        .set_charge(np.zeros_like(particle_radii, dtype=np.float64))
        .set_volume(1.0, "cm^3")
        .build()
    )
    return par.Aerosol(atmosphere=atmosphere, particles=particles)


def run_example() -> dict[str, float | list[float] | str | int]:
    """Run the latent-heat condensation example and return diagnostics."""
    aerosol = _build_aerosol()
    latent_heat_strategy = par.gas.LatentHeatFactory().get_strategy(
        strategy_type="constant",
        parameters={
            "latent_heat_ref": 2.26e6,
            "latent_heat_ref_units": "J/kg",
        },
    )
    condensation_strategy = par.dynamics.CondensationFactory().get_strategy(
        "latent_heat",
        {
            "molar_mass": 18.015e-3,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
            "latent_heat_strategy": latent_heat_strategy,
        },
    )
    condensation = par.dynamics.MassCondensation(
        condensation_strategy=condensation_strategy
    )

    current = par.Aerosol(
        atmosphere=copy.deepcopy(aerosol.atmosphere),
        particles=copy.deepcopy(aerosol.particles),
    )
    initial_gas_concentration = _as_float(
        current.atmosphere.partitioning_species.get_concentration()
    )
    initial_particle_mass_concentration = _as_float(
        current.particles.get_mass_concentration()
    )

    per_step_latent_heat_energies: list[float] = []
    time_step = 0.05
    sub_steps = 2
    n_steps = 5
    for _ in range(n_steps):
        current = condensation.execute(
            current, time_step=time_step, sub_steps=sub_steps
        )
        per_step_latent_heat_energies.append(
            float(condensation_strategy.last_latent_heat_energy)
        )

    final_gas_concentration = _as_float(
        current.atmosphere.partitioning_species.get_concentration()
    )
    final_particle_mass_concentration = _as_float(
        current.particles.get_mass_concentration()
    )
    cumulative_latent_heat_energy = float(sum(per_step_latent_heat_energies))

    result: dict[str, float | list[float] | str | int] = {
        "initial_gas_concentration": initial_gas_concentration,
        "final_gas_concentration": final_gas_concentration,
        "initial_particle_mass_concentration": (
            initial_particle_mass_concentration
        ),
        "final_particle_mass_concentration": final_particle_mass_concentration,
        "particle_mass_change": (
            final_particle_mass_concentration
            - initial_particle_mass_concentration
        ),
        "per_step_latent_heat_energies": per_step_latent_heat_energies,
        "cumulative_latent_heat_energy": cumulative_latent_heat_energy,
        "cpu_only_note": CPU_ONLY_NOTE,
        "bookkeeping_only_note": BOOKKEEPING_ONLY_NOTE,
        "iteration_count": n_steps,
    }
    if cumulative_latent_heat_energy == 0.0:
        result["zero_transfer_explanation"] = (
            "No net latent-heat signal was produced because the chosen setup "
            "did not transfer measurable vapor mass."
        )
    return result


def main() -> None:
    """Run the example and print concise user-facing diagnostics."""
    result = run_example()
    print(result["cpu_only_note"])
    print(result["bookkeeping_only_note"])
    print(
        "Gas concentration [kg/m^3]: "
        f"{result['initial_gas_concentration']:.6e} -> "
        f"{result['final_gas_concentration']:.6e}"
    )
    print(
        f"Particle mass change [kg/m^3]: {result['particle_mass_change']:.6e}"
    )
    print(
        "Per-step latent heat energy [J]: "
        f"{result['per_step_latent_heat_energies']}"
    )
    print(
        "Cumulative latent heat energy [J]: "
        f"{result['cumulative_latent_heat_energy']:.6e}"
    )
    if "zero_transfer_explanation" in result:
        print(
            f"Zero-transfer explanation: {result['zero_transfer_explanation']}"
        )


if __name__ == "__main__":
    main()
