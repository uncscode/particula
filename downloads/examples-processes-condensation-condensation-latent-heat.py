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
# # How much energy accompanies a condensation step?
#
# A mass-conserving isothermal calculation does not tell us how the temperature
# would change. This small CPU example isolates the latent-energy bookkeeping:
# five 0.05 s condensation calls transfer water between four resolved particles
# and a finite vapor reservoir in 10⁻⁶ m³. Initial water saturation is 1.03 at
# 298.15 K. The energy is diagnostic only and never updates temperature.
#
# Read [the finite-reservoir example](../../../tutorials/condensation/condensation_3_massresolved.py) for coupled
# mass transfer. Here the constant latent heat is 2.26 × 10⁶ J kg⁻¹. Follow
# `run_example()` from copied initial state through mass transfer to per-call and
# cumulative energy density. Positive energy denotes net condensation; negative
# energy denotes evaporation.
#
# `last_latent_heat_energy` stores only the most recent substep's extensive energy
# in joules. This example deliberately uses one substep per call, divides once by
# physical volume to obtain J m⁻³, and sums the five call values explicitly.
# Reading only the final value would not recover the accumulated signal.

# %%
"""CPU-only latent-heat condensation example.

This runnable example demonstrates latent-heat bookkeeping from real
condensation mass transfer using public CPU condensation APIs. The
reported energy is diagnostic only and does not feed back into the
temperature state.
"""

import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

CPU_ONLY_NOTE = (
    "CPU-only example: this workflow uses public CPU condensation APIs only."
)
BOOKKEEPING_ONLY_NOTE = (
    "Bookkeeping only: latent-heat energy density is diagnostic only, not "
    "thermal feedback. Positive values indicate condensation and negative "
    "values indicate evaporation."
)
EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE = 1.0e-18
LATENT_HEAT_REFERENCE = 2.26e6  # J/kg
BOX_VOLUME = 1.0e-6  # m^3
RUN_TIME_STEP = 0.05
RUN_SUB_STEPS = 1
RUN_ITERATION_COUNT = 5


def _as_float(value: float | np.ndarray) -> float:
    """Return the first scalar value from a scalar-like numeric input.

    Args:
        value: Scalar or array-like numeric value to coerce to ``float``.

    Returns:
        The first value from ``value`` as a Python ``float``.
    """
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _build_aerosol() -> par.Aerosol:
    """Build a small supersaturated single-species aerosol state.

    Returns:
        Aerosol configured to produce a measurable condensation signal in
        a single CPU box.
    """
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
        par.gas.GasDataBuilder()
        .set_molar_mass([molar_mass_water], "kg/mol")
        .set_concentration([saturation_concentration * 1.03], "kg/m^3")
        .set_names(["H2O"])
        .set_partitioning([True])
        .build()
    )
    environment = par.gas.EnvironmentData(
        temperature=[temperature],
        pressure=[101325.0],
        saturation_ratio=[[1.03]],
    )

    particle_radii = np.array([30e-9, 45e-9, 60e-9, 90e-9], dtype=np.float64)
    density = 1000.0  # kg/m^3
    particle_mass = (4.0 / 3.0) * np.pi * particle_radii**3 * density
    particles = par.particles.create_resolved_particle_data(
        masses=particle_mass.reshape(-1, 1),
        density=[density],
        volume=BOX_VOLUME,
    )
    return par.Aerosol(
        particles=particles, gas=gas_species, environment=environment
    )


def run_example() -> dict[str, float | list[float] | str | int]:
    """Run the latent-heat example and return bookkeeping diagnostics.

    The example executes several short CPU condensation steps, records
    the per-call latent-heat bookkeeping reported by the strategy, and
    accumulates the cumulative energy density explicitly. The example uses
    ``sub_steps=1`` so each stored value corresponds to the full runnable
    call. Positive latent-heat values indicate condensation, while
    negative values indicate evaporation.

    Returns:
        Dictionary containing initial and final gas concentrations,
        initial and final particle mass concentrations, particle mass
        change, per-call latent-heat energy densities, cumulative
        latent-heat energy density, user-facing note strings, and the
        iteration count. A
        ``zero_transfer_explanation`` entry is added when the cumulative
        energy is effectively zero within a small bookkeeping tolerance.
    """
    aerosol = _build_aerosol()
    latent_heat_strategy = par.gas.LatentHeatFactory().get_strategy(
        strategy_type="constant",
        parameters={
            "latent_heat_ref": LATENT_HEAT_REFERENCE,
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
            "activity_strategy": par.particles.ActivityIdealMass(),
            "surface_strategy": par.particles.SurfaceStrategyVolume(),
            "vapor_pressure_strategy": par.gas.WaterBuckStrategy(),
            "species_map": CondensationSpeciesMap(("H2O",), ((0, 0),)),
        },
    )
    condensation = par.dynamics.MassCondensation(
        condensation_strategy=condensation_strategy
    )

    current = par.Aerosol(
        gas=aerosol.gas.copy(),
        environment=aerosol.environment.copy(),
        particles=aerosol.particles.copy(),
    )
    initial_gas_concentration = _as_float(current.gas.concentration)
    initial_particle_mass_concentration = _as_float(
        current.particles.species_mass_density
    )

    # Native bookkeeping reports extensive J; divide by physical volume once.
    per_call_latent_heat_energy_densities: list[float] = []
    for _ in range(RUN_ITERATION_COUNT):
        current = condensation.execute(
            current,
            time_step=RUN_TIME_STEP,
            sub_steps=RUN_SUB_STEPS,
        )
        per_call_latent_heat_energy_densities.append(
            float(condensation_strategy.last_latent_heat_energy)
            / float(current.particles.volume[0])
        )

    final_gas_concentration = _as_float(current.gas.concentration)
    final_particle_mass_concentration = _as_float(
        current.particles.species_mass_density
    )
    cumulative_latent_heat_energy_density = float(
        sum(per_call_latent_heat_energy_densities)
    )

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
        "per_call_latent_heat_energy_densities": (
            per_call_latent_heat_energy_densities
        ),
        "cumulative_latent_heat_energy_density": (
            cumulative_latent_heat_energy_density
        ),
        "latent_heat_reference": LATENT_HEAT_REFERENCE,
        "cpu_only_note": CPU_ONLY_NOTE,
        "bookkeeping_only_note": BOOKKEEPING_ONLY_NOTE,
        "iteration_count": RUN_ITERATION_COUNT,
        "sub_steps_per_call": RUN_SUB_STEPS,
    }
    if np.isclose(
        cumulative_latent_heat_energy_density,
        0.0,
        rtol=0.0,
        atol=EFFECTIVE_ZERO_LATENT_HEAT_ENERGY_TOLERANCE,
    ):
        result["zero_transfer_explanation"] = (
            "No net latent-heat signal was produced because the chosen setup "
            "did not transfer measurable vapor mass concentration."
        )
    return result


def main() -> None:
    """Run the example and print concise user-facing diagnostics.

    Returns:
        None.
    """
    result = run_example()
    print(result["cpu_only_note"])
    print(result["bookkeeping_only_note"])
    print(
        "Gas concentration [kg/m^3]: "
        f"{result['initial_gas_concentration']:.6e} -> "
        f"{result['final_gas_concentration']:.6e}"
    )
    print(
        "Particle mass concentration [kg/m^3]: "
        f"{result['initial_particle_mass_concentration']:.6e} -> "
        f"{result['final_particle_mass_concentration']:.6e}"
    )
    print(
        f"Particle mass change [kg/m^3]: {result['particle_mass_change']:.6e}"
    )
    print(
        "Per-call latent heat energy density [J/m^3] "
        "(each entry is the last value reported after one runnable call): "
        f"{result['per_call_latent_heat_energy_densities']}"
    )
    print(
        "Cumulative latent heat energy density [J/m^3]: "
        f"{result['cumulative_latent_heat_energy_density']:.6e}"
    )
    if "zero_transfer_explanation" in result:
        print(
            f"Zero-transfer explanation: {result['zero_transfer_explanation']}"
        )


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Interpret energy without claiming thermal feedback
#
# The saved cumulative energy density is about 0.0350 J m⁻³, consistent with the
# reported particle mass-density gain of 1.549 × 10⁻⁸ kg m⁻³ times the constant
# latent heat. Gas depletion supplies that gain. This is an accounting comparison,
# not a prediction or measurement of temperature rise.
#
# To estimate a thermal response, specify the relevant heat capacity, heat exchange
# with the chamber or surrounding gas, and whether properties change during the
# step. Compare a coupled result with this isothermal reference and check energy
# and species inventories independently. If you increase `sub_steps`, accumulate
# energy at the appropriate update level rather than treating the last-substep
# property as a full-call total.
