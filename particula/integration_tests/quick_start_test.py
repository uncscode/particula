"""Integration test for quick start example in particula."""

import numpy as np

import particula as par


def test_quick_start_runs():
    """Test the quick start example runs without errors."""
    # 1. Build the GasSpecies for an organic vapor:
    organic = (
        par.gas.GasSpeciesBuilder()
        .set_name("organic")
        .set_molar_mass(180e-3, "kg/mol")
        .set_vapor_pressure_strategy(
            par.gas.ConstantVaporPressureStrategy(1e2)  # Pa
        )
        .set_partitioning(True)
        .set_concentration(np.array([1e2]), "kg/m^3")
        .build()
    )

    # 2. Use AtmosphereBuilder to configure temperature, pressure, and species:
    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_temperature(298.15, "K")
        .set_pressure(101325, "Pa")
        .set_more_partitioning_species(organic)
        .build()
    )

    # 3. Build the particle distribution:
    #    Using PresetParticleRadiusBuilder, we set mode radius, GSD, etc.
    particle = (
        par.particles.PresetParticleRadiusBuilder()
        .set_mode(np.array([100e-9]), "m")
        .set_geometric_standard_deviation(np.array([1.2]))
        .set_number_concentration(np.array([1e8]), "1/m^3")
        .set_density(1e3, "kg/m^3")
        .build()
    )

    # 4. Create the Aerosol combining the atmosphere and particle distribution:
    aerosol = (
        par.AerosolBuilder()
        .set_atmosphere(atmosphere)
        .set_particles(particle)
        .build()
    )

    # 5. Define the isothermal condensation strategy:
    condensation_strategy = par.dynamics.CondensationIsothermal(
        molar_mass=180e-3,  # kg/mol
        diffusion_coefficient=2e-5,  # m^2/s
        accommodation_coefficient=1.0,
    )

    # 6. Build the MassCondensation process:
    process = par.dynamics.MassCondensation(condensation_strategy)

    # 7. Execute the condensation process over 10 seconds:
    result = process.execute(aerosol, time_step=10.0)

    #   The result is an Aerosol instance with updated particle properties.
    # print(result)
    # 8. Confirm the result is an Aerosol instance:
    assert isinstance(result, par.Aerosol)
