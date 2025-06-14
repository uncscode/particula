import numpy as np
import particula as par

def test_quick_start_runs():
    organic = par.gas.GasSpecies(
        name="H2O",
        molar_mass=180e-3,
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e2, "Pa"),
        partitioning=True,
        concentration=1e-4,
    )

    atm = par.gas.Atmosphere(
        temperature=298.15,
        total_pressure=101325,
        partitioning_species=organic,
    )

    particle = (
        par.particles.PresetParticleRadiusBuilder()
        .set_mode(np.array([100e-9]), "m")
        .set_geometric_standard_deviation(np.array([1.2]))
        .set_number_concentration(np.array([1e8]), "1/m^3")
        .set_density(1e3, "kg/m^3")
        .build()
    )

    aerosol = par.Aerosol(atmosphere=atm, particles=particle)
    process = par.dynamics.MassCondensation(
        par.dynamics.CondensationIsothermal(0.018)
    )

    result = process.execute(aerosol, 10.0)

    # Basic sanity check: execution returns an Aerosol instance
    assert isinstance(result, par.Aerosol)
