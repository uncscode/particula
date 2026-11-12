"""Native replacements for former nested atmosphere construction contracts."""

import numpy as np
import numpy.testing as npt
import pytest
from particula import Aerosol
from particula.gas import EnvironmentData, GasDataBuilder
from particula.particles import create_resolved_particle_data


@pytest.mark.parametrize(
    ("inert_name", "inert_mass"),
    [("Hydrogen", 0.002), ("Nitrogen", 0.028)],
)
def test_native_gas_environment_construction_and_extension(
    inert_name, inert_mass
):
    """Full-gas lane extension requires matching environment replacement."""
    names = ["Oxygen", inert_name]
    gas = (
        GasDataBuilder()
        .set_names(names)
        .set_molar_mass([0.032, inert_mass])
        .set_concentration([0.21, 0.79])
        .set_partitioning([True, False])
        .build()
    )
    environment = EnvironmentData(
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.array([[0.0, 0.0]]),
    )
    particle = create_resolved_particle_data(masses=[1e-18], density=1000.0)
    aerosol = Aerosol(particles=particle, gas=gas, environment=environment)
    npt.assert_array_equal(aerosol.environment.temperature, [298.15])
    npt.assert_array_equal(aerosol.environment.pressure, [101325.0])
    assert gas.name == names
    npt.assert_array_equal(gas.molar_mass, [0.032, inert_mass])
    npt.assert_array_equal(gas.concentration, [[0.21, 0.79]])
    assert gas.n_species == 2
    assert np.count_nonzero(gas.partitioning) == 1

    added_name = "Nitrogen" if inert_name == "Hydrogen" else "Water"
    extended = (
        GasDataBuilder()
        .set_names([*names, added_name])
        .set_molar_mass(
            [0.032, inert_mass, 0.028 if added_name == "Nitrogen" else 0.018]
        )
        .set_concentration([0.21, 0.79, 0.0])
        .set_partitioning([True, False, True])
        .build()
    )
    with pytest.raises(ValueError):
        aerosol.gas = extended
    assert aerosol.gas is gas
    assert aerosol.environment is environment
    matching = EnvironmentData(
        temperature=environment.temperature.copy(),
        pressure=environment.pressure.copy(),
        saturation_ratio=np.zeros((1, 3)),
    )
    aerosol.replace_data(particles=particle, gas=extended, environment=matching)
    assert aerosol.gas is extended
    assert aerosol.environment is matching
    assert extended.name == [*names, added_name]
    assert extended.n_species == 3
    assert np.count_nonzero(extended.partitioning) == 2
    assert gas.name == names
    npt.assert_array_equal(gas.concentration, [[0.21, 0.79]])
    npt.assert_array_equal(environment.saturation_ratio, [[0.0, 0.0]])


def test_native_environment_requires_pressure():
    """Temperature alone cannot supply the required pressure field."""
    with pytest.raises(TypeError, match="pressure"):
        EnvironmentData(
            temperature=np.array([298.15]), saturation_ratio=np.zeros((1, 1))
        )  # type: ignore[call-arg]
