"""A warm illuminated solution sphere conserves water and energy."""

import numpy as np
import numpy.testing as npt
import pytest
from particula import Aerosol
from particula.dynamics.optical_heating import HeatingPhysics, OpticalHeating
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.optics import OpticalConfig, get_mie_efficiencies
from particula.particles import ParticleData
from particula.util.constants import GAS_CONSTANT
from scipy.constants import Stefan_Boltzmann
from scipy.optimize import brentq

TEMPERATURE = 298.15
MOLAR_WATER = 0.01801528
DT = 5e-5


def illuminated_state(*, irradiance=9e4, gas_offset=0):
    """Return one illuminated resolved box with an inert gas and an empty slot."""
    model = HeatingPhysics(radiation_temperature=TEMPERATURE)
    radius = np.array([4e-6, 6e-6, 0.0])
    wet = (4 * np.pi / 3) * radius**3
    mass = wet[:, None] * np.array([0.72 * 997, 0.28 * 1600])
    particles = ParticleData(
        masses=mass[None, :, :],
        concentration=np.array([[24.0, 40.0, 0.0]]),
        charge=np.zeros((1, 3)),
        density=np.array([997.0, 1600.0]),
        volume=np.array([0.04]),
        distribution_type="particle_resolved",
    )
    vapor = (
        0.72
        / (0.72 + model.kappa * 0.28)
        * WaterBuckStrategy().pure_vapor_pressure(TEMPERATURE)
        * MOLAR_WATER
        / (GAS_CONSTANT * TEMPERATURE)
    )
    gas = GasData(
        name=["inert", "water"],
        molar_mass=np.array([0.05, MOLAR_WATER]),
        concentration=np.array([[4e-7, vapor + gas_offset]]),
        partitioning=np.array([False, True]),
    )
    state = Aerosol(
        particles=particles,
        gas=gas,
        environment=EnvironmentData(
            temperature=np.array([TEMPERATURE]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.zeros((1, 2)),
        ),
    )
    light = OpticalHeating(
        OpticalConfig([550e-9], particle_index=1.5 + 0.08j),
        [irradiance],
        gas_water="water",
        physics=model,
    )
    light.attach_fields(state)
    return state, light


def test_energy_water_and_temperature_match_independent_roots():
    """Compare a directly calculated first-step energy and mass ledger."""
    aerosol, process = illuminated_state()
    start = aerosol.particles.masses.copy()
    initial_water = aerosol.gas.concentration[0, 1]
    weights = aerosol.particles.concentration[0, :2] / 0.04
    radius = aerosol.particles.radii[0, :2]
    q = get_mie_efficiencies(radius, 550e-9, 1.5 + 0.08j)
    power = 9e4 * q.absorption * np.pi * radius**2
    expected_loss, expected_temp = [], []
    p = process.physics
    for i, r in enumerate(radius):
        water, solute = start[0, i]
        aw = 0.72 / (0.72 + p.kappa * 0.28)

        def loss(t, r=r, water=water, aw=aw):
            c = t - 273.15
            buck = 611.21 * np.exp((18.678 - c / 234.5) * c / (257.14 + c))
            saturated = aw * buck * MOLAR_WATER / (GAS_CONSTANT * t)
            return min(
                water,
                4
                * np.pi
                * r
                * p.diffusivity
                * DT
                * max(saturated - initial_water, 0),
            )

        def balance(
            t, r=r, water=water, solute=solute, absorbed=power[i], loss=loss
        ):
            area = 4 * np.pi * r**2
            return (
                (t - TEMPERATURE)
                * (
                    p.water_heat_capacity * water
                    + p.solute_heat_capacity * solute
                    + DT * p.conductivity * area / r
                )
                + DT
                * p.emissivity
                * Stefan_Boltzmann
                * area
                * (t**4 - TEMPERATURE**4)
                + p.latent_heat * loss(t)
                - DT * absorbed
            )

        root = brentq(balance, TEMPERATURE, TEMPERATURE + 3, xtol=1e-12)
        expected_temp.append(root)
        expected_loss.append(loss(root))
    owner = CPUSimulation(aerosol, process, time_step=DT)
    owner.step(DT)
    state = owner.snapshot()
    temp = state.fields[process.temperature_field][0]
    npt.assert_allclose(temp[:2], expected_temp, rtol=2e-14)
    npt.assert_allclose(
        start[0, :2, 0] - state.particles.masses[0, :2, 0],
        expected_loss,
        rtol=1e-10,
    )
    npt.assert_array_equal(state.particles.masses[..., 1], start[..., 1])
    npt.assert_allclose(
        state.gas.concentration[0, 1] - initial_water,
        weights @ np.array(expected_loss),
        atol=5e-18,
    )
    npt.assert_array_equal(state.gas.concentration[0, 0], 4e-7)
    npt.assert_array_equal(state.fields[process.power_field][0, 2], 0.0)
    energy = np.array([state.fields[key][0] for key in process.energy_fields])
    npt.assert_allclose(energy[0], weights @ power * DT, rtol=1e-13)
    npt.assert_allclose(energy[0], energy[1:].sum(), rtol=1e-10)


def test_dark_control_and_failed_temperature_step():
    """A dark equilibrium stays at rest and an invalid solve changes no state."""
    dark, process = illuminated_state(irradiance=0)
    initial = dark.particles.masses.copy()
    process.execute(dark, DT)
    npt.assert_allclose(dark.particles.masses, initial, rtol=3e-15)

    hot, process = illuminated_state(irradiance=1e12)
    initial = hot.particles.masses.copy()
    with pytest.raises(ValueError, match="temperature left"):
        process.execute(hot, DT)
    npt.assert_array_equal(hot.particles.masses, initial)
    npt.assert_array_equal(hot.fields[process.temperature_field], 0)
