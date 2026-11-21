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


def test_discrete_heating_rejects_before_registering_fields():
    """Reject storage that cannot hold per-slot temperature and power."""
    resolved, process = illuminated_state()
    particles = ParticleData(
        masses=resolved.particles.masses,
        concentration=resolved.particles.slot_concentration_density,
        charge=resolved.particles.charge,
        density=resolved.particles.density,
        volume=np.ones(1),
        distribution_type="discrete",
    )
    aerosol = Aerosol(
        particles=particles, gas=resolved.gas, environment=resolved.environment
    )
    with pytest.raises(ValueError, match="requires particle_resolved storage"):
        process.attach_fields(aerosol)
    with pytest.raises(ValueError, match="requires particle_resolved storage"):
        process.validate(aerosol)
    assert not aerosol.fields.specs


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


@pytest.fixture(params=["cpu", "cuda:0"])
def resident_device(request):
    """Run resident kernels on each available device."""
    import warp as wp

    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def _register_heating(capacity=512):
    from particula import ProcessRegistry
    from particula.dynamics.optical_heating_gpu import OpticalHeatingAdapter

    registry = ProcessRegistry()
    registry.register(
        OpticalHeating, OpticalHeatingAdapter(coefficient_capacity=capacity)
    )
    return registry


def test_resident_optics_follow_mass_and_temperature(resident_device, tmp_path):
    """Recompute optical feedback on device and record post-step state."""
    from particula.execution.optical_recording import (
        OpticalHistory,
        OpticalObserver,
        read_optical_history,
    )
    from particula.execution.prepared_recording import run_prepared
    from particula.optics import population_optics

    initial, process = illuminated_state()
    config = OpticalConfig(
        [450e-9, 550e-9, 700e-9],
        species_index=[(1.33, 1.73 + 0.03j)] * 3,
    )
    process = OpticalHeating(
        config,
        [3e4, 4e4, 2e4],
        gas_water="water",
        physics=process.physics,
    )
    reference = CPUSimulation(initial, process, time_step=DT)
    count = 12
    reference.step(DT)
    first_power = reference.snapshot().fields[process.power_field]
    for step in range(2, count + 1):
        reference.step(step * DT)
    expected = reference.snapshot()
    path = tmp_path / "heating.sqlite"
    registry = _register_heating()
    with process.prepare(
        initial,
        device=resident_device,
        time_step=DT,
        process_registry=registry,
    ) as owner:
        history = OpticalHistory(
            path, config, variables=("absorption",), water_vapor="water"
        )
        observer = OpticalObserver(
            owner, history, batch_size=3, coefficient_capacity=512
        )
        run_prepared(owner, count * DT, optics=observer, optical_initial=True)
        actual = owner.snapshot()
        checkpoint = owner.checkpoint()
        owner.run(2)
        continued = owner.snapshot()
        with process.resume(checkpoint, process_registry=registry) as resumed:
            resumed.run(2)
            npt.assert_array_equal(
                resumed.snapshot().particles.masses,
                continued.particles.masses,
            )
    assert not np.array_equal(actual.fields[process.power_field], first_power)
    npt.assert_allclose(
        actual.particles.masses, expected.particles.masses, rtol=6e-12
    )
    npt.assert_allclose(
        actual.gas.concentration, expected.gas.concentration, rtol=2e-14
    )
    for name, array in expected.fields.items():
        npt.assert_allclose(actual.fields[name], array, rtol=2e-9, atol=1e-21)
    metadata, observations = list(read_optical_history(path))[-1]
    assert metadata["relative_humidity"]["gas_species"] == "water"
    assert metadata["relative_humidity"]["axes"] == ["time", "box"]
    humidity = WaterBuckStrategy().saturation_ratio(
        expected.gas.concentration[:, 1],
        MOLAR_WATER,
        expected.environment.temperature,
    )
    npt.assert_allclose(
        observations["relative_humidity"][-1], humidity, rtol=2e-14
    )
    npt.assert_allclose(
        observations["absorption"][-1],
        population_optics(actual.particles, config).absorption,
        rtol=2e-10,
    )


@pytest.mark.parametrize(
    ("capacity", "irradiance", "status"), [(1, 9e4, 102), (512, 1e12, 103)]
)
def test_resident_failed_batch_does_not_publish(
    resident_device,
    capacity,
    irradiance,
    status,
):
    """Reject invalid optics or an unbracketed root before touching state."""
    initial, process = illuminated_state(irradiance=irradiance)
    with process.prepare(
        initial,
        device=resident_device,
        time_step=DT,
        process_registry=_register_heating(capacity),
    ) as owner:
        owner.step()
        with pytest.raises(ValueError, match=f"failure {status}"):
            owner.synchronize()
        with owner._scope():
            npt.assert_array_equal(
                owner._session.particles.masses.numpy(),
                initial.particles.masses,
            )
            npt.assert_array_equal(
                owner._session.gas.concentration.numpy(),
                initial.gas.concentration,
            )
            for name, buffer in owner._extension_fields.arrays.items():
                npt.assert_array_equal(buffer.numpy(), initial.fields[name])
