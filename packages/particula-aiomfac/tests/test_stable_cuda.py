"""Resident and standalone phase-equilibrium qualification on real CUDA."""

# Optional library imports follow pytest.importorskip.
# ruff: noqa: E402
from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("particula")
pytestmark = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="requires NVIDIA CUDA"
)

from particula.dynamics.multivapor_parcel import (
    MultiVaporParcelInitialState,
    ParcelVapor,
)
from particula.dynamics.parcel import ParcelROS3, run_parcel
from particula.execution.multivapor_parcel import ResidentMultiVaporParcel
from particula.particles.activity_factories import ActivityFactory
from particula_aiomfac import AIOMFACModel, MixtureSpec

MIXTURE = MixtureSpec(
    {
        "water": {16: 1},
        "2-propanol": {141: 2, 151: 1, 153: 1},
        "ammonium sulfate": {204: 2, 261: 1},
    }
)
FEED = np.array([0.41, 0.413, 0.177])
TEMPERATURE = 303.15


def _provider(mixture=MIXTURE):
    return ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": mixture,
            "material_mapping": dict(enumerate(mixture.components)),
            "phase_mode": "stable",
        },
    )


@pytest.mark.parametrize(
    "water_mass_fraction,split", [(0.41, True), (0.61, False), (0.97, False)]
)
def test_stable_cuda_phase_equilibrium(water_mass_fraction, split):
    """The same dry ratio traverses the miscibility gap as humidity changes."""
    feed = np.r_[
        water_mass_fraction, (1 - water_mass_fraction) * np.array([0.7, 0.3])
    ]
    expected = _provider().phase_equilibrium(feed, TEMPERATURE)
    model = AIOMFACModel.prepare(MIXTURE, execution="cuda")
    actual = model.phase_equilibrium(feed, TEMPERATURE, 0)
    np.testing.assert_equal(actual.beta_masses is not None, split)
    np.testing.assert_allclose(
        actual.alpha_masses, expected.alpha_masses, atol=3e-8
    )
    np.testing.assert_allclose(
        actual.alpha.neutral_activities,
        expected.alpha.neutral_activities,
        rtol=1e-7,
    )
    if split:
        np.testing.assert_allclose(
            actual.alpha_masses + actual.beta_masses, feed, rtol=1e-14
        )
        np.testing.assert_allclose(
            actual.alpha.neutral_activities,
            actual.beta.neutral_activities,
            rtol=1e-7,
        )
        np.testing.assert_array_less(actual.chemical_potential_error, 1e-7)


def test_two_liquid_resident_ros3_box():
    """Compare GPU water uptake with a refined CPU trajectory."""
    from particula import Aerosol
    from particula.dynamics.condensation.condensation_strategies import (
        CondensationIsothermal,
    )
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.execution.gpu_resident_condensation import (
        GPUResidentCondensation,
    )
    from particula.gas import EnvironmentData, GasData
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
    )
    from particula.particles import ParticleData, SurfaceStrategyVolume

    def box():
        start = Aerosol(
            particles=ParticleData(
                masses=np.array([[FEED * 5e-15, np.zeros(3)]]),
                concentration=np.array([[1e6, 0.0]]),
                density=np.full(3, 1000.0),
                charge=np.zeros((1, 2)),
                volume=np.ones(1),
                distribution_type="particle_resolved",
            ),
            gas=GasData(
                ["water"],
                np.array([0.01801528]),
                np.array([[0.026]]),
                np.array([True]),
            ),
            environment=EnvironmentData([TEMPERATURE], [101325.0], [[1.0]]),
        )
        proc = CondensationIsothermal(
            molar_mass=np.array([0.01801528]),
            diffusion_coefficient=2.5e-5,
            activity_strategy=_provider(),
            surface_strategy=SurfaceStrategyVolume(
                density=1000.0, surface_tension=0.072
            ),
            vapor_pressure_strategy=ConstantVaporPressureStrategy(4246.0),
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
        return start, proc

    state, strategy = box()
    duration = 1.0e-4
    owner = GPUResidentCondensation(
        state,
        strategy,
        device="cuda:0",
        method="ros3",
        particle_atol=1e-29,
        gas_atol=1e-15,
        rtol=1e-7,
    )
    start = perf_counter()
    owner.step(duration)
    out = owner.snapshot()
    cuda_seconds = perf_counter() - start
    cpu, strategy = box()
    start = perf_counter()
    for _ in range(96):
        strategy.step(
            cpu.particles,
            cpu.gas,
            TEMPERATURE,
            101325.0,
            time_step=duration / 96,
        )
    cpu_seconds = perf_counter() - start
    print(
        f"stable two-liquid box, 100 µs, CUDA ROS3 {cuda_seconds:.4g} s, "
        f"CPU 96 substeps {cpu_seconds:.4g} s"
    )
    np.testing.assert_array_less(
        state.particles.masses[0, 0, 0], out.particles.masses[0, 0, 0]
    )
    np.testing.assert_allclose(
        out.particles.masses, cpu.particles.masses, rtol=2e-5, atol=0
    )
    np.testing.assert_array_equal(
        out.particles.masses[..., 1:], state.particles.masses[..., 1:]
    )

    def inventory(aerosol):
        return (
            aerosol.gas.concentration[0, 0]
            + aerosol.particles.masses[0, 0, 0] * 1e6
        )

    np.testing.assert_allclose(
        inventory(out), inventory(state), rtol=0, atol=1e-15
    )
    # The coupled box archives the activity identity in its restart record.
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.execution.gpu_coupled import (
        GPUCondensationControls,
        GPUCoupledSimulation,
    )
    from particula.execution.gpu_coupled_restart import resume_coupled

    interval = 1e-6
    events = (Dilution(DilutionStrategy(0.0)),)
    controls = GPUCondensationControls(
        method="ros3", first_step=interval, max_step=interval
    )
    initial, proc = box()
    with GPUCoupledSimulation(
        initial,
        events,
        proc,
        time_step=interval,
        device="cuda:0",
        controls=controls,
        resident=True,
    ) as coupled:
        coupled.step()
        saved = coupled.checkpoint()
        with resume_coupled(saved, events, proc, device="cuda:0") as resumed:
            coupled.step()
            resumed.step()
            actual = coupled.snapshot()
            replay = resumed.snapshot()
            np.testing.assert_array_equal(
                actual.particles.masses, replay.particles.masses
            )
            np.testing.assert_array_equal(
                actual.gas.concentration, replay.gas.concentration
            )


def test_resident_two_liquid_parcel_restart():
    """Replay a rising parcel with a split droplet and vacant slot."""
    state = MultiVaporParcelInitialState(
        masses=np.array([FEED * 5e-15, np.zeros(3)]),
        number=np.array([1e6, 0.0]),
        vapor=np.array([0.020]),
        material_names=tuple(MIXTURE.components),
        density=np.full(3, 1000.0),
        vapors=(
            ParcelVapor(
                "water", 0, 0.01801528, 4246.0, TEMPERATURE, 2.5e6, 2.11e-5
            ),
        ),
        activity_strategy=_provider(),
        pressure=101325.0,
        temperature=TEMPERATURE,
    )
    dt = 0.0001
    owner = ResidentMultiVaporParcel(
        state, device="cuda:0", adaptive=ParcelROS3(), timestep=dt, updraft=1.0
    )
    before = owner.checkpoint()
    start = perf_counter()
    owner.advance(dt)
    after = owner.checkpoint()
    elapsed = perf_counter() - start
    print(f"stable two-liquid rising parcel, CUDA ROS3 100 µs: {elapsed:.4g} s")
    np.testing.assert_equal(
        np.array_equal(before.coordinates, after.coordinates), False
    )
    owner.restore(before)
    owner.advance(dt)
    np.testing.assert_array_equal(
        owner.checkpoint().coordinates, after.coordinates
    )
    native = run_parcel(
        state,
        steps=1,
        timestep=dt,
        updraft=1.0,
        adaptive=ParcelROS3(),
        device="cuda:0",
    )
    start = perf_counter()
    cpu = run_parcel(state, steps=2, timestep=dt / 2, updraft=1.0)
    print(
        f"stable two-liquid rising parcel, CPU two steps 100 µs: "
        f"{perf_counter() - start:.4g} s"
    )
    np.testing.assert_allclose(
        native.wet_radius[-1], cpu.wet_radius[-1], rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        native.temperature[-1], cpu.temperature[-1], rtol=0, atol=1e-6
    )


def test_stable_parcel_trial_failure():
    """Check the accepted parcel coordinates after a rejected trial."""
    from test_parcel_cuda import make_state

    state = make_state("coupled-trace-bound")
    state = replace(
        state, activity_strategy=_provider(state.activity_strategy.mixture)
    )
    owner = ResidentMultiVaporParcel(
        state,
        device="cuda:0",
        adaptive=ParcelROS3(),
        timestep=0.001,
        updraft=1.0,
    )
    before = owner.checkpoint()
    with pytest.raises(RuntimeError, match="backend status"):
        owner.advance(0.001)
    np.testing.assert_array_equal(before.coordinates, owner.y.numpy())
    np.testing.assert_equal(owner.clock.numpy()[0], 0)


def test_independent_ions_on_cuda():
    """Two salts repartition, conserving each ion and both source labels."""
    mixture = MixtureSpec(
        {
            "water": {16: 1},
            "2-propanol": {141: 2, 151: 1, 153: 1},
            "NaCl": {202: 1, 242: 1},
            "K2SO4": {203: 2, 261: 1},
        }
    )
    feed = np.array([0.4, 0.4, 0.1, 0.1])
    provider = _provider(mixture)
    cpu = provider.phase_equilibrium(feed, TEMPERATURE)
    gpu = provider.phase_equilibrium(feed, TEMPERATURE, execution="cuda")
    np.testing.assert_allclose(
        gpu.alpha_masses + gpu.beta_masses, feed, rtol=2e-13
    )
    np.testing.assert_allclose(cpu.alpha_masses, gpu.alpha_masses, rtol=1e-7)
    all_species = provider.prepared.evaluate(
        feed, TEMPERATURE
    ).species_moles_per_kg_input
    recovered = np.zeros_like(all_species)
    for label in ("alpha", "beta"):
        phase = getattr(gpu, label)
        phase_mass = getattr(gpu, label + "_masses").sum()
        recovered += phase.species_moles_per_kg_input * phase_mass
        np.testing.assert_allclose(
            phase.ion_molalities @ provider.prepared.charge, 0.0, atol=1e-11
        )
        np.testing.assert_allclose(
            phase.ion_molalities, getattr(cpu, label).ion_molalities, rtol=2e-7
        )
    np.testing.assert_allclose(all_species, recovered, rtol=2e-11)
    np.testing.assert_allclose(
        gpu.alpha.ion_activity_products,
        gpu.beta.ion_activity_products,
        rtol=1e-7,
    )
    na = gpu.alpha.ion_ids.index(202)
    cl = gpu.alpha.ion_ids.index(242)
    np.testing.assert_array_less(
        0.05, abs(gpu.alpha.ion_molalities[na] - gpu.alpha.ion_molalities[cl])
    )


def test_device_solid_redissolution_and_mask():
    """The same slot precipitates two crystals, then redissolves both."""
    groups = {
        "water": {16: 1},
        "NaCl": {202: 1, 242: 1},
        "KCl": {203: 1, 242: 1},
    }
    provider = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
            phase_mode="stable",
            solids={"NaCl": "halite", "KCl": "sylvite"},
        ),
    )
    binding = provider.bind_device((1, 2, 3), (0,), "cuda:0")
    counts = wp.array([[1.0, 0.0]], dtype=wp.float64, device="cuda:0")
    tk = wp.array([298.15], dtype=wp.float64, device="cuda:0")
    invalid = wp.zeros(1, dtype=int, device="cuda:0")
    for water, crystallizes in ((0.2, True), (10.0, False), (0.2, True)):
        mass = np.array([[water, 0.5, 0.3], [0.0, 0.0, 0.0]]) * 1e-15
        binding.evaluate(
            wp.array(mass[None], device="cuda:0"), counts, tk, invalid
        )
        expected = provider.neutral_activities(
            mass, 298.15, (0,), np.array([True, False])
        )
        np.testing.assert_equal(invalid.numpy()[0], 0)
        np.testing.assert_allclose(
            binding.values.numpy()[0, :, 0], expected[:, 0], rtol=2e-7
        )
        dissolved = provider.solid_liquid_equilibrium(mass[0], 298.15)
        extent = binding.phase_solver.minerals.work.numpy()[0, -2:]
        np.testing.assert_equal(extent > 0.0, [crystallizes] * 2)
        np.testing.assert_equal(
            dissolved.solid_masses > 0.0, [crystallizes] * 2
        )

    # The qualified Ksp constants are isothermal. A failed trial cannot
    # reuse activities from the previous warm row.
    bad = wp.array([310.0], dtype=wp.float64, device="cuda:0")
    binding.evaluate(
        wp.array(mass[None], device="cuda:0"), counts, bad, invalid
    )
    np.testing.assert_equal(invalid.numpy()[0], 1)
    np.testing.assert_equal(binding.status.numpy()[0, 0], -21)


def test_halitedroplet_rising_parcel():
    """Water uptake redissolves halite during native ROS3 parcel integration."""
    from particula_aiomfac.solids import SolidSalt

    groups = {"water": {16: 1}, "NaCl": {202: 1, 242: 1}}
    # An explicit narrow temperature interval approximates the 298.15 K
    # constant during the parcel's millisecond ascent.
    provider = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
            phase_mode="stable",
            solids={"NaCl": SolidSalt(1.57, 298.14, 298.16)},
        ),
    )
    start = MultiVaporParcelInitialState(
        masses=np.array([[0.715, 0.285], [0.0, 0.0]]) * 5e-15,
        number=np.array([1e6, 0.0]),
        vapor=np.array([0.021]),
        material_names=tuple(groups),
        density=np.array([1000.0, 2165.0]),
        vapors=(
            ParcelVapor("water", 0, 0.01801528, 3169.0, 298.15, 2.5e6, 2.11e-5),
        ),
        activity_strategy=provider,
        pressure=101325.0,
        temperature=298.15,
    )
    owner = ResidentMultiVaporParcel(
        start,
        device="cuda:0",
        adaptive=ParcelROS3(),
        timestep=1e-4,
        updraft=0.5,
    )
    for time, solid in ((0.0004, True), (0.001, False)):
        _, temperature, _, masses = owner.advance(time)
        phases = provider.solid_liquid_equilibrium(masses[0], temperature)
        np.testing.assert_equal(phases.solid_masses[0] > 0.0, solid)
        np.testing.assert_equal(masses[:, 1], start.masses[:, 1])
        np.testing.assert_array_less(start.masses[0, 0], masses[0, 0])


def test_gibbs_carbonate_activities_and_gradient_cuda():
    """Analytical potentials equal finite differences of the extensive G/RT."""
    from particula_aiomfac.phases import component_molar_masses

    spec = MixtureSpec({"water": {16: 1}, "KHCO3": {203: 1, 250: 1}})
    cpu = AIOMFACModel.prepare(spec, carbonate_model="gibbs_consistent")
    gpu = AIOMFACModel.prepare(
        spec, carbonate_model="gibbs_consistent", execution="cuda"
    )
    masses = np.array([[1.0, 0.0], [0.98, 0.02]])
    host = cpu.evaluate(masses, 298.15, diagnostics=True)
    device = gpu.evaluate(masses, 298.15, diagnostics=True)
    np.testing.assert_equal(device.model_identity, host.model_identity)
    np.testing.assert_allclose(
        device.neutral_activities, host.neutral_activities
    )
    np.testing.assert_allclose(device.ion_molalities, host.ion_molalities)
    for name in ("sr", "mr", "lr", "molality_conversion"):
        np.testing.assert_allclose(
            device.diagnostics[name],
            host.diagnostics[name],
            rtol=1.0e-8,
            atol=1.0e-12,
        )

    mw = component_molar_masses(cpu)
    amounts = masses[1] / mw

    def potential(n):
        state = gpu.evaluate(n * mw, 298.15)
        ion = np.log(state.ion_molalities) + state.ion_log_coefficients
        ions = [state.ion_ids.index(i) for i in (203, 250)]
        return np.array([np.log(state.neutral_activities[0]), ion[ions].sum()])

    mu = potential(amounts)
    for j in range(len(amounts)):
        delta = np.zeros(len(amounts))
        delta[j] = amounts[j] * 1.0e-5
        low, high = amounts - delta, amounts + delta
        derivative = (high @ potential(high) - low @ potential(low)) / (
            2 * delta[j]
        )
        np.testing.assert_allclose(derivative, mu[j], rtol=0, atol=3e-8)


@pytest.mark.parametrize(
    "carbon_sources,fractions",
    [
        (
            {"NH4HSO4": {204: 1, 248: 1}, "KHCO3": {203: 1, 250: 1}},
            [0.18, 0.02],
        ),
        ({"K2CO3": {203: 2, 262: 1}}, [0.2]),
        (
            {
                "CO2": {173: 1},
                "NH4HSO4": {204: 1, 248: 1},
                "NaCl": {202: 1, 242: 1},
            },
            [0.002, 0.188, 0.01],
        ),
    ],
)
def test_reactive_carbonate_phase_split_cuda(carbon_sources, fractions):
    """Verify all reactive species across acidic and alkaline phase splits."""
    from particula_aiomfac.phases import phase_equilibrium

    spec = MixtureSpec(
        {"water": {16: 1}, "IPA": {141: 2, 151: 1, 153: 1}, **carbon_sources}
    )
    cpu = AIOMFACModel.prepare(spec, carbonate_model="gibbs_consistent")
    gpu = AIOMFACModel.prepare(
        spec, carbonate_model="gibbs_consistent", execution="cuda"
    )
    feed = np.array([0.4, 0.4, *fractions])
    expected = phase_equilibrium(cpu, feed, 298.15, 0)
    result = gpu.phase_equilibrium(feed, 298.15, 0)
    np.testing.assert_allclose(
        result.alpha_masses + result.beta_masses, feed, rtol=2e-13
    )
    np.testing.assert_allclose(
        result.alpha_masses, expected.alpha_masses, rtol=1e-7
    )
    np.testing.assert_array_less(result.chemical_potential_error, 1.0e-8)
    for phase in ("alpha", "beta"):
        state = getattr(result, phase)
        np.testing.assert_allclose(
            state.ion_molalities,
            getattr(expected, phase).ion_molalities,
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            state.ion_molalities @ cpu.charge, 0.0, atol=1.0e-10
        )
    np.testing.assert_allclose(
        result.alpha.neutral_activities,
        result.beta.neutral_activities,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.alpha.ion_activity_products,
        result.beta.ion_activity_products,
        rtol=1.0e-8,
    )


def test_dissolved_carbon_donor_and_water_on_cuda():
    """Aqueous CO2 and water jointly supply the analytical carbonate."""
    from particula_aiomfac.phases import phase_equilibrium

    spec = MixtureSpec(
        {"water": {16: 1}, "CO2": {173: 1}, "NaCl": {202: 1, 242: 1}}
    )
    cpu = AIOMFACModel.prepare(spec, carbonate_model="gibbs_consistent")
    gpu = AIOMFACModel.prepare(
        spec, carbonate_model="gibbs_consistent", execution="cuda"
    )
    feed = np.array([0.97, 0.005, 0.025])
    expected = phase_equilibrium(cpu, feed, 298.15, 0)
    result = gpu.phase_equilibrium(feed, 298.15, 0)
    np.testing.assert_equal(result.beta is None, True)
    np.testing.assert_allclose(
        result.alpha.ion_molalities, expected.alpha.ion_molalities, rtol=1e-8
    )


@pytest.mark.parametrize("mineral,ion", [("halite", 202), ("sylvite", 203)])
def test_dry_crystal_deliquescence_cuda(mineral, ion):
    """Dry seeds take up water only above their saturated-brine fugacity."""
    from particula_aiomfac.solids import SolidSalt

    groups = {"water": {16: 1}, "seed": {ion: 1, 242: 1}}
    p = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
            solids={"seed": mineral},
            phase_mode="stable",
        ),
    )
    masses = np.array([[0.0, 5.0e-15], [0.0, 0.0]])
    tk = wp.array([298.15], dtype=wp.float64, device="cuda:0")
    active = wp.array([[1.0, 0.0]], dtype=wp.float64, device="cuda:0")
    invalid = wp.zeros(1, dtype=int, device="cuda:0")
    binding = p.bind_device((1, 2, 2), (0,), "cuda:0")
    binding.evaluate(
        wp.array(masses[None], device="cuda:0"), active, tk, invalid
    )
    drh = p.neutral_activities(masses, 298.15, (0,), np.array([True, False]))
    np.testing.assert_equal(invalid.numpy()[0], 0)
    np.testing.assert_allclose(
        binding.values.numpy()[0, :, 0], drh[:, 0], rtol=2.0e-7
    )
    np.testing.assert_array_less(0.7, drh[0, 0])
    np.testing.assert_array_less(drh[0, 0], 0.9)

    # A narrow isothermal Ksp interval permits the initial parcel trial's
    # temperature perturbation. No water is added to the native initial state.
    p = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
            solids={
                "seed": SolidSalt(1.57 if ion == 202 else 0.9, 298.14, 298.16)
            },
            phase_mode="stable",
        ),
    )
    for vapor, wet in ((0.010, False), (0.021, True)):
        start = MultiVaporParcelInitialState(
            masses=masses.copy(),
            number=np.array([1e6, 0.0]),
            vapor=np.array([vapor]),
            material_names=tuple(groups),
            density=np.array([1000.0, 2165.0]),
            vapors=(
                ParcelVapor(
                    "water", 0, 0.01801528, 3169.0, 298.15, 2.5e6, 2.11e-5
                ),
            ),
            activity_strategy=p,
            pressure=101325.0,
            temperature=298.15,
        )
        owner = ResidentMultiVaporParcel(
            start,
            device="cuda:0",
            adaptive=ParcelROS3(),
            timestep=1.0e-6,
            updraft=0.0,
        )
        _, _, _, final = owner.advance(1.0e-6)
        np.testing.assert_equal(final[0, 0] > 0.0, wet)
        np.testing.assert_array_equal(final[1], masses[1])
        np.testing.assert_equal(final[:, 1], masses[:, 1])
        cpu = run_parcel(start, steps=1, timestep=1.0e-6, updraft=0.0)
        np.testing.assert_allclose(
            cpu.wet_radius[-1, 0],
            (3.0 * (final[0] / start.density).sum() / (4.0 * np.pi))
            ** (1.0 / 3.0),
            rtol=1.0e-6,
        )


def test_precipitate_carbonate_with_no_carbonate_source_on_device():
    """CaCO3 can draw on the bicarbonate inventory of a different salt."""
    from particula_aiomfac.phases import component_molar_masses
    from particula_aiomfac.solids import SolidSalt

    groups = {
        "water": {16: 1},
        "CaCl2": {221: 1, 242: 2},
        "NaHCO3": {202: 1, 250: 1},
        "CaCO3": {221: 1, 262: 1},
    }
    p = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            carbonate_model="gibbs_consistent",
            material_mapping=dict(enumerate(groups)),
            solids={"CaCO3": SolidSalt(-8.48)},
            phase_mode="stable",
        ),
    )
    mass = np.array([10.0, 0.1, 0.12, 0.0])
    binding = p.bind_device((1, 1, 4), (0,), "cuda:0")
    invalid = wp.zeros(1, dtype=int, device="cuda:0")
    binding.evaluate(
        wp.array(mass[None, None], device="cuda:0"),
        wp.ones((1, 1), dtype=wp.float64, device="cuda:0"),
        wp.array([298.15], dtype=wp.float64, device="cuda:0"),
        invalid,
    )
    np.testing.assert_equal(invalid.numpy()[0], 0)
    expected = p.solid_liquid_equilibrium(mass, 298.15)
    # Resident extents are moles per largest expanded input mass. Water is
    # the largest input here and CaCO3 is the last original formula.
    calcite = (
        binding.phase_solver.minerals.work.numpy()[0, -1]
        * mass[0]
        * component_molar_masses(p.prepared)[-1]
    )
    np.testing.assert_array_less(0.06, calcite)
    np.testing.assert_allclose(calcite, expected.solid_masses[0], rtol=1e-7)
    np.testing.assert_allclose(
        binding.values.numpy()[0, 0, 0],
        expected.liquids.alpha.neutral_activities[0],
        rtol=1e-8,
    )


def test_small_new_liquid_near_binodal_cuda():
    """Backtracking produces an energy-decreasing incipient organic phase."""
    groups = {
        "IPA": {141: 2, 151: 1, 153: 1},
        "NaCl": {202: 1, 242: 1},
        "water": {16: 1},
    }
    feed = np.array([2.5812405, 15.85285599, 47.4187595])
    p = _provider(MixtureSpec(groups))
    ref = p.phase_equilibrium(feed, 298.15)
    gpu = p.phase_equilibrium(feed, 298.15, execution="cuda")
    np.testing.assert_allclose(gpu.beta_masses, ref.beta_masses, rtol=2e-7)
    np.testing.assert_array_less(gpu.beta_masses.sum() / feed.sum(), 0.001)


@pytest.mark.parametrize(
    "feed",
    [
        [
            0.6620006564835846,
            0.2720568201925576,
            0.03489894965646451,
            0.031043573667393257,
        ],
        [
            0.6962452632978388,
            0.2286434854507551,
            0.05568961458106756,
            0.019421636670338565,
        ],
    ],
)
def test_two_salt_incipient_liquid_amount_cuda(feed):
    """Resolve small minority phases at two kinetic humidity-ramp stages."""
    mixture = MixtureSpec(
        {
            "water": {16: 1},
            "IPA": {141: 2, 151: 1, 153: 1},
            "NaCl": {202: 1, 242: 1},
            "KCl": {203: 1, 242: 1},
        }
    )
    provider = _provider(mixture)
    mass = np.array(feed)
    cpu = provider.phase_equilibrium(mass, 298.15)
    gpu = provider.phase_equilibrium(mass, 298.15, execution="cuda")
    np.testing.assert_allclose(
        gpu.beta_masses, cpu.beta_masses, rtol=0, atol=2.0e-8
    )
    np.testing.assert_array_less(gpu.beta_masses.sum(), 0.01)
    np.testing.assert_array_less(gpu.chemical_potential_error, 1.0e-8)
