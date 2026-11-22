"""Qualify per-species transport, equilibrium refresh and failed-row isolation."""

import os

import numpy as np
import pytest
from scripts.docs_examples import import_study

pytest.importorskip("particula_aiomfac")
box = import_study("aiomfac-closed-box")
require_cuda = pytest.mark.skipif(
    os.getenv("AIOMFAC_TEST_CUDA") != "1",
    reason="requires NVIDIA qualification",
)


@pytest.mark.parametrize("permuted", [False, True])
def test_flux_directions_and_trajectory_refinement(permuted):
    """Compare two finite reservoirs to equilibrium pressures and Radau."""
    result = box.simulate(cuda=False, permuted=permuted)
    for case in result["reservoirs"]:
        signs = np.sign(case["initial"]["flux_kg_s"])
        np.testing.assert_array_equal(
            signs, case["direction"] * np.array([1, -1])
        )
        assert max(abs(np.array(case["initial"]["pressure_error_Pa"]))) < 2e-12
        assert case["metrics"]["cpu-fixed-8"]["budget"] <= 1e-10
        activity = np.asarray(case["reference"]["activity"])
        assert np.all(activity[0, 0] != activity[-1, 0])
        if permuted:
            np.testing.assert_array_equal(activity[:, -1], 0)
            assert np.all(activity[0, 0] != activity[0, 1])


@require_cuda
def test_permuted_cuda_box():
    """Exercise both reservoir signs, all five CUDA solvers and restart."""
    report = box.simulate(cuda=True, permuted=True)
    for record in report["reservoirs"]:
        assert all(m["budget"] < 1e-10 for m in record["metrics"].values())
    for state in report["restart"]:
        assert state["coupling_uploads"] == state["coupling_downloads"] == 0


def electrolyte_case(trace=False):
    """Water-only exchange for an analytical ammonium bisulfate mixture."""
    from particula import Aerosol
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.gas import EnvironmentData, GasData
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
    )
    from particula.particles import ParticleData
    from particula.particles.activity_factories import ActivityFactory
    from particula_aiomfac import MixtureSpec

    names = {"water": {16: 1}, "NH4HSO4": {204: 1, 248: 1}}
    rows = [[0.88, 0.12]]
    temperature = 298.15
    if trace:
        names["KHCO3"] = {203: 1, 250: 1}
        rows = [[0.97, 0.02, 0.01], [0.999999998, 1e-9, 1e-9]]
        temperature = 295.0
    mass = 5e-15 * np.asarray(rows)[None]
    particle = ParticleData(
        masses=mass,
        concentration=np.full(mass.shape[:2], 1e10),
        charge=np.zeros(mass.shape[:2]),
        density=np.full(mass.shape[-1], 1000.0),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        name=["water"],
        molar_mass=np.array([0.01801528]),
        concentration=np.array([[0.026]]),
        partitioning=np.array([True]),
    )
    environment = EnvironmentData(
        np.array([temperature]), np.array([101325.0]), np.ones((1, 1))
    )
    process = box.CondensationIsothermal(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=2.5e-5,
        activity_strategy=ActivityFactory().get_strategy(
            "aiomfac",
            {
                "mixture": MixtureSpec(names),
                "material_mapping": dict(enumerate(names)),
            },
        ),
        surface_strategy=box.SurfaceStrategyVolume(
            density=1000.0, surface_tension=0.072
        ),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(3167.0),
        species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
    )
    return Aerosol(
        particles=particle, gas=gas, environment=environment
    ), process


@require_cuda
@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_resident_sulfate_refresh(method):
    """GPU uptake changes sulfate dissociation and matches CPU water gain."""
    from particula.execution.gpu_resident_condensation import (
        GPUResidentCondensation,
    )

    initial, strategy = electrolyte_case()
    evaluator = strategy.activity_strategy.prepared
    before = evaluator.evaluate(initial.particles.masses[0], 298.15)
    owner = GPUResidentCondensation(
        initial,
        strategy,
        device="cuda:0",
        method=method,
        particle_atol=1e-29,
        gas_atol=1e-15,
        rtol=1e-7,
    )
    owner.step(1e-4)
    state = owner.snapshot()
    after = evaluator.evaluate(state.particles.masses[0], 298.15)
    assert (
        after.ion_molalities[0, after.ion_ids.index(261)]
        < before.ion_molalities[0, before.ion_ids.index(261)]
    )
    np.testing.assert_array_equal(
        state.particles.masses[:, :, 1], initial.particles.masses[:, :, 1]
    )
    reference, cpu = electrolyte_case()
    for _ in range(100):
        cpu.step(
            reference.particles, reference.gas, 298.15, 101325.0, time_step=1e-6
        )
    np.testing.assert_allclose(
        state.particles.masses[:, :, 0],
        reference.particles.masses[:, :, 0],
        rtol=1e-5,
        atol=0,
    )


@require_cuda
@pytest.mark.parametrize("driver", ["fixed", "host", "resident"])
def test_trace_failure_preserves_accepted_state(driver):
    """A failing second occupied slot cannot advance the first slot or gas."""
    from particula.dynamics import MassCondensation
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.gpu_resident_condensation import (
        GPUResidentCondensation,
    )

    initial, strategy = electrolyte_case(trace=True)
    original = initial.particles.masses.copy(), initial.gas.concentration.copy()
    owner = None
    with pytest.raises(
        (RuntimeError, ValueError),
        match=r"box 0, slot 1, .*stage .*backend status",
    ):
        if driver == "fixed":
            owner = MassCondensation(strategy).prepare(
                initial, device="cuda:0", time_step=1e-6
            )
            owner.run(1)
            owner.synchronize()
        else:
            cls = {
                "host": GPUCondensationODE,
                "resident": GPUResidentCondensation,
            }[driver]
            owner = cls(initial, strategy, device="cuda:0", max_trials=2)
            owner.step(1e-6)
    np.testing.assert_array_equal(initial.particles.masses, original[0])
    np.testing.assert_array_equal(initial.gas.concentration, original[1])
    if owner is not None and driver != "fixed":
        assert owner.time == 0.0
        snapshot = owner.snapshot()
        np.testing.assert_array_equal(snapshot.particles.masses, original[0])
        np.testing.assert_array_equal(snapshot.gas.concentration, original[1])
    if owner is not None and driver == "fixed":
        # Faulted fixed-step owners reject snapshots. Verify accepted device
        # storage directly to distinguish rollback from unmodified host inputs.
        prepared = owner._bindings["condensation"].prepared
        np.testing.assert_array_equal(
            prepared.particles.masses.numpy(), original[0]
        )
        np.testing.assert_array_equal(
            prepared.gas.concentration.numpy(), original[1]
        )
        owner.close()


@require_cuda
def test_new_organic_seed_changes_first_resident_flux():
    """Inject a vanillic acid seed into a vacant slot between resident segments."""
    from particula.dynamics import Nucleation
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.dynamics.nucleation.nucleation_configuration import (
        NucleationSourceConfig,
    )
    from particula.dynamics.nucleation.nucleation_strategies import (
        ActivationNucleationStrategy,
        ClosedInterval,
        FormationMetadata,
        InjectionComposition,
        NucleationValidityDomain,
    )
    from particula.dynamics.nucleation.species_mapping import (
        NucleationSpeciesMap,
    )
    from particula.dynamics.particle_process import NucleationCommitConfig
    from particula.execution.gpu_coupled import (
        GPUCondensationControls,
        GPUCoupledSimulation,
    )
    from particula.particles.exhaustion import ExhaustionControls
    from particula.util.constants import AVOGADRO_NUMBER

    case = box.make_box(permuted=True)
    aerosol, process = case.aerosol, case.process
    particle, gas = aerosol.particles, aerosol.gas
    particle.volume[0] = 1e-10
    particle.concentration[0] = [1, 1, 0]
    names = ["butanediol", "vanillic_acid", "water"]
    gas.name = names
    gas.partitioning[1] = True
    gas.molar_mass[1] = 0.16815
    process.molar_mass[1] = gas.molar_mass[1]
    process.species_map = CondensationSpeciesMap(
        tuple(names), tuple(zip(case.sources, case.targets, strict=True))
    )

    # The pulse provides exactly one seed's inventory. The first segment sees
    # no precursor and retains the vacant slot. Both rates there start at zero.
    molecules = 100_000_000
    interval = 1e-8
    seed_mass = molecules * gas.molar_mass[1] / AVOGADRO_NUMBER
    source = ActivationNucleationStrategy(
        2 / (molecules * interval),
        NucleationValidityDomain(
            ClosedInterval(0, 1e30), ClosedInterval(200, 400)
        ),
        InjectionComposition((0, molecules, 0)),
        FormationMetadata(1e-9),
    )
    event = Nucleation(
        NucleationSourceConfig(
            source, 1, NucleationSpeciesMap(tuple(names), ((1, 4),))
        ),
        NucleationCommitConfig(
            1.0,
            0.0,
            ExhaustionControls(False, False),
            np.ones(1),
            np.array([0.001]),
            np.array([1e-30]),
        ),
    )
    controls = GPUCondensationControls(
        method="rk23",
        first_step=interval,
        max_step=interval,
        rtol=1e-7,
        particle_atol=1e-30,
        gas_atol=1e-16,
    )
    with GPUCoupledSimulation(
        aerosol,
        (event,),
        process,
        device="cuda:0",
        time_step=interval,
        controls=controls,
        resident=True,
    ) as owner:
        owner.capture_events()
        owner.step()
        before = owner.snapshot()
        assert before.particles.concentration[0, -1] == 0
        owner.update_gas_species(
            concentrations={"vanillic_acid": seed_mass / particle.volume[0]}
        )
        owner.step()
        observed = owner.snapshot()
    assert observed.particles.concentration[0, -1] == 1
    np.testing.assert_allclose(
        observed.particles.masses[0, -1, 4], seed_mass, rtol=1e-13, atol=0
    )
    before.particles.concentration[0, -1] = 1
    before.particles.masses[0, -1, 4] = seed_mass
    expected = box.sample(case, before)["flux"][-1]
    measured = observed.particles.masses[0, -1, case.targets] / interval
    assert np.all(expected > 0)
    np.testing.assert_allclose(measured, expected, rtol=1e-3, atol=0)
