"""Prepared ownership, transfers, stream selection and failure boundaries."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from typing import no_type_check

import numpy as np
import pytest

from particula._cpu_batch import arrays
from particula.dynamics.tests.native_runnable_test import state
from particula.execution import Backend, Device
from particula.execution.prepared import PreparedSimulation

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require actual CUDA for the CUDA parameter."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return Device(Backend.WARP, request.param)


def upload(aerosol, device):
    """Exercise ownership before process translation is connected."""
    return PreparedSimulation._upload(
        aerosol,
        device=device,
        time_step=0.125,
        root_seed=92,
        logical_box_ids=("chamber",),
    )


@no_type_check
@wp.kernel
def _consume_gas(values: wp.array2d(dtype=wp.float64)):
    box, species = wp.tid()
    values[box, species] *= wp.float64(0.5)


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_snapshot_detaches_every_field_and_preserves_semantics(device, kind):
    """Names, independent species widths and raw counts survive both ways."""
    initial = state(
        volume=0.25 if kind == "particle_resolved" else 1.0, kind=kind
    )
    baseline = {name: value.copy() for name, value in arrays(initial).items()}
    with upload(initial, device) as simulation:
        first = simulation.snapshot()
        second = simulation.snapshot()
        assert first.gas.name == ["inert", "B", "A"]
        assert first.particles.distribution_type == kind
        assert first.particles.n_species == 2
        assert first.gas.concentration.shape == (1, 3)
        for name, value in arrays(first).items():
            np.testing.assert_array_equal(value, baseline[name])
            assert not np.shares_memory(value, arrays(initial)[name])
            assert not np.shares_memory(value, arrays(second)[name])
            value[...] = 0
        first.gas.name[0] = "changed"
        initial.gas.concentration[:] = 1
        initial.particles.masses[:] = 1
        third = simulation.snapshot()
        for name, value in arrays(third).items():
            np.testing.assert_array_equal(value, baseline[name])
        assert third.gas.name == ["inert", "B", "A"]
        assert simulation.device == device
        assert simulation._session.metadata.stream.logical_box_ids == (
            "chamber",
        )
    assert simulation.closed
    assert (
        simulation._session is simulation._registry is simulation._guard is None
    )
    simulation.close()
    for observe in (
        simulation.snapshot,
        simulation.synchronize,
        simulation.step,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            observe()


def test_enqueue_uses_owned_stream_and_no_host_observation(device, monkeypatch):
    """Prepared stepping retains buffers and does no synchronization/readback."""
    initial = state()
    with upload(initial, device) as simulation:
        gas = simulation._session.gas.concentration

        def enqueue():
            if device.native != "cpu":
                assert wp.get_stream(device.native) is simulation._stream
            wp.launch(
                _consume_gas, dim=gas.shape, inputs=[gas], device=device.native
            )

        simulation._operations = (enqueue,)
        # Compile/load before applying the no-allocation observation guard.
        wp.load_module(device=device.native)
        saved_stream = (
            wp.get_stream(device.native) if device.native != "cpu" else None
        )

        def forbidden(*_args, **_kwargs):
            pytest.fail("a prepared step performed host work or allocation")

        with monkeypatch.context() as patch:
            for name in (
                "synchronize",
                "synchronize_device",
                "synchronize_stream",
                "zeros",
                "empty",
            ):
                patch.setattr(wp, name, forbidden)
            patch.setattr(wp.array, "numpy", forbidden)
            simulation.run(3)
        if saved_stream is not None:
            assert wp.get_stream(device.native) is saved_stream
        assert simulation.completed_steps == 3
        assert simulation.simulated_time == 0.375
        np.testing.assert_array_equal(
            simulation.snapshot().gas.concentration,
            initial.gas.concentration / 8,
        )
    assert simulation.completed_steps == 3
    assert simulation.simulated_time == 0.375


def test_writer_failure_faults_and_close_never_downloads(device, monkeypatch):
    """A failed launch sequence cannot be reused or inspected as successful."""
    simulation = upload(state(), device)
    gas = simulation._session.gas.concentration

    def failing_operation():
        wp.launch(
            _consume_gas, dim=gas.shape, inputs=[gas], device=device.native
        )
        raise RuntimeError("writer failed")

    simulation._operations = (failing_operation,)
    with pytest.raises(RuntimeError, match="writer failed"):
        simulation.step()
    assert simulation.completed_steps == 0
    with pytest.raises(RuntimeError, match="faulted"):
        simulation.snapshot()
    monkeypatch.setattr(
        wp.array, "numpy", lambda *_: pytest.fail("close downloaded")
    )
    simulation.close()
    assert simulation.closed


@pytest.mark.parametrize("steps", [-1, 0.5, True])
def test_invalid_run_control_does_not_enqueue(device, steps):
    """Invalid work counts cannot advance simulation time."""
    with upload(state(), device) as simulation:
        simulation._operations = (lambda: pytest.fail("invalid run enqueued"),)
        with pytest.raises((TypeError, ValueError)):
            simulation.run(steps)
        assert simulation.completed_steps == 0


def test_unsupported_population_rejects_before_device_lookup(monkeypatch):
    """PDF state cannot allocate or silently fall back to a different backend."""
    monkeypatch.setattr(
        wp, "get_device", lambda *_: pytest.fail("device lookup")
    )
    with pytest.raises(ValueError, match="continuous_pdf"):
        upload(
            state(volume=1.0, kind="continuous_pdf"),
            Device(Backend.WARP, "cuda:0"),
        )


def test_async_failure_faults_on_observation(device, monkeypatch):
    """A deferred runtime error faults a binding even after enqueue returns."""
    simulation = upload(state(), device)
    simulation._operations = ()
    simulation.run(2)

    def failure():
        raise RuntimeError("asynchronous failure")

    with monkeypatch.context() as patch:
        patch.setattr(simulation, "_wait", failure)
        with pytest.raises(RuntimeError, match="asynchronous failure"):
            simulation.synchronize()
    with pytest.raises(RuntimeError, match="faulted"):
        simulation.run(1)
    simulation.close()


def test_context_exception_closes_owner(device):
    """Caller exceptions release resources without requiring a snapshot."""
    with pytest.raises(ValueError, match="analysis failed"):
        with upload(state(), device) as simulation:
            raise ValueError("analysis failed")
    assert simulation.closed


def test_translated_dilution_matches_independent_decay_and_freezes_config(
    device,
):
    """All gas lanes and raw counts decay while metadata and masses persist."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.execution.prepared import prepare_simulation

    initial = state()
    strategy = DilutionStrategy(0.2)
    with prepare_simulation(
        (Dilution(strategy),), initial, device=device.native, time_step=0.5
    ) as simulation:
        strategy.coefficient = 5
        simulation.run(4)
        result = simulation.snapshot()
    factor = np.exp(-0.4)
    np.testing.assert_allclose(
        result.particles.concentration,
        initial.particles.concentration * factor,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        result.gas.concentration, initial.gas.concentration * factor, rtol=1e-14
    )
    np.testing.assert_array_equal(
        result.particles.masses, initial.particles.masses
    )


def test_translated_brownian_has_collisions_and_conserves_inventory(device):
    """Equal-count collision events conserve species mass and signed charge."""
    from particula.dynamics import BrownianCoagulationStrategy, Coagulation
    from particula.execution.prepared import prepare_simulation

    initial = state(volume=1e-20)
    with prepare_simulation(
        (Coagulation(BrownianCoagulationStrategy("particle_resolved")),),
        initial,
        device=device.native,
        time_step=1,
        root_seed=17,
    ) as simulation:
        simulation.step()
        result = simulation.snapshot()
    assert np.count_nonzero(result.particles.concentration) == 1
    for name in ("masses", "charge"):
        before = getattr(initial.particles, name)
        after = getattr(result.particles, name)
        weights = initial.particles.concentration
        result_weights = result.particles.concentration
        if before.ndim == 3:
            weights = weights[..., None]
            result_weights = result_weights[..., None]
        np.testing.assert_allclose(
            np.sum(after * result_weights, axis=1),
            np.sum(before * weights, axis=1),
            rtol=1e-14,
        )


@pytest.mark.parametrize("charged", [False, True])
def test_translated_wall_loss_clears_lost_slots(device, charged):
    """The existing neutral/charged kernels retire complete resolved slots."""
    from particula.dynamics import WallLoss
    from particula.dynamics.wall_loss.wall_loss_strategies import (
        ChargedWallLossStrategy,
        SphericalWallLossStrategy,
    )
    from particula.execution.prepared import prepare_simulation

    strategy = (
        ChargedWallLossStrategy(
            1e-3,
            "spherical",
            chamber_radius=0.5,
            distribution_type="particle_resolved",
        )
        if charged
        else SphericalWallLossStrategy(1e-3, 0.5, "particle_resolved")
    )
    initial = state()
    with prepare_simulation(
        (WallLoss(strategy),), initial, device=device.native, time_step=1e30
    ) as simulation:
        simulation.step()
        result = simulation.snapshot()
    for name in ("masses", "charge", "concentration"):
        np.testing.assert_array_equal(getattr(result.particles, name), 0)
    np.testing.assert_array_equal(
        result.gas.concentration, initial.gas.concentration
    )


def test_complete_plan_rejects_before_first_upload(monkeypatch):
    """Unsupported later stages and duplicate families cannot partially upload."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.execution.prepared import prepare_simulation
    from particula.gpu import conversion
    from particula.runnable import RunnableABC

    def forbidden(*_args, **_kwargs):
        pytest.fail("rejected sequence uploaded state")

    monkeypatch.setattr(conversion, "to_warp_particle_data", forbidden)
    process = Dilution(DilutionStrategy(0.2))
    for processes in ((process, RunnableABC()), (process, process)):
        with pytest.raises(ValueError):
            prepare_simulation(processes, state(), time_step=1)


def condensation_case(*, latent=False, kappa=False, weighted=False):
    """One volatile material, one dry material and two uncoupled gas lanes."""
    from particula.dynamics import MassCondensation
    from particula.dynamics.condensation.condensation_strategies import (
        CondensationIsothermal,
        CondensationLatentHeat,
    )
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
        WaterBuckStrategy,
    )
    from particula.particles.activity_strategies import (
        ActivityIdealMolar,
        ActivityKappaParameter,
    )
    from particula.particles.surface_strategies import SurfaceStrategyVolume

    initial = state(volume=1e-12)
    initial.particles.density[:] = 1000
    initial.gas.concentration[:, 2] = 0.03
    molar = np.array([0.018, 0.046])
    activity = (
        ActivityKappaParameter(
            kappa=np.array([0.0, 0.4]),
            density=initial.particles.density.copy(),
            molar_mass=molar,
            water_index=0,
        )
        if kappa
        else ActivityIdealMolar(molar)
    )
    cls = CondensationLatentHeat if latent else CondensationIsothermal
    kwargs = {"latent_heat": 2.4e6} if latent else {}
    strategy = cls(
        molar_mass=initial.gas.molar_mass.copy(),
        diffusion_coefficient=np.array([1.1e-5, 1.3e-5, 1.7e-5]),
        accommodation_coefficient=np.array([0.2, 0.3, 0.4]),
        activity_strategy=activity,
        surface_strategy=SurfaceStrategyVolume(
            density=1000.0,
            surface_tension=np.array([0.06, 0.08]),
            phase_index=np.array([0, 0]) if weighted else None,
        ),
        vapor_pressure_strategy=(
            ConstantVaporPressureStrategy(1.0),
            ConstantVaporPressureStrategy(2.0),
            WaterBuckStrategy(),
        ),
        species_map=CondensationSpeciesMap(tuple(initial.gas.name), ((2, 0),)),
        **kwargs,
    )
    return initial, MassCondensation(strategy)


@pytest.mark.parametrize(
    "latent,kappa,weighted",
    [
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_translated_condensation_conserves_and_preserves_uncoupled_lanes(
    device, latent, kappa, weighted
):
    """Supported live composition models transfer mass with nondefault physics."""
    from particula.execution.prepared import prepare_simulation

    initial, process = condensation_case(
        latent=latent, kappa=kappa, weighted=weighted
    )
    with prepare_simulation(
        (process,), initial, device=device.native, time_step=1e-4
    ) as simulation:
        simulation.run(3)
        result = simulation.snapshot()
    gain = (
        np.sum(
            (result.particles.masses[..., 0] - initial.particles.masses[..., 0])
            * initial.particles.concentration,
            axis=1,
        )
        / initial.particles.volume
    )
    loss = initial.gas.concentration[:, 2] - result.gas.concentration[:, 2]
    assert np.all(gain > 0)
    np.testing.assert_allclose(gain, loss, rtol=1e-10, atol=1e-18)
    np.testing.assert_array_equal(
        result.particles.masses[..., 1], initial.particles.masses[..., 1]
    )
    np.testing.assert_array_equal(
        result.gas.concentration[:, :2], initial.gas.concentration[:, :2]
    )
    np.testing.assert_array_equal(
        result.particles.concentration, initial.particles.concentration
    )


def test_condensation_coefficients_are_frozen_and_affect_transfer(device):
    """Mutating the CPU strategy cannot alter the prepared transport model."""
    from particula.execution.prepared import prepare_simulation

    initial, process = condensation_case()
    with prepare_simulation(
        (process,), initial, device=device.native, time_step=1e-4
    ) as frozen:
        process.condensation_strategy.diffusion_coefficient *= 0.1
        with prepare_simulation(
            (process,), initial, device=device.native, time_step=1e-4
        ) as changed:
            frozen.step()
            changed.step()
            first, second = frozen.snapshot(), changed.snapshot()
    assert np.all(
        first.particles.masses[..., 0] > second.particles.masses[..., 0]
    )


def nucleation_case(*, resampling=False):
    """An exactly integral two-event source with independently sized species."""
    from dataclasses import replace

    from particula.dynamics.tests.native_nucleation_test import (
        fixture,
        runnable,
    )
    from particula.particles.exhaustion import ExhaustionControls
    from particula.util.constants import AVOGADRO_NUMBER

    initial, config = fixture(volume=1.0, capacity=4)
    initial.gas.concentration[0, 2] = 8 * 0.2 / AVOGADRO_NUMBER
    initial.gas.concentration[0, 1] = 32 * 0.1 / AVOGADRO_NUMBER
    initial.particles.concentration[0, 0] = 1
    process = runnable(initial, config)
    process.source_config = replace(
        process.source_config,
        strategy=replace(process.source_config.strategy, coefficient=0.25),
    )
    process.commit_config = replace(
        process.commit_config,
        maximum_slot_weight=1.0,
        source_charge=0.0,
        exhaustion_controls=ExhaustionControls(resampling, False),
        radius_cubed_relative_error=0.01,
    )
    return initial, process


def test_translated_nucleation_injects_mapped_neutral_unit_events(device):
    """Two admitted molecules become two slots without touching inert lanes."""
    from particula.dynamics.tests.native_nucleation_test import inventory
    from particula.execution.prepared import prepare_simulation

    initial, process = nucleation_case()
    with prepare_simulation(
        (process,), initial, device=device.native, time_step=1.0
    ) as simulation:
        simulation.step()
        result = simulation.snapshot()
    np.testing.assert_array_equal(
        result.particles.concentration, [[1, 1, 1, 0]]
    )
    np.testing.assert_array_equal(result.particles.charge, 0)
    np.testing.assert_allclose(
        inventory(result), inventory(initial), rtol=1e-14
    )
    assert result.gas.concentration[0, 2] < initial.gas.concentration[0, 2]
    np.testing.assert_array_equal(
        result.gas.concentration[:, [0, 3]],
        initial.gas.concentration[:, [0, 3]],
    )


def test_nucleation_fractional_demand_surfaces_at_observation(device):
    """Unsupported live fractional event counts cannot appear as success."""
    from particula.execution.prepared import prepare_simulation

    initial, process = nucleation_case()
    with prepare_simulation(
        (process,), initial, device=device.native, time_step=0.1
    ) as simulation:
        simulation.step()
        with pytest.raises(ValueError, match="integral"):
            simulation.snapshot()
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.step()


def test_nucleation_resampling_uses_live_capacity_and_retains_bounds(device):
    """A full cohort releases one slot without losing species or count moments."""
    from particula.dynamics.tests.native_nucleation_test import inventory
    from particula.execution.prepared import prepare_simulation

    initial, process = nucleation_case(resampling=True)
    initial.particles.masses[0, 1:3] = initial.particles.masses[0, 0]
    initial.particles.concentration[0, :3] = 1
    with prepare_simulation(
        (process,), initial, device=device.native, time_step=1.0
    ) as simulation:
        simulation.step()
        result = simulation.snapshot()
    np.testing.assert_array_equal(
        result.particles.concentration, [[1.5, 1.5, 1, 1]]
    )
    np.testing.assert_allclose(
        inventory(result), inventory(initial), rtol=1e-14
    )


@pytest.mark.parametrize(
    "failure",
    [
        "activity",
        "surface_density",
        "zero_vapor",
        "uncoupled",
        "latent_bound",
        "nucleation_weight",
        "nucleation_charge",
        "nucleation_scale",
    ],
)
def test_unsupported_physics_is_rejected_before_upload(monkeypatch, failure):
    """Capability rejection must precede allocation even for valid CPU models."""
    from dataclasses import replace

    from particula.execution.prepared import prepare_simulation
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
    )
    from particula.gpu import conversion
    from particula.particles.activity_strategies import ActivityIdealMass

    initial, process = condensation_case(latent=failure == "latent_bound")
    strategy = process.condensation_strategy
    if failure == "activity":
        strategy.activity_strategy = ActivityIdealMass()
    elif failure == "surface_density":
        strategy.surface_strategy.density = np.array([1000.0, 1200.0])
    elif failure == "zero_vapor":
        strategy.vapor_pressure_strategy = ConstantVaporPressureStrategy(0)
    elif failure == "uncoupled":
        strategy.update_gases = False
    elif failure == "latent_bound":
        from particula.gas.latent_heat_strategies import ConstantLatentHeat

        strategy._latent_heat_strategy = ConstantLatentHeat(1e10)
    else:
        initial, process = nucleation_case()
        changes = {
            "nucleation_weight": {"maximum_slot_weight": 2.0},
            "nucleation_charge": {"source_charge": 1.0},
            "nucleation_scale": {"requested_scale": np.array([2.0])},
        }
        process.commit_config = replace(
            process.commit_config, **changes[failure]
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("unsupported physics reached upload")

    monkeypatch.setattr(conversion, "to_warp_particle_data", forbidden)
    with pytest.raises(ValueError):
        prepare_simulation((process,), initial, time_step=1.0)


def test_source_failure_latches_across_steps_and_stops_later_writers(device):
    """A fractional source cannot be forgotten or followed by gas dilution."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.execution.prepared import prepare_simulation

    initial, process = nucleation_case()
    with prepare_simulation(
        (process, Dilution(DilutionStrategy(1.0))),
        initial,
        device=device.native,
        time_step=0.1,
    ) as simulation:
        simulation.run(3)
        simulation._wait()
        np.testing.assert_array_equal(
            simulation._session.gas.concentration.numpy(),
            initial.gas.concentration,
        )
        np.testing.assert_array_equal(
            simulation._session.particles.masses.numpy(),
            initial.particles.masses,
        )
        with pytest.raises(ValueError, match="integral"):
            simulation.synchronize()


def test_new_unit_particles_guard_unequal_coagulation_before_rng_draw(device):
    """Nucleation may make a valid initial equal-count cohort incompatible."""
    from particula.dynamics import (
        BrownianCoagulationStrategy,
        Coagulation,
        Dilution,
        DilutionStrategy,
    )
    from particula.execution.prepared import prepare_simulation

    initial, process = nucleation_case()
    initial.particles.concentration[0, 0] = 2
    with prepare_simulation(
        (
            process,
            Coagulation(BrownianCoagulationStrategy("particle_resolved")),
            Dilution(DilutionStrategy(1.0)),
        ),
        initial,
        device=device.native,
        time_step=1.0,
    ) as simulation:
        resources = simulation._registry.acquire_coagulation(4)
        before_rng = resources.rng_states.numpy().copy()
        simulation.run(3)
        simulation._wait()
        np.testing.assert_array_equal(
            simulation._session.particles.concentration.numpy(), [[2, 1, 1, 0]]
        )
        np.testing.assert_array_equal(resources.rng_states.numpy(), before_rng)
        with pytest.raises(ValueError, match="equal active counts"):
            simulation.synchronize()


@pytest.mark.parametrize("kind", ["particle_resolved", "discrete"])
def test_public_prepare_forcing_and_reduced_diagnostics(device, kind):
    """Box-varying forcing and density reductions preserve the population basis."""
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.dynamics.tests.multibox_runnable_test import batch
    from particula.execution import PreparedSimulation, SimulationDiagnostics

    initial = batch(state(volume=1.0, kind=kind))
    if kind == "discrete":
        initial.particles.volume[:] = 1.0
    process = Dilution(DilutionStrategy(0.2))
    with process.prepare(
        initial, device=device.native, time_step=0.5
    ) as simulation:
        assert isinstance(simulation, PreparedSimulation)
        before = simulation.diagnostics()
        assert isinstance(before, SimulationDiagnostics)
        expected_count = initial.particles.concentration.sum(axis=1)
        expected_mass = np.sum(
            initial.particles.masses
            * initial.particles.concentration[..., None],
            axis=1,
        )
        if kind == "particle_resolved":
            expected_count /= initial.particles.volume
            expected_mass /= initial.particles.volume[:, None]
        np.testing.assert_allclose(before.number_concentration, expected_count)
        np.testing.assert_allclose(
            before.particle_mass_concentration, expected_mass
        )
        np.testing.assert_array_equal(before.active_slots, [2, 2])
        gas = np.array([0.01, 0.02, 0.03])
        simulation.update_gas(gas)
        gas[:] = 0
        simulation.update_environment(temperature=[290, 310], pressure=90000)
        simulation.run(2)
        after = simulation.diagnostics()
        np.testing.assert_allclose(
            after.gas_concentration,
            np.tile([0.01, 0.02, 0.03], (2, 1)) * np.exp(-0.2),
        )
        np.testing.assert_array_equal(after.temperature, [290, 310])
        np.testing.assert_array_equal(after.pressure, [90000, 90000])
        np.testing.assert_allclose(
            after.number_concentration, expected_count * np.exp(-0.2)
        )
        np.testing.assert_array_equal(
            before.temperature, initial.environment.temperature
        )
        with pytest.raises(ValueError, match="nonphysical"):
            simulation.update_environment(
                temperature=[300, 301], pressure=np.nan
            )
        np.testing.assert_array_equal(
            simulation.diagnostics().temperature, [290, 310]
        )
        after.number_concentration[:] = -1
        assert np.all(simulation.diagnostics().number_concentration >= 0)


def test_condensation_forcing_refreshes_saturation_and_last_step_energy(device):
    """Warm forcing changes Buck pressure immediately; latent output is extensive."""
    from particula.util.constants import GAS_CONSTANT

    initial, process = condensation_case(latent=True)
    with process.prepare(
        initial, device=device.native, time_step=1e-4
    ) as simulation:
        first = simulation.diagnostics()
        simulation.update_environment(temperature=310)
        warm = simulation.diagnostics()
        assert warm.saturation_ratio[0, 2] < first.saturation_ratio[0, 2]
        gas = initial.gas.concentration.copy()
        gas[:, 2] = 0.08
        simulation.update_gas(gas)
        before = simulation.diagnostics()
        simulation.step()
        after = simulation.diagnostics()
        vapor = simulation._session.gas.vapor_pressure.numpy()
        np.testing.assert_allclose(
            after.saturation_ratio,
            after.gas_concentration
            * GAS_CONSTANT
            * after.temperature[:, None]
            / (initial.gas.molar_mass[None, :] * vapor),
            rtol=1e-14,
        )
        expected_energy = (
            (before.gas_concentration - after.gas_concentration)
            * after.volume[:, None]
            * 2.4e6
        )
        np.testing.assert_allclose(
            after.latent_heat_energy, expected_energy, rtol=1e-9, atol=1e-24
        )
        assert after.latent_heat_energy[0, 2] > 0
        with pytest.raises(ValueError, match="owns derived"):
            simulation.update_environment(saturation_ratio=1)


def test_real_prepared_sequence_has_no_stepping_allocations_or_readbacks(
    device, monkeypatch
):
    """Compiled condensation/dilution loops only enqueue retained device work."""
    from particula.dynamics import Dilution, DilutionStrategy

    initial, process = condensation_case()
    sequence = process | Dilution(DilutionStrategy(0.01))
    with sequence.prepare(
        initial, device=device.native, time_step=1e-4
    ) as simulation:
        simulation.step()
        simulation.synchronize()

        def forbidden(*_args, **_kwargs):
            pytest.fail("resident sequence performed host work or allocation")

        with monkeypatch.context() as patch:
            for name in (
                "zeros",
                "empty",
                "synchronize",
                "synchronize_device",
                "synchronize_stream",
            ):
                patch.setattr(wp, name, forbidden)
            patch.setattr(wp.array, "numpy", forbidden)
            simulation.run(3)
        assert simulation.completed_steps == 4
        assert np.all(np.isfinite(simulation.diagnostics().gas_concentration))


@pytest.mark.parametrize(
    "family", ["condensation", "coagulation", "wall_loss", "dilution"]
)
def test_latched_failure_preserves_primary_arrays_and_random_streams(
    device, family
):
    """Every selectable writer respects the same persistent sequence gate."""
    from particula.dynamics import (
        BrownianCoagulationStrategy,
        Coagulation,
        Dilution,
        DilutionStrategy,
        WallLoss,
    )
    from particula.dynamics.wall_loss.wall_loss_strategies import (
        ChargedWallLossStrategy,
    )

    initial = state()
    if family == "condensation":
        initial, process = condensation_case()
    elif family == "coagulation":
        process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    elif family == "wall_loss":
        process = WallLoss(
            ChargedWallLossStrategy(
                1e-3,
                "rectangular",
                chamber_dimensions=(1.0, 2.0, 3.0),
                wall_electric_field=np.array([1.0, 2.0, 3.0]),
                distribution_type="particle_resolved",
            )
        )
    else:
        process = Dilution(DilutionStrategy(0.2))
    with process.prepare(
        initial, device=device.native, time_step=1
    ) as simulation:
        simulation.synchronize()
        resident = simulation._session
        targets = [
            getattr(resident.particles, name)
            for name in (
                "masses",
                "concentration",
                "charge",
                "density",
                "volume",
            )
        ]
        targets += [
            getattr(resident.gas, name)
            for name in (
                "concentration",
                "molar_mass",
                "partitioning",
                "vapor_pressure",
            )
        ]
        targets += [
            resident.environment.temperature,
            resident.environment.pressure,
            resident.environment.saturation_ratio,
        ]
        if family == "wall_loss":
            targets.append(simulation._registry.acquire_wall_loss().rng_states)
        elif family == "coagulation":
            targets.append(
                simulation._registry.acquire_coagulation(2).rng_states
            )
        before = [array.numpy().copy() for array in targets]
        with simulation._scope():
            simulation._device_failure.assign(np.array([1], dtype=np.int32))
        simulation.run(3)
        simulation._wait()
        for array, original in zip(targets, before, strict=True):
            np.testing.assert_array_equal(array.numpy(), original)
        with pytest.raises(ValueError, match="dependent work stopped"):
            simulation.synchronize()


def test_nucleation_resampling_tolerance_failure_stops_later_process(device):
    """User error bounds remain active and a failed remap cannot dilute gas."""
    from dataclasses import replace

    from particula.dynamics import Dilution, DilutionStrategy

    initial, process = nucleation_case(resampling=True)
    initial.particles.masses[0, 1] = initial.particles.masses[0, 0] * 1000
    initial.particles.masses[0, 2] = initial.particles.masses[0, 0] * 1e6
    initial.particles.concentration[0, :3] = 1
    process.commit_config = replace(
        process.commit_config, mean_radius_relative_error=0.0
    )
    with (process | Dilution(DilutionStrategy(1.0))).prepare(
        initial, device=device.native, time_step=1
    ) as simulation:
        simulation.run(2)
        simulation._wait()
        np.testing.assert_array_equal(
            simulation._session.gas.concentration.numpy(),
            initial.gas.concentration,
        )
        np.testing.assert_array_equal(
            simulation._session.particles.masses.numpy(),
            initial.particles.masses,
        )
        with pytest.raises(ValueError, match="diagnostic exceeds bound"):
            simulation.synchronize()


def test_prepared_stochastic_streams_follow_logical_boxes_and_run_segments(
    device,
):
    """Scheduling chunks and physical box order cannot reseed scientific paths."""
    from particula._cpu_batch import detached
    from particula.dynamics import BrownianCoagulationStrategy, Coagulation
    from particula.dynamics.tests.multibox_runnable_test import batch
    from particula.particles import ParticleData

    initial = batch(state())
    old = initial.particles
    initial.particles = ParticleData(
        masses=np.tile(old.masses, (1, 32, 1)),
        concentration=np.ones((2, 64)),
        charge=np.tile(old.charge, (1, 32)),
        density=old.density.copy(),
        volume=np.array([1e-13, 2e-13]),
        distribution_type="particle_resolved",
    )
    swapped = detached(initial)
    for carrier, names in (
        (swapped.particles, ("masses", "concentration", "charge", "volume")),
        (swapped.gas, ("concentration",)),
        (swapped.environment, ("temperature", "pressure", "saturation_ratio")),
    ):
        for name in names:
            values = getattr(carrier, name)
            values[:] = values[::-1].copy()
    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    results = []
    for aerosol, identities, segments in (
        (initial, ("cell A", "cell B"), (4,)),
        (initial, ("cell A", "cell B"), (2, 2)),
        (swapped, ("cell B", "cell A"), (1, 3)),
    ):
        with process.prepare(
            aerosol,
            device=device.native,
            time_step=0.1,
            root_seed=123,
            logical_box_ids=identities,
        ) as simulation:
            for steps in segments:
                simulation.run(steps)
            results.append(simulation.snapshot())
    assert np.count_nonzero(results[0].particles.concentration) < 128
    for name in ("masses", "concentration", "charge", "volume"):
        first = getattr(results[0].particles, name)
        np.testing.assert_array_equal(
            first, getattr(results[1].particles, name)
        )
        np.testing.assert_array_equal(
            first, getattr(results[2].particles, name)[::-1]
        )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("latent,kappa", [(False, False), (True, True)])
def test_prepared_weighted_surface_matches_each_cpu_particle(
    device, weighted, latent, kappa
):
    """Mapped vapor transport and composition agree across CPU and CUDA."""
    from particula.aerosol import Aerosol

    initial, process = condensation_case(
        weighted=weighted, latent=latent, kappa=kappa
    )
    initial.particles.masses[0, :2] = [[9e-19, 1e-19], [1e-19, 9e-19]]
    cpu = Aerosol(
        particles=initial.particles.copy(),
        gas=initial.gas.copy(),
        environment=initial.environment.copy(),
    )
    process.execute(cpu, time_step=1e-4, sub_steps=4)
    with process.prepare(
        initial, device=device.native, time_step=1e-4
    ) as simulation:
        simulation.step()
        actual = simulation.snapshot()
    np.testing.assert_allclose(
        actual.particles.masses, cpu.particles.masses, rtol=2e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        actual.gas.concentration, cpu.gas.concentration, rtol=2e-12, atol=1e-30
    )


def test_native_condensation_minimum_radius_matches_cpu(device):
    """Native transport and Kelvin evaluation share the CPU radius floor."""
    from particula._cpu_batch import detached

    initial, process = condensation_case()
    total_mass = 4 * np.pi / 3 * 1000 * (5e-11) ** 3
    initial.particles.masses[:] = total_mass * np.array([0.5, 0.5])
    initial.gas.concentration[:, 2] = 0
    cpu = detached(initial)
    process.execute(cpu, time_step=1e-16, sub_steps=4)
    with process.prepare(
        initial, device=device.native, time_step=1e-16
    ) as owner:
        owner.step()
        result = owner.snapshot()
    assert np.any(cpu.particles.masses != initial.particles.masses)
    np.testing.assert_allclose(
        result.particles.masses, cpu.particles.masses, rtol=2e-12, atol=1e-45
    )
    np.testing.assert_allclose(
        result.gas.concentration[:, 2],
        cpu.gas.concentration[:, 2],
        rtol=2e-12,
        atol=1e-45,
    )


def test_zero_surface_tension_rejects_before_gpu_upload(monkeypatch):
    """The prepared model honors the native CPU Kelvin positive-tension domain."""
    from particula.gpu import conversion

    initial, process = condensation_case()
    process.condensation_strategy.surface_strategy.surface_tension[:] = 0

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid surface reached upload")

    monkeypatch.setattr(conversion, "to_warp_particle_data", forbidden)
    with pytest.raises(ValueError, match="surface_tension"):
        process.prepare(initial, device="cpu", time_step=1)


@pytest.mark.parametrize(
    "counter, expected", [("completed_steps", 3), ("simulated_time", 0.375)]
)
def test_counter_read_serializes_with_close(device, counter, expected):
    """A blocked status reader sees cached final counters after teardown."""
    simulation = upload(state(), device)
    simulation._operations = ()
    simulation.run(3)
    started = Event()

    def read_counter():
        started.set()
        return getattr(simulation, counter)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with simulation._lock:
                future = executor.submit(read_counter)
                assert started.wait(timeout=5)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.1)
                simulation.close()
            assert future.result(timeout=5) == expected
    finally:
        simulation.close()
