"""Native complete prepared loops with explicit mappings and restart ownership."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.execution import (
    Backend,
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
    Device,
)
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    ResidentBrownianCoagulationExecutionState,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationState,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    WarpCondensationExecutionState,
    WarpCondensationState,
)
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.gpu_resources import (
    CaptureResourceRequirements,
    GPUResourceRegistry,
    PreparedResourceViews,
    ResourceInventoryCapacities,
)
from particula.execution.gpu_session import (
    ResidentStepGuard,
    setup_resident_session,
)
from particula.execution.graph_capture import (
    GraphCaptureAvailability,
    GraphCaptureCapability,
    ResidentGraphCaptureBinding,
    _attach_resident_graph_capture_binding,
    create_graph_capture_lifecycle,
    create_resident_graph_capture_signature,
)
from particula.execution.process_adapters import (
    ResidentDilutionRequest,
    ResidentNucleationRequest,
    ResidentWallLossRequest,
)
from particula.execution.resident_communication import (
    ResidentCommunicationRequest,
)
from particula.execution.resident_scheduler import (
    ResidentSimulationRequest,
    enqueue_prepared_resident_simulation,
    prepare_resident_simulation,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
)
from particula.execution.tests.multi_box_loop_test import _resident_graph
from particula.execution.tests.native_resident_test import diagnostics_plan
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.nucleation import (
    NucleationConfig,
    NucleationExhaustionControls,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.gpu.nucleation_species_mapping import (
    prepare_nucleation_species_map,
)
from particula.gpu.species_mapping import prepare_condensation_species_map
from particula.particles import ParticleData
from particula.util.constants import AVOGADRO_NUMBER

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Use real native CUDA when available alongside Warp CPU."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture(autouse=True)
def forbid_facades(monkeypatch):
    """Native prepared setup and continuation never construct a legacy facade."""
    from particula._legacy_aerosol import Aerosol as LegacyAerosol
    from particula.gas.species import GasSpecies
    from particula.particles.representation import ParticleRepresentation

    def forbidden(*args, **kwargs):
        pytest.fail("native loop constructed a legacy facade")

    for carrier in (LegacyAerosol, GasSpecies, ParticleRepresentation):
        monkeypatch.setattr(carrier, "__init__", forbidden)


def build_loop(device, source=False, restored=None):
    """Prepare all real adapters for a tagged native population, without mocks."""
    if restored is None:
        masses = np.zeros((1, 8, 2))
        masses[0, :2] = [[1e-18, 2e-18], [2e-18, 1e-18]]
        counts = np.zeros((1, 8))
        counts[0, :2] = 1
        cpu_p = ParticleData(
            masses,
            counts,
            np.zeros_like(counts),
            np.array([1000.0, 1200.0]),
            np.array([0.25]),
            distribution_type="particle_resolved",
        )
        cpu_g = GasData(
            ["inert", "B", "A"],
            np.full(3, AVOGADRO_NUMBER)
            if source
            else np.array([0.028, 0.1, 0.018]),
            np.array([[7.0, 100.0, 100.0]])
            if source
            else np.array([[7e-9, 1e-8, 2e-8]]),
            np.array([False, True, True]),
        )
        cpu_e = EnvironmentData(
            np.array([300.0]), np.array([101325.0]), np.zeros((1, 3))
        )
        session = setup_resident_session(
            cpu_p, cpu_g, cpu_e, Device(Backend.WARP, device), root_seed=734
        )
        registry = GPUResourceRegistry(session)
        guard = ResidentStepGuard(session, registry)
    else:
        session, registry, guard = restored
    duration = 1.0
    graph, schedule, by_id = _resident_graph()
    configuration = CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            CommunicationTransportMode.GAS,
            0,
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.float64, device=device),
        ),
        PrescribedVolumeUpdate(None),
        (),
    )
    communication = (
        registry.acquire_communication(configuration)
        if restored is None
        else registry._views["communication_gas"]
    )
    condensation_resources = registry.acquire_condensation()
    coagulation_resources = registry.acquire_coagulation(8)
    wall_loss_resources = registry.acquire_wall_loss()
    nucleation_resources = registry.acquire_nucleation()
    for name in ("requested_scale", "minimum_scale", "minimum_volume"):
        getattr(nucleation_resources.exhaustion, name).assign(np.ones(1))
    thermodynamics = ThermodynamicsConfig(
        wp.zeros(3, dtype=wp.int32, device=device),
        wp.array(
            np.tile([1e-12, 0.0, 0.0, 0.0], (3, 1)),
            dtype=wp.float64,
            device=device,
        ),
        wp.array(
            session.gas.molar_mass.numpy().copy(),
            dtype=wp.float64,
            device=device,
        ),
    )
    mapping = CondensationSpeciesMap(
        session.metadata.gas_names, ((2, 0), (1, 1))
    )
    binding = prepare_condensation_species_map(
        session.particles,
        session.gas,
        gas_names=session.metadata.gas_names,
        species_map=mapping,
        skip_partitioning_indices=(1, 2) if source else (),
    )
    condensation = WarpCondensationExecutionState(
        WarpCondensationState(
            CondensationExecutionConfig(
                CondensationConfiguration(
                    CondensationExecutionMode.EQUAL_STEP,
                    False,
                    CondensationActivityMode.IDEAL,
                    CondensationSurfaceMode.STATIC,
                )
            ),
            session.particles,
            session.gas,
            session.environment,
            thermodynamics,
            scratch_buffers=condensation_resources.scratch_buffers,
            species_binding=binding,
        ),
        duration,
    )
    coagulation = ResidentBrownianCoagulationExecutionState(
        WarpBrownianCoagulationExecutionState(
            WarpBrownianCoagulationState(
                BrownianCoagulationConfig(),
                session.particles,
                None,
                None,
                duration,
                collision_pairs=coagulation_resources.collision_pairs,
                n_collisions=coagulation_resources.n_collisions,
                rng_states=coagulation_resources.rng_states,
                initialize_rng=False,
                environment=session.environment,
            )
        ),
        session,
        registry,
        coagulation_resources,
    )
    original_plan, outputs, _ = diagnostics_plan(session)
    diagnostics = replace(
        original_plan,
        registry=registry,
        graph=graph,
        schedule=schedule,
        node=by_id["diagnostics"],
    )
    registrations = list(diagnostics.registrations)
    # Keep separate accounting storage: source/sink may evolve independently.
    registrations[-1] = replace(
        registrations[-1],
        source_ledger=wp.zeros((1, 3), dtype=wp.float64, device=device),
        sink_ledger=wp.zeros((1, 3), dtype=wp.float64, device=device),
    )
    diagnostics = replace(diagnostics, registrations=tuple(registrations))
    inventory = registry.register_capture_resources(
        session, communication, diagnostics.registrations
    )
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(8, 0, 0),
        inventory,
        PreparedResourceViews(
            condensation_resources,
            coagulation_resources,
            None,
            wall_loss_resources,
            nucleation_resources,
        ),
        communication,
        ("condensation", "coagulation", "wall_loss", "nucleation", "dilution"),
        condensation_resources.scratch_buffers,
        coagulation_resources,
        wall_loss_resources,
        nucleation_resources,
    )
    published = registry.prepare_capture_resources(requirements)
    alpha = 0.0 if source else 0.1
    published.dilution.normalized_coefficient.assign(np.array([alpha]))
    source_binding = prepare_nucleation_species_map(
        session.particles,
        session.gas,
        gas_names=session.metadata.gas_names,
        species_map=NucleationSpeciesMap(
            session.metadata.gas_names, ((2, 0), (1, 1))
        ),
    )
    config = NucleationConfig(
        rate_law="activation",
        coefficient=0.04 if source else 0.0,
        survival_factor=1.0,
        precursor_index=2,
        molecule_counts=(0, 2, 1),
        formation_diameter=1e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1e40,
        temperature_lower=100.0,
        temperature_upper=500.0,
    )
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    prescribed_gas = (
        np.array([[7.0, 100.0, 100.0]])
        if source
        else np.array([[9e-9, 1e-8, 2e-8]])
    )
    request = ResidentSimulationRequest(
        session,
        registry,
        guard,
        graph,
        schedule,
        thermodynamics,
        condensation,
        coagulation,
        ResidentDilutionRequest(
            session,
            registry,
            published.dilution.normalized_coefficient,
            duration,
            published.dilution,
        ),
        ResidentWallLossRequest(
            session,
            registry,
            wall_loss_resources,
            NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
            duration,
            enabled_box_indices=(),
        ),
        ResidentNucleationRequest(
            session,
            registry,
            nucleation_resources,
            config,
            duration,
            NucleationExhaustionControls(False, False),
            source_binding,
        ),
        diagnostics,
        ResidentEnvironmentUpdateRequest(
            session,
            registry,
            graph,
            by_id["environment_update"],
            wp.array([300.0], dtype=wp.float64, device=device),
            wp.array([101325.0], dtype=wp.float64, device=device),
        ),
        ResidentGasUpdateRequest(
            session,
            registry,
            graph,
            by_id["gas_update"],
            wp.array(prescribed_gas, dtype=wp.float64, device=device),
        ),
        ResidentCommunicationRequest(
            session,
            registry,
            graph,
            communication,
            by_id["communication"],
            by_id["volume_evolution"],
            duration,
        ),
        requirements,
    )
    signature = create_resident_graph_capture_signature(request)
    capture = ResidentGraphCaptureBinding(
        request,
        session,
        registry,
        guard,
        create_graph_capture_lifecycle(
            GraphCaptureCapability(
                session.metadata.device, GraphCaptureAvailability.AVAILABLE
            ),
            signature,
        ),
    )
    _attach_resident_graph_capture_binding(request, capture)
    prepared = prepare_resident_simulation(request, duration)
    return SimpleNamespace(
        session=session,
        registry=registry,
        guard=guard,
        request=request,
        prepared=prepared,
        outputs=outputs,
        alpha=alpha,
        prescribed_gas=prescribed_gas,
        source=source,
    )


def inventory(loop):
    """Independent extensive inventory in full gas order."""
    p, g = loop.session.particles, loop.session.gas
    result = g.concentration.numpy().copy() * p.volume.numpy()[:, None]
    mass = np.sum(p.masses.numpy() * p.concentration.numpy()[..., None], axis=1)
    result[:, 2] += mass[:, 0]
    result[:, 1] += mass[:, 1]
    return result


def advance(loop, monkeypatch):
    """Account explicitly for prescribed gas, then enqueue without host work."""
    p, g = loop.session.particles, loop.session.gas
    before = inventory(loop)
    before += (
        loop.prescribed_gas - g.concentration.numpy()
    ) * p.volume.numpy()[:, None]
    registration = loop.request.diagnostics.registrations[-1]
    registration.baseline_total_mass.assign(before)
    factor = np.exp(-loop.alpha)
    registration.sink_ledger.assign(before * (1 - factor))

    def forbidden(*args, **kwargs):
        pytest.fail("prepared native enqueue performed host work")

    with monkeypatch.context() as patch:
        for name in ("zeros", "empty", "synchronize", "synchronize_device"):
            patch.setattr(wp, name, forbidden)
        patch.setattr(type(p.masses), "numpy", forbidden)
        enqueue_prepared_resident_simulation(loop.prepared)
    wp.synchronize()
    np.testing.assert_allclose(
        inventory(loop), before * factor, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        loop.outputs[2].numpy(), inventory(loop), rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        loop.outputs[5].numpy(),
        0.0,
        rtol=0,
        atol=max(1e-30, float(np.max(before)) * 1e-12),
    )


@pytest.mark.parametrize("source", [False, True])
def test_native_complete_prepared_loop_and_restart(device, source, monkeypatch):
    """Real process composition retains counts, mappings, conservation and RNG."""
    from particula.execution.checkpoint import restart_resident_session

    loop = build_loop(device, source)
    restored_loop = None
    try:
        initial_mass = loop.session.particles.masses.numpy().copy()
        for _ in range(2):
            advance(loop, monkeypatch)
        if source:
            assert (
                np.count_nonzero(loop.session.particles.concentration.numpy())
                == 4
            )
        else:
            assert np.any(loop.session.particles.masses.numpy() > initial_mass)
            np.testing.assert_allclose(
                loop.outputs[0].numpy()[0, 0],
                9e-9 * np.exp(-0.1),
                rtol=1e-12,
                atol=0,
            )
        checkpoint = loop.session.checkpoint(loop.registry, loop.guard)
        restored = restart_resident_session(
            checkpoint, loop.session.metadata.device
        )
        restored_loop = build_loop(device, source, restored)
        advance(loop, monkeypatch)
        advance(restored_loop, monkeypatch)
        for field in ("masses", "concentration", "charge", "volume"):
            np.testing.assert_array_equal(
                getattr(loop.session.particles, field).numpy(),
                getattr(restored_loop.session.particles, field).numpy(),
            )
        assert (
            loop.guard.completed_steps
            == restored_loop.guard.completed_steps
            == 3
        )
    finally:
        if restored_loop is not None:
            restored_loop.session.close(
                restored_loop.registry, restored_loop.guard
            )
        loop.session.close(loop.registry, loop.guard)


@pytest.mark.parametrize("process", ["condensation", "nucleation"])
def test_native_prepared_loop_rejects_replaced_map_before_token(
    device, process
):
    """Changed uploaded mapping identities cannot silently alter a ready loop."""
    loop = build_loop(device)
    state = (
        loop.request.condensation.state
        if process == "condensation"
        else loop.request.nucleation
    )
    binding = state.species_binding
    original = binding.gas_to_particle
    before = inventory(loop)
    try:
        object.__setattr__(binding, "gas_to_particle", wp.clone(original))
        with pytest.raises(ValueError):
            enqueue_prepared_resident_simulation(loop.prepared)
        assert loop.guard.completed_steps == 0
        np.testing.assert_array_equal(inventory(loop), before)
    finally:
        object.__setattr__(binding, "gas_to_particle", original)
        loop.session.close(loop.registry, loop.guard)
