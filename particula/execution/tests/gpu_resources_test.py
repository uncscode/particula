"""Tests for concrete-only fixed-shape GPU resource acquisition."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("warp")

import particula.execution.gpu_resources as gpu_resources
from particula.execution import Backend, Device
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.diagnostics import (
    ResidentDiagnosticOperation,
    ResidentDiagnosticRegistration,
    ResidentDiagnosticsExecutor,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_resources import (
    _MAX_SIZE,
    CaptureResourceRequirements,
    DilutionResources,
    GPUResourceRegistry,
    ManifestEntry,
    PreparedResourceViews,
    ResourceInventoryCapacities,
    _item_size,
)
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentMetadata,
    ResidentSession,
)
from particula.execution.process_graph import (
    TimestepPlan,
)
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    SchedulerProfile,
    resolve_timestep_schedule,
)


def _session(
    boxes: int = 1, particle_count: int = 2, species: int = 1
) -> ResidentSession:
    """Build a small valid active Warp CPU resident session lazily."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    particles = WarpParticleData()
    particles.masses = wp.ones(
        (boxes, particle_count, species), dtype=wp.float64, device="cpu"
    )
    particles.concentration = wp.ones(
        (boxes, particle_count), dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros(
        (boxes, particle_count), dtype=wp.float64, device="cpu"
    )
    particles.density = wp.ones(species, dtype=wp.float64, device="cpu")
    particles.volume = wp.ones(boxes, dtype=wp.float64, device="cpu")
    gas = WarpGasData()
    gas.molar_mass = wp.ones(species, dtype=wp.float64, device="cpu")
    gas.concentration = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.vapor_pressure = wp.zeros(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    gas.partitioning = wp.ones((boxes, species), dtype=wp.int32, device="cpu")
    environment = WarpEnvironmentData()
    environment.temperature = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.pressure = wp.ones(boxes, dtype=wp.float64, device="cpu")
    environment.saturation_ratio = wp.ones(
        (boxes, species), dtype=wp.float64, device="cpu"
    )
    return ResidentSession(
        particles,
        gas,
        environment,
        ResidentDimensions(boxes, particle_count, species),
        ResidentMetadata(
            Device(Backend.WARP, "cpu"),
            tuple(str(index) for index in range(species)),
        ),
        ResidentLifecycle.ACTIVE,
    )


def _diagnostics_plan(
    session: ResidentSession,
    registry: GPUResourceRegistry,
    outputs: tuple[Any, ...],
) -> ResidentDiagnosticsPlan:
    """Build the smallest resolver-produced final diagnostics schedule."""
    from particula.execution import CapabilityRequirements, process_graph
    from particula.execution.process_graph import ProcessNode

    node_ids = (
        "environment_update",
        "gas_update",
        "vapor_pressure_refresh",
        "saturation_refresh",
        "diagnostics",
    )

    nodes = tuple(
        ProcessNode(
            schema.node_id,
            schema.kind,
            schema.process,
            CapabilityRequirements(frozenset()),
            schema.resources,
            schema.invalidates,
        )
        for schema in process_graph._NODE_CATALOGUE
        if schema.node_id in node_ids
    )
    plan = TimestepPlan(nodes, ())
    schedule = resolve_timestep_schedule(
        plan,
        EnabledNodeSelection(frozenset(node_ids)),
        SchedulerProfile(
            NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
        ),
    )
    graph = cast(Any, schedule.source_graph)
    wp = pytest.importorskip("warp")
    dimensions = session.dimensions
    device = cast(Any, session.particles).masses.device

    def matrix() -> Any:
        """Allocate one local diagnostics matrix for plan construction."""
        return wp.zeros(
            (dimensions.n_boxes, dimensions.n_species),
            dtype=wp.float64,
            device=device,
        )

    return ResidentDiagnosticsPlan(
        session,
        registry,
        graph,
        schedule,
        next(node for node in graph.nodes if node.node_id == "diagnostics"),
        (
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT,
                outputs[0],
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT,
                outputs[1],
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.TOTAL_SPECIES_MASS,
                matrix(),
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
                wp.zeros(dimensions.n_boxes, dtype=wp.float64, device=device),
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
                matrix(),
                energy_transfer=matrix(),
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
                matrix(),
                baseline_total_mass=matrix(),
                source_ledger=matrix(),
                sink_ledger=matrix(),
            ),
        ),
    )


@pytest.mark.warp
def test_capture_registration_without_communication_retains_exact_report() -> (
    None
):
    """Test absent communication registration returns its exact pinned report."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wp = pytest.importorskip("warp")
    outputs = tuple(
        wp.zeros((1, 1), dtype=wp.float64, device="cpu") for _ in range(2)
    )
    plan = _diagnostics_plan(session, registry, outputs)

    with pytest.raises(ValueError, match="have not been registered"):
        registry.selected_resource_report()
    inventory = registry.register_capture_resources(
        session, None, plan.registrations
    )

    assert registry.selected_resource_report() is inventory
    assert (
        registry.register_capture_resources(session, None, plan.registrations)
        is inventory
    )
    assert inventory.communication_resources is None
    assert inventory.registrations is plan.registrations
    assert inventory.families[0].family == "diagnostics"
    assert [role.shape for role in inventory.families[0].roles][:4] == [
        (1, 1),
        (1, 1),
        (1, 1),
        (1,),
    ]

    with pytest.raises(ValueError, match="already been registered"):
        registry.register_capture_resources(
            session, None, tuple([*plan.registrations])
        )
    assert registry.selected_resource_report() is inventory


@pytest.mark.warp
def test_capture_registration_repeat_bypasses_candidate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test retained repeats use only pinned-session and identity checks."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wp = pytest.importorskip("warp")
    plan = _diagnostics_plan(
        session,
        registry,
        tuple(
            wp.zeros((1, 1), dtype=wp.float64, device="cpu") for _ in range(2)
        ),
    )
    inventory = registry.register_capture_resources(
        session, None, plan.registrations
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        """Fail if a repeat attempts candidate construction or validation."""
        pytest.fail("retained repeat must not validate or build candidates")

    monkeypatch.setattr(registry, "_capture_candidate_roles", fail)
    monkeypatch.setattr(registry, "validate_diagnostic_registrations", fail)

    assert (
        registry.register_capture_resources(session, None, plan.registrations)
        is inventory
    )
    with pytest.raises(ValueError, match="already been registered"):
        registry.register_capture_resources(
            session, None, tuple([*plan.registrations])
        )
    assert registry.selected_resource_report() is inventory


def _communication_with_final_volumes(
    registry: GPUResourceRegistry,
    final_volumes: Any,
    mode: CommunicationTransportMode = CommunicationTransportMode.GAS,
) -> Any:
    """Acquire a minimal closed communication view with final volumes."""
    wp = pytest.importorskip("warp")
    configuration = CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            mode,
            1,
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([0.1], dtype=wp.float64, device="cpu"),
        ),
        PrescribedVolumeUpdate(final_volumes),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )
    return registry.acquire_communication(configuration)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("mode", "family"),
    (
        (CommunicationTransportMode.GAS, "communication_gas"),
        (CommunicationTransportMode.PARTICLES, "communication_particles"),
    ),
)
def test_capture_registration_protects_unselected_final_volumes(
    mode: CommunicationTransportMode, family: str
) -> None:
    """Test either published communication mode protects final volumes."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=3)
    registry = GPUResourceRegistry(session)
    final_volumes = wp.ones(3, dtype=wp.float64, device="cpu")
    resources = _communication_with_final_volumes(registry, final_volumes, mode)
    aliased = ResidentDiagnosticRegistration(
        ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
        final_volumes,
    )

    with pytest.raises(ValueError, match="byte ranges must not overlap"):
        registry.register_capture_resources(session, None, (aliased,))
    assert registry._capture_inventory is None

    output = wp.zeros(3, dtype=wp.float64, device="cpu")
    valid = ResidentDiagnosticRegistration(
        ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
        output,
    )
    inventory = registry.register_capture_resources(
        session, resources, (valid,)
    )

    assert inventory.communication_resources is resources
    assert inventory.families[0].family == family


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(1, 2, 1), (2, 1, 2)])
def test_all_families_allocate_complete_stable_resources(
    shape: tuple[int, int, int],
) -> None:
    """Test every family supplies complete records and stable repeats."""
    registry = GPUResourceRegistry(_session(*shape))
    collision_capacity = min(3, max(1, shape[1] ** 2))

    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(collision_capacity)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()

    condensation_buffers = cast(Any, condensation.scratch_buffers)
    nucleation_exhaustion = cast(Any, nucleation.exhaustion)
    nucleation_finalized_demand = cast(Any, nucleation.finalized_demand)

    assert registry.acquire_condensation() is condensation
    assert registry.acquire_coagulation(collision_capacity) is coagulation
    assert registry.acquire_wall_loss() is wall_loss
    assert registry.acquire_nucleation() is nucleation
    assert condensation_buffers.work_mass_transfer.shape == shape
    assert coagulation.collision_pairs.shape == (
        shape[0],
        collision_capacity,
        2,
    )
    assert coagulation.n_collisions.shape == (shape[0],)
    assert wall_loss.rng_states is not coagulation.rng_states
    assert (
        nucleation_exhaustion.resampling_buffers.replacement_masses.shape
        == shape
    )
    assert nucleation_finalized_demand.precursor_mass_change.shape == (
        shape[0],
        shape[2],
    )


@pytest.mark.warp
def test_coagulation_acquisition_initializes_one_persistent_stream() -> None:
    """Test acquisition initializes manifest-derived words only once."""
    session = _session(boxes=2)
    object.__setattr__(
        session,
        "metadata",
        session.metadata.__class__(
            session.metadata.device,
            session.metadata.gas_names,
            session.metadata.stream.__class__(
                2, 41, ("north", "south"), (1, 0)
            ),
        ),
    )
    registry = GPUResourceRegistry(session)

    resources = registry.acquire_coagulation(1)
    stream = registry._coagulation_stream_registry

    assert stream is not None
    np.testing.assert_array_equal(
        resources.rng_states.numpy(), stream.words_by_lane("coagulation")
    )
    before = resources.rng_states.numpy().copy()
    assert registry.acquire_coagulation(1) is resources
    np.testing.assert_array_equal(resources.rng_states.numpy(), before)


@pytest.mark.warp
def test_published_stream_reset_targets_only_selected_sidecars_and_lanes() -> (
    None
):
    """Test explicit resets retain sidecar identity and skip unselected lanes."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=2)
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    wp.copy(
        coagulation.rng_states,
        wp.array(
            np.full(2, 17, dtype=np.uint32), dtype=wp.uint32, device="cpu"
        ),
    )
    wp.copy(
        wall_loss.rng_states,
        wp.array(
            np.full(2, 19, dtype=np.uint32), dtype=wp.uint32, device="cpu"
        ),
    )

    registry.initialize_published_streams(
        session,
        process_ids=("coagulation",),
        logical_box_ids=("0",),
    )

    stream = registry._coagulation_stream_registry
    assert stream is not None
    assert coagulation.rng_states.numpy().tolist() == [
        stream.word_for("coagulation", "0"),
        17,
    ]
    assert wall_loss.rng_states.numpy().tolist() == [19, 19]
    inspection = registry.inspect_published_streams(session)
    assert inspection.published_process_ids == ("coagulation", "wall_loss")
    assert not hasattr(inspection, "rng_states")


@pytest.mark.warp
def test_published_stream_reset_rejects_empty_registry_selectors_without_allocation() -> (
    None
):
    """Test empty publication still validates selectors before sidecar work."""
    session = _session(boxes=1)
    registry = GPUResourceRegistry(session)

    with pytest.raises(ValueError, match="has not been acquired"):
        registry.initialize_published_streams(
            session, process_ids=("coagulation",)
        )
    with pytest.raises(LookupError, match="No lane"):
        registry.initialize_published_streams(
            session, logical_box_ids=("missing",)
        )

    assert registry._bindings == {}
    assert (
        registry.inspect_published_streams(session).published_process_ids == ()
    )


@pytest.mark.warp
def test_unpublished_process_rejection_precedes_published_stream_write() -> (
    None
):
    """Test a mixed request cannot partially reset an acquired process."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=1)
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    wp.copy(
        coagulation.rng_states,
        wp.array(
            np.array([17], dtype=np.uint32), dtype=wp.uint32, device="cpu"
        ),
    )

    with pytest.raises(ValueError, match="has not been acquired"):
        registry.initialize_published_streams(
            session,
            process_ids=("coagulation", "wall_loss"),
        )

    assert coagulation.rng_states.numpy().tolist() == [17]


@pytest.mark.warp
def test_published_stream_reset_preflights_every_registry_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a malformed later stream prevents every selected writer launch."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    stream = registry._wall_loss_stream_registry
    assert stream is not None
    invalid = wp.full(1, 19.0, dtype=wp.float64, device="cpu")
    object.__setattr__(
        stream,
        "_state_arrays",
        (("coagulation", wall_loss.rng_states), ("wall_loss", invalid)),
    )
    writes: list[object] = []
    monkeypatch.setattr(
        wp, "launch", lambda *args, **kwargs: writes.append(args)
    )

    with pytest.raises(
        TypeError, match="wall_loss state array must have dtype"
    ):
        registry.initialize_published_streams(session)

    assert writes == []
    assert coagulation.rng_states.numpy().tolist() != [19]
    assert session.lifecycle is ResidentLifecycle.ACTIVE


@pytest.mark.warp
def test_published_stream_reset_writer_failure_faults_bound_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a later stream writer failure faults without rolling back prior work."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    wp.copy(
        coagulation.rng_states, wp.full(1, 17, dtype=wp.uint32, device="cpu")
    )
    wp.copy(wall_loss.rng_states, wp.full(1, 19, dtype=wp.uint32, device="cpu"))
    original_launch = wp.launch
    launches = 0

    def fail_second_launch(*args: object, **kwargs: object) -> object:
        """Launch the first selected writer and fail the second deterministically."""
        nonlocal launches
        launches += 1
        if launches == 2:
            raise RuntimeError("second reset writer failed")
        return cast(Any, original_launch)(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", fail_second_launch)
    with pytest.raises(RuntimeError, match="second reset writer failed"):
        registry.initialize_published_streams(session)

    stream = registry._coagulation_stream_registry
    assert stream is not None
    assert coagulation.rng_states.numpy().tolist() == list(
        stream.words_by_lane("coagulation")
    )
    assert wall_loss.rng_states.numpy().tolist() == [19]
    assert session.lifecycle is ResidentLifecycle.FAULTED


@pytest.mark.warp
def test_published_stream_default_reset_uses_nondefault_root_seed() -> None:
    """Test all published streams reset every lane from their retained root."""
    session = _session(boxes=2)
    object.__setattr__(
        session,
        "metadata",
        session.metadata.__class__(
            session.metadata.device,
            session.metadata.gas_names,
            session.metadata.stream.__class__(
                2, 73, ("north", "south"), (1, 0)
            ),
        ),
    )
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    coagulation.rng_states.assign(np.full(2, 17, dtype=np.uint32))
    wall_loss.rng_states.assign(np.full(2, 19, dtype=np.uint32))

    registry.initialize_published_streams(session)

    for process_id, states, stream in (
        (
            "coagulation",
            coagulation.rng_states,
            registry._coagulation_stream_registry,
        ),
        (
            "wall_loss",
            wall_loss.rng_states,
            registry._wall_loss_stream_registry,
        ),
    ):
        assert stream is not None
        assert states.numpy().tolist() == list(stream.words_by_lane(process_id))


@pytest.mark.warp
def test_wall_loss_acquisition_initializes_its_distinct_persistent_stream() -> (
    None
):
    """Wall loss uses its own manifest-derived words and stable sidecar."""
    session = _session(boxes=2)
    registry = GPUResourceRegistry(session)

    wall_loss = registry.acquire_wall_loss()
    stream = registry._wall_loss_stream_registry

    assert stream is not None
    np.testing.assert_array_equal(
        wall_loss.rng_states.numpy(), stream.words_by_lane("wall_loss")
    )
    assert registry.acquire_wall_loss() is wall_loss
    coagulation = registry.acquire_coagulation(1)
    assert wall_loss.rng_states is not coagulation.rng_states


@pytest.mark.warp
def test_wall_loss_acquisition_does_not_reseed_coagulation() -> None:
    """Wall-loss acquisition preserves an already advanced coagulation stream."""
    session = _session(boxes=2)
    registry = GPUResourceRegistry(session)
    coagulation = registry.acquire_coagulation(1)
    advanced = coagulation.rng_states.numpy().copy()
    advanced[0] += np.uint32(1)
    wp = pytest.importorskip("warp")
    wp.copy(
        coagulation.rng_states,
        wp.array(advanced, dtype=wp.uint32, device="cpu"),
    )

    registry.acquire_wall_loss()

    np.testing.assert_array_equal(coagulation.rng_states.numpy(), advanced)


@pytest.mark.warp
@pytest.mark.parametrize(
    "failure_point", ("allocation", "registry", "initialize")
)
def test_coagulation_acquisition_failure_is_transactional(
    monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    """Test failed stream setup leaves no state published and retries cleanly."""
    registry = GPUResourceRegistry(_session())
    original_allocate = registry._allocate
    original_stream_registry: type[Any] = gpu_resources.StreamRegistry

    if failure_point == "allocation":
        monkeypatch.setattr(
            registry,
            "_allocate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("allocation failed")
            ),
        )
    elif failure_point == "registry":
        monkeypatch.setattr(
            gpu_resources,
            "StreamRegistry",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("registry failed")
            ),
        )
    else:

        class FailingStreamRegistry(original_stream_registry):
            """Fail only the staged initialization before publication."""

            def initialize(self) -> None:
                raise RuntimeError("initialize failed")

        monkeypatch.setattr(
            gpu_resources, "StreamRegistry", FailingStreamRegistry
        )

    with pytest.raises(RuntimeError, match=failure_point):
        registry.acquire_coagulation(1)

    assert registry._bindings == {}
    assert registry._capacities == {}
    assert registry._views == {}
    assert registry._coagulation_stream_registry is None

    monkeypatch.setattr(registry, "_allocate", original_allocate)
    monkeypatch.setattr(
        gpu_resources, "StreamRegistry", original_stream_registry
    )
    resources = registry.acquire_coagulation(1)

    assert registry._views["coagulation"] is resources
    assert registry._coagulation_stream_registry is not None


@pytest.mark.warp
def test_enumerate_resources_returns_established_arrays_in_manifest_order() -> (
    None
):
    """Checkpoint enumeration omits absent families and preserves live identities."""
    registry = GPUResourceRegistry(_session())

    assert registry._enumerate_resources() == ()

    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(2)
    entries = registry._enumerate_resources()

    expected = [
        (manifest.family, entry.role)
        for manifest in (
            gpu_resources._CONDENSATION,
            gpu_resources._COAGULATION,
        )
        for entry in manifest.entries
        if entry.role != "rng_states"
    ]
    assert [(family, role) for family, role, _, _ in entries] == expected
    assert entries[0][2] is condensation.scratch_buffers.work_mass_transfer
    assert entries[-1][2] is coagulation.n_collisions
    assert all(capacity is None for *_, capacity in entries[:7])
    assert all(capacity == 2 for *_, capacity in entries[7:])


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(1, 0, 0), (1, 0, 1), (1, 2, 0)])
def test_registry_allocates_canonical_zero_dimension_schemas(
    shape: tuple[int, int, int],
) -> None:
    """Test zero particle/species dimensions retain complete schemas."""
    wp = pytest.importorskip("warp")
    registry = GPUResourceRegistry(_session(*shape))

    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(1)
    nucleation = registry.acquire_nucleation()
    condensation_buffers = cast(Any, condensation.scratch_buffers)
    nucleation_diagnostics = cast(Any, nucleation.diagnostics)
    nucleation_finalized_demand = cast(Any, nucleation.finalized_demand)

    assert condensation_buffers.work_mass_transfer.shape == shape
    assert condensation_buffers.work_mass_transfer.strides == (
        shape[1] * shape[2] * 8,
        shape[2] * 8,
        8,
    )
    assert coagulation.collision_pairs.shape == (shape[0], 1, 2)
    assert coagulation.collision_pairs.dtype == wp.int32
    assert nucleation_diagnostics.selected_slot_indices.shape == (
        shape[0],
        shape[1],
    )
    assert nucleation_finalized_demand.precursor_mass_change.shape == (
        shape[0],
        shape[2],
    )


@pytest.mark.warp
def test_registry_rejects_replacement_capacity_and_primary_alias() -> None:
    """Test established role identity and protected primary ownership checks."""
    session = _session()
    registry = GPUResourceRegistry(session)
    with pytest.raises(ValueError, match="fixed-capacity bounds"):
        registry.acquire_coagulation(5)
    view = registry.acquire_wall_loss()
    session_temperature = cast(Any, session.environment).temperature
    with pytest.raises(ValueError, match="replaced"):
        registry.acquire_wall_loss(rng_states=session_temperature)
    registry.acquire_coagulation(1)
    with pytest.raises(ValueError, match="cannot change"):
        registry.acquire_coagulation(2)
    assert registry.acquire_wall_loss() is view


@pytest.mark.warp
def test_registry_rejects_session_drift_before_acquisition() -> None:
    """Test a fabricated changed frozen session cannot silently resize."""
    session = _session()
    registry = GPUResourceRegistry(session)
    object.__setattr__(session, "lifecycle", ResidentLifecycle.FAULTED)
    with pytest.raises(ValueError, match="ACTIVE"):
        registry.acquire_condensation()


@pytest.mark.warp
def test_registry_rejects_non_warp_and_invalid_capacity() -> None:
    """Test public metadata validation has no permissive binding path."""
    registry = GPUResourceRegistry(_session())
    with pytest.raises(TypeError, match="Warp array"):
        registry.acquire_wall_loss(rng_states=object())
    with pytest.raises(TypeError, match="non-boolean integral"):
        registry.acquire_coagulation(True)
    with pytest.raises(ValueError, match="positive"):
        registry.acquire_coagulation(0)
    with pytest.raises(TypeError):
        registry.acquire_wall_loss(unexpected=object())  # type: ignore[call-arg]


@pytest.mark.warp
def test_registry_accepts_complete_supplied_condensation_record() -> None:
    """Test complete native records retain their caller-owned array identity."""
    source_resources = GPUResourceRegistry(_session()).acquire_condensation()
    registry = GPUResourceRegistry(_session())

    acquired = registry.acquire_condensation(
        buffers=source_resources.scratch_buffers,
    )

    assert acquired.scratch_buffers is not source_resources.scratch_buffers
    assert (
        acquired.scratch_buffers.work_mass_transfer
        is source_resources.scratch_buffers.work_mass_transfer
    )
    assert (
        registry.acquire_condensation(
            buffers=source_resources.scratch_buffers,
        )
        is acquired
    )


@pytest.mark.warp
def test_registry_rejects_incomplete_and_inexact_condensation_records() -> None:
    """Test supplied records must be complete exact native scratch records."""
    source_resources = GPUResourceRegistry(_session()).acquire_condensation()
    buffers = source_resources.scratch_buffers
    object.__setattr__(buffers, "work_mass_transfer", None)
    registry = GPUResourceRegistry(_session())

    with pytest.raises(ValueError, match="complete"):
        registry.acquire_condensation(buffers=buffers)
    with pytest.raises(TypeError, match="exact CondensationScratchBuffers"):
        registry.acquire_condensation(buffers=object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_registry_rejects_invalid_sidecar_schema_and_session_signature() -> (
    None
):
    """Test sidecar schemas and immutable session identity are enforced."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wp = pytest.importorskip("warp")
    wrong_dtype = wp.zeros((1,), dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="incompatible schema"):
        registry.acquire_wall_loss(rng_states=wrong_dtype)
    object.__setattr__(session, "dimensions", ResidentDimensions(1, 3, 1))
    with pytest.raises(ValueError, match="signature changed"):
        registry.acquire_wall_loss()


@pytest.mark.warp
def test_registry_rejects_replaced_primary_identity_before_publication() -> (
    None
):
    """Test changing a protected primary rejects before another family binds."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    object.__setattr__(
        cast(Any, session.environment),
        "temperature",
        wp.ones((1,), dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(ValueError, match="signature changed"):
        registry.acquire_wall_loss()

    assert registry._bindings == {}


@pytest.mark.warp
def test_registry_rejects_replaced_container_before_allocation() -> None:
    """Test replacing a container rejects even when its arrays are unchanged."""
    from particula.gpu.warp_types import WarpParticleData

    session = _session()
    registry = GPUResourceRegistry(session)
    original_particles = cast(Any, session.particles)
    replacement_particles = WarpParticleData()
    for name in ("masses", "concentration", "charge", "density", "volume"):
        setattr(replacement_particles, name, getattr(original_particles, name))
    object.__setattr__(session, "particles", replacement_particles)

    with pytest.raises(ValueError, match="signature changed"):
        registry.acquire_wall_loss()

    assert registry._bindings == {}


@pytest.mark.warp
def test_registry_rejects_non_warp_primary_metadata_impostor() -> None:
    """Test primary metadata-shaped values cannot impersonate Warp arrays."""
    wp = pytest.importorskip("warp")

    class MetadataImpostor:
        """Provide the metadata shape of a Warp array without its type."""

        dtype = wp.float64
        shape = (1, 2, 1)
        device = "cpu"
        ptr = 0
        strides = (16, 8, 8)

    session = _session()
    object.__setattr__(
        cast(Any, session.particles), "masses", MetadataImpostor()
    )

    with pytest.raises(ValueError, match="Warp array"):
        GPUResourceRegistry(session)


@pytest.mark.warp
@pytest.mark.parametrize(
    ("carrier", "field", "role"),
    [
        ("particles", "masses", "work_mass_transfer"),
        ("gas", "concentration", "positive_mass_transfer_demand"),
        ("environment", "temperature", "dynamic_viscosity"),
    ],
)
def test_registry_rejects_particle_gas_and_environment_primary_aliases(
    carrier: str,
    field: str,
    role: str,
) -> None:
    """Test each protected primary category cannot become a sidecar."""
    session = _session()
    registry = GPUResourceRegistry(session)
    buffers = GPUResourceRegistry(_session()).acquire_condensation()
    object.__setattr__(
        buffers.scratch_buffers,
        role,
        getattr(getattr(session, carrier), field),
    )

    with pytest.raises(ValueError, match="alias"):
        registry.acquire_condensation(buffers=buffers.scratch_buffers)

    assert registry._bindings == {}


@pytest.mark.warp
def test_registry_rejects_cross_family_duplicate_sidecar_identity() -> None:
    """Test persistent RNG storage cannot be reused across resource families."""
    registry = GPUResourceRegistry(_session())
    wall_loss = registry.acquire_wall_loss()

    with pytest.raises(ValueError, match="share identity"):
        registry.acquire_coagulation(1, rng_states=wall_loss.rng_states)

    assert set(registry._bindings) == {"wall_loss"}


@pytest.mark.warp
def test_cross_family_alias_rejects_before_omitted_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test supplied cross-family aliases fail before omitted buffers allocate."""
    registry = GPUResourceRegistry(_session())
    wall_loss = registry.acquire_wall_loss()
    calls = 0
    original_zeros = cast(Any, gpu_resources.wp.zeros)

    def tracked_zeros(*args: object, **kwargs: object) -> Any:
        """Track unexpected omitted-sidecar allocations."""
        nonlocal calls
        calls += 1
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(cast(Any, gpu_resources.wp), "zeros", tracked_zeros)
    with pytest.raises(ValueError, match="share identity"):
        registry.acquire_coagulation(1, rng_states=wall_loss.rng_states)

    assert calls == 0
    assert set(registry._bindings) == {"wall_loss"}


@pytest.mark.warp
def test_registry_accepts_complete_supplied_nucleation_records() -> None:
    """Test every supplied nucleation record is retained by array identity."""
    source = GPUResourceRegistry(_session()).acquire_nucleation()
    registry = GPUResourceRegistry(_session())

    acquired = registry.acquire_nucleation(
        scratch=source.scratch,
        finalized_demand=source.finalized_demand,
        diagnostics=source.diagnostics,
        exhaustion=source.exhaustion,
    )

    for record_name in (
        "scratch",
        "finalized_demand",
        "diagnostics",
        "exhaustion",
    ):
        supplied = getattr(source, record_name)
        retained = getattr(acquired, record_name)
        for field in fields(supplied):
            if field.name == "resampling_buffers":
                for nested in fields(getattr(supplied, field.name)):
                    assert getattr(
                        getattr(retained, field.name), nested.name
                    ) is getattr(getattr(supplied, field.name), nested.name)
            else:
                assert getattr(retained, field.name) is getattr(
                    supplied, field.name
                )
    assert registry.acquire_nucleation() is acquired


@pytest.mark.warp
def test_registry_rejects_incomplete_and_inexact_nucleation_records() -> None:
    """Test native nucleation records must be exact and complete."""
    source = GPUResourceRegistry(_session()).acquire_nucleation()
    object.__setattr__(source.diagnostics, "gate_codes", None)
    registry = GPUResourceRegistry(_session())

    with pytest.raises(ValueError, match="complete"):
        registry.acquire_nucleation(diagnostics=source.diagnostics)
    with pytest.raises(TypeError, match="exact native types"):
        registry.acquire_nucleation(diagnostics=object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_capture_resource_set_reuses_exact_p3_and_family_identities() -> None:
    """A compatible P4 request publishes and returns one exact frozen set."""
    session = _session()
    registry = GPUResourceRegistry(session)
    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()
    inventory = registry.register_capture_resources(session, None, ())
    views = PreparedResourceViews(
        condensation,
        coagulation,
        None,
        wall_loss,
        nucleation,
    )
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(1, 0, 0),
        inventory,
        views,
        None,
        condensation.scratch_buffers,
        coagulation,
        wall_loss,
        nucleation,
    )

    capture_set = registry.prepare_capture_resources(requirements)

    assert capture_set.requirements is requirements
    assert capture_set.inventory is inventory
    assert capture_set.prepared_views is not views
    assert capture_set.prepared_views.condensation is condensation
    assert capture_set.prepared_views.coagulation is coagulation
    assert capture_set.prepared_views.wall_loss is wall_loss
    assert capture_set.prepared_views.nucleation is nucleation
    assert capture_set.condensation is condensation
    assert capture_set.coagulation is coagulation
    assert capture_set.wall_loss is wall_loss
    assert capture_set.nucleation is nucleation
    assert registry.validate_capture_resource_set(requirements) is capture_set
    assert registry.prepare_capture_resources(requirements) is capture_set


@pytest.mark.warp
def test_capture_resource_set_rejects_distinct_requirements_without_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value-equal but distinct request cannot replace a published set."""
    session = _session()
    registry = GPUResourceRegistry(session)
    condensation = registry.acquire_condensation()
    inventory = registry.register_capture_resources(session, None, ())
    views = PreparedResourceViews(condensation=condensation)
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(0, 0, 0),
        inventory,
        views,
        None,
        condensation.scratch_buffers,
    )
    capture_set = registry.prepare_capture_resources(requirements)
    distinct = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(0, 0, 0),
        inventory,
        views,
        None,
        condensation.scratch_buffers,
    )
    monkeypatch.setattr(
        registry,
        "logical_resource_report",
        lambda *_args: pytest.fail("repeat must not build a report"),
    )

    with pytest.raises(ValueError, match="identities are incompatible"):
        registry.prepare_capture_resources(distinct)
    with pytest.raises(ValueError, match="identities are incompatible"):
        registry.validate_capture_resource_set(distinct)
    assert registry.validate_capture_resource_set(requirements) is capture_set


@pytest.mark.warp
def test_capture_resource_view_drift_rejects_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonpublished prepared view cannot trigger candidate allocation."""
    session = _session()
    registry = GPUResourceRegistry(session)
    inventory = registry.register_capture_resources(session, None, ())
    placeholder = gpu_resources.CondensationResources(
        gpu_resources.CondensationScratchBuffers(
            None, None, None, None, None, None, None
        )
    )
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(0, 0, 0),
        inventory,
        PreparedResourceViews(condensation=placeholder),
        None,
    )
    monkeypatch.setattr(
        registry,
        "_allocate",
        lambda *_args, **_kwargs: pytest.fail("drift must not allocate"),
    )

    with pytest.raises(ValueError, match="view identity changed"):
        registry.prepare_capture_resources(requirements)

    assert registry._capture_resource_set is None
    assert registry.selected_resource_report() is inventory
    assert registry._bindings == {}


@pytest.mark.warp
def test_capture_resource_capacity_rejects_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4 reuses ordinary collision bounds before any candidate work."""
    session = _session(particle_count=2)
    registry = GPUResourceRegistry(session)
    inventory = registry.register_capture_resources(session, None, ())
    placeholder = gpu_resources.CoagulationResources(0, None, None, None)
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(5, 0, 0),
        inventory,
        PreparedResourceViews(coagulation=placeholder),
        None,
    )
    monkeypatch.setattr(
        registry,
        "_allocate",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid capacity must not allocate"
        ),
    )

    with pytest.raises(ValueError, match="fixed-capacity bounds"):
        registry.prepare_capture_resources(requirements)

    assert registry._bindings == {}


@pytest.mark.warp
def test_capture_resource_rejects_inventory_alias_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4 protects retained diagnostic inventory storage from staged aliases."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wp = pytest.importorskip("warp")
    output = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    plan = _diagnostics_plan(
        session,
        registry,
        (output, wp.zeros((1, 1), dtype=wp.float64, device="cpu")),
    )
    inventory = registry.register_capture_resources(
        session, None, plan.registrations
    )
    source = GPUResourceRegistry(session).acquire_condensation().scratch_buffers
    supplied = gpu_resources.CondensationScratchBuffers(
        source.work_mass_transfer,
        source.total_mass_transfer,
        source.dynamic_viscosity,
        source.mean_free_path,
        output,
        source.negative_mass_transfer_release,
        source.positive_mass_transfer_scale,
    )
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(0, 0, 0),
        inventory,
        PreparedResourceViews(),
        None,
        condensation=supplied,
    )
    monkeypatch.setattr(
        registry,
        "_allocate",
        lambda *_args, **_kwargs: pytest.fail(
            "inventory alias must not allocate"
        ),
    )

    with pytest.raises(ValueError, match="Sidecar"):
        registry.prepare_capture_resources(requirements)

    assert registry._bindings == {}


def test_execution_package_remains_dependency_neutral() -> None:
    """Test the package import neither exports nor eagerly loads resources."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import sys
import particula.execution as execution
assert 'GPUResourceRegistry' not in execution.__all__
assert 'particula.execution.gpu_resources' not in sys.modules
assert 'warp' not in sys.modules
assert not any(name == 'particula.gpu' or name.startswith('particula.gpu.') for name in sys.modules)
"""
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-Werror", "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_manifest_helpers_reject_unsupported_dtype_and_size() -> None:
    """Test private manifest helpers reject unsupported allocation metadata."""
    with pytest.raises(ValueError, match="Unsupported manifest dtype"):
        _item_size(object())

    registry = object.__new__(GPUResourceRegistry)
    with pytest.raises(ValueError, match="exceeds supported range"):
        registry._checked_product(2**63, 2)
    with pytest.raises(ValueError, match="exceeds supported range"):
        registry._checked_product(1, -1)
    with pytest.raises(ValueError, match="exceeds supported range"):
        registry._checked_product(2**62, 2)
    entry = ManifestEntry("pairs", "coagulation", object(), "bc2")
    registry._session = _session()
    with pytest.raises(ValueError, match="collision capacity"):
        registry._shape(entry)


def test_array_range_rejects_stride_overflow_before_pointer_access() -> None:
    """Test centralized stride arithmetic rejects overflow before range use."""
    array = type(
        "array",
        (),
        {
            "dtype": gpu_resources.wp.float64,
            "shape": (2**62, 2),
            "strides": (16, 8),
            "ptr": 8,
            "capacity": 16,
        },
    )()
    with pytest.raises(ValueError, match="exceeds supported range"):
        GPUResourceRegistry._array_range(array)


@pytest.mark.warp
def test_byte_range_endpoint_overflow_rejects_supplied_and_registered_arrays() -> (
    None
):
    """Test range endpoints reject overflow before alias comparisons."""
    session = _session()
    registry = GPUResourceRegistry(session)
    device = cast(Any, session.particles).masses.device
    array = type(
        "array",
        (),
        {
            "dtype": gpu_resources.wp.float64,
            "shape": (1,),
            "strides": (8,),
            "device": device,
            "ptr": _MAX_SIZE - 7,
            "capacity": 8,
        },
    )()
    type(array).__module__ = "warp"

    with pytest.raises(ValueError, match="byte range exceeds supported range"):
        registry._contiguous_range(
            array, (1,), gpu_resources.wp.float64, "supplied"
        )
    with pytest.raises(ValueError, match="byte range exceeds supported range"):
        GPUResourceRegistry._array_range(array)

    assert registry._bindings == {}
    assert registry._views == {}
    assert registry._capacities == {}


def test_registry_requires_exact_active_session() -> None:
    """Test construction rejects values outside the exact active boundary."""
    with pytest.raises(TypeError, match="exact ResidentSession"):
        GPUResourceRegistry(object())  # type: ignore[arg-type]


@pytest.mark.warp
def test_registry_requires_active_session_and_exposes_all_manifests() -> None:
    """Test construction enforces ACTIVE state and publishes fixed schemas."""
    session = _session()
    object.__setattr__(session, "lifecycle", ResidentLifecycle.CLOSED)
    with pytest.raises(ValueError, match="ACTIVE"):
        GPUResourceRegistry(session)

    manifests = GPUResourceRegistry(_session()).manifests
    assert tuple(manifest.family for manifest in manifests) == (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
        "communication_gas",
        "communication_particles",
        "dilution",
    )
    assert all(
        entry.role and entry.dtype
        for manifest in manifests
        for entry in manifest.entries
    )


@pytest.mark.warp
def test_logical_resource_report_resolves_manifest_schemas_without_acquisition() -> (
    None
):
    """Test inventory reports use only logical manifest schemas and capacities."""
    registry = GPUResourceRegistry(_session(2, 3, 4))
    capacities = ResourceInventoryCapacities(5, 6, 7)
    report = registry.logical_resource_report(capacities)

    assert tuple(family.family for family in report.families) == (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
        "communication_gas",
        "communication_particles",
        "dilution",
    )
    assert registry._bindings == {}
    assert registry._views == {}
    by_family = {family.family: family for family in report.families}
    condensation = by_family["condensation"].roles[0]
    pairs = by_family["coagulation"].roles[0]
    gas_edge = by_family["communication_gas"].roles[0]
    particle_edges = by_family["communication_particles"].roles[6]
    assert (condensation.entry.shape, condensation.element_count) == (
        (2, 3, 4),
        24,
    )
    assert condensation.logical_byte_count == 192
    assert (
        pairs.entry.shape,
        pairs.element_count,
        pairs.logical_byte_count,
    ) == (
        (2, 5, 2),
        20,
        80,
    )
    assert (gas_edge.entry.shape, gas_edge.entry.capacity_source) == (
        (6,),
        "gas_edge_capacity",
    )
    assert particle_edges.entry.shape == (7, 3)
    assert particle_edges.entry.capacity_source == "particle_edge_capacity"
    assert gas_edge.entry.ownership == "caller_configuration"
    assert pairs.entry.ownership == "registry_or_caller_sidecar"
    dilution = by_family["dilution"].roles
    assert [role.entry.role for role in dilution] == [
        "normalized_coefficient",
        "factors",
    ]
    assert all(role.entry.shape == (2,) for role in dilution)
    assert report.logical_byte_count == sum(
        family.logical_byte_count for family in report.families
    )
    assert all(
        family.logical_byte_count
        == sum(role.logical_byte_count for role in family.roles)
        for family in report.families
    )
    assert report == registry.logical_resource_report(capacities)

    registry.acquire_condensation()
    bindings = registry._bindings.copy()
    views = registry._views.copy()
    assert registry.logical_resource_report(capacities) == report
    assert registry._bindings == bindings
    assert registry._views == views


@pytest.mark.warp
def test_descriptor_only_dilution_view_validates_without_publication() -> None:
    """Test prepared dilution schemas do not create registry bindings."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=2)
    registry = GPUResourceRegistry(session)
    view = DilutionResources(
        wp.zeros(2, dtype=wp.float64, device="cpu"),
        wp.zeros(2, dtype=wp.float64, device="cpu"),
    )

    registry.validate_prepared_resource_views(
        session, PreparedResourceViews(dilution=view)
    )

    assert registry._bindings == {}
    assert registry._views == {}
    with pytest.raises(ValueError, match="share identity"):
        registry.validate_dilution_resources(
            session,
            DilutionResources(
                view.normalized_coefficient, view.normalized_coefficient
            ),
        )


@pytest.mark.warp
def test_dilution_resources_accept_canonical_empty_factors_without_mutation() -> (
    None
):
    """An empty resident layout accepts both canonical empty dilution sidecars."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=0)
    registry = GPUResourceRegistry(session)
    coefficient = wp.zeros(0, dtype=wp.float64, device="cpu")
    factors = wp.zeros(0, dtype=wp.float64, device="cpu")

    registry.validate_dilution_resources(
        session, DilutionResources(coefficient, factors)
    )

    assert coefficient.shape == (0,)
    assert factors.shape == (0,)
    assert registry._bindings == {}
    assert registry._views == {}


@pytest.mark.warp
@pytest.mark.parametrize(
    ("kind", "match"),
    (
        ("carrier", "exact DilutionResources"),
        ("type", "normalized_coefficient must be a Warp array"),
        ("dtype", "normalized_coefficient has incompatible schema"),
        ("shape", "normalized_coefficient has incompatible schema"),
        ("self", "must not share identity"),
        ("primary", "must not alias session primaries"),
        ("established", "must not share identity"),
    ),
)
def test_dilution_resource_validation_rejects_invalid_views_without_mutation(
    kind: str, match: str
) -> None:
    """Invalid dilution views reject before changing primaries or publications."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    coefficient = wp.full(1, 11.0, dtype=wp.float64, device="cpu")
    factors = wp.full(1, 13.0, dtype=wp.float64, device="cpu")
    candidate: Any = DilutionResources(coefficient, factors)
    established = None
    if kind == "carrier":
        candidate = object()
    elif kind == "type":
        candidate = DilutionResources(object(), factors)
    elif kind == "dtype":
        candidate = DilutionResources(
            wp.zeros(1, dtype=wp.float32, device="cpu"), factors
        )
    elif kind == "shape":
        candidate = DilutionResources(
            wp.zeros((1, 1), dtype=wp.float64, device="cpu"), factors
        )
    elif kind == "self":
        candidate = DilutionResources(coefficient, coefficient)
    elif kind == "primary":
        candidate = DilutionResources(
            cast(Any, session.environment).temperature, factors
        )
    elif kind == "established":
        established = (
            registry.acquire_condensation().scratch_buffers.dynamic_viscosity
        )
        candidate = DilutionResources(coefficient, established)
    primaries_before = tuple(
        array.numpy().copy()
        for array in (
            cast(Any, session.particles).masses,
            cast(Any, session.particles).concentration,
            cast(Any, session.gas).concentration,
        )
    )
    candidate_before = (coefficient.numpy().copy(), factors.numpy().copy())
    bindings = registry._bindings.copy()
    views = registry._views.copy()

    with pytest.raises((TypeError, ValueError), match=match):
        registry.validate_dilution_resources(session, candidate)

    np.testing.assert_array_equal(coefficient.numpy(), candidate_before[0])
    np.testing.assert_array_equal(factors.numpy(), candidate_before[1])
    for current, before in zip(
        (
            cast(Any, session.particles).masses,
            cast(Any, session.particles).concentration,
            cast(Any, session.gas).concentration,
        ),
        primaries_before,
        strict=True,
    ):
        np.testing.assert_array_equal(current.numpy(), before)
    assert registry._bindings == bindings
    assert registry._views == views
    if established is not None:
        np.testing.assert_array_equal(established.numpy(), np.zeros(1))


@pytest.mark.warp
def test_dilution_resource_validation_rejects_overlapping_nonempty_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct valid sidecars with overlapping ranges reject before mutation."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    coefficient = wp.full(1, 11.0, dtype=wp.float64, device="cpu")
    factors = wp.full(1, 13.0, dtype=wp.float64, device="cpu")
    original_validate = registry._validate_array

    def overlapping_ranges(
        entry: ManifestEntry, value: Any, capacity: int | None
    ) -> tuple[int, int] | None:
        """Preserve schema checks while modeling overlapping supplied storage."""
        original_validate(entry, value, capacity)
        return (8, 16)

    monkeypatch.setattr(registry, "_validate_array", overlapping_ranges)
    before = (coefficient.numpy().copy(), factors.numpy().copy())

    with pytest.raises(ValueError, match="byte ranges must not overlap"):
        registry.validate_dilution_resources(
            session, DilutionResources(coefficient, factors)
        )

    np.testing.assert_array_equal(coefficient.numpy(), before[0])
    np.testing.assert_array_equal(factors.numpy(), before[1])
    assert registry._bindings == {}
    assert registry._views == {}


@pytest.mark.warp
@pytest.mark.parametrize("capacities", ((0, 0, 0), (0, 2, 0)))
def test_logical_resource_report_retains_zero_extent_roles(
    capacities: tuple[int, int, int],
) -> None:
    """Test valid zero capacities remain present with zero logical bytes."""
    report = GPUResourceRegistry(_session(0, 3, 0)).logical_resource_report(
        ResourceInventoryCapacities(*capacities)
    )
    roles = tuple(role for family in report.families for role in family.roles)
    assert any(role.entry.shape == (0, 0, 2) for role in roles)
    assert any(role.entry.shape == (0,) for role in roles)
    assert any(
        role.entry.capacity_source == "collision_capacity"
        and role.logical_byte_count == 0
        for role in roles
    )
    assert any(
        role.entry.capacity_source == "particle_edge_capacity"
        and role.logical_byte_count == 0
        for role in roles
    )
    assert report.logical_byte_count > 0


@pytest.mark.warp
@pytest.mark.parametrize(
    "capacities",
    (
        (-1, 1, 1),
        (True, 1, 1),
        (1.5, 1, 1),
        (1, -1, 1),
        (1, True, 1),
        (1, 1.5, 1),
        (1, 1, -1),
        (1, 1, True),
        (1, 1, 1.5),
    ),
)
def test_logical_resource_report_rejects_invalid_capacities_before_mutation(
    capacities: tuple[Any, Any, Any],
) -> None:
    """Test invalid logical capacities reject without publishing resources."""
    registry = GPUResourceRegistry(_session())
    with pytest.raises(ValueError, match="non-boolean nonnegative integers"):
        registry.logical_resource_report(
            ResourceInventoryCapacities(*capacities)
        )
    with pytest.raises(TypeError, match="exact ResourceInventoryCapacities"):
        registry.logical_resource_report((1, 1, 1))  # type: ignore[arg-type]
    assert registry._bindings == {}
    assert registry._views == {}
    assert registry._capacities == {}


@pytest.mark.warp
@pytest.mark.parametrize(
    ("dimensions", "capacities"),
    (
        ((0, 3, 2), (_MAX_SIZE + 1, 0, 0)),
        ((0, 3, 2), (0, _MAX_SIZE + 1, 0)),
        ((2, 0, 2), (0, 0, _MAX_SIZE + 1)),
    ),
)
def test_logical_resource_report_rejects_oversized_capacity_before_mutation(
    dimensions: tuple[int, int, int],
    capacities: tuple[int, int, int],
) -> None:
    """Test zero dimensions cannot mask an oversized dynamic capacity."""
    registry = GPUResourceRegistry(_session(*dimensions))

    with pytest.raises(ValueError, match="exceeds supported range"):
        registry.logical_resource_report(
            ResourceInventoryCapacities(*capacities)
        )

    assert registry._bindings == {}
    assert registry._views == {}
    assert registry._capacities == {}


@pytest.mark.warp
def test_logical_resource_report_lists_complete_canonical_inventory() -> None:
    """Test every canonical role has independently expected report metadata."""
    report = GPUResourceRegistry(_session(2, 3, 4)).logical_resource_report(
        ResourceInventoryCapacities(5, 6, 7)
    )
    expected = (
        (
            "condensation",
            (
                ("work_mass_transfer", (2, 3, 4), 8),
                ("total_mass_transfer", (2, 3, 4), 8),
                ("dynamic_viscosity", (2,), 8),
                ("mean_free_path", (2,), 8),
                ("positive_mass_transfer_demand", (2, 4), 8),
                ("negative_mass_transfer_release", (2, 4), 8),
                ("positive_mass_transfer_scale", (2, 4), 8),
            ),
        ),
        (
            "coagulation",
            (
                ("collision_pairs", (2, 5, 2), 4),
                ("n_collisions", (2,), 4),
                ("rng_states", (2,), 4),
            ),
        ),
        ("wall_loss", (("rng_states", (2,), 4),)),
        (
            "nucleation",
            (
                ("precursor_number_concentration", (2,), 8),
                ("potential_rate", (2,), 8),
                ("potential_demand", (2,), 8),
                ("accepted_counts", (2,), 4),
                ("accepted_demand", (2,), 8),
                ("precursor_mass_change", (2, 4), 8),
                ("gate_codes", (2,), 4),
                ("selected_slot_indices", (2, 3), 4),
                ("free_slot_indices", (2, 3), 4),
                ("active_slot_counts", (2,), 4),
                ("free_slot_counts", (2,), 4),
                ("retained_counts", (2,), 4),
                ("released_counts", (2,), 4),
                ("retained_indices", (2, 3), 4),
                ("released_indices", (2, 3), 4),
                ("sorted_indices", (2, 3), 4),
                ("replacement_masses", (2, 3, 4), 8),
                ("replacement_concentration", (2, 3), 8),
                ("replacement_charge", (2, 3), 8),
                ("source_radii", (2, 3), 8),
                ("radius_cubed_relative_error", (2,), 8),
                ("mean_radius_relative_error", (2,), 8),
                ("surface_relative_error", (2,), 8),
                ("diversity_absolute_error", (2,), 8),
                ("planning_status", (2,), 4),
                ("demand_workspace", (2,), 8),
                ("final_demand", (2,), 8),
                ("requested_scale", (2,), 8),
                ("minimum_scale", (2,), 8),
                ("minimum_volume", (2,), 8),
                ("resolved_scale", (2,), 8),
                ("resampling_releasable_counts", (2,), 4),
                ("required_release_counts", (2,), 4),
                ("scaling_required", (2,), 4),
                ("final_counts", (2,), 4),
                ("final_selected_slot_indices", (2, 3), 4),
            ),
        ),
        (
            "communication_gas",
            (
                ("source_boxes", (6,), 4),
                ("destination_boxes", (6,), 4),
                ("enabled", (6,), 4),
                ("rates", (6,), 8),
                ("amounts", (2, 4), 8),
                ("amount_deltas", (2, 4), 8),
                ("outbound_amounts", (2, 4), 8),
                ("invalid", (1,), 4),
                ("active_or_demand", (1,), 4),
                ("volume_invalid", (1,), 4),
                ("volume_changed", (1,), 4),
            ),
        ),
        (
            "communication_particles",
            (
                ("source_boxes", (7,), 4),
                ("destination_boxes", (7,), 4),
                ("enabled", (7,), 4),
                ("rates", (7,), 8),
                ("source_debits", (2, 3), 8),
                ("destination_credits", (2, 3), 8),
                ("assignments", (7, 3), 4),
                ("request_concentrations", (7, 3), 8),
                ("invalid", (1,), 4),
                ("active_or_demand", (1,), 4),
                ("volume_invalid", (1,), 4),
                ("volume_changed", (1,), 4),
                ("initial_masses", (2, 3, 4), 8),
                ("initial_concentration", (2, 3), 8),
                ("initial_charge", (2, 3), 8),
            ),
        ),
        (
            "dilution",
            (
                ("normalized_coefficient", (2,), 8),
                ("factors", (2,), 8),
            ),
        ),
    )
    expected_total = 0
    assert tuple(family.family for family in report.families) == tuple(
        family for family, _ in expected
    )
    for actual_family, (family_name, expected_roles) in zip(
        report.families, expected, strict=True
    ):
        expected_family_total = 0
        assert actual_family.family == family_name
        assert tuple(role.entry.role for role in actual_family.roles) == tuple(
            role for role, _, _ in expected_roles
        )
        for actual_role, (role, shape, item_size) in zip(
            actual_family.roles, expected_roles, strict=True
        ):
            element_count = 1
            for extent in shape:
                element_count *= extent
            expected_bytes = element_count * item_size
            expected_family_total += expected_bytes
            is_configuration = family_name.startswith(
                "communication_"
            ) and role in {
                "source_boxes",
                "destination_boxes",
                "enabled",
                "rates",
            }
            assert actual_role.entry.shape == shape
            assert actual_role.element_count == element_count
            assert actual_role.logical_byte_count == expected_bytes
            assert actual_role.entry.ownership == (
                "caller_configuration"
                if is_configuration
                else "registry_or_caller_sidecar"
            )
            assert actual_role.entry.capacity_source == (
                "collision_capacity"
                if role == "collision_pairs"
                else "gas_edge_capacity"
                if family_name == "communication_gas" and shape[0] == 6
                else "particle_edge_capacity"
                if family_name == "communication_particles" and shape[0] == 7
                else "fixed"
            )
        assert actual_family.logical_byte_count == expected_family_total
        expected_total += expected_family_total
    assert report.logical_byte_count == expected_total


@pytest.mark.warp
def test_logical_resource_report_rejects_closed_session_before_reporting() -> (
    None
):
    """Test inventory validates the pinned session before resolving schemas."""
    session = _session()
    registry = GPUResourceRegistry(session)
    object.__setattr__(session, "lifecycle", ResidentLifecycle.CLOSED)

    with pytest.raises(ValueError, match="ACTIVE"):
        registry.logical_resource_report(ResourceInventoryCapacities(1, 1, 1))

    assert registry._bindings == {}
    assert registry._views == {}
    assert registry._capacities == {}


@pytest.mark.warp
def test_logical_resource_report_carriers_are_frozen_and_concrete_only() -> (
    None
):
    """Test reports are immutable and remain absent from package exports."""
    report = GPUResourceRegistry(_session()).logical_resource_report(
        ResourceInventoryCapacities(1, 1, 1)
    )
    with pytest.raises(FrozenInstanceError):
        report.logical_byte_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.families[0].roles[0].entry.shape = ()  # type: ignore[misc]
    import particula
    import particula.execution as execution

    inventory_names = (
        "ResourceInventoryCapacities",
        "ResolvedResourceInventoryEntry",
        "LogicalResourceRoleReport",
        "LogicalResourceFamilyReport",
        "LogicalResourceReport",
    )
    for name in inventory_names:
        assert hasattr(gpu_resources, name)
        assert not hasattr(execution, name)
        assert not hasattr(particula, name)


@pytest.mark.warp
def test_validate_pinned_session_rejects_other_or_drift_without_allocation() -> (
    None
):
    """Test direct binding validation is exact and metadata-only."""
    session = _session()
    registry = GPUResourceRegistry(session)
    bindings = registry._bindings.copy()
    views = registry._views.copy()

    with pytest.raises(ValueError, match="pinned ResidentSession"):
        registry.validate_pinned_session(_session())
    registry.validate_pinned_session(session)
    object.__setattr__(session, "lifecycle", ResidentLifecycle.FINALIZED)
    with pytest.raises(ValueError, match="ACTIVE"):
        registry.validate_pinned_session(session)

    assert registry._bindings == bindings
    assert registry._views == views


@pytest.mark.warp
def test_validate_pinned_session_rejects_primary_and_container_drift() -> None:
    """Test binding validation catches identity drift without acquisition."""
    wp = pytest.importorskip("warp")
    from particula.gpu.warp_types import WarpParticleData

    session = _session()
    registry = GPUResourceRegistry(session)
    particles = cast(Any, session.particles)
    original_masses = particles.masses
    object.__setattr__(
        particles,
        "masses",
        wp.ones((1, 2, 1), dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(ValueError, match="signature changed"):
        registry.validate_pinned_session(session)
    object.__setattr__(particles, "masses", original_masses)

    replacement = WarpParticleData()
    for name in ("masses", "concentration", "charge", "density", "volume"):
        setattr(replacement, name, getattr(particles, name))
    object.__setattr__(session, "particles", replacement)
    with pytest.raises(ValueError, match="signature changed"):
        registry.validate_pinned_session(session)

    assert registry._bindings == {}
    assert registry._views == {}


@pytest.mark.warp
@pytest.mark.parametrize(
    ("pointer", "capacity", "match"),
    (
        (0, 8, "valid pointer"),
        (4, 8, "8-byte aligned"),
        (8, 7, "sufficient integral storage capacity"),
        (8, 9, "sufficient integral storage capacity"),
    ),
)
def test_diagnostics_reject_invalid_nonempty_pointer_backing_before_alias_checks(
    monkeypatch: pytest.MonkeyPatch,
    pointer: int,
    capacity: int,
    match: str,
) -> None:
    """Test diagnostic outputs validate pointer alignment and capacity first."""
    session = _session()
    registry = GPUResourceRegistry(session)
    device = cast(Any, session.particles).masses.device
    invalid_output = type(
        "array",
        (),
        {
            "dtype": gpu_resources.wp.float64,
            "shape": (1, 1),
            "strides": (8, 8),
            "device": device,
            "ptr": pointer,
            "capacity": capacity,
        },
    )()
    type(invalid_output).__module__ = "warp"
    monkeypatch.setattr(gpu_resources.wp, "array", None)

    with pytest.raises(ValueError, match=match):
        registry.validate_diagnostic_outputs(session, (invalid_output,))


@pytest.mark.warp
def test_established_view_validators_require_exact_published_views() -> None:
    """Test adapter seams accept only their exact acquired resource views."""
    session = _session()
    registry = GPUResourceRegistry(session)
    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()
    bindings = registry._bindings.copy()
    views = registry._views.copy()
    capacities = registry._capacities.copy()

    registry.validate_wall_loss_resources(session, wall_loss)
    registry.validate_nucleation_resources(session, nucleation)
    with pytest.raises(TypeError, match="exact WallLossResources"):
        registry.validate_wall_loss_resources(session, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact NucleationResources"):
        registry.validate_nucleation_resources(session, object())  # type: ignore[arg-type]

    assert registry._bindings == bindings
    assert registry._views == views
    assert registry._capacities == capacities


@pytest.mark.warp
def test_condensation_and_coagulation_validators_accept_published_views() -> (
    None
):
    """Test established condensation and coagulation views validate by identity."""
    session = _session()
    registry = GPUResourceRegistry(session)
    condensation = registry.acquire_condensation()
    coagulation = registry.acquire_coagulation(1)
    bindings = registry._bindings.copy()
    views = registry._views.copy()

    registry.validate_condensation_resources(session, condensation)
    registry.validate_coagulation_resources(session, coagulation)

    assert registry._bindings == bindings
    assert registry._views == views


@pytest.mark.warp
def test_established_view_validators_reject_absence_identity_and_binding_drift() -> (
    None
):
    """Test published-view rejection paths leave registry bookkeeping intact."""
    session = _session()
    registry = GPUResourceRegistry(session)
    capacities = registry._capacities.copy()
    primaries = (session.particles, session.gas, session.environment)
    token = object()
    registry.reserve_open_step(token)

    foreign = GPUResourceRegistry(_session())
    foreign_wall = foreign.acquire_wall_loss()
    foreign_nucleation = foreign.acquire_nucleation()
    with pytest.raises(ValueError, match="have not been acquired"):
        registry.validate_wall_loss_resources(session, foreign_wall)
    with pytest.raises(ValueError, match="have not been acquired"):
        registry.validate_nucleation_resources(session, foreign_nucleation)

    wall_loss = registry.acquire_wall_loss()
    nucleation = registry.acquire_nucleation()
    with pytest.raises(ValueError, match="published wall_loss"):
        registry.validate_wall_loss_resources(
            session, gpu_resources.WallLossResources(wall_loss.rng_states)
        )
    with pytest.raises(ValueError, match="published nucleation"):
        registry.validate_nucleation_resources(
            session,
            gpu_resources.NucleationResources(
                nucleation.scratch,
                nucleation.finalized_demand,
                nucleation.diagnostics,
                nucleation.exhaustion,
            ),
        )

    object.__setattr__(wall_loss, "rng_states", foreign_wall.rng_states)
    with pytest.raises(ValueError, match="bindings changed"):
        registry.validate_wall_loss_resources(session, wall_loss)

    replacement_scratch = type(nucleation.scratch)(
        **{
            field.name: getattr(nucleation.scratch, field.name)
            for field in fields(nucleation.scratch)
        }
    )
    object.__setattr__(nucleation, "scratch", replacement_scratch)
    with pytest.raises(ValueError, match="record bindings changed"):
        registry.validate_nucleation_resources(session, nucleation)

    assert registry._bindings.keys() == {"wall_loss", "nucleation"}
    assert registry._views["wall_loss"] is wall_loss
    assert registry._views["nucleation"] is nucleation
    assert registry._capacities == capacities
    assert registry._open_step_token is token
    assert session.particles is primaries[0]
    assert session.gas is primaries[1]
    assert session.environment is primaries[2]
    registry.release_open_step(token)


@pytest.mark.warp
def test_registry_allocation_failure_does_not_publish_partial_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a failed allocation leaves all registry bindings unpublished."""
    registry = GPUResourceRegistry(_session())

    def fail_allocation(*_args: object, **_kwargs: object) -> object:
        """Raise a deterministic allocation failure for the local candidate."""
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(registry, "_allocate", fail_allocation)
    with pytest.raises(RuntimeError, match="allocation failed"):
        registry.acquire_wall_loss()

    assert registry._bindings == {}
    assert registry._views == {}


@pytest.mark.warp
@pytest.mark.parametrize("coagulation_first", (False, True))
@pytest.mark.parametrize(
    "failure_point", ("allocation", "registry", "initialize")
)
def test_wall_loss_initialization_failure_is_transactional_and_retryable(
    coagulation_first: bool,
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each staged wall-loss setup failure leaves an unpublished retry path."""
    registry = GPUResourceRegistry(_session())
    if coagulation_first:
        registry.acquire_coagulation(1)

    with monkeypatch.context() as patch:
        if failure_point == "allocation":
            patch.setattr(
                registry,
                "_allocate",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("allocation failed")
                ),
            )
        elif failure_point == "registry":
            patch.setattr(
                gpu_resources,
                "StreamRegistry",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("registry failed")
                ),
            )
        else:
            target = "initialize_process" if coagulation_first else "initialize"
            patch.setattr(
                gpu_resources.StreamRegistry,
                target,
                lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("initialize failed")
                ),
            )
        with pytest.raises(RuntimeError, match=failure_point):
            registry.acquire_wall_loss()

    assert "wall_loss" not in registry._bindings
    assert "wall_loss" not in registry._views
    assert registry._wall_loss_stream_registry is None
    assert registry.acquire_wall_loss().rng_states is not None


@pytest.mark.warp
def test_registry_partial_allocation_failure_does_not_publish_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a later allocation failure cannot publish a partial candidate."""
    registry = GPUResourceRegistry(_session())
    original_allocate = cast(Any, registry._allocate)
    calls = 0

    def fail_after_one(*args: object, **kwargs: object) -> Any:
        """Allocate one local candidate sidecar then fail deterministically."""
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("allocation failed")
        return original_allocate(*args, **kwargs)

    monkeypatch.setattr(registry, "_allocate", fail_after_one)
    with pytest.raises(RuntimeError, match="allocation failed"):
        registry.acquire_condensation()

    assert calls == 2
    assert registry._bindings == {}
    assert registry._views == {}


@pytest.mark.warp
def test_diagnostics_executor_snapshots_closed_operations_in_order() -> None:
    """Test diagnostics copies current resident fields into owned outputs."""
    wp = pytest.importorskip("warp")
    session = _session(boxes=2, particle_count=1, species=2)
    registry = GPUResourceRegistry(session)
    gas_output = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    saturation_output = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    plan = _diagnostics_plan(session, registry, (gas_output, saturation_output))

    ResidentDiagnosticsExecutor().execute(plan)
    wp.synchronize()

    np.testing.assert_array_equal(
        gas_output.numpy(), cast(Any, session.gas).concentration.numpy()
    )
    np.testing.assert_array_equal(
        saturation_output.numpy(),
        cast(Any, session.environment).saturation_ratio.numpy(),
    )


@pytest.mark.warp
@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (0, 0)])
def test_diagnostics_accepts_canonical_empty_outputs_without_dispatch(
    shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test canonical empty diagnostic schemas are successful no-ops."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

    session = _session(boxes=shape[0], particle_count=1, species=shape[1])
    registry = GPUResourceRegistry(session)
    outputs = (
        wp.zeros(shape, dtype=wp.float64, device="cpu"),
        wp.zeros(shape, dtype=wp.float64, device="cpu"),
    )
    launches: list[object] = []
    original_launch: Any = diagnostics.wp.launch

    def record_launch(*args: object, **kwargs: object) -> object:
        """Record nonempty writer dispatches without changing their behavior."""
        launches.append(args[0])
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(diagnostics.wp, "launch", record_launch)

    ResidentDiagnosticsExecutor().execute(
        _diagnostics_plan(session, registry, outputs)
    )

    assert tuple(output.shape for output in outputs) == (shape, shape)
    assert len(launches) == (2 if shape == (1, 0) else 0)


@pytest.mark.warp
def test_diagnostics_rejects_duplicate_operations_before_writing() -> None:
    """Test closed diagnostics reject duplicate operations before a launch."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    first = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    second = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    plan = _diagnostics_plan(session, registry, (first, second))
    duplicate = ResidentDiagnosticsPlan(
        plan.session,
        plan.registry,
        plan.graph,
        plan.schedule,
        plan.node,
        (
            plan.registrations[0],
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT,
                second,
            ),
        ),
    )

    with pytest.raises(ValueError, match="operations must be unique"):
        ResidentDiagnosticsExecutor().execute(duplicate)

    np.testing.assert_array_equal(first.numpy(), np.zeros((1, 1)))
    np.testing.assert_array_equal(second.numpy(), np.zeros((1, 1)))


@pytest.mark.warp
def test_diagnostic_accounting_inputs_may_alias_each_other() -> None:
    """Test read-only accounting inputs may share one caller-owned array."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    shared = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    output = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            output,
            baseline_total_mass=shared,
            source_ledger=shared,
            sink_ledger=shared,
        ),
    )

    registry.validate_diagnostic_registrations(session, registrations)


@pytest.mark.warp
def test_diagnostic_output_cannot_alias_accounting_input() -> None:
    """Test output/input aliasing is rejected during registry preflight."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    shared = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            shared,
            energy_transfer=shared,
        ),
    )

    with pytest.raises(ValueError, match="outputs must not overlap"):
        registry.validate_diagnostic_registrations(session, registrations)


@pytest.mark.warp
def test_diagnostic_input_schema_rejects_before_executor_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test invalid accounting metadata prevents any diagnostics writer launch."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

    session = _session()
    registry = GPUResourceRegistry(session)
    outputs = (
        wp.zeros((1, 1), dtype=wp.float64, device="cpu"),
        wp.zeros((1, 1), dtype=wp.float64, device="cpu"),
    )
    plan = _diagnostics_plan(session, registry, outputs)
    invalid_energy = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    invalid_registration = ResidentDiagnosticRegistration(
        ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
        plan.registrations[4].output,
        energy_transfer=invalid_energy,
    )
    invalid_plan = ResidentDiagnosticsPlan(
        plan.session,
        plan.registry,
        plan.graph,
        plan.schedule,
        plan.node,
        plan.registrations[:4]
        + (invalid_registration,)
        + plan.registrations[5:],
    )
    launches: list[object] = []
    monkeypatch.setattr(
        diagnostics.wp,
        "launch",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    with pytest.raises(
        ValueError, match="accounting input has incompatible schema"
    ):
        ResidentDiagnosticsExecutor().execute(invalid_plan)

    assert not launches


@pytest.mark.warp
def test_particle_number_output_schema_rejects_before_any_diagnostic_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the vector-only particle-number output cannot use a matrix schema."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

    session = _session()
    registry = GPUResourceRegistry(session)
    outputs = (
        wp.zeros((1, 1), dtype=wp.float64, device="cpu"),
        wp.zeros((1, 1), dtype=wp.float64, device="cpu"),
    )
    plan = _diagnostics_plan(session, registry, outputs)
    invalid_number = wp.full((1, 1), 17.0, dtype=wp.float64, device="cpu")
    registrations = (
        plan.registrations[:3]
        + (
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
                invalid_number,
            ),
        )
        + plan.registrations[4:]
    )
    invalid_plan = ResidentDiagnosticsPlan(
        plan.session,
        plan.registry,
        plan.graph,
        plan.schedule,
        plan.node,
        registrations,
    )
    launches: list[object] = []
    monkeypatch.setattr(
        diagnostics.wp,
        "launch",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    with pytest.raises(
        ValueError, match="diagnostic output has incompatible schema"
    ):
        ResidentDiagnosticsExecutor().execute(invalid_plan)

    assert not launches
    np.testing.assert_array_equal(invalid_number.numpy(), [[17.0]])


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("baseline_total_mass", np.nan),
        ("source_ledger", np.nan),
        ("sink_ledger", np.inf),
        ("source_ledger", -1.0),
        ("sink_ledger", -1.0),
    ),
)
def test_diagnostic_accounting_preflight_is_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
) -> None:
    """Test accounting payload values are not scanned or read back."""
    wp = pytest.importorskip("warp")

    session = _session()
    registry = GPUResourceRegistry(session)
    outputs = (
        wp.full((1, 1), 11.0, dtype=wp.float64, device="cpu"),
        wp.full((1, 1), 12.0, dtype=wp.float64, device="cpu"),
    )
    plan = _diagnostics_plan(session, registry, outputs)
    accounting = wp.full((1, 1), value, dtype=wp.float64, device="cpu")
    residual = plan.registrations[5]
    invalid_residual = ResidentDiagnosticRegistration(
        residual.operation,
        residual.output,
        baseline_total_mass=(
            accounting
            if field == "baseline_total_mass"
            else residual.baseline_total_mass
        ),
        source_ledger=(
            accounting if field == "source_ledger" else residual.source_ledger
        ),
        sink_ledger=(
            accounting if field == "sink_ledger" else residual.sink_ledger
        ),
    )
    invalid_plan = ResidentDiagnosticsPlan(
        plan.session,
        plan.registry,
        plan.graph,
        plan.schedule,
        plan.node,
        plan.registrations[:5] + (invalid_residual,),
    )
    launches: list[object] = []
    monkeypatch.setattr(
        wp, "launch", lambda *args, **kwargs: launches.append(args)
    )

    registry.validate_diagnostic_registrations(
        session, invalid_plan.registrations
    )

    assert launches == []
    np.testing.assert_array_equal(outputs[0].numpy(), [[11.0]])
    np.testing.assert_array_equal(outputs[1].numpy(), [[12.0]])


@pytest.mark.warp
@pytest.mark.parametrize("use_sidecar", (False, True))
@pytest.mark.parametrize(
    "field",
    ("energy_transfer", "baseline_total_mass", "source_ledger", "sink_ledger"),
)
def test_diagnostic_accounting_aliases_reject_before_mutation(
    use_sidecar: bool, field: str
) -> None:
    """Test accounting aliases of primaries and pinned sidecars reject safely."""
    wp = pytest.importorskip("warp")
    session = _session()
    registry = GPUResourceRegistry(session)
    sidecar = (
        registry.acquire_nucleation().finalized_demand.precursor_mass_change
    )
    protected = sidecar if use_sidecar else cast(Any, session.gas).concentration
    output = wp.full((1, 1), 13.0, dtype=wp.float64, device="cpu")
    ordinary = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    if field == "energy_transfer":
        registration = ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            output,
            energy_transfer=protected,
        )
    else:
        registration = ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            output,
            baseline_total_mass=(
                protected if field == "baseline_total_mass" else ordinary
            ),
            source_ledger=(protected if field == "source_ledger" else ordinary),
            sink_ledger=(protected if field == "sink_ledger" else ordinary),
        )
    protected_before = protected.numpy().copy()
    masses_before = cast(Any, session.particles).masses.numpy().copy()

    with pytest.raises(ValueError, match="must not alias resident resources"):
        registry.validate_diagnostic_registrations(session, (registration,))

    np.testing.assert_array_equal(output.numpy(), [[13.0]])
    np.testing.assert_array_equal(protected.numpy(), protected_before)
    np.testing.assert_array_equal(
        cast(Any, session.particles).masses.numpy(), masses_before
    )
