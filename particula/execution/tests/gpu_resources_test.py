"""Tests for concrete-only fixed-shape GPU resource acquisition."""

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("warp")

import particula.execution.gpu_resources as gpu_resources
from particula.execution import Backend, Device
from particula.execution.diagnostics import (
    ResidentDiagnosticOperation,
    ResidentDiagnosticRegistration,
    ResidentDiagnosticsExecutor,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_resources import (
    GPUResourceRegistry,
    ManifestEntry,
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
        return original_launch(*args, **kwargs)

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
    entry = ManifestEntry("pairs", "coagulation", object(), "bc2")
    registry._session = _session()
    with pytest.raises(ValueError, match="collision capacity"):
        registry._shape(entry)


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
    )
    assert all(
        entry.role and entry.dtype
        for manifest in manifests
        for entry in manifest.entries
    )


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
    original_launch = diagnostics.wp.launch

    def record_launch(*args: object, **kwargs: object) -> object:
        """Record nonempty writer dispatches without changing their behavior."""
        launches.append(args[0])
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(diagnostics.wp, "launch", record_launch)

    ResidentDiagnosticsExecutor().execute(
        _diagnostics_plan(session, registry, outputs)
    )

    assert tuple(output.shape for output in outputs) == (shape, shape)
    assert len(launches) == (1 if shape == (1, 0) else 0)


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
    ("field", "value", "match"),
    (
        ("baseline_total_mass", np.nan, "must be finite"),
        ("source_ledger", np.nan, "finite and nonnegative"),
        ("sink_ledger", np.inf, "finite and nonnegative"),
        ("source_ledger", -1.0, "finite and nonnegative"),
        ("sink_ledger", -1.0, "finite and nonnegative"),
    ),
)
def test_diagnostic_accounting_values_reject_before_writer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    match: str,
) -> None:
    """Test malformed ledger values preserve resident and output sentinels."""
    wp = pytest.importorskip("warp")
    import particula.execution.diagnostics as diagnostics

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
    writers = {
        diagnostics._copy_snapshot,
        diagnostics._total_species_mass,
        diagnostics._particle_number,
        diagnostics._conservation_residual,
    }
    launches: list[object] = []
    original_launch = wp.launch

    def record_writers(
        kernel: object, *args: object, **kwargs: object
    ) -> object:
        """Record diagnostics writers while allowing read-only validation scans."""
        if kernel in writers:
            launches.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_writers)
    before_masses = cast(Any, session.particles).masses.numpy().copy()

    with pytest.raises(ValueError, match=match):
        ResidentDiagnosticsExecutor().execute(invalid_plan)

    assert launches == []
    np.testing.assert_array_equal(outputs[0].numpy(), [[11.0]])
    np.testing.assert_array_equal(outputs[1].numpy(), [[12.0]])
    np.testing.assert_array_equal(
        cast(Any, session.particles).masses.numpy(), before_masses
    )


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
