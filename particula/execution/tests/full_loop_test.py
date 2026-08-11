"""Regression coverage for complete resident-loop dispatch ordering.

These tests exercise the real resident session, registry, schedule, and loop
coordinator while keeping the direct process adapters bounded and deterministic.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import (
    CONDENSATION_CAPABILITY_MATRIX,
    CONDENSATION_PROCESS,
    Backend,
    CapabilityRequirements,
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
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.gpu_session import (
    ResidentLifecycle,
    ResidentStepGuard,
    setup_resident_session,
)
from particula.execution.process_adapters import (
    ResidentDilutionRequest,
    ResidentNucleationRequest,
    ResidentWallLossRequest,
)
from particula.execution.process_graph import (
    DependencyEdge,
    ProcessNode,
    TimestepPlan,
)
from particula.execution.resident_communication import (
    ResidentCommunicationExecutor,
    ResidentCommunicationRequest,
)
from particula.execution.resident_scheduler import (
    ResidentSimulationRequest,
    ResidentSimulationScheduler,
)
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    SchedulerProfile,
    resolve_timestep_schedule,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
    ResidentStateUpdateExecutor,
)
from particula.execution.tests.gpu_session_test import _cpu_resources
from particula.execution.thermodynamic_updates import (
    ResidentThermodynamicUpdateCoordinator,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.gpu.kernels.wall_loss import NeutralWallLossConfig
from particula.util.constants import GAS_CONSTANT

_NODE_IDS = (
    "communication",
    "volume_evolution",
    "environment_update",
    "gas_update",
    "vapor_pressure_refresh",
    "saturation_refresh",
    "condensation",
    "brownian_coagulation",
    "dilution",
    "wall_loss",
    "nucleation",
    "diagnostics",
)


@pytest.fixture(autouse=True)
def _remove_resolver_schedule_registrations() -> Any:
    """Remove resolver registrations created by each isolated full-loop test."""
    import particula.execution.scheduler as scheduler_module

    schedules = scheduler_module._RESOLVER_SCHEDULES
    initial_count = len(schedules)
    yield
    del schedules[initial_count:]


@dataclass
class _LoopFixture:
    """Track resident-loop state, traces, and expected observations."""

    wp: Any
    session: Any
    registry: Any
    guard: Any
    request: Any
    scheduler: Any
    trace: list[str]
    condensation_windows: list[tuple[str, ...]]
    diagnostics_windows: list[tuple[str, ...]]
    sync_calls: list[str]
    layout_before: dict[str, object]
    gas_snapshot: Any
    saturation_snapshot: Any
    total_mass: Any
    particle_number: Any
    latent_energy: Any
    conservation_residual: Any
    expected_temperature: np.ndarray
    expected_pressure: np.ndarray
    expected_vapor_pressure: np.ndarray
    expected_saturation_ratio: np.ndarray
    initial_inventory: np.ndarray


def _scheduler_module() -> Any:
    """Import the Warp-dependent scheduler module only when Warp is present."""
    pytest.importorskip("warp")
    import particula.execution.resident_scheduler as resident_scheduler

    return resident_scheduler


def _node(node_id: str) -> ProcessNode:
    """Build one exact catalogue node for the resolved resident graph."""
    from particula.execution import process_graph

    schema = next(
        item
        for item in process_graph._NODE_CATALOGUE
        if item.node_id == node_id
    )
    if node_id == "condensation":
        requirements = next(
            declaration.requirements
            for declaration in CONDENSATION_CAPABILITY_MATRIX.declarations
            if declaration.process == CONDENSATION_PROCESS
        )
    else:
        requirements = CapabilityRequirements(frozenset())
    return ProcessNode(
        schema.node_id,
        schema.kind,
        schema.process,
        requirements,
        schema.resources,
        schema.invalidates,
    )


def _array_signature(value: Any) -> tuple[int, tuple[int, ...], str, str, Any]:
    """Capture a resident array's immutable schema without copying payloads."""
    return (
        id(value),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        getattr(value, "capacity", None),
    )


def _layout_signature(fixture: _LoopFixture) -> dict[str, object]:
    """Capture the stable identities and schemas used by the regression rows."""
    session = fixture.session
    registry = fixture.registry
    return {
        "session": (id(session), session.dimensions),
        "particles": (
            id(session.particles),
            _array_signature(session.particles.masses),
            _array_signature(session.particles.concentration),
            _array_signature(session.particles.charge),
            _array_signature(session.particles.density),
            _array_signature(session.particles.volume),
        ),
        "gas": (
            id(session.gas),
            _array_signature(session.gas.molar_mass),
            _array_signature(session.gas.concentration),
            _array_signature(session.gas.vapor_pressure),
            _array_signature(session.gas.partitioning),
        ),
        "environment": (
            id(session.environment),
            _array_signature(session.environment.temperature),
            _array_signature(session.environment.pressure),
            _array_signature(session.environment.saturation_ratio),
        ),
        "registry": id(registry),
        "communication": (
            id(fixture.request.communication),
            id(fixture.request.communication.resources),
            id(fixture.request.communication.resources.buffers),
            id(fixture.request.communication.resources.execution_state),
        ),
        "condensation": id(fixture.request.condensation),
        "coagulation": (
            id(fixture.request.coagulation),
            _array_signature(
                fixture.request.coagulation.resources.collision_pairs
            ),
            _array_signature(
                fixture.request.coagulation.resources.n_collisions
            ),
            _array_signature(fixture.request.coagulation.resources.rng_states),
        ),
        "dilution": id(fixture.request.dilution),
        "wall_loss": (
            id(fixture.request.wall_loss),
            _array_signature(fixture.request.wall_loss.resources.rng_states),
        ),
        "nucleation": (
            id(fixture.request.nucleation),
            id(fixture.request.nucleation.resources.scratch),
            id(fixture.request.nucleation.resources.finalized_demand),
            id(fixture.request.nucleation.resources.diagnostics),
            id(fixture.request.nucleation.resources.exhaustion),
        ),
        "diagnostics": (
            id(fixture.request.diagnostics),
            _array_signature(fixture.gas_snapshot),
            _array_signature(fixture.saturation_snapshot),
            _array_signature(fixture.total_mass),
            _array_signature(fixture.particle_number),
            _array_signature(fixture.latent_energy),
            _array_signature(fixture.conservation_residual),
        ),
    }


def _expected_inventory(session: Any) -> np.ndarray:
    """Compute the closed-system particle-plus-gas inventory from NumPy data."""
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    volume = particles.volume.numpy()
    particle_mass = particles.masses.numpy()
    particle_concentration = particles.concentration.numpy()
    gas_concentration = gas.concentration.numpy()
    return volume[:, None] * (
        np.sum(particle_mass * particle_concentration[:, :, None], axis=1)
        + gas_concentration
    )


def _communication_configuration(
    wp: Any, family: CommunicationTransportMode
) -> CommunicationConfiguration:
    """Build one closed zero-rate communication configuration for a test row."""
    return CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            family,
            2,
            wp.array([0, 1], dtype=wp.int32, device="cpu"),
            wp.array([1, 2], dtype=wp.int32, device="cpu"),
            wp.array([1, 1], dtype=wp.int32, device="cpu"),
            wp.array([0.0, 0.0], dtype=wp.float64, device="cpu"),
        ),
        PrescribedVolumeUpdate(None),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )


def _build_resident_graph() -> tuple[Any, Any, dict[str, Any]]:
    """Resolve the canonical twelve-node resident graph and schedule."""
    nodes = tuple(_node(node_id) for node_id in _NODE_IDS)
    dependencies = (
        (DependencyEdge("communication", "volume_evolution"),)
        + tuple(
            DependencyEdge("volume_evolution", node_id)
            for node_id in _NODE_IDS
            if node_id not in {"communication", "volume_evolution"}
        )
        + tuple(
            DependencyEdge(node_id, "diagnostics")
            for node_id in (
                "condensation",
                "brownian_coagulation",
                "dilution",
                "wall_loss",
                "nucleation",
            )
        )
    )
    schedule = resolve_timestep_schedule(
        TimestepPlan(nodes, dependencies),
        EnabledNodeSelection(frozenset(_NODE_IDS)),
        SchedulerProfile(
            NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION
        ),
    )
    graph = cast(Any, schedule.source_graph)
    assert graph is not None
    by_id = {node.node_id: node for node in graph.nodes}
    return graph, schedule, by_id


def _build_condensation_state(
    session: Any,
    thermodynamics: ThermodynamicsConfig,
    scratch_buffers: Any,
) -> Any:
    """Build one exact resident Warp condensation state for a no-op row."""
    configuration = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        False,
        CondensationActivityMode.IDEAL,
        CondensationSurfaceMode.STATIC,
    )
    state = WarpCondensationState(
        CondensationExecutionConfig(configuration),
        session.particles,
        session.gas,
        session.environment,
        thermodynamics,
        scratch_buffers=scratch_buffers,
    )
    return WarpCondensationExecutionState(state, 0.0)


def _build_coagulation_state(session: Any, registry: Any) -> Any:
    """Build one exact resident Brownian execution state for a no-op row."""
    resources = registry.acquire_coagulation(1)
    request = WarpBrownianCoagulationExecutionState(
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            session.particles,
            None,
            None,
            0.0,
            collision_pairs=resources.collision_pairs,
            n_collisions=resources.n_collisions,
            rng_states=resources.rng_states,
            initialize_rng=False,
            environment=session.environment,
        )
    )
    return ResidentBrownianCoagulationExecutionState(
        request, session, registry, resources
    )


def _build_diagnostics_plan(
    session: Any,
    registry: Any,
    graph: Any,
    schedule: Any,
    by_id: dict[str, Any],
    wp: Any,
) -> tuple[ResidentDiagnosticsPlan, dict[str, Any]]:
    """Build the closed diagnostics plan and its caller-owned output arrays."""
    outputs = {
        "gas_snapshot": wp.zeros((3, 1), dtype=wp.float64, device="cpu"),
        "saturation_snapshot": wp.zeros((3, 1), dtype=wp.float64, device="cpu"),
        "total_mass": wp.zeros((3, 1), dtype=wp.float64, device="cpu"),
        "particle_number": wp.zeros(3, dtype=wp.float64, device="cpu"),
        "latent_energy": wp.zeros((3, 1), dtype=wp.float64, device="cpu"),
        "conservation_residual": wp.zeros(
            (3, 1), dtype=wp.float64, device="cpu"
        ),
    }
    latent_transfer = wp.zeros((3, 1), dtype=wp.float64, device="cpu")
    baseline_total_mass = wp.zeros((3, 1), dtype=wp.float64, device="cpu")
    source_ledger = wp.zeros((3, 1), dtype=wp.float64, device="cpu")
    sink_ledger = wp.zeros((3, 1), dtype=wp.float64, device="cpu")
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT,
            outputs["gas_snapshot"],
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT,
            outputs["saturation_snapshot"],
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS,
            outputs["total_mass"],
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
            outputs["particle_number"],
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            outputs["latent_energy"],
            energy_transfer=latent_transfer,
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            outputs["conservation_residual"],
            baseline_total_mass=baseline_total_mass,
            source_ledger=source_ledger,
            sink_ledger=sink_ledger,
        ),
    )
    return (
        ResidentDiagnosticsPlan(
            session,
            registry,
            graph,
            schedule,
            by_id["diagnostics"],
            registrations,
        ),
        outputs,
    )


def _build_loop_fixture(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
    *,
    wall_loss_failure: bool = False,
) -> _LoopFixture:
    """Construct the real resident loop with deterministic spies and snapshots."""
    wp = pytest.importorskip("warp")
    import particula.execution.resident_scheduler as resident_scheduler
    import particula.execution.thermodynamic_updates as thermodynamic_updates
    from particula.gpu import conversion

    particles, gas, environment = _cpu_resources(3, 2, 1)
    particles.masses = np.array(
        [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], dtype=np.float64
    )
    particles.concentration = np.array(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float64
    )
    particles.charge = np.zeros((3, 2), dtype=np.float64)
    particles.density = np.array([1000.0], dtype=np.float64)
    particles.volume = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gas.molar_mass = np.array([0.018], dtype=np.float64)
    gas.concentration = np.array([[0.25], [0.5], [0.75]], dtype=np.float64)
    gas.partitioning = np.array([True], dtype=np.bool_)
    environment.temperature = np.array([295.0, 296.0, 297.0], dtype=np.float64)
    environment.pressure = np.array(
        [101325.0, 100800.0, 100000.0], dtype=np.float64
    )
    environment.saturation_ratio = np.ones((3, 1), dtype=np.float64)

    upload_calls: list[tuple[str, str]] = []
    original_particle_upload = conversion.to_warp_particle_data
    original_gas_upload = conversion.to_warp_gas_data
    original_environment_upload = conversion.to_warp_environment_data

    def particle_upload(value: object, *, device: str) -> object:
        upload_calls.append(("particles", device))
        return cast(Any, original_particle_upload)(value, device=device)

    def gas_upload(value: object, *, device: str) -> object:
        upload_calls.append(("gas", device))
        return cast(Any, original_gas_upload)(value, device=device)

    def environment_upload(value: object, *, device: str) -> object:
        upload_calls.append(("environment", device))
        return cast(Any, original_environment_upload)(value, device=device)

    monkeypatch.setattr(conversion, "to_warp_particle_data", particle_upload)
    monkeypatch.setattr(conversion, "to_warp_gas_data", gas_upload)
    monkeypatch.setattr(
        conversion, "to_warp_environment_data", environment_upload
    )

    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, "cpu"),
    )
    assert upload_calls == [
        ("particles", "cpu"),
        ("gas", "cpu"),
        ("environment", "cpu"),
    ]

    def reject_late_upload(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resident dispatch must not upload CPU state")

    monkeypatch.setattr(conversion, "to_warp_particle_data", reject_late_upload)
    monkeypatch.setattr(conversion, "to_warp_gas_data", reject_late_upload)
    monkeypatch.setattr(
        conversion, "to_warp_environment_data", reject_late_upload
    )

    registry = GPUResourceRegistry(session)
    graph, schedule, by_id = _build_resident_graph()

    communication = registry.acquire_communication(
        _communication_configuration(wp, communication_family)
    )
    condensation_resources = registry.acquire_condensation()
    coagulation = _build_coagulation_state(session, registry)
    wall_loss_resources = registry.acquire_wall_loss()
    nucleation_resources = registry.acquire_nucleation()

    thermodynamics = ThermodynamicsConfig(
        modes=wp.zeros(1, dtype=wp.int32, device="cpu"),
        parameters=wp.array(
            [[800.0, 0.0, 0.0, 0.0]], dtype=wp.float64, device="cpu"
        ),
        molar_mass_reference=wp.array(
            cast(Any, session.gas).molar_mass.numpy(),
            dtype=wp.float64,
            device="cpu",
        ),
    )
    condensation_state = _build_condensation_state(
        session, thermodynamics, condensation_resources.scratch_buffers
    )
    environment_update = ResidentEnvironmentUpdateRequest(
        session,
        registry,
        graph,
        by_id["environment_update"],
        wp.array(environment.temperature, dtype=wp.float64, device="cpu"),
        wp.array(environment.pressure, dtype=wp.float64, device="cpu"),
    )
    gas_update = ResidentGasUpdateRequest(
        session,
        registry,
        graph,
        by_id["gas_update"],
        wp.array(gas.concentration, dtype=wp.float64, device="cpu"),
    )
    communication_request = ResidentCommunicationRequest(
        session,
        registry,
        graph,
        communication,
        by_id["communication"],
        by_id["volume_evolution"],
        0.0,
    )
    dilution_request = ResidentDilutionRequest(session, registry, 0.0, 0.0)
    wall_loss_request = ResidentWallLossRequest(
        session,
        registry,
        wall_loss_resources,
        NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
        0.0,
        enabled_box_indices=(0, 1, 2),
    )
    nucleation_request = ResidentNucleationRequest(
        session,
        registry,
        nucleation_resources,
        object(),
        0.0,
        object(),
    )
    diagnostics_plan, outputs = _build_diagnostics_plan(
        session, registry, graph, schedule, by_id, wp
    )

    request = ResidentSimulationRequest(
        session,
        registry,
        ResidentStepGuard(session, registry),
        graph,
        schedule,
        thermodynamics,
        condensation_state,
        coagulation,
        dilution_request,
        wall_loss_request,
        nucleation_request,
        diagnostics_plan,
        environment_update,
        gas_update,
        communication_request,
    )
    scheduler = ResidentSimulationScheduler(request)

    trace: list[str] = []
    condensation_windows: list[tuple[str, ...]] = []
    diagnostics_windows: list[tuple[str, ...]] = []
    refresh_events: list[str] = []
    sync_calls: list[str] = []

    class SpyStateUpdates:
        """Record state-update dispatches while preserving actual mutation."""

        def __init__(self) -> None:
            self._inner = ResidentStateUpdateExecutor()

        def execute(self, item: object) -> object:
            trace.append(cast(Any, item).node.node_id)
            return cast(Any, self._inner).execute(item)

    class SpyCommunication:
        """Record resident communication dispatches and preserve execution."""

        def __init__(self, request: object) -> None:
            self._inner = ResidentCommunicationExecutor(cast(Any, request))

        def execute_communication(self) -> object:
            trace.append("communication")
            return self._inner.execute_communication()

        def execute_volume_evolution(self) -> object | None:
            trace.append("volume_evolution")
            return self._inner.execute_volume_evolution()

        def validate(self, *args: object, **kwargs: object) -> None:
            """Delegate resident-request validation unchanged."""
            cast(Any, self._inner).validate(*args, **kwargs)

    class SpyThermalCoordinator:
        """Record thermodynamic coordinator use and keep real freshness logic."""

        def __init__(self, request: object) -> None:
            self._inner = ResidentThermodynamicUpdateCoordinator(
                cast(Any, request)
            )

        def record_completed(self, node: object) -> None:
            node_id = cast(Any, node).node_id
            if node_id not in {
                "communication",
                "volume_evolution",
                "environment_update",
                "gas_update",
            }:
                trace.append(node_id)
            self._inner.record_completed(cast(Any, node))

        def execute_consumer(self, node: object, callback: Any) -> object:
            return self._inner.execute_consumer(cast(Any, node), callback)

        def _refresh_saturation_ratio(self) -> None:
            return self._inner._refresh_saturation_ratio()

    class SpyDiagnostics:
        """Record the diagnostics consumer and preserve output writes."""

        def __init__(self) -> None:
            self._inner = ResidentDiagnosticsExecutor()

        def execute(self, plan: object) -> None:
            trace.append("diagnostics")
            self._inner.execute(cast(Any, plan))
            wp.synchronize()
            diagnostics_windows.append(tuple(refresh_events))
            refresh_events.clear()
            assert diagnostics_windows[-1] == ("saturation_refresh",)
            _assert_observation_state(
                session,
                outputs,
                expected_temperature=environment.temperature,
                expected_pressure=environment.pressure,
                expected_vapor_pressure=expected_vapor_pressure,
                expected_saturation_ratio=expected_saturation_ratio,
            )

        def validate(self, *args: object, **kwargs: object) -> None:
            """Delegate resident diagnostics validation unchanged."""
            cast(Any, self._inner).validate(*args, **kwargs)

    class NoOpCondensation:
        """Record a condensation completion without native kernel launch."""

        def execute(self, _state: object) -> None:
            trace.append("condensation")
            wp.synchronize()
            condensation_windows.append(tuple(refresh_events))
            refresh_events.clear()
            assert condensation_windows[-1] == (
                "vapor_pressure_refresh",
                "saturation_refresh",
            )
            _assert_observation_state(
                session,
                outputs,
                expected_temperature=environment.temperature,
                expected_pressure=environment.pressure,
                expected_vapor_pressure=np.full(
                    (3, 1), 800.0, dtype=np.float64
                ),
                expected_saturation_ratio=_expected_saturation_ratio(session),
            )

    class NoOpCoagulation:
        """Record the resident coagulation completion without mutation."""

        def execute(self, _state: object) -> None:
            return None

    class NoOpDilution:
        """Record the resident dilution completion without mutation."""

        def execute(self, _request: object) -> None:
            return None

    class NoOpWallLoss:
        """Record wall-loss completion or inject the writer-failure row."""

        def execute(self, _request: object) -> None:
            if wall_loss_failure:
                raise RuntimeError("wall-loss writer failed")

    class NoOpNucleation:
        """Record ordinary nucleation completion without mutation."""

        def execute(self, _request: object) -> None:
            return None

    def spy_refresh_vapor_pressure(*args: object, **kwargs: object) -> object:
        """Record the vapor-pressure refresh boundary before delegating."""
        refresh_events.append("vapor_pressure_refresh")
        return cast(Any, original_refresh_vapor_pressure)(*args, **kwargs)

    def spy_refresh_saturation_ratio(self: object) -> None:
        """Record the saturation-ratio refresh boundary before delegating."""
        refresh_events.append("saturation_refresh")
        cast(Any, original_refresh_saturation_ratio)(self)

    def spy_sync() -> None:
        """Count explicit host synchronization boundaries in the row."""
        sync_calls.append("sync")
        original_sync()

    original_refresh_vapor_pressure = (
        thermodynamic_updates.refresh_vapor_pressure_gpu
    )
    original_refresh_saturation_ratio = (
        ResidentThermodynamicUpdateCoordinator._refresh_saturation_ratio
    )
    original_sync = wp.synchronize

    monkeypatch.setattr(
        resident_scheduler, "ResidentStateUpdateExecutor", SpyStateUpdates
    )
    monkeypatch.setattr(
        resident_scheduler, "ResidentCommunicationExecutor", SpyCommunication
    )
    monkeypatch.setattr(
        resident_scheduler,
        "ResidentThermodynamicUpdateCoordinator",
        SpyThermalCoordinator,
    )
    monkeypatch.setattr(
        resident_scheduler, "ResidentDiagnosticsExecutor", SpyDiagnostics
    )
    monkeypatch.setattr(
        resident_scheduler,
        "WarpCondensationExecutionAdapter",
        NoOpCondensation,
    )
    monkeypatch.setattr(
        resident_scheduler,
        "ResidentBrownianCoagulationExecutionAdapter",
        NoOpCoagulation,
    )
    monkeypatch.setattr(
        resident_scheduler, "ResidentDilutionAdapter", NoOpDilution
    )
    monkeypatch.setattr(
        resident_scheduler, "ResidentWallLossAdapter", NoOpWallLoss
    )
    monkeypatch.setattr(
        resident_scheduler, "ResidentNucleationAdapter", NoOpNucleation
    )
    monkeypatch.setattr(
        thermodynamic_updates,
        "refresh_vapor_pressure_gpu",
        spy_refresh_vapor_pressure,
    )
    monkeypatch.setattr(
        ResidentThermodynamicUpdateCoordinator,
        "_refresh_saturation_ratio",
        spy_refresh_saturation_ratio,
    )
    monkeypatch.setattr(wp, "synchronize", spy_sync)

    expected_vapor_pressure = np.full((3, 1), 800.0, dtype=np.float64)
    expected_saturation_ratio = _expected_saturation_ratio(session)
    initial_inventory = _expected_inventory(session)

    fixture = _LoopFixture(
        wp=wp,
        session=session,
        registry=registry,
        guard=request.guard,
        request=request,
        scheduler=scheduler,
        trace=trace,
        condensation_windows=condensation_windows,
        diagnostics_windows=diagnostics_windows,
        sync_calls=sync_calls,
        layout_before={},
        gas_snapshot=outputs["gas_snapshot"],
        saturation_snapshot=outputs["saturation_snapshot"],
        total_mass=outputs["total_mass"],
        particle_number=outputs["particle_number"],
        latent_energy=outputs["latent_energy"],
        conservation_residual=outputs["conservation_residual"],
        expected_temperature=environment.temperature.copy(),
        expected_pressure=environment.pressure.copy(),
        expected_vapor_pressure=expected_vapor_pressure,
        expected_saturation_ratio=expected_saturation_ratio,
        initial_inventory=initial_inventory,
    )
    fixture.layout_before = _layout_signature(fixture)
    return fixture


def _expected_saturation_ratio(session: Any) -> np.ndarray:
    """Compute the current saturation ratio with an independent NumPy oracle."""
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    vapor_pressure = np.full((3, 1), 800.0, dtype=np.float64)
    return (
        gas.concentration.numpy()
        * GAS_CONSTANT
        * environment.temperature.numpy()[:, None]
        / (gas.molar_mass.numpy()[None, :] * vapor_pressure)
    )


def _assert_observation_state(
    session: Any,
    outputs: dict[str, Any],
    *,
    expected_temperature: np.ndarray,
    expected_pressure: np.ndarray,
    expected_vapor_pressure: np.ndarray,
    expected_saturation_ratio: np.ndarray,
) -> None:
    """Assert the resident derived state matches the independent NumPy oracle."""
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    npt.assert_allclose(
        environment.temperature.numpy(),
        expected_temperature,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        environment.pressure.numpy(), expected_pressure, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        gas.vapor_pressure.numpy(),
        expected_vapor_pressure,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        environment.saturation_ratio.numpy(),
        expected_saturation_ratio,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_equal(
        outputs["saturation_snapshot"].shape, expected_saturation_ratio.shape
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    [
        CommunicationTransportMode.GAS,
        CommunicationTransportMode.PARTICLES,
    ],
    ids=("gas-map", "particles-map"),
)
def test_complete_loop_repeats_canonical_order_without_bulk_transfer(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Two resident rows preserve identity and canonical node order."""
    fixture = _build_loop_fixture(monkeypatch, communication_family)

    fixture.scheduler.execute(0.0)
    fixture.scheduler.execute(0.0)

    ordinary_nodes = tuple(
        node_id
        for node_id in fixture.request.schedule.ordered_node_ids
        if node_id not in {"vapor_pressure_refresh", "saturation_refresh"}
    )
    assert fixture.trace == list(ordinary_nodes * 2)
    assert fixture.condensation_windows == [
        ("vapor_pressure_refresh", "saturation_refresh"),
        ("vapor_pressure_refresh", "saturation_refresh"),
    ]
    assert fixture.diagnostics_windows == [
        ("saturation_refresh",),
        ("saturation_refresh",),
    ]
    assert fixture.guard.completed_steps == 2
    assert fixture.session.lifecycle is ResidentLifecycle.ACTIVE
    assert len(fixture.sync_calls) == 4
    assert _layout_signature(fixture) == fixture.layout_before


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_closed_system_inventory_remains_conserved_after_two_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed GAS-map execution preserves concentration-weighted inventory."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)

    fixture.scheduler.execute(0.0)
    fixture.scheduler.execute(0.0)

    wp = fixture.wp
    wp.synchronize()
    npt.assert_allclose(
        _expected_inventory(fixture.session),
        fixture.initial_inventory,
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.warp
def test_wall_loss_failure_faults_session_and_blocks_later_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer failure faults the session and rejects later lifecycle reuse."""
    fixture = _build_loop_fixture(
        monkeypatch,
        CommunicationTransportMode.GAS,
        wall_loss_failure=True,
    )

    with pytest.raises(RuntimeError, match="wall-loss writer failed"):
        fixture.scheduler.execute(0.0)

    fixture.guard.assert_step_closed()
    assert fixture.session.lifecycle is ResidentLifecycle.FAULTED
    with pytest.raises(ValueError, match="ACTIVE"):
        fixture.scheduler.execute(0.0)
