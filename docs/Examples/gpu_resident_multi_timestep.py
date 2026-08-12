"""Run a complete multi-box GPU-resident scheduler example.

Warp and the concrete resident seams are optional at import time.  When Warp is
enabled, this example uploads primary CPU state once during setup, stages
per-request gas and environment forcing, runs two resident timesteps, observes
caller-owned diagnostics, then manually checkpoints and restarts on the same
device.  It is a bounded resident-loop example, not a CPU fallback,
automatic restart facility, direct-kernel orchestration, graph-capture example,
or performance claim.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"
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


@dataclass
class ExampleRun:
    """Retain address-free observations from the resident-loop example.

    Attributes:
        output: Deterministic status and ownership statements.
        session: Source resident session, or ``None`` without Warp.
        registry: Source registry, or ``None`` without Warp.
        guard: Source closed step guard, or ``None`` without Warp.
        checkpoint: Active source checkpoint created before restart.
        restarted: Fresh restarted ``(session, registry, guard)`` binding.
        terminal_checkpoint: Cached checkpoint from source finalization.
        gas_snapshot: Caller-owned gas diagnostic with shape ``(3, 1)``.
        saturation_snapshot: Caller-owned saturation diagnostic with shape
            ``(3, 1)``.
        initial_total_mass: Initial particle-plus-gas extensive inventory.
        conservation_residual: Final residual measured against that inventory.
        restart_gas_before_physics: Restarted gas observed before dispatch.
        restart_temperature_before_physics: Restarted temperature observed
            before dispatch.
        source_steps: Completed source scheduler steps.
        restarted_steps: Completed restarted scheduler steps.
    """

    output: list[str]
    session: Any | None = None
    registry: Any | None = None
    guard: Any | None = None
    checkpoint: Any | None = None
    restarted: tuple[Any, Any, Any] | None = None
    terminal_checkpoint: Any | None = None
    gas_snapshot: np.ndarray | None = None
    saturation_snapshot: np.ndarray | None = None
    initial_total_mass: np.ndarray | None = None
    conservation_residual: np.ndarray | None = None
    restart_gas_before_physics: np.ndarray | None = None
    restart_temperature_before_physics: np.ndarray | None = None
    source_steps: int = 0
    restarted_steps: int = 0


def _warp_enabled() -> bool:
    """Return whether optional Warp is explicitly enabled and importable."""
    if os.getenv(_FORCE_NO_WARP_ENV) == "1":
        return False
    try:
        importlib.import_module("warp")
    except ModuleNotFoundError as error:
        if error.name == "warp":
            return False
        raise
    return True


def _disabled_output() -> list[str]:
    """Return deterministic guidance for the intentional no-Warp path."""
    return [
        "Canonical path: docs/Examples/gpu_resident_multi_timestep.py",
        "Warp is unavailable or disabled; install warp-lang or enable Warp.",
        "No CPU fallback ran; no fixture, upload, diagnostics, or restart ran.",
    ]


def _load_enabled_runtime() -> SimpleNamespace:
    """Load Warp and concrete resident seams only for enabled execution."""
    names = (
        "warp",
        "particula.execution",
        "particula.execution.availability",
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
        "particula.execution.process_graph",
        "particula.execution.scheduler",
        "particula.execution.resident_scheduler",
        "particula.execution.diagnostics",
        "particula.execution.state_updates",
        "particula.execution.process_adapters",
        "particula.execution.communication",
        "particula.execution.resident_communication",
        "particula.execution.adapters.condensation",
        "particula.execution.adapters.coagulation",
        "particula.gpu.kernels.thermodynamics",
        "particula.gpu.kernels.wall_loss",
        "particula.gpu.kernels.nucleation",
    )
    loaded = {
        name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names
    }
    return SimpleNamespace(**loaded)


def _build_cpu_state() -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build a deterministic float64 three-box fixture for one source upload."""
    return (
        ParticleData(
            masses=np.array(
                [
                    [[1.0e-18], [2.0e-18]],
                    [[3.0e-18], [4.0e-18]],
                    [[5.0e-18], [6.0e-18]],
                ],
                dtype=np.float64,
            ),
            concentration=np.array(
                [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float64
            ),
            charge=np.zeros((3, 2), dtype=np.float64),
            density=np.array([1000.0], dtype=np.float64),
            volume=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
        GasData(
            name=["water"],
            molar_mass=np.array([0.018], dtype=np.float64),
            concentration=np.array([[0.25], [0.5], [0.75]], dtype=np.float64),
            partitioning=np.array([True], dtype=np.bool_),
        ),
        EnvironmentData(
            temperature=np.array([295.0, 296.0, 297.0], dtype=np.float64),
            pressure=np.array([101325.0, 100800.0, 100000.0], dtype=np.float64),
            saturation_ratio=np.ones((3, 1), dtype=np.float64),
        ),
    )


def _resolved_graph(
    runtime: SimpleNamespace,
) -> tuple[Any, Any, dict[str, Any]]:
    """Create the resolver-produced canonical twelve-node graph and schedule."""
    execution = runtime.execution
    graph_module = runtime.process_graph
    resource = graph_module.ResourceRequirement
    invalidated = graph_module.InvalidatedState
    schemas = {
        "communication": (
            graph_module.NodeKind.COMMUNICATION,
            None,
            frozenset(
                {resource.PARTICLES, resource.GAS, resource.PROCESS_SIDECARS}
            ),
            frozenset({invalidated.SATURATION_RATIO}),
        ),
        "volume_evolution": (
            graph_module.NodeKind.VOLUME_EVOLUTION,
            None,
            frozenset(
                {resource.PARTICLES, resource.GAS, resource.PROCESS_SIDECARS}
            ),
            frozenset({invalidated.SATURATION_RATIO}),
        ),
        "environment_update": (
            graph_module.NodeKind.ENVIRONMENT_UPDATE,
            None,
            frozenset({resource.ENVIRONMENT}),
            frozenset(
                {invalidated.VAPOR_PRESSURE, invalidated.SATURATION_RATIO}
            ),
        ),
        "gas_update": (
            graph_module.NodeKind.GAS_UPDATE,
            None,
            frozenset({resource.GAS}),
            frozenset({invalidated.SATURATION_RATIO}),
        ),
        "vapor_pressure_refresh": (
            graph_module.NodeKind.VAPOR_PRESSURE_REFRESH,
            None,
            frozenset(
                {resource.GAS, resource.ENVIRONMENT, resource.THERMODYNAMICS}
            ),
            frozenset(),
        ),
        "saturation_refresh": (
            graph_module.NodeKind.SATURATION_REFRESH,
            None,
            frozenset(
                {resource.GAS, resource.ENVIRONMENT, resource.THERMODYNAMICS}
            ),
            frozenset(),
        ),
        "condensation": (
            graph_module.NodeKind.PROCESS,
            execution.CONDENSATION_PROCESS,
            frozenset(
                {
                    resource.PARTICLES,
                    resource.GAS,
                    resource.ENVIRONMENT,
                    resource.THERMODYNAMICS,
                    resource.PROCESS_SIDECARS,
                }
            ),
            frozenset({invalidated.SATURATION_RATIO}),
        ),
        "brownian_coagulation": (
            graph_module.NodeKind.PROCESS,
            execution.Process("brownian_coagulation"),
            frozenset(
                {
                    resource.PARTICLES,
                    resource.ENVIRONMENT,
                    resource.PROCESS_SIDECARS,
                }
            ),
            frozenset(),
        ),
        "dilution": (
            graph_module.NodeKind.PROCESS,
            execution.Process("dilution"),
            frozenset({resource.PARTICLES, resource.GAS}),
            frozenset(),
        ),
        "wall_loss": (
            graph_module.NodeKind.PROCESS,
            execution.Process("wall_loss"),
            frozenset(
                {
                    resource.PARTICLES,
                    resource.ENVIRONMENT,
                    resource.PROCESS_SIDECARS,
                }
            ),
            frozenset(),
        ),
        "nucleation": (
            graph_module.NodeKind.PROCESS,
            execution.Process("nucleation"),
            frozenset(
                {
                    resource.PARTICLES,
                    resource.GAS,
                    resource.ENVIRONMENT,
                    resource.PROCESS_SIDECARS,
                }
            ),
            frozenset({invalidated.SATURATION_RATIO}),
        ),
        "diagnostics": (
            graph_module.NodeKind.DIAGNOSTIC,
            None,
            frozenset(
                {
                    resource.PARTICLES,
                    resource.GAS,
                    resource.ENVIRONMENT,
                    resource.THERMODYNAMICS,
                    resource.DIAGNOSTICS,
                }
            ),
            frozenset(),
        ),
    }
    condensation_requirements = next(
        item.requirements
        for item in execution.CONDENSATION_CAPABILITY_MATRIX.declarations
        if item.process == execution.CONDENSATION_PROCESS
    )
    nodes = tuple(
        graph_module.ProcessNode(
            node_id,
            schemas[node_id][0],
            schemas[node_id][1],
            condensation_requirements
            if node_id == "condensation"
            else execution.CapabilityRequirements(frozenset()),
            schemas[node_id][2],
            schemas[node_id][3],
        )
        for node_id in _NODE_IDS
    )
    dependencies = (
        (graph_module.DependencyEdge("communication", "volume_evolution"),)
        + tuple(
            graph_module.DependencyEdge("volume_evolution", node_id)
            for node_id in _NODE_IDS
            if node_id not in {"communication", "volume_evolution"}
        )
        + tuple(
            graph_module.DependencyEdge(node_id, "diagnostics")
            for node_id in (
                "condensation",
                "brownian_coagulation",
                "dilution",
                "wall_loss",
                "nucleation",
            )
        )
    )
    schedule = runtime.scheduler.resolve_timestep_schedule(
        graph_module.TimestepPlan(nodes, dependencies),
        runtime.scheduler.EnabledNodeSelection(frozenset(_NODE_IDS)),
        runtime.scheduler.SchedulerProfile(
            runtime.scheduler.NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION
        ),
    )
    graph = schedule.source_graph
    if graph is None:
        raise RuntimeError("resolved schedule did not produce a source graph.")
    return graph, schedule, {node.node_id: node for node in graph.nodes}


def _diagnostics_plan(
    runtime: SimpleNamespace,
    session: Any,
    registry: Any,
    graph: Any,
    schedule: Any,
    nodes: dict[str, Any],
    initial_total_mass: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Bind all closed diagnostics, retaining two caller-owned observations."""
    wp = runtime.warp
    diagnostics = runtime.diagnostics
    shape = (session.dimensions.n_boxes, session.dimensions.n_species)
    device = session.particles.masses.device

    def matrix() -> Any:
        """Allocate one caller-owned diagnostic matrix."""
        return wp.zeros(shape, dtype=wp.float64, device=device)

    gas_snapshot, saturation_snapshot, total = matrix(), matrix(), matrix()
    baseline = wp.array(initial_total_mass, dtype=wp.float64, device=device)
    registrations = (
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT,
            gas_snapshot,
        ),
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT,
            saturation_snapshot,
        ),
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.TOTAL_SPECIES_MASS,
            total,
        ),
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
            wp.zeros(shape[0], dtype=wp.float64, device=device),
        ),
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            matrix(),
            energy_transfer=matrix(),
        ),
        diagnostics.ResidentDiagnosticRegistration(
            diagnostics.ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            matrix(),
            baseline_total_mass=baseline,
            source_ledger=matrix(),
            sink_ledger=matrix(),
        ),
    )
    return (
        diagnostics.ResidentDiagnosticsPlan(
            session,
            registry,
            graph,
            schedule,
            nodes["diagnostics"],
            registrations,
        ),
        gas_snapshot,
        saturation_snapshot,
    )


def _request(
    runtime: SimpleNamespace,
    session: Any,
    registry: Any,
    guard: Any,
    duration: float,
    cpu_gas: GasData,
    cpu_environment: EnvironmentData,
    initial_total_mass: np.ndarray,
) -> tuple[Any, Any, Any]:  # noqa: C901
    """Build fresh exact request carriers from the binding's resource views."""
    wp = runtime.warp
    execution = runtime.execution
    graph, schedule, nodes = _resolved_graph(runtime)
    device = session.particles.masses.device
    communication_view = registry.get_communication_resources()
    if communication_view is None:
        map_data = runtime.communication.CommunicationMap(
            runtime.communication.CommunicationMapForm.ARBITRARY_PAIRS,
            runtime.communication.CommunicationTransportMode.GAS,
            0,
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.float64, device=device),
        )
        communication_view = registry.acquire_communication(
            runtime.communication.CommunicationConfiguration(
                map_data,
                runtime.communication.PrescribedVolumeUpdate(
                    wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=device)
                ),
                (
                    runtime.communication.CommunicationResourceShape(
                        "edge_rates",
                        wp.float64,
                        runtime.communication.CommunicationShapeKind.E,
                    ),
                ),
            )
        )
    condensation_view = registry.acquire_condensation()
    coagulation_view = registry.acquire_coagulation(1)
    wall_loss_view = registry.acquire_wall_loss()
    nucleation_view = registry.acquire_nucleation()
    exhaustion = nucleation_view.exhaustion
    unit_scale = wp.ones(
        session.dimensions.n_boxes, dtype=wp.float64, device=device
    )
    wp.copy(exhaustion.requested_scale, unit_scale)
    wp.copy(exhaustion.minimum_scale, unit_scale)
    wp.copy(exhaustion.minimum_volume, unit_scale)
    thermodynamics = runtime.thermodynamics.ThermodynamicsConfig(
        modes=wp.zeros(1, dtype=wp.int32, device=device),
        parameters=wp.array(
            [[800.0, 0.0, 0.0, 0.0]], dtype=wp.float64, device=device
        ),
        molar_mass_reference=wp.array(
            cpu_gas.molar_mass, dtype=wp.float64, device=device
        ),
    )
    condensation = runtime.condensation.WarpCondensationExecutionState(
        runtime.condensation.WarpCondensationState(
            runtime.condensation.CondensationExecutionConfig(
                execution.CondensationConfiguration(
                    execution.CondensationExecutionMode.EQUAL_STEP,
                    False,
                    execution.CondensationActivityMode.IDEAL,
                    execution.CondensationSurfaceMode.STATIC,
                )
            ),
            session.particles,
            session.gas,
            session.environment,
            thermodynamics,
            scratch_buffers=condensation_view.scratch_buffers,
        ),
        duration,
    )
    coagulation = runtime.coagulation.ResidentBrownianCoagulationExecutionState(
        runtime.coagulation.WarpBrownianCoagulationExecutionState(
            runtime.coagulation.WarpBrownianCoagulationState(
                runtime.coagulation.BrownianCoagulationConfig(),
                session.particles,
                None,
                None,
                duration,
                collision_pairs=coagulation_view.collision_pairs,
                n_collisions=coagulation_view.n_collisions,
                rng_states=coagulation_view.rng_states,
                initialize_rng=False,
                environment=session.environment,
            )
        ),
        session,
        registry,
        coagulation_view,
    )
    diagnostics, gas_snapshot, saturation_snapshot = _diagnostics_plan(
        runtime,
        session,
        registry,
        graph,
        schedule,
        nodes,
        initial_total_mass,
    )
    request = runtime.resident_scheduler.ResidentSimulationRequest(
        session,
        registry,
        guard,
        graph,
        schedule,
        thermodynamics,
        condensation,
        coagulation,
        runtime.process_adapters.ResidentDilutionRequest(
            session, registry, 0.0, duration
        ),
        runtime.process_adapters.ResidentWallLossRequest(
            session,
            registry,
            wall_loss_view,
            runtime.wall_loss.NeutralWallLossConfig(
                "spherical", 1.0, chamber_radius=1.0
            ),
            duration,
            enabled_box_indices=(0, 1, 2),
        ),
        runtime.process_adapters.ResidentNucleationRequest(
            session,
            registry,
            nucleation_view,
            runtime.nucleation.NucleationConfig(
                "activation",
                0.0,
                1.0,
                0,
                (1,),
                1e-9,
                0.0,
                1e30,
                200.0,
                400.0,
            ),
            duration,
            runtime.nucleation.NucleationExhaustionControls(False, False),
        ),
        diagnostics,
        runtime.state_updates.ResidentEnvironmentUpdateRequest(
            session,
            registry,
            graph,
            nodes["environment_update"],
            wp.array(
                cpu_environment.temperature, dtype=wp.float64, device=device
            ),
            wp.array(cpu_environment.pressure, dtype=wp.float64, device=device),
        ),
        runtime.state_updates.ResidentGasUpdateRequest(
            session,
            registry,
            graph,
            nodes["gas_update"],
            wp.array(cpu_gas.concentration, dtype=wp.float64, device=device),
        ),
        runtime.resident_communication.ResidentCommunicationRequest(
            session,
            registry,
            graph,
            communication_view,
            nodes["communication"],
            nodes["volume_evolution"],
            duration,
        ),
    )
    return request, gas_snapshot, saturation_snapshot


def run_example(device: str = "cpu") -> ExampleRun:
    """Run the bounded complete resident loop and explicit checkpoint restart.

    Device arrays and diagnostic buffers remain caller-owned.  This example
    synchronizes only to observe diagnostics; setup, ordinary dispatch, and
    restart do not hide transfers or host inspection.
    """
    if not _warp_enabled():
        return ExampleRun(output=_disabled_output())
    runtime = _load_enabled_runtime()
    selected_device = runtime.execution.Device(
        runtime.execution.Backend.WARP, device
    )
    requirements = runtime.execution.CapabilityRequirements(frozenset())
    process = runtime.execution.Process("resident_simulation")
    request = runtime.execution.ExecutionRequest(
        runtime.execution.Backend.WARP, selected_device, process, requirements
    )
    matrix = runtime.execution.CapabilityMatrix(
        frozenset(
            {
                runtime.execution.CapabilityDeclaration(
                    selected_device, process, requirements
                )
            }
        )
    )
    decision = runtime.availability.resolve_availability(request, matrix)
    if decision.request is not request:
        raise RuntimeError(
            "availability resolution returned a different request."
        )
    particles, gas, environment = _build_cpu_state()
    initial_total_mass = particles.volume[:, None] * (
        np.sum(particles.masses * particles.concentration[:, :, None], axis=1)
        + gas.concentration
    )
    if not np.any(initial_total_mass > 0.0):
        raise RuntimeError(
            "resident example initial inventory must be nonzero."
        )
    session = runtime.gpu_session.setup_resident_session(
        particles, gas, environment, selected_device
    )
    registry = runtime.gpu_resources.GPUResourceRegistry(session)
    guard = runtime.gpu_session.ResidentStepGuard(session, registry)
    source_request, gas_output, saturation_output = _request(
        runtime,
        session,
        registry,
        guard,
        1.0,
        gas,
        environment,
        initial_total_mass,
    )
    scheduler = runtime.resident_scheduler.ResidentSimulationScheduler(
        source_request
    )
    for _ in range(2):
        scheduler.execute(1.0)
        guard.assert_step_closed()
    runtime.warp.synchronize_device(session.particles.masses.device)
    gas_snapshot, saturation_snapshot = (
        gas_output.numpy().copy(),
        saturation_output.numpy().copy(),
    )
    conservation_residual = (
        source_request.diagnostics.registrations[5].output.numpy().copy()
    )
    checkpoint = session.checkpoint(registry, guard)
    if session.lifecycle is not runtime.gpu_session.ResidentLifecycle.ACTIVE:
        raise RuntimeError(
            "source session did not remain active at checkpoint."
        )
    restarted_session, restarted_registry, restarted_guard = (
        runtime.checkpoint.restart_resident_session(checkpoint, selected_device)
    )
    if restarted_session is session or restarted_registry is registry:
        raise RuntimeError("restart reused source identities.")
    if restarted_guard is guard:
        raise RuntimeError("restart reused the source guard.")
    restart_gas_before_physics: np.ndarray | None = None
    restart_temperature_before_physics: np.ndarray | None = None
    try:
        restart_request, _, _ = _request(
            runtime,
            restarted_session,
            restarted_registry,
            restarted_guard,
            1.0,
            checkpoint.gas,
            checkpoint.environment,
            checkpoint.particles.volume[:, None]
            * (
                np.sum(
                    checkpoint.particles.masses
                    * checkpoint.particles.concentration[:, :, None],
                    axis=1,
                )
                + checkpoint.gas.concentration
            ),
        )
        runtime.warp.synchronize_device(
            restarted_session.particles.masses.device
        )
        restart_gas_before_physics = (
            restarted_session.gas.concentration.numpy().copy()
        )
        restart_temperature_before_physics = (
            restarted_session.environment.temperature.numpy().copy()
        )
        if not np.array_equal(
            restart_gas_before_physics, checkpoint.gas.concentration
        ) or not np.array_equal(
            restart_temperature_before_physics,
            checkpoint.environment.temperature,
        ):
            raise RuntimeError(
                "restart forcing does not match checkpoint state."
            )
        runtime.resident_scheduler.ResidentSimulationScheduler(
            restart_request
        ).execute(1.0)
        restarted_guard.assert_step_closed()
    finally:
        restarted_session.close(restarted_registry, restarted_guard)
    terminal_checkpoint = session.finalize(registry, guard)
    if session.finalize(registry, guard) is not terminal_checkpoint:
        raise RuntimeError("finalization did not return the cached checkpoint.")
    return ExampleRun(
        output=[
            "Warp CPU is the installed-Warp baseline; CUDA is optional.",
            "Caller owns resident data and diagnostic buffers; "
            "synchronization is explicit.",
            "Restart is manual, exact-device, and uses canonical "
            "checkpoint bytes.",
            # No graph capture or performance guarantee is claimed.
            "No CPU fallback, hidden transfer, automatic restart, graph "
            "capture, or performance guarantee.",
            "Unsupported physics and exact cross-backend RNG replay are "
            "not claimed.",
        ],
        session=session,
        registry=registry,
        guard=guard,
        checkpoint=checkpoint,
        restarted=(restarted_session, restarted_registry, restarted_guard),
        terminal_checkpoint=terminal_checkpoint,
        gas_snapshot=gas_snapshot,
        saturation_snapshot=saturation_snapshot,
        initial_total_mass=initial_total_mass,
        conservation_residual=conservation_residual,
        restart_gas_before_physics=restart_gas_before_physics,
        restart_temperature_before_physics=restart_temperature_before_physics,
        source_steps=2,
        restarted_steps=1,
    )


def main() -> None:
    """Run the example and print its deterministic status lines."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
