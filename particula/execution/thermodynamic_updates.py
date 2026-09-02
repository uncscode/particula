"""Coordinate resident thermodynamic derived-state freshness.

This direct-import-only Warp boundary consumes resolver-produced graph and
schedule metadata. Legacy callers explicitly report successful ordinary nodes
and bracket supported consumer callbacks. The READY-prepared path validates
and freezes both consumer windows during setup, then enqueues only retained
writers and records successful-node freshness metadata. Both paths refresh
stale vapor-pressure and saturation-ratio fields immediately before a consumer.
Communication and volume-evolution barriers invalidate saturation ratio
without invalidating or refreshing vapor pressure.

It does not own lifecycle, resource acquisition, scheduling, transfers,
fallbacks, or general process dispatch. Prepared setup freezes the consumer
windows and enqueue repeats no host/setup metadata validation or performs
allocation, host readback, transfer, synchronization, mutable-carrier lookup,
or rebinding. Retained refresh writers still perform device-side status and
physical-state validation. Setup rejection is pre-writer; rollback is not
promised after a refresh writer launches. Refreshing vapor pressure delegates to
the authoritative GPU primitive; saturation refresh is a private device writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, cast

import warp as wp

from particula.execution.gpu_session import ResidentSession
from particula.execution.process_graph import (
    InvalidatedState,
    NodeKind,
    ProcessNode,
    ResolvedProcessGraph,
    ResourceRequirement,
    _is_resolver_produced_graph,
    resolve_canonical_topological_order,
)
from particula.execution.resident_enqueue import PreparedResidentTimestep
from particula.execution.scheduler import (
    ResolvedTimestepSchedule,
    is_resolver_produced_schedule,
)
from particula.gpu.kernels.thermodynamics import (
    ThermodynamicsConfig,
    _refresh_vapor_pressure_kernel,
    refresh_vapor_pressure_gpu,
)
from particula.util.constants import GAS_CONSTANT

_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))


@wp.kernel
def _refresh_saturation_ratio_kernel(
    concentration: Any,
    molar_mass: Any,
    vapor_pressure: Any,
    temperature: Any,
    saturation_ratio: Any,
) -> None:
    """Write saturation ratios from resident gas and environment fields.

    The kernel applies ``c × R × T / (M × p_v)`` independently to each
    box/species lane, where concentration ``c`` is in kg/m³, ``R`` is the gas
    constant, temperature ``T`` is in K, molar mass ``M`` is in kg/mol, and
    vapor pressure ``p_v`` is in Pa.
    """
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    saturation_ratio[box_idx, species_idx] = (
        concentration[box_idx, species_idx]
        * _GAS_CONSTANT
        * temperature[box_idx]
        / (molar_mass[species_idx] * vapor_pressure[box_idx, species_idx])
    )


def _registry_type() -> type[object]:
    """Lazily resolve the concrete registry type without a package export.

    Returns:
        The direct-module-only ``GPUResourceRegistry`` class.
    """
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


@dataclass(frozen=True, eq=False)
class ResidentThermodynamicUpdateRequest:
    """Bind a coordinator to exact resident state and resolved metadata.

    Construction intentionally validates only exact concrete carrier types.
    Graph provenance, schedule membership, session ownership, and array schemas
    are checked immediately before a reported node or consumer is executed.

    Attributes:
        session: Active resident session whose gas and environment fields are
            refreshed in place.
        registry: Pinned registry that must be bound to ``session``.
        graph: Resolver-produced process graph that owns canonical nodes.
        schedule: Resolver-produced ordered schedule of graph nodes.
        thermodynamics: Configuration passed unchanged to the authoritative
            vapor-pressure writer.
    """

    session: ResidentSession
    registry: object
    graph: ResolvedProcessGraph
    schedule: ResolvedTimestepSchedule
    thermodynamics: ThermodynamicsConfig

    def __post_init__(self) -> None:
        """Validate exact retained carrier types without inspecting payloads.

        Raises:
            TypeError: If any retained object is not its required exact
                concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(self.registry) is not _registry_type():
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.graph) is not ResolvedProcessGraph:
            raise TypeError("graph must be an exact ResolvedProcessGraph.")
        if type(self.schedule) is not ResolvedTimestepSchedule:
            raise TypeError(
                "schedule must be an exact ResolvedTimestepSchedule."
            )
        if type(self.thermodynamics) is not ThermodynamicsConfig:
            raise TypeError(
                "thermodynamics must be an exact ThermodynamicsConfig."
            )


@dataclass(frozen=True, eq=False)
class PreparedResidentThermodynamicBinding:
    """Bind validated P1 thermodynamic writer identities.

    This concrete-only record retains the exact arrays and configuration
    selected during setup. Enqueue uses those references only and does not
    rediscover resident containers or resolve another refresh policy.

    Attributes:
        request: Exact thermodynamic update request validated during setup.
        gas: Resident gas container retained for vapor-pressure refresh.
        gas_concentration: Bound resident gas concentration array.
        gas_molar_mass: Bound resident gas molar-mass array.
        gas_vapor_pressure: Bound resident vapor-pressure destination array.
        environment_temperature: Bound resident temperature array in K.
        environment_saturation_ratio: Bound saturation-ratio destination array.
        device: Device shared by all bound writer arrays.
        dimensions: Exact resident dimensions used for empty-schema checks.
        thermodynamics: Exact configuration for vapor-pressure refresh.
    """

    request: ResidentThermodynamicUpdateRequest
    gas: object
    gas_concentration: object
    gas_molar_mass: object
    gas_vapor_pressure: object
    environment_temperature: object
    environment_saturation_ratio: object
    device: object
    dimensions: object
    thermodynamics: ThermodynamicsConfig


@dataclass(frozen=True, eq=False)
class PreparedResidentThermodynamicConsumer:
    """Bind a current consumer refresh window without changing state.

    Attributes:
        binding: Validated common thermodynamic writer binding.
        node: Exact consumer node expected at the current cursor.
        virtual_nodes: Current preceding virtual refresh nodes.
        refresh_vapor: Whether vapor pressure is stale at preparation time.
        refresh_saturation: Whether saturation ratio is stale at preparation
            time.
    """

    binding: PreparedResidentThermodynamicBinding
    node: ProcessNode
    virtual_nodes: tuple[ProcessNode, ...]
    refresh_vapor: bool
    refresh_saturation: bool


@dataclass(eq=False)
class PreparedResidentThermodynamicSequence:
    """Freeze the two thermodynamic consumer windows for resident enqueue.

    The coordinator is used only while constructing this carrier. Enqueue uses
    the retained writer inputs and consumer products directly, then records
    successful nodes, so it does not validate bindings, resolve nodes, or
    construct a coordinator.

    Attributes:
        binding: Common resident arrays and thermodynamic configuration.
        condensation: Prepared refresh window before condensation.
        diagnostics: Prepared refresh window before diagnostics.
        cursor: Number of successful operations recorded during enqueue.
        stale_states: Derived fields requiring refresh before consumption.
    """

    binding: PreparedResidentThermodynamicBinding
    condensation: PreparedResidentThermodynamicConsumer
    diagnostics: PreparedResidentThermodynamicConsumer
    cursor: int = 0
    stale_states: set[InvalidatedState] = field(
        default_factory=lambda: {
            InvalidatedState.VAPOR_PRESSURE,
            InvalidatedState.SATURATION_RATIO,
        }
    )


def setup_prepared_thermodynamic_sequence(
    prepared_timestep: object,
    request: object,
    condensation_node: object,
    diagnostics_node: object,
) -> PreparedResidentThermodynamicSequence:
    """Validate and freeze the condensation and diagnostics windows.

    Args:
        prepared_timestep: Exact READY P1 timestep retaining resident arrays.
        request: Exact thermodynamic request associated with the timestep.
        condensation_node: Resolver-produced condensation consumer node.
        diagnostics_node: Resolver-produced diagnostics consumer node.

    Returns:
        Prepared sequence with both consumer refresh windows retained by
        identity.

    Raises:
        TypeError: If a supplied carrier or node has an invalid exact type.
        ValueError: If nodes do not belong to the prepared canonical schedule.
    """
    binding = setup_prepared_thermodynamic_binding(prepared_timestep, request)
    if type(condensation_node) is not ProcessNode:
        raise TypeError("condensation_node must be an exact ProcessNode.")
    if type(diagnostics_node) is not ProcessNode:
        raise TypeError("diagnostics_node must be an exact ProcessNode.")
    nodes = {node.node_id: node for node in binding.request.schedule.nodes}
    if (
        condensation_node is not nodes.get("condensation")
        or diagnostics_node is not nodes.get("diagnostics")
        or nodes.get("vapor_pressure_refresh") is None
        or nodes.get("saturation_refresh") is None
    ):
        raise ValueError(
            "thermodynamic nodes do not match the prepared schedule."
        )
    condensation = PreparedResidentThermodynamicConsumer(
        binding,
        condensation_node,
        (nodes["vapor_pressure_refresh"], nodes["saturation_refresh"]),
        True,
        True,
    )
    diagnostics = PreparedResidentThermodynamicConsumer(
        binding, diagnostics_node, (), False, True
    )
    return PreparedResidentThermodynamicSequence(
        binding, condensation, diagnostics
    )


def setup_prepared_thermodynamic_binding(
    prepared_timestep: object, request: object
) -> PreparedResidentThermodynamicBinding:
    """Validate a P1 timestep and bind thermodynamic arrays once.

    Args:
        prepared_timestep: Exact P1 timestep retaining the request
            configuration.
        request: Exact thermodynamic update request to validate and bind.

    Returns:
        Immutable common binding for later current-cursor consumer preparation.

    Raises:
        TypeError: If either carrier has an unsupported exact type.
        ValueError: If retained P1 identities, resident ownership, metadata, or
            primary arrays are invalid.
    """
    if type(prepared_timestep) is not PreparedResidentTimestep:
        raise TypeError(
            "prepared_timestep must be an exact PreparedResidentTimestep."
        )
    if type(request) is not ResidentThermodynamicUpdateRequest:
        raise TypeError(
            "request must be an exact ResidentThermodynamicUpdateRequest."
        )
    prepared = cast(PreparedResidentTimestep, prepared_timestep)
    typed = cast(ResidentThermodynamicUpdateRequest, request)
    if prepared.request.thermodynamics is not typed.thermodynamics:
        raise ValueError(
            "prepared timestep does not retain the supplied request."
        )
    if (
        prepared.session is not typed.session
        or prepared.registry is not typed.registry
        or prepared.graph is not typed.graph
        or prepared.schedule is not typed.schedule
        or prepared.dimensions is not typed.session.dimensions
    ):
        raise ValueError("prepared timestep identities do not match request.")
    coordinator = ResidentThermodynamicUpdateCoordinator(typed)
    coordinator._validate_binding()
    session = typed.session
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    primaries = (
        cast(Any, session.particles).masses,
        cast(Any, session.particles).concentration,
        cast(Any, session.particles).density,
        cast(Any, session.particles).volume,
        cast(Any, session.particles).charge,
        gas.molar_mass,
        gas.concentration,
        gas.partitioning,
        gas.vapor_pressure,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )
    if len(prepared.primary_arrays) != len(primaries) or any(
        left is not right
        for left, right in zip(prepared.primary_arrays, primaries, strict=True)
    ):
        raise ValueError(
            "prepared timestep primary arrays do not match session."
        )
    return PreparedResidentThermodynamicBinding(
        typed,
        gas,
        gas.concentration,
        gas.molar_mass,
        gas.vapor_pressure,
        environment.temperature,
        environment.saturation_ratio,
        gas.concentration.device,
        cast(Any, session.dimensions),
        typed.thermodynamics,
    )


def _enqueue_prepared_vapor_pressure(
    prepared: PreparedResidentThermodynamicConsumer,
) -> None:
    """Enqueue the selected bound vapor-pressure refresh only.

    This private retained-reference seam does no validation, allocation,
    readback, transfer, synchronization, lookup, or rebinding. An empty window
    is write-free; rollback is not promised after its writer launches.

    Args:
        prepared: Current consumer binding that selects the vapor refresh.
    """
    if not prepared.refresh_vapor:
        return
    binding = prepared.binding
    dimensions = cast(Any, binding.dimensions)
    if not dimensions.n_boxes or not dimensions.n_species:
        return
    wp.launch(
        _refresh_vapor_pressure_kernel,
        dim=(dimensions.n_boxes, dimensions.n_species),
        inputs=[
            binding.thermodynamics.modes,
            binding.thermodynamics.parameters,
            binding.environment_temperature,
            binding.gas_vapor_pressure,
        ],
        device=cast(Any, binding.device),
    )


def _enqueue_prepared_saturation_ratio(
    prepared: PreparedResidentThermodynamicConsumer,
) -> None:
    """Enqueue the selected bound saturation-ratio refresh only.

    This private retained-reference seam does no validation, allocation,
    readback, transfer, synchronization, lookup, or rebinding. An empty window
    is write-free; rollback is not promised after its writer launches.

    Args:
        prepared: Current consumer binding that selects the saturation refresh.
    """
    if not prepared.refresh_saturation:
        return
    binding = prepared.binding
    dimensions = cast(Any, binding.dimensions)
    if not dimensions.n_boxes or not dimensions.n_species:
        return
    wp.launch(
        _refresh_saturation_ratio_kernel,
        dim=(dimensions.n_boxes, dimensions.n_species),
        inputs=[
            binding.gas_concentration,
            binding.gas_molar_mass,
            binding.gas_vapor_pressure,
            binding.environment_temperature,
            binding.environment_saturation_ratio,
        ],
        device=cast(Any, binding.device),
    )


def record_prepared_thermodynamic_success(
    sequence: PreparedResidentThermodynamicSequence,
    node: ProcessNode,
) -> None:
    """Advance prepared freshness metadata after one successful node.

    The caller invokes this only after the retained writer or process callable
    returns. A failed callable consequently leaves the cursor and stale markers
    unchanged.
    """
    if node.node_id == "vapor_pressure_refresh":
        sequence.stale_states.discard(InvalidatedState.VAPOR_PRESSURE)
    elif node.node_id == "saturation_refresh":
        sequence.stale_states.discard(InvalidatedState.SATURATION_RATIO)
    elif node.node_id == "diagnostics":
        sequence.stale_states.discard(InvalidatedState.SATURATION_RATIO)
    else:
        sequence.stale_states.update(node.invalidates)
    sequence.cursor += 1


_ROLE_SCHEMAS = {
    "communication": (
        NodeKind.COMMUNICATION,
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    "volume_evolution": (
        NodeKind.VOLUME_EVOLUTION,
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    "environment_update": (
        NodeKind.ENVIRONMENT_UPDATE,
        frozenset({ResourceRequirement.ENVIRONMENT}),
        frozenset(
            {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO}
        ),
    ),
    "gas_update": (
        NodeKind.GAS_UPDATE,
        frozenset({ResourceRequirement.GAS}),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    "vapor_pressure_refresh": (
        NodeKind.VAPOR_PRESSURE_REFRESH,
        frozenset(
            {
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
            }
        ),
        frozenset(),
    ),
    "saturation_refresh": (
        NodeKind.SATURATION_REFRESH,
        frozenset(
            {
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
            }
        ),
        frozenset(),
    ),
    "condensation": (
        NodeKind.PROCESS,
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    "diagnostics": (
        NodeKind.DIAGNOSTIC,
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
                ResourceRequirement.DIAGNOSTICS,
            }
        ),
        frozenset(),
    ),
}
_VIRTUAL_IDS = frozenset({"vapor_pressure_refresh", "saturation_refresh"})
_CONSUMER_IDS = frozenset({"condensation", "diagnostics"})


class ResidentThermodynamicUpdateCoordinator:
    """Track derived-state freshness around explicit resident consumer calls.

    The coordinator begins with vapor pressure and saturation ratio stale. A
    successfully reported ordinary node contributes its declared invalidations.
    Before each scheduled condensation or diagnostics callback, virtual refresh
    nodes write only stale fields in vapor-pressure then saturation order. A
    failed writer or callback does not advance the schedule cursor; a successful
    vapor refresh remains fresh if a later saturation refresh fails.

    Attributes:
        cursor: Number of successfully consumed schedule node IDs.
        stale_states: Immutable view of coordinator-owned stale derived-state
            markers.
    """

    def __init__(self, request: ResidentThermodynamicUpdateRequest) -> None:
        """Retain a request and initialize both derived fields as stale.

        Args:
            request: Exact concrete binding to the resident session, registry,
                resolved graph, schedule, and thermodynamic configuration.

        Raises:
            TypeError: If ``request`` is not an exact
                ``ResidentThermodynamicUpdateRequest``.
        """
        if type(request) is not ResidentThermodynamicUpdateRequest:
            raise TypeError(
                "request must be an exact ResidentThermodynamicUpdateRequest."
            )
        self._request = request
        self._cursor = 0
        self._stale = {
            InvalidatedState.VAPOR_PRESSURE,
            InvalidatedState.SATURATION_RATIO,
        }

    @property
    def cursor(self) -> int:
        """Return the number of schedule IDs successfully consumed.

        Returns:
            Number of consumed ordinary, virtual, and consumer nodes.
        """
        return self._cursor

    @property
    def stale_states(self) -> frozenset[InvalidatedState]:
        """Return coordinator-owned derived-state stale markers.

        Returns:
            Immutable set of fields that require refresh before their next
            supported consumer.
        """
        return frozenset(self._stale)

    def _validate_binding(self) -> None:
        """Validate resident ownership and exact graph/schedule provenance."""
        request = self._request
        registry = cast(Any, request.registry)
        registry.validate_pinned_session(request.session)
        if not _is_resolver_produced_graph(request.graph):
            raise ValueError(
                "request graph must be produced by plan resolution."
            )
        if not is_resolver_produced_schedule(request.schedule, request.graph):
            raise ValueError(
                "request schedule must be produced for the exact graph."
            )
        graph_by_id = {node.node_id: node for node in request.graph.nodes}
        schedule_ids = request.schedule.ordered_node_ids
        if len(schedule_ids) != len(set(schedule_ids)) or any(
            node_id not in graph_by_id for node_id in schedule_ids
        ):
            raise ValueError(
                "schedule IDs must be an ordered subset of graph IDs."
            )
        if schedule_ids != resolve_canonical_topological_order(
            request.schedule.nodes,
            request.schedule.dependencies,
        ):
            raise ValueError("schedule IDs must be an ordered schedule.")
        for node in request.schedule.nodes:
            graph_node = graph_by_id.get(node.node_id)
            if node is not graph_node:
                raise ValueError(
                    "schedule nodes must be identical graph members."
                )
            self._validate_role(node)
        if tuple(node.node_id for node in request.schedule.nodes) != tuple(
            sorted(schedule_ids)
        ):
            raise ValueError("schedule nodes must match ordered schedule IDs.")
        positions = {
            node_id: index for index, node_id in enumerate(schedule_ids)
        }
        for edge in request.schedule.dependencies:
            if (
                edge.before_id not in positions
                or edge.after_id not in positions
                or positions[edge.before_id] >= positions[edge.after_id]
            ):
                raise ValueError(
                    "schedule dependencies must be valid schedule dependencies."
                )

    @staticmethod
    def _validate_role(node: ProcessNode) -> None:
        """Validate the closed canonical role schema used by this boundary."""
        schema = _ROLE_SCHEMAS.get(node.node_id)
        if schema is None:
            return
        kind, resources, invalidates = schema
        if (
            node.kind is not kind
            or node.resources != resources
            or node.invalidates != invalidates
        ):
            raise ValueError(
                "node has an invalid canonical thermodynamic role."
            )

    def _next_node(self) -> ProcessNode:
        """Return the next node after complete metadata validation."""
        self._validate_binding()
        ids = self._request.schedule.ordered_node_ids
        if self._cursor >= len(ids):
            raise ValueError("schedule cursor has no remaining node.")
        node_id = ids[self._cursor]
        return next(
            node
            for node in self._request.schedule.nodes
            if node.node_id == node_id
        )

    def record_completed(self, node: ProcessNode) -> None:
        """Record one successfully completed ordinary schedule node.

        The node must be the next scheduled graph member and must not be a
        virtual refresh or supported consumer. Its declared invalidations are
        added to the coordinator-owned stale markers. Thus completed
        communication and volume barriers mark saturation stale only; neither
        writes vapor pressure or saturation, launches a refresh, transfers, or
        synchronizes.

        Args:
            node: Exact next canonical graph node, reported only after its
                caller-owned execution succeeds.

        Raises:
            TypeError: If ``node`` is not an exact ``ProcessNode``.
            ValueError: If resident binding or schedule metadata is invalid, or
                if ``node`` is out of order, virtual, or a consumer.
        """
        if type(node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")
        expected = self._next_node()
        if node is not expected:
            raise ValueError("node must be the next scheduled graph node.")
        if node.node_id in _VIRTUAL_IDS or node.node_id in _CONSUMER_IDS:
            raise ValueError(
                "record_completed accepts only non-consumer non-virtual nodes."
            )
        self._stale.update(node.invalidates)
        self._cursor += 1

    def _refresh_vapor_pressure(self) -> None:
        """Delegate the authoritative vapor-pressure writer unchanged."""
        request = self._request
        refresh_vapor_pressure_gpu(
            request.thermodynamics,
            cast(Any, request.session.gas),
            cast(Any, request.session.environment).temperature,
        )

    def _refresh_saturation_ratio(self) -> None:
        """Launch the private resident saturation-ratio writer when nonempty."""
        session = self._request.session
        dimensions = session.dimensions
        if not dimensions.n_boxes or not dimensions.n_species:
            return
        gas = cast(Any, session.gas)
        environment = cast(Any, session.environment)
        wp.launch(
            _refresh_saturation_ratio_kernel,
            dim=(dimensions.n_boxes, dimensions.n_species),
            inputs=[
                gas.concentration,
                gas.molar_mass,
                gas.vapor_pressure,
                environment.temperature,
                environment.saturation_ratio,
            ],
            device=gas.concentration.device,
        )

    def prepare_consumer(
        self,
        binding: PreparedResidentThermodynamicBinding,
        node: ProcessNode,
    ) -> PreparedResidentThermodynamicConsumer:
        """Bind the current consumer window without changing coordinator state.

        This P1-consumer seam intentionally reads the coordinator's current
        state on every call. It does not execute writers or consume schedule
        entries; the caller must report successful completion separately.

        Args:
            binding: Exact common P1 binding for this coordinator request.
            node: Exact expected condensation or diagnostics consumer node.

        Returns:
            Immutable current-window binding for enqueue-only refresh writers.

        Raises:
            TypeError: If ``binding`` or ``node`` has an unsupported exact type.
            ValueError: If the binding drifts, metadata is invalid, or ``node``
                is not the current supported consumer.
        """
        if type(binding) is not PreparedResidentThermodynamicBinding:
            raise TypeError(
                "binding must be an exact PreparedResidentThermodynamicBinding."
            )
        if type(node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")
        if binding.request is not self._request:
            raise ValueError("binding must retain the coordinator request.")
        session = self._request.session
        gas = cast(Any, session.gas)
        environment = cast(Any, session.environment)
        if (
            binding.gas is not gas
            or binding.gas_concentration is not gas.concentration
            or binding.gas_molar_mass is not gas.molar_mass
            or binding.gas_vapor_pressure is not gas.vapor_pressure
            or binding.environment_temperature is not environment.temperature
            or binding.environment_saturation_ratio
            is not environment.saturation_ratio
            or binding.device != gas.concentration.device
            or binding.dimensions is not session.dimensions
        ):
            raise ValueError(
                "binding does not retain coordinator primary arrays."
            )
        self._validate_binding()
        virtual_nodes, expected = self._resolve_consumer_window()
        if expected.node_id not in _CONSUMER_IDS or node is not expected:
            raise ValueError(
                "node must be the next scheduled thermodynamic consumer."
            )
        virtual_ids = {item.node_id for item in virtual_nodes}
        vapor = InvalidatedState.VAPOR_PRESSURE in self._stale
        saturation = InvalidatedState.SATURATION_RATIO in self._stale
        if vapor and "vapor_pressure_refresh" not in virtual_ids:
            raise ValueError(
                "stale vapor pressure requires a virtual refresh node."
            )
        if (
            saturation
            and "saturation_refresh" not in virtual_ids
            and virtual_nodes
        ):
            raise ValueError(
                "stale saturation ratio requires a virtual refresh node."
            )
        return PreparedResidentThermodynamicConsumer(
            binding, expected, tuple(virtual_nodes), vapor, saturation
        )

    def execute_consumer(
        self, node: ProcessNode, callback: Callable[[], object]
    ) -> object:
        """Refresh stale fields and call the next scheduled consumer once.

        Only condensation and diagnostics consumers are supported. The method
        consumes immediately preceding virtual refresh nodes when present,
        writes stale fields in dependency order, then invokes ``callback``.
        Stale fields may persist across reported ordinary nodes, so diagnostics
        may refresh them without an adjacent virtual-node window. It applies the
        consumer invalidations and advances the cursor only after the callback
        returns successfully.

        Args:
            node: Exact next scheduled condensation or diagnostics graph node.
            callback: Zero-argument callable that executes that consumer.

        Returns:
            The value returned by ``callback``.

        Raises:
            TypeError: If ``node`` is not an exact ``ProcessNode`` or callback
                is not callable.
            ValueError: If resident binding, schedule metadata, refresh-window
                ordering, or the requested consumer is invalid.
            Exception: Propagates a refresh-writer or callback failure without
                consuming the consumer node.
        """
        if type(node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")
        if not callable(callback):
            raise TypeError("callback must be callable.")
        binding = PreparedResidentThermodynamicBinding(
            self._request,
            self._request.session.gas,
            cast(Any, self._request.session.gas).concentration,
            cast(Any, self._request.session.gas).molar_mass,
            cast(Any, self._request.session.gas).vapor_pressure,
            cast(Any, self._request.session.environment).temperature,
            cast(Any, self._request.session.environment).saturation_ratio,
            cast(Any, self._request.session.gas).concentration.device,
            self._request.session.dimensions,
            self._request.thermodynamics,
        )
        prepared = self.prepare_consumer(binding, node)
        virtual_nodes = prepared.virtual_nodes
        self._refresh_stale_fields()
        result = callback()
        self._stale.update(node.invalidates)
        self._cursor += len(virtual_nodes) + 1
        return result

    def _refresh_stale_fields(self) -> None:
        """Write only stale derived fields in canonical dependency order."""
        if InvalidatedState.VAPOR_PRESSURE in self._stale:
            self._refresh_vapor_pressure()
            self._stale.discard(InvalidatedState.VAPOR_PRESSURE)
        if InvalidatedState.SATURATION_RATIO in self._stale:
            self._refresh_saturation_ratio()
            self._stale.discard(InvalidatedState.SATURATION_RATIO)

    def _resolve_consumer_window(self) -> tuple[list[ProcessNode], ProcessNode]:
        """Return the canonical virtual-writer window and its consumer."""
        schedule = self._request.schedule
        if self._cursor >= len(schedule.ordered_node_ids):
            raise ValueError("schedule cursor has no remaining node.")
        by_id = {item.node_id: item for item in schedule.nodes}
        index = self._cursor
        virtual_nodes: list[ProcessNode] = []
        while schedule.ordered_node_ids[index] in _VIRTUAL_IDS:
            virtual_nodes.append(by_id[schedule.ordered_node_ids[index]])
            index += 1
            if index >= len(schedule.ordered_node_ids):
                raise ValueError(
                    "virtual refresh nodes must precede one consumer."
                )
        expected = by_id[schedule.ordered_node_ids[index]]
        if not virtual_nodes and expected.node_id == "diagnostics":
            # Condensation can leave saturation stale after its virtual window
            # has been consumed. Ordinary nodes report successfully without
            # changing that marker, so diagnostics performs the pending refresh.
            return virtual_nodes, expected
        if not virtual_nodes:
            raise ValueError(
                "a consumer requires immediately preceding virtual refresh "
                "nodes."
            )
        if tuple(item.node_id for item in virtual_nodes) != (
            "vapor_pressure_refresh",
            "saturation_refresh",
        ):
            raise ValueError(
                "a consumer requires vapor then saturation virtual refresh "
                "nodes."
            )
        return virtual_nodes, expected
