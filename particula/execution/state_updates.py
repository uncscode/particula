"""Apply caller-prescribed resident environment and gas replacements.

This direct-import-only module binds immutable update requests to one exact
resident session, resource registry, resolved graph, and graph node. It uses
temporary device scalar validation storage before copying only the designated
resident arrays in place. Canonical empty schemas are intentional write-free
no-ops. It neither schedules, refreshes derived state, transfers host data,
acquires resources, changes lifecycle state, nor provides a fallback.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import warp as wp

from particula.execution.gpu_session import ResidentSession
from particula.execution.process_graph import (
    InvalidatedState,
    NodeKind,
    ProcessNode,
    ResolvedProcessGraph,
    ResourceRequirement,
    _is_resolver_produced_graph,
)

if TYPE_CHECKING:
    from particula.execution.gpu_resources import GPUResourceRegistry


@wp.kernel
def _scan_positive(
    values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    index = wp.tid()
    value = values[index]
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_nonnegative(
    values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    index = wp.tid()
    value = values[index]
    if not wp.isfinite(value) or value < 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_nonnegative_matrix(
    values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    box, species = wp.tid()
    value = values[box, species]
    if not wp.isfinite(value) or value < 0.0:
        wp.atomic_max(invalid, 0, 1)


def _registry_type() -> type[object]:
    """Lazily resolve the concrete registry type for request validation."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


@dataclass(frozen=True, eq=False)
class ResidentEnvironmentUpdateRequest:
    """Retain one exact graph-bound environment replacement request.

    The immutable carrier preserves session, registry, graph, node, and Warp
    input-array identities. Construction validates only the concrete dependency
    carriers; execution validates the graph binding, array schemas, ownership,
    and positive finite temperature and pressure values.

    Attributes:
        session: Exact resident session whose environment arrays may be updated.
        registry: Exact registry pinned to ``session``.
        graph: Exact resolved graph containing ``node`` by identity.
        node: Canonical ``environment_update`` graph node.
        temperature: Caller-owned float64 Warp array shaped ``(n_boxes,)`` in K.
        pressure: Caller-owned float64 Warp array shaped ``(n_boxes,)`` in Pa.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    graph: ResolvedProcessGraph
    node: ProcessNode
    temperature: object
    pressure: object

    def __post_init__(self) -> None:
        """Validate only exact dependency carrier types in field order.

        Raises:
            TypeError: If a dependency is not its exact required concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(self.registry) is not _registry_type():
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.graph) is not ResolvedProcessGraph:
            raise TypeError("graph must be an exact ResolvedProcessGraph.")
        if type(self.node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")


@dataclass(frozen=True, eq=False)
class ResidentGasUpdateRequest:
    """Retain one exact graph-bound gas concentration replacement request.

    The immutable carrier preserves session, registry, graph, node, and Warp
    input-array identities. Construction validates only the concrete dependency
    carriers; execution validates the graph binding, array schema, ownership,
    and finite nonnegative concentration values.

    Attributes:
        session: Exact resident session whose gas concentration may be updated.
        registry: Exact registry pinned to ``session``.
        graph: Exact resolved graph containing ``node`` by identity.
        node: Canonical ``gas_update`` graph node.
        concentration: Caller-owned float64 Warp array shaped
            ``(n_boxes, n_species)`` in the resident gas concentration units.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    graph: ResolvedProcessGraph
    node: ProcessNode
    concentration: object

    def __post_init__(self) -> None:
        """Validate only exact dependency carrier types in field order.

        Raises:
            TypeError: If a dependency is not its exact required concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(self.registry) is not _registry_type():
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.graph) is not ResolvedProcessGraph:
            raise TypeError("graph must be an exact ResolvedProcessGraph.")
        if type(self.node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")


def _array_range(
    value: Any,
    shape: tuple[int, ...],
    name: str,
    item_size: int = 8,
) -> tuple[int, int] | None:
    """Validate contiguous metadata and return a nonempty half-open range."""
    expected: list[int] = []
    stride = item_size
    for length in reversed(shape):
        expected.insert(0, stride)
        stride *= length
    if getattr(value, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    count = 1
    for length in shape:
        count *= length
    if count == 0:
        return None
    pointer = getattr(value, "ptr", None)
    if not isinstance(pointer, Integral) or pointer <= 0:
        raise ValueError(f"{name} must have a valid pointer.")
    if pointer % item_size:
        raise ValueError(f"{name} pointer must be {item_size}-byte aligned.")
    capacity = getattr(value, "capacity", None)
    required = count * item_size
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity < required
        or capacity % item_size
    ):
        raise ValueError(
            f"{name} must have sufficient integral storage capacity."
        )
    return int(pointer), int(pointer) + count * item_size


def _validate_array(
    value: object, shape: tuple[int, ...], device: object, name: str
) -> tuple[int, int] | None:
    """Validate one caller-prescribed float64 Warp array and its byte range."""
    if not (
        type(value).__module__.startswith("warp")
        and type(value).__name__ == "array"
    ):
        raise ValueError(f"{name} must be a Warp array.")
    array = cast(Any, value)
    if array.dtype != wp.float64 or array.shape != shape:
        raise ValueError(f"{name} has incompatible schema.")
    if array.device != device:
        raise ValueError(f"{name} device must match session device.")
    return _array_range(array, shape, name)


def _overlaps(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> bool:
    """Return whether two nonempty half-open byte ranges overlap."""
    return (
        left is not None
        and right is not None
        and left[0] < right[1]
        and right[0] < left[1]
    )


def _primary_arrays(session: ResidentSession) -> tuple[Any, ...]:
    """Return the protected resident primary arrays in canonical order."""
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    return (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.vapor_pressure,
        gas.partitioning,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )


def _reject_primary_aliases(
    session: ResidentSession,
    values: tuple[object, ...],
    ranges: tuple[tuple[int, int] | None, ...],
) -> None:
    """Reject input identities and nonempty ranges shared with primaries."""
    primaries = _primary_arrays(session)
    if any(value is primary for value in values for primary in primaries):
        raise ValueError("Update inputs must not alias session primaries.")
    primary_ranges = tuple(
        _array_range(
            array,
            array.shape,
            "session primary",
            8 if array.dtype == wp.float64 else 4,
        )
        for array in primaries
    )
    if any(
        _overlaps(value_range, primary_range)
        for value_range in ranges
        for primary_range in primary_ranges
    ):
        raise ValueError(
            "Update input byte ranges must not overlap session primaries."
        )


def _validate_node(request: object, *, environment: bool) -> None:
    """Validate resolver provenance, membership, and canonical update role."""
    typed = cast(
        ResidentEnvironmentUpdateRequest | ResidentGasUpdateRequest, request
    )
    if not _is_resolver_produced_graph(typed.graph):
        raise ValueError("request graph must be produced by plan resolution.")
    if not any(node is typed.node for node in typed.graph.nodes):
        raise ValueError("request node must be a member of graph.nodes.")
    expected = (
        (
            "environment_update",
            NodeKind.ENVIRONMENT_UPDATE,
            frozenset({ResourceRequirement.ENVIRONMENT}),
            frozenset(
                {
                    InvalidatedState.VAPOR_PRESSURE,
                    InvalidatedState.SATURATION_RATIO,
                }
            ),
        )
        if environment
        else (
            "gas_update",
            NodeKind.GAS_UPDATE,
            frozenset({ResourceRequirement.GAS}),
            frozenset({InvalidatedState.SATURATION_RATIO}),
        )
    )
    node = typed.node
    if (
        node.node_id,
        node.kind,
        node.process,
        node.requirements.values,
        node.resources,
        node.invalidates,
    ) != (
        expected[0],
        expected[1],
        None,
        frozenset(),
        expected[2],
        expected[3],
    ):
        raise ValueError("request node has an invalid canonical update role.")


def _scan(
    value: Any,
    size: int,
    *,
    positive: bool,
    name: str,
    matrix_shape: tuple[int, int] | None = None,
) -> None:
    """Scan a nonempty device payload with transient scalar validation state."""
    if size == 0:
        return
    invalid = wp.zeros(1, dtype=wp.int32, device=value.device)
    kernel = _scan_positive if positive else _scan_nonnegative
    launch_dim: int | tuple[int, int] = size
    if matrix_shape is not None:
        kernel = _scan_nonnegative_matrix
        launch_dim = matrix_shape
    wp.launch(
        kernel, dim=launch_dim, inputs=[value, invalid], device=value.device
    )
    if int(invalid.numpy()[0]) != 0:
        constraint = (
            "finite and positive" if positive else "finite and nonnegative"
        )
        raise ValueError(f"{name} values must be {constraint}.")


class ResidentStateUpdateExecutor:
    """Apply one graph-bound resident replacement without scheduler behavior.

    Execution preserves all resident container and primary-array identities. It
    updates only environment temperature and pressure or gas concentration;
    canonical empty schemas are successful write-free no-ops. It does not
    refresh derived state, schedule work, transfer host data, alter lifecycle
    state, or provide rollback after a copy writer launches.
    """

    def execute(self, request: object) -> object:
        """Validate and apply one exact environment or gas update request.

        Args:
            request: Exact immutable environment or gas update request.

        Returns:
            The identical resident environment or gas container that was
            targeted by ``request``.

        Raises:
            TypeError: If ``request`` is not an exact supported request type.
            ValueError: If the pinned binding, graph role, input schema,
                ownership, or scalar values are invalid.
        """
        if type(request) is ResidentEnvironmentUpdateRequest:
            return self._execute_environment(request)
        if type(request) is ResidentGasUpdateRequest:
            return self._execute_gas(request)
        raise TypeError(
            "request must be an exact resident state update request."
        )

    @staticmethod
    def _execute_environment(
        request: ResidentEnvironmentUpdateRequest,
    ) -> object:
        request.registry.validate_pinned_session(request.session)
        _validate_node(request, environment=True)
        dimensions = request.session.dimensions
        target = cast(Any, request.session.environment)
        device = target.temperature.device
        shape = (dimensions.n_boxes,)
        temperature_range = _validate_array(
            request.temperature, shape, device, "temperature"
        )
        pressure_range = _validate_array(
            request.pressure, shape, device, "pressure"
        )
        if request.temperature is request.pressure or _overlaps(
            temperature_range, pressure_range
        ):
            raise ValueError(
                "Environment update inputs must not alias each other."
            )
        _reject_primary_aliases(
            request.session,
            (request.temperature, request.pressure),
            (temperature_range, pressure_range),
        )
        _scan(
            cast(Any, request.temperature),
            dimensions.n_boxes,
            positive=True,
            name="temperature",
        )
        _scan(
            cast(Any, request.pressure),
            dimensions.n_boxes,
            positive=True,
            name="pressure",
        )
        if dimensions.n_boxes:
            wp.copy(target.temperature, cast(Any, request.temperature))
            wp.copy(target.pressure, cast(Any, request.pressure))
        return target

    @staticmethod
    def _execute_gas(request: ResidentGasUpdateRequest) -> object:
        request.registry.validate_pinned_session(request.session)
        _validate_node(request, environment=False)
        dimensions = request.session.dimensions
        target = cast(Any, request.session.gas)
        shape = (dimensions.n_boxes, dimensions.n_species)
        concentration_range = _validate_array(
            request.concentration,
            shape,
            target.concentration.device,
            "concentration",
        )
        _reject_primary_aliases(
            request.session, (request.concentration,), (concentration_range,)
        )
        size = dimensions.n_boxes * dimensions.n_species
        _scan(
            cast(Any, request.concentration),
            size,
            positive=False,
            name="concentration",
            matrix_shape=shape,
        )
        if size:
            wp.copy(target.concentration, cast(Any, request.concentration))
        return target
