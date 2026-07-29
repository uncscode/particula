"""Declare and validate neutral process-graph plans.

This module validates and normalizes immutable process declarations and
provides a read-only canonical topological-order helper. It does not resolve
scheduler policy, execute work, access resources, or load optional backends.
"""

import re
import weakref
from dataclasses import dataclass
from enum import Enum

from particula.execution import (
    CONDENSATION_CAPABILITY_MATRIX,
    CONDENSATION_PROCESS,
    CapabilityRequirements,
    Process,
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_EMPTY_REQUIREMENTS = CapabilityRequirements(frozenset())
_RESOLVED_GRAPH_IDENTITIES: weakref.WeakValueDictionary[
    int, "ResolvedProcessGraph"
] = weakref.WeakValueDictionary()


class NodeKind(str, Enum):
    """Identify the role of a declaration-only process-graph node.

    Node kinds describe graph declarations only; they do not select or run an
    execution backend.
    """

    PROCESS = "process"
    ENVIRONMENT_UPDATE = "environment_update"
    GAS_UPDATE = "gas_update"
    VAPOR_PRESSURE_REFRESH = "vapor_pressure_refresh"
    SATURATION_REFRESH = "saturation_refresh"
    DIAGNOSTIC = "diagnostic"


class ResourceRequirement(str, Enum):
    """Identify backend-neutral state required by a declared graph node.

    These symbolic requirements describe state ownership without accessing a
    resource or selecting a backend.
    """

    PARTICLES = "particles"
    GAS = "gas"
    ENVIRONMENT = "environment"
    THERMODYNAMICS = "thermodynamics"
    PROCESS_SIDECARS = "process_sidecars"
    DIAGNOSTICS = "diagnostics"


class InvalidatedState(str, Enum):
    """Identify derived state invalidated by a declared graph node.

    Invalidations are declarations for validation and do not refresh or mutate
    the named state.
    """

    VAPOR_PRESSURE = "vapor_pressure"
    SATURATION_RATIO = "saturation_ratio"


@dataclass(frozen=True)
class DependencyEdge:
    """Declare one ordered dependency between distinct graph node identifiers.

    Attributes:
        before_id: Canonical identifier of the predecessor node.
        after_id: Canonical identifier of the successor node.
    """

    before_id: str
    after_id: str

    def __post_init__(self) -> None:
        """Validate canonical, distinct dependency endpoints."""
        _validate_node_id(self.before_id, "DependencyEdge.before_id")
        _validate_node_id(self.after_id, "DependencyEdge.after_id")
        if self.before_id == self.after_id:
            raise ValueError("DependencyEdge endpoints must differ.")


@dataclass(frozen=True)
class ProcessNode:
    """Declare one immutable, backend-neutral process-graph node.

    Attributes:
        node_id: Canonical identifier for the declared node.
        kind: Role assigned to the node.
        process: Process declaration for process nodes, otherwise ``None``.
        requirements: Capability requirements for a process node.
        resources: Symbolic resources required by the node.
        invalidates: Derived state the node declares invalid.
    """

    node_id: str
    kind: NodeKind
    process: Process | None
    requirements: CapabilityRequirements
    resources: frozenset[ResourceRequirement]
    invalidates: frozenset[InvalidatedState]

    def __post_init__(self) -> None:
        """Validate the node's typed immutable declaration fields."""
        _validate_node_id(self.node_id, "ProcessNode.node_id")
        if not isinstance(self.kind, NodeKind):
            raise TypeError("ProcessNode.kind must be a NodeKind.")
        if self.process is not None and not isinstance(self.process, Process):
            raise TypeError("ProcessNode.process must be a Process or None.")
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "ProcessNode.requirements must be a CapabilityRequirements."
            )
        _validate_frozenset_members(
            self.resources,
            ResourceRequirement,
            "ProcessNode.resources",
        )
        _validate_frozenset_members(
            self.invalidates,
            InvalidatedState,
            "ProcessNode.invalidates",
        )
        if self.kind is NodeKind.PROCESS:
            if self.process is None:
                raise ValueError("Process nodes must declare a Process.")
        elif self.process is not None or self.requirements.values:
            raise ValueError(
                "Non-process nodes must have no Process and empty requirements."
            )


@dataclass(frozen=True)
class TimestepPlan:
    """Declare an immutable process graph before validation and normalization.

    Attributes:
        nodes: Supplied process-node declarations in caller-provided order.
        dependencies: Supplied dependency declarations in caller-provided order.
    """

    nodes: tuple[ProcessNode, ...]
    dependencies: tuple[DependencyEdge, ...]

    def __post_init__(self) -> None:
        """Validate exact graph-plan container and member types."""
        _validate_plan_members(self.nodes, self.dependencies, "TimestepPlan")


@dataclass(frozen=True)
class ResolvedProcessGraph:
    """Store typed node and dependency tuples for a resolved declaration.

    Direct construction validates only exact tuple and member types. Use
    ``resolve_timestep_plan`` to validate catalogue membership, dependencies,
    and cycles, then canonicalize declaration order.

    Attributes:
        nodes: Process-node declarations.
        dependencies: Dependency declarations.
    """

    nodes: tuple[ProcessNode, ...]
    dependencies: tuple[DependencyEdge, ...]

    def __post_init__(self) -> None:
        """Validate exact resolved-graph container and member types."""
        _validate_plan_members(
            self.nodes,
            self.dependencies,
            "ResolvedProcessGraph",
        )


@dataclass(frozen=True)
class _NodeSchema:
    """Retain one private closed-catalogue node declaration."""

    node_id: str
    kind: NodeKind
    process: Process | None
    resources: frozenset[ResourceRequirement]
    invalidates: frozenset[InvalidatedState]


_NODE_CATALOGUE = (
    _NodeSchema(
        "environment_update",
        NodeKind.ENVIRONMENT_UPDATE,
        None,
        frozenset({ResourceRequirement.ENVIRONMENT}),
        frozenset(
            {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO}
        ),
    ),
    _NodeSchema(
        "gas_update",
        NodeKind.GAS_UPDATE,
        None,
        frozenset({ResourceRequirement.GAS}),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    _NodeSchema(
        "vapor_pressure_refresh",
        NodeKind.VAPOR_PRESSURE_REFRESH,
        None,
        frozenset(
            {
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
            }
        ),
        frozenset(),
    ),
    _NodeSchema(
        "saturation_refresh",
        NodeKind.SATURATION_REFRESH,
        None,
        frozenset(
            {
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
            }
        ),
        frozenset(),
    ),
    _NodeSchema(
        "condensation",
        NodeKind.PROCESS,
        CONDENSATION_PROCESS,
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
    _NodeSchema(
        "brownian_coagulation",
        NodeKind.PROCESS,
        Process("brownian_coagulation"),
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset(),
    ),
    _NodeSchema(
        "dilution",
        NodeKind.PROCESS,
        Process("dilution"),
        frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS}),
        frozenset(),
    ),
    _NodeSchema(
        "wall_loss",
        NodeKind.PROCESS,
        Process("wall_loss"),
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset(),
    ),
    _NodeSchema(
        "nucleation",
        NodeKind.PROCESS,
        Process("nucleation"),
        frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.PROCESS_SIDECARS,
            }
        ),
        frozenset({InvalidatedState.SATURATION_RATIO}),
    ),
    _NodeSchema(
        "diagnostics",
        NodeKind.DIAGNOSTIC,
        None,
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
)

_ALLOWED_EDGES = frozenset(
    {
        ("environment_update", "vapor_pressure_refresh"),
        ("environment_update", "saturation_refresh"),
        ("gas_update", "saturation_refresh"),
        ("vapor_pressure_refresh", "saturation_refresh"),
        ("saturation_refresh", "condensation"),
        ("saturation_refresh", "diagnostics"),
        ("condensation", "diagnostics"),
        ("brownian_coagulation", "diagnostics"),
        ("dilution", "diagnostics"),
        ("wall_loss", "diagnostics"),
        ("nucleation", "diagnostics"),
        ("nucleation", "condensation"),
        ("condensation", "nucleation"),
    }
)


def resolve_timestep_plan(plan: TimestepPlan) -> ResolvedProcessGraph:
    """Validate and deterministically normalize a declaration-only plan.

    This pure boundary neither schedules nor executes the declared graph,
    accesses resources, or selects an execution backend.

    Args:
        plan: Exact immutable plan declaration to validate and normalize.

    Returns:
        A new graph with nodes and dependencies in canonical declaration order.

    Raises:
        TypeError: If the plan is not a ``TimestepPlan``.
        ValueError: If declarations violate the closed catalogue, dependency
            schema, or acyclic graph constraints.
    """
    if type(plan) is not TimestepPlan:
        raise TypeError("plan must be a TimestepPlan.")
    node_ids = _validate_nodes(plan.nodes)
    _validate_dependencies(plan.dependencies, node_ids)
    normalized_nodes = tuple(sorted(plan.nodes, key=lambda node: node.node_id))
    normalized_edges = tuple(
        sorted(
            plan.dependencies, key=lambda edge: (edge.before_id, edge.after_id)
        )
    )
    _raise_for_cycle(normalized_nodes, normalized_edges)
    graph = ResolvedProcessGraph(normalized_nodes, normalized_edges)
    _RESOLVED_GRAPH_IDENTITIES[id(graph)] = graph
    return graph


def _is_resolver_produced_graph(graph: ResolvedProcessGraph) -> bool:
    """Return whether ``graph`` is the exact result of plan resolution."""
    return _RESOLVED_GRAPH_IDENTITIES.get(id(graph)) is graph


def resolve_canonical_topological_order(
    nodes: tuple[ProcessNode, ...],
    dependencies: tuple[DependencyEdge, ...],
) -> tuple[str, ...]:
    """Return a deterministic canonical dependency order for graph records.

    Ready nodes and successor traversal use lexical node identifiers, so the
    result never depends on declaration order. This read-only helper validates
    tuple/member shape, unique node IDs, and edge endpoints, but does not
    validate the closed P1 catalogue, resolve scheduler policy, or execute
    graph nodes.

    Args:
        nodes: Exact tuple of declared graph nodes with unique identifiers.
        dependencies: Exact tuple of dependency edges between declared nodes.

    Returns:
        Canonical topological node identifiers.

    Raises:
        TypeError: If containers or members do not have the declared types.
        ValueError: If node identifiers repeat, an endpoint is absent, or the
            effective dependencies contain a cycle.
    """
    _validate_topology_members(nodes, dependencies)
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Topology node IDs must be unique.")
    declared_ids = frozenset(node_ids)
    adjacent: dict[str, list[str]] = {node_id: [] for node_id in declared_ids}
    indegree = {node_id: 0 for node_id in declared_ids}
    for edge in dependencies:
        if (
            edge.before_id not in declared_ids
            or edge.after_id not in declared_ids
        ):
            raise ValueError("Dependency endpoints must be declared node IDs.")
        adjacent[edge.before_id].append(edge.after_id)
        indegree[edge.after_id] += 1

    ready = sorted(
        node_id for node_id, degree in indegree.items() if degree == 0
    )
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for successor in sorted(adjacent[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort()

    if len(ordered) != len(declared_ids):
        remaining = sorted(
            node_id for node_id, degree in indegree.items() if degree > 0
        )
        raise ValueError("Cycle detected: " + ", ".join(remaining) + ".")
    return tuple(ordered)


def _validate_topology_members(
    nodes: object,
    dependencies: object,
) -> None:
    """Validate exact topology utility containers and record members."""
    if type(nodes) is not tuple:
        raise TypeError("Topology.nodes must be a tuple.")
    if not all(type(node) is ProcessNode for node in nodes):
        raise TypeError(
            "Topology.nodes must contain only ProcessNode instances."
        )
    if type(dependencies) is not tuple:
        raise TypeError("Topology.dependencies must be a tuple.")
    if not all(type(edge) is DependencyEdge for edge in dependencies):
        raise TypeError(
            "Topology.dependencies must contain only DependencyEdge instances."
        )


def _validate_node_id(value: object, field_name: str) -> None:
    """Validate a canonical lower-case identifier-style node identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str.")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match ^[a-z][a-z0-9_]*$.")


def _validate_frozenset_members(
    values: object,
    member_type: type[Enum],
    field_name: str,
) -> None:
    """Validate an exact frozenset containing only one enum type."""
    if type(values) is not frozenset:
        raise TypeError(f"{field_name} must be a frozenset.")
    if not all(isinstance(value, member_type) for value in values):
        raise TypeError(
            f"{field_name} must contain only {member_type.__name__} instances."
        )


def _validate_plan_members(
    nodes: object,
    dependencies: object,
    owner: str,
) -> None:
    """Validate exact tuple containers and their declared member types."""
    if type(nodes) is not tuple:
        raise TypeError(f"{owner}.nodes must be a tuple.")
    if not all(type(node) is ProcessNode for node in nodes):
        raise TypeError(
            f"{owner}.nodes must contain only ProcessNode instances."
        )
    if type(dependencies) is not tuple:
        raise TypeError(f"{owner}.dependencies must be a tuple.")
    if not all(type(edge) is DependencyEdge for edge in dependencies):
        raise TypeError(
            f"{owner}.dependencies must contain only DependencyEdge instances."
        )


def _validate_nodes(nodes: tuple[ProcessNode, ...]) -> frozenset[str]:
    """Validate supplied nodes in order against the closed catalogue."""
    node_ids: set[str] = set()
    for node in nodes:
        if node.node_id in node_ids:
            raise ValueError(f"Duplicate node ID: {node.node_id}.")
        node_ids.add(node.node_id)
    for node in nodes:
        schema = next(
            (
                entry
                for entry in _NODE_CATALOGUE
                if entry.node_id == node.node_id
            ),
            None,
        )
        if schema is None:
            raise ValueError(f"Unknown node ID: {node.node_id}.")
        if (node.kind, node.process, node.resources, node.invalidates) != (
            schema.kind,
            schema.process,
            schema.resources,
            schema.invalidates,
        ):
            raise ValueError(f"Invalid declaration for node: {node.node_id}.")
        if node.node_id == "condensation":
            if not any(
                declaration.process == CONDENSATION_PROCESS
                and declaration.requirements == node.requirements
                for declaration in CONDENSATION_CAPABILITY_MATRIX.declarations
            ):
                raise ValueError("Invalid requirements for node: condensation.")
        elif node.requirements != _EMPTY_REQUIREMENTS:
            raise ValueError(f"Invalid requirements for node: {node.node_id}.")
    return frozenset(node_ids)


def _validate_dependencies(
    dependencies: tuple[DependencyEdge, ...],
    node_ids: frozenset[str],
) -> None:
    """Validate supplied dependency declarations in their input order."""
    edges: set[tuple[str, str]] = set()
    for edge in dependencies:
        pair = (edge.before_id, edge.after_id)
        if edge.before_id not in node_ids or edge.after_id not in node_ids:
            raise ValueError("Dependency endpoints must be declared node IDs.")
        if edge.before_id == edge.after_id:
            raise ValueError("Dependency endpoints must differ.")
        if pair in edges:
            raise ValueError(
                "Duplicate dependency edge: "
                f"{edge.before_id} -> {edge.after_id}."
            )
        edges.add(pair)
        if pair not in _ALLOWED_EDGES:
            raise ValueError(
                f"Unsupported dependency edge: {edge.before_id} -> "
                f"{edge.after_id}."
            )


def _raise_for_cycle(
    nodes: tuple[ProcessNode, ...],
    edges: tuple[DependencyEdge, ...],
) -> None:
    """Raise an error when normalized declarations contain a cycle."""
    resolve_canonical_topological_order(nodes, edges)
