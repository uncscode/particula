"""Resolve direct-import-only, declaration-only scheduling metadata.

This concrete module validates immutable P1 graph declarations before applying
enabled-node selection, direction policy, and freshness closure. Resolution is
prelaunch-only: it neither loads backends nor enters lifecycle state, allocates
resources, executes refreshes, or mutates caller-owned data. Its names are not
package exports and must be imported from this module directly.
"""

import re
from dataclasses import dataclass
from enum import Enum

from particula.execution.process_graph import (
    DependencyEdge,
    ProcessNode,
    ResolvedProcessGraph,
    TimestepPlan,
    resolve_canonical_topological_order,
    resolve_timestep_plan,
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_NUCLEATION_CONDENSATION_EDGES = frozenset(
    {
        ("nucleation", "condensation"),
        ("condensation", "nucleation"),
    }
)
_FRESHNESS_EDGES = (
    ("environment_update", "vapor_pressure_refresh"),
    ("environment_update", "saturation_refresh"),
    ("gas_update", "saturation_refresh"),
    ("vapor_pressure_refresh", "saturation_refresh"),
    ("saturation_refresh", "condensation"),
    ("saturation_refresh", "diagnostics"),
)


@dataclass(frozen=True)
class EnabledNodeSelection:
    """Declare the complete immutable set of enabled P1 node identifiers.

    Graph membership is deliberately deferred to schedule resolution, after
    complete P1 plan validation.

    Args:
        enabled_node_ids: Exact frozenset of syntactically valid node IDs.
    """

    enabled_node_ids: frozenset[str]

    def __post_init__(self) -> None:
        """Validate selection container and identifier syntax."""
        if type(self.enabled_node_ids) is not frozenset:
            raise TypeError(
                "EnabledNodeSelection.enabled_node_ids must be a frozenset."
            )
        for node_id in self.enabled_node_ids:
            if type(node_id) is not str:
                raise TypeError(
                    "EnabledNodeSelection.enabled_node_ids must contain only "
                    "str instances."
                )
            if not _NAME_PATTERN.fullmatch(node_id):
                raise ValueError(
                    "EnabledNodeSelection.enabled_node_ids must contain valid "
                    "node IDs."
                )


class NucleationCondensationDirection(str, Enum):
    """Declare the reviewed nucleation/condensation ordering."""

    NUCLEATION_THEN_CONDENSATION = "nucleation_then_condensation"
    CONDENSATION_THEN_NUCLEATION = "condensation_then_nucleation"


@dataclass(frozen=True)
class SchedulerProfile:
    """Declare immutable scheduler direction policy without selecting execution.

    The enum represents exactly one reviewed nucleation/condensation direction;
    it cannot encode both directions or no direction.

    Args:
        nucleation_condensation_direction: The one permitted direction policy.
    """

    nucleation_condensation_direction: NucleationCondensationDirection

    def __post_init__(self) -> None:
        """Validate the single direction declaration."""
        if not isinstance(
            self.nucleation_condensation_direction,
            NucleationCondensationDirection,
        ):
            raise TypeError(
                "SchedulerProfile.nucleation_condensation_direction must be a "
                "NucleationCondensationDirection."
            )


@dataclass(frozen=True)
class ResolvedTimestepSchedule:
    """Store canonical, immutable declaration-only scheduling metadata.

    Nodes and dependencies are canonically sorted. The order is a permutation
    of node IDs created before any lifecycle entry, resource action, or process
    launch.

    Args:
        nodes: Sorted exact tuple of enabled process nodes.
        dependencies: Sorted exact tuple of effective dependency edges.
        ordered_node_ids: Canonical dependency order for exactly these nodes.
    """

    nodes: tuple[ProcessNode, ...]
    dependencies: tuple[DependencyEdge, ...]
    ordered_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate canonical immutable schedule fields."""
        _validate_schedule_types(self)
        _validate_schedule_content(self)


def _validate_schedule_types(schedule: ResolvedTimestepSchedule) -> None:
    """Validate exact immutable schedule container and member types."""
    if type(schedule.nodes) is not tuple:
        raise TypeError("ResolvedTimestepSchedule.nodes must be a tuple.")
    if not all(type(node) is ProcessNode for node in schedule.nodes):
        raise TypeError(
            "ResolvedTimestepSchedule.nodes must contain only ProcessNode "
            "instances."
        )
    if type(schedule.dependencies) is not tuple:
        raise TypeError(
            "ResolvedTimestepSchedule.dependencies must be a tuple."
        )
    if not all(type(edge) is DependencyEdge for edge in schedule.dependencies):
        raise TypeError(
            "ResolvedTimestepSchedule.dependencies must contain only "
            "DependencyEdge instances."
        )
    if type(schedule.ordered_node_ids) is not tuple:
        raise TypeError(
            "ResolvedTimestepSchedule.ordered_node_ids must be a tuple."
        )
    if not all(type(node_id) is str for node_id in schedule.ordered_node_ids):
        raise TypeError(
            "ResolvedTimestepSchedule.ordered_node_ids must contain only str "
            "instances."
        )


def _validate_schedule_content(schedule: ResolvedTimestepSchedule) -> None:
    """Validate canonical schedule ordering, endpoints, and permutation."""
    node_ids = tuple(node.node_id for node in schedule.nodes)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("ResolvedTimestepSchedule node IDs must be unique.")
    if schedule.nodes != tuple(
        sorted(schedule.nodes, key=lambda node: node.node_id)
    ):
        raise ValueError(
            "ResolvedTimestepSchedule.nodes must be sorted by node ID."
        )
    expected_edges = tuple(
        sorted(
            schedule.dependencies,
            key=lambda edge: (edge.before_id, edge.after_id),
        )
    )
    if schedule.dependencies != expected_edges:
        raise ValueError(
            "ResolvedTimestepSchedule.dependencies must be sorted by endpoints."
        )
    edge_pairs = {
        (edge.before_id, edge.after_id) for edge in schedule.dependencies
    }
    if len(schedule.dependencies) != len(edge_pairs):
        raise ValueError(
            "ResolvedTimestepSchedule.dependencies must be unique."
        )
    if any(
        edge.before_id not in node_ids or edge.after_id not in node_ids
        for edge in schedule.dependencies
    ):
        raise ValueError(
            "ResolvedTimestepSchedule dependency endpoints must be schedule "
            "nodes."
        )
    if set(schedule.ordered_node_ids) != set(node_ids) or len(
        schedule.ordered_node_ids
    ) != len(node_ids):
        raise ValueError(
            "ResolvedTimestepSchedule.ordered_node_ids must be a permutation "
            "of nodes."
        )


def resolve_timestep_schedule(
    plan: TimestepPlan,
    selection: EnabledNodeSelection,
    profile: SchedulerProfile,
) -> ResolvedTimestepSchedule:
    """Resolve an immutable effective schedule without running any process.

    After exact carrier-type checks, complete P1 graph validation happens before
    selection or profile inspection. This function then applies selected IDs,
    direction policy, required freshness closure, and canonical topology
    ordering. It returns metadata only and performs no lifecycle, resource, or
    GPU work.

    Args:
        plan: Exact P1 plan declaration.
        selection: Exact enabled-node selection.
        profile: Exact immutable direction policy.

    Returns:
        A new canonical schedule with no disabled dependency endpoints.

    Raises:
        TypeError: If a carrier is not its exact declared type.
        ValueError: If selection, direction, closure, or effective topology is
            invalid. No input is mutated when resolution fails; no rollback is
            required because resolution has no side effects.
    """
    _validate_resolution_inputs(plan, selection, profile)
    resolved = resolve_timestep_plan(plan)
    _validate_selected_ids(selection, resolved)
    direction = _direction_edge(profile)
    _validate_profile_direction(resolved, direction)
    pairs = _resolve_effective_pairs(resolved, selection, profile, direction)
    nodes = tuple(
        node
        for node in resolved.nodes
        if node.node_id in selection.enabled_node_ids
    )
    dependencies = tuple(DependencyEdge(*pair) for pair in sorted(pairs))
    order = resolve_canonical_topological_order(nodes, dependencies)
    return ResolvedTimestepSchedule(nodes, dependencies, order)


def _validate_resolution_inputs(
    plan: object,
    selection: object,
    profile: object,
) -> None:
    """Validate exact public scheduler carriers before P1 resolution."""
    if type(plan) is not TimestepPlan:
        raise TypeError("plan must be a TimestepPlan.")
    if type(selection) is not EnabledNodeSelection:
        raise TypeError("selection must be an EnabledNodeSelection.")
    if type(profile) is not SchedulerProfile:
        raise TypeError("profile must be a SchedulerProfile.")


def _validate_selected_ids(
    selection: EnabledNodeSelection,
    resolved: ResolvedProcessGraph,
) -> None:
    """Reject selected IDs that are absent from an already resolved P1 graph."""
    resolved_ids = frozenset(node.node_id for node in resolved.nodes)
    unknown_ids = sorted(selection.enabled_node_ids - resolved_ids)
    if unknown_ids:
        raise ValueError(
            "Selected node IDs must be declared: "
            + ", ".join(unknown_ids)
            + "."
        )


def _validate_profile_direction(
    resolved: ResolvedProcessGraph,
    direction: tuple[str, str],
) -> None:
    """Reject an explicit P1 direction that opposes the profile."""
    opposite = (direction[1], direction[0])
    if any(
        (edge.before_id, edge.after_id) == opposite
        for edge in resolved.dependencies
    ):
        raise ValueError("P1 direction conflicts with SchedulerProfile.")


def _resolve_effective_pairs(
    resolved: ResolvedProcessGraph,
    selection: EnabledNodeSelection,
    profile: SchedulerProfile,
    direction: tuple[str, str],
) -> set[tuple[str, str]]:
    """Combine retained declarations, profile policy, and freshness closure."""
    enabled = selection.enabled_node_ids
    pairs = _retained_pairs(resolved, enabled)
    if direction[0] in enabled and direction[1] in enabled:
        pairs.add(direction)
    _add_freshness_pairs(pairs, enabled)
    if (
        profile.nucleation_condensation_direction
        is NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
        and "nucleation" in enabled
        and "condensation" in enabled
    ):
        if "saturation_refresh" not in enabled:
            _raise_unsatisfied_closure("nucleation", "saturation_refresh")
        pairs.add(("nucleation", "saturation_refresh"))
    return pairs


def _retained_pairs(
    resolved: ResolvedProcessGraph,
    enabled: frozenset[str],
) -> set[tuple[str, str]]:
    """Return selected P1 edges after enforcing explicit dependency closure."""
    pairs: set[tuple[str, str]] = set()
    for edge in resolved.dependencies:
        if edge.after_id in enabled and edge.before_id not in enabled:
            _raise_unsatisfied_closure(edge.before_id, edge.after_id)
        if edge.before_id in enabled and edge.after_id in enabled:
            pairs.add((edge.before_id, edge.after_id))
    return pairs


def _add_freshness_pairs(
    pairs: set[tuple[str, str]],
    enabled: frozenset[str],
) -> None:
    """Add required freshness edges after enforcing producer closure."""
    for before_id, after_id in _FRESHNESS_EDGES:
        if after_id in enabled:
            if before_id not in enabled:
                _raise_unsatisfied_closure(before_id, after_id)
            pairs.add((before_id, after_id))


def _direction_edge(profile: SchedulerProfile) -> tuple[str, str]:
    """Return the declared direction as one graph edge."""
    if (
        profile.nucleation_condensation_direction
        is NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
    ):
        return ("nucleation", "condensation")
    return ("condensation", "nucleation")


def _raise_unsatisfied_closure(before_id: str, after_id: str) -> None:
    """Raise the stable error for a missing required predecessor."""
    raise ValueError(
        f"Unsatisfied dependency closure: {before_id} -> {after_id}."
    )
