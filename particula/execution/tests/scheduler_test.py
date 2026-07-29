"""Test declaration-only scheduling resolution."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from particula.execution import (
    CONDENSATION_CAPABILITY_MATRIX,
    CONDENSATION_PROCESS,
    CapabilityRequirements,
)
from particula.execution.process_graph import (
    DependencyEdge,
    NodeKind,
    ProcessNode,
    TimestepPlan,
)
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    ResolvedTimestepSchedule,
    SchedulerProfile,
    resolve_timestep_schedule,
)


def _node(node_id: str) -> ProcessNode:
    """Build one exact catalogue declaration for scheduling tests."""
    from particula.execution import process_graph

    schema = next(
        item
        for item in process_graph._NODE_CATALOGUE
        if item.node_id == node_id
    )
    requirements = next(
        declaration.requirements
        for declaration in CONDENSATION_CAPABILITY_MATRIX.declarations
        if declaration.process == CONDENSATION_PROCESS
    )
    if node_id != "condensation":
        from particula.execution import CapabilityRequirements

        requirements = CapabilityRequirements(frozenset())
    return ProcessNode(
        schema.node_id,
        schema.kind,
        schema.process,
        requirements,
        schema.resources,
        schema.invalidates,
    )


def _profile(
    direction: NucleationCondensationDirection = (
        NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
    ),
) -> SchedulerProfile:
    """Return a valid direction profile."""
    return SchedulerProfile(direction)


def test_scheduler_carriers_are_immutable_and_validate_types() -> None:
    """Test scheduler records enforce their immutable carrier contracts."""
    selection = EnabledNodeSelection(frozenset({"dilution"}))
    profile = _profile()
    schedule = ResolvedTimestepSchedule((), (), ())

    with pytest.raises(FrozenInstanceError):
        selection.enabled_node_ids = frozenset()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        profile.nucleation_condensation_direction = (  # type: ignore[misc]
            NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION
        )
    with pytest.raises(FrozenInstanceError):
        schedule.nodes = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be a frozenset"):
        EnabledNodeSelection({"dilution"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid node IDs"):
        EnabledNodeSelection(frozenset({"Bad"}))
    with pytest.raises(TypeError, match="only str instances"):
        EnabledNodeSelection(frozenset({1}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NucleationCondensationDirection"):
        SchedulerProfile("both")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NucleationCondensationDirection"):
        SchedulerProfile(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SchedulerProfile()  # type: ignore[call-arg]


def test_schedule_requires_canonical_members_and_topological_order() -> None:
    """Test direct schedule construction rejects noncanonical metadata."""
    dilution = _node("dilution")
    diagnostics = _node("diagnostics")
    edge = DependencyEdge("dilution", "diagnostics")
    with pytest.raises(ValueError, match="nodes must be sorted"):
        ResolvedTimestepSchedule((dilution, diagnostics), (), ())
    with pytest.raises(ValueError, match="canonical topological order"):
        ResolvedTimestepSchedule((diagnostics,), (), ("dilution",))
    with pytest.raises(ValueError, match="endpoints"):
        ResolvedTimestepSchedule((diagnostics,), (edge,), ("diagnostics",))


def test_schedule_rejects_noncanonical_or_cyclic_direct_orders() -> None:
    """Test direct schedule construction requires one acyclic lexical order."""
    condensation = _node("condensation")
    diagnostics = _node("diagnostics")
    dilution = _node("dilution")
    nucleation = _node("nucleation")

    with pytest.raises(ValueError, match="canonical topological order"):
        ResolvedTimestepSchedule(
            (diagnostics, dilution),
            (DependencyEdge("dilution", "diagnostics"),),
            ("diagnostics", "dilution"),
        )
    with pytest.raises(ValueError, match="canonical topological order"):
        ResolvedTimestepSchedule(
            (dilution, _node("wall_loss")),
            (),
            ("wall_loss", "dilution"),
        )
    with pytest.raises(
        ValueError, match="^Cycle detected: condensation, nucleation.$"
    ):
        ResolvedTimestepSchedule(
            (condensation, nucleation),
            (
                DependencyEdge("condensation", "nucleation"),
                DependencyEdge("nucleation", "condensation"),
            ),
            ("condensation", "nucleation"),
        )


@pytest.mark.parametrize(
    ("nodes", "edges", "order", "match"),
    [
        (cast(Any, []), (), (), "nodes must be a tuple"),
        ((), cast(Any, []), (), "dependencies must be a tuple"),
        ((), (), cast(Any, []), "ordered_node_ids must be a tuple"),
        ((object(),), (), (), "only ProcessNode instances"),
        ((), (object(),), (), "only DependencyEdge instances"),
        ((), (), (1,), "only str instances"),
    ],
)
def test_schedule_rejects_noncanonical_container_members(
    nodes: object, edges: object, order: object, match: str
) -> None:
    """Test schedule records reject every non-tuple or invalid member shape."""
    with pytest.raises(TypeError, match=match):
        ResolvedTimestepSchedule(nodes, edges, order)  # type: ignore[arg-type]


def test_schedule_rejects_duplicate_nodes_edges_and_unsorted_edges() -> None:
    """Test schedule records reject duplicate and noncanonical declarations."""
    diagnostics = _node("diagnostics")
    dilution = _node("dilution")
    edge = DependencyEdge("dilution", "diagnostics")
    with pytest.raises(ValueError, match="node IDs must be unique"):
        ResolvedTimestepSchedule(
            (diagnostics, diagnostics),
            (),
            ("diagnostics", "diagnostics"),
        )
    with pytest.raises(ValueError, match="dependencies must be unique"):
        ResolvedTimestepSchedule(
            (diagnostics, dilution), (edge, edge), ("dilution", "diagnostics")
        )
    first = DependencyEdge("dilution", "diagnostics")
    second = DependencyEdge("brownian_coagulation", "diagnostics")
    with pytest.raises(ValueError, match="dependencies must be sorted"):
        ResolvedTimestepSchedule(
            (_node("brownian_coagulation"), diagnostics, dilution),
            (first, second),
            ("brownian_coagulation", "dilution", "diagnostics"),
        )


def test_schedule_requires_exact_record_members() -> None:
    """Test schedule carriers reject ProcessNode and edge subclasses."""

    class NodeSubclass(ProcessNode):
        """Exercise the exact schedule node-record boundary."""

    class EdgeSubclass(DependencyEdge):
        """Exercise the exact schedule edge-record boundary."""

    node = _node("dilution")
    node_subclass = NodeSubclass(
        node.node_id,
        node.kind,
        node.process,
        node.requirements,
        node.resources,
        node.invalidates,
    )
    with pytest.raises(TypeError, match="only ProcessNode instances"):
        ResolvedTimestepSchedule((node_subclass,), (), ("dilution",))
    with pytest.raises(TypeError, match="only DependencyEdge instances"):
        ResolvedTimestepSchedule(
            (_node("diagnostics"), _node("dilution")),
            (EdgeSubclass("dilution", "diagnostics"),),
            ("dilution", "diagnostics"),
        )


def test_resolution_requires_exact_carriers() -> None:
    """Test public resolution rejects carrier subclasses at its boundary."""

    class SelectionSubclass(EnabledNodeSelection):
        """Exercise the exact selection type boundary."""

    class ProfileSubclass(SchedulerProfile):
        """Exercise the exact profile type boundary."""

    class PlanSubclass(TimestepPlan):
        """Exercise the exact plan type boundary."""

    plan = TimestepPlan((), ())
    with pytest.raises(TypeError, match="plan must be"):
        resolve_timestep_schedule(
            PlanSubclass((), ()), EnabledNodeSelection(frozenset()), _profile()
        )
    with pytest.raises(TypeError, match="selection must be"):
        resolve_timestep_schedule(
            plan, SelectionSubclass(frozenset()), _profile()
        )
    with pytest.raises(TypeError, match="profile must be"):
        resolve_timestep_schedule(
            plan,
            EnabledNodeSelection(frozenset()),
            ProfileSubclass(_profile().nucleation_condensation_direction),
        )


def test_empty_selection_and_unknown_selection_are_handled_after_p1() -> None:
    """Test empty schedules and selected-ID validation."""
    assert resolve_timestep_schedule(
        TimestepPlan((), ()), EnabledNodeSelection(frozenset()), _profile()
    ) == ResolvedTimestepSchedule((), (), ())
    with pytest.raises(ValueError, match="Selected node IDs must be declared"):
        resolve_timestep_schedule(
            TimestepPlan((), ()),
            EnabledNodeSelection(frozenset({"missing"})),
            _profile(),
        )


def test_p1_validation_precedes_selection_and_profile_resolution() -> None:
    """Test invalid full declarations cannot be hidden by an empty selection."""
    unknown = ProcessNode(
        "unknown",
        NodeKind.DIAGNOSTIC,
        None,
        CapabilityRequirements(frozenset()),
        frozenset(),
        frozenset(),
    )
    plan = TimestepPlan((unknown,), ())

    with pytest.raises(ValueError, match="^Unknown node ID: unknown.$"):
        resolve_timestep_schedule(
            plan,
            EnabledNodeSelection(frozenset()),
            _profile(),
        )


@pytest.mark.parametrize(
    "plan, message",
    [
        (
            TimestepPlan(
                (_node("condensation"),),
                (DependencyEdge("condensation", "diagnostics"),),
            ),
            "Dependency endpoints must be declared node IDs.",
        ),
        (
            TimestepPlan(
                (_node("nucleation"), _node("condensation")),
                (
                    DependencyEdge("nucleation", "condensation"),
                    DependencyEdge("condensation", "nucleation"),
                ),
            ),
            "Cycle detected: condensation, nucleation.",
        ),
    ],
)
def test_p1_dependency_failures_precede_empty_selection(
    plan: TimestepPlan, message: str
) -> None:
    """Test P1 dependency errors cannot be hidden by disabled selections."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        resolve_timestep_schedule(
            plan,
            EnabledNodeSelection(frozenset()),
            _profile(),
        )


def test_explicit_dependency_and_freshness_closure_reject_missing_producers() -> (
    None
):
    """Test selected consumers cannot silently discard required predecessors."""
    explicit_plan = TimestepPlan(
        (_node("dilution"), _node("diagnostics")),
        (DependencyEdge("dilution", "diagnostics"),),
    )
    with pytest.raises(ValueError, match="dilution -> diagnostics"):
        resolve_timestep_schedule(
            explicit_plan,
            EnabledNodeSelection(frozenset({"diagnostics"})),
            _profile(),
        )

    closure_cases = (
        (
            "environment_update",
            "vapor_pressure_refresh",
            {"environment_update", "vapor_pressure_refresh"},
        ),
        (
            "environment_update",
            "saturation_refresh",
            {"environment_update", "saturation_refresh"},
        ),
        (
            "gas_update",
            "saturation_refresh",
            {
                "environment_update",
                "gas_update",
                "vapor_pressure_refresh",
                "saturation_refresh",
            },
        ),
        (
            "vapor_pressure_refresh",
            "saturation_refresh",
            {
                "environment_update",
                "gas_update",
                "vapor_pressure_refresh",
                "saturation_refresh",
            },
        ),
        (
            "saturation_refresh",
            "condensation",
            {"saturation_refresh", "condensation"},
        ),
        (
            "saturation_refresh",
            "diagnostics",
            {"saturation_refresh", "diagnostics"},
        ),
    )
    for before_id, after_id, available_ids in closure_cases:
        plan = TimestepPlan(
            tuple(_node(node_id) for node_id in available_ids), ()
        )
        with pytest.raises(
            ValueError,
            match=f"{before_id} -> {after_id}",
        ):
            resolve_timestep_schedule(
                plan,
                EnabledNodeSelection(frozenset(available_ids - {before_id})),
                _profile(),
            )


def test_freshness_closure_and_canonical_order_are_derived() -> None:
    """Test complete selected consumers receive all required freshness edges."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "diagnostics",
        }
    )
    plan = TimestepPlan(
        tuple(_node(node_id) for node_id in reversed(sorted(ids))), ()
    )
    schedule = resolve_timestep_schedule(
        plan, EnabledNodeSelection(ids), _profile()
    )
    pairs = {(edge.before_id, edge.after_id) for edge in schedule.dependencies}
    assert pairs == {
        ("environment_update", "vapor_pressure_refresh"),
        ("environment_update", "saturation_refresh"),
        ("gas_update", "saturation_refresh"),
        ("vapor_pressure_refresh", "saturation_refresh"),
        ("saturation_refresh", "condensation"),
        ("saturation_refresh", "diagnostics"),
    }
    assert schedule.ordered_node_ids.index(
        "saturation_refresh"
    ) < schedule.ordered_node_ids.index("condensation")
    with pytest.raises(
        ValueError, match="environment_update -> saturation_refresh"
    ):
        resolve_timestep_schedule(
            TimestepPlan((_node("saturation_refresh"),), ()),
            EnabledNodeSelection(frozenset({"saturation_refresh"})),
            _profile(),
        )


@pytest.mark.parametrize(
    "direction, expected",
    [
        (
            NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION,
            ("nucleation", "condensation"),
        ),
        (
            NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION,
            ("condensation", "nucleation"),
        ),
    ],
)
def test_direction_policy_adds_exact_edge(
    direction: NucleationCondensationDirection, expected: tuple[str, str]
) -> None:
    """Test both direction policies produce one effective direction edge."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "nucleation",
        }
    )
    schedule = resolve_timestep_schedule(
        TimestepPlan(tuple(_node(node_id) for node_id in ids), ()),
        EnabledNodeSelection(ids),
        _profile(direction),
    )
    pairs = {(edge.before_id, edge.after_id) for edge in schedule.dependencies}
    assert expected in pairs
    if (
        direction
        is NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION
    ):
        assert ("nucleation", "saturation_refresh") in pairs
        assert (
            schedule.ordered_node_ids.index("nucleation")
            < schedule.ordered_node_ids.index("saturation_refresh")
            < schedule.ordered_node_ids.index("condensation")
        )
    else:
        assert schedule.ordered_node_ids.index(
            "condensation"
        ) < schedule.ordered_node_ids.index("nucleation")


def test_direction_policy_deduplicates_matching_and_rejects_opposite_p1_edge() -> (
    None
):
    """Test profile policy handles explicit compatible and conflicting edges."""
    ids = frozenset(
        {
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "nucleation",
            "condensation",
        }
    )
    matching = TimestepPlan(
        tuple(_node(node_id) for node_id in ids),
        (DependencyEdge("condensation", "nucleation"),),
    )
    schedule = resolve_timestep_schedule(
        matching,
        EnabledNodeSelection(ids),
        _profile(NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION),
    )
    assert (
        sum(
            edge == DependencyEdge("condensation", "nucleation")
            for edge in schedule.dependencies
        )
        == 1
    )

    opposite = TimestepPlan(
        tuple(_node(node_id) for node_id in ids),
        (DependencyEdge("nucleation", "condensation"),),
    )
    with pytest.raises(ValueError, match="conflicts with SchedulerProfile"):
        resolve_timestep_schedule(
            opposite,
            EnabledNodeSelection(ids),
            _profile(
                NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION
            ),
        )


def test_direction_policy_is_absent_when_an_endpoint_is_disabled() -> None:
    """Test a one-endpoint selection derives no policy direction edge."""
    plan = TimestepPlan((_node("nucleation"), _node("condensation")), ())
    schedule = resolve_timestep_schedule(
        plan,
        EnabledNodeSelection(frozenset({"nucleation"})),
        _profile(),
    )

    assert schedule.dependencies == ()
    assert schedule.ordered_node_ids == ("nucleation",)


def test_scheduler_resolution_is_registration_order_independent() -> None:
    """Test equivalent plans resolve to one canonical effective schedule."""
    ids = (
        "environment_update",
        "gas_update",
        "vapor_pressure_refresh",
        "saturation_refresh",
        "condensation",
        "diagnostics",
        "dilution",
    )
    selection = EnabledNodeSelection(frozenset(ids))
    forward = resolve_timestep_schedule(
        TimestepPlan(tuple(_node(node_id) for node_id in ids), ()),
        selection,
        _profile(),
    )
    reverse = resolve_timestep_schedule(
        TimestepPlan(tuple(_node(node_id) for node_id in reversed(ids)), ()),
        selection,
        _profile(),
    )

    assert forward == reverse
    dilution_index = forward.ordered_node_ids.index("dilution")
    environment_index = forward.ordered_node_ids.index("environment_update")
    assert dilution_index < environment_index


def test_importing_scheduler_does_not_load_execution_or_gpu_backends() -> None:
    """Test scheduling remains a direct-import-only prelaunch boundary."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import builtins
import sys
original_import = builtins.__import__
blocked = ('warp', 'particula.gpu', 'particula.execution.gpu_session',
           'particula.execution.gpu_resources', 'particula.execution.checkpoint',
           'particula.execution.adapters')
def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked):
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from particula.execution.process_graph import TimestepPlan
from particula.execution.scheduler import EnabledNodeSelection, SchedulerProfile
from particula.execution.scheduler import NucleationCondensationDirection, resolve_timestep_schedule
schedule = resolve_timestep_schedule(TimestepPlan((), ()), EnabledNodeSelection(frozenset()), SchedulerProfile(NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION))
assert schedule.ordered_node_ids == ()
assert not any(name == prefix or name.startswith(prefix + '.') for prefix in blocked for name in sys.modules)
"""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_execution_package_does_not_export_scheduler_symbols() -> None:
    """Test scheduler remains absent from the ten-name execution API."""
    import particula.execution as execution

    assert len(execution.__all__) == 10
    assert "scheduler" not in execution.__all__
    assert not hasattr(execution, "resolve_timestep_schedule")
