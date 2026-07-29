"""Test declaration-only process graph validation and normalization."""

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
    Capability,
    CapabilityRequirements,
    Process,
)
from particula.execution.process_graph import (
    DependencyEdge,
    InvalidatedState,
    NodeKind,
    ProcessNode,
    ResolvedProcessGraph,
    ResourceRequirement,
    TimestepPlan,
    resolve_canonical_topological_order,
    resolve_timestep_plan,
)


def _condensation_requirements() -> CapabilityRequirements:
    """Return a requirement declaration from the condensation matrix."""
    return next(
        declaration.requirements
        for declaration in CONDENSATION_CAPABILITY_MATRIX.declarations
        if declaration.process == CONDENSATION_PROCESS
    )


def _node(node_id: str) -> ProcessNode:
    """Build an exact closed-catalogue node declaration."""
    process_nodes = {
        "condensation": (
            CONDENSATION_PROCESS,
            _condensation_requirements(),
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
        "brownian_coagulation": (
            Process("brownian_coagulation"),
            CapabilityRequirements(frozenset()),
            frozenset(
                {
                    ResourceRequirement.PARTICLES,
                    ResourceRequirement.ENVIRONMENT,
                    ResourceRequirement.PROCESS_SIDECARS,
                }
            ),
            frozenset(),
        ),
        "dilution": (
            Process("dilution"),
            CapabilityRequirements(frozenset()),
            frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS}),
            frozenset(),
        ),
        "wall_loss": (
            Process("wall_loss"),
            CapabilityRequirements(frozenset()),
            frozenset(
                {
                    ResourceRequirement.PARTICLES,
                    ResourceRequirement.ENVIRONMENT,
                    ResourceRequirement.PROCESS_SIDECARS,
                }
            ),
            frozenset(),
        ),
        "nucleation": (
            Process("nucleation"),
            CapabilityRequirements(frozenset()),
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
    }
    if node_id in process_nodes:
        process, requirements, resources, invalidates = process_nodes[node_id]
        return ProcessNode(
            node_id,
            NodeKind.PROCESS,
            process,
            requirements,
            resources,
            invalidates,
        )
    declarations = {
        "environment_update": (
            NodeKind.ENVIRONMENT_UPDATE,
            frozenset({ResourceRequirement.ENVIRONMENT}),
            frozenset(
                {
                    InvalidatedState.VAPOR_PRESSURE,
                    InvalidatedState.SATURATION_RATIO,
                }
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
    kind, resources, invalidates = declarations[node_id]
    return ProcessNode(
        node_id,
        kind,
        None,
        CapabilityRequirements(frozenset()),
        resources,
        invalidates,
    )


def _unchecked_edge(before_id: str, after_id: str) -> DependencyEdge:
    """Build an edge whose constructor-only invariants are bypassed."""
    edge = object.__new__(DependencyEdge)
    object.__setattr__(edge, "before_id", before_id)
    object.__setattr__(edge, "after_id", after_id)
    return edge


def test_enums_and_records_are_immutable() -> None:
    """Test enums, frozen records, and their declaration-only values."""
    node = _node("dilution")
    edge = DependencyEdge("dilution", "diagnostics")

    assert NodeKind.PROCESS.value == "process"
    assert ResourceRequirement.PARTICLES.value == "particles"
    assert InvalidatedState.SATURATION_RATIO.value == "saturation_ratio"
    with pytest.raises(FrozenInstanceError):
        node.node_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        edge.before_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TimestepPlan((), ()).nodes = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ResolvedProcessGraph((), ()).nodes = ()  # type: ignore[misc]


def test_enums_expose_the_complete_closed_value_sets() -> None:
    """Test each symbolic enum exposes exactly its contract values."""
    assert {kind.value for kind in NodeKind} == {
        "process",
        "environment_update",
        "gas_update",
        "vapor_pressure_refresh",
        "saturation_refresh",
        "diagnostic",
    }
    assert {resource.value for resource in ResourceRequirement} == {
        "particles",
        "gas",
        "environment",
        "thermodynamics",
        "process_sidecars",
        "diagnostics",
    }
    assert {state.value for state in InvalidatedState} == {
        "vapor_pressure",
        "saturation_ratio",
    }


def test_every_catalogue_node_resolves_only_with_exact_declaration() -> None:
    """Test each catalogue row is accepted by the closed resolver."""
    node_ids = (
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
    for node_id in node_ids:
        assert resolve_timestep_plan(TimestepPlan((_node(node_id),), ())).nodes


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (lambda: DependencyEdge("Bad", "good"), "before_id"),
        (lambda: DependencyEdge("same", "same"), "endpoints must differ"),
        (
            lambda: TimestepPlan(cast(Any, []), ()),
            "nodes must be a tuple",
        ),
        (
            lambda: TimestepPlan((), cast(Any, [])),
            "dependencies must be a tuple",
        ),
        (
            lambda: ProcessNode(
                "x",
                NodeKind.PROCESS,
                None,
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            "declare a Process",
        ),
    ],
)
def test_record_contracts_reject_invalid_values(
    constructor, match: str
) -> None:
    """Test record constructors reject invalid values with stable text."""
    with pytest.raises((TypeError, ValueError), match=match):
        constructor()


@pytest.mark.parametrize(
    ("constructor", "error", "match"),
    [
        (
            lambda: ProcessNode(
                1,  # type: ignore[arg-type]
                NodeKind.PROCESS,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            TypeError,
            "node_id must be a str",
        ),
        (
            lambda: ProcessNode(
                "Bad",
                NodeKind.PROCESS,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            ValueError,
            "node_id must match",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                "process",  # type: ignore[arg-type]
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            TypeError,
            "kind must be a NodeKind",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                NodeKind.PROCESS,
                object(),  # type: ignore[arg-type]
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            TypeError,
            "process must be a Process or None",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                NodeKind.PROCESS,
                Process("dilution"),
                object(),  # type: ignore[arg-type]
                frozenset(),
                frozenset(),
            ),
            TypeError,
            "requirements must be a CapabilityRequirements",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                NodeKind.PROCESS,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                set(),  # type: ignore[arg-type]
                frozenset(),
            ),
            TypeError,
            "resources must be a frozenset",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                NodeKind.PROCESS,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset({"particles"}),  # type: ignore[arg-type]
                frozenset(),
            ),
            TypeError,
            "resources must contain only ResourceRequirement",
        ),
        (
            lambda: ProcessNode(
                "dilution",
                NodeKind.PROCESS,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset({"saturation_ratio"}),  # type: ignore[arg-type]
            ),
            TypeError,
            "invalidates must contain only InvalidatedState",
        ),
        (
            lambda: ProcessNode(
                "diagnostics",
                NodeKind.DIAGNOSTIC,
                Process("dilution"),
                CapabilityRequirements(frozenset()),
                frozenset(),
                frozenset(),
            ),
            ValueError,
            "Non-process nodes",
        ),
        (
            lambda: ProcessNode(
                "diagnostics",
                NodeKind.DIAGNOSTIC,
                None,
                CapabilityRequirements(frozenset({Capability("extra")})),
                frozenset(),
                frozenset(),
            ),
            ValueError,
            "Non-process nodes",
        ),
    ],
)
def test_process_node_constructor_rejects_each_field_contract(
    constructor, error: type[Exception], match: str
) -> None:
    """Test ProcessNode validates field types before graph resolution."""
    with pytest.raises(error, match=match):
        constructor()


@pytest.mark.parametrize("owner", [TimestepPlan, ResolvedProcessGraph])
@pytest.mark.parametrize(
    ("nodes", "dependencies", "match"),
    [
        ((), (), ""),
        ([], (), "nodes must be a tuple"),
        ((object(),), (), "nodes must contain only ProcessNode"),
        ((), [], "dependencies must be a tuple"),
        ((), (object(),), "dependencies must contain only DependencyEdge"),
    ],
)
def test_plan_records_require_exact_immutable_containers(
    owner, nodes: object, dependencies: object, match: str
) -> None:
    """Test both graph records validate exact tuple containers and members."""
    if not match:
        assert owner(nodes, dependencies).nodes == ()
        return
    with pytest.raises(TypeError, match=match):
        owner(nodes, dependencies)  # type: ignore[arg-type]


@pytest.mark.parametrize("owner", [TimestepPlan, ResolvedProcessGraph])
def test_plan_records_reject_process_graph_record_subclasses(owner) -> None:
    """Test every P1 carrier rejects node and edge subclasses."""

    class NodeSubclass(ProcessNode):
        """Exercise the exact P1 node-record boundary."""

    class EdgeSubclass(DependencyEdge):
        """Exercise the exact P1 edge-record boundary."""

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
        owner((node_subclass,), ())
    with pytest.raises(TypeError, match="only DependencyEdge instances"):
        owner(
            (_node("diagnostics"), _node("dilution")),
            (EdgeSubclass("dilution", "diagnostics"),),
        )


def test_resolver_requires_exact_plan_type_before_inspecting_plan() -> None:
    """Test resolver rejects subclasses and unrelated values at its boundary."""

    class PlanSubclass(TimestepPlan):
        """Exercise the exact plan-type boundary."""

    for plan in (object(), PlanSubclass((), ())):
        with pytest.raises(TypeError, match="^plan must be a TimestepPlan.$"):
            resolve_timestep_plan(plan)  # type: ignore[arg-type]


def test_resolver_normalizes_noncanonical_dag_without_execution_order() -> None:
    """Test normalization sorts declarations and does not create scheduling data."""
    nodes = (
        _node("diagnostics"),
        _node("condensation"),
        _node("saturation_refresh"),
        _node("environment_update"),
    )
    edges = (
        DependencyEdge("condensation", "diagnostics"),
        DependencyEdge("saturation_refresh", "condensation"),
        DependencyEdge("environment_update", "saturation_refresh"),
    )

    resolved = resolve_timestep_plan(TimestepPlan(nodes, edges))

    assert [node.node_id for node in resolved.nodes] == sorted(
        node.node_id for node in nodes
    )
    assert [
        (edge.before_id, edge.after_id) for edge in resolved.dependencies
    ] == sorted((edge.before_id, edge.after_id) for edge in edges)
    assert not hasattr(resolved, "execution_order")
    assert resolved.nodes[0] is nodes[1]
    assert resolved.dependencies[0] is edges[0]


def test_resolver_accepts_empty_and_dependency_free_declarations() -> None:
    """Test valid empty and single-node plans remain declaration-only values."""
    empty = resolve_timestep_plan(TimestepPlan((), ()))
    single_node = _node("dilution")
    single = resolve_timestep_plan(TimestepPlan((single_node,), ()))

    assert empty == ResolvedProcessGraph((), ())
    assert single.nodes == (single_node,)
    assert single.dependencies == ()


def test_canonical_topology_is_independent_of_declaration_order() -> None:
    """Test lexical Kahn ordering for independent and dependent declarations."""
    nodes = (_node("wall_loss"), _node("diagnostics"), _node("dilution"))
    edges = (DependencyEdge("wall_loss", "diagnostics"),)

    assert resolve_canonical_topological_order(nodes, edges) == (
        "dilution",
        "wall_loss",
        "diagnostics",
    )
    assert resolve_canonical_topological_order(nodes[::-1], edges) == (
        "dilution",
        "wall_loss",
        "diagnostics",
    )


def test_canonical_topology_rejects_missing_endpoints_and_cycles() -> None:
    """Test topology utility does not return partial orders for invalid graphs."""
    nodes = (_node("nucleation"), _node("condensation"))
    with pytest.raises(ValueError, match="endpoints must be declared"):
        resolve_canonical_topological_order(
            (_node("dilution"),),
            (DependencyEdge("dilution", "diagnostics"),),
        )
    with pytest.raises(
        ValueError, match="^Cycle detected: condensation, nucleation.$"
    ):
        resolve_canonical_topological_order(
            nodes,
            (
                DependencyEdge("nucleation", "condensation"),
                DependencyEdge("condensation", "nucleation"),
            ),
        )


def test_canonical_topology_requires_exact_records_and_unique_ids() -> None:
    """Test topology records cannot bypass exact type or uniqueness checks."""

    class NodeSubclass(ProcessNode):
        """Exercise the exact topology node-record boundary."""

    node = _node("dilution")
    duplicate = (_node("dilution"), _node("dilution"))
    subclass = NodeSubclass(
        node.node_id,
        node.kind,
        node.process,
        node.requirements,
        node.resources,
        node.invalidates,
    )

    with pytest.raises(TypeError, match="only ProcessNode instances"):
        resolve_canonical_topological_order((subclass,), ())
    with pytest.raises(ValueError, match="node IDs must be unique"):
        resolve_canonical_topological_order(duplicate, ())


def test_resolver_rejects_unknown_duplicate_and_invalid_requirements() -> None:
    """Test node validation rejects before dependency validation."""
    unknown = ProcessNode(
        "unknown",
        NodeKind.DIAGNOSTIC,
        None,
        CapabilityRequirements(frozenset()),
        frozenset(),
        frozenset(),
    )
    with pytest.raises(ValueError, match="^Unknown node ID: unknown.$"):
        resolve_timestep_plan(TimestepPlan((unknown,), ()))
    with pytest.raises(ValueError, match="^Duplicate node ID: dilution.$"):
        resolve_timestep_plan(
            TimestepPlan((_node("dilution"), _node("dilution")), ())
        )
    invalid = ProcessNode(
        "dilution",
        NodeKind.PROCESS,
        Process("dilution"),
        _condensation_requirements(),
        _node("dilution").resources,
        frozenset(),
    )
    with pytest.raises(
        ValueError, match="Invalid requirements for node: dilution"
    ):
        resolve_timestep_plan(TimestepPlan((invalid,), ()))


def test_node_rejection_order_and_closed_catalogue_fields() -> None:
    """Test node identity errors precede catalogue and dependency validation."""
    unknown = ProcessNode(
        "unknown",
        NodeKind.DIAGNOSTIC,
        None,
        CapabilityRequirements(frozenset()),
        frozenset(),
        frozenset(),
    )
    with pytest.raises(ValueError, match="^Duplicate node ID: dilution.$"):
        resolve_timestep_plan(
            TimestepPlan(
                (_node("dilution"), _node("dilution"), unknown),
                (DependencyEdge("dilution", "unknown"),),
            )
        )
    valid = _node("dilution")
    variants = (
        ProcessNode(
            "dilution",
            NodeKind.PROCESS,
            Process("other"),
            valid.requirements,
            valid.resources,
            valid.invalidates,
        ),
        ProcessNode(
            "dilution",
            NodeKind.PROCESS,
            valid.process,
            valid.requirements,
            frozenset(),
            valid.invalidates,
        ),
        ProcessNode(
            "dilution",
            NodeKind.PROCESS,
            valid.process,
            valid.requirements,
            valid.resources,
            frozenset({InvalidatedState.SATURATION_RATIO}),
        ),
    )
    for node in variants:
        with pytest.raises(
            ValueError, match="^Invalid declaration for node: dilution.$"
        ):
            resolve_timestep_plan(TimestepPlan((node,), ()))
    invalid_condensation = ProcessNode(
        "condensation",
        NodeKind.PROCESS,
        CONDENSATION_PROCESS,
        CapabilityRequirements(frozenset({Capability("not_in_matrix")})),
        _node("condensation").resources,
        _node("condensation").invalidates,
    )
    with pytest.raises(
        ValueError, match="^Invalid requirements for node: condensation.$"
    ):
        resolve_timestep_plan(
            TimestepPlan(
                (invalid_condensation,),
                (DependencyEdge("condensation", "diagnostics"),),
            )
        )


def test_dependency_schema_and_cycle_validation_are_deterministic() -> None:
    """Test allowed edges, malformed endpoints, and canonical cycle failures."""
    nodes = (_node("nucleation"), _node("condensation"), _node("diagnostics"))
    for edge in (
        DependencyEdge("nucleation", "diagnostics"),
        DependencyEdge("nucleation", "condensation"),
        DependencyEdge("condensation", "nucleation"),
    ):
        assert resolve_timestep_plan(TimestepPlan(nodes, (edge,)))
    with pytest.raises(ValueError, match="endpoints must be declared"):
        resolve_timestep_plan(
            TimestepPlan(
                (_node("dilution"),),
                (DependencyEdge("dilution", "diagnostics"),),
            )
        )
    with pytest.raises(ValueError, match="Unsupported dependency edge"):
        resolve_timestep_plan(
            TimestepPlan(
                (_node("dilution"), _node("nucleation")),
                (DependencyEdge("dilution", "nucleation"),),
            )
        )
    cycle = (
        DependencyEdge("nucleation", "condensation"),
        DependencyEdge("condensation", "nucleation"),
    )
    with pytest.raises(
        ValueError, match="^Cycle detected: condensation, nucleation.$"
    ):
        resolve_timestep_plan(TimestepPlan(nodes[:2], cycle))


def test_all_dependency_pairs_obey_the_closed_schema() -> None:
    """Test every allowed pair accepts and every other distinct pair rejects."""
    node_ids = (
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
    allowed = {
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
    for before_id in node_ids:
        for after_id in node_ids:
            if before_id == after_id:
                continue
            edge = DependencyEdge(before_id, after_id)
            plan = TimestepPlan((_node(before_id), _node(after_id)), (edge,))
            if (before_id, after_id) in allowed:
                assert resolve_timestep_plan(plan).dependencies == (edge,)
            else:
                with pytest.raises(
                    ValueError,
                    match=(
                        "^Unsupported dependency edge: "
                        f"{before_id} -> {after_id}.$"
                    ),
                ):
                    resolve_timestep_plan(plan)


def test_dependency_validation_order_and_constructor_contracts() -> None:
    """Test dependency checks occur in declared order before cycle detection."""
    with pytest.raises(TypeError, match="before_id must be a str"):
        DependencyEdge(1, "dilution")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after_id must match"):
        DependencyEdge("dilution", "Bad")
    nodes = (_node("dilution"), _node("diagnostics"))
    edge = DependencyEdge("dilution", "diagnostics")
    with pytest.raises(
        ValueError,
        match="^Duplicate dependency edge: dilution -> diagnostics.$",
    ):
        resolve_timestep_plan(TimestepPlan(nodes, (edge, edge)))
    with pytest.raises(ValueError, match="^Dependency endpoints must differ.$"):
        resolve_timestep_plan(
            TimestepPlan(
                (_node("dilution"),),
                (_unchecked_edge("dilution", "dilution"),),
            )
        )
    with pytest.raises(
        ValueError, match="^Dependency endpoints must be declared node IDs.$"
    ):
        resolve_timestep_plan(
            TimestepPlan(
                (_node("dilution"),),
                (_unchecked_edge("dilution", "missing"),),
            )
        )


def test_cycle_error_is_canonical_for_permuted_declarations() -> None:
    """Test cycle reporting does not depend on input node or edge order."""
    forward = TimestepPlan(
        (_node("nucleation"), _node("condensation")),
        (
            DependencyEdge("nucleation", "condensation"),
            DependencyEdge("condensation", "nucleation"),
        ),
    )
    reverse = TimestepPlan(
        (_node("condensation"), _node("nucleation")),
        (
            DependencyEdge("condensation", "nucleation"),
            DependencyEdge("nucleation", "condensation"),
        ),
    )
    messages = []
    for plan in (forward, reverse):
        with pytest.raises(ValueError) as error:
            resolve_timestep_plan(plan)
        messages.append(str(error.value))

    assert messages == ["Cycle detected: condensation, nucleation."] * 2


def test_rejected_plan_can_be_corrected_and_resubmitted() -> None:
    """Test immutable rejected declarations do not affect a corrected retry."""
    rejected = TimestepPlan(
        (_node("dilution"), _node("nucleation")),
        (DependencyEdge("dilution", "nucleation"),),
    )
    with pytest.raises(ValueError, match="^Unsupported dependency edge"):
        resolve_timestep_plan(rejected)

    corrected = TimestepPlan(
        (_node("dilution"), _node("diagnostics")),
        (DependencyEdge("dilution", "diagnostics"),),
    )
    resolved = resolve_timestep_plan(corrected)

    assert rejected.dependencies[0].after_id == "nucleation"
    assert resolved.dependencies == corrected.dependencies


def test_importing_graph_does_not_load_optional_backend() -> None:
    """Test graph validation remains neutral to optional GPU backends."""
    root = Path(__file__).parents[3]
    environment = os.environ | {"PYTHONPATH": str(root)}
    script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'warp' or name.startswith('warp.') or name == 'particula.gpu' or name.startswith('particula.gpu.'):
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from particula.execution.process_graph import TimestepPlan, resolve_timestep_plan
assert resolve_timestep_plan(TimestepPlan((), ())).nodes == ()
assert not any(name == 'warp' or name.startswith('warp.') or name == 'particula.gpu' or name.startswith('particula.gpu.') for name in sys.modules)
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
