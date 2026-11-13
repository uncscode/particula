# particula.execution.process_graph

*module*

Declare and validate neutral process-graph plans.

This module validates and normalizes immutable process declarations and
provides a read-only canonical topological-order helper. Its closed resident
catalogue declares communication followed by optional volume evolution before
the ten ordinary loop nodes; both barriers invalidate saturation ratio only.
It does not resolve scheduler policy, execute work, access resources, or load
optional backends.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L1-L1)

## particula.execution.process_graph._ALLOWED_EDGES

*attribute* · *module attribute*

```python
_ALLOWED_EDGES = frozenset({('communication', 'volume_evolution'), *(('volume_evolution', schema.node_id) for schema in _NODE_CATALOGUE if schema.node_id not in {'communication', 'volume_evolution'}), ('environment_update', 'vapor_pressure_refresh'), ('environment_update', 'saturation_refresh'), ('gas_update', 'saturation_refresh'), ('vapor_pressure_refresh', 'saturation_refresh'), ('saturation_refresh', 'condensation'), ('saturation_refresh', 'diagnostics'), ('condensation', 'diagnostics'), ('brownian_coagulation', 'diagnostics'), ('dilution', 'diagnostics'), ('wall_loss', 'diagnostics'), ('nucleation', 'diagnostics'), ('nucleation', 'condensation'), ('condensation', 'nucleation')})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L345-L367)

## particula.execution.process_graph._EMPTY_REQUIREMENTS

*attribute* · *module attribute*

```python
_EMPTY_REQUIREMENTS = CapabilityRequirements(frozenset())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L24-L24)

## particula.execution.process_graph._NAME_PATTERN

*attribute* · *module attribute*

```python
_NAME_PATTERN = re.compile('^[a-z][a-z0-9_]*$')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L23-L23)

## particula.execution.process_graph._NODE_CATALOGUE

*attribute* · *module attribute*

```python
_NODE_CATALOGUE = (_NodeSchema('communication', NodeKind.COMMUNICATION, None, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), _NodeSchema('volume_evolution', NodeKind.VOLUME_EVOLUTION, None, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), _NodeSchema('environment_update', NodeKind.ENVIRONMENT_UPDATE, None, frozenset({ResourceRequirement.ENVIRONMENT}), frozenset({InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO})), _NodeSchema('gas_update', NodeKind.GAS_UPDATE, None, frozenset({ResourceRequirement.GAS}), frozenset({InvalidatedState.SATURATION_RATIO})), _NodeSchema('vapor_pressure_refresh', NodeKind.VAPOR_PRESSURE_REFRESH, None, frozenset({ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS}), frozenset()), _NodeSchema('saturation_refresh', NodeKind.SATURATION_REFRESH, None, frozenset({ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS}), frozenset()), _NodeSchema('condensation', NodeKind.PROCESS, CONDENSATION_PROCESS, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), _NodeSchema('brownian_coagulation', NodeKind.PROCESS, Process('brownian_coagulation'), frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.ENVIRONMENT, ResourceRequirement.PROCESS_SIDECARS}), frozenset()), _NodeSchema('dilution', NodeKind.PROCESS, Process('dilution'), frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS}), frozenset()), _NodeSchema('wall_loss', NodeKind.PROCESS, Process('wall_loss'), frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.ENVIRONMENT, ResourceRequirement.PROCESS_SIDECARS}), frozenset()), _NodeSchema('nucleation', NodeKind.PROCESS, Process('nucleation'), frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), _NodeSchema('diagnostics', NodeKind.DIAGNOSTIC, None, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS, ResourceRequirement.DIAGNOSTICS}), frozenset()))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L197-L343)

## particula.execution.process_graph._RESOLVED_GRAPH_IDENTITIES

*attribute* · *module attribute*

```python
_RESOLVED_GRAPH_IDENTITIES: weakref.WeakValueDictionary[int, ResolvedProcessGraph] = weakref.WeakValueDictionary()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L25-L27)

## particula.execution.process_graph.DependencyEdge

*class* · *dataclass*

```python
class DependencyEdge
```

Declare one ordered dependency between distinct graph node identifiers.

**Attributes**

- `before_id` (`str`) — Canonical identifier of the predecessor node.
- `after_id` (`str`) — Canonical identifier of the successor node.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L74-L91)

### particula.execution.process_graph.DependencyEdge.after_id

*attribute* · *instance attribute*

```python
after_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L84-L84)

### particula.execution.process_graph.DependencyEdge.before_id

*attribute* · *instance attribute*

```python
before_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L83-L83)

### particula.execution.process_graph.DependencyEdge.__init__

*method*

```python
def __init__(before_id: str, after_id: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L0-L0)

### particula.execution.process_graph.DependencyEdge.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate canonical, distinct dependency endpoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L86-L91)

## particula.execution.process_graph.InvalidatedState

*class*

```python
class InvalidatedState(str, Enum)
```

Bases: `str`, `Enum`

Identify derived state invalidated by a declared graph node.

Invalidations are declarations for validation and do not refresh or mutate
the named state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L63-L71)

### particula.execution.process_graph.InvalidatedState.SATURATION_RATIO

*attribute* · *class attribute* · *instance attribute*

```python
SATURATION_RATIO = 'saturation_ratio'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L71-L71)

### particula.execution.process_graph.InvalidatedState.VAPOR_PRESSURE

*attribute* · *class attribute* · *instance attribute*

```python
VAPOR_PRESSURE = 'vapor_pressure'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L70-L70)

## particula.execution.process_graph.NodeKind

*class*

```python
class NodeKind(str, Enum)
```

Bases: `str`, `Enum`

Identify the role of a declaration-only process-graph node.

Node kinds describe graph declarations only; they do not select or run an
execution backend. ``COMMUNICATION`` and ``VOLUME_EVOLUTION`` are the
ordered resident barrier declarations, not process or backend APIs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L30-L45)

### particula.execution.process_graph.NodeKind.COMMUNICATION

*attribute* · *class attribute* · *instance attribute*

```python
COMMUNICATION = 'communication'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L44-L44)

### particula.execution.process_graph.NodeKind.DIAGNOSTIC

*attribute* · *class attribute* · *instance attribute*

```python
DIAGNOSTIC = 'diagnostic'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L43-L43)

### particula.execution.process_graph.NodeKind.ENVIRONMENT_UPDATE

*attribute* · *class attribute* · *instance attribute*

```python
ENVIRONMENT_UPDATE = 'environment_update'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L39-L39)

### particula.execution.process_graph.NodeKind.GAS_UPDATE

*attribute* · *class attribute* · *instance attribute*

```python
GAS_UPDATE = 'gas_update'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L40-L40)

### particula.execution.process_graph.NodeKind.PROCESS

*attribute* · *class attribute* · *instance attribute*

```python
PROCESS = 'process'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L38-L38)

### particula.execution.process_graph.NodeKind.SATURATION_REFRESH

*attribute* · *class attribute* · *instance attribute*

```python
SATURATION_REFRESH = 'saturation_refresh'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L42-L42)

### particula.execution.process_graph.NodeKind.VAPOR_PRESSURE_REFRESH

*attribute* · *class attribute* · *instance attribute*

```python
VAPOR_PRESSURE_REFRESH = 'vapor_pressure_refresh'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L41-L41)

### particula.execution.process_graph.NodeKind.VOLUME_EVOLUTION

*attribute* · *class attribute* · *instance attribute*

```python
VOLUME_EVOLUTION = 'volume_evolution'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L45-L45)

## particula.execution.process_graph.ProcessNode

*class* · *dataclass*

```python
class ProcessNode
```

Declare one immutable, backend-neutral process-graph node.

**Attributes**

- `node_id` (`str`) — Canonical identifier for the declared node.
- `kind` (`NodeKind`) — Role assigned to the node.
- `process` (`Process | None`) — Process declaration for process nodes, otherwise ``None``.
- `requirements` (`CapabilityRequirements`) — Capability requirements for a process node.
- `resources` (`frozenset[ResourceRequirement]`) — Symbolic resources required by the node.
- `invalidates` (`frozenset[InvalidatedState]`) — Derived state the node declares invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L94-L141)

### particula.execution.process_graph.ProcessNode.invalidates

*attribute* · *instance attribute*

```python
invalidates: frozenset[InvalidatedState]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L112-L112)

### particula.execution.process_graph.ProcessNode.kind

*attribute* · *instance attribute*

```python
kind: NodeKind
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L108-L108)

### particula.execution.process_graph.ProcessNode.node_id

*attribute* · *instance attribute*

```python
node_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L107-L107)

### particula.execution.process_graph.ProcessNode.process

*attribute* · *instance attribute*

```python
process: Process | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L109-L109)

### particula.execution.process_graph.ProcessNode.requirements

*attribute* · *instance attribute*

```python
requirements: CapabilityRequirements
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L110-L110)

### particula.execution.process_graph.ProcessNode.resources

*attribute* · *instance attribute*

```python
resources: frozenset[ResourceRequirement]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L111-L111)

### particula.execution.process_graph.ProcessNode.__init__

*method*

```python
def __init__(node_id: str, kind: NodeKind, process: Process | None, requirements: CapabilityRequirements, resources: frozenset[ResourceRequirement], invalidates: frozenset[InvalidatedState]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L0-L0)

### particula.execution.process_graph.ProcessNode.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the node's typed immutable declaration fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L114-L141)

## particula.execution.process_graph.ResolvedProcessGraph

*class* · *dataclass*

```python
class ResolvedProcessGraph
```

Store typed node and dependency tuples for a resolved declaration.

Direct construction validates only exact tuple and member types. Use
``resolve_timestep_plan`` to validate catalogue membership, dependencies,
and cycles, then canonicalize declaration order.

**Attributes**

- `nodes` (`tuple[ProcessNode, ...]`) — Process-node declarations.
- `dependencies` (`tuple[DependencyEdge, ...]`) — Dependency declarations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L161-L183)

### particula.execution.process_graph.ResolvedProcessGraph.dependencies

*attribute* · *instance attribute*

```python
dependencies: tuple[DependencyEdge, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L175-L175)

### particula.execution.process_graph.ResolvedProcessGraph.nodes

*attribute* · *instance attribute*

```python
nodes: tuple[ProcessNode, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L174-L174)

### particula.execution.process_graph.ResolvedProcessGraph.__init__

*method*

```python
def __init__(nodes: tuple[ProcessNode, ...], dependencies: tuple[DependencyEdge, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L0-L0)

### particula.execution.process_graph.ResolvedProcessGraph.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact resolved-graph container and member types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L177-L183)

## particula.execution.process_graph.ResourceRequirement

*class*

```python
class ResourceRequirement(str, Enum)
```

Bases: `str`, `Enum`

Identify backend-neutral state required by a declared graph node.

These symbolic requirements describe state ownership without accessing a
resource or selecting a backend.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L48-L60)

### particula.execution.process_graph.ResourceRequirement.DIAGNOSTICS

*attribute* · *class attribute* · *instance attribute*

```python
DIAGNOSTICS = 'diagnostics'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L60-L60)

### particula.execution.process_graph.ResourceRequirement.ENVIRONMENT

*attribute* · *class attribute* · *instance attribute*

```python
ENVIRONMENT = 'environment'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L57-L57)

### particula.execution.process_graph.ResourceRequirement.GAS

*attribute* · *class attribute* · *instance attribute*

```python
GAS = 'gas'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L56-L56)

### particula.execution.process_graph.ResourceRequirement.PARTICLES

*attribute* · *class attribute* · *instance attribute*

```python
PARTICLES = 'particles'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L55-L55)

### particula.execution.process_graph.ResourceRequirement.PROCESS_SIDECARS

*attribute* · *class attribute* · *instance attribute*

```python
PROCESS_SIDECARS = 'process_sidecars'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L59-L59)

### particula.execution.process_graph.ResourceRequirement.THERMODYNAMICS

*attribute* · *class attribute* · *instance attribute*

```python
THERMODYNAMICS = 'thermodynamics'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L58-L58)

## particula.execution.process_graph.TimestepPlan

*class* · *dataclass*

```python
class TimestepPlan
```

Declare an immutable process graph before validation and normalization.

**Attributes**

- `nodes` (`tuple[ProcessNode, ...]`) — Supplied process-node declarations in caller-provided order.
- `dependencies` (`tuple[DependencyEdge, ...]`) — Supplied dependency declarations in caller-provided order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L144-L158)

### particula.execution.process_graph.TimestepPlan.dependencies

*attribute* · *instance attribute*

```python
dependencies: tuple[DependencyEdge, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L154-L154)

### particula.execution.process_graph.TimestepPlan.nodes

*attribute* · *instance attribute*

```python
nodes: tuple[ProcessNode, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L153-L153)

### particula.execution.process_graph.TimestepPlan.__init__

*method*

```python
def __init__(nodes: tuple[ProcessNode, ...], dependencies: tuple[DependencyEdge, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L0-L0)

### particula.execution.process_graph.TimestepPlan.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact graph-plan container and member types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L156-L158)

## particula.execution.process_graph._NodeSchema

*class* · *dataclass*

```python
class _NodeSchema
```

Retain one private closed-catalogue node declaration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L186-L194)

### particula.execution.process_graph._NodeSchema.invalidates

*attribute* · *instance attribute*

```python
invalidates: frozenset[InvalidatedState]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L194-L194)

### particula.execution.process_graph._NodeSchema.kind

*attribute* · *instance attribute*

```python
kind: NodeKind
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L191-L191)

### particula.execution.process_graph._NodeSchema.node_id

*attribute* · *instance attribute*

```python
node_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L190-L190)

### particula.execution.process_graph._NodeSchema.process

*attribute* · *instance attribute*

```python
process: Process | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L192-L192)

### particula.execution.process_graph._NodeSchema.resources

*attribute* · *instance attribute*

```python
resources: frozenset[ResourceRequirement]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L193-L193)

### particula.execution.process_graph._NodeSchema.__init__

*method*

```python
def __init__(node_id: str, kind: NodeKind, process: Process | None, resources: frozenset[ResourceRequirement], invalidates: frozenset[InvalidatedState]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L0-L0)

## particula.execution.process_graph._is_resolver_produced_graph

*function*

```python
def _is_resolver_produced_graph(graph: ResolvedProcessGraph) -> bool
```

Return whether ``graph`` is the exact result of plan resolution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L406-L408)

## particula.execution.process_graph._raise_for_cycle

*function*

```python
def _raise_for_cycle(nodes: tuple[ProcessNode, ...], edges: tuple[DependencyEdge, ...]) -> None
```

Raise an error when normalized declarations contain a cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L596-L601)

## particula.execution.process_graph._validate_dependencies

*function*

```python
def _validate_dependencies(dependencies: tuple[DependencyEdge, ...], node_ids: frozenset[str]) -> None
```

Validate supplied dependency declarations in their input order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L571-L593)

## particula.execution.process_graph._validate_frozenset_members

*function*

```python
def _validate_frozenset_members(values: object, member_type: type[Enum], field_name: str) -> None
```

Validate an exact frozenset containing only one enum type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L500-L511)

## particula.execution.process_graph._validate_node_id

*function*

```python
def _validate_node_id(value: object, field_name: str) -> None
```

Validate a canonical lower-case identifier-style node identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L492-L497)

## particula.execution.process_graph._validate_nodes

*function*

```python
def _validate_nodes(nodes: tuple[ProcessNode, ...]) -> frozenset[str]
```

Validate supplied nodes in order against the closed catalogue.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L534-L568)

## particula.execution.process_graph._validate_plan_members

*function*

```python
def _validate_plan_members(nodes: object, dependencies: object, owner: str) -> None
```

Validate exact tuple containers and their declared member types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L514-L531)

## particula.execution.process_graph._validate_topology_members

*function*

```python
def _validate_topology_members(nodes: object, dependencies: object) -> None
```

Validate exact topology utility containers and record members.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L473-L489)

## particula.execution.process_graph.resolve_canonical_topological_order

*function*

```python
def resolve_canonical_topological_order(nodes: tuple[ProcessNode, ...], dependencies: tuple[DependencyEdge, ...]) -> tuple[str, ...]
```

Return a deterministic canonical dependency order for graph records.

Ready nodes and successor traversal use lexical node identifiers, so the
result never depends on declaration order. This read-only helper validates
tuple/member shape, unique node IDs, and edge endpoints, but does not
validate the closed declaration catalogue, resolve scheduler policy, or
execute
graph nodes.

**Parameters**

- `nodes` (`tuple[ProcessNode, ...]`) — Exact tuple of declared graph nodes with unique identifiers.
- `dependencies` (`tuple[DependencyEdge, ...]`) — Exact tuple of dependency edges between declared nodes.

**Returns**

- (`tuple[str, ...]`) — Canonical topological node identifiers.

**Raises**

- `TypeError` — If containers or members do not have the declared types.
- `ValueError` — If node identifiers repeat, an endpoint is absent, or the
effective dependencies contain a cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L411-L470)

## particula.execution.process_graph.resolve_timestep_plan

*function*

```python
def resolve_timestep_plan(plan: TimestepPlan) -> ResolvedProcessGraph
```

Validate and deterministically normalize a declaration-only plan.

This pure boundary neither schedules nor executes the declared graph,
accesses resources, or selects an execution backend. For complete resident
graphs, the closed catalogue permits the communication-to-volume barrier
edges before every ordinary loop node; it does not acquire those resources
or mutate derived-state freshness.

**Parameters**

- `plan` (`TimestepPlan`) — Exact immutable plan declaration to validate and normalize.

**Returns**

- (`ResolvedProcessGraph`) — A new graph with nodes and dependencies in canonical declaration order.

**Raises**

- `TypeError` — If the plan is not a ``TimestepPlan``.
- `ValueError` — If declarations violate the closed catalogue, dependency
schema, or acyclic graph constraints.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_graph.py#L370-L403)
