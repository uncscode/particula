# particula.execution.scheduler

*module*

Resolve direct-import-only, declaration-only scheduling metadata.

This concrete module validates immutable declaration graph declarations before
applying
enabled-node selection, direction policy, and freshness closure. Resolution is
prelaunch-only: it neither loads backends nor enters lifecycle state, allocates
resources, executes refreshes, or mutates caller-owned data. Its names are not
package exports and must be imported from this module directly.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L1-L1)

## particula.execution.scheduler._FRESHNESS_EDGES

*attribute* · *module attribute*

```python
_FRESHNESS_EDGES = (('environment_update', 'vapor_pressure_refresh'), ('environment_update', 'saturation_refresh'), ('gas_update', 'saturation_refresh'), ('environment_update', 'brownian_coagulation'), ('environment_update', 'wall_loss'), ('environment_update', 'nucleation'), ('gas_update', 'condensation'), ('gas_update', 'dilution'), ('gas_update', 'nucleation'), ('vapor_pressure_refresh', 'saturation_refresh'), ('saturation_refresh', 'condensation'), ('saturation_refresh', 'diagnostics'))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L31-L44)

## particula.execution.scheduler._NAME_PATTERN

*attribute* · *module attribute*

```python
_NAME_PATTERN = re.compile('^[a-z][a-z0-9_]*$')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L24-L24)

## particula.execution.scheduler._NUCLEATION_CONDENSATION_EDGES

*attribute* · *module attribute*

```python
_NUCLEATION_CONDENSATION_EDGES = frozenset({('nucleation', 'condensation'), ('condensation', 'nucleation')})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L25-L30)

## particula.execution.scheduler._RESOLVER_SCHEDULES

*attribute* · *module attribute*

```python
_RESOLVER_SCHEDULES: list[object] = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L46-L46)

## particula.execution.scheduler._SCHEDULE_PROVENANCE

*attribute* · *module attribute*

```python
_SCHEDULE_PROVENANCE = object()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L45-L45)

## particula.execution.scheduler.EnabledNodeSelection

*class* · *dataclass*

```python
class EnabledNodeSelection
```

Declare the complete immutable set of enabled declaration node
    identifiers.

Graph membership is deliberately deferred to schedule resolution, after
complete declaration plan validation.

**Parameters**

- `enabled_node_ids` (`frozenset[str]`) — Exact frozenset of syntactically valid node IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L49-L79)

### particula.execution.scheduler.EnabledNodeSelection.enabled_node_ids

*attribute* · *instance attribute*

```python
enabled_node_ids: frozenset[str]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L61-L61)

### particula.execution.scheduler.EnabledNodeSelection.__init__

*method*

```python
def __init__(enabled_node_ids: frozenset[str]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L0-L0)

### particula.execution.scheduler.EnabledNodeSelection.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate selection container and identifier syntax.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L63-L79)

## particula.execution.scheduler.NucleationCondensationDirection

*class*

```python
class NucleationCondensationDirection(str, Enum)
```

Bases: `str`, `Enum`

Declare the reviewed nucleation/condensation ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L82-L86)

### particula.execution.scheduler.NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION

*attribute* · *class attribute* · *instance attribute*

```python
CONDENSATION_THEN_NUCLEATION = 'condensation_then_nucleation'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L86-L86)

### particula.execution.scheduler.NucleationCondensationDirection.NUCLEATION_THEN_CONDENSATION

*attribute* · *class attribute* · *instance attribute*

```python
NUCLEATION_THEN_CONDENSATION = 'nucleation_then_condensation'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L85-L85)

## particula.execution.scheduler.ResolvedTimestepSchedule

*class* · *dataclass*

```python
class ResolvedTimestepSchedule
```

Store canonical, immutable declaration-only scheduling metadata.

Nodes and dependencies are canonically sorted. The order is a permutation
of node IDs created before any lifecycle entry, resource action, or process
launch.

**Parameters**

- `nodes` (`tuple[ProcessNode, ...]`) — Sorted exact tuple of enabled process nodes.
- `dependencies` (`tuple[DependencyEdge, ...]`) — Sorted exact tuple of effective dependency edges.
- `ordered_node_ids` (`tuple[str, ...]`) — Canonical dependency order for exactly these nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L114-L139)

### particula.execution.scheduler.ResolvedTimestepSchedule._provenance

*attribute* · *class attribute* · *instance attribute*

```python
_provenance: object | None = field(default=None, repr=False, compare=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L134-L134)

### particula.execution.scheduler.ResolvedTimestepSchedule.dependencies

*attribute* · *instance attribute*

```python
dependencies: tuple[DependencyEdge, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L129-L129)

### particula.execution.scheduler.ResolvedTimestepSchedule.nodes

*attribute* · *instance attribute*

```python
nodes: tuple[ProcessNode, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L128-L128)

### particula.execution.scheduler.ResolvedTimestepSchedule.ordered_node_ids

*attribute* · *instance attribute*

```python
ordered_node_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L130-L130)

### particula.execution.scheduler.ResolvedTimestepSchedule.source_graph

*attribute* · *class attribute* · *instance attribute*

```python
source_graph: ResolvedProcessGraph | None = field(default=None, repr=False, compare=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L131-L133)

### particula.execution.scheduler.ResolvedTimestepSchedule.__init__

*method*

```python
def __init__(nodes: tuple[ProcessNode, ...], dependencies: tuple[DependencyEdge, ...], ordered_node_ids: tuple[str, ...], source_graph: ResolvedProcessGraph | None = None, _provenance: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L0-L0)

### particula.execution.scheduler.ResolvedTimestepSchedule.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate canonical immutable schedule fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L136-L139)

## particula.execution.scheduler.SchedulerProfile

*class* · *dataclass*

```python
class SchedulerProfile
```

Declare immutable scheduler direction policy without selecting execution.

The enum represents exactly one reviewed nucleation/condensation direction;
it cannot encode both directions or no direction.

**Parameters**

- `nucleation_condensation_direction` (`NucleationCondensationDirection`) — The one permitted direction policy.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L89-L111)

### particula.execution.scheduler.SchedulerProfile.nucleation_condensation_direction

*attribute* · *instance attribute*

```python
nucleation_condensation_direction: NucleationCondensationDirection
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L100-L100)

### particula.execution.scheduler.SchedulerProfile.__init__

*method*

```python
def __init__(nucleation_condensation_direction: NucleationCondensationDirection) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L0-L0)

### particula.execution.scheduler.SchedulerProfile.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the single direction declaration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L102-L111)

## particula.execution.scheduler._add_freshness_pairs

*function*

```python
def _add_freshness_pairs(pairs: set[tuple[str, str]], enabled: frozenset[str]) -> None
```

Add freshness edges only when both endpoints are selected.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L367-L374)

## particula.execution.scheduler._direction_edge

*function*

```python
def _direction_edge(profile: SchedulerProfile) -> tuple[str, str]
```

Return the declared direction as one graph edge.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L377-L384)

## particula.execution.scheduler._raise_unsatisfied_closure

*function*

```python
def _raise_unsatisfied_closure(before_id: str, after_id: str) -> None
```

Raise the stable error for a missing required predecessor.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L387-L391)

## particula.execution.scheduler._resolve_effective_pairs

*function*

```python
def _resolve_effective_pairs(resolved: ResolvedProcessGraph, selection: EnabledNodeSelection, profile: SchedulerProfile, direction: tuple[str, str]) -> set[tuple[str, str]]
```

Combine retained declarations, profile policy, and freshness closure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L327-L348)

## particula.execution.scheduler._retained_pairs

*function*

```python
def _retained_pairs(resolved: ResolvedProcessGraph, enabled: frozenset[str]) -> set[tuple[str, str]]
```

Return selected declaration edges after enforcing explicit dependency
closure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L351-L364)

## particula.execution.scheduler._validate_profile_direction

*function*

```python
def _validate_profile_direction(resolved: ResolvedProcessGraph, direction: tuple[str, str]) -> None
```

Reject an explicit declaration direction that opposes the profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L314-L324)

## particula.execution.scheduler._validate_resolution_inputs

*function*

```python
def _validate_resolution_inputs(plan: object, selection: object, profile: object) -> None
```

Validate public scheduler carriers before declaration resolution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L283-L294)

## particula.execution.scheduler._validate_schedule_content

*function*

```python
def _validate_schedule_content(schedule: ResolvedTimestepSchedule) -> None
```

Validate canonical schedule ordering, endpoints, and permutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L171-L215)

## particula.execution.scheduler._validate_schedule_types

*function*

```python
def _validate_schedule_types(schedule: ResolvedTimestepSchedule) -> None
```

Validate exact immutable schedule container and member types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L142-L168)

## particula.execution.scheduler._validate_selected_ids

*function*

```python
def _validate_selected_ids(selection: EnabledNodeSelection, resolved: ResolvedProcessGraph) -> None
```

Reject selected IDs that are absent from an already resolved declaration
graph.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L297-L311)

## particula.execution.scheduler.is_resolver_produced_schedule

*function*

```python
def is_resolver_produced_schedule(schedule: ResolvedTimestepSchedule, graph: ResolvedProcessGraph) -> bool
```

Return whether a schedule retains its exact resolver-produced graph.

This internal provenance check prevents a structurally similar hand-built
schedule from entering the resident execution boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L270-L280)

## particula.execution.scheduler.resolve_timestep_schedule

*function*

```python
def resolve_timestep_schedule(plan: TimestepPlan, selection: EnabledNodeSelection, profile: SchedulerProfile) -> ResolvedTimestepSchedule
```

Resolve an immutable effective schedule without running any process.

After exact carrier-type checks, complete declaration graph validation
happens before
selection or profile inspection. This function then applies selected IDs,
direction policy, required freshness closure, and canonical topology
ordering. It returns metadata only and performs no lifecycle, resource, or
GPU work.

**Parameters**

- `plan` (`TimestepPlan`) — Exact declaration plan declaration.
- `selection` (`EnabledNodeSelection`) — Exact enabled-node selection.
- `profile` (`SchedulerProfile`) — Exact immutable direction policy.

**Returns**

- (`ResolvedTimestepSchedule`) — A new canonical schedule with no disabled dependency endpoints.

**Raises**

- `TypeError` — If a carrier is not its exact declared type.
- `ValueError` — If selection, direction, closure, or effective topology is
invalid. No input is mutated when resolution fails; no rollback is
required because resolution has no side effects.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/scheduler.py#L218-L267)
