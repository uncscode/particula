# particula.execution

*module*

Declare dependency-neutral public execution selection and error contracts.

This module validates immutable backend, device, process, and capability
declarations. An ``ExecutionContext`` capability-validates a typed request and
returns one exact context-local adapter. Selection does not invoke adapters,
load or probe optional backends, resolve native devices, transfer state, or
validate state ownership state or result contracts.

state ownership result validation retains caller-owned state by identity, keeps
backend
payloads and results opaque, and records explicit in-place mutation permission.
It does not import a backend, provide fallback behavior, transfer state, or
execute adapters. Selection remains context-local and callable-only.

The direct-module-only CPU adapter is a narrow dispatch dispatch boundary. It
preflights concrete CPU state and execution controls, delegates exactly once to
a caller-supplied CPU runnable, and retains the original state and aerosol by
identity. It does not select adapters, validate process physics, transfer or
convert data, or load optional backends.

The direct-module-only condensation profile catalogue declares semantic support
only. It neither selects an adapter nor validates runtime or native-device
Availability, imports optional backends, allocates state, or exposes a
user-facing API.

The public API includes selection, capability errors, fallback policies, and
the caller-owned ``CPUExecutionContext`` for CPU box random streams.
Its three public fallback policy enums are
``FallbackPolicy``, ``FallbackBoundary``, and ``CPUStateAuthority``. Concrete
fallback mechanics and their carriers remain direct-module-only. Importing this
package does not import optional backends, Warp, or ``particula.gpu``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1-L1)

## particula.execution.Backend

*class*

```python
class Backend(str, Enum)
```

Bases: `str`, `Enum`

Identify a declared execution backend without loading it.

**Attributes**

- `CPU` — The declared CPU backend.
- `WARP` — The declared Warp backend.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L82-L91)

### particula.execution.Backend.CPU

*attribute* · *class attribute* · *instance attribute*

```python
CPU = 'cpu'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L90-L90)

### particula.execution.Backend.WARP

*attribute* · *class attribute* · *instance attribute*

```python
WARP = 'warp'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L91-L91)

## particula.execution.BackendResult

*class* · *dataclass*

```python
class BackendResult
```

Retain opaque backend result values by identity without inspection.

**Parameters**

- `value` (`object`) — Opaque backend-owned result value.
- `diagnostics` (`object | None`) (default: `None`) — Optional opaque backend-owned diagnostic value.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L851-L861)

### particula.execution.BackendResult.diagnostics

*attribute* · *class attribute* · *instance attribute*

```python
diagnostics: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L861-L861)

### particula.execution.BackendResult.value

*attribute* · *instance attribute*

```python
value: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L860-L860)

### particula.execution.BackendResult.__init__

*method*

```python
def __init__(value: object, diagnostics: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

## particula.execution.CPUExecutionAdapter

*class*

```python
class CPUExecutionAdapter
```

Dispatch one validated CPU state to one caller-supplied runnable.

The adapter accepts only an exact ``CPUExecutionState``, locally validates
finite nonnegative duration and a positive integral substep count, then
delegates once using those state-held objects. It requires the runnable to
return the original aerosol and reports ``MutationScope.STATE``. This
direct-module-only boundary neither selects adapters nor transfers,
converts, or otherwise validates process state. The stored runnable owns
aerosol and process-physics validation.

**Parameters**

- `runnable` (`_CPURunnable | RunnableABC`) — Runnable invoked once for every valid execution state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1018-L1092)

### particula.execution.CPUExecutionAdapter._runnable

*attribute* · *instance attribute*

```python
_runnable = runnable
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1040-L1040)

### particula.execution.CPUExecutionAdapter.__init__

*method*

```python
def __init__(runnable: _CPURunnable | RunnableABC) -> None
```

Retain the runnable by identity without inspecting it.

**Parameters**

- `runnable` (`_CPURunnable | RunnableABC`) — CPU runnable invoked once by :meth:`execute` for each valid state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1033-L1040)

### particula.execution.CPUExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Dispatch one concrete CPU state and retain its identity.

This method performs local state and control preflight before one
positional runnable call. It does not split time steps, catch runnable
exceptions, inspect aerosol physics, select adapters, or use a
conversion or fallback path.

**Parameters**

- `state` (`ExecutionState`) — Exact ``CPUExecutionState`` containing the aerosol and execution controls.

**Returns**

- (`ExecutionResult`) — An ``ExecutionResult`` retaining the original state and aerosol by
- (`ExecutionResult`) — identity, with ``MutationScope.STATE`` and no metadata.

**Raises**

- `TypeError` — If state is not an exact CPU state or time_step is not a
non-boolean real scalar.
- `ValueError` — If controls are invalid or the runnable returns a
different aerosol object.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1042-L1092)

## particula.execution.CPUExecutionContext

*class*

```python
class CPUExecutionContext
```

Re-exported from: `particula.execution.cpu_context`

Retain persistent process-specific streams keyed by logical box ID.

**Parameters**

- `root_seed` (`int`) — Nonnegative integer seed; booleans are not accepted.
- `box_ids` (`tuple[str, ...]`) — Unique nonempty strings in current numeric-array box order.

Reordering numeric storage requires a matching ``with_box_ids`` view.
Reusing a context and process namespace deliberately continues its streams.
Create a new context to reset them. Contexts are not thread-safe and must
not be used concurrently. No CPU/GPU bitwise equivalence is implied.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L29-L90)

### particula.execution.CPUExecutionContext._box_ids

*attribute* · *instance attribute*

```python
_box_ids: tuple[str, ...] = _box_ids(box_ids)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L51-L51)

### particula.execution.CPUExecutionContext._root_seed

*attribute* · *instance attribute*

```python
_root_seed: int = int(root_seed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L50-L50)

### particula.execution.CPUExecutionContext._streams

*attribute* · *instance attribute*

```python
_streams: dict[tuple[str, str], np.random.Generator] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L52-L52)

### particula.execution.CPUExecutionContext.box_ids

*property*

```python
box_ids: tuple[str, ...]
```

Return logical IDs in this view's numeric-array order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L60-L62)

### particula.execution.CPUExecutionContext.root_seed

*property*

```python
root_seed: int
```

Return the immutable root seed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L55-L57)

### particula.execution.CPUExecutionContext.__init__

*method*

```python
def __init__(root_seed: int, box_ids: tuple[str, ...]) -> None
```

Validate metadata without creating or advancing random streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L42-L52)

### particula.execution.CPUExecutionContext._begin

*method*

```python
def _begin(n_boxes: int) -> _CPUStreamTransaction
```

Validate layout and begin a private substep transaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L74-L78)

### particula.execution.CPUExecutionContext._new_generator

*method*

```python
def _new_generator(namespace: str, box_id: str) -> np.random.Generator
```

Derive a versioned PCG64 stream independently of lane/order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L80-L90)

### particula.execution.CPUExecutionContext.with_box_ids

*method*

```python
def with_box_ids(box_ids: tuple[str, ...]) -> CPUExecutionContext
```

Create a reordered, reduced, or extended continuation-sharing view.

New IDs acquire streams lazily; existing IDs retain continuation even
when absent from a view. The caller must reorder arrays consistently.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L64-L72)

## particula.execution.CPUExecutionState

*class* · *dataclass*

```python
class CPUExecutionState
```

Retain CPU execution inputs for the direct CPU adapter.

Construction is intentionally side-effect-free: control validation occurs
at ``CPUExecutionAdapter.execute`` so invalid states can reach that
boundary. The carrier neither copies nor validates the aerosol or controls.

**Parameters**

- `aerosol` (`object`) — Caller-owned aerosol supplied to the CPU runnable.
- `time_step` (`object`) — Caller-supplied total execution duration.
- `sub_steps` (`object`) — Caller-supplied number of runnable substeps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L993-L1015)

### particula.execution.CPUExecutionState.aerosol

*attribute* · *instance attribute*

```python
aerosol: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1007-L1007)

### particula.execution.CPUExecutionState.context

*attribute* · *class attribute* · *instance attribute*

```python
context: object = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1010-L1010)

### particula.execution.CPUExecutionState.sub_steps

*attribute* · *instance attribute*

```python
sub_steps: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1009-L1009)

### particula.execution.CPUExecutionState.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1008-L1008)

### particula.execution.CPUExecutionState.backend_payload

*property*

```python
backend_payload: object
```

Return the caller-owned aerosol as the opaque CPU payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1013-L1015)

### particula.execution.CPUExecutionState.__init__

*method*

```python
def __init__(aerosol: object, time_step: object, sub_steps: object, context: object = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

## particula.execution.CPUStateAuthority

*class*

```python
class CPUStateAuthority(str, Enum)
```

Bases: `str`, `Enum`

Re-exported from: `particula.execution.values`

Declare the asserted authority of CPU state for a fallback request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L23-L29)

### particula.execution.CPUStateAuthority.CPU_AUTHORITATIVE

*attribute* · *class attribute* · *instance attribute*

```python
CPU_AUTHORITATIVE = 'cpu_authoritative'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L26-L26)

### particula.execution.CPUStateAuthority.MUTATED

*attribute* · *class attribute* · *instance attribute*

```python
MUTATED = 'mutated'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L29-L29)

### particula.execution.CPUStateAuthority.RESIDENT

*attribute* · *class attribute* · *instance attribute*

```python
RESIDENT = 'resident'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L27-L27)

### particula.execution.CPUStateAuthority.UPLOADED

*attribute* · *class attribute* · *instance attribute*

```python
UPLOADED = 'uploaded'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L28-L28)

## particula.execution.Capability

*class* · *dataclass*

```python
class Capability
```

Declare an immutable, validated capability name.

**Parameters**

- `name` (`str`) — Lowercase identifier-style capability name.

**Raises**

- `TypeError` — If ``name`` is not a str.
- `ValueError` — If ``name`` is not a lowercase identifier-style name.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L155-L176)

### particula.execution.Capability.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L167-L167)

### particula.execution.Capability.__init__

*method*

```python
def __init__(name: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.Capability.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the capability name.

**Raises**

- `TypeError` — If the name is not a str.
- `ValueError` — If the name is not a lowercase identifier-style name.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L169-L176)

## particula.execution.CapabilityDeclaration

*class* · *dataclass*

```python
class CapabilityDeclaration
```

Declare immutable capability support for one device and process.

**Parameters**

- `device` (`Device`) — Device receiving the declaration.
- `process` (`Process`) — Process supported by the device.
- `requirements` (`CapabilityRequirements`) — Exact required capability set for the support entry.

**Raises**

- `TypeError` — If any field is not its declared metadata type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L244-L275)

### particula.execution.CapabilityDeclaration.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L257-L257)

### particula.execution.CapabilityDeclaration.process

*attribute* · *instance attribute*

```python
process: Process
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L258-L258)

### particula.execution.CapabilityDeclaration.requirements

*attribute* · *instance attribute*

```python
requirements: CapabilityRequirements
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L259-L259)

### particula.execution.CapabilityDeclaration.__init__

*method*

```python
def __init__(device: Device, process: Process, requirements: CapabilityRequirements) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.CapabilityDeclaration.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the declaration's typed value objects.

**Raises**

- `TypeError` — If any declaration field has the wrong type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L261-L275)

## particula.execution.CapabilityMatrix

*class* · *dataclass*

```python
class CapabilityMatrix
```

Provide immutable, pure lookup of exact capability declarations.

Nonempty requirements must match one complete declaration. Empty
requirements are supported when at least one declaration has the requested
device and process. Lookup neither probes backend availability nor selects
an execution adapter.

**Parameters**

- `declarations` (`frozenset[CapabilityDeclaration]`) — Exact frozenset of capability declarations.

**Raises**

- `TypeError` — If ``declarations`` is not a frozenset of declarations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L278-L368)

### particula.execution.CapabilityMatrix.declarations

*attribute* · *instance attribute*

```python
declarations: frozenset[CapabilityDeclaration]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L294-L294)

### particula.execution.CapabilityMatrix.__init__

*method*

```python
def __init__(declarations: frozenset[CapabilityDeclaration]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.CapabilityMatrix.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact immutable declaration collection type.

**Raises**

- `TypeError` — If the collection or one of its members is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L296-L313)

### particula.execution.CapabilityMatrix.require

*method*

```python
def require(device: Device, process: Process, requirements: CapabilityRequirements) -> None
```

Require a declared capability request without executing it.

**Parameters**

- `device` (`Device`) — Device requested for the process.
- `process` (`Process`) — Process requested on the device.
- `requirements` (`CapabilityRequirements`) — Exact capability requirements for the request.

**Raises**

- `TypeError` — If an argument is not its declared metadata type.
- `ValueError` — If the valid request is not declared by this matrix.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L345-L368)

### particula.execution.CapabilityMatrix.supports

*method*

```python
def supports(device: Device, process: Process, requirements: CapabilityRequirements) -> bool
```

Return whether the matrix exactly declares a capability request.

**Parameters**

- `device` (`Device`) — Device requested for the process.
- `process` (`Process`) — Process requested on the device.
- `requirements` (`CapabilityRequirements`) — Exact capability requirements for the request.

**Returns**

- (`bool`) — True if the request is declared by this matrix; otherwise, False.

**Raises**

- `TypeError` — If an argument is not its declared metadata type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L315-L343)

## particula.execution.CapabilityRequirements

*class* · *dataclass*

```python
class CapabilityRequirements
```

Declare an immutable exact set of required capabilities.

The empty set is valid. Requirements are declarations, not composable
features inferred by the matrix.

**Parameters**

- `values` (`frozenset[Capability]`) — Exact frozenset of capability declarations.

**Raises**

- `TypeError` — If ``values`` is not a frozenset of Capability instances.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L211-L241)

### particula.execution.CapabilityRequirements.values

*attribute* · *instance attribute*

```python
values: frozenset[Capability]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L225-L225)

### particula.execution.CapabilityRequirements.__init__

*method*

```python
def __init__(values: frozenset[Capability]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.CapabilityRequirements.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact immutable capability collection type.

**Raises**

- `TypeError` — If the collection or one of its members is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L227-L241)

## particula.execution.CondensationActivityMode

*class*

```python
class CondensationActivityMode(str, Enum)
```

Bases: `str`, `Enum`

Declare the semantic condensation activity-coefficient mode.

``NONREPRESENTABLE`` is a valid semantic category that a profile may not
support. It does not inspect or detect concrete activity implementations.

**Attributes**

- `IDEAL` — Ideal activity-coefficient semantics.
- `KAPPA` — Kappa activity-coefficient semantics.
- `NONREPRESENTABLE` — Semantics not representable by a profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L386-L400)

### particula.execution.CondensationActivityMode.IDEAL

*attribute* · *class attribute* · *instance attribute*

```python
IDEAL = 'ideal'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L398-L398)

### particula.execution.CondensationActivityMode.KAPPA

*attribute* · *class attribute* · *instance attribute*

```python
KAPPA = 'kappa'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L399-L399)

### particula.execution.CondensationActivityMode.NONREPRESENTABLE

*attribute* · *class attribute* · *instance attribute*

```python
NONREPRESENTABLE = 'nonrepresentable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L400-L400)

## particula.execution.CondensationConfiguration

*class* · *dataclass*

```python
class CondensationConfiguration
```

Declare immutable semantic condensation configuration metadata.

This metadata does not inspect or translate strategy objects, device names,
optional-backend sidecars, or concrete activity and surface implementations.
It also does not select an adapter or determine runtime availability.

**Parameters**

- `execution_mode` (`CondensationExecutionMode`) — Semantic time-stepping mode.
- `latent_heat` (`bool`) — Whether latent-heat semantics are required.
- `activity_mode` (`CondensationActivityMode`) — Semantic activity-coefficient mode.
- `surface_mode` (`CondensationSurfaceMode`) — Semantic surface-tension mode.

**Raises**

- `TypeError` — If a field is not its exact declared metadata type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L420-L463)

### particula.execution.CondensationConfiguration.activity_mode

*attribute* · *instance attribute*

```python
activity_mode: CondensationActivityMode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L440-L440)

### particula.execution.CondensationConfiguration.execution_mode

*attribute* · *instance attribute*

```python
execution_mode: CondensationExecutionMode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L438-L438)

### particula.execution.CondensationConfiguration.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L439-L439)

### particula.execution.CondensationConfiguration.surface_mode

*attribute* · *instance attribute*

```python
surface_mode: CondensationSurfaceMode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L441-L441)

### particula.execution.CondensationConfiguration.__init__

*method*

```python
def __init__(execution_mode: CondensationExecutionMode, latent_heat: bool, activity_mode: CondensationActivityMode, surface_mode: CondensationSurfaceMode) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.CondensationConfiguration.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact semantic configuration field types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L443-L463)

## particula.execution.CondensationExecutionMode

*class*

```python
class CondensationExecutionMode(str, Enum)
```

Bases: `str`, `Enum`

Declare the semantic condensation time-stepping mode.

This vocabulary describes configuration semantics only; it does not select
an execution implementation.

**Attributes**

- `EQUAL_STEP` — Equal-step condensation semantics.
- `STAGGERED` — Staggered condensation semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L371-L383)

### particula.execution.CondensationExecutionMode.EQUAL_STEP

*attribute* · *class attribute* · *instance attribute*

```python
EQUAL_STEP = 'equal_step'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L382-L382)

### particula.execution.CondensationExecutionMode.STAGGERED

*attribute* · *class attribute* · *instance attribute*

```python
STAGGERED = 'staggered'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L383-L383)

## particula.execution.CondensationSurfaceMode

*class*

```python
class CondensationSurfaceMode(str, Enum)
```

Bases: `str`, `Enum`

Declare the semantic condensation surface-tension mode.

``NONREPRESENTABLE`` is a valid semantic category that a profile may not
support. It does not inspect or detect concrete surface implementations.

**Attributes**

- `STATIC` — Static surface-tension semantics.
- `COMPOSITION_WEIGHTED` — Composition-weighted surface-tension semantics.
- `NONREPRESENTABLE` — Semantics not representable by a profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L403-L417)

### particula.execution.CondensationSurfaceMode.COMPOSITION_WEIGHTED

*attribute* · *class attribute* · *instance attribute*

```python
COMPOSITION_WEIGHTED = 'composition_weighted'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L416-L416)

### particula.execution.CondensationSurfaceMode.NONREPRESENTABLE

*attribute* · *class attribute* · *instance attribute*

```python
NONREPRESENTABLE = 'nonrepresentable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L417-L417)

### particula.execution.CondensationSurfaceMode.STATIC

*attribute* · *class attribute* · *instance attribute*

```python
STATIC = 'static'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L415-L415)

## particula.execution.Device

*class* · *dataclass*

```python
class Device
```

Declare an immutable backend and opaque native device identifier.

The native identifier is retained verbatim and is never parsed, resolved,
or checked against backend availability.

**Parameters**

- `backend` (`Backend`) — Declared backend for the device.
- `native` (`str`) — Nonempty native identifier without surrounding whitespace.

**Raises**

- `TypeError` — If ``backend`` is not a Backend or ``native`` is not a str.
- `ValueError` — If ``native`` is empty or has surrounding whitespace.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L94-L128)

### particula.execution.Device.backend

*attribute* · *instance attribute*

```python
backend: Backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L110-L110)

### particula.execution.Device.native

*attribute* · *instance attribute*

```python
native: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L111-L111)

### particula.execution.Device.__init__

*method*

```python
def __init__(backend: Backend, native: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.Device.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the declared backend and opaque native identifier.

**Raises**

- `TypeError` — If the backend or native identifier has the wrong type.
- `ValueError` — If the native identifier is empty or padded.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L113-L128)

## particula.execution.ExecutionAdapter

*class*

```python
class ExecutionAdapter(Protocol)
```

Bases: `Protocol`

Declare the structural registration and selection seam.

Registration validates that an adapter exposes a callable ``execute``
attribute without invoking or probing it. Selection returns the registered
adapter by identity and does not establish an execution, state, or result
contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L759-L771)

### particula.execution.ExecutionAdapter.execute

*property*

```python
execute: Callable[..., object]
```

Return the callable execution seam without fixing its signature.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L770-L771)

## particula.execution.ExecutionCapabilityError

*class*

```python
class ExecutionCapabilityError(Exception)
```

Bases: `Exception`

Re-exported from: `particula.execution.errors`

Describe an unavailable, unsupported, or invalid execution request.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L24-L109)

### particula.execution.ExecutionCapabilityError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.ExecutionCapabilityError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.ExecutionCapabilityError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.ExecutionCapabilityError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.ExecutionCapabilityError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.ExecutionCapabilityError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.ExecutionCapabilityError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.ExecutionCapabilityError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.ExecutionCapabilityError.__str__

*method*

```python
def __str__() -> str
```

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.ExecutionCapabilityReason

*class*

```python
class ExecutionCapabilityReason(str, Enum)
```

Bases: `str`, `Enum`

Re-exported from: `particula.execution.errors`

Closed reason codes for execution capability errors.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L11-L21)

### particula.execution.ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED

*attribute* · *class attribute* · *instance attribute*

```python
CAPABILITY_UNSUPPORTED = 'capability_unsupported'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L19-L19)

### particula.execution.ExecutionCapabilityReason.DEVICE_UNAVAILABLE

*attribute* · *class attribute* · *instance attribute*

```python
DEVICE_UNAVAILABLE = 'device_unavailable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L17-L17)

### particula.execution.ExecutionCapabilityReason.FALLBACK_DISALLOWED

*attribute* · *class attribute* · *instance attribute*

```python
FALLBACK_DISALLOWED = 'fallback_disallowed'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L21-L21)

### particula.execution.ExecutionCapabilityReason.INVALID_STATE

*attribute* · *class attribute* · *instance attribute*

```python
INVALID_STATE = 'invalid_state'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L20-L20)

### particula.execution.ExecutionCapabilityReason.PROCESS_UNSUPPORTED

*attribute* · *class attribute* · *instance attribute*

```python
PROCESS_UNSUPPORTED = 'process_unsupported'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L18-L18)

### particula.execution.ExecutionCapabilityReason.RUNTIME_UNAVAILABLE

*attribute* · *class attribute* · *instance attribute*

```python
RUNTIME_UNAVAILABLE = 'runtime_unavailable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L16-L16)

### particula.execution.ExecutionCapabilityReason.UNKNOWN_BACKEND

*attribute* · *class attribute* · *instance attribute*

```python
UNKNOWN_BACKEND = 'unknown_backend'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L14-L14)

### particula.execution.ExecutionCapabilityReason.UNKNOWN_DEVICE

*attribute* · *class attribute* · *instance attribute*

```python
UNKNOWN_DEVICE = 'unknown_device'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L15-L15)

## particula.execution.ExecutionContext

*class*

```python
class ExecutionContext
```

Register and select declared adapters without executing or transferring.

Each context owns its private registry, preventing mutable module-global
registration state. Registration and selection are context-local only:
neither invokes an adapter, probes availability, loads a backend, nor
transfers data.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1167-L1235)

### particula.execution.ExecutionContext._matrix

*attribute* · *instance attribute*

```python
_matrix = matrix
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1187-L1187)

### particula.execution.ExecutionContext._registry

*attribute* · *instance attribute*

```python
_registry = _AdapterRegistry()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1188-L1188)

### particula.execution.ExecutionContext.__init__

*method*

```python
def __init__(matrix: CapabilityMatrix) -> None
```

Store a matrix and create an empty private adapter registry.

**Parameters**

- `matrix` (`CapabilityMatrix`) — Immutable capability declarations used for selection.

**Raises**

- `TypeError` — If ``matrix`` is not a CapabilityMatrix.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1176-L1188)

### particula.execution.ExecutionContext.register_adapter

*method*

```python
def register_adapter(process: Process, backend: Backend, adapter: ExecutionAdapter) -> None
```

Register one context-local adapter for selection only.

Registration never invokes or probes the adapter, transfers data, or
loads an optional backend. The adapter is available only from this
context and is selected later by its exact process/backend pair.

**Parameters**

- `process` (`Process`) — Process to associate with the adapter.
- `backend` (`Backend`) — Backend to associate with the adapter.
- `adapter` (`ExecutionAdapter`) — Structurally callable adapter to register.

**Raises**

- `TypeError` — If arguments are invalid in process, backend, adapter
order.
- `ValueError` — If the process/backend pair is already registered.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1190-L1212)

### particula.execution.ExecutionContext.resolve

*method*

```python
def resolve(request: ExecutionRequest) -> ExecutionAdapter
```

Return one exact adapter after validation without invoking it.

Capability support is checked before the single registry lookup. This
method does not catch adapter errors because it never executes adapters.

**Parameters**

- `request` (`ExecutionRequest`) — Typed selection request.

**Returns**

- (`ExecutionAdapter`) — The registered adapter object by identity.

**Raises**

- `TypeError` — If request is not an ExecutionRequest.
- `ValueError` — If CPU device spelling is invalid or support is absent.
- `LookupError` — If supported request has no exact registered adapter.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L1214-L1235)

## particula.execution.ExecutionRequest

*class* · *dataclass*

```python
class ExecutionRequest
```

Declare a typed request for adapter selection without execution.

This value only describes selection. It neither probes a backend, transfers
state, nor establishes execution state or result contracts.

**Parameters**

- `backend` (`Backend`) — Backend requested for selection.
- `device` (`Device`) — Device requested on the backend.
- `process` (`Process`) — Process requested on the device.
- `requirements` (`CapabilityRequirements`) — Exact capability requirements for the process.

**Raises**

- `TypeError` — If a field is not its declared metadata type.
- `ValueError` — If the backend does not match the device backend.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L699-L743)

### particula.execution.ExecutionRequest.backend

*attribute* · *instance attribute*

```python
backend: Backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L717-L717)

### particula.execution.ExecutionRequest.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L718-L718)

### particula.execution.ExecutionRequest.process

*attribute* · *instance attribute*

```python
process: Process
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L719-L719)

### particula.execution.ExecutionRequest.requirements

*attribute* · *instance attribute*

```python
requirements: CapabilityRequirements
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L720-L720)

### particula.execution.ExecutionRequest.__init__

*method*

```python
def __init__(backend: Backend, device: Device, process: Process, requirements: CapabilityRequirements) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.ExecutionRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate request metadata without resolving or executing it.

**Raises**

- `TypeError` — If a field is not its declared metadata type.
- `ValueError` — If backend and device backend differ.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L722-L743)

## particula.execution.ExecutionResult

*class* · *dataclass*

```python
class ExecutionResult
```

Declare immutable state ownership result ownership and mutation metadata.

``validate_execution_result`` verifies this carrier at the execution
boundary, including that ``state`` is the original caller-owned object.

**Parameters**

- `state` (`ExecutionState`) — Execution state expected to be the original caller-owned object.
- `metadata` (`tuple[tuple[str, str], ...]`) — Ordered immutable key-value result metadata for validation.
- `mutation` (`MutationDeclaration`) — Explicit permission for in-place state mutation.
- `backend_result` (`BackendResult | None`) (default: `None`) — Optional opaque backend result retained by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L864-L881)

### particula.execution.ExecutionResult.backend_result

*attribute* · *class attribute* · *instance attribute*

```python
backend_result: BackendResult | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L881-L881)

### particula.execution.ExecutionResult.metadata

*attribute* · *instance attribute*

```python
metadata: tuple[tuple[str, str], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L879-L879)

### particula.execution.ExecutionResult.mutation

*attribute* · *instance attribute*

```python
mutation: MutationDeclaration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L880-L880)

### particula.execution.ExecutionResult.state

*attribute* · *instance attribute*

```python
state: ExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L878-L878)

### particula.execution.ExecutionResult.__init__

*method*

```python
def __init__(state: ExecutionState, metadata: tuple[tuple[str, str], ...], mutation: MutationDeclaration, backend_result: BackendResult | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

## particula.execution.ExecutionState

*class*

```python
class ExecutionState(Protocol)
```

Bases: `Protocol`

Describe retained caller-owned state with an opaque backend payload.

The payload is not inspected by this module. Boundary validation requires
the same state object supplied by the caller and retains it by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L746-L756)

### particula.execution.ExecutionState.backend_payload

*property*

```python
backend_payload: object
```

Return opaque backend-owned payload state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L755-L756)

## particula.execution.FallbackBoundary

*class*

```python
class FallbackBoundary(str, Enum)
```

Bases: `str`, `Enum`

Re-exported from: `particula.execution.values`

Declare the caller-visible boundary for an explicit fallback request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L16-L20)

### particula.execution.FallbackBoundary.PRE_UPLOAD

*attribute* · *class attribute* · *instance attribute*

```python
PRE_UPLOAD = 'pre_upload'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L19-L19)

### particula.execution.FallbackBoundary.RESTORED

*attribute* · *class attribute* · *instance attribute*

```python
RESTORED = 'restored'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L20-L20)

## particula.execution.FallbackDisallowedError

*class*

```python
class FallbackDisallowedError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Re-exported from: `particula.execution.errors`

Report a fallback request outside its permitted boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L272-L303)

### particula.execution.FallbackDisallowedError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.FallbackDisallowedError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.FallbackDisallowedError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.FallbackDisallowedError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.FallbackDisallowedError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.FallbackDisallowedError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.FallbackDisallowedError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.FallbackDisallowedError.__init__

*method*

```python
def __init__(fallback_boundary: str, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None) -> None
```

Initialize a fallback-disallowed error.

**Parameters**

- `fallback_boundary` (`str`) — Boundary that disallows the fallback request.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L275-L303)

### particula.execution.FallbackDisallowedError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.FallbackPolicy

*class*

```python
class FallbackPolicy(str, Enum)
```

Bases: `str`, `Enum`

Re-exported from: `particula.execution.values`

Declare whether an eligible capability error is re-raised.

Or dispatched to CPU fallback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L6-L13)

### particula.execution.FallbackPolicy.CPU

*attribute* · *class attribute* · *instance attribute*

```python
CPU = 'cpu'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L13-L13)

### particula.execution.FallbackPolicy.RAISE

*attribute* · *class attribute* · *instance attribute*

```python
RAISE = 'raise'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L12-L12)

## particula.execution.InvalidExecutionStateError

*class*

```python
class InvalidExecutionStateError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Re-exported from: `particula.execution.errors`

Report an invalid state for an execution request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L241-L269)

### particula.execution.InvalidExecutionStateError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.InvalidExecutionStateError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.InvalidExecutionStateError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.InvalidExecutionStateError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.InvalidExecutionStateError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.InvalidExecutionStateError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.InvalidExecutionStateError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.InvalidExecutionStateError.__init__

*method*

```python
def __init__(state: str, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None) -> None
```

Initialize an invalid-execution-state error.

**Parameters**

- `state` (`str`) — Execution state that is invalid for the request.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L244-L269)

### particula.execution.InvalidExecutionStateError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.MutationDeclaration

*class* · *dataclass*

```python
class MutationDeclaration
```

Declare exactly one explicit state-mutation permission.

**Parameters**

- `scopes` (`frozenset[MutationScope]`) — Exact frozenset containing either ``NONE`` or ``STATE``.

**Raises**

- `TypeError` — If scopes is not an exact frozenset of MutationScope values.
- `ValueError` — If scopes is empty or combines mutation scopes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L786-L802)

### particula.execution.MutationDeclaration.scopes

*attribute* · *instance attribute*

```python
scopes: frozenset[MutationScope]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L798-L798)

### particula.execution.MutationDeclaration.__init__

*method*

```python
def __init__(scopes: frozenset[MutationScope]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.MutationDeclaration.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the closed, immutable mutation declaration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L800-L802)

## particula.execution.MutationScope

*class*

```python
class MutationScope(str, Enum)
```

Bases: `str`, `Enum`

Declare the one allowed mutation scope for an execution result.

``STATE`` permits in-place mutation of retained state. State ownership does
not infer or
verify that a declared mutation occurred.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L774-L783)

### particula.execution.MutationScope.NONE

*attribute* · *class attribute* · *instance attribute*

```python
NONE = 'none'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L782-L782)

### particula.execution.MutationScope.STATE

*attribute* · *class attribute* · *instance attribute*

```python
STATE = 'state'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L783-L783)

## particula.execution.Process

*class* · *dataclass*

```python
class Process
```

Declare an immutable, validated process name.

**Parameters**

- `name` (`str`) — Lowercase identifier-style process name.

**Raises**

- `TypeError` — If ``name`` is not a str.
- `ValueError` — If ``name`` is not a lowercase identifier-style name.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L131-L152)

### particula.execution.Process.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L143-L143)

### particula.execution.Process.__init__

*method*

```python
def __init__(name: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L0-L0)

### particula.execution.Process.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the process name.

**Raises**

- `TypeError` — If the name is not a str.
- `ValueError` — If the name is not a lowercase identifier-style name.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L145-L152)

## particula.execution.UnavailableDeviceError

*class*

```python
class UnavailableDeviceError(UnavailableExecutionTargetError)
```

Bases: `particula.execution.errors.UnavailableExecutionTargetError`

Re-exported from: `particula.execution.errors`

Report a device that is unavailable for execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L171-L185)

### particula.execution.UnavailableDeviceError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnavailableDeviceError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnavailableDeviceError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnavailableDeviceError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnavailableDeviceError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnavailableDeviceError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnavailableDeviceError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnavailableDeviceError.__init__

*method*

```python
def __init__(device: str, *, backend: str | None = None) -> None
```

Initialize an unavailable-device error.

**Parameters**

- `device` (`str`) — Device that is unavailable for execution.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L174-L185)

### particula.execution.UnavailableDeviceError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnavailableExecutionTargetError

*class*

```python
class UnavailableExecutionTargetError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Re-exported from: `particula.execution.errors`

Base category for unavailable execution targets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L116-L117)

### particula.execution.UnavailableExecutionTargetError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnavailableExecutionTargetError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnavailableExecutionTargetError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnavailableExecutionTargetError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnavailableExecutionTargetError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnavailableExecutionTargetError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnavailableExecutionTargetError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnavailableExecutionTargetError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.UnavailableExecutionTargetError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnavailableRuntimeError

*class*

```python
class UnavailableRuntimeError(UnavailableExecutionTargetError)
```

Bases: `particula.execution.errors.UnavailableExecutionTargetError`

Re-exported from: `particula.execution.errors`

Report an execution backend whose runtime is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L156-L168)

### particula.execution.UnavailableRuntimeError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnavailableRuntimeError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnavailableRuntimeError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnavailableRuntimeError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnavailableRuntimeError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnavailableRuntimeError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnavailableRuntimeError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnavailableRuntimeError.__init__

*method*

```python
def __init__(backend: str) -> None
```

Initialize an unavailable-runtime error.

**Parameters**

- `backend` (`str`) — Backend whose required runtime is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L159-L168)

### particula.execution.UnavailableRuntimeError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnknownBackendError

*class*

```python
class UnknownBackendError(UnknownExecutionTargetError)
```

Bases: `particula.execution.errors.UnknownExecutionTargetError`

Re-exported from: `particula.execution.errors`

Report a backend identifier that is not recognized.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L124-L136)

### particula.execution.UnknownBackendError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnknownBackendError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnknownBackendError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnknownBackendError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnknownBackendError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnknownBackendError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnknownBackendError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnknownBackendError.__init__

*method*

```python
def __init__(backend: str) -> None
```

Initialize an unknown-backend error.

**Parameters**

- `backend` (`str`) — Unrecognized execution backend identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L127-L136)

### particula.execution.UnknownBackendError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnknownDeviceError

*class*

```python
class UnknownDeviceError(UnknownExecutionTargetError)
```

Bases: `particula.execution.errors.UnknownExecutionTargetError`

Re-exported from: `particula.execution.errors`

Report a device identifier that is not recognized.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L139-L153)

### particula.execution.UnknownDeviceError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnknownDeviceError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnknownDeviceError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnknownDeviceError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnknownDeviceError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnknownDeviceError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnknownDeviceError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnknownDeviceError.__init__

*method*

```python
def __init__(device: str, *, backend: str | None = None) -> None
```

Initialize an unknown-device error.

**Parameters**

- `device` (`str`) — Unrecognized execution device identifier.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L142-L153)

### particula.execution.UnknownDeviceError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnknownExecutionTargetError

*class*

```python
class UnknownExecutionTargetError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Re-exported from: `particula.execution.errors`

Base category for unknown execution targets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L112-L113)

### particula.execution.UnknownExecutionTargetError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnknownExecutionTargetError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnknownExecutionTargetError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnknownExecutionTargetError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnknownExecutionTargetError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnknownExecutionTargetError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnknownExecutionTargetError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnknownExecutionTargetError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.UnknownExecutionTargetError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnsupportedCapabilityError

*class*

```python
class UnsupportedCapabilityError(UnsupportedExecutionRequestError)
```

Bases: `particula.execution.errors.UnsupportedExecutionRequestError`

Re-exported from: `particula.execution.errors`

Report an execution capability that is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L213-L238)

### particula.execution.UnsupportedCapabilityError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnsupportedCapabilityError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnsupportedCapabilityError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnsupportedCapabilityError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnsupportedCapabilityError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnsupportedCapabilityError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnsupportedCapabilityError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnsupportedCapabilityError.__init__

*method*

```python
def __init__(capability: str, *, backend: str | None = None, device: str | None = None, process: str | None = None) -> None
```

Initialize an unsupported-capability error.

**Parameters**

- `capability` (`str`) — Execution capability that is not supported.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L216-L238)

### particula.execution.UnsupportedCapabilityError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnsupportedExecutionRequestError

*class*

```python
class UnsupportedExecutionRequestError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Re-exported from: `particula.execution.errors`

Base category for unsupported execution requests.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L120-L121)

### particula.execution.UnsupportedExecutionRequestError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnsupportedExecutionRequestError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnsupportedExecutionRequestError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnsupportedExecutionRequestError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnsupportedExecutionRequestError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnsupportedExecutionRequestError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnsupportedExecutionRequestError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnsupportedExecutionRequestError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.UnsupportedExecutionRequestError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.UnsupportedProcessError

*class*

```python
class UnsupportedProcessError(UnsupportedExecutionRequestError)
```

Bases: `particula.execution.errors.UnsupportedExecutionRequestError`

Re-exported from: `particula.execution.errors`

Report an execution process that is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L188-L210)

### particula.execution.UnsupportedProcessError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.UnsupportedProcessError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.UnsupportedProcessError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.UnsupportedProcessError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.UnsupportedProcessError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.UnsupportedProcessError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.UnsupportedProcessError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.UnsupportedProcessError.__init__

*method*

```python
def __init__(process: str, *, backend: str | None = None, device: str | None = None) -> None
```

Initialize an unsupported-process error.

**Parameters**

- `process` (`str`) — Execution process that is not supported.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L191-L210)

### particula.execution.UnsupportedProcessError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.condensation_profile_supports

*function*

```python
def condensation_profile_supports(backend: Backend, configuration: CondensationConfiguration) -> bool
```

Return whether a fixed backend profile declares a configuration.

This pure catalogue query does not accept or parse a native device, select
an adapter, import an optional backend, or determine runtime availability.

**Parameters**

- `backend` (`Backend`) — Catalogue backend profile to query.
- `configuration` (`CondensationConfiguration`) — Semantic condensation configuration to query.

**Returns**

- (`bool`) — True when the exact configuration is declared by the backend profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L624-L646)

## particula.execution.get_condensation_requirements

*function*

```python
def get_condensation_requirements(configuration: CondensationConfiguration) -> CapabilityRequirements
```

Return the exact four-axis semantic requirements for a configuration.

The pure mapping does not compose declarations, select an adapter, or
inspect runtime, device, or optional-backend state.

**Parameters**

- `configuration` (`CondensationConfiguration`) — Validated semantic condensation configuration.

**Returns**

- (`CapabilityRequirements`) — A new requirements value containing exactly one capability per axis.

**Raises**

- `TypeError` — If ``configuration`` is not a CondensationConfiguration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L523-L557)

## particula.execution.require_condensation_profile

*function*

```python
def require_condensation_profile(backend: Backend, configuration: CondensationConfiguration) -> None
```

Require that a fixed backend profile declares a configuration.

This pure catalogue check preserves the matrix's unsupported-declaration
error. It does not accept or parse a native device, select an adapter,
import an optional backend, or determine runtime availability.

**Parameters**

- `backend` (`Backend`) — Catalogue backend profile to query.
- `configuration` (`CondensationConfiguration`) — Semantic condensation configuration to require.

**Raises**

- `TypeError` — If either argument is not its declared metadata type.
- `ValueError` — If the configuration is not declared by the profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L649-L673)

## particula.execution.validate_execution_result

*function*

```python
def validate_execution_result(original_state: object, result: object) -> ExecutionResult
```

Validate and return an immutable state ownership result without executing
    an adapter.

**Parameters**

- `original_state` (`object`) — Caller-owned state supplied before execution.
- `result` (`object`) — Candidate result that retains the original state by identity.

**Returns**

- (`ExecutionResult`) — The exact validated result object.

**Raises**

- `TypeError` — If state, result layout, metadata, or declarations are
invalid.
- `ValueError` — If state was replaced or metadata is otherwise invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/__init__.py#L927-L963)

**Modules**

- `particula.execution.adapters`
- `particula.execution.availability`
- `particula.execution.checkpoint`
- `particula.execution.communication`
- `particula.execution.cpu_context`
- `particula.execution.diagnostics`
- `particula.execution.errors`
- `particula.execution.fallback`
- `particula.execution.gpu_resources`
- `particula.execution.gpu_session`
- `particula.execution.graph_capture`
- `particula.execution.process_adapters`
- `particula.execution.process_graph`
- `particula.execution.resident_communication`
- `particula.execution.resident_enqueue`
- `particula.execution.resident_scheduler`
- `particula.execution.rng`
- `particula.execution.scheduler`
- `particula.execution.state_updates`
- `particula.execution.thermodynamic_updates`
- `particula.execution.validation`
- `particula.execution.values`
