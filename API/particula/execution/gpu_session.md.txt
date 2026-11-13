# particula.execution.gpu_session

*module*

Define concrete-only GPU-resident sessions and lifecycle bookkeeping.

Immutable carriers retain caller-owned Warp containers and CPU gas-name
metadata. ``setup_resident_session`` is the sole CPU-to-Warp upload boundary;
it preflights locally, then converts particles, gas, and environment once in
that order before publishing an ACTIVE session. Gas names remain CPU metadata.

``ResidentStepGuard`` and ``ResidentStepToken`` track one open timestep for an
exact active session-registry binding. They execute no adapters and perform no
implicit transfer, synchronization, allocation, fallback, or payload work.
Direct owners explicitly classify failures: read-only failures release the open
 token and retain ACTIVE state, while possible post-launch writer failures fault
 the session without rollback. Exact writer and terminal lifecycle events lazily
 notify an attached graph-capture owner to stale and release issued handles.
 ``close`` and ``discard`` only perform terminal lifecycle transitions after the
 guard closes; they never restore, synchronize, or mutate resident payloads.
 Checkpoint, restart, and raw low-level helpers remain separate concrete-only
 boundaries.

Each factory-created session also retains immutable host-only session metadata
stream
metadata. Setup validates that metadata before importing conversion helpers or
uploading CPU carriers; it does not allocate, initialize, inspect, or persist
an RNG sidecar. First coagulation resource acquisition owns that later,
coagulation-only initialization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1-L1)

## particula.execution.gpu_session.ResidentDimensions

*class* · *dataclass*

```python
class ResidentDimensions
```

Declare immutable dimensions for caller-owned resident state.

**Attributes**

- `n_boxes` (`int`) — Nonnegative number of resident boxes.
- `n_particles` (`int`) — Nonnegative fixed particle capacity per box.
- `n_species` (`int`) — Nonnegative number of particle material species.
- `n_gas_species` (`int`) — Explicit independent nonnegative gas width.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L224-L255)

### particula.execution.gpu_session.ResidentDimensions.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L235-L235)

### particula.execution.gpu_session.ResidentDimensions.n_gas_species

*attribute* · *instance attribute*

```python
n_gas_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L238-L238)

### particula.execution.gpu_session.ResidentDimensions.n_particles

*attribute* · *instance attribute*

```python
n_particles: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L236-L236)

### particula.execution.gpu_session.ResidentDimensions.n_species

*attribute* · *instance attribute*

```python
n_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L237-L237)

### particula.execution.gpu_session.ResidentDimensions.gas_species

*property*

```python
gas_species: int
```

Return the explicitly declared independent gas width.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L241-L243)

### particula.execution.gpu_session.ResidentDimensions.__init__

*method*

```python
def __init__(n_boxes: int, n_particles: int, n_species: int, n_gas_species: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L0-L0)

### particula.execution.gpu_session.ResidentDimensions.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the immutable resident dimensions.

**Raises**

- `TypeError` — If a dimension is not an integral value or is boolean.
- `ValueError` — If a dimension violates its required lower bound.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L245-L255)

## particula.execution.gpu_session.ResidentLifecycle

*class*

```python
class ResidentLifecycle(Enum)
```

Bases: `Enum`

Declare resident lifecycle vocabulary.

``ResidentSession`` starts ACTIVE. Checkpoint finalization transitions
ACTIVE to FINALIZED after caching an immutable checkpoint. A direct owner
can transition ACTIVE to FAULTED after a writer may have launched; close or
discard transitions ACTIVE or FAULTED to CLOSED. Faulting and closing do
not roll back resident payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L333-L346)

### particula.execution.gpu_session.ResidentLifecycle.ACTIVE

*attribute* · *class attribute* · *instance attribute*

```python
ACTIVE = 'active'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L343-L343)

### particula.execution.gpu_session.ResidentLifecycle.CLOSED

*attribute* · *class attribute* · *instance attribute*

```python
CLOSED = 'closed'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L346-L346)

### particula.execution.gpu_session.ResidentLifecycle.FAULTED

*attribute* · *class attribute* · *instance attribute*

```python
FAULTED = 'faulted'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L344-L344)

### particula.execution.gpu_session.ResidentLifecycle.FINALIZED

*attribute* · *class attribute* · *instance attribute*

```python
FINALIZED = 'finalized'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L345-L345)

## particula.execution.gpu_session.ResidentMetadata

*class* · *dataclass*

```python
class ResidentMetadata
```

Retain declared Warp-device and ordered gas-name metadata by identity.

**Attributes**

- `device` (`Device`) — Exact Warp-backed execution device declaration.
- `gas_names` (`tuple[str, ...]`) — Ordered caller-declared gas names; entries are not inspected or normalized at this boundary.
- `stream` (`ResidentStreamMetadata`) — Immutable host-only session metadata stream identity matching resident boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L362-L398)

### particula.execution.gpu_session.ResidentMetadata.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L374-L374)

### particula.execution.gpu_session.ResidentMetadata.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str = ''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L377-L377)

### particula.execution.gpu_session.ResidentMetadata.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L375-L375)

### particula.execution.gpu_session.ResidentMetadata.stream

*attribute* · *class attribute* · *instance attribute*

```python
stream: ResidentStreamMetadata = ResidentStreamMetadata(0, 0, (), ())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L376-L376)

### particula.execution.gpu_session.ResidentMetadata.__init__

*method*

```python
def __init__(device: Device, gas_names: tuple[str, ...], stream: ResidentStreamMetadata = ResidentStreamMetadata(0, 0, (), ()), distribution_type: str = '') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L0-L0)

### particula.execution.gpu_session.ResidentMetadata.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate fixed-cost metadata declarations without normalizing names.

**Raises**

- `TypeError` — If the device or name tuple has the wrong type.
- `ValueError` — If the device does not declare the Warp backend.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L379-L398)

## particula.execution.gpu_session.ResidentSession

*class* · *dataclass*

```python
class ResidentSession
```

Retain one validated caller-owned GPU-resident session by identity.

Construction performs only O(1) schema and device metadata validation. It
neither reads resident payloads nor transfers, allocates, synchronizes,
schedules, or operates on the retained lifecycle state. Concrete-only
checkpoint methods bind exact session, registry, and guard identities;
checkpoint is nonterminal, whereas successful finalization caches a
checkpoint and makes the session FINALIZED. They neither select a device nor
provide migration, CPU fallback, or rollback after launched device work.

**Attributes**

- `particles` (`object`) — Caller-owned generated Warp particle container.
- `gas` (`object`) — Caller-owned generated Warp gas container.
- `environment` (`object`) — Caller-owned generated Warp environment container.
- `dimensions` (`ResidentDimensions`) — Immutable dimensions that describe the retained containers.
- `metadata` (`ResidentMetadata`) — Immutable device, gas-name, and host-only stream metadata.
- `lifecycle` (`ResidentLifecycle`) — Current lifecycle value. Checkpoint finalization can change ACTIVE to FINALIZED; classified writer failures can fault ACTIVE; and close or discard can close ACTIVE or FAULTED.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L762-L1091)

### particula.execution.gpu_session.ResidentSession._lifecycle_lock

*attribute* · *class attribute* · *instance attribute*

```python
_lifecycle_lock: Any = field(init=False, repr=False, compare=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L791-L791)

### particula.execution.gpu_session.ResidentSession._native_operation

*attribute* · *class attribute* · *instance attribute*

```python
_native_operation: object | None = field(init=False, repr=False, compare=False, default=None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L795-L797)

### particula.execution.gpu_session.ResidentSession._terminal_intent

*attribute* · *class attribute* · *instance attribute*

```python
_terminal_intent: tuple[str, object, int] | None = field(init=False, repr=False, compare=False, default=None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L792-L794)

### particula.execution.gpu_session.ResidentSession.dimensions

*attribute* · *instance attribute*

```python
dimensions: ResidentDimensions
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L788-L788)

### particula.execution.gpu_session.ResidentSession.environment

*attribute* · *instance attribute*

```python
environment: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L787-L787)

### particula.execution.gpu_session.ResidentSession.gas

*attribute* · *instance attribute*

```python
gas: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L786-L786)

### particula.execution.gpu_session.ResidentSession.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: ResidentLifecycle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L790-L790)

### particula.execution.gpu_session.ResidentSession.metadata

*attribute* · *instance attribute*

```python
metadata: ResidentMetadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L789-L789)

### particula.execution.gpu_session.ResidentSession.particles

*attribute* · *instance attribute*

```python
particles: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L785-L785)

### particula.execution.gpu_session.ResidentSession.__init__

*method*

```python
def __init__(particles: object, gas: object, environment: object, dimensions: ResidentDimensions, metadata: ResidentMetadata, lifecycle: ResidentLifecycle) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L0-L0)

### particula.execution.gpu_session.ResidentSession.__post_init__

*method*

```python
def __post_init__() -> None
```

Perform ordered O(1) metadata-only resident-session preflight.

**Raises**

- `TypeError` — If carrier, lifecycle, or generated container types are
invalid.
- `ValueError` — If declared metadata, schema, or device values disagree.
- `RuntimeError` — If resident validation requires an unavailable Warp
runtime.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L799-L841)

### particula.execution.gpu_session.ResidentSession._acquire_native_operation

*method*

```python
def _acquire_native_operation(kind: str) -> object
```

Acquire this session's private capture/replay lifecycle lease.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L843-L859)

### particula.execution.gpu_session.ResidentSession._begin_terminal_transition

*method*

```python
def _begin_terminal_transition(intent: str) -> object
```

Publish terminal intent before any caller-controlled callback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L868-L879)

### particula.execution.gpu_session.ResidentSession._checkpoint_controller

*method*

```python
def _checkpoint_controller(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> Any
```

Return the exact checkpoint controller bound to this session.

The controller is intentionally attached outside the frozen data-carrier
fields so construction remains a metadata-only operation. Its binding is
exact: a different registry or guard is rejected rather than rebinding
the session or transferring ownership.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L891-L914)

### particula.execution.gpu_session.ResidentSession._end_terminal_transition

*method*

```python
def _end_terminal_transition(token: object) -> None
```

Clear the exact private terminal-transition intent.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L881-L889)

### particula.execution.gpu_session.ResidentSession._finalize_checkpoint

*method*

```python
def _finalize_checkpoint() -> None
```

Transition this exact active session to its terminal lifecycle.

This checkpoint-only helper performs no transfer, allocation, or
payload mutation.  Its narrow transition prevents a controller from
silently overwriting any lifecycle state other than ``ACTIVE``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1013-L1022)

### particula.execution.gpu_session.ResidentSession._release_native_operation

*method*

```python
def _release_native_operation(lease: object) -> None
```

Release the exact private capture/replay lifecycle lease.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L861-L866)

### particula.execution.gpu_session.ResidentSession.checkpoint

*method*

```python
def checkpoint(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> Any
```

Create a nonterminal immutable in-memory resident checkpoint.

The session, registry, and closed guard must be the controller's exact
active binding. The returned concrete-only record owns detached host
inspection carriers and canonical primary/sidecar bytes; this session
and all resident identities remain active and caller-owned.

**Parameters**

- `registry` (`GPUResourceRegistry`) — Exact registry pinned to this active session.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to the same session and registry.

**Returns**

- (`Any`) — A fresh immutable checkpoint record.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L916-L933)

### particula.execution.gpu_session.ResidentSession.close

*method*

```python
def close(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> None
```

Terminally discard this binding without touching resident payloads.

An active session validates its exact pinned binding once and requires a
closed guard before becoming CLOSED. A faulted session uses only exact
identity checks because active-only registry validation is deliberately
unavailable after a possible writer failure. CLOSED and FINALIZED are
idempotent write-free no-ops. Finalization retains its cached
 checkpoint. If attached graph-handle teardown fails, the session closes
 and the teardown error propagates.

**Parameters**

- `registry` (`GPUResourceRegistry`) — Exact registry pinned to this session.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to this session and registry.

**Raises**

- `TypeError` — If the registry or guard is not an exact concrete type.
- `ValueError` — If the binding is wrong or the lifecycle is invalid.
- `RuntimeError` — If a resident timestep remains open.
- `BaseException` — Propagates an attached graph-handle teardown failure
after transitioning this session to CLOSED.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1024-L1072)

### particula.execution.gpu_session.ResidentSession.discard

*method*

```python
def discard(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> None
```

Discard this binding using the same terminal semantics as ``close``.

This alias only changes lifecycle state when ``close`` permits it. It
does not restore, synchronize, allocate, or mutate resident payloads.

**Parameters**

- `registry` (`GPUResourceRegistry`) — Exact registry pinned to this session.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to this session and registry.

**Raises**

- `TypeError` — If the registry or guard is not an exact concrete type.
- `ValueError` — If the binding is wrong or the lifecycle is invalid.
- `RuntimeError` — If a resident timestep remains open.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1074-L1091)

### particula.execution.gpu_session.ResidentSession.finalize

*method*

```python
def finalize(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> Any
```

Create or return the terminal cached resident checkpoint.

The first successful call snapshots the exact active binding, caches the
resulting record, and transitions this session to FINALIZED. Later calls
return that identical record without further device work. A failure
before the transition leaves this session active; no rollback is
guaranteed after an already-launched device operation.

**Parameters**

- `registry` (`GPUResourceRegistry`) — Exact registry pinned to this active session.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to the same session and registry.

**Returns**

- (`Any`) — The immutable cached terminal checkpoint.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L935-L953)

### particula.execution.gpu_session.ResidentSession.initialize_streams

*method*

```python
def initialize_streams(registry: GPUResourceRegistry, guard: ResidentStepGuard, *, process_ids: tuple[str, ...] | None = None, logical_box_ids: tuple[str, ...] | None = None) -> None
```

Explicitly reinitialize selected published resident RNG streams.

The exact active session, registry, and closed guard are validated
before stream work. A stream-writer failure faults the exact binding
and notifies any attached graph-capture owner; no rollback is promised.

**Parameters**

- `registry` (`GPUResourceRegistry`) — Exact registry pinned to this active session.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to this session and registry.
- `process_ids` (`tuple[str, ...] | None`) (default: `None`) — Optional published processes to initialize.
- `logical_box_ids` (`tuple[str, ...] | None`) (default: `None`) — Optional stable logical box IDs to initialize.

**Raises**

- `TypeError` — If the binding or stream selection has an invalid type.
- `ValueError` — If the binding, lifecycle, or stream selection is
invalid.
- `BaseException` — Propagates a stream-writer failure after faulting the
binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L962-L995)

### particula.execution.gpu_session.ResidentSession.inspect_streams

*method*

```python
def inspect_streams(registry: GPUResourceRegistry, guard: ResidentStepGuard) -> Any
```

Return immutable metadata for published binding RNG streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L955-L960)

### particula.execution.gpu_session.ResidentSession.reset_streams

*method*

```python
def reset_streams(registry: GPUResourceRegistry, guard: ResidentStepGuard, *, process_ids: tuple[str, ...] | None = None, logical_box_ids: tuple[str, ...] | None = None) -> None
```

Alias :meth:`initialize_streams` for deliberate stream resets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L997-L1011)

## particula.execution.gpu_session.ResidentStepGuard

*class*

```python
class ResidentStepGuard
```

Track one open timestep for one pinned active resident session.

The guard retains one exact :class:`ResidentSession` and
:class:`GPUResourceRegistry` binding by identity. A successful
:meth:`begin_step` creates the sole open token; only successful matching
:meth:`complete_step` advances completed-step and simulated-time metadata.
The private failure seam can instead abort the exact token without advancing
either counter.

This concrete-only scheduler bookkeeping does not execute adapters,
transfer or restore data, synchronize Warp, acquire sidecars, resize
resident state, evolve the environment, or provide a CPU fallback. Future
checkpoint, finalize, close, conversion, resize, rebind, and fault
boundaries must call :meth:`assert_step_closed` before their own work. The
gate does not intercept callers that bypass those lifecycle boundaries with
raw low-level helpers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1114-L1347)

### particula.execution.gpu_session.ResidentStepGuard._completed_steps

*attribute* · *instance attribute*

```python
_completed_steps = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1160-L1160)

### particula.execution.gpu_session.ResidentStepGuard._open_token

*attribute* · *instance attribute*

```python
_open_token: ResidentStepToken | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1159-L1159)

### particula.execution.gpu_session.ResidentStepGuard._registry

*attribute* · *instance attribute*

```python
_registry = registry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1158-L1158)

### particula.execution.gpu_session.ResidentStepGuard._session

*attribute* · *instance attribute*

```python
_session = session
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1157-L1157)

### particula.execution.gpu_session.ResidentStepGuard._simulated_time

*attribute* · *instance attribute*

```python
_simulated_time: Any = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1164-L1164)

### particula.execution.gpu_session.ResidentStepGuard.completed_steps

*property*

```python
completed_steps: int
```

Return the number of successfully completed guarded timesteps.

**Returns**

- (`int`) — Count advanced only after successful matching completion.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1167-L1173)

### particula.execution.gpu_session.ResidentStepGuard.simulated_time

*property*

```python
simulated_time: Real
```

Return the sum of successfully completed timestep durations.

**Returns**

- (`Real`) — Uncoerced-real sum advanced only after successful matching
- (`Real`) — completion.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1176-L1183)

### particula.execution.gpu_session.ResidentStepGuard.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry) -> None
```

Create a closed guard for one exact active session-registry binding.

Construction validates the pinned binding without acquiring sidecars or
executing, transferring, synchronizing, or mutating resident state.

**Parameters**

- `session` (`ResidentSession`) — Exact active resident session retained by ``registry``.
- `registry` (`GPUResourceRegistry`) — Exact resource registry pinned to ``session``.

**Raises**

- `TypeError` — If either argument is not its exact concrete type.
- `ValueError` — If the registry does not retain ``session`` or its
active metadata-only signature is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1133-L1164)

### particula.execution.gpu_session.ResidentStepGuard._abort_step

*method*

```python
def _abort_step(token: ResidentStepToken) -> None
```

Release an exact open token without advancing timestep bookkeeping.

This private failure seam only closes the token held by this guard and
its pinned registry. It does not execute work, allocate, transfer,
synchronize, or roll back any device payload mutation.

**Parameters**

- `token` (`ResidentStepToken`) — Exact outstanding token owned by this guard.

**Raises**

- `ValueError` — If the active pinned binding or token identity is
invalid.
- `RuntimeError` — If no resident timestep is open.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1270-L1293)

### particula.execution.gpu_session.ResidentStepGuard._restore_checkpoint_counters

*method*

```python
def _restore_checkpoint_counters(completed_steps: object, simulated_time: object) -> None
```

Restore counters only into a fresh closed active guard binding.

This private restart seam changes bookkeeping only after a checkpoint
restart has established every fresh primary and acquired sidecar. It
does not restore arrays, synchronize, launch device work, or roll back a
launched device operation.

**Parameters**

- `completed_steps` (`object`) — Nonnegative count to restore exactly.
- `simulated_time` (`object`) — Finite nonnegative real duration to restore without coercion.

**Raises**

- `ValueError` — If the guard is not fresh and closed, its pinned binding
is inactive, or a counter violates its invariant.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1309-L1347)

### particula.execution.gpu_session.ResidentStepGuard._validate_duration

*method* · *staticmethod*

```python
def _validate_duration(duration: object) -> Any
```

Validate one nonnegative finite real duration without coercion.

**Parameters**

- `duration` (`object`) — Candidate timestep duration.

**Returns**

- (`Any`) — The validated duration unchanged.

**Raises**

- `TypeError` — If the duration is boolean or not a real number.
- `ValueError` — If the duration is negative or non-finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1185-L1203)

### particula.execution.gpu_session.ResidentStepGuard.assert_step_closed

*method*

```python
def assert_step_closed() -> None
```

Reject a lifecycle boundary while a guarded timestep remains open.

Future checkpoint, finalize, close, conversion, resize, rebind, and
fault boundaries must call this side-effect-free gate before their own
transfer, synchronization, mutation, or state transition. It does not
globally intercept direct raw helper calls.

**Raises**

- `RuntimeError` — If this guard has an outstanding timestep token.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1295-L1307)

### particula.execution.gpu_session.ResidentStepGuard.begin_step

*method*

```python
def begin_step(duration: object) -> ResidentStepToken
```

Open and return the sole token for a validated timestep duration.

The guard validates duration before its pinned active session-registry
binding. It performs bookkeeping only; callers execute any process work
outside this method and must complete the returned exact token only
after that work succeeds.

**Parameters**

- `duration` (`object`) — Nonnegative finite real timestep duration.

**Returns**

- (`ResidentStepToken`) — The new opaque token for the sole open timestep.

**Raises**

- `TypeError` — If ``duration`` is boolean or not a real number.
- `ValueError` — If ``duration`` is invalid or the pinned session binding
is no longer valid and active.
- `RuntimeError` — If another timestep is already open.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1205-L1237)

### particula.execution.gpu_session.ResidentStepGuard.complete_step

*method*

```python
def complete_step(token: ResidentStepToken) -> None
```

Complete the exact outstanding token and advance bookkeeping.

Completed-step count and simulated time change only after the current
pinned binding validates and ``token`` is the exact open token. This
method executes no adapter and supplies no failure recovery or rollback
for external process work.

**Parameters**

- `token` (`ResidentStepToken`) — Exact opaque token returned by this guard's open step.

**Raises**

- `ValueError` — If the pinned session binding is invalid or ``token``
does not match the open token by identity.
- `RuntimeError` — If no timestep is open.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1239-L1268)

## particula.execution.gpu_session.ResidentStepToken

*class* · *dataclass*

```python
class ResidentStepToken
```

Retain one guard-owned timestep identity and validated duration.

A token is opaque to scheduler callers. ``complete_step`` accepts only the
exact outstanding token by identity. Its private origin and duration fields
are guard bookkeeping, not a process-execution interface.
This concrete-only token does not execute work, transfer data, synchronize,
allocate resources, or provide a fallback.

**Attributes**

- `_guard` (`object`) — Guard that created this token.
- `_duration` (`Any`) — Validated nonnegative finite duration retained without coercion.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1094-L1111)

### particula.execution.gpu_session.ResidentStepToken._duration

*attribute* · *instance attribute*

```python
_duration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1111-L1111)

### particula.execution.gpu_session.ResidentStepToken._guard

*attribute* · *instance attribute*

```python
_guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1110-L1110)

### particula.execution.gpu_session.ResidentStepToken.__init__

*method*

```python
def __init__(_guard: object, _duration: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L0-L0)

## particula.execution.gpu_session.ResidentStreamMetadata

*class* · *dataclass*

```python
class ResidentStreamMetadata
```

Retain immutable host-only session metadata stream identity for one
    session.

This metadata validates session metadata root-seed, logical-box-ID, and
lane-permutation
rules without importing Warp. It identifies a future resident coagulation
stream but does not allocate, initialize, inspect, reset, synchronize, or
persist any native RNG state.

**Attributes**

- `n_boxes` (`int`) — Number of resident boxes represented by the metadata.
- `root_seed` (`int`) — Unsigned session metadata root seed.
- `logical_box_ids` (`tuple[str, ...]`) — Unique logical identifiers in box order.
- `lanes` (`tuple[int, ...]`) — session metadata lane permutation for the resident boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L258-L330)

### particula.execution.gpu_session.ResidentStreamMetadata.lanes

*attribute* · *instance attribute*

```python
lanes: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L279-L279)

### particula.execution.gpu_session.ResidentStreamMetadata.logical_box_ids

*attribute* · *instance attribute*

```python
logical_box_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L278-L278)

### particula.execution.gpu_session.ResidentStreamMetadata.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L276-L276)

### particula.execution.gpu_session.ResidentStreamMetadata.root_seed

*attribute* · *instance attribute*

```python
root_seed: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L277-L277)

### particula.execution.gpu_session.ResidentStreamMetadata.__init__

*method*

```python
def __init__(n_boxes: int, root_seed: int, logical_box_ids: tuple[str, ...], lanes: tuple[int, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L0-L0)

### particula.execution.gpu_session.ResidentStreamMetadata.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the host-only session metadata stream manifest metadata.

**Raises**

- `TypeError` — If metadata fields have invalid exact container or scalar
types.
- `ValueError` — If the seed, ID uniqueness, lengths, or lane permutation
violates the session metadata contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L281-L294)

### particula.execution.gpu_session.ResidentStreamMetadata._validate_root_seed

*method*

```python
def _validate_root_seed(maximum: int) -> None
```

Validate the immutable root seed against the session metadata
unsigned range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L296-L305)

### particula.execution.gpu_session.ResidentStreamMetadata._validate_stream_arrays

*method*

```python
def _validate_stream_arrays() -> None
```

Validate logical IDs and lane ordering against the box count.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L307-L330)

## particula.execution.gpu_session._ResidentOperationOutcome

*class*

```python
class _ResidentOperationOutcome(Enum)
```

Bases: `Enum`

Classify an operation-owner failure without inspecting its exception.

Direct operation owners must explicitly select this classification. A
read-only failure leaves a successfully aborted session active; a failure
after a writer may have launched faults it because resident payload mutation
is observable and no rollback is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L349-L359)

### particula.execution.gpu_session._ResidentOperationOutcome.READ_ONLY

*attribute* · *class attribute* · *instance attribute*

```python
READ_ONLY = 'read_only'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L358-L358)

### particula.execution.gpu_session._ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED

*attribute* · *class attribute* · *instance attribute*

```python
WRITER_MAY_HAVE_LAUNCHED = 'writer_may_have_launched'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L359-L359)

## particula.execution.gpu_session._fault_resident_session

*function*

```python
def _fault_resident_session(session: ResidentSession) -> None
```

Mark one resident session as faulted without any recovery work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1404-L1406)

## particula.execution.gpu_session._fault_resident_session_with_context

*function*

```python
def _fault_resident_session_with_context(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard) -> None
```

Notify graph capture, then fault even if notification fails.

**Parameters**

- `session` (`ResidentSession`) — Resident session to fault.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to ``session`` and ``registry``.

**Raises**

- `BaseException` — Propagates a graph-capture notification failure after
marking ``session`` faulted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1433-L1452)

## particula.execution.gpu_session._gas_shape

*function*

```python
def _gas_shape(gas: GasData) -> tuple[int, int]
```

Validate gas rank before using its independent width.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L117-L122)

## particula.execution.gpu_session._handle_failed_resident_operation

*function*

```python
def _handle_failed_resident_operation(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard, token: ResidentStepToken, outcome: _ResidentOperationOutcome) -> None
```

Release a failed operation token and apply its explicit outcome.

Direct operation owners call this only after opening ``token``. They retain
their operational exception while calling this seam, so a cleanup exception
can be chained as diagnostic context without replacing the operation error.
Classification is explicit rather than inferred from exception type.
READ_ONLY failures leave the session active after aborting the token; a
possible writer launch faults the session after that same abort. If writer
cleanup or active-binding validation fails, the session is faulted before
that cleanup error propagates. No device payload rollback, transfer,
synchronization, retry, or recovery occurs here.

**Parameters**

- `session` (`ResidentSession`) — Exact resident session owning the failure.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `guard` (`ResidentStepGuard`) — Exact guard bound to ``session`` and ``registry``.
- `token` (`ResidentStepToken`) — Exact open timestep token created by ``guard``.
- `outcome` (`_ResidentOperationOutcome`) — Explicit classification chosen by the direct owner.

**Raises**

- `TypeError` — If any argument is not the exact required concrete type.
- `ValueError` — If the binding, lifecycle, or token identity is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1487-L1527)

## particula.execution.gpu_session._notify_resident_graph_capture

*function*

```python
def _notify_resident_graph_capture(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard, cause: str) -> None
```

Lazily route an exact lifecycle event to graph-capture ownership.

This seam transfers no native handle and does not inspect graph state. The
graph-capture module validates the exact context and owns all teardown.

**Parameters**

- `session` (`ResidentSession`) — Resident session reporting the event.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `guard` (`ResidentStepGuard`) — Exact closed guard bound to ``session`` and ``registry``.
- `cause` (`str`) — Graph-capture teardown cause.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1409-L1430)

## particula.execution.gpu_session._preflight_cpu_session

*function*

```python
def _preflight_cpu_session(particles: object, gas: object, environment: object, device: object) -> tuple[ParticleData, GasData, EnvironmentData, ResidentDimensions, tuple[str, ...]]
```

Validate local CPU session inputs before importing conversion helpers.

This function intentionally does not probe native-device availability. That
policy is an upstream execution capability policy responsibility and the
selected device is
assumed to have been capability-approved before this boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L125-L221)

## particula.execution.gpu_session._reject_primary_aliases

*function*

```python
def _reject_primary_aliases(particles: object, gas: object, environment: object) -> None
```

Reject overlapping primary storage before publishing a session.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L664-L697)

## particula.execution.gpu_session._require_native_population_kind

*function*

```python
def _require_native_population_kind(particles: ParticleData) -> None
```

Reject ambiguous or unsupported population tags before payload access.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L99-L104)

## particula.execution.gpu_session._validate_active_stream_binding

*function*

```python
def _validate_active_stream_binding(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard) -> None
```

Require an exact active, closed session/registry/guard stream binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1377-L1401)

## particula.execution.gpu_session._validate_array

*function*

```python
def _validate_array(name: str, array: Any, dtype: Any, shape: tuple[int, ...], device: Any) -> None
```

Validate only one Warp primary-array metadata declaration.

**Parameters**

- `name` (`str`) — Fully qualified primary-array name for validation errors.
- `array` (`Any`) — Candidate Warp array whose metadata is inspected.
- `dtype` (`Any`) — Required Warp dtype.
- `shape` (`tuple[int, ...]`) — Required immutable array shape.
- `device` (`Any`) — Required shared Warp device.

**Raises**

- `ValueError` — If the array metadata is absent or differs from the schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L401-L442)

## particula.execution.gpu_session._validate_contiguous_array

*function*

```python
def _validate_contiguous_array(name: str, array: Any, shape: tuple[int, ...], item_size: int) -> None
```

Validate contiguous-stride primary metadata and nonempty pointers.

**Parameters**

- `name` (`str`) — Fully qualified array name used in validation errors.
- `array` (`Any`) — Candidate Warp array.
- `shape` (`tuple[int, ...]`) — Required immutable shape.
- `item_size` (`int`) — Element size in bytes.

**Raises**

- `ValueError` — If the array is not contiguous or its nonempty storage has
invalid pointer, alignment, or capacity metadata. Canonical empty
schemas require no pointer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L445-L491)

## particula.execution.gpu_session._validate_cpu_shape

*function*

```python
def _validate_cpu_shape(value: object, name: str, expected_shape: tuple[int, ...]) -> None
```

Validate one CPU field shape without reading or copying values.

**Parameters**

- `value` (`object`) — Candidate CPU-backed field.
- `name` (`str`) — Field name used in validation errors.
- `expected_shape` (`tuple[int, ...]`) — Required immutable shape.

**Raises**

- `ValueError` — If the value does not expose the required shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L80-L96)

## particula.execution.gpu_session._validate_dimension

*function*

```python
def _validate_dimension(value: object, name: str, *, positive: bool) -> None
```

Validate one immutable resident dimension.

**Parameters**

- `value` (`object`) — Candidate dimension value.
- `name` (`str`) — Dimension name used in validation errors.
- `positive` (`bool`) — Whether the value must be greater than zero instead of nonnegative.

**Raises**

- `TypeError` — If ``value`` is not an integral value or is a boolean.
- `ValueError` — If ``value`` violates its required lower bound.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L44-L62)

## particula.execution.gpu_session._validate_failed_operation_context

*function*

```python
def _validate_failed_operation_context(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard, token: ResidentStepToken, outcome: _ResidentOperationOutcome) -> None
```

Validate the exact resident failure context before cleanup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1455-L1484)

## particula.execution.gpu_session._validate_gas_names

*function*

```python
def _validate_gas_names(gas_names: object) -> None
```

Validate an exact tuple of exact string gas names.

**Parameters**

- `gas_names` (`object`) — Candidate ordered gas-name metadata.

**Raises**

- `TypeError` — If the metadata is not an exact tuple of exact strings.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L65-L77)

## particula.execution.gpu_session._validate_generated_containers

*function*

```python
def _validate_generated_containers(particles: object, gas: object, environment: object) -> None
```

Validate generated Warp structs without reading their payloads.

**Parameters**

- `particles` (`object`) — Candidate generated particle container.
- `gas` (`object`) — Candidate generated gas container.
- `environment` (`object`) — Candidate generated environment container.

**Raises**

- `TypeError` — If a container is not its required generated Warp struct.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L494-L523)

## particula.execution.gpu_session._validate_native_cpu_population

*function*

```python
def _validate_native_cpu_population(particles: ParticleData) -> None
```

Check explicit population admission before upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L107-L114)

## particula.execution.gpu_session._validate_resident_carriers

*function*

```python
def _validate_resident_carriers(dimensions: object, metadata: object, lifecycle: object) -> tuple[ResidentDimensions, ResidentMetadata, ResidentLifecycle]
```

Validate exact resident carrier declarations.

**Parameters**

- `dimensions` (`object`) — Declared resident dimensions.
- `metadata` (`object`) — Declared resident metadata.
- `lifecycle` (`object`) — Declared lifecycle value.

**Returns**

- (`tuple[ResidentDimensions, ResidentMetadata, ResidentLifecycle]`) — The validated resident carriers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L700-L738)

## particula.execution.gpu_session._validate_resident_imports

*function*

```python
def _validate_resident_imports() -> Any
```

Import Warp for resident checks and preserve missing-runtime errors.

**Returns**

- (`Any`) — The imported Warp module.

**Raises**

- `RuntimeError` — If Warp is unavailable.
- `ModuleNotFoundError` — If a different import name is missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L741-L759)

## particula.execution.gpu_session._validate_schema

*function*

```python
def _validate_schema(particles: object, gas: object, environment: object, dimensions: ResidentDimensions, metadata: ResidentMetadata, wp: Any) -> None
```

Validate fixed primary schemas and shared Warp-device metadata.

**Parameters**

- `particles` (`object`) — Validated generated particle container.
- `gas` (`object`) — Validated generated gas container.
- `environment` (`object`) — Validated generated environment container.
- `dimensions` (`ResidentDimensions`) — Declared resident dimensions to match the particle schema.
- `metadata` (`ResidentMetadata`) — Declared device metadata to match the shared Warp device.
- `wp` (`Any`) — Lazily imported Warp module.

**Raises**

- `ValueError` — If a primary array has incompatible metadata or device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L526-L661)

## particula.execution.gpu_session._validate_terminal_binding

*function*

```python
def _validate_terminal_binding(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard) -> None
```

Validate exact session, registry, and guard ownership identities.

Unlike ``GPUResourceRegistry.validate_pinned_session``, this local check is
valid for a FAULTED session and intentionally performs no lifecycle/schema
validation or runtime work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1350-L1374)

## particula.execution.gpu_session.setup_resident_session

*function*

```python
def setup_resident_session(particles: ParticleData, gas: GasData, environment: EnvironmentData, device: Device, *, root_seed: int = 0, logical_box_ids: tuple[str, ...] | None = None, lanes: tuple[int, ...] | None = None) -> ResidentSession
```

Upload CPU carriers once and publish one active resident session.

This concrete-only, unexported factory is session creation's only
CPU-to-Warp upload
point. Local CPU preflight completes before conversion helpers are imported.
It then converts particles, gas, and environment once in that order,
snapshots ordered gas names into immutable CPU-owned tuple metadata, and
publishes one complete, validated ACTIVE session retaining converted
containers by identity. It provides no lifecycle operations, fallback,
synchronization, restore, or process sidecars. The selected native Warp
device must already be availability-approved by upstream execution
capability policy; this
revision does not probe it.

**Parameters**

- `particles` (`ParticleData`) — CPU particle carrier to upload.
- `gas` (`GasData`) — CPU gas carrier to upload; its ordered names remain CPU metadata.
- `environment` (`EnvironmentData`) — CPU environment carrier to upload.
- `device` (`Device`) — Exact, upstream-approved Warp device declaration.
- `root_seed` (`int`) (default: `0`) — Unsigned session metadata root seed retained in immutable session metadata.
- `logical_box_ids` (`tuple[str, ...] | None`) (default: `None`) — Optional unique session metadata box IDs. ``None`` uses stringified ascending box indices.
- `lanes` (`tuple[int, ...] | None`) (default: `None`) — Optional session metadata lane permutation. ``None`` uses ascending box lanes.

**Returns**

- (`ResidentSession`) — One complete ACTIVE session retaining conversion results by identity.

**Raises**

- `TypeError` — If a local device, carrier, gas-name, or stream declaration
is invalid.
- `ValueError` — If the backend, CPU schema, or session metadata stream
metadata is
invalid.
- `RuntimeError` — If optional Warp conversion or session validation requires
an unavailable runtime.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_session.py#L1530-L1625)
