# particula.execution.checkpoint

*module*

Provide concrete-only immutable in-memory resident-session checkpoints.

This direct-import-only module snapshots an active resident Warp session after
its guard is closed. A snapshot contains immutable host bytes for canonical
primary arrays and ordered registry-owned continuation state: acquired
sidecars, resident ledgers, diagnostics, and at most one complete closed-map
communication family. It excludes arbitrary caller-owned outputs. Detached CPU
inspection carriers are non-authoritative. Checkpointing needs approximately
one additional host copy of resident payload bytes and the inspection copies.

Restart is explicit and same-device: it creates a fresh compatible session from
the canonical bytes, never migrates data, serializes to disk, chooses a device,
or falls back to CPU execution. Schema v5 records explicit population semantics
and independent species widths. Older schemas reject before setup or upload;
there is no inferred migration. A snapshot may retain one complete closed-map
GAS or PARTICLES communication family and its optional final-volume sidecar.
``checkpoint()`` is nonterminal; ``finalize()`` caches a snapshot and ends
normal session use. Finalization tears down any issued graph handle; teardown
failure faults the source session and does not cache a final checkpoint.
No rollback is promised once a device operation has launched.

Schema v5 requires published-stream continuation metadata and an explicit
acquired RNG-process inventory, with optional coagulation and wall-loss current
words after the single synchronization boundary. Words and acquired families
must agree. They are the exact-device continuation authority; normal dispatch
and reacquisition retain restored arrays until reset is explicitly requested.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L1-L1)

## particula.execution.checkpoint._PRIMARY_ROLES

*attribute* · *module attribute*

```python
_PRIMARY_ROLES = (('particles', 'masses'), ('particles', 'concentration'), ('particles', 'charge'), ('particles', 'density'), ('particles', 'volume'), ('gas', 'molar_mass'), ('gas', 'concentration'), ('gas', 'vapor_pressure'), ('gas', 'partitioning'), ('environment', 'temperature'), ('environment', 'pressure'), ('environment', 'saturation_ratio'))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L55-L68)

## particula.execution.checkpoint._SCHEMA_VERSION

*attribute* · *module attribute*

```python
_SCHEMA_VERSION = 5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L54-L54)

## particula.execution.checkpoint.restart_checkpoint

*attribute* · *module attribute*

```python
restart_checkpoint = restart_resident_session
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L1268-L1268)

## particula.execution.checkpoint.CheckpointPayload

*class* · *dataclass*

```python
class CheckpointPayload
```

Store one immutable exact host representation of a resident array.

Payload bytes are canonical recovery state, never a live Warp reference or
mutable NumPy alias. They may describe a primary carrier field or an
acquired registry sidecar.

**Attributes**

- `family` (`str`) — Primary carrier family or acquired registry-owned resource family.
- `role` (`str`) — Array role within its family.
- `dtype` (`str`) — Exact NumPy-compatible dtype spelling.
- `shape` (`tuple[int, ...]`) — Exact array shape.
- `data` (`bytes`) — Immutable contiguous array bytes.
- `capacity` (`int | None`) — Optional collision or communication-edge capacity metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L71-L94)

### particula.execution.checkpoint.CheckpointPayload.capacity

*attribute* · *class attribute* · *instance attribute*

```python
capacity: int | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L94-L94)

### particula.execution.checkpoint.CheckpointPayload.data

*attribute* · *instance attribute*

```python
data: bytes
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L93-L93)

### particula.execution.checkpoint.CheckpointPayload.dtype

*attribute* · *instance attribute*

```python
dtype: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L91-L91)

### particula.execution.checkpoint.CheckpointPayload.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L89-L89)

### particula.execution.checkpoint.CheckpointPayload.role

*attribute* · *instance attribute*

```python
role: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L90-L90)

### particula.execution.checkpoint.CheckpointPayload.shape

*attribute* · *instance attribute*

```python
shape: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L92-L92)

### particula.execution.checkpoint.CheckpointPayload.__init__

*method*

```python
def __init__(family: str, role: str, dtype: str, shape: tuple[int, ...], data: bytes, capacity: int | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint.CommunicationCheckpointMetadata

*class* · *dataclass*

```python
class CommunicationCheckpointMetadata
```

Describe an optional pinned communication family without live arrays.

This schema-v5 metadata identifies the sole restored communication mode and
its fixed edge capacity. It is validated against canonical sidecar payloads
before restart setup; it neither owns nor retains source-device arrays.

**Attributes**

- `mode` (`str`) — Canonical ``"gas"`` or ``"particles"`` transport mode.
- `form` (`str`) — Canonical closed-map communication form.
- `edge_capacity` (`int`) — Fixed number of resident communication edges.
- `has_final_volumes` (`bool`) — Whether a prescribed-volume payload is present.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L157-L175)

### particula.execution.checkpoint.CommunicationCheckpointMetadata.edge_capacity

*attribute* · *instance attribute*

```python
edge_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L174-L174)

### particula.execution.checkpoint.CommunicationCheckpointMetadata.form

*attribute* · *instance attribute*

```python
form: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L173-L173)

### particula.execution.checkpoint.CommunicationCheckpointMetadata.has_final_volumes

*attribute* · *instance attribute*

```python
has_final_volumes: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L175-L175)

### particula.execution.checkpoint.CommunicationCheckpointMetadata.mode

*attribute* · *instance attribute*

```python
mode: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L172-L172)

### particula.execution.checkpoint.CommunicationCheckpointMetadata.__init__

*method*

```python
def __init__(mode: str, form: str, edge_capacity: int, has_final_volumes: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint.RNGContinuationCheckpoint

*class* · *dataclass*

```python
class RNGContinuationCheckpoint
```

Store required schema-v5 stream metadata and optional continuation words.

The payloads, rather than root-seed-derived initialization, are the
authority for restored published stream state. Explicit stream reset is the
only operation that may subsequently derive new words from this metadata.

**Attributes**

- `stream_schema_version` (`int`) — Exact RNG stream schema version.
- `root_seed` (`int`) — Seed used only by explicit stream initialization or reset.
- `logical_box_ids` (`tuple[str, ...]`) — Ordered unique logical resident-box identifiers.
- `lanes` (`tuple[int, ...]`) — Ordered unique stream lanes corresponding to logical box IDs.
- `descriptors` (`tuple[Any, ...]`) — Canonical descriptors for published process streams.
- `payloads` (`tuple[RNGContinuationPayload, ...]`) — Immutable current-word payloads in canonical process order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L195-L217)

### particula.execution.checkpoint.RNGContinuationCheckpoint.descriptors

*attribute* · *instance attribute*

```python
descriptors: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L216-L216)

### particula.execution.checkpoint.RNGContinuationCheckpoint.lanes

*attribute* · *instance attribute*

```python
lanes: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L215-L215)

### particula.execution.checkpoint.RNGContinuationCheckpoint.logical_box_ids

*attribute* · *instance attribute*

```python
logical_box_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L214-L214)

### particula.execution.checkpoint.RNGContinuationCheckpoint.payloads

*attribute* · *instance attribute*

```python
payloads: tuple[RNGContinuationPayload, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L217-L217)

### particula.execution.checkpoint.RNGContinuationCheckpoint.root_seed

*attribute* · *instance attribute*

```python
root_seed: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L213-L213)

### particula.execution.checkpoint.RNGContinuationCheckpoint.stream_schema_version

*attribute* · *instance attribute*

```python
stream_schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L212-L212)

### particula.execution.checkpoint.RNGContinuationCheckpoint.__init__

*method*

```python
def __init__(stream_schema_version: int, root_seed: int, logical_box_ids: tuple[str, ...], lanes: tuple[int, ...], descriptors: tuple[Any, ...], payloads: tuple[RNGContinuationPayload, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint.RNGContinuationPayload

*class* · *dataclass*

```python
class RNGContinuationPayload
```

Store immutable current RNG words for one published process.

**Attributes**

- `process_id` (`str`) — Canonical published process identifier.
- `dtype` (`str`) — Exact NumPy dtype spelling for the current-word array.
- `shape` (`tuple[int, ...]`) — Exact current-word array shape.
- `data` (`bytes`) — Immutable little-endian ``uint32`` current-word bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L178-L192)

### particula.execution.checkpoint.RNGContinuationPayload.data

*attribute* · *instance attribute*

```python
data: bytes
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L192-L192)

### particula.execution.checkpoint.RNGContinuationPayload.dtype

*attribute* · *instance attribute*

```python
dtype: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L190-L190)

### particula.execution.checkpoint.RNGContinuationPayload.process_id

*attribute* · *instance attribute*

```python
process_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L189-L189)

### particula.execution.checkpoint.RNGContinuationPayload.shape

*attribute* · *instance attribute*

```python
shape: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L191-L191)

### particula.execution.checkpoint.RNGContinuationPayload.__init__

*method*

```python
def __init__(process_id: str, dtype: str, shape: tuple[int, ...], data: bytes) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint.ResidentCheckpoint

*class* · *dataclass*

```python
class ResidentCheckpoint
```

Retain a versioned immutable inspection and same-device recovery state.

The detached inspection carriers are convenient but non-authoritative:
``gas`` intentionally omits GPU-only vapor pressure. Restart always uses
canonical payload bytes, which preserve vapor pressure and all acquired
sidecars exactly. The record is an in-memory, concrete-only checkpoint, not
a portable serialization or device-migration format.

**Attributes**

- `schema_version` (`int`) — Exact checkpoint schema version.
- `carrier_type` (`str`) — Exact resident carrier type name.
- `dimensions` (`ResidentDimensions`) — Immutable resident dimensions.
- `device` (`Device`) — Declared Warp device that restart must match exactly.
- `gas_names` (`tuple[str, ...]`) — Ordered CPU-owned gas names.
- `completed_steps` (`int`) — Number of completed guarded timesteps.
- `simulated_time` (`Real`) — Finite nonnegative cumulative timestep duration.
- `lifecycle` (`ResidentLifecycle`) — Lifecycle state captured before a normal checkpoint.
- `particles` (`'ParticleData'`) — Detached CPU particle inspection carrier.
- `gas` (`'GasData'`) — Detached CPU gas inspection carrier without vapor pressure.
- `environment` (`'EnvironmentData'`) — Detached CPU environment inspection carrier.
- `payloads` (`tuple[CheckpointPayload, ...]`) — Canonical immutable primaries followed by the exact ordered acquired registry sidecars, resident ledgers, diagnostics, and at most one complete closed-map communication family. Arbitrary caller-owned outputs are never checkpoint authority.
- `communication` (`'CommunicationCheckpointMetadata | None'`) — Optional schema-v5 metadata for one pinned, complete closed-map communication family.
- `rng_continuation` (`'RNGContinuationCheckpoint | None'`) — Required schema-v5 metadata with optional current ``uint32`` words for published coagulation and wall-loss streams. The words are authoritative continuation state on exact-device restart.
- `rng_resource_processes` (`tuple[str, ...]`) — Schema-v5 canonical process-family inventory for acquired RNG resources. This preserves the wall-loss acquisition boundary, whose sole ``rng_states`` sidecar is intentionally excluded from ordinary payloads.
- `distribution_type` (`str`) — Explicit "discrete" or "particle_resolved" semantics for canonical population bytes. Missing or untagged records reject before restart setup or upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L97-L154)

### particula.execution.checkpoint.ResidentCheckpoint.carrier_type

*attribute* · *instance attribute*

```python
carrier_type: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L140-L140)

### particula.execution.checkpoint.ResidentCheckpoint.communication

*attribute* · *class attribute* · *instance attribute*

```python
communication: 'CommunicationCheckpointMetadata | None' = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L151-L151)

### particula.execution.checkpoint.ResidentCheckpoint.completed_steps

*attribute* · *instance attribute*

```python
completed_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L144-L144)

### particula.execution.checkpoint.ResidentCheckpoint.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L142-L142)

### particula.execution.checkpoint.ResidentCheckpoint.dimensions

*attribute* · *instance attribute*

```python
dimensions: ResidentDimensions
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L141-L141)

### particula.execution.checkpoint.ResidentCheckpoint.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str = ''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L154-L154)

### particula.execution.checkpoint.ResidentCheckpoint.environment

*attribute* · *instance attribute*

```python
environment: 'EnvironmentData'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L149-L149)

### particula.execution.checkpoint.ResidentCheckpoint.gas

*attribute* · *instance attribute*

```python
gas: 'GasData'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L148-L148)

### particula.execution.checkpoint.ResidentCheckpoint.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L143-L143)

### particula.execution.checkpoint.ResidentCheckpoint.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: ResidentLifecycle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L146-L146)

### particula.execution.checkpoint.ResidentCheckpoint.particles

*attribute* · *instance attribute*

```python
particles: 'ParticleData'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L147-L147)

### particula.execution.checkpoint.ResidentCheckpoint.payloads

*attribute* · *instance attribute*

```python
payloads: tuple[CheckpointPayload, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L150-L150)

### particula.execution.checkpoint.ResidentCheckpoint.rng_continuation

*attribute* · *class attribute* · *instance attribute*

```python
rng_continuation: 'RNGContinuationCheckpoint | None' = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L152-L152)

### particula.execution.checkpoint.ResidentCheckpoint.rng_resource_processes

*attribute* · *class attribute* · *instance attribute*

```python
rng_resource_processes: tuple[str, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L153-L153)

### particula.execution.checkpoint.ResidentCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L139-L139)

### particula.execution.checkpoint.ResidentCheckpoint.simulated_time

*attribute* · *instance attribute*

```python
simulated_time: Real
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L145-L145)

### particula.execution.checkpoint.ResidentCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, carrier_type: str, dimensions: ResidentDimensions, device: Device, gas_names: tuple[str, ...], completed_steps: int, simulated_time: Real, lifecycle: ResidentLifecycle, particles: 'ParticleData', gas: 'GasData', environment: 'EnvironmentData', payloads: tuple[CheckpointPayload, ...], communication: 'CommunicationCheckpointMetadata | None' = None, rng_continuation: 'RNGContinuationCheckpoint | None' = None, rng_resource_processes: tuple[str, ...] = (), distribution_type: str = '') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint.ResidentCheckpointController

*class*

```python
class ResidentCheckpointController
```

Bind checkpoint and terminal finalization to one resident lifecycle.

The controller is identity-bound to one active session, its pinned registry,
and its closed-step guard. A normal checkpoint preserves ACTIVE usability
and returns a fresh equivalent record. The first finalization snapshots,
then atomically changes the session to FINALIZED; later finalizations return
the cached record without validation, synchronization, transfer, or
allocation. A failed pre-transition snapshot leaves the source ACTIVE.
Schema-v3 snapshots always capture published coagulation and wall-loss
stream metadata and optionally capture their current words at the
checkpoint's single synchronization boundary. An exact-device restart
restores fresh bindings from captured words without reseeding. Explicit
reset rederives restored published streams; normal first acquisition still
initializes streams absent from a checkpoint.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L326-L598)

### particula.execution.checkpoint.ResidentCheckpointController._finalized

*attribute* · *instance attribute*

```python
_finalized: ResidentCheckpoint | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L372-L372)

### particula.execution.checkpoint.ResidentCheckpointController._guard

*attribute* · *instance attribute*

```python
_guard = guard
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L371-L371)

### particula.execution.checkpoint.ResidentCheckpointController._registry

*attribute* · *instance attribute*

```python
_registry = registry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L370-L370)

### particula.execution.checkpoint.ResidentCheckpointController._session

*attribute* · *instance attribute*

```python
_session = session
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L369-L369)

### particula.execution.checkpoint.ResidentCheckpointController.__init__

*method*

```python
def __init__(session: ResidentSession, registry: 'GPUResourceRegistry', guard: ResidentStepGuard) -> None
```

Bind exact active session, registry, and guard identities.

**Parameters**

- `session` (`ResidentSession`) — Active resident session to checkpoint.
- `registry` (`'GPUResourceRegistry'`) — Registry pinned to ``session``.
- `guard` (`ResidentStepGuard`) — Closed-step guard bound to the same registry and session.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L343-L372)

### particula.execution.checkpoint.ResidentCheckpointController._communication_metadata

*method*

```python
def _communication_metadata() -> CommunicationCheckpointMetadata | None
```

Return metadata for the sole optional pinned communication view.

**Returns**

- (`CommunicationCheckpointMetadata | None`) — Detached metadata for the published gas or particle communication
- (`CommunicationCheckpointMetadata | None`) — family, or ``None`` when no communication resources are pinned.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L529-L549)

### particula.execution.checkpoint.ResidentCheckpointController._validate

*method*

```python
def _validate() -> None
```

Validate active identity-bound lifecycle state before readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L374-L389)

### particula.execution.checkpoint.ResidentCheckpointController.checkpoint

*method*

```python
def checkpoint() -> ResidentCheckpoint
```

Return a fresh immutable snapshot while leaving the session active.

Validation occurs before synchronization, transfer, payload access, or
mutation. On success, this operation synchronizes once, reads the three
primary carriers in order, and captures canonical primary and acquired
sidecar bytes. It neither transfers ownership nor rolls back device work
that was already launched by another operation.

**Returns**

- (`ResidentCheckpoint`) — A new independent checkpoint record.

**Raises**

- `ValueError` — If the lifecycle binding or published stream metadata
is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L391-L527)

### particula.execution.checkpoint.ResidentCheckpointController.finalize

*method*

```python
def finalize() -> ResidentCheckpoint
```

Create once, cache, and terminally finalize this resident session.

The first successful call first makes any attached issued graph handle
non-replayable and tears it down, then snapshots the active binding,
transitions the session from ACTIVE to FINALIZED, and caches the
snapshot. A teardown failure faults the source session and leaves no
cached final checkpoint. Repeated calls return the identical cached
checkpoint in O(1) without device activity.

**Returns**

- (`ResidentCheckpoint`) — The cached immutable terminal checkpoint.

**Raises**

- `ValueError` — If a first finalization has an invalid lifecycle or
stream binding. The session remains ACTIVE.
- `BaseException` — Propagates graph-handle teardown failure after
faulting the source session without caching a checkpoint.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L551-L598)

## particula.execution.checkpoint._RestartPlan

*class* · *dataclass*

```python
class _RestartPlan
```

Retain normalized, fully validated restart inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L220-L226)

### particula.execution.checkpoint._RestartPlan.continuation

*attribute* · *instance attribute*

```python
continuation: RNGContinuationCheckpoint | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L226-L226)

### particula.execution.checkpoint._RestartPlan.primary

*attribute* · *instance attribute*

```python
primary: dict[tuple[str, str], CheckpointPayload]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L224-L224)

### particula.execution.checkpoint._RestartPlan.resource_payloads

*attribute* · *instance attribute*

```python
resource_payloads: tuple[CheckpointPayload, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L225-L225)

### particula.execution.checkpoint._RestartPlan.__init__

*method*

```python
def __init__(primary: dict[tuple[str, str], CheckpointPayload], resource_payloads: tuple[CheckpointPayload, ...], continuation: RNGContinuationCheckpoint | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L0-L0)

## particula.execution.checkpoint._capture_rng_continuation

*function*

```python
def _capture_rng_continuation(streams: tuple[tuple[str, Any, Any], ...], manifest: Any) -> RNGContinuationCheckpoint
```

Copy stream metadata and up to two stream states after sync.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L248-L274)

## particula.execution.checkpoint._payload

*function*

```python
def _payload(family: str, role: str, value: Any, capacity: int | None = None) -> CheckpointPayload
```

Copy one synchronized Warp array into immutable canonical bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L229-L245)

## particula.execution.checkpoint._preflight_restart

*function*

```python
def _preflight_restart(checkpoint: object, device: object) -> _RestartPlan
```

Validate every descriptor before resident setup or Warp writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L820-L982)

## particula.execution.checkpoint._validate_payload

*function*

```python
def _validate_payload(item: object) -> CheckpointPayload
```

Validate one immutable payload descriptor without materializing it.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L313-L323)

## particula.execution.checkpoint._validate_payload_bytes

*function*

```python
def _validate_payload_bytes(item: CheckpointPayload) -> None
```

Validate the exact dtype and immutable byte payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L298-L310)

## particula.execution.checkpoint._validate_payload_metadata

*function*

```python
def _validate_payload_metadata(item: CheckpointPayload) -> None
```

Validate the exact descriptor metadata and size contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L277-L295)

## particula.execution.checkpoint._validate_population_metadata

*function*

```python
def _validate_population_metadata(checkpoint: ResidentCheckpoint) -> None
```

Require declared population meaning and gas width before any upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L798-L817)

## particula.execution.checkpoint._validate_resource_payloads

*function*

```python
def _validate_resource_payloads(payloads: tuple[CheckpointPayload, ...], dimensions: ResidentDimensions) -> tuple[str | None, bool]
```

Validate ordered acquired-sidecar descriptors before resident setup.

Communication payloads may contain exactly one complete gas or particle
family and an optional final-volume array. This descriptor-only preflight
reads no device payload, creates no session, and rejects mixed, partial, or
schema-incompatible families before any restart upload can begin.

**Parameters**

- `payloads` (`tuple[CheckpointPayload, ...]`) — Ordered immutable sidecar descriptors after primary payloads.
- `dimensions` (`ResidentDimensions`) — Restored resident dimensions used to resolve role shapes.

**Returns**

- (`tuple[str | None, bool]`) — Communication family name, if present, and final-volume presence.

**Raises**

- `ValueError` — If sidecar order, completeness, capacity, or schema is
incompatible with the checkpoint contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L1124-L1264)

## particula.execution.checkpoint._validate_rng_continuation

*function*

```python
def _validate_rng_continuation(checkpoint: ResidentCheckpoint, dimensions: ResidentDimensions) -> RNGContinuationCheckpoint | None
```

Fail closed on malformed v5 continuation metadata before setup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L985-L1083)

## particula.execution.checkpoint._validate_rng_resource_pairing

*function*

```python
def _validate_rng_resource_pairing(payloads: tuple[CheckpointPayload, ...], continuation: RNGContinuationCheckpoint | None, resource_processes: object) -> None
```

Require acquired RNG families and continuation words to pair exactly.

The explicit inventory preserves wall-loss acquisition, whose sole
``rng_states`` sidecar is continuation-only. Coagulation ordinary sidecars
independently corroborate that inventory for legacy records.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L1086-L1121)

## particula.execution.checkpoint.restart_resident_session

*function*

```python
def restart_resident_session(checkpoint: ResidentCheckpoint, device: Device) -> tuple[ResidentSession, 'GPUResourceRegistry', ResidentStepGuard]
```

Restart a fresh compatible resident session from canonical bytes.

This concrete-only operation requires a nonterminal checkpoint and an exact
matching target device. It never reuses source containers, primary arrays,
sidecars, registry, or guard identities. Full descriptor preflight happens
before setup and upload; after an asynchronous device write has launched,
rollback is not guaranteed.

**Parameters**

- `checkpoint` (`ResidentCheckpoint`) — Exact immutable active-session checkpoint to materialize.
- `device` (`Device`) — Explicit target device exactly equal to checkpoint metadata.

**Returns**

- (`tuple[ResidentSession, 'GPUResourceRegistry', ResidentStepGuard]`) — Fresh compatible session, pinned registry, and restored closed guard.

**Raises**

- `TypeError` — If the checkpoint carrier or required metadata is invalid.
- `ValueError` — If compatibility, lifecycle, counters, or payload schemas
are invalid, or the device does not exactly match.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint.py#L601-L795)
