# particula.execution.gpu_resources

*module*

Allocate concrete reusable Warp sidecars for one active resident session.

This direct-import-only, Warp-dependent boundary pins complete fixed-shape
native sidecar families, one capture resource selection, and one atomically
prepared capture resource set to one exact ``ACTIVE`` :class:`ResidentSession`.
It allocates and validates resources only: it neither executes a process,
transfers, synchronizes, nor resizes. Coagulation and wall-loss acquisition
initialize distinct resource setup-derived persistent RNG streams exactly once
before
publishing their resident resources. The checkpoint-private restoration seam
instead publishes prevalidated current stream words without reseeding. The
manifests and views here are concrete-only and are deliberately not exported
from :mod:`particula.execution`.

The registry retains array identities and performs metadata-only schema and
nonaliasing checks. It does not establish allocator provenance, execute a
kernel, or change session lifecycle. Its direct-module-only logical inventory
reports immutable manifest schema metadata and logical bytes, not
allocator-reserved bytes. Capture registration retains one exact closed
communication view, if selected, and an ordered diagnostic registration tuple.
It validates schemas and byte-range nonaliasing on the host without inspecting
payloads; reports reuse that retained metadata. Inventory reporting neither
inspects payloads nor acquires, allocates, binds, or mutates sidecars. Explicit
 lifecycle methods may inspect frozen stream metadata or reset selected
 published lanes without hidden transfer or synchronization. A stream-writer
 failure faults the exact resident binding and notifies any attached
 graph-capture owner to tear down issued handles.

capture-set preparation is setup-only. It stages all requested sidecars,
validates metadata and nonaliasing, and initializes only newly created resident
RNG streams before one final publication. Compatible repeats return the exact
published set; failed candidates do not publish staged bindings. Its matching
accessor is metadata-only and neither allocates, initializes, executes,
transfers, synchronizes, nor inspects payloads. These carriers and seams are
available only from this module, not from :mod:`particula.execution` or the
top-level package.
``validate_pinned_session`` is the narrow direct-module-only integration seam
for resident timestep guards. It requires the exact retained session, then
revalidates its active lifecycle, pinned container and primary-array identities,
and schema metadata without inspecting payloads, acquiring sidecars, allocating,
transferring, or synchronizing.

For concrete checkpointing, the private deterministic enumeration seam exposes
established live sidecars only after the same active pinned-session validation.
Checkpoint code owns any immutable host copy it creates; this registry retains
caller- or registry-owned device arrays. Enumeration neither copies nor
transfers payloads, and this module offers no restart, migration, or rollback
after launched device work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1-L1)

## particula.execution.gpu_resources.CaptureResourceRequirements

*class* · *dataclass*

```python
class CaptureResourceRequirements
```

Describe the complete setup-only resource request for one capture set.

This direct-module-only carrier retains all supplied objects by exact
identity for setup. It is host metadata only: registry-dependent schema,
alias, and publication checks are deliberately performed by
:meth:`GPUResourceRegistry.prepare_capture_resources`.
``None`` for an enabled allocatable family requests private allocation.

**Attributes**

- `session` (`ResidentSession`) — Exact resident session pinned by the target registry.
- `capacities` (`ResourceInventoryCapacities`) — Exact immutable logical inventory capacities.
- `inventory` (`SelectedResourceInventory`) — Exact resource inventory selected resource inventory.
- `prepared_views` (`PreparedResourceViews`) — Canonical process views required by preparation.
- `communication_resources` (`CommunicationResources | None`) — Exact selected communication view, if present.
- `enabled_process_families` (`tuple[str, ...]`) — Canonically ordered process families retained by the capture set. Enabled families may omit resources and views to request private allocation.
- `condensation` (`CondensationScratchBuffers | None`) — Supplied condensation record or an allocation request.
- `coagulation` (`CoagulationResources | None`) — Supplied coagulation view or an allocation request.
- `wall_loss` (`WallLossResources | None`) — Supplied wall-loss view or an allocation request.
- `nucleation` (`NucleationResources | None`) — Supplied nucleation view or an allocation request.
- `family_order` (`tuple[str, ...]`) — Required canonical order for all capture resource families.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L524-L697)

### particula.execution.gpu_resources.CaptureResourceRequirements.capacities

*attribute* · *instance attribute*

```python
capacities: ResourceInventoryCapacities
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L552-L552)

### particula.execution.gpu_resources.CaptureResourceRequirements.coagulation

*attribute* · *class attribute* · *instance attribute*

```python
coagulation: CoagulationResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L558-L558)

### particula.execution.gpu_resources.CaptureResourceRequirements.communication_resources

*attribute* · *instance attribute*

```python
communication_resources: CommunicationResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L555-L555)

### particula.execution.gpu_resources.CaptureResourceRequirements.condensation

*attribute* · *class attribute* · *instance attribute*

```python
condensation: CondensationScratchBuffers | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L557-L557)

### particula.execution.gpu_resources.CaptureResourceRequirements.enabled_process_families

*attribute* · *instance attribute*

```python
enabled_process_families: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L556-L556)

### particula.execution.gpu_resources.CaptureResourceRequirements.family_order

*attribute* · *class attribute* · *instance attribute*

```python
family_order: tuple[str, ...] = ('condensation', 'coagulation', 'wall_loss', 'nucleation', 'communication_gas', 'communication_particles', 'dilution')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L561-L569)

### particula.execution.gpu_resources.CaptureResourceRequirements.inventory

*attribute* · *instance attribute*

```python
inventory: SelectedResourceInventory
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L553-L553)

### particula.execution.gpu_resources.CaptureResourceRequirements.nucleation

*attribute* · *class attribute* · *instance attribute*

```python
nucleation: NucleationResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L560-L560)

### particula.execution.gpu_resources.CaptureResourceRequirements.prepared_views

*attribute* · *instance attribute*

```python
prepared_views: PreparedResourceViews
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L554-L554)

### particula.execution.gpu_resources.CaptureResourceRequirements.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L551-L551)

### particula.execution.gpu_resources.CaptureResourceRequirements.wall_loss

*attribute* · *class attribute* · *instance attribute*

```python
wall_loss: WallLossResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L559-L559)

### particula.execution.gpu_resources.CaptureResourceRequirements.__init__

*method*

```python
def __init__(session: ResidentSession, capacities: ResourceInventoryCapacities, inventory: SelectedResourceInventory, prepared_views: PreparedResourceViews, communication_resources: CommunicationResources | None, enabled_process_families: tuple[str, ...], condensation: CondensationScratchBuffers | None = None, coagulation: CoagulationResources | None = None, wall_loss: WallLossResources | None = None, nucleation: NucleationResources | None = None, family_order: tuple[str, ...] = ('condensation', 'coagulation', 'wall_loss', 'nucleation', 'communication_gas', 'communication_particles', 'dilution')) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

### particula.execution.gpu_resources.CaptureResourceRequirements.__post_init__

*method*

```python
def __post_init__() -> None
```

Reject incomplete or noncanonical host-only request metadata.

**Raises**

- `TypeError` — If a required carrier or optional resource has an invalid
exact type.
- `ValueError` — If communication identity, canonical family order, or
required supplied-record completeness is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L571-L697)

## particula.execution.gpu_resources.CaptureResourceSet

*class* · *dataclass*

```python
class CaptureResourceSet
```

Retain one atomically published capture resource set by exact identity.

This direct-module-only, setup-only carrier is published only after complete
staging and transactional initialization of newly created resident RNG
streams. It owns no payload copies, bytes, RNG words, or accounting. It
exposes retained metadata and views solely for later identity validation;
callers cannot use it to acquire, execute, transfer, synchronize, or mutate
resource bindings. It is not exported from :mod:`particula.execution` or
the top-level package.

**Attributes**

- `requirements` (`CaptureResourceRequirements`) — Exact setup request retained by identity.
- `capacities` (`ResourceInventoryCapacities`) — Exact logical inventory capacities from the request.
- `inventory` (`SelectedResourceInventory`) — Exact resource inventory-selected resource inventory.
- `report` (`LogicalResourceReport`) — Immutable logical resource report for the published set.
- `prepared_views` (`PreparedResourceViews`) — Canonical process views retained for preparation.
- `communication_resources` (`CommunicationResources | None`) — Selected communication view, if present.
- `condensation` (`CondensationResources | None`) — Published condensation resources, if requested.
- `coagulation` (`CoagulationResources | None`) — Published coagulation resources, if requested.
- `wall_loss` (`WallLossResources | None`) — Published wall-loss resources, if requested.
- `nucleation` (`NucleationResources | None`) — Published nucleation resources, if requested.
- `dilution` (`DilutionResources | None`) — Capture-owned dilution resources, if requested.
- `coagulation_stream_registry` (`_PublishedStreamRegistry | None`) — Persistent coagulation stream metadata.
- `wall_loss_stream_registry` (`_PublishedStreamRegistry | None`) — Persistent wall-loss stream metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L700-L740)

### particula.execution.gpu_resources.CaptureResourceSet.capacities

*attribute* · *instance attribute*

```python
capacities: ResourceInventoryCapacities
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L729-L729)

### particula.execution.gpu_resources.CaptureResourceSet.coagulation

*attribute* · *instance attribute*

```python
coagulation: CoagulationResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L735-L735)

### particula.execution.gpu_resources.CaptureResourceSet.coagulation_stream_registry

*attribute* · *instance attribute*

```python
coagulation_stream_registry: _PublishedStreamRegistry | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L739-L739)

### particula.execution.gpu_resources.CaptureResourceSet.communication_resources

*attribute* · *instance attribute*

```python
communication_resources: CommunicationResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L733-L733)

### particula.execution.gpu_resources.CaptureResourceSet.condensation

*attribute* · *instance attribute*

```python
condensation: CondensationResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L734-L734)

### particula.execution.gpu_resources.CaptureResourceSet.dilution

*attribute* · *instance attribute*

```python
dilution: DilutionResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L738-L738)

### particula.execution.gpu_resources.CaptureResourceSet.inventory

*attribute* · *instance attribute*

```python
inventory: SelectedResourceInventory
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L730-L730)

### particula.execution.gpu_resources.CaptureResourceSet.nucleation

*attribute* · *instance attribute*

```python
nucleation: NucleationResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L737-L737)

### particula.execution.gpu_resources.CaptureResourceSet.prepared_views

*attribute* · *instance attribute*

```python
prepared_views: PreparedResourceViews
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L732-L732)

### particula.execution.gpu_resources.CaptureResourceSet.report

*attribute* · *instance attribute*

```python
report: LogicalResourceReport
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L731-L731)

### particula.execution.gpu_resources.CaptureResourceSet.requirements

*attribute* · *instance attribute*

```python
requirements: CaptureResourceRequirements
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L728-L728)

### particula.execution.gpu_resources.CaptureResourceSet.wall_loss

*attribute* · *instance attribute*

```python
wall_loss: WallLossResources | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L736-L736)

### particula.execution.gpu_resources.CaptureResourceSet.wall_loss_stream_registry

*attribute* · *instance attribute*

```python
wall_loss_stream_registry: _PublishedStreamRegistry | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L740-L740)

### particula.execution.gpu_resources.CaptureResourceSet.__init__

*method*

```python
def __init__(requirements: CaptureResourceRequirements, capacities: ResourceInventoryCapacities, inventory: SelectedResourceInventory, report: LogicalResourceReport, prepared_views: PreparedResourceViews, communication_resources: CommunicationResources | None, condensation: CondensationResources | None, coagulation: CoagulationResources | None, wall_loss: WallLossResources | None, nucleation: NucleationResources | None, dilution: DilutionResources | None, coagulation_stream_registry: _PublishedStreamRegistry | None, wall_loss_stream_registry: _PublishedStreamRegistry | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.CoagulationResources

*class* · *dataclass*

```python
class CoagulationResources
```

Expose native coagulation outputs and one resource setup-initialized RNG
    sidecar.

The registry publishes this identity-bound view only after validating all
supplied sidecars and initializing the coagulation-only ``rng_states`` once
from immutable resident stream metadata. Repeated compatible acquisition
returns this view and its arrays by identity without allocation, reseeding,
readback, transfer, or synchronization. The sidecar has no wall-loss,
public checkpoint, reset, or inspection API. Schema-v3 checkpoint restart
can privately restore fresh bindings from captured current words.

**Attributes**

- `collision_capacity` (`int`) — Fixed maximum number of collision pairs per box.
- `collision_pairs` (`Any`) — Caller- or registry-owned ``int32`` pair storage.
- `n_collisions` (`Any`) — Per-box ``int32`` collision-count storage.
- `rng_states` (`Any`) — Persistent per-box ``uint32`` RNG state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L368-L391)

### particula.execution.gpu_resources.CoagulationResources.collision_capacity

*attribute* · *instance attribute*

```python
collision_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L388-L388)

### particula.execution.gpu_resources.CoagulationResources.collision_pairs

*attribute* · *instance attribute*

```python
collision_pairs: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L389-L389)

### particula.execution.gpu_resources.CoagulationResources.n_collisions

*attribute* · *instance attribute*

```python
n_collisions: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L390-L390)

### particula.execution.gpu_resources.CoagulationResources.rng_states

*attribute* · *instance attribute*

```python
rng_states: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L391-L391)

### particula.execution.gpu_resources.CoagulationResources.__init__

*method*

```python
def __init__(collision_capacity: int, collision_pairs: Any, n_collisions: Any, rng_states: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.CommunicationResources

*class* · *dataclass*

```python
class CommunicationResources
```

Expose one exact resident communication configuration and work record.

The registry publishes this identity-bound view only after resource setup
configuration
validation and complete sidecar schema/nonaliasing checks. The
configuration, native work record, and optional final volumes remain
caller- or registry-owned device state; this view performs no transfer,
synchronization, payload inspection, or mutation. It represents either the
GAS or PARTICLES family, never a combined or open-map configuration.

**Attributes**

- `configuration` (`CommunicationConfiguration`) — Exact closed-map configuration retained by identity.
- `buffers` (`GasCommunicationBuffers | ParticleCommunicationBuffers`) — Exact mode-matched native work record retained by identity.
- `final_volumes` (`Any | None`) — Optional pinned ``float64`` per-box target volumes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L451-L472)

### particula.execution.gpu_resources.CommunicationResources.buffers

*attribute* · *instance attribute*

```python
buffers: GasCommunicationBuffers | ParticleCommunicationBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L470-L470)

### particula.execution.gpu_resources.CommunicationResources.configuration

*attribute* · *instance attribute*

```python
configuration: CommunicationConfiguration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L469-L469)

### particula.execution.gpu_resources.CommunicationResources.execution_state

*attribute* · *instance attribute*

```python
execution_state: 'ResidentCommunicationState'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L472-L472)

### particula.execution.gpu_resources.CommunicationResources.final_volumes

*attribute* · *instance attribute*

```python
final_volumes: Any | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L471-L471)

### particula.execution.gpu_resources.CommunicationResources.__init__

*method*

```python
def __init__(configuration: CommunicationConfiguration, buffers: GasCommunicationBuffers | ParticleCommunicationBuffers, final_volumes: Any | None, execution_state: 'ResidentCommunicationState') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.CondensationResources

*class* · *dataclass*

```python
class CondensationResources
```

Expose a complete native condensation scratch record.

**Attributes**

- `scratch_buffers` (`CondensationScratchBuffers`) — Pinned native scratch arrays for condensation steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L357-L365)

### particula.execution.gpu_resources.CondensationResources.scratch_buffers

*attribute* · *instance attribute*

```python
scratch_buffers: CondensationScratchBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L365-L365)

### particula.execution.gpu_resources.CondensationResources.__init__

*method*

```python
def __init__(scratch_buffers: CondensationScratchBuffers) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.DilutionResources

*class* · *dataclass*

```python
class DilutionResources
```

Expose complete caller-owned dilution preparation sidecars.

This descriptor-only record is deliberately neither acquired nor published
by the registry. Prepared callers may retain both validated sidecars by
identity. The registry read-only validates their exact ``float64``
``(B,)`` schemas, device, contiguity, and nonaliasing before preparation;
it does not allocate, bind, or mutate them. Direct dilution calls retain
their existing fallback-allocation behavior when no record is supplied.

**Attributes**

- `normalized_coefficient` (`Any`) — Caller-owned normalized per-box dilution coefficient sidecar with shape ``(B,)``.
- `factors` (`Any`) — Caller-owned per-box dilution factor sidecar with shape ``(B,)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L394-L413)

### particula.execution.gpu_resources.DilutionResources.factors

*attribute* · *instance attribute*

```python
factors: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L413-L413)

### particula.execution.gpu_resources.DilutionResources.normalized_coefficient

*attribute* · *instance attribute*

```python
normalized_coefficient: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L412-L412)

### particula.execution.gpu_resources.DilutionResources.__init__

*method*

```python
def __init__(normalized_coefficient: Any, factors: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.GPUResourceRegistry

*class*

```python
class GPUResourceRegistry
```

Pin reusable complete native sidecars to one exact active session.

Publication pins caller- or registry-allocated Warp objects by role. This
validates identity and nonaliasing, not unverifiable allocator provenance.
No payload is read, copied, synchronized, or mutated by acquisition, except
that first coagulation or wall-loss acquisition initializes its distinct
Resource setup-derived RNG sidecar before publication. Its
concrete-only :meth:`validate_pinned_session` seam lets lifecycle guards
verify the exact active binding without resource acquisition or execution.
Its private checkpoint enumeration reports ordinary acquired sidecars in
manifest order and published RNG bindings in canonical process order,
without changing ownership or creating host copies. Checkpoint restart may
privately publish prevalidated fresh RNG bindings without reseeding.

capture registration is a separate concrete-only host-metadata seam. It
pins one optional published closed communication view and one ordered
diagnostic registration tuple by identity after complete schema and
nonaliasing validation. It neither changes normal resource enumeration nor
participates in checkpoint or restart state.

:meth:`prepare_capture_resources` is a separate direct-module-only,
setup-only transaction. It reuses exact compatible publications, stages a
complete resource set privately, initializes only new coagulation or
wall-loss streams, and assigns the set only after validation succeeds.
:meth:`validate_capture_resource_set` only verifies retained identities and
returns the published set and its cached immutable report by identity; it
performs no allocation, initialization, payload inspection, transfer,
synchronization, or execution. Neither seam
is re-exported from :mod:`particula.execution` or the top-level package.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1137-L4208)

### particula.execution.gpu_resources.GPUResourceRegistry._bindings

*attribute* · *instance attribute*

```python
_bindings: dict[str, dict[str, Any]] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1187-L1187)

### particula.execution.gpu_resources.GPUResourceRegistry._capacities

*attribute* · *instance attribute*

```python
_capacities: dict[str, int] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1190-L1190)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_inventory

*attribute* · *instance attribute*

```python
_capture_inventory: SelectedResourceInventory | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1196-L1196)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_resource_fingerprint

*attribute* · *instance attribute*

```python
_capture_resource_fingerprint: tuple[Any, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1198-L1198)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_resource_set

*attribute* · *instance attribute*

```python
_capture_resource_set: CaptureResourceSet | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1197-L1197)

### particula.execution.gpu_resources.GPUResourceRegistry._coagulation_stream_registry

*attribute* · *instance attribute*

```python
_coagulation_stream_registry: _PublishedStreamRegistry | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1192-L1194)

### particula.execution.gpu_resources.GPUResourceRegistry._nucleation_records

*attribute* · *instance attribute*

```python
_nucleation_records: tuple[Any, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1189-L1189)

### particula.execution.gpu_resources.GPUResourceRegistry._open_step_token

*attribute* · *instance attribute*

```python
_open_step_token: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1191-L1191)

### particula.execution.gpu_resources.GPUResourceRegistry._session

*attribute* · *instance attribute*

```python
_session = session
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1184-L1184)

### particula.execution.gpu_resources.GPUResourceRegistry._signature

*attribute* · *instance attribute*

```python
_signature = self._session_signature()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1186-L1186)

### particula.execution.gpu_resources.GPUResourceRegistry._views

*attribute* · *instance attribute*

```python
_views: dict[str, Any] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1188-L1188)

### particula.execution.gpu_resources.GPUResourceRegistry._wall_loss_stream_registry

*attribute* · *instance attribute*

```python
_wall_loss_stream_registry: _PublishedStreamRegistry | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1195-L1195)

### particula.execution.gpu_resources.GPUResourceRegistry.manifests

*property*

```python
manifests: tuple[ResourceManifest, ...]
```

Return the canonical immutable direct-module manifest set.

**Returns**

- (`ResourceManifest`) — All established sidecar manifests, including the mutually exclusive
- (`...`) — gas and particle communication families.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1201-L1216)

### particula.execution.gpu_resources.GPUResourceRegistry.__init__

*method*

```python
def __init__(session: ResidentSession) -> None
```

Create a sidecar registry pinned to one active resident session.

**Parameters**

- `session` (`ResidentSession`) — Exact active resident session that supplies the fixed dimensions, device, and protected primary-array identities.

**Raises**

- `TypeError` — If ``session`` or its lifecycle carriers are not exact
resident-session types.
- `ValueError` — If the session is not active or fails its metadata
validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1169-L1198)

### particula.execution.gpu_resources.GPUResourceRegistry._acquire

*method*

```python
def _acquire(manifest: ResourceManifest, supplied: dict[str, Any], capacity: int | None = None, *, publish: bool = True) -> dict[str, Any]
```

Validate, allocate, and atomically publish one resource family.

**Parameters**

- `manifest` (`ResourceManifest`) — Complete schema for the resource family.
- `supplied` (`dict[str, Any]`) — Role-to-array bindings; ``None`` requests allocation.
- `capacity` (`int | None`) (default: `None`) — Collision or communication-edge capacity for a manifest that requires it.
- `publish` (`bool`) (default: `True`) — Whether to register a newly validated family immediately.

**Returns**

- (`dict[str, Any]`) — Pinned role-to-array bindings for the established family.

**Raises**

- `TypeError` — If supplied arrays fail the required Warp type checks.
- `ValueError` — If the session drifted, bindings are incompatible,
alias protected storage, or coagulation capacity changes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3520-L3573)

### particula.execution.gpu_resources.GPUResourceRegistry._allocate

*method*

```python
def _allocate(entry: ManifestEntry, capacity: int | None) -> Any
```

Allocate one manifest-conforming sidecar on the pinned device.

**Parameters**

- `entry` (`ManifestEntry`) — Manifest role to allocate.
- `capacity` (`int | None`) — Collision or communication-edge capacity for entries whose schema requires it.

**Returns**

- (`Any`) — Zero-filled Warp array matching the entry's fixed schema.

**Raises**

- `ValueError` — If the computed allocation element or byte count
exceeds the supported range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3360-L3377)

### particula.execution.gpu_resources.GPUResourceRegistry._array_range

*method* · *staticmethod*

```python
def _array_range(array: Any) -> tuple[int, int] | None
```

Return one validated registry array's nonempty byte range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3481-L3518)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_candidate_roles

*method*

```python
def _capture_candidate_roles(communication_resources: CommunicationResources | None, registrations: tuple[Any, ...]) -> tuple[tuple[SelectedResourceFamilyReport, ...], tuple[SelectedResourceRole, ...], tuple[tuple[int, int] | None, ...]]
```

Build deterministic selected-role metadata before publication.

The caller has already validated the selection's exact session,
communication view, and diagnostic registrations. This helper resolves
each selected schema and byte range once; it does not perform device I/O
or mutate the capture inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2065-L2212)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_fingerprint

*method*

```python
def _capture_fingerprint(requirements: CaptureResourceRequirements) -> tuple[Any, ...]
```

Return immutable identity and capacity metadata for one request.

The private fingerprint compares only retained host identities,
presence, and logical capacities. It neither reads payloads nor
performs resource work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1312-L1346)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_inventory_values

*method*

```python
def _capture_inventory_values() -> list[Any]
```

Return all arrays retained by the selected capture inventory.

Inventory roles are protected storage just like ordinary published
sidecars.  This includes diagnostic accounting arrays and optional
communication final volumes, which do not necessarily appear in
``_bindings``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1400-L1422)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_publication_report

*method*

```python
def _capture_publication_report(requirements: CaptureResourceRequirements) -> LogicalResourceReport
```

Report only roles retained by the exact capture publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1659-L1698)

### particula.execution.gpu_resources.GPUResourceRegistry._capture_set_matches

*method*

```python
def _capture_set_matches(requirements: CaptureResourceRequirements) -> bool
```

Return whether a request retains every published capture identity.

This is an identity-and-metadata comparison only. It does not allocate,
initialize streams, inspect payloads, or mutate registry state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1384-L1398)

### particula.execution.gpu_resources.GPUResourceRegistry._checked_product

*method* · *staticmethod*

```python
def _checked_product(left: int, right: int) -> int
```

Multiply resource extents with central checked arithmetic.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3355-L3358)

### particula.execution.gpu_resources.GPUResourceRegistry._contiguous_range

*method*

```python
def _contiguous_range(value: Any, shape: tuple[int, ...], dtype: Any, role: str) -> tuple[int, int] | None
```

Validate contiguous metadata and return a nonempty byte range.

**Parameters**

- `value` (`Any`) — Warp array whose storage metadata is checked.
- `shape` (`tuple[int, ...]`) — Expected logical shape of the array.
- `dtype` (`Any`) — Expected Warp dtype.
- `role` (`str`) — Role name used in validation errors.

**Returns**

- (`tuple[int, int] | None`) — Half-open byte range for nonempty storage, or ``None`` for an empty
- (`tuple[int, int] | None`) — logical array.

**Raises**

- `ValueError` — If strides, pointer alignment, or storage capacity is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3300-L3353)

### particula.execution.gpu_resources.GPUResourceRegistry._diagnostic_binding_entries

*method* · *staticmethod*

```python
def _diagnostic_binding_entries(registrations: tuple[Any, ...]) -> tuple[list[Any], list[ManifestEntry], list[Any], list[ManifestEntry]]
```

Return caller-owned diagnostic bindings and their exact schemas.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2680-L2718)

### particula.execution.gpu_resources.GPUResourceRegistry._enumerate_published_rng_streams

*method*

```python
def _enumerate_published_rng_streams() -> tuple[tuple[str, StreamManifest, Any], ...]
```

Return live published RNG bindings in canonical order.

This checkpoint-private preflight performs only metadata and identity
validation.  It deliberately does not read state words or synchronize.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3058-L3118)

### particula.execution.gpu_resources.GPUResourceRegistry._enumerate_resources

*method*

```python
def _enumerate_resources() -> tuple[tuple[str, str, Any, int | None], ...]
```

Return established sidecars in deterministic manifest order.

This checkpoint-only seam validates the exact active pinned session but
does not synchronize, copy, allocate, or inspect array payloads. Each
item is ``(family, role, live_array, capacity)``. The returned arrays
remain registry-owned live device arrays; a checkpoint controller alone
decides whether and how to capture immutable host bytes.

**Returns**

- (`tuple[str, str, Any, int | None]`) — Established sidecars as deterministic family/role descriptors with
- (`...`) — live arrays and optional coagulation capacity.

**Raises**

- `ValueError` — If the exact pinned session is inactive or its protected
metadata, containers, or primary-array identities drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3025-L3056)

### particula.execution.gpu_resources.GPUResourceRegistry._has_resident_coagulation_stream

*method*

```python
def _has_resident_coagulation_stream() -> bool
```

Return whether an initialized resident coagulation stream exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3120-L3122)

### particula.execution.gpu_resources.GPUResourceRegistry._has_resident_rng_stream

*method*

```python
def _has_resident_rng_stream() -> bool
```

Return whether any initialized resident RNG stream exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3124-L3129)

### particula.execution.gpu_resources.GPUResourceRegistry._new_stream_registry

*method*

```python
def _new_stream_registry(process_id: str, state: Any) -> _PublishedStreamRegistry
```

Create and initialize exactly one unpublished resident RNG stream.

The stream remains private to capture-set setup until publication, so
only a newly staged coagulation or wall-loss sidecar is initialized.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1635-L1657)

### particula.execution.gpu_resources.GPUResourceRegistry._nucleation_supplied_bindings

*method*

```python
def _nucleation_supplied_bindings(scratch: NucleationScratchBuffers | None, finalized_demand: NucleationFinalizedDemandBuffers | None, diagnostics: NucleationDiagnosticBuffers | None, exhaustion: NucleationExhaustionBuffers | None) -> dict[str, Any]
```

Validate and flatten optional complete nucleation records.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L4089-L4135)

### particula.execution.gpu_resources.GPUResourceRegistry._nucleation_view

*method*

```python
def _nucleation_view(bindings: dict[str, Any]) -> NucleationResources
```

Construct complete native nucleation records from pinned bindings.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L4137-L4163)

### particula.execution.gpu_resources.GPUResourceRegistry._preflight_capture_family

*method*

```python
def _preflight_capture_family(manifest: ResourceManifest, supplied: dict[str, Any], capacity: int | None) -> None
```

Validate one requested preallocation family without allocating or
publishing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1577-L1604)

### particula.execution.gpu_resources.GPUResourceRegistry._protected_ranges

*method*

```python
def _protected_ranges() -> list[tuple[int, int] | None]
```

Return metadata-only byte ranges for protected primary arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3391-L3395)

### particula.execution.gpu_resources.GPUResourceRegistry._published_stream_registry

*method*

```python
def _published_stream_registry(process_id: str) -> _PublishedStreamRegistry | None
```

Return the published process registry without exposing sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2459-L2465)

### particula.execution.gpu_resources.GPUResourceRegistry._ranges_overlap

*method* · *staticmethod*

```python
def _ranges_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool
```

Return whether two nonempty half-open byte ranges overlap.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3379-L3389)

### particula.execution.gpu_resources.GPUResourceRegistry._record_bindings

*method* · *staticmethod*

```python
def _record_bindings(record: Any) -> dict[str, Any]
```

Return dataclass field bindings for one exact native record.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L4082-L4087)

### particula.execution.gpu_resources.GPUResourceRegistry._reject_primary_aliases

*method*

```python
def _reject_primary_aliases(values: list[Any]) -> None
```

Reject sidecars that share identity or bytes with primaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3406-L3417)

### particula.execution.gpu_resources.GPUResourceRegistry._reject_shared_identities

*method* · *staticmethod*

```python
def _reject_shared_identities(values: list[Any], others: list[Any]) -> None
```

Reject identity reuse within or across sidecar ownership sets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3397-L3404)

### particula.execution.gpu_resources.GPUResourceRegistry._restore_published_rng_views

*method*

```python
def _restore_published_rng_views(process_ids: tuple[str, ...]) -> None
```

Publish prevalidated restored stream bindings without reseeding.

Checkpoint restart calls this only after it has bulk-uploaded and bound
the ordinary sidecars and current-word arrays.  It deliberately creates
no initial words and invokes no acquisition API.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3131-L3162)

### particula.execution.gpu_resources.GPUResourceRegistry._selected_role

*method*

```python
def _selected_role(canonical_name: str, family: str, entry: ManifestEntry, value: Any, capacity: int | None, *, read_only: bool = False) -> tuple[SelectedResourceRole, tuple[int, int] | None]
```

Resolve selected-role metadata once without inspecting payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2038-L2063)

### particula.execution.gpu_resources.GPUResourceRegistry._session_signature

*method*

```python
def _session_signature() -> tuple[Any, ...]
```

Build the pinned lifecycle, dimension, device, and identity
signature.

**Returns**

- (`tuple[Any, ...]`) — Immutable metadata used to detect session drift before acquisition.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2392-L2416)

### particula.execution.gpu_resources.GPUResourceRegistry._shape

*method*

```python
def _shape(entry: ManifestEntry, capacity: int | None = None) -> tuple[int, ...]
```

Resolve a manifest entry to its fixed resident-session shape.

**Parameters**

- `entry` (`ManifestEntry`) — Manifest role whose shape formula is resolved.
- `capacity` (`int | None`) (default: `None`) — Required collision or communication-edge capacity for ``"bc2"``, ``"e"``, and ``"en"`` entries.

**Returns**

- (`tuple[int, ...]`) — Exact Warp-array shape for the entry.

**Raises**

- `ValueError` — If a collision-pair or communication-edge shape lacks
its required capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3224-L3265)

### particula.execution.gpu_resources.GPUResourceRegistry._stage_family

*method*

```python
def _stage_family(manifest: ResourceManifest, supplied: dict[str, Any], capacity: int | None, staged: list[dict[str, Any]]) -> dict[str, Any]
```

Build one unpublished family after complete supplied preflight.

Existing compatible bindings are reused by identity. Newly allocated
bindings remain private to capture-set setup until the transaction's
final publication assignments.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1535-L1575)

### particula.execution.gpu_resources.GPUResourceRegistry._stream_metadata

*method*

```python
def _stream_metadata() -> tuple[int, tuple[str, ...], tuple[int, ...]]
```

Return normalized host stream identity for sessions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2447-L2457)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_array

*method*

```python
def _validate_array(entry: ManifestEntry, value: Any, capacity: int | None) -> tuple[int, int] | None
```

Validate one sidecar's Warp metadata against its manifest role.

**Parameters**

- `entry` (`ManifestEntry`) — Expected dtype and shape specification.
- `value` (`Any`) — Caller-supplied Warp array to inspect without reading.
- `capacity` (`int | None`) — Collision or communication-edge capacity for entries whose schema requires it.

**Returns**

- (`tuple[int, int] | None`) — Nonempty half-open byte range, or ``None`` for an empty array.

**Raises**

- `TypeError` — If ``value`` is not a Warp array.
- `ValueError` — If its schema, device, pointer, or strides are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3267-L3298)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_capture_collision_capacity

*method*

```python
def _validate_capture_collision_capacity(capacity: Any) -> int
```

Validate preallocation collision capacity using ordinary acquisition
bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1424-L1445)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_capture_nonalias

*method*

```python
def _validate_capture_nonalias(roles: tuple[SelectedResourceRole, ...], ranges: tuple[tuple[int, int] | None, ...], selected_published: set[int]) -> None
```

Reject capture overlap using one host-only sorted interval sweep.

Read-only diagnostic accounting inputs may share one another.  Every
other selected, primary, or established-sidecar overlap is forbidden.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2214-L2288)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_capture_requirements

*method*

```python
def _validate_capture_requirements(requirements: CaptureResourceRequirements) -> None
```

Validate request identities before capture allocation or staging.

This setup-only helper checks only the existing pinned session and
resource inventory
selection relationships. Schema, alias, and candidate-resource checks
remain in the preparation transaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1348-L1382)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_capture_supplied_families

*method*

```python
def _validate_capture_supplied_families(supplied_families: tuple[dict[str, Any], ...]) -> None
```

Reject cross-family supplied aliases before any preallocation
allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1606-L1633)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_diagnostic_binding_nonalias

*method*

```python
def _validate_diagnostic_binding_nonalias(outputs: list[Any], output_ranges: list[tuple[int, int] | None], inputs: list[Any], input_ranges: list[tuple[int, int] | None], protected: list[Any], protected_ranges: list[tuple[int, int] | None]) -> None
```

Reject diagnostic bindings that overlap protected/output storage.

This metadata-only sorted sweep permits only accounting-input aliases;
it avoids pairwise work for large diagnostic selections.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2720-L2778)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_nonalias

*method*

```python
def _validate_nonalias(bindings: dict[str, Any], entries: tuple[ManifestEntry, ...], capacity: int | None) -> None
```

Reject overlaps among candidate, registered, and primary arrays.

**Parameters**

- `bindings` (`dict[str, Any]`) — Candidate role-to-array bindings to validate.
- `entries` (`tuple[ManifestEntry, ...]`) — Manifest entries defining the candidate schemas.
- `capacity` (`int | None`) — Dynamic capacity required by the manifest, if any.

**Raises**

- `TypeError` — If a candidate binding is not a Warp array.
- `ValueError` — If a binding has invalid metadata or overlaps protected
storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3419-L3455)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_session_carriers

*method*

```python
def _validate_session_carriers() -> None
```

Validate exact lifecycle carriers before inspecting schemas.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3213-L3222)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_session_signature

*method*

```python
def _validate_session_signature() -> None
```

Reject a fabricated, inactive, or replaced resident session state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2418-L2424)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_session_state

*method*

```python
def _validate_session_state() -> None
```

Recheck the metadata-only invariants needed by this boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3206-L3211)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_staged_nonalias

*method*

```python
def _validate_staged_nonalias(bindings: dict[str, Any], entries: tuple[ManifestEntry, ...], capacity: int | None, staged: tuple[dict[str, Any], ...]) -> None
```

Validate one private candidate family against protected sidecars.

Candidate ranges remain transaction-local until capture-set publication.
This metadata-only check does not inspect sidecar payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1499-L1533)

### particula.execution.gpu_resources.GPUResourceRegistry._validate_supplied_nonalias

*method*

```python
def _validate_supplied_nonalias(supplied: dict[str, Any], entries: tuple[ManifestEntry, ...]) -> None
```

Reject supplied aliases before allocating omitted sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3457-L3479)

### particula.execution.gpu_resources.GPUResourceRegistry.acquire_coagulation

*method*

```python
def acquire_coagulation(collision_capacity: int, *, collision_pairs: Any | None = None, n_collisions: Any | None = None, rng_states: Any | None = None) -> CoagulationResources
```

Acquire fixed-capacity coagulation outputs and one RNG sidecar.

The first successful acquisition validates supplied sidecars and
nonaliasing before allocating omitted arrays. It then initializes the
single ``(n_boxes,)`` ``wp.uint32`` coagulation sidecar from immutable
session stream metadata and publishes the view. Compatible later calls
return that exact view without allocation or reseeding. This is not a
wall-loss stream, reset or inspection API, hidden transfer,
synchronization, or public checkpoint-persistence boundary. A
schema-v3 restart privately restores captured current words without
invoking this acquisition method or reseeding.

**Parameters**

- `collision_capacity` (`int`) — Positive, non-boolean integral collision bound.
- `collision_pairs` (`Any | None`) (default: `None`) — Optional ``int32`` collision-pair sidecar.
- `n_collisions` (`Any | None`) (default: `None`) — Optional ``int32`` per-box count sidecar.
- `rng_states` (`Any | None`) (default: `None`) — Optional ``uint32`` persistent per-box RNG sidecar.

**Returns**

- (`CoagulationResources`) — Stable view with the fixed capacity and pinned native sidecars.

**Raises**

- `TypeError` — If capacity is not a non-boolean integral or a supplied
sidecar is not a Warp array.
- `ValueError` — If capacity, schema, aliasing, session signature, or a
replacement request is incompatible with the established state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3616-L3719)

### particula.execution.gpu_resources.GPUResourceRegistry.acquire_communication

*method*

```python
def acquire_communication(configuration: CommunicationConfiguration, *, buffers: GasCommunicationBuffers | ParticleCommunicationBuffers | None = None) -> CommunicationResources
```

Pin one closed resident communication map and native work record.

This is the sole resource setup validation and optional-allocation
boundary for this
family. It accepts only GAS or PARTICLES closed maps, validates the
configuration once, then pins maps, work arrays, and optional prescribed
volumes by identity after schema and byte-range nonaliasing checks.
Reacquisition may return the established binding but never replaces it.
It does not execute a communication primitive, inspect payload values,
transfer, synchronize, initialize RNG state, or recover a writer error.

**Parameters**

- `configuration` (`CommunicationConfiguration`) — Exact resource setup-validated closed resident map and optional prescribed-volume update.
- `buffers` (`GasCommunicationBuffers | ParticleCommunicationBuffers | None`) (default: `None`) — Optional complete native mode-matched work record. Omitted work arrays are allocated on the pinned device.

**Returns**

- (`CommunicationResources`) — The stable identity-bound published communication resource view.

**Raises**

- `TypeError` — If the configuration or supplied buffer record has an
inexact or mode-incompatible type.
- `ValueError` — If resource setup validation, session identity, resource
schema,
capacity, or nonaliasing checks fail, or a binding is replaced.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3796-L3965)

### particula.execution.gpu_resources.GPUResourceRegistry.acquire_condensation

*method*

```python
def acquire_condensation(buffers: CondensationScratchBuffers | None = None) -> CondensationResources
```

Acquire one complete pinned condensation scratch record.

**Parameters**

- `buffers` (`CondensationScratchBuffers | None`) (default: `None`) — Optional complete exact native scratch record. Missing records are allocated as a complete fixed-shape set.

**Returns**

- (`CondensationResources`) — Stable view containing the native scratch record by identity.

**Raises**

- `TypeError` — If ``buffers`` is not an exact native record.
- `ValueError` — If it is incomplete, incompatible, aliases protected
storage, or the pinned session has drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3575-L3614)

### particula.execution.gpu_resources.GPUResourceRegistry.acquire_nucleation

*method*

```python
def acquire_nucleation(scratch: NucleationScratchBuffers | None = None, finalized_demand: NucleationFinalizedDemandBuffers | None = None, diagnostics: NucleationDiagnosticBuffers | None = None, exhaustion: NucleationExhaustionBuffers | None = None) -> NucleationResources
```

Acquire complete pinned native nucleation records and scratch.

**Parameters**

- `scratch` (`NucleationScratchBuffers | None`) (default: `None`) — Optional complete exact nucleation scratch record.
- `finalized_demand` (`NucleationFinalizedDemandBuffers | None`) (default: `None`) — Optional complete exact finalized-demand record.
- `diagnostics` (`NucleationDiagnosticBuffers | None`) (default: `None`) — Optional complete exact diagnostic record.
- `exhaustion` (`NucleationExhaustionBuffers | None`) (default: `None`) — Optional complete exact exhaustion record, including its complete nested resampling buffers.

**Returns**

- (`NucleationResources`) — Stable view holding complete native records built from pinned
- (`NucleationResources`) — arrays.

**Raises**

- `TypeError` — If supplied records are not exact native record types.
- `ValueError` — If records are incomplete, their sidecars are
incompatible or aliasing, or the session signature drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L4165-L4208)

### particula.execution.gpu_resources.GPUResourceRegistry.acquire_wall_loss

*method*

```python
def acquire_wall_loss(rng_states: Any | None = None) -> WallLossResources
```

Acquire one initialized persistent wall-loss RNG sidecar.

The first successful acquisition validates or allocates the single
``(n_boxes,)`` ``wp.uint32`` sidecar, initializes it from the wall-loss
namespace, then publishes the view. Compatible later calls return the
exact view without allocation or reseeding. Initializing this sidecar
does not reseed a published coagulation stream. Schema-v3 checkpoint
restart can privately restore captured current words without invoking
this acquisition method or reseeding.

**Parameters**

- `rng_states` (`Any | None`) (default: `None`) — Optional ``uint32`` per-box native RNG sidecar.

**Returns**

- (`WallLossResources`) — Stable view containing the pinned RNG sidecar.

**Raises**

- `TypeError` — If a supplied sidecar is not a Warp array.
- `ValueError` — If the sidecar schema, aliasing, session signature, or
an established binding is incompatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3721-L3794)

### particula.execution.gpu_resources.GPUResourceRegistry.assert_step_closed

*method*

```python
def assert_step_closed() -> None
```

Reject lifecycle work while this registry has an open timestep.

This binding-wide check covers every guard sharing the registry rather
than only the guard passed to a lifecycle boundary.

**Raises**

- `RuntimeError` — If a resident timestep remains open.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3194-L3204)

### particula.execution.gpu_resources.GPUResourceRegistry.get_communication_resources

*method*

```python
def get_communication_resources() -> CommunicationResources | None
```

Return the established concrete communication view, if any.

This metadata-only accessor exposes the single identity-pinned resident
communication family so an explicit checkpoint restart can reuse its
restored configuration without accessing registry internals.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3967-L3981)

### particula.execution.gpu_resources.GPUResourceRegistry.initialize_published_streams

*method*

```python
def initialize_published_streams(session: ResidentSession, *, guard: 'ResidentStepGuard', process_ids: tuple[str, ...] | None = None, logical_box_ids: tuple[str, ...] | None = None) -> None
```

Explicitly reinitialize selected currently published stream lanes.

The caller must provide the exact closed guard for this registry's
active session before any stream work. A writer failure faults that
binding and notifies its attached graph-capture owner; no rollback is
promised after a stream writer launches.

**Parameters**

- `session` (`ResidentSession`) — Exact active resident session pinned to this registry.
- `guard` (`'ResidentStepGuard'`) — Exact closed guard bound to ``session`` and this registry.
- `process_ids` (`tuple[str, ...] | None`) (default: `None`) — Optional published processes to initialize.
- `logical_box_ids` (`tuple[str, ...] | None`) (default: `None`) — Optional stable logical box IDs to initialize.

**Raises**

- `TypeError` — If the guard or stream selection has an invalid type.
- `ValueError` — If the binding, lifecycle, or stream selection is
invalid.
- `BaseException` — Propagates a stream-writer failure after faulting the
exact resident binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2495-L2572)

### particula.execution.gpu_resources.GPUResourceRegistry.inspect_published_streams

*method*

```python
def inspect_published_streams(session: ResidentSession) -> PublishedStreamManifest
```

Return frozen metadata for currently published resident streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2467-L2493)

### particula.execution.gpu_resources.GPUResourceRegistry.logical_resource_report

*method*

```python
def logical_resource_report(capacities: ResourceInventoryCapacities) -> LogicalResourceReport
```

Return a frozen direct-module-only logical resource report.

Resolves every canonical manifest role, including both communication
families, from pinned dimensions and explicit capacities. The result
reports logical schema bytes rather than allocator-reserved bytes. This
read-only accessor does not inspect payloads or pointers, or acquire,
allocate, bind, or mutate resources.

**Parameters**

- `capacities` (`ResourceInventoryCapacities`) — Exact frozen carrier of logical collision, gas-edge, and particle-edge capacities.

**Returns**

- (`LogicalResourceReport`) — Pointer-free frozen family and role reports in canonical declaration
- (`LogicalResourceReport`) — order.

**Raises**

- `TypeError` — If ``capacities`` is not an exact
``ResourceInventoryCapacities`` carrier.
- `ValueError` — If a capacity, checked logical-byte calculation, or
pinned session metadata is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1218-L1291)

### particula.execution.gpu_resources.GPUResourceRegistry.prepare_capture_resources

*method*

```python
def prepare_capture_resources(requirements: CaptureResourceRequirements) -> CaptureResourceSet
```

Atomically stage and publish the complete frozen capture set.

This direct-module-only, setup-only transaction reuses an exact
compatible publication by identity. Otherwise, all allocator, schema,
alias, view, report, and new-stream initialization work remains private
until the final non-fallible publication assignments. It does not
capture or replay graphs, admit execution, transfer, synchronize, or
provide a broad package or top-level API.

**Parameters**

- `requirements` (`CaptureResourceRequirements`) — Exact setup-only request describing required resource families and any caller-supplied sidecars.

**Returns**

- (`CaptureResourceSet`) — The newly published capture set, or the existing set for an exact
- (`CaptureResourceSet`) — compatible repeat.

**Raises**

- `TypeError` — If ``requirements`` is not an exact request carrier.
- `ValueError` — If session, inventory, capacities, schemas, identities,
or alias constraints are incompatible.
- `RuntimeError` — If an internal staged family is unexpectedly missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1700-L2036)

### particula.execution.gpu_resources.GPUResourceRegistry.register_capture_resources

*method*

```python
def register_capture_resources(session: ResidentSession, communication_resources: CommunicationResources | None, registrations: tuple[Any, ...]) -> SelectedResourceInventory
```

Pin one exact metadata-only communication and diagnostics selection.

The first successful call retains an immutable identity inventory. A
later exact repeat returns it; all other repeats reject without changing
the retained inventory. A communication view, when supplied, must be
an already-published closed GAS or PARTICLES view. Candidate metadata
and nonaliasing are completely validated before first publication. This
method performs no allocation, device dispatch, synchronization,
transfer, or payload inspection.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `communication_resources` (`CommunicationResources | None`) — Optional published closed communication view to include in the selection.
- `registrations` (`tuple[Any, ...]`) — Exact ordered tuple of diagnostic registrations.

**Returns**

- (`SelectedResourceInventory`) — The newly retained inventory, or the same inventory for an exact
- (`SelectedResourceInventory`) — repeat registration.

**Raises**

- `TypeError` — If a carrier or registration tuple has the wrong exact
type.
- `ValueError` — If session identity, publication identity, schemas,
or nonaliasing constraints are invalid, or a different
selection was already registered.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2290-L2390)

### particula.execution.gpu_resources.GPUResourceRegistry.release_open_step

*method*

```python
def release_open_step(token: Any) -> None
```

Release the exact outstanding binding-level timestep token.

**Parameters**

- `token` (`Any`) — The exact token previously reserved for this binding.

**Raises**

- `ValueError` — If ``token`` is not the outstanding token by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3181-L3192)

### particula.execution.gpu_resources.GPUResourceRegistry.reserve_open_step

*method*

```python
def reserve_open_step(token: Any) -> None
```

Reserve the binding's sole open timestep token by identity.

The resident step guard calls this only after validating the pinned
session. This bookkeeping-only seam neither acquires resources nor
performs runtime work.

**Parameters**

- `token` (`Any`) — Newly created opaque resident-step token.

**Raises**

- `RuntimeError` — If this exact registry binding already has a token.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3164-L3179)

### particula.execution.gpu_resources.GPUResourceRegistry.selected_resource_report

*method*

```python
def selected_resource_report() -> SelectedResourceInventory
```

Return the one previously registered concrete capture inventory.

The retained report is returned by identity. This accessor validates
only the pinned active session and does not rescan arrays, allocate,
synchronize, transfer, inspect payloads, or rebuild role metadata.

**Returns**

- (`SelectedResourceInventory`) — The immutable inventory retained during registration.

**Raises**

- `ValueError` — If no capture selection has been registered or the
pinned session has drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1293-L1310)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_capture_resource_set

*method*

```python
def validate_capture_resource_set(requirements: CaptureResourceRequirements) -> CaptureResourceSet
```

Return the retained matching capture set without resource work.

This direct-module-only accessor is metadata-only. It returns the exact
set, nested views, and cached immutable logical-byte report retained by
setup. It neither constructs reports nor acquires, allocates,
initializes, reads, transfers, synchronizes, executes, or mutates any
resource binding. It is not a broad package or top-level API.

**Parameters**

- `requirements` (`CaptureResourceRequirements`) — Exact request previously used to publish the set.

**Returns**

- (`CaptureResourceSet`) — The retained capture set, including its cached report, returned by
- (`CaptureResourceSet`) — object identity.

**Raises**

- `TypeError` — If ``requirements`` is not an exact request carrier.
- `ValueError` — If the request or pinned session has drifted, or no set
has been published.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L1447-L1497)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_coagulation_resources

*method*

```python
def validate_coagulation_resources(session: ResidentSession, resources: CoagulationResources) -> None
```

Require the exact established coagulation view.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `resources` (`CoagulationResources`) — Exact published coagulation resource view.

**Raises**

- `TypeError` — If ``resources`` is not an exact coagulation view.
- `ValueError` — If capacity, view identity, or sidecar bindings changed.

This does not acquire a new resource binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2843-L2873)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_communication_resources

*method*

```python
def validate_communication_resources(session: ResidentSession, resources: CommunicationResources) -> None
```

Metadata-validate an established communication resource view.

This execution-time seam requires the exact active pinned session and
published view, then rechecks mode, identities, shapes, device,
contiguity, and nonaliasing metadata. It intentionally does not repeat
Resource setup payload validation, allocate, acquire, inspect values,
transfer,
synchronize, mutate bindings, or invoke a native primitive.

**Parameters**

- `session` (`ResidentSession`) — Exact active session retained by this registry.
- `resources` (`CommunicationResources`) — Exact published communication resource view.

**Raises**

- `TypeError` — If ``resources`` or its configuration has an inexact
concrete type.
- `ValueError` — If the session, mode, view, sidecar identity, or schema
no longer matches the pinned binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L3983-L4080)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_condensation_resources

*method*

```python
def validate_condensation_resources(session: ResidentSession, resources: CondensationResources) -> None
```

Require the exact established condensation view.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `resources` (`CondensationResources`) — Exact published condensation resource view.

**Raises**

- `TypeError` — If ``resources`` is not an exact condensation view.
- `ValueError` — If the view or any pinned sidecar binding has changed.

This does not acquire a new resource binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2815-L2841)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_diagnostic_outputs

*method*

```python
def validate_diagnostic_outputs(session: ResidentSession, outputs: tuple[Any, ...]) -> None
```

Validate caller-owned diagnostic outputs without publishing them.

The diagnostics boundary owns neither these arrays nor their lifetime.
This metadata-only check rejects aliasing with resident primaries and
acquired sidecars, while accepting canonical empty ``(B, S)`` arrays.
It does not allocate, launch, synchronize, transfer, inspect payloads,
acquire a sidecar, or mutate registry state.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `outputs` (`tuple[Any, ...]`) — Exact tuple of caller-owned ``float64`` ``(B, S)`` Warp arrays to validate in registration order.

**Raises**

- `TypeError` — If ``outputs`` is not an exact tuple or an output is not
a Warp array.
- `ValueError` — If session ownership, output schema, device, pointer,
contiguity, or byte-range nonaliasing validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2574-L2639)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_diagnostic_registrations

*method*

```python
def validate_diagnostic_registrations(session: ResidentSession, registrations: tuple[Any, ...]) -> None
```

Validate closed diagnostic bindings without acquiring resources.

Outputs and accounting inputs are caller-owned, same-device contiguous
arrays. Inputs may alias other inputs because they are read-only, but
no input or output may overlap resident or acquired storage, and outputs
may not overlap any diagnostic binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2641-L2678)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_dilution_resources

*method*

```python
def validate_dilution_resources(session: ResidentSession, resources: DilutionResources) -> None
```

Validate one complete unbound dilution view without publishing it.

Dilution is descriptor-only in this phase, so this validation accepts a
caller-owned complete view but deliberately does not retain it in the
registry. It performs metadata-only exact-schema, device, contiguity,
primary-alias, role-reuse, and cross-family nonaliasing checks without
allocating, publishing, initializing, or mutating sidecars.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `resources` (`DilutionResources`) — Complete caller-owned dilution descriptor view.

**Raises**

- `TypeError` — If ``resources`` is not an exact ``DilutionResources``
carrier or either sidecar is not a Warp array.
- `ValueError` — If the pinned session, sidecar schema, device,
contiguity, identity, or byte-range nonaliasing is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2941-L2987)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_nucleation_resources

*method*

```python
def validate_nucleation_resources(session: ResidentSession, resources: NucleationResources) -> None
```

Validate one established nucleation view without acquiring sidecars.

This concrete-only adapter seam verifies exact active-session ownership,
the exact published view identity, and every pinned record binding. It
retains all resource records and sidecars by identity and does not
allocate, acquire, inspect payloads, mutate state, transfer,
synchronize, or recover failures.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `resources` (`NucleationResources`) — Exact established nucleation resource view.

**Raises**

- `TypeError` — If ``resources`` is not an exact nucleation view.
- `ValueError` — If the family is unavailable, the view differs from the
published view, or a pinned sidecar binding changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2875-L2939)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_pinned_session

*method*

```python
def validate_pinned_session(session: ResidentSession) -> None
```

Validate the exact active pinned session without acquiring resources.

This direct-module-only guard seam first requires ``session is`` the
registry's retained session. It then performs the existing metadata-only
active-lifecycle, container identity, primary-array identity, and schema
validation. It does not inspect payloads, acquire sidecars, allocate,
transfer, synchronize, execute, or mutate registry bindings or views.

**Parameters**

- `session` (`ResidentSession`) — The exact resident session retained at registry creation.

**Raises**

- `ValueError` — If ``session`` is not the retained object or its active
lifecycle, schema, protected container identity, or primary
array identity signature changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2426-L2445)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_prepared_resource_views

*method*

```python
def validate_prepared_resource_views(session: ResidentSession, views: PreparedResourceViews) -> None
```

Validate a complete private prepared-view carrier read-only.

Bound family views must be the exact established publication. The
descriptor-only dilution view is checked against the same protected and
established storage, but is not acquired or retained. This method is a
setup-only seam; enqueue paths retain the validated references and do
not call it again. It does not inspect payloads, acquire, allocate,
publish, initialize, transfer, synchronize, or mutate state.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `views` (`PreparedResourceViews`) — Exact optional-family carrier to validate.

**Raises**

- `TypeError` — If ``views`` or a supplied nested resource carrier has
the wrong exact type.
- `ValueError` — If session, publication identity, sidecar schema, or
nonaliasing validation fails for a supplied family.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2989-L3023)

### particula.execution.gpu_resources.GPUResourceRegistry.validate_wall_loss_resources

*method*

```python
def validate_wall_loss_resources(session: ResidentSession, resources: WallLossResources) -> None
```

Validate one established wall-loss view without acquiring sidecars.

This concrete-only adapter seam verifies the exact active session before
checking that ``resources`` is the already-published wall-loss view. It
retains the view and its RNG sidecar by identity, performs metadata and
identity checks only, and does not allocate, acquire, inspect payloads,
mutate registry state, transfer, synchronize, or recover failures.

**Parameters**

- `session` (`ResidentSession`) — Exact active session pinned by this registry.
- `resources` (`WallLossResources`) — Exact established wall-loss resource view.

**Raises**

- `TypeError` — If ``resources`` is not an exact wall-loss view.
- `ValueError` — If the family is unavailable, the view differs from the
published view, or its pinned sidecar binding changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L2780-L2813)

## particula.execution.gpu_resources.LogicalResourceFamilyReport

*class* · *dataclass*

```python
class LogicalResourceFamilyReport
```

Report frozen logical accounting for one direct-module-only family.

Counts cover manifest-defined roles only, not allocator-reserved bytes.
Constructing the report does not inspect payloads or acquire resources.

**Attributes**

- `family` (`str`) — Canonical manifest family name.
- `roles` (`tuple[LogicalResourceRoleReport, ...]`) — Ordered frozen reports for the family's declared roles.
- `logical_byte_count` (`int`) — Sum of role logical schema bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L233-L248)

### particula.execution.gpu_resources.LogicalResourceFamilyReport.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L246-L246)

### particula.execution.gpu_resources.LogicalResourceFamilyReport.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L248-L248)

### particula.execution.gpu_resources.LogicalResourceFamilyReport.roles

*attribute* · *instance attribute*

```python
roles: tuple[LogicalResourceRoleReport, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L247-L247)

### particula.execution.gpu_resources.LogicalResourceFamilyReport.__init__

*method*

```python
def __init__(family: str, roles: tuple[LogicalResourceRoleReport, ...], logical_byte_count: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.LogicalResourceReport

*class* · *dataclass*

```python
class LogicalResourceReport
```

Report frozen aggregate accounting for direct-module-only manifests.

The pointer-free report covers manifest-defined roles only and reports
logical schema bytes, not allocator-reserved bytes. Its read-only creation
neither inspects payloads nor acquires, allocates, binds, or mutates
sidecars.

**Attributes**

- `families` (`tuple[LogicalResourceFamilyReport, ...]`) — Ordered frozen reports for all canonical manifest families.
- `logical_byte_count` (`int`) — Sum of family logical schema bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L251-L266)

### particula.execution.gpu_resources.LogicalResourceReport.families

*attribute* · *instance attribute*

```python
families: tuple[LogicalResourceFamilyReport, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L265-L265)

### particula.execution.gpu_resources.LogicalResourceReport.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L266-L266)

### particula.execution.gpu_resources.LogicalResourceReport.__init__

*method*

```python
def __init__(families: tuple[LogicalResourceFamilyReport, ...], logical_byte_count: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.LogicalResourceRoleReport

*class* · *dataclass*

```python
class LogicalResourceRoleReport
```

Report frozen logical accounting for one direct-module-only role.

Counts describe the resolved schema, not allocator-reserved bytes. Report
construction is read-only: it neither inspects device payloads nor acquires,
allocates, binds, or mutates resources.

**Attributes**

- `entry` (`ResolvedResourceInventoryEntry`) — Pointer-free resolved metadata for the manifest role.
- `element_count` (`int`) — Logical element count of ``entry.shape``.
- `logical_byte_count` (`int`) — Logical schema bytes, excluding allocator overhead.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L214-L230)

### particula.execution.gpu_resources.LogicalResourceRoleReport.element_count

*attribute* · *instance attribute*

```python
element_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L229-L229)

### particula.execution.gpu_resources.LogicalResourceRoleReport.entry

*attribute* · *instance attribute*

```python
entry: ResolvedResourceInventoryEntry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L228-L228)

### particula.execution.gpu_resources.LogicalResourceRoleReport.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L230-L230)

### particula.execution.gpu_resources.LogicalResourceRoleReport.__init__

*method*

```python
def __init__(entry: ResolvedResourceInventoryEntry, element_count: int, logical_byte_count: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.ManifestEntry

*class* · *dataclass*

```python
class ManifestEntry
```

Describe one fixed-shape concrete sidecar role.

**Attributes**

- `role` (`str`) — Name of the sidecar within its resource family.
- `family` (`str`) — Canonical resource family that owns the role.
- `dtype` (`Any`) — Warp dtype declared for the sidecar.
- `shape_kind` (`_ShapeKind`) — Symbolic shape formula resolved by the registry.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L135-L149)

### particula.execution.gpu_resources.ManifestEntry.dtype

*attribute* · *instance attribute*

```python
dtype: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L148-L148)

### particula.execution.gpu_resources.ManifestEntry.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L147-L147)

### particula.execution.gpu_resources.ManifestEntry.role

*attribute* · *instance attribute*

```python
role: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L146-L146)

### particula.execution.gpu_resources.ManifestEntry.shape_kind

*attribute* · *instance attribute*

```python
shape_kind: _ShapeKind
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L149-L149)

### particula.execution.gpu_resources.ManifestEntry.__init__

*method*

```python
def __init__(role: str, family: str, dtype: Any, shape_kind: _ShapeKind) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.NucleationResources

*class* · *dataclass*

```python
class NucleationResources
```

Expose complete native nucleation sidecar records.

**Attributes**

- `scratch` (`NucleationScratchBuffers`) — Pinned working arrays for nucleation planning.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Pinned finalized-demand arrays.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Pinned diagnostic arrays.
- `exhaustion` (`NucleationExhaustionBuffers`) — Pinned exhaustion and resampling arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L434-L448)

### particula.execution.gpu_resources.NucleationResources.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: NucleationDiagnosticBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L447-L447)

### particula.execution.gpu_resources.NucleationResources.exhaustion

*attribute* · *instance attribute*

```python
exhaustion: NucleationExhaustionBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L448-L448)

### particula.execution.gpu_resources.NucleationResources.finalized_demand

*attribute* · *instance attribute*

```python
finalized_demand: NucleationFinalizedDemandBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L446-L446)

### particula.execution.gpu_resources.NucleationResources.scratch

*attribute* · *instance attribute*

```python
scratch: NucleationScratchBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L445-L445)

### particula.execution.gpu_resources.NucleationResources.__init__

*method*

```python
def __init__(scratch: NucleationScratchBuffers, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers, exhaustion: NucleationExhaustionBuffers) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.PreparedResourceViews

*class* · *dataclass*

```python
class PreparedResourceViews
```

Carry optional complete prepared-process views by exact identity.

This private concrete carrier groups independently optional resource views
for prepared process adapters. Validation requires exact view carriers and
validates every supplied family read-only against the pinned active session;
established families must be their published views, while dilution remains
descriptor-only. Constructing or validating this carrier never acquires,
publishes, allocates, initializes, or mutates resource state.

**Attributes**

- `condensation` (`CondensationResources | None`) — Optional established condensation resource view.
- `coagulation` (`CoagulationResources | None`) — Optional established coagulation resource view.
- `dilution` (`DilutionResources | None`) — Optional complete caller-owned dilution descriptor view.
- `wall_loss` (`WallLossResources | None`) — Optional established wall-loss resource view.
- `nucleation` (`NucleationResources | None`) — Optional established nucleation resource view.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L498-L521)

### particula.execution.gpu_resources.PreparedResourceViews.coagulation

*attribute* · *class attribute* · *instance attribute*

```python
coagulation: CoagulationResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L518-L518)

### particula.execution.gpu_resources.PreparedResourceViews.condensation

*attribute* · *class attribute* · *instance attribute*

```python
condensation: CondensationResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L517-L517)

### particula.execution.gpu_resources.PreparedResourceViews.dilution

*attribute* · *class attribute* · *instance attribute*

```python
dilution: DilutionResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L519-L519)

### particula.execution.gpu_resources.PreparedResourceViews.nucleation

*attribute* · *class attribute* · *instance attribute*

```python
nucleation: NucleationResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L521-L521)

### particula.execution.gpu_resources.PreparedResourceViews.wall_loss

*attribute* · *class attribute* · *instance attribute*

```python
wall_loss: WallLossResources | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L520-L520)

### particula.execution.gpu_resources.PreparedResourceViews.__init__

*method*

```python
def __init__(condensation: CondensationResources | None = None, coagulation: CoagulationResources | None = None, dilution: DilutionResources | None = None, wall_loss: WallLossResources | None = None, nucleation: NucleationResources | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.PublishedStreamManifest

*class* · *dataclass*

```python
class PublishedStreamManifest
```

Describe immutable identity metadata for currently published streams.

No live device arrays, pointers, device values, or current stream words are
exposed by this inspection carrier.

**Attributes**

- `stream` (`StreamManifest`) — Frozen stream identity and descriptor metadata.
- `published_process_ids` (`tuple[str, ...]`) — Processes with currently published streams.
- `sidecar_roles` (`tuple[tuple[str, str], ...]`) — Published process and sidecar-role pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L339-L354)

### particula.execution.gpu_resources.PublishedStreamManifest.published_process_ids

*attribute* · *instance attribute*

```python
published_process_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L353-L353)

### particula.execution.gpu_resources.PublishedStreamManifest.sidecar_roles

*attribute* · *instance attribute*

```python
sidecar_roles: tuple[tuple[str, str], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L354-L354)

### particula.execution.gpu_resources.PublishedStreamManifest.stream

*attribute* · *instance attribute*

```python
stream: StreamManifest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L352-L352)

### particula.execution.gpu_resources.PublishedStreamManifest.__init__

*method*

```python
def __init__(stream: StreamManifest, published_process_ids: tuple[str, ...], sidecar_roles: tuple[tuple[str, str], ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.ResidentCommunicationState

*class* · *dataclass*

```python
class ResidentCommunicationState
```

Expose registry-pinned status and snapshot storage for barriers.

**Attributes**

- `invalid` (`Any`) — Device status lane for invalid communication state.
- `active_or_demand` (`Any`) — Device status lane for active or demanded work.
- `volume_invalid` (`Any`) — Device status lane for invalid volume state.
- `volume_changed` (`Any`) — Device status lane for volume-change detection.
- `initial_masses` (`Any | None`) — Optional particle-mass snapshot storage.
- `initial_concentration` (`Any | None`) — Optional particle-concentration snapshot.
- `initial_charge` (`Any | None`) — Optional particle-charge snapshot.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L475-L495)

### particula.execution.gpu_resources.ResidentCommunicationState.active_or_demand

*attribute* · *instance attribute*

```python
active_or_demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L490-L490)

### particula.execution.gpu_resources.ResidentCommunicationState.initial_charge

*attribute* · *class attribute* · *instance attribute*

```python
initial_charge: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L495-L495)

### particula.execution.gpu_resources.ResidentCommunicationState.initial_concentration

*attribute* · *class attribute* · *instance attribute*

```python
initial_concentration: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L494-L494)

### particula.execution.gpu_resources.ResidentCommunicationState.initial_masses

*attribute* · *class attribute* · *instance attribute*

```python
initial_masses: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L493-L493)

### particula.execution.gpu_resources.ResidentCommunicationState.invalid

*attribute* · *instance attribute*

```python
invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L489-L489)

### particula.execution.gpu_resources.ResidentCommunicationState.volume_changed

*attribute* · *instance attribute*

```python
volume_changed: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L492-L492)

### particula.execution.gpu_resources.ResidentCommunicationState.volume_invalid

*attribute* · *instance attribute*

```python
volume_invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L491-L491)

### particula.execution.gpu_resources.ResidentCommunicationState.__init__

*method*

```python
def __init__(invalid: Any, active_or_demand: Any, volume_invalid: Any, volume_changed: Any, initial_masses: Any | None = None, initial_concentration: Any | None = None, initial_charge: Any | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.ResolvedResourceInventoryEntry

*class* · *dataclass*

```python
class ResolvedResourceInventoryEntry
```

Describe one frozen direct-module-only resolved manifest role.

The pointer-free entry contains declaration-resolved schema metadata only.
It has no device payload, acquisition state, or allocator-capacity
information.

**Attributes**

- `family` (`str`) — Canonical manifest family containing the role.
- `role` (`str`) — Canonical role name within ``family``.
- `dtype` (`Any`) — Declared Warp dtype for the logical role.
- `shape` (`tuple[int, ...]`) — Resolved logical shape.
- `capacity_source` (`Literal['fixed', 'collision_capacity', 'gas_edge_capacity', 'particle_edge_capacity']`) — Input that resolved any dynamic shape extent.
- `ownership` (`Literal['registry_or_caller_sidecar', 'caller_configuration']`) — Declared owner category for the role.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L184-L211)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.capacity_source

*attribute* · *instance attribute*

```python
capacity_source: Literal['fixed', 'collision_capacity', 'gas_edge_capacity', 'particle_edge_capacity']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L205-L210)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.dtype

*attribute* · *instance attribute*

```python
dtype: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L203-L203)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L201-L201)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.ownership

*attribute* · *instance attribute*

```python
ownership: Literal['registry_or_caller_sidecar', 'caller_configuration']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L211-L211)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.role

*attribute* · *instance attribute*

```python
role: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L202-L202)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.shape

*attribute* · *instance attribute*

```python
shape: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L204-L204)

### particula.execution.gpu_resources.ResolvedResourceInventoryEntry.__init__

*method*

```python
def __init__(family: str, role: str, dtype: Any, shape: tuple[int, ...], capacity_source: Literal['fixed', 'collision_capacity', 'gas_edge_capacity', 'particle_edge_capacity'], ownership: Literal['registry_or_caller_sidecar', 'caller_configuration']) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.ResourceInventoryCapacities

*class* · *dataclass*

```python
class ResourceInventoryCapacities
```

Hold frozen capacities for direct-module-only inventory reports.

The carrier resolves declared manifest shapes only. It contains no device
payload or allocator-capacity data, and using it neither acquires nor
allocates resources.

**Attributes**

- `collision_capacity` (`int`) — Logical collision-sidecar capacity.
- `gas_edge_capacity` (`int`) — Logical gas-communication edge capacity.
- `particle_edge_capacity` (`int`) — Logical particle-communication edge capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L165-L181)

### particula.execution.gpu_resources.ResourceInventoryCapacities.collision_capacity

*attribute* · *instance attribute*

```python
collision_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L179-L179)

### particula.execution.gpu_resources.ResourceInventoryCapacities.gas_edge_capacity

*attribute* · *instance attribute*

```python
gas_edge_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L180-L180)

### particula.execution.gpu_resources.ResourceInventoryCapacities.particle_edge_capacity

*attribute* · *instance attribute*

```python
particle_edge_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L181-L181)

### particula.execution.gpu_resources.ResourceInventoryCapacities.__init__

*method*

```python
def __init__(collision_capacity: int, gas_edge_capacity: int, particle_edge_capacity: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.ResourceManifest

*class* · *dataclass*

```python
class ResourceManifest
```

Declare immutable sidecar schemas for one resource family.

**Attributes**

- `family` (`str`) — Canonical name of the resource family.
- `entries` (`tuple[ManifestEntry, ...]`) — Ordered role declarations comprising the family schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L152-L162)

### particula.execution.gpu_resources.ResourceManifest.entries

*attribute* · *instance attribute*

```python
entries: tuple[ManifestEntry, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L162-L162)

### particula.execution.gpu_resources.ResourceManifest.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L161-L161)

### particula.execution.gpu_resources.ResourceManifest.__init__

*method*

```python
def __init__(family: str, entries: tuple[ManifestEntry, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.SelectedResourceFamilyReport

*class* · *dataclass*

```python
class SelectedResourceFamilyReport
```

Report ordered metadata-only roles selected from one capture family.

Reports retain zero-extent roles in deterministic order even though those
roles do not participate in byte-range overlap checks.

**Attributes**

- `family` (`str`) — Canonical name of the selected resource family.
- `roles` (`tuple[SelectedResourceRole, ...]`) — Ordered identity-pinned role records in the family.
- `logical_byte_count` (`int`) — Checked sum of the roles' logical byte counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L299-L314)

### particula.execution.gpu_resources.SelectedResourceFamilyReport.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L312-L312)

### particula.execution.gpu_resources.SelectedResourceFamilyReport.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L314-L314)

### particula.execution.gpu_resources.SelectedResourceFamilyReport.roles

*attribute* · *instance attribute*

```python
roles: tuple[SelectedResourceRole, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L313-L313)

### particula.execution.gpu_resources.SelectedResourceFamilyReport.__init__

*method*

```python
def __init__(family: str, roles: tuple[SelectedResourceRole, ...], logical_byte_count: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.SelectedResourceInventory

*class* · *dataclass*

```python
class SelectedResourceInventory
```

Retain one exact concrete capture resource selection.

The optional communication view is one already-published, closed GAS or
PARTICLES view; diagnostic registrations remain caller-owned.
Re-registration succeeds only for this exact retained identity selection.
No device data is copied, inspected, or made checkpoint-visible.

**Attributes**

- `communication_resources` (`'CommunicationResources | None'`) — Selected closed communication view, if present.
- `registrations` (`tuple[Any, ...]`) — Exact ordered diagnostic-registration tuple.
- `families` (`tuple[SelectedResourceFamilyReport, ...]`) — Deterministic selected-family reports.
- `logical_byte_count` (`int`) — Checked total of all selected role bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L317-L336)

### particula.execution.gpu_resources.SelectedResourceInventory.communication_resources

*attribute* · *instance attribute*

```python
communication_resources: 'CommunicationResources | None'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L333-L333)

### particula.execution.gpu_resources.SelectedResourceInventory.families

*attribute* · *instance attribute*

```python
families: tuple[SelectedResourceFamilyReport, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L335-L335)

### particula.execution.gpu_resources.SelectedResourceInventory.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L336-L336)

### particula.execution.gpu_resources.SelectedResourceInventory.registrations

*attribute* · *instance attribute*

```python
registrations: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L334-L334)

### particula.execution.gpu_resources.SelectedResourceInventory.__init__

*method*

```python
def __init__(communication_resources: 'CommunicationResources | None', registrations: tuple[Any, ...], families: tuple[SelectedResourceFamilyReport, ...], logical_byte_count: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.SelectedResourceRole

*class* · *dataclass*

```python
class SelectedResourceRole
```

Describe one identity-pinned capture resource role.

This concrete-only carrier retains a caller- or registry-owned array by
identity for the lifetime of one capture selection. It records schema and
logical-byte metadata only; it neither transfers ownership nor becomes
checkpoint authority.

**Attributes**

- `canonical_name` (`str`) — Deterministic name for the selected role.
- `family` (`str`) — Resource family containing the role.
- `dtype` (`Any`) — Warp dtype of the retained array.
- `shape` (`tuple[int, ...]`) — Resolved fixed-shape schema.
- `value` (`Any`) — Retained array reference, preserved by identity.
- `element_count` (`int`) — Number of logical elements in ``shape``.
- `logical_byte_count` (`int`) — Logical bytes required by the role.
- `read_only` (`bool`) — Whether the role is a read-only accounting input.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L269-L296)

### particula.execution.gpu_resources.SelectedResourceRole.canonical_name

*attribute* · *instance attribute*

```python
canonical_name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L289-L289)

### particula.execution.gpu_resources.SelectedResourceRole.dtype

*attribute* · *instance attribute*

```python
dtype: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L291-L291)

### particula.execution.gpu_resources.SelectedResourceRole.element_count

*attribute* · *instance attribute*

```python
element_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L294-L294)

### particula.execution.gpu_resources.SelectedResourceRole.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L290-L290)

### particula.execution.gpu_resources.SelectedResourceRole.logical_byte_count

*attribute* · *instance attribute*

```python
logical_byte_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L295-L295)

### particula.execution.gpu_resources.SelectedResourceRole.read_only

*attribute* · *class attribute* · *instance attribute*

```python
read_only: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L296-L296)

### particula.execution.gpu_resources.SelectedResourceRole.shape

*attribute* · *instance attribute*

```python
shape: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L292-L292)

### particula.execution.gpu_resources.SelectedResourceRole.value

*attribute* · *instance attribute*

```python
value: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L293-L293)

### particula.execution.gpu_resources.SelectedResourceRole.__init__

*method*

```python
def __init__(canonical_name: str, family: str, dtype: Any, shape: tuple[int, ...], value: Any, element_count: int, logical_byte_count: int, read_only: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)

## particula.execution.gpu_resources.WallLossResources

*class* · *dataclass*

```python
class WallLossResources
```

Expose one independently initialized wall-loss RNG sidecar.

The registry publishes this identity-bound view only after initializing its
``rng_states`` once from the wall-loss process namespace. Compatible
reacquisition returns the same view and sidecar by identity without
allocation or reseeding. The sidecar is distinct from coagulation state and
has no public reset or inspection API. Schema-v3 checkpoint restart can
privately restore fresh bindings from captured current words.

**Attributes**

- `rng_states` (`Any`) — Persistent per-box ``uint32`` wall-loss RNG state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L416-L431)

### particula.execution.gpu_resources.WallLossResources.rng_states

*attribute* · *instance attribute*

```python
rng_states: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L431-L431)

### particula.execution.gpu_resources.WallLossResources.__init__

*method*

```python
def __init__(rng_states: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resources.py#L0-L0)
