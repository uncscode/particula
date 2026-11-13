# particula.execution.process_adapters

*module*

Delegate resident processes through established concrete resource views.

This direct-import-only module retains resident sessions, registries, and
published resource views by identity. Its adapters perform metadata-only
preflight and call one supported direct GPU kernel without transfers,
synchronization, acquisition, fallback, retry, rollback, or physics. Prepared
bindings freeze selected controls and delegate only retained device calls, with
no enqueue-time validation, readback, allocation, lookup, rebinding, or RNG
initialization/reset. Setup rejection is pre-writer; direct-kernel writer
failures retain their no-rollback boundary. These
concrete names are deliberately not exported from :mod:`particula.execution`.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L1-L1)

## particula.execution.process_adapters.ResidentDilutionAdapter

*class*

```python
class ResidentDilutionAdapter
```

Delegate one dilution request through its exact pinned session.

This concrete-only adapter resolves one supported direct kernel after
metadata-only preflight. It does not acquire resources, transfer data,
synchronize, retry, roll back, or recover writer failures.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L337-L430)

### particula.execution.process_adapters.ResidentDilutionAdapter.execute

*method*

```python
def execute(request: object) -> object
```

Validate and delegate one dilution request.

**Parameters**

- `request` (`object`) — Exact dilution request retaining the pinned session and registry.

**Returns**

- (`object`) — Native result returned by the supported direct dilution kernel.

**Raises**

- `TypeError` — If ``request`` is not an exact dilution request.
- `ValueError` — If the request session is not the registry's active,
pinned session.

Direct-kernel exceptions and mutations propagate without adapter retry,
rollback, recovery, transfer, or synchronization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L412-L430)

### particula.execution.process_adapters.ResidentDilutionAdapter.prepare

*method*

```python
def prepare(request: object) -> _PreparedResidentProcessBinding
```

Validate resident ownership and prepare one pinned dilution call.

An optional descriptor-only dilution view is schema-validated before
resolving the direct step. Its normalized coefficient must be the
request coefficient by identity, and the returned binding retains the
complete view without acquiring, publishing, allocating, or replacing
resources. Omitting the view preserves the direct kernel's normal
fallback behavior.

**Parameters**

- `request` (`object`) — Exact resident dilution request to validate and bind.

**Returns**

- (`_PreparedResidentProcessBinding`) — A binding whose execution reuses the validated direct-kernel state.

**Raises**

- `TypeError` — If ``request`` is not the exact request carrier.
- `ValueError` — If the request session is not pinned, an optional
descriptor view is invalid or mismatched, or direct-kernel
preparation rejects its inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L345-L410)

## particula.execution.process_adapters.ResidentDilutionRequest

*class* · *dataclass*

```python
class ResidentDilutionRequest
```

Retain one dilution call's resident binding and opaque kernel inputs.

This concrete-only carrier retains references without inspecting physical
values. When present, its descriptor-only resource view supplies complete
caller-owned prepared sidecars that the adapter validates read-only through
the registry and retains by identity. The direct dilution kernel owns all
numerical validation and writer failure semantics. The carrier neither
copies, publishes, allocates, nor recovers retained state.

**Attributes**

- `session` (`ResidentSession`) — Exact concrete resident-session reference. Execution validates that it is active and pinned by ``registry``.
- `registry` (`GPUResourceRegistry`) — Exact concrete registry reference. Execution validates its binding to ``session``.
- `coefficient` (`object`) — Opaque dilution coefficient for the direct kernel.
- `time_step` (`object`) — Opaque duration for the direct kernel.
- `resources` (`object | None`) — Optional complete descriptor-only dilution view. When supplied, the registry read-only validates both ``(B,)`` sidecars; its normalized coefficient must be ``coefficient`` by identity, and the prepared binding retains the view and both sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L47-L93)

### particula.execution.process_adapters.ResidentDilutionRequest.coefficient

*attribute* · *instance attribute*

```python
coefficient: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L73-L73)

### particula.execution.process_adapters.ResidentDilutionRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L72-L72)

### particula.execution.process_adapters.ResidentDilutionRequest.resources

*attribute* · *class attribute* · *instance attribute*

```python
resources: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L75-L75)

### particula.execution.process_adapters.ResidentDilutionRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L71-L71)

### particula.execution.process_adapters.ResidentDilutionRequest.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L74-L74)

### particula.execution.process_adapters.ResidentDilutionRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, coefficient: object, time_step: object, resources: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L0-L0)

### particula.execution.process_adapters.ResidentDilutionRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact resident dependency types without kernel imports.

**Raises**

- `TypeError` — If ``session`` or ``registry`` is not its exact required
concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L77-L93)

## particula.execution.process_adapters.ResidentNucleationAdapter

*class*

```python
class ResidentNucleationAdapter
```

Delegate one nucleation request through its exact published view.

This concrete-only adapter validates the exact resident publication before
resolving the private prepare/enqueue seam.  It pins container and sidecar
identity, while mutable device payload values remain the direct kernel's
live authority.  It provides no acquisition, transfer, synchronization,
scheduler-token entry, caching, recovery, or rollback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L635-L757)

### particula.execution.process_adapters.ResidentNucleationAdapter.execute

*method*

```python
def execute(request: object) -> object
```

Validate and delegate one nucleation request without altering result.

**Parameters**

- `request` (`object`) — Exact nucleation request retaining the pinned published view.

**Returns**

- (`object`) — Native result returned by the supported direct nucleation kernel.

**Raises**

- `TypeError` — If ``request`` is not an exact nucleation request or its
resource view has the wrong concrete type.
- `ValueError` — If the session or resource view is no longer the active
registry-pinned publication.

This is a fresh ``prepare(request).execute()`` delegation and does not
cache a convenience binding. Direct-kernel exceptions, including any
observer error, and mutations propagate without adapter retry, rollback,
recovery, transfer, or synchronization. A writer-capable direct-kernel
failure therefore retains that kernel's no-rollback boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L735-L757)

### particula.execution.process_adapters.ResidentNucleationAdapter.prepare

*method*

```python
def prepare(request: object) -> _PreparedResidentProcessBinding
```

Pin an exact resident nucleation request before its one launch.

This concrete-only boundary validates resident and published-resource
identity before resolving a kernel.  Identity drift rejects before a
resolver call or device launch.  It neither enters a scheduler token nor
acquires, replaces, transfers, synchronizes, caches scheduler state, or
inspects payload state.  Production delegates to nucleation's private
prepare/enqueue seam; a replaced public resolver remains a one-call
compatibility path for dependency-injection tests.  The returned record
pins identities, not mutable payload values.

**Parameters**

- `request` (`object`) — Exact resident nucleation request to validate and bind.

**Returns**

- (`_PreparedResidentProcessBinding`) — A frozen binding whose execution issues the already prepared call.

**Raises**

- `TypeError` — If ``request`` is not the exact request carrier.
- `ValueError` — If session or published-resource identity has drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L645-L733)

## particula.execution.process_adapters.ResidentNucleationRequest

*class* · *dataclass*

```python
class ResidentNucleationRequest
```

Retain one nucleation call and its established sidecar resource view.

This concrete-only carrier preserves references by identity and delegates
controls, physical validation, and writer failure semantics unchanged. It
neither copies retained state nor provides recovery.

**Attributes**

- `session` (`ResidentSession`) — Exact concrete resident-session reference. Execution validates that it is active and pinned by ``registry``.
- `registry` (`GPUResourceRegistry`) — Exact concrete registry reference. Execution validates its binding to ``session``.
- `resources` (`NucleationResources`) — Exact concrete nucleation resource view. Execution validates that it is the established publication.
- `config` (`object`) — Opaque direct-kernel configuration.
- `time_step` (`object`) — Opaque duration for the direct kernel.
- `exhaustion_controls` (`object`) — Opaque direct-kernel exhaustion controls.
- `species_binding` (`object | None`) — Explicit native source mapping pinned to the session carriers and complete gas-name order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L191-L234)

### particula.execution.process_adapters.ResidentNucleationRequest.config

*attribute* · *instance attribute*

```python
config: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L216-L216)

### particula.execution.process_adapters.ResidentNucleationRequest.exhaustion_controls

*attribute* · *instance attribute*

```python
exhaustion_controls: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L218-L218)

### particula.execution.process_adapters.ResidentNucleationRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L214-L214)

### particula.execution.process_adapters.ResidentNucleationRequest.resources

*attribute* · *instance attribute*

```python
resources: NucleationResources
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L215-L215)

### particula.execution.process_adapters.ResidentNucleationRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L213-L213)

### particula.execution.process_adapters.ResidentNucleationRequest.species_binding

*attribute* · *class attribute* · *instance attribute*

```python
species_binding: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L219-L219)

### particula.execution.process_adapters.ResidentNucleationRequest.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L217-L217)

### particula.execution.process_adapters.ResidentNucleationRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, resources: NucleationResources, config: object, time_step: object, exhaustion_controls: object, species_binding: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L0-L0)

### particula.execution.process_adapters.ResidentNucleationRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact dependencies without inspecting kernel arguments.

**Raises**

- `TypeError` — If a resident dependency or resource view is not its
exact required concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L221-L234)

## particula.execution.process_adapters.ResidentWallLossAdapter

*class*

```python
class ResidentWallLossAdapter
```

Delegate one wall-loss request through its exact published view.

This concrete-only adapter resolves one supported direct kernel after
metadata-only preflight. It preserves container and sidecar identity and
dispatches selected logical boxes through a private selected-box kernel when
selection is not all boxes, preventing disabled lanes from reaching the
public direct kernel.
It provides no acquisition, transfer, synchronization, rollback, or
recovery.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L433-L632)

### particula.execution.process_adapters.ResidentWallLossAdapter._prepare_selected

*method*

```python
def _prepare_selected(request: ResidentWallLossRequest, enabled_logical_boxes: tuple[int, ...]) -> _PreparedResidentProcessBinding
```

Prepare the complete private selected-lane launch once.

Selection validation, physical-lane resolution, device allocation, and
direct-boundary lookup all occur before the retained binding is
returned. Its execution is strictly enqueue-only.

**Parameters**

- `request` (`ResidentWallLossRequest`) — Validated resident wall-loss request to prepare.
- `enabled_logical_boxes` (`tuple[int, ...]`) — Ascending logical boxes selected for launch.

**Returns**

- (`_PreparedResidentProcessBinding`) — A binding that dispatches only the selected physical lanes.

**Raises**

- `ValueError` — If selected boxes cannot be mapped to resident lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L534-L608)

### particula.execution.process_adapters.ResidentWallLossAdapter.execute

*method*

```python
def execute(request: object) -> object
```

Validate and delegate one wall-loss request.

**Parameters**

- `request` (`object`) — Exact wall-loss request retaining the pinned published view.

**Returns**

- (`object`) — Native result returned by the supported direct wall-loss kernel.

**Raises**

- `TypeError` — If ``request`` is not an exact wall-loss request or its
resource view has the wrong concrete type.
- `ValueError` — If the session or resource view is no longer the active
registry-pinned publication.

An empty selection returns without resolving the kernel. For a partial
selection, preparation freezes a private selected-box launch set. The
retained binding executes only its already-resolved delegate.
Direct-kernel exceptions and mutations propagate without adapter retry,
rollback, recovery, transfer, or synchronization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L610-L632)

### particula.execution.process_adapters.ResidentWallLossAdapter.prepare

*method*

```python
def prepare(request: object) -> _PreparedResidentProcessBinding
```

Preflight resident ownership and prepare one wall-loss call.

**Parameters**

- `request` (`object`) — Exact resident wall-loss request to validate and bind.

**Returns**

- (`_PreparedResidentProcessBinding`) — A binding for the selected, all-box, or valid empty dispatch.

**Raises**

- `TypeError` — If ``request`` is not the exact request carrier.
- `ValueError` — If the session or resource publication is stale, or
selection validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L445-L532)

## particula.execution.process_adapters.ResidentWallLossRequest

*class* · *dataclass*

```python
class ResidentWallLossRequest
```

Retain one wall-loss call and its established RNG resource view.

This concrete-only carrier preserves every reference by identity. It
validates scheduler-owned logical-box selection and requires direct-kernel
RNG reset to remain disabled; physical validation and writer failure
semantics remain with the direct step. It neither copies retained state nor
provides recovery.

**Attributes**

- `session` (`ResidentSession`) — Exact concrete resident-session reference. Execution validates that it is active and pinned by ``registry``.
- `registry` (`GPUResourceRegistry`) — Exact concrete registry reference. Execution validates its binding to ``session``.
- `resources` (`WallLossResources`) — Exact concrete wall-loss resource view. Execution validates that it is the established publication.
- `config` (`object`) — Opaque direct-kernel configuration.
- `time_step` (`object`) — Opaque duration for the direct kernel.
- `rng_seed` (`object`) — Opaque seed forwarded unchanged to the direct kernel.
- `initialize_rng` (`object`) — Legacy reset flag. Resident dispatch accepts only the literal ``False`` and always disables direct-kernel resets.
- `enabled_box_indices` (`tuple[int, ...] | None`) — Scheduler-resolved, ascending logical box indices to dispatch. ``None`` retains the legacy all-box selection.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L96-L188)

### particula.execution.process_adapters.ResidentWallLossRequest.config

*attribute* · *instance attribute*

```python
config: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L125-L125)

### particula.execution.process_adapters.ResidentWallLossRequest.enabled_box_indices

*attribute* · *class attribute* · *instance attribute*

```python
enabled_box_indices: tuple[int, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L129-L129)

### particula.execution.process_adapters.ResidentWallLossRequest.initialize_rng

*attribute* · *class attribute* · *instance attribute*

```python
initialize_rng: object = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L128-L128)

### particula.execution.process_adapters.ResidentWallLossRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L123-L123)

### particula.execution.process_adapters.ResidentWallLossRequest.resources

*attribute* · *instance attribute*

```python
resources: WallLossResources
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L124-L124)

### particula.execution.process_adapters.ResidentWallLossRequest.rng_seed

*attribute* · *class attribute* · *instance attribute*

```python
rng_seed: object = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L127-L127)

### particula.execution.process_adapters.ResidentWallLossRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L122-L122)

### particula.execution.process_adapters.ResidentWallLossRequest.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L126-L126)

### particula.execution.process_adapters.ResidentWallLossRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, resources: WallLossResources, config: object, time_step: object, rng_seed: object = 0, initialize_rng: object = False, enabled_box_indices: tuple[int, ...] | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L0-L0)

### particula.execution.process_adapters.ResidentWallLossRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact dependencies without inspecting kernel arguments.

**Raises**

- `TypeError` — If a resident dependency or resource view is not its
exact required concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L131-L147)

### particula.execution.process_adapters.ResidentWallLossRequest.validate_enabled_box_indices

*method*

```python
def validate_enabled_box_indices() -> tuple[int, ...]
```

Validate and return the scheduler-owned wall-loss launch set.

``None`` selects every logical box in ascending order. An explicit tuple
must already be strictly ascending, unique, and within the resident
box range; this preserves the one-to-one logical-lane mapping used by
the persistent wall-loss RNG sidecar.

**Returns**

- (`int`) — Ascending selected logical-box indices. ``None`` returns all
- (`...`) — resident boxes in ascending order.

**Raises**

- `TypeError` — If an explicit selection is not a tuple or contains a
non-integral or boolean index.
- `ValueError` — If an index is negative, repeated, unordered, or
outside the resident box range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L149-L188)

## particula.execution.process_adapters._PreparedResidentProcessBinding

*class* · *dataclass*

```python
class _PreparedResidentProcessBinding
```

Privately keep one prepared process record and enqueue delegate together.

For prepared direct-kernel calls, the record freezes validated references
and ``execute`` only invokes the retained delegate.  It performs no repeated
adapter validation, resource lookup, allocation, normalization, token entry,
or RNG setup.  The direct-kernel record pins object identity, not a snapshot
of mutable device payloads; the delegate's live inputs remain authoritative.
Any direct-kernel status observation and writer rollback boundary remain
with that delegate rather than this resident adapter. This is a
concrete-only seam, not a scheduler gate or public execution protocol.

**Attributes**

- `prepared` (`Any`) — Frozen direct-kernel preparation record, or adapter-local no-op state.
- `enqueue` (`Callable[[Any], object]`) — Callable that issues the already prepared native operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L267-L301)

### particula.execution.process_adapters._PreparedResidentProcessBinding.enqueue

*attribute* · *instance attribute*

```python
enqueue: Callable[[Any], object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L287-L287)

### particula.execution.process_adapters._PreparedResidentProcessBinding.prepared

*attribute* · *instance attribute*

```python
prepared: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L286-L286)

### particula.execution.process_adapters._PreparedResidentProcessBinding.retained_resources

*attribute* · *class attribute* · *instance attribute*

```python
retained_resources: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L288-L288)

### particula.execution.process_adapters._PreparedResidentProcessBinding.__init__

*method*

```python
def __init__(prepared: Any, enqueue: Callable[[Any], object], retained_resources: Any | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L0-L0)

### particula.execution.process_adapters._PreparedResidentProcessBinding.execute

*method*

```python
def execute() -> object
```

Invoke the pinned delegate without repeating adapter setup.

**Returns**

- (`object`) — Result returned by the retained direct-kernel delegate.

The binding does not enter a scheduler token, inspect payloads, acquire
resources, synchronize, or provide rollback. Delegate exceptions and
any observer readback propagate unchanged; a writer-capable failure has
no rollback guarantee.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L290-L301)

## particula.execution.process_adapters._PreparedSelectedWallLossCall

*class* · *dataclass*

```python
class _PreparedSelectedWallLossCall
```

Freeze private selected-lane wall-loss launch state at setup.

The selected physical lanes, device array, direct delegate, and all launch
references are established once. Enqueue only invokes ``delegate`` and
never looks up, validates, allocates, or rereads the request/session.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L304-L334)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.config

*attribute* · *instance attribute*

```python
config: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L316-L316)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.delegate

*attribute* · *instance attribute*

```python
delegate: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L313-L313)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.environment

*attribute* · *instance attribute*

```python
environment: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L320-L320)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L314-L314)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.rng_seed

*attribute* · *instance attribute*

```python
rng_seed: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L317-L317)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.rng_states

*attribute* · *instance attribute*

```python
rng_states: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L318-L318)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.selected_boxes

*attribute* · *instance attribute*

```python
selected_boxes: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L319-L319)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L315-L315)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.__init__

*method*

```python
def __init__(delegate: Callable[..., object], particles: Any, time_step: object, config: object, rng_seed: object, rng_states: Any, selected_boxes: Any, environment: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L0-L0)

### particula.execution.process_adapters._PreparedSelectedWallLossCall.enqueue

*method*

```python
def enqueue() -> object
```

Issue the frozen selected-lane call without setup work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L322-L334)

## particula.execution.process_adapters._get_dilution_step_gpu

*function*

```python
def _get_dilution_step_gpu() -> Callable[..., object]
```

Lazily import the sole supported direct dilution boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L237-L241)

## particula.execution.process_adapters._get_nucleation_step_gpu

*function*

```python
def _get_nucleation_step_gpu() -> Callable[..., object]
```

Lazily import the sole supported direct nucleation boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L260-L264)

## particula.execution.process_adapters._get_wall_loss_selected_boxes_step_gpu

*function*

```python
def _get_wall_loss_selected_boxes_step_gpu() -> Callable[..., object]
```

Lazily import the private batched resident wall-loss dispatch seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L251-L257)

## particula.execution.process_adapters._get_wall_loss_step_gpu

*function*

```python
def _get_wall_loss_step_gpu() -> Callable[..., object]
```

Lazily import the sole supported direct wall-loss boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L244-L248)

## particula.execution.process_adapters._resource_types

*function*

```python
def _resource_types() -> tuple[type[object], type[object], type[object]]
```

Import concrete resource types only when a carrier needs them.

**Returns**

- (`type[object]`) — The exact registry, wall-loss resource, and nucleation resource types,
- (`type[object]`) — in that order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_adapters.py#L31-L44)
