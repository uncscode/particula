# particula.execution.adapters.condensation

*module*

Provide concrete-only condensation state carriers and selected adapters.

Import these carriers from ``particula.execution.adapters.condensation``, not
from ``particula.execution`` or top-level ``particula``. They retain
caller-owned resources by identity. Inventory finalization carriers perform
read-only
construction-time metadata checks; thermal correction adapters prepare then make
one selected
native call. Adapter carriers and bindings neither transfer, restore,
synchronize, retry, nor fall back. Kernel preparation owns any direct-kernel
fallback allocation, while a prepared binding reuses its retained resources for
enqueue. Frozen fields prevent rebinding only; retained resources remain
mutable and caller-owned. Native calls may mutate particle masses, gas
concentration or vapor pressure, and writable sidecars. Callers own resource
lifetime, synchronization, concurrency, and any post-launch recovery limits.
Setup preserves direct-kernel validation ordering, normalization, and private
fallback allocation; prepared enqueue uses retained references only and performs
no validation, readback, allocation, transfer, synchronization, lookup,
rebinding, or reset. Rejection is pre-writer; no recovery or rollback is
promised after a writer-capable failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L1-L1)

## particula.execution.adapters.condensation.CPUCondensationAdapter

*attribute* · *module attribute*

```python
CPUCondensationAdapter = CPUCondensationExecutionAdapter
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L1022-L1022)

## particula.execution.adapters.condensation.WarpCondensationAdapter

*attribute* · *module attribute*

```python
WarpCondensationAdapter = WarpCondensationExecutionAdapter
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L1023-L1023)

## particula.execution.adapters.condensation.CPUCondensationExecutionAdapter

*class*

```python
class CPUCondensationExecutionAdapter
```

Dispatch one selected isothermal CPU condensation request exactly once.

This concrete-only adapter performs local preflight, then invokes the
caller-owned ``MassCondensation`` runnable once without splitting controls.
It neither converts state nor catches delegate exceptions. The runnable must
return the original aerosol, and successful results declare state mutation.

Backend exceptions propagate unchanged. This adapter is concrete-only and
must be imported from ``particula.execution.adapters.condensation``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L794-L845)

### particula.execution.adapters.condensation.CPUCondensationExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Execute one exact CPU thermal correction state without fallback or
    recovery.

**Parameters**

- `state` (`ExecutionState`) — Exact selected CPU thermal correction execution state.

**Returns**

- (`ExecutionResult`) — A result retaining ``state`` and the original aerosol by identity.

**Raises**

- `TypeError` — If ``state`` or its time step has an invalid type.
- `ValueError` — If controls or the selected profile are invalid, latent
heat is enabled, or the runnable returns another aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L806-L845)

## particula.execution.adapters.condensation.CPUCondensationExecutionState

*class* · *dataclass*

```python
class CPUCondensationExecutionState
```

Retain selected CPU condensation controls and runnable by identity.

Import this concrete-only thermal correction carrier from
``particula.execution.adapters.condensation``. Construction validates only
exact inventory finalization state and runnable types; execution controls
remain opaque until
Dispatch. The retained runnable and aerosol are caller-owned and mutable;
frozen fields prevent rebinding only.

**Parameters**

- `state` (`CPUCondensationState`) — Exact inventory finalization CPU state retained by identity.
- `time_step` (`object`) — Original execution time step in s, validated at dispatch.
- `sub_steps` (`object`) — Original number of CPU substeps, validated at dispatch.
- `runnable` (`MassCondensation`) — Exact caller-owned ``MassCondensation`` runnable.

**Raises**

- `TypeError` — If ``state`` or ``runnable`` is not the required exact type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L611-L657)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.runnable

*attribute* · *instance attribute*

```python
runnable: MassCondensation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L635-L635)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.state

*attribute* · *instance attribute*

```python
state: CPUCondensationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L632-L632)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.sub_steps

*attribute* · *instance attribute*

```python
sub_steps: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L634-L634)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L633-L633)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.backend_payload

*property*

```python
backend_payload: Aerosol
```

Return the exact caller-owned CPU aerosol payload.

**Returns**

- (`Aerosol`) — The aerosol retained by the inventory finalization state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L651-L657)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.__init__

*method*

```python
def __init__(state: CPUCondensationState, time_step: object, sub_steps: object, runnable: MassCondensation) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation.CPUCondensationExecutionState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact inventory finalization-state and runnable carrier
    types.

**Raises**

- `TypeError` — If ``state`` or ``runnable`` is not the required exact
type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L637-L648)

## particula.execution.adapters.condensation.CPUCondensationState

*class* · *dataclass*

```python
class CPUCondensationState
```

Retain caller-owned CPU condensation state without executing it.

This concrete-only inventory finalization carrier validates only the
configuration and aerosol
types, in that order. It does not inspect aerosol backing data, validate
condensation physics, add execution controls, copy resources, or invoke a
runnable. Frozen status prevents field rebinding only; ``aerosol`` remains
caller-owned and mutable.

**Parameters**

- `config` (`CondensationExecutionConfig`) — Exact inventory finalization condensation execution configuration.
- `aerosol` (`Aerosol`) — Caller-owned aerosol retained by identity.

**Raises**

- `TypeError` — If ``config`` is not an exact
``CondensationExecutionConfig`` or ``aerosol`` is not an
``Aerosol``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L76-L123)

### particula.execution.adapters.condensation.CPUCondensationState.aerosol

*attribute* · *instance attribute*

```python
aerosol: Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L99-L99)

### particula.execution.adapters.condensation.CPUCondensationState.config

*attribute* · *instance attribute*

```python
config: CondensationExecutionConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L98-L98)

### particula.execution.adapters.condensation.CPUCondensationState.backend_payload

*property*

```python
backend_payload: Aerosol
```

Return the caller-owned aerosol without inspecting it.

**Returns**

- (`Aerosol`) — The exact aerosol retained by this state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L117-L123)

### particula.execution.adapters.condensation.CPUCondensationState.__init__

*method*

```python
def __init__(config: CondensationExecutionConfig, aerosol: Aerosol) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation.CPUCondensationState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate state inputs in configuration then aerosol order.

**Raises**

- `TypeError` — If ``config`` is not an exact
``CondensationExecutionConfig`` or ``aerosol`` is not an
``Aerosol``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L101-L114)

## particula.execution.adapters.condensation.CondensationExecutionConfig

*class* · *dataclass*

```python
class CondensationExecutionConfig
```

Retain an exact semantic condensation configuration by identity.

This concrete-only inventory finalization carrier does not select a profile
or inspect its
configuration. A future selection caller remains responsible for calling
``require_condensation_profile()`` when that semantic check is needed.
Frozen status prevents field rebinding but does not copy the configuration.

**Parameters**

- `configuration` (`CondensationConfiguration`) — Exact concrete condensation configuration.

**Raises**

- `TypeError` — If ``configuration`` is not an exact
``CondensationConfiguration`` instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L43-L73)

### particula.execution.adapters.condensation.CondensationExecutionConfig.configuration

*attribute* · *instance attribute*

```python
configuration: CondensationConfiguration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L61-L61)

### particula.execution.adapters.condensation.CondensationExecutionConfig.__init__

*method*

```python
def __init__(configuration: CondensationConfiguration) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation.CondensationExecutionConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact configuration carrier type.

**Raises**

- `TypeError` — If ``configuration`` is not an exact
``CondensationConfiguration`` instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L63-L73)

## particula.execution.adapters.condensation.WarpCondensationExecutionAdapter

*class*

```python
class WarpCondensationExecutionAdapter
```

Dispatch one selected Warp request to its direct kernel.

Preflight completes before the optional kernel import. The adapter makes one
native call without conversion, adapter allocation, restoration,
synchronization, fallback, or exception recovery. Profile preflight
completes before lazy resolution. It forwards valid caller-owned
``latent_heat``,
``energy_transfer``, and deferred ``thermal_work`` sidecars by identity.
The direct kernel owns detailed thermal-sidecar validation, execution, and
post-launch mutation limits. Omitted direct-kernel property arrays retain
the native step's documented step-local fallback allocations; callers that
require reuse must supply the direct kernel's caller-owned sidecars.

This concrete-only adapter is imported from
``particula.execution.adapters.condensation``. It forwards the exact
native kernel tuple without reconstructing it.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L848-L952)

### particula.execution.adapters.condensation.WarpCondensationExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Execute one exact Warp thermal correction state with no post-launch
    recovery.

Profile preflight occurs before lazy kernel resolution. After that
preflight, this method forwards the caller-owned thermal sidecars by
identity and leaves detailed validation and any post-launch behavior to
the direct kernel.

**Parameters**

- `state` (`ExecutionState`) — Exact selected Warp thermal correction execution state.

**Returns**

- (`ExecutionResult`) — A result retaining ``state`` and the native kernel result by
- (`ExecutionResult`) — identity.

**Raises**

- `TypeError` — If ``state`` or its time step has an invalid type.
- `ValueError` — If controls or the selected profile are invalid, or if
the direct kernel rejects detailed thermal-sidecar validation.
Direct errors propagate unchanged.
- `ImportError` — If the optional Warp kernel cannot be imported after
successful preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L921-L952)

### particula.execution.adapters.condensation.WarpCondensationExecutionAdapter.prepare

*method*

```python
def prepare(state: WarpCondensationExecutionState) -> _PreparedWarpCondensationBinding
```

Validate and bind one resident state for repeated device enqueues.

**Parameters**

- `state` (`WarpCondensationExecutionState`) — Exact selected Warp thermal correction execution state.

**Returns**

- (`_PreparedWarpCondensationBinding`) — A concrete binding retaining the prepared kernel record and enqueue
- (`_PreparedWarpCondensationBinding`) — delegate by identity.

**Raises**

- `TypeError` — If ``state`` or its time step has an invalid type.
- `ValueError` — If controls, profile, or selected sidecars are invalid.
- `ImportError` — If the optional Warp kernel cannot be imported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L867-L919)

## particula.execution.adapters.condensation.WarpCondensationExecutionState

*class* · *dataclass*

```python
class WarpCondensationExecutionState
```

Retain selected resident-Warp condensation controls by identity.

This concrete-only thermal correction carrier preserves the exact inventory
finalization state and time step.
Construction neither imports a kernel nor inspects retained resources.

**Parameters**

- `state` (`WarpCondensationState`) — Exact inventory finalization Warp state retained by identity.
- `time_step` (`object`) — Original execution time step in s, validated at dispatch.

**Raises**

- `TypeError` — If ``state`` is not an exact ``WarpCondensationState``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L660-L696)

### particula.execution.adapters.condensation.WarpCondensationExecutionState.state

*attribute* · *instance attribute*

```python
state: WarpCondensationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L676-L676)

### particula.execution.adapters.condensation.WarpCondensationExecutionState.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L677-L677)

### particula.execution.adapters.condensation.WarpCondensationExecutionState.backend_payload

*property*

```python
backend_payload: tuple[object, object, object]
```

Return the exact caller-owned resident primary payload.

**Returns**

- (`object`) — The ``(particles, gas, environment)`` tuple from the inventory
- (`object`) — finalization state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L689-L696)

### particula.execution.adapters.condensation.WarpCondensationExecutionState.__init__

*method*

```python
def __init__(state: WarpCondensationState, time_step: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation.WarpCondensationExecutionState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact inventory finalization Warp state carrier type.

**Raises**

- `TypeError` — If ``state`` is not an exact ``WarpCondensationState``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L679-L686)

## particula.execution.adapters.condensation.WarpCondensationState

*class* · *dataclass*

```python
class WarpCondensationState
```

Retain validated caller-owned resident Warp condensation resources.

This concrete-only inventory finalization carrier lazily imports Warp during
construction. It
validates primary-container metadata and writable-output ownership only.
Thermodynamics need only be non-None. ``latent_heat`` and deferred
``thermal_work`` remain opaque, while writable ``mass_transfer`` and
``energy_transfer`` receive metadata and ownership checks. The selected
Warp adapter forwards all three thermal sidecars by identity; the direct
kernel owns their dependency, schema, value, and execution validation.
Construction does not select a profile, execute a kernel or runnable,
transfer, allocate, synchronize, or validate direct-kernel physics.

Rejection occurs before a writer launch and mutates nothing. That
construction guarantee does not promise post-launch rollback for a future
adapter. Frozen status prevents field rebinding only. All retained
containers and sidecars remain caller-owned and mutable.

**Parameters**

- `config` (`CondensationExecutionConfig`) — Exact inventory finalization condensation execution configuration.
- `particles` (`object`) — Resident ``WarpParticleData`` primary container.
- `gas` (`object`) — Resident ``WarpGasData`` primary container.
- `environment` (`object`) — Resident ``WarpEnvironmentData`` primary container.
- `thermodynamics` (`object`) — Required opaque thermodynamics reference.
- `activity_surface` (`object | None`) (default: `None`) — Optional opaque activity and surface reference.
- `scratch_buffers` (`object | None`) (default: `None`) — Optional opaque scratch-buffer reference.
- `mass_transfer` (`object | None`) (default: `None`) — Optional writable mass-transfer output sidecar.
- `latent_heat` (`object | None`) (default: `None`) — Optional opaque latent-heat reference.
- `energy_transfer` (`object | None`) (default: `None`) — Optional writable energy-transfer output sidecar.
- `thermal_work` (`object | None`) (default: `None`) — Optional opaque thermal-work reference.
- `species_binding` (`object | None`) (default: `None`) — Required explicit pinned species map for discrete or resolved particles; untagged carriers are rejected.

**Raises**

- `TypeError` — If the configuration or a primary container has an invalid
type.
- `RuntimeError` — If the optional Warp runtime is unavailable when state
validation requires it.
- `ValueError` — If primary metadata, required thermodynamics, writable
output metadata, or writable-output ownership is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L310-L608)

### particula.execution.adapters.condensation.WarpCondensationState.activity_surface

*attribute* · *class attribute* · *instance attribute*

```python
activity_surface: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L360-L360)

### particula.execution.adapters.condensation.WarpCondensationState.config

*attribute* · *instance attribute*

```python
config: CondensationExecutionConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L355-L355)

### particula.execution.adapters.condensation.WarpCondensationState.energy_transfer

*attribute* · *class attribute* · *instance attribute*

```python
energy_transfer: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L364-L364)

### particula.execution.adapters.condensation.WarpCondensationState.environment

*attribute* · *instance attribute*

```python
environment: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L358-L358)

### particula.execution.adapters.condensation.WarpCondensationState.gas

*attribute* · *instance attribute*

```python
gas: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L357-L357)

### particula.execution.adapters.condensation.WarpCondensationState.latent_heat

*attribute* · *class attribute* · *instance attribute*

```python
latent_heat: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L363-L363)

### particula.execution.adapters.condensation.WarpCondensationState.mass_transfer

*attribute* · *class attribute* · *instance attribute*

```python
mass_transfer: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L362-L362)

### particula.execution.adapters.condensation.WarpCondensationState.particles

*attribute* · *instance attribute*

```python
particles: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L356-L356)

### particula.execution.adapters.condensation.WarpCondensationState.scratch_buffers

*attribute* · *class attribute* · *instance attribute*

```python
scratch_buffers: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L361-L361)

### particula.execution.adapters.condensation.WarpCondensationState.species_binding

*attribute* · *class attribute* · *instance attribute*

```python
species_binding: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L366-L366)

### particula.execution.adapters.condensation.WarpCondensationState.thermal_work

*attribute* · *class attribute* · *instance attribute*

```python
thermal_work: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L365-L365)

### particula.execution.adapters.condensation.WarpCondensationState.thermodynamics

*attribute* · *instance attribute*

```python
thermodynamics: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L359-L359)

### particula.execution.adapters.condensation.WarpCondensationState.backend_payload

*property*

```python
backend_payload: tuple[object, object, object]
```

Return the three caller-owned primary containers by identity.

**Returns**

- (`tuple[object, object, object]`) — The ``(particles, gas, environment)`` primary-container tuple.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L602-L608)

### particula.execution.adapters.condensation.WarpCondensationState.__init__

*method*

```python
def __init__(config: CondensationExecutionConfig, particles: object, gas: object, environment: object, thermodynamics: object, activity_surface: object | None = None, scratch_buffers: object | None = None, mass_transfer: object | None = None, latent_heat: object | None = None, energy_transfer: object | None = None, thermal_work: object | None = None, species_binding: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation.WarpCondensationState.__post_init__

*method*

```python
def __post_init__() -> None
```

Perform ordered metadata and writable-output ownership validation.

The lazy imports make importing this concrete module and constructing
CPU-only carriers independent of the optional Warp runtime.

**Raises**

- `TypeError` — If the configuration or primary-container types are
invalid.
- `RuntimeError` — If Warp is unavailable.
- `ValueError` — If required metadata, thermodynamics, or writable
output ownership is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L368-L419)

### particula.execution.adapters.condensation.WarpCondensationState._validate_optional_outputs

*method*

```python
def _validate_optional_outputs(wp: Any, boxes: int, particle_count: int, species: int, device: Any) -> None
```

Validate optional outputs and ownership against primary state.

**Parameters**

- `wp` (`Any`) — Imported Warp module used for dtype metadata.
- `boxes` (`int`) — Number of validated boxes in the primary state.
- `particle_count` (`int`) — Number of validated particles in the primary state.
- `species` (`int`) — Number of validated species in the primary state.
- `device` (`Any`) — Device metadata from the validated mass storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L545-L599)

### particula.execution.adapters.condensation.WarpCondensationState._validate_population_binding

*method*

```python
def _validate_population_binding() -> None
```

Require a matching native mapping without reading payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L421-L429)

### particula.execution.adapters.condensation.WarpCondensationState._validate_primary_arrays

*method*

```python
def _validate_primary_arrays(wp: Any) -> Any
```

Validate all primary Warp arrays and return the mass storage.

**Parameters**

- `wp` (`Any`) — Imported Warp module used for dtype metadata.

**Returns**

- (`Any`) — The validated ``particles.masses`` array.

**Raises**

- `ValueError` — If a required primary array is missing or malformed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L431-L543)

## particula.execution.adapters.condensation._PreparedWarpCondensationBinding

*class* · *dataclass*

```python
class _PreparedWarpCondensationBinding
```

Retain one validated Warp condensation call for device enqueue.

The binding keeps the exact selected state, prepared kernel record, and
enqueue delegate together so repeated execution cannot silently replace
caller-owned resident resources. It is concrete-module-only. Construction
delegates validation and any fallback resolution to kernel preparation;
``execute()`` delegates to the retained enqueue callable without repeating
adapter setup, validation, allocation, transfer, synchronization, lookup,
or rebinding. Rejection before launch is pre-writer; post-launch rollback is
not provided.

**Attributes**

- `state` (`WarpCondensationExecutionState`) — Exact selected Warp execution state retained by identity.
- `prepared` (`Any`) — Private kernel record produced during preparation.
- `enqueue` (`Any`) — Private device-only enqueue callable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L699-L734)

### particula.execution.adapters.condensation._PreparedWarpCondensationBinding.enqueue

*attribute* · *instance attribute*

```python
enqueue: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L720-L720)

### particula.execution.adapters.condensation._PreparedWarpCondensationBinding.prepared

*attribute* · *instance attribute*

```python
prepared: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L719-L719)

### particula.execution.adapters.condensation._PreparedWarpCondensationBinding.state

*attribute* · *instance attribute*

```python
state: WarpCondensationExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L718-L718)

### particula.execution.adapters.condensation._PreparedWarpCondensationBinding.__init__

*method*

```python
def __init__(state: WarpCondensationExecutionState, prepared: Any, enqueue: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L0-L0)

### particula.execution.adapters.condensation._PreparedWarpCondensationBinding.execute

*method*

```python
def execute() -> tuple[Any, Any]
```

Enqueue the retained kernel call without repeating setup.

**Returns**

- (`Any`) — The direct-kernel result, normally the resident particle container
- (`Any`) — and finalized total-transfer buffer.

**Raises**

- `ValueError` — If the already-prepared device sequence encounters an
existing runtime validation failure. Post-launch rollback is
not provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L722-L734)

## particula.execution.adapters.condensation._dtype_itemsize

*function*

```python
def _dtype_itemsize(dtype: Any) -> int
```

Return the fixed byte size of a inventory finalization primary Warp
    dtype.

**Parameters**

- `dtype` (`Any`) — inventory finalization-validated ``wp.float64`` or ``wp.int32`` dtype metadata.

**Returns**

- (`int`) — The dtype item size in bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L277-L293)

## particula.execution.adapters.condensation._get_condensation_step_gpu

*function*

```python
def _get_condensation_step_gpu() -> Any
```

Lazily resolve the prepared direct Warp kernel path after preflight.

**Returns**

- (`Any`) — A direct-call-compatible prepared setup/enqueue delegate.

**Raises**

- `ImportError` — If the optional Warp kernel dependencies are unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L768-L791)

## particula.execution.adapters.condensation._memory_range

*function*

```python
def _memory_range(name: str, array: Any, itemsize: int, *, require_contiguous: bool = True) -> tuple[int, int] | None
```

Return a metadata-only storage range, or ``None`` for an empty array.

**Parameters**

- `name` (`str`) — Qualified field name used in ownership errors.
- `array` (`Any`) — Metadata-valid Warp array whose storage is inspected.
- `itemsize` (`int`) — Size in bytes of one array element.
- `require_contiguous` (`bool`) (default: `True`) — Whether to reject non-contiguous storage.

**Returns**

- (`tuple[int, int] | None`) — The half-open byte range occupied by nonempty storage, or ``None`` for
- (`tuple[int, int] | None`) — an empty array.

**Raises**

- `ValueError` — If required pointer or stride metadata is absent, or if
contiguous storage is required but unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L161-L222)

## particula.execution.adapters.condensation._overlaps

*function*

```python
def _overlaps(first: tuple[int, int], second: tuple[int, int] | None) -> bool
```

Return whether two half-open byte ranges overlap.

**Parameters**

- `first` (`tuple[int, int]`) — Nonempty first byte range.
- `second` (`tuple[int, int] | None`) — Optional second byte range, with ``None`` representing empty storage.

**Returns**

- (`bool`) — True if both ranges contain at least one common byte.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L296-L307)

## particula.execution.adapters.condensation._require_isothermal

*function*

```python
def _require_isothermal(configuration: CondensationConfiguration) -> None
```

Reject semantic latent heat at the selected isothermal boundary.

**Parameters**

- `configuration` (`CondensationConfiguration`) — Selected condensation configuration to inspect.

**Raises**

- `ValueError` — If the configuration enables latent heat.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L753-L765)

## particula.execution.adapters.condensation._validate_array

*function*

```python
def _validate_array(name: str, array: Any, dtype: Any, shape: tuple[int, ...] | None, device: Any | None = None) -> None
```

Validate Warp-array metadata without device operations.

**Parameters**

- `name` (`str`) — Qualified field name used in validation errors.
- `array` (`Any`) — Candidate Warp array whose metadata is read.
- `dtype` (`Any`) — Required Warp dtype.
- `shape` (`tuple[int, ...] | None`) — Required array shape, or ``None`` to validate only metadata.
- `device` (`Any | None`) (default: `None`) — Required device, or ``None`` when no device comparison applies.

**Raises**

- `ValueError` — If the candidate lacks required Warp metadata or its dtype,
shape, or device is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L126-L158)

## particula.execution.adapters.condensation._validate_output_ownership

*function*

```python
def _validate_output_ownership(mass_transfer: Any | None, energy_transfer: Any | None, primary_fields: tuple[Any, ...]) -> None
```

Reject writable outputs overlapping primary fields or each other.

This check reads pointer, shape, stride, and dtype metadata only. Omitted
and empty outputs have no storage range and therefore cannot overlap.

**Parameters**

- `mass_transfer` (`Any | None`) — Optional writable mass-transfer output.
- `energy_transfer` (`Any | None`) — Optional writable energy-transfer output.
- `primary_fields` (`tuple[Any, ...]`) — Primary state arrays that outputs must not overlap.

**Raises**

- `ValueError` — If a supplied output is non-contiguous or aliases primary
state or the other supplied output.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L225-L274)

## particula.execution.adapters.condensation._validate_selected_warp_sidecars

*function*

```python
def _validate_selected_warp_sidecars(state: WarpCondensationState, configuration: CondensationConfiguration) -> None
```

Validate semantic-profile sidecars before native kernel resolution.

**Parameters**

- `state` (`WarpCondensationState`) — Validated inventory finalization state holding caller-owned sidecars.
- `configuration` (`CondensationConfiguration`) — Selected semantic configuration.

**Raises**

- `ValueError` — If a required sidecar is missing or incompatible with the
selected latent-heat, activity, or surface-tension semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L955-L1018)

## particula.execution.adapters.condensation._validate_time_step

*function*

```python
def _validate_time_step(time_step: object) -> None
```

Validate a non-boolean, finite, nonnegative real time step.

**Parameters**

- `time_step` (`object`) — Candidate execution time step in s.

**Raises**

- `TypeError` — If ``time_step`` is not a non-boolean real scalar.
- `ValueError` — If ``time_step`` is non-finite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/condensation.py#L737-L750)
