# particula.execution.adapters.coagulation

*module*

Provide concrete-only CPU and resident-Warp Brownian adapters.

state carriers and execution states retain caller-owned resources by identity.
Frozen dataclasses prohibit field rebinding only; retained resources remain
mutable. The adapters dispatch only their already-selected CPU runnable or
resident-Warp kernel. They do not select another backend, transfer,
synchronize, convert, retry, recover, or roll back caller-owned state.

The persistent RNG sidecar records direct-Warp dispatch intent. A dispatch may
seed it only when ``initialize_rng`` is true; otherwise it reuses the supplied
sidecar, including when the seed is unchanged. Callers own synchronization and
recovery after a kernel launch.

Prepared bindings freeze operation selection and dispatch retained references
without repeating host/setup metadata validation or performing allocation, host
readback, transfer, synchronization, lookup, or rebinding or RNG
initialization/reset. Retained resident-Warp operations still perform
device-side status and physical-state validation. Rejection is pre-writer; no
rollback is promised after a writer-capable call. CPU and resident-Warp calls
have independent stochastic trajectories. This concrete boundary provides no
seed-by-seed cross-backend trajectory comparison.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L1-L1)

## particula.execution.adapters.coagulation.BrownianCoagulationStrategy

*attribute* · *module attribute*

```python
BrownianCoagulationStrategy = brownian_coagulation_strategy.BrownianCoagulationStrategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L43-L45)

## particula.execution.adapters.coagulation.BrownianCoagulationConfig

*class* · *dataclass*

```python
class BrownianCoagulationConfig
```

Denote the direct-kernel default Brownian coagulation mechanism.

This fieldless, concrete-only marker selects no backend and performs no
mechanism or physical-value validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L55-L61)

### particula.execution.adapters.coagulation.BrownianCoagulationConfig.__init__

*method*

```python
def __init__() -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

## particula.execution.adapters.coagulation.CPUCoagulationExecutionAdapter

*class*

```python
class CPUCoagulationExecutionAdapter
```

Dispatch one selected CPU coagulation request exactly once.

This concrete-only adapter validates local controls and invokes the
caller-owned runnable once without splitting, converting, retrying, or
recovering. Successful results retain the original aerosol and declare
state mutation. Delegate exceptions propagate unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L340-L384)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Execute one exact CPU execution state without fallback or recovery.

**Parameters**

- `state` (`ExecutionState`) — Exact selected CPU execution state.

**Returns**

- (`ExecutionResult`) — A result retaining ``state`` and the original aerosol by identity.

**Raises**

- `TypeError` — If ``state`` or its time step has an invalid type.
- `ValueError` — If controls are invalid or the runnable returns another
aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L349-L384)

## particula.execution.adapters.coagulation.CPUCoagulationExecutionState

*class* · *dataclass*

```python
class CPUCoagulationExecutionState
```

Retain selected CPU coagulation controls and runnable by identity.

This concrete-only execution carrier validates only exact state-state and
runnable
types. Execution controls remain opaque until dispatch, and retained
resources remain caller-owned and mutable.

**Parameters**

- `state` (`CPUCoagulationState`) — Exact state CPU state retained by identity.
- `time_step` (`object`) — Original execution time step in s, validated at dispatch.
- `sub_steps` (`object`) — Original number of runnable substeps, validated at dispatch.
- `runnable` (`Coagulation`) — Exact caller-owned ``Coagulation`` runnable.

**Raises**

- `TypeError` — If ``state`` or ``runnable`` is not the required exact type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L151-L189)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.runnable

*attribute* · *instance attribute*

```python
runnable: Coagulation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L173-L173)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.state

*attribute* · *instance attribute*

```python
state: CPUCoagulationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L170-L170)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.sub_steps

*attribute* · *instance attribute*

```python
sub_steps: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L172-L172)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L171-L171)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.backend_payload

*property*

```python
backend_payload: Aerosol
```

Return the exact caller-owned CPU aerosol payload.

**Returns**

- (`Aerosol`) — The aerosol retained by the state state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L183-L189)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.__init__

*method*

```python
def __init__(state: CPUCoagulationState, time_step: object, sub_steps: object, runnable: Coagulation) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.CPUCoagulationExecutionState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact state-state and runnable carrier types.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L175-L180)

## particula.execution.adapters.coagulation.CPUCoagulationResult

*class* · *dataclass*

```python
class CPUCoagulationResult
```

Retain a CPU result that preserves its state's aerosol identity.

This carrier is write-free and validates that its result aerosol is the
exact resource retained by the supplied CPU state.

**Parameters**

- `state` (`CPUCoagulationState`) — Exact CPU request state retained by identity.
- `aerosol` (`Aerosol`) — The state's caller-owned aerosol.

**Raises**

- `TypeError` — If ``state`` is not an exact ``CPUCoagulationState`` or
``aerosol`` is not an ``Aerosol``.
- `ValueError` — If ``aerosol`` is not ``state.aerosol``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L107-L148)

### particula.execution.adapters.coagulation.CPUCoagulationResult.aerosol

*attribute* · *instance attribute*

```python
aerosol: Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L125-L125)

### particula.execution.adapters.coagulation.CPUCoagulationResult.state

*attribute* · *instance attribute*

```python
state: CPUCoagulationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L124-L124)

### particula.execution.adapters.coagulation.CPUCoagulationResult.backend_payload

*property*

```python
backend_payload: Aerosol
```

Return the retained caller-owned aerosol by identity.

**Returns**

- (`Aerosol`) — The exact aerosol retained by this result.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L142-L148)

### particula.execution.adapters.coagulation.CPUCoagulationResult.__init__

*method*

```python
def __init__(state: CPUCoagulationState, aerosol: Aerosol) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.CPUCoagulationResult.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate result ownership without inspecting the aerosol.

**Raises**

- `TypeError` — If the state or aerosol has an invalid type.
- `ValueError` — If the result aerosol differs from the state's aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L127-L139)

## particula.execution.adapters.coagulation.CPUCoagulationState

*class* · *dataclass*

```python
class CPUCoagulationState
```

Retain one caller-owned CPU Brownian coagulation request.

Construction validates the exact marker type and the aerosol type without
inspecting aerosol state or strategy. It does not execute a runnable, copy
resources, or validate coagulation physics. Frozen status prevents field
rebinding only; the retained aerosol remains mutable and caller-owned.

**Parameters**

- `config` (`BrownianCoagulationConfig`) — Exact Brownian mechanism marker.
- `aerosol` (`Aerosol`) — Caller-owned aerosol retained by identity.

**Raises**

- `TypeError` — If ``config`` is not an exact ``BrownianCoagulationConfig``
or ``aerosol`` is not an ``Aerosol``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L64-L104)

### particula.execution.adapters.coagulation.CPUCoagulationState.aerosol

*attribute* · *instance attribute*

```python
aerosol: Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L83-L83)

### particula.execution.adapters.coagulation.CPUCoagulationState.config

*attribute* · *instance attribute*

```python
config: BrownianCoagulationConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L82-L82)

### particula.execution.adapters.coagulation.CPUCoagulationState.backend_payload

*property*

```python
backend_payload: Aerosol
```

Return the retained caller-owned aerosol by identity.

**Returns**

- (`Aerosol`) — The exact aerosol retained by this state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L98-L104)

### particula.execution.adapters.coagulation.CPUCoagulationState.__init__

*method*

```python
def __init__(config: BrownianCoagulationConfig, aerosol: Aerosol) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.CPUCoagulationState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate carrier types without inspecting aerosol state.

**Raises**

- `TypeError` — If the configuration marker or aerosol has an invalid
type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L85-L95)

## particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionAdapter

*class*

```python
class ResidentBrownianCoagulationExecutionAdapter
```

Dispatch one pre-acquired resident Brownian request by identity.

The adapter preflights exact session, registry, resource, output, and RNG
identities before lazily importing the kernel. It then supplies the
resident-owned configuration-derived coagulation sidecar with literal
``initialize_rng=False``. It does not acquire, allocate, seed, reset,
inspect, transfer, or synchronize the stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L808-L924)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Preflight bindings and dispatch one forced-no-reset kernel call.

**Parameters**

- `state` (`ExecutionState`) — Exact resident Brownian execution state.

**Returns**

- (`ExecutionResult`) — A result retaining the request and kernel result resources by
- (`ExecutionResult`) — identity.

**Raises**

- `TypeError` — If ``state`` is not the exact resident execution state.
- `ValueError` — If reset is requested or resident identities are stale
or inconsistent.
- `ImportError` — If the optional direct Warp kernel is unavailable after
preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L894-L924)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionAdapter.prepare

*method*

```python
def prepare(state: ExecutionState) -> _PreparedResidentBrownianCoagulationBinding
```

Preflight and freeze one resident Brownian kernel call.

**Parameters**

- `state` (`ExecutionState`) — Exact resident Brownian state containing the pinned request and published resource view.

**Returns**

- (`_PreparedResidentBrownianCoagulationBinding`) — A binding that can enqueue the native call without repeating setup.

**Raises**

- `TypeError` — If ``state`` is not the exact resident state carrier.
- `ValueError` — If reset is requested or a resident identity is stale
or inconsistent.
- `ImportError` — If the optional direct Warp kernel is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L818-L892)

## particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState

*class* · *dataclass*

```python
class ResidentBrownianCoagulationExecutionState
```

Bind a Brownian request to one established resident RNG stream.

The retained registry view owns the exact configuration-initialized,
coagulation-only
``wp.uint32`` RNG sidecar. Resident dispatch never acquires, allocates,
resets, inspects, transfers, or synchronizes that sidecar; it always
forwards it with ``initialize_rng=False``. Wall-loss streams and public
reset or inspection APIs are outside this concrete-only boundary.

**Attributes**

- `request` (`WarpBrownianCoagulationExecutionState`) — Exact direct Brownian request retaining process inputs.
- `session` (`ResidentSession`) — Exact active resident session owning the primary containers.
- `registry` (`GPUResourceRegistry`) — Exact registry that published ``resources`` for ``session``.
- `resources` (`CoagulationResources`) — Published collision outputs and persistent RNG sidecar.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L734-L780)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L754-L754)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.request

*attribute* · *instance attribute*

```python
request: WarpBrownianCoagulationExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L752-L752)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.resources

*attribute* · *instance attribute*

```python
resources: CoagulationResources
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L755-L755)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L753-L753)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.backend_payload

*property*

```python
backend_payload: object
```

Return the exact caller-owned resident particle payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L778-L780)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.__init__

*method*

```python
def __init__(request: WarpBrownianCoagulationExecutionState, session: ResidentSession, registry: GPUResourceRegistry, resources: CoagulationResources) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.ResidentBrownianCoagulationExecutionState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate concrete carrier types without acquiring resources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L757-L775)

## particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionAdapter

*class*

```python
class WarpBrownianCoagulationExecutionAdapter
```

Dispatch one selected resident-Warp Brownian request exactly once.

This concrete-only adapter resolves the selected kernel lazily after local
state validation and calls it once. It forwards state resources by identity
without conversion, synchronization, fallback, retry, or recovery. Kernel
exceptions and any post-launch mutation remain governed by the kernel.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L941-L999)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionAdapter.execute

*method*

```python
def execute(state: ExecutionState) -> ExecutionResult
```

Execute one exact Warp execution state without fallback or recovery.

**Parameters**

- `state` (`ExecutionState`) — Exact selected resident-Warp execution state.

**Returns**

- (`ExecutionResult`) — A result retaining ``state`` and native result resources by
- (`ExecutionResult`) — identity.

**Raises**

- `TypeError` — If ``state`` is not the required exact execution state.
- `ValueError` — If the direct kernel returns resources that violate
state
result identity requirements. Direct kernel errors propagate
unchanged.
- `ImportError` — If optional Warp kernel dependencies are unavailable
after local preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L950-L999)

## particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionState

*class* · *dataclass*

```python
class WarpBrownianCoagulationExecutionState
```

Retain one selected resident-Warp coagulation request by identity.

This concrete-only execution carrier validates only the exact state state.
It neither
imports a kernel nor inspects or synchronizes caller-owned resident
resources.

**Parameters**

- `state` (`WarpBrownianCoagulationState`) — Exact state resident-Warp state retained by identity.

**Raises**

- `TypeError` — If ``state`` is not an exact
``WarpBrownianCoagulationState``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L698-L731)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionState.state

*attribute* · *instance attribute*

```python
state: WarpBrownianCoagulationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L715-L715)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionState.backend_payload

*property*

```python
backend_payload: object
```

Return the exact caller-owned resident particle payload.

**Returns**

- (`object`) — The particle container retained by the state state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L725-L731)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionState.__init__

*method*

```python
def __init__(state: WarpBrownianCoagulationState) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationExecutionState.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact state Warp state carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L717-L722)

## particula.execution.adapters.coagulation.WarpBrownianCoagulationResult

*class* · *dataclass*

```python
class WarpBrownianCoagulationResult
```

Retain a direct-Warp result and supplied resource identities.

State-supplied diagnostic outputs must be returned by identity. Diagnostics
omitted from the state are allocated and returned according to the native
direct-kernel contract.

**Parameters**

- `state` (`WarpBrownianCoagulationState`) — Exact resident-Warp request state retained by identity.
- `particles` (`object`) — The state's caller-owned particle container.
- `collision_pairs` (`object | None`) — Returned collision-pair output, when applicable.
- `n_collisions` (`object | None`) — Returned collision-count output, when applicable.

**Raises**

- `TypeError` — If ``state`` is not an exact
``WarpBrownianCoagulationState``.
- `ValueError` — If particles or a state-supplied diagnostic lacks the
required identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L640-L695)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.collision_pairs

*attribute* · *instance attribute*

```python
collision_pairs: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L663-L663)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.n_collisions

*attribute* · *instance attribute*

```python
n_collisions: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L664-L664)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.particles

*attribute* · *instance attribute*

```python
particles: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L662-L662)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.state

*attribute* · *instance attribute*

```python
state: WarpBrownianCoagulationState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L661-L661)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.backend_payload

*property*

```python
backend_payload: object
```

Return the caller-owned particle container by identity.

**Returns**

- (`object`) — The exact particle container retained by this result.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L689-L695)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.__init__

*method*

```python
def __init__(state: WarpBrownianCoagulationState, particles: object, collision_pairs: object | None, n_collisions: object | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationResult.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate state and state-supplied result-sidecar identities.

**Raises**

- `TypeError` — If the state is not the exact required carrier type.
- `ValueError` — If a required particle or diagnostic identity differs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L666-L686)

## particula.execution.adapters.coagulation.WarpBrownianCoagulationState

*class* · *dataclass*

```python
class WarpBrownianCoagulationState
```

Retain a resident direct-Brownian request without native validation.

Only marker, container kind, environment/direct-input form, selected time,
required persistent RNG, and metadata-detectable sidecar ownership are
checked. Particle, environment/volume, output-buffer, device, dtype,
capacity, and detailed RNG schemas remain native-kernel concerns.

Construction is write-free: it does not import a kernel, transfer,
synchronize, allocate, seed, reset, or advance ``rng_states``. Dispatch
seeds the caller-owned ``(n_boxes,)`` ``wp.uint32`` sidecar only when the
exact boolean ``initialize_rng`` is true, and otherwise reuses it even for
the same seed. Callers own synchronization and recovery after a launch.

**Parameters**

- `config` (`BrownianCoagulationConfig`) — Exact Brownian mechanism marker.
- `particles` (`object`) — Caller-owned resident ``WarpParticleData`` container.
- `temperature` (`object | None`) — Direct caller-owned temperature, or ``None`` with ``environment``.
- `pressure` (`object | None`) — Direct caller-owned pressure, or ``None`` with ``environment``.
- `time_step` (`object`) — Finite, nonnegative real direct-kernel time step in s.
- `volume` (`object | None`) (default: `None`) — Optional opaque direct-kernel volume input.
- `collision_pairs` (`object | None`) (default: `None`) — Optional caller-owned collision-pair output.
- `n_collisions` (`object | None`) (default: `None`) — Optional caller-owned collision-count output.
- `rng_states` (`object | None`) (default: `None`) — Required caller-owned persistent RNG sidecar.
- `rng_seed` (`object`) (default: `0`) — Opaque seed intent retained without interpretation.
- `initialize_rng` (`object`) (default: `False`) — Exact boolean request to reset the supplied RNG sidecar.
- `environment` (`object | None`) (default: `None`) — Caller-owned ``WarpEnvironmentData``, or ``None`` for direct temperature and pressure inputs.

**Raises**

- `TypeError` — If the configuration, particles, or non-``None`` environment
has an invalid type.
- `RuntimeError` — If the optional Warp runtime is unavailable after valid
configuration preflight.
- `ValueError` — If input form or selected time is invalid, no RNG sidecar
is supplied, or metadata establishes forbidden caller-resource
aliasing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L538-L637)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.collision_pairs

*attribute* · *class attribute* · *instance attribute*

```python
collision_pairs: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L586-L586)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.config

*attribute* · *instance attribute*

```python
config: BrownianCoagulationConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L580-L580)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.environment

*attribute* · *class attribute* · *instance attribute*

```python
environment: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L591-L591)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.initialize_rng

*attribute* · *class attribute* · *instance attribute*

```python
initialize_rng: object = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L590-L590)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.n_collisions

*attribute* · *class attribute* · *instance attribute*

```python
n_collisions: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L587-L587)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.particles

*attribute* · *instance attribute*

```python
particles: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L581-L581)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.pressure

*attribute* · *instance attribute*

```python
pressure: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L583-L583)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.rng_seed

*attribute* · *class attribute* · *instance attribute*

```python
rng_seed: object = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L589-L589)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.rng_states

*attribute* · *class attribute* · *instance attribute*

```python
rng_states: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L588-L588)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.temperature

*attribute* · *instance attribute*

```python
temperature: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L582-L582)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.time_step

*attribute* · *instance attribute*

```python
time_step: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L584-L584)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.volume

*attribute* · *class attribute* · *instance attribute*

```python
volume: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L585-L585)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.backend_payload

*property*

```python
backend_payload: object
```

Return the caller-owned particle container by identity.

**Returns**

- (`object`) — The exact particle container retained by this state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L631-L637)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.__init__

*method*

```python
def __init__(config: BrownianCoagulationConfig, particles: object, temperature: object | None, pressure: object | None, time_step: object, volume: object | None = None, collision_pairs: object | None = None, n_collisions: object | None = None, rng_states: object | None = None, rng_seed: object = 0, initialize_rng: object = False, environment: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation.WarpBrownianCoagulationState.__post_init__

*method*

```python
def __post_init__() -> None
```

Perform ordered kind, form, time, required-RNG, and ownership checks.

The configuration check occurs before the lazy Warp import, preserving
CPU-only construction and deterministic error ordering.

**Raises**

- `TypeError` — If a selection-owned kind check fails.
- `RuntimeError` — If Warp is unavailable after configuration preflight.
- `ValueError` — If resource form, selected time, RNG presence, or
ownership is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L593-L628)

## particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding

*class* · *dataclass*

```python
class _PreparedResidentBrownianCoagulationBinding
```

Privately retain one preflighted resident coagulation enqueue.

The prepared record freezes direct-kernel references after resident binding
preflight. ``execute`` only invokes the retained enqueue delegate; it does
not repeat validation, allocation, normalization, lookup, rebinding, or RNG
setup. It is concrete-only and retains the direct writer's no-rollback
boundary.

**Attributes**

- `state` (`ResidentBrownianCoagulationExecutionState`) — Resident request and resource binding validated during setup.
- `prepared` (`Any`) — Frozen direct-kernel preparation record.
- `enqueue` (`Any`) — Callable that issues the already prepared native operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L783-L805)

### particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding.enqueue

*attribute* · *instance attribute*

```python
enqueue: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L801-L801)

### particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding.prepared

*attribute* · *instance attribute*

```python
prepared: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L800-L800)

### particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding.state

*attribute* · *instance attribute*

```python
state: ResidentBrownianCoagulationExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L799-L799)

### particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding.__init__

*method*

```python
def __init__(state: ResidentBrownianCoagulationExecutionState, prepared: Any, enqueue: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L0-L0)

### particula.execution.adapters.coagulation._PreparedResidentBrownianCoagulationBinding.execute

*method*

```python
def execute() -> tuple[Any, Any, Any]
```

Enqueue the frozen native call without repeating setup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L803-L805)

## particula.execution.adapters.coagulation._available_fields

*function*

```python
def _available_fields(resource: object, names: tuple[str, ...]) -> tuple[object, ...]
```

Return protected fields without validating their native schema.

**Parameters**

- `resource` (`object`) — Primary resource whose available attributes are protected.
- `names` (`tuple[str, ...]`) — Attribute names to retrieve when present.

**Returns**

- (`tuple[object, ...]`) — Available attribute values in ``names`` order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L475-L489)

## particula.execution.adapters.coagulation._dtype_itemsize

*function*

```python
def _dtype_itemsize(dtype: object, wp: Any) -> int | None
```

Return a supported Warp scalar size, deferring unknown metadata.

**Parameters**

- `dtype` (`object`) — Candidate Warp scalar dtype metadata.
- `wp` (`Any`) — Imported Warp module containing supported scalar dtype singletons.

**Returns**

- (`int | None`) — The scalar size in bytes, or ``None`` when metadata is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L387-L401)

## particula.execution.adapters.coagulation._get_coagulation_step_gpu

*function*

```python
def _get_coagulation_step_gpu() -> Any
```

Lazily resolve the optional direct Warp coagulation kernel.

**Returns**

- (`Any`) — The direct ``coagulation_step_gpu`` kernel entry point.

**Raises**

- `ImportError` — If optional Warp kernel dependencies are unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L927-L938)

## particula.execution.adapters.coagulation._memory_range

*function*

```python
def _memory_range(array: object, wp: Any) -> tuple[int, int] | None
```

Return a usable contiguous storage range, otherwise defer validation.

This metadata-only helper deliberately treats unknown or malformed arrays as
native-kernel validation concerns.  It never reads device data.

**Parameters**

- `array` (`object`) — Candidate resource with optional Warp-array metadata.
- `wp` (`Any`) — Imported Warp module used to recognize scalar dtypes.

**Returns**

- (`tuple[int, int] | None`) — A half-open byte range for usable contiguous storage, or ``None`` when
- (`tuple[int, int] | None`) — metadata cannot establish a range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L404-L452)

## particula.execution.adapters.coagulation._overlaps

*function*

```python
def _overlaps(first: tuple[int, int] | None, second: tuple[int, int] | None) -> bool
```

Return whether two usable, nonempty storage ranges overlap.

**Parameters**

- `first` (`tuple[int, int] | None`) — Optional first half-open byte range.
- `second` (`tuple[int, int] | None`) — Optional second half-open byte range.

**Returns**

- (`bool`) — True when both ranges are present and share at least one byte.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L455-L472)

## particula.execution.adapters.coagulation._validate_cpu_brownian_runnable

*function*

```python
def _validate_cpu_brownian_runnable(runnable: Coagulation) -> None
```

Validate the selected CPU Brownian particle-resolved capability.

**Parameters**

- `runnable` (`Coagulation`) — Exact caller-owned coagulation runnable selected for dispatch.

**Raises**

- `ValueError` — If the runnable is not the supported Brownian,
particle-resolved configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L221-L238)

## particula.execution.adapters.coagulation._validate_initialize_rng

*function*

```python
def _validate_initialize_rng(initialize_rng: object) -> None
```

Validate the explicit persistent-RNG reset control.

**Parameters**

- `initialize_rng` (`object`) — Candidate request to initialize the supplied RNG state.

**Raises**

- `TypeError` — If ``initialize_rng`` is not an exact boolean.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L208-L218)

## particula.execution.adapters.coagulation._validate_ownership

*function*

```python
def _validate_ownership(wp: Any, protected: tuple[object, ...], collision_pairs: object | None, n_collisions: object | None, rng_states: object) -> None
```

Reject aliases between writable sidecars and protected fields.

Unknown metadata defers to native validation, but identical objects and
metadata-detectable overlapping byte ranges are rejected. This function
reads metadata only and does not mutate or synchronize device resources.

**Parameters**

- `wp` (`Any`) — Imported Warp module used for metadata interpretation.
- `protected` (`tuple[object, ...]`) — Primary resources that writable sidecars cannot alias.
- `collision_pairs` (`object | None`) — Optional caller-owned collision-pair output.
- `n_collisions` (`object | None`) — Optional caller-owned collision-count output.
- `rng_states` (`object`) — Required caller-owned persistent RNG sidecar.

**Raises**

- `ValueError` — If writable sidecars alias a protected resource or each
other.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L492-L535)

## particula.execution.adapters.coagulation._validate_time_step

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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L192-L205)

## particula.execution.adapters.coagulation._validate_warp_brownian_state_controls

*function*

```python
def _validate_warp_brownian_state_controls(state: WarpBrownianCoagulationState) -> None
```

Validate scalar execution controls before ownership checks.

**Parameters**

- `state` (`WarpBrownianCoagulationState`) — Selected resident-Warp request carrier.

**Raises**

- `TypeError` — If the selected time step or reset control has an invalid
type.
- `ValueError` — If the selected time step is non-finite, negative, or the
persistent RNG sidecar is missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L278-L295)

## particula.execution.adapters.coagulation._validate_warp_brownian_state_form

*function*

```python
def _validate_warp_brownian_state_form(state: WarpBrownianCoagulationState, particle_type: type, environment_type: type) -> bool
```

Validate resident-Warp payload form and return environment usage.

**Parameters**

- `state` (`WarpBrownianCoagulationState`) — Selected resident-Warp request carrier.
- `particle_type` (`type`) — Exact Warp particle-data class.
- `environment_type` (`type`) — Exact Warp environment-data class.

**Returns**

- (`bool`) — ``True`` when the environment form is used, otherwise ``False``.

**Raises**

- `TypeError` — If particles or the optional environment has an invalid type.
- `ValueError` — If the selected temperature/pressure form is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L241-L275)

## particula.execution.adapters.coagulation._validate_warp_brownian_state_ownership

*function*

```python
def _validate_warp_brownian_state_ownership(state: WarpBrownianCoagulationState, wp: Any, environment_form: bool) -> None
```

Validate writable sidecar ownership for resident-Warp dispatch.

**Parameters**

- `state` (`WarpBrownianCoagulationState`) — Selected resident-Warp request carrier.
- `wp` (`Any`) — Imported Warp module used for metadata interpretation.
- `environment_form` (`bool`) — Whether the selected request uses environment input.

**Raises**

- `ValueError` — If writable sidecars alias a protected resource or each
other.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/adapters/coagulation.py#L298-L337)
