# particula.execution.prepared

*module*

Owned preparation, execution and observation of native Warp simulations.

Construction is routed through runnable preparation. Primary arrays and process
sidecars remain private and resident until an explicit observation. Warp CPU is
an explicit regression target; requesting CUDA never falls back to the CPU.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L1-L1)

## particula.execution.prepared.PreparedSimulation

*class*

```python
class PreparedSimulation
```

Own a fixed-layout simulation and its explicit CPU observation boundary.

Use a runnable's ``prepare`` method to construct this object. ``step`` and
``run`` enqueue work without downloading state. ``synchronize`` waits for
pending work, and ``snapshot`` returns detached CPU data for analysis.
Closing waits for outstanding work and releases owned storage without an
automatic download. Instances serialize their own operations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L23-L655)

### particula.execution.prepared.PreparedSimulation._auxiliary

*attribute* · *instance attribute*

```python
_auxiliary: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L51-L51)

### particula.execution.prepared.PreparedSimulation._bindings

*attribute* · *instance attribute*

```python
_bindings: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L48-L48)

### particula.execution.prepared.PreparedSimulation._capture

*attribute* · *instance attribute*

```python
_capture: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L52-L52)

### particula.execution.prepared.PreparedSimulation._clock

*attribute* · *instance attribute*

```python
_clock: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L54-L54)

### particula.execution.prepared.PreparedSimulation._closed

*attribute* · *instance attribute*

```python
_closed: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L37-L37)

### particula.execution.prepared.PreparedSimulation._device

*attribute* · *instance attribute*

```python
_device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L34-L34)

### particula.execution.prepared.PreparedSimulation._device_failure

*attribute* · *instance attribute*

```python
_device_failure: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L47-L47)

### particula.execution.prepared.PreparedSimulation._diagnostics

*attribute* · *instance attribute*

```python
_diagnostics: Callable[[], Any] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L45-L45)

### particula.execution.prepared.PreparedSimulation._extension_fields

*attribute* · *instance attribute*

```python
_extension_fields: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L55-L55)

### particula.execution.prepared.PreparedSimulation._extension_state

*attribute* · *instance attribute*

```python
_extension_state: dict[str, dict[str, Any]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L56-L56)

### particula.execution.prepared.PreparedSimulation._guard

*attribute* · *instance attribute*

```python
_guard: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L43-L43)

### particula.execution.prepared.PreparedSimulation._host_context

*attribute* · *instance attribute*

```python
_host_context: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L57-L57)

### particula.execution.prepared.PreparedSimulation._initial_time

*attribute* · *instance attribute*

```python
_initial_time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L53-L53)

### particula.execution.prepared.PreparedSimulation._last_steps

*attribute* · *instance attribute*

```python
_last_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L38-L38)

### particula.execution.prepared.PreparedSimulation._last_time

*attribute* · *instance attribute*

```python
_last_time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L39-L39)

### particula.execution.prepared.PreparedSimulation._lock

*attribute* · *instance attribute*

```python
_lock: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L36-L36)

### particula.execution.prepared.PreparedSimulation._observers

*attribute* · *instance attribute*

```python
_observers: list[Callable[[], Any]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L49-L49)

### particula.execution.prepared.PreparedSimulation._operations

*attribute* · *instance attribute*

```python
_operations: tuple[Callable[[], None], ...] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L40-L40)

### particula.execution.prepared.PreparedSimulation._plans

*attribute* · *instance attribute*

```python
_plans: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L50-L50)

### particula.execution.prepared.PreparedSimulation._refresh

*attribute* · *instance attribute*

```python
_refresh: Callable[[], None] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L46-L46)

### particula.execution.prepared.PreparedSimulation._registry

*attribute* · *instance attribute*

```python
_registry: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L42-L42)

### particula.execution.prepared.PreparedSimulation._session

*attribute* · *instance attribute*

```python
_session: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L41-L41)

### particula.execution.prepared.PreparedSimulation._stream

*attribute* · *instance attribute*

```python
_stream: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L44-L44)

### particula.execution.prepared.PreparedSimulation._time_step

*attribute* · *instance attribute*

```python
_time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L35-L35)

### particula.execution.prepared.PreparedSimulation._wp

*attribute* · *instance attribute*

```python
_wp: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L33-L33)

### particula.execution.prepared.PreparedSimulation.captured

*property*

```python
captured: bool
```

Report whether subsequent steps use this owner's live CUDA graph.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L592-L595)

### particula.execution.prepared.PreparedSimulation.closed

*property*

```python
closed: bool
```

Report whether owned resources have been released.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L279-L281)

### particula.execution.prepared.PreparedSimulation.completed_steps

*property*

```python
completed_steps: int
```

Return enqueued steps; synchronization establishes device success.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L258-L263)

### particula.execution.prepared.PreparedSimulation.current_time

*property*

```python
current_time: float
```

Return absolute enqueued time [s], including the initial time.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L274-L276)

### particula.execution.prepared.PreparedSimulation.device

*property*

```python
device: Device
```

Return the explicit backend and canonical native device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L248-L250)

### particula.execution.prepared.PreparedSimulation.gas_names

*property*

```python
gas_names: tuple[str, ...]
```

Return the fixed full-gas order for updates and analysis.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L450-L454)

### particula.execution.prepared.PreparedSimulation.simulated_time

*property*

```python
simulated_time: float
```

Return enqueued simulated time in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L266-L271)

### particula.execution.prepared.PreparedSimulation.time_step

*property*

```python
time_step: float
```

Return the frozen duration of one process sequence, in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L253-L255)

### particula.execution.prepared.PreparedSimulation.__enter__

*method*

```python
def __enter__() -> PreparedSimulation
```

Enter an active simulation's explicit ownership scope.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L648-L651)

### particula.execution.prepared.PreparedSimulation.__exit__

*method*

```python
def __exit__(*_exc: object) -> None
```

Release resources on normal exit and exceptions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L653-L655)

### particula.execution.prepared.PreparedSimulation.__init__

*method*

```python
def __init__() -> None
```

Reject direct construction without a validated process plan.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L59-L61)

### particula.execution.prepared.PreparedSimulation._bind

*method*

```python
def _bind(plans: tuple[Any, ...]) -> None
```

Compile a validated process plan against the owned resident state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L159-L210)

### particula.execution.prepared.PreparedSimulation._empty

*method* · *classmethod*

```python
def _empty(device: Device, time_step: float) -> PreparedSimulation
```

Create an owner and stream before either upload or exact restart.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L125-L157)

### particula.execution.prepared.PreparedSimulation._fault

*method*

```python
def _fault() -> None
```

Preserve resident fault and graph invalidation semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L234-L245)

### particula.execution.prepared.PreparedSimulation._require_active

*method*

```python
def _require_active() -> None
```

Reject terminal state before observing or enqueueing anything.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L219-L227)

### particula.execution.prepared.PreparedSimulation._scope

*method*

```python
def _scope()
```

Select this owner's device/stream without inter-stream events.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L212-L217)

### particula.execution.prepared.PreparedSimulation._upload

*method* · *classmethod*

```python
def _upload(aerosol: Aerosol, *, device: Device, time_step: float, root_seed: int = 0, logical_box_ids: tuple[str, ...] | None = None) -> PreparedSimulation
```

Upload preflighted state and own its exact resident binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L63-L123)

### particula.execution.prepared.PreparedSimulation._wait

*method*

```python
def _wait() -> None
```

Wait only for this simulation's CUDA stream (CPU work is eager).

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L229-L232)

### particula.execution.prepared.PreparedSimulation.capture

*method*

```python
def capture() -> None
```

Record one CUDA timestep without executing it or advancing time.

This explicit setup boundary synchronizes prior work. Later step/run
calls replay the owned graph. Shape, process order, duration and buffer
identities stay fixed; forcing updates may change values in place.
Warp CPU rejects. Warp may instantiate its executable on first replay;
performance measurements should warm up that replay separately.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L597-L609)

### particula.execution.prepared.PreparedSimulation.checkpoint

*method*

```python
def checkpoint() -> SimulationCheckpoint
```

Observe and save exact continuation bytes plus frozen configuration.

Unlike a CPU snapshot, this in-memory checkpoint preserves random
streams, GPU-only fields and step counters. Resume on the same device
through the original runnable or an equivalently configured sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L579-L589)

### particula.execution.prepared.PreparedSimulation.close

*method*

```python
def close() -> None
```

Wait and release owned storage without downloading a final state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L617-L646)

### particula.execution.prepared.PreparedSimulation.coagulation_diagnostics

*method*

```python
def coagulation_diagnostics() -> dict[str, Any]
```

Observe compact work counts for the latest successful coagulation.

Counts describe this owner since preparation/restart; fresh owners
report zeros until a step runs. Capture recording does not advance them.
This observer does not download particle arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L421-L447)

### particula.execution.prepared.PreparedSimulation.cohort_diagnostics

*method*

```python
def cohort_diagnostics()
```

Observe detached per-box original-lineage outcome counts.

Counts partition initial seeds into reached, captured and unresolved.
Unresolved lineages are censored at the chosen observation end time.
These are lineage counts, not the number of surviving particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L560-L577)

### particula.execution.prepared.PreparedSimulation.diagnostics

*method*

```python
def diagnostics() -> SimulationDiagnostics
```

Observe detached box/species reductions without downloading slots.

**Returns**

- (`SimulationDiagnostics`) — SimulationDiagnostics with number, charge and mass concentrations,
- (`SimulationDiagnostics`) — active-slot counts, forcing, and optional last-step latent energy.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L403-L419)

### particula.execution.prepared.PreparedSimulation.ion_cleanup_diagnostics

*method*

```python
def ion_cleanup_diagnostics()
```

Observe cumulative exported particles and mass [kg, box by species].

Exported charge is exactly zero by construction. Returned arrays are
detached; this explicit observation synchronizes the prepared owner.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L541-L558)

### particula.execution.prepared.PreparedSimulation.ion_source_diagnostics

*method*

```python
def ion_source_diagnostics()
```

Return cumulative injected pairs and mass [kg] for each box.

This explicit observation synchronizes and returns detached arrays.
Mass is extensive, in the native particle species order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L518-L539)

### particula.execution.prepared.PreparedSimulation.run

*method*

```python
def run(steps: int) -> None
```

Enqueue a nonnegative integer number of fixed-duration steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L311-L320)

### particula.execution.prepared.PreparedSimulation.snapshot

*method*

```python
def snapshot() -> Aerosol
```

Download detached native CPU state, preserving names and measures.

Snapshot arrays never alias initial state, resident state, or earlier
snapshots, including when explicitly targeting Warp CPU. Derived vapor
pressure and random streams belong to execution/checkpoint state rather
than the CPU aerosol schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L367-L401)

### particula.execution.prepared.PreparedSimulation.step

*method*

```python
def step() -> None
```

Enqueue one frozen process sequence without a host observation.

A launch failure faults the binding. Earlier writes are not rolled back.
Asynchronous errors surface at a later synchronization or snapshot.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L283-L309)

### particula.execution.prepared.PreparedSimulation.synchronize

*method*

```python
def synchronize() -> None
```

Wait for pending work and surface asynchronous execution failures.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L322-L365)

### particula.execution.prepared.PreparedSimulation.uncapture

*method*

```python
def uncapture() -> None
```

Wait, release the graph and return to ordinary prepared execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L611-L615)

### particula.execution.prepared.PreparedSimulation.update_environment

*method*

```python
def update_environment(temperature=None, pressure=None, saturation_ratio=None) -> None
```

Replace specified environment fields between run segments.

Temperature [K] and pressure [Pa] accept scalars or per-box vectors.
Saturation accepts a scalar, full-gas vector or box-by-gas matrix; it
is externally prescribed when no condensation model is present. With
condensation, saturation is derived and cannot be prescribed here.
This explicit forcing boundary waits for prior work and its uploads.
Invalid values reject before any resident field is changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L456-L495)

### particula.execution.prepared.PreparedSimulation.update_gas

*method*

```python
def update_gas(concentration) -> None
```

Replace full-gas concentration [kg/m³] between run segments.

Accept a full-gas vector or box-by-gas matrix in ``gas_names`` order.
This explicit boundary waits for pending work, uploads owned values,
and refreshes derived saturation when condensation supplies its law.
It leaves particle state and process configuration unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L497-L516)

## particula.execution.prepared.prepare_simulation

*function*

```python
def prepare_simulation(processes: tuple[Any, ...], aerosol: Aerosol, *, backend: Backend = Backend.WARP, device: str = 'cuda:0', time_step: float, root_seed: int = 0, logical_box_ids: tuple[str, ...] | None = None, process_registry=None, coagulation_work_config=None, time: float = 0.0) -> PreparedSimulation
```

Validate the ordered process plan before uploading native state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared.py#L658-L749)
