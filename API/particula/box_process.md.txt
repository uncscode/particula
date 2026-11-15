# particula.box_process

*module*

Public, staged single-box Python process extensions.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L1-L1)

## particula.box_process._WRITABLE

*attribute* · *module attribute*

```python
_WRITABLE = frozenset({'particles.masses', 'particles.concentration', 'particles.charge', 'particles.volume', 'gas.concentration', 'environment.temperature', 'environment.pressure'})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L15-L25)

## particula.box_process.BoxContext

*class* · *dataclass*

```python
class BoxContext
```

Logical box identity and optional staged random generator.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L37-L42)

### particula.box_process.BoxContext.box_id

*attribute* · *instance attribute*

```python
box_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L41-L41)

### particula.box_process.BoxContext.rng

*attribute* · *class attribute* · *instance attribute*

```python
rng: np.random.Generator | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L42-L42)

### particula.box_process.BoxContext.__init__

*method*

```python
def __init__(box_id: str, rng: np.random.Generator | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L0-L0)

## particula.box_process.BoxProcess

*class*

```python
class BoxProcess(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Implement validate and advance to compose a custom Python process.

Advances receive ordinary native containers with a leading box axis.
Only declared writes are published after the complete batch succeeds.
External side effects and hidden callback state cannot be rolled back.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L45-L186)

### particula.box_process.BoxProcess.extension_id

*attribute* · *instance attribute*

```python
extension_id = extension_id or name
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L100-L100)

### particula.box_process.BoxProcess.name

*attribute* · *instance attribute*

```python
name = name
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L95-L95)

### particula.box_process.BoxProcess.parameters

*attribute* · *instance attribute*

```python
parameters = parameters
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L105-L105)

### particula.box_process.BoxProcess.reads

*attribute* · *instance attribute*

```python
reads = tuple(reads)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L96-L96)

### particula.box_process.BoxProcess.representations

*attribute* · *instance attribute*

```python
representations = tuple(representations)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L99-L99)

### particula.box_process.BoxProcess.stochastic

*attribute* · *instance attribute*

```python
stochastic = stochastic
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L98-L98)

### particula.box_process.BoxProcess.version

*attribute* · *instance attribute*

```python
version = version
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L106-L106)

### particula.box_process.BoxProcess.writes

*attribute* · *instance attribute*

```python
writes = tuple(writes)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L97-L97)

### particula.box_process.BoxProcess.__init__

*method*

```python
def __init__(name, reads=(), writes=(), stochastic=False, representations=('discrete', 'continuous_pdf', 'particle_resolved'), extension_id=None, version='1', parameters=())
```

Declare instance identity, field access and scientific support.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L53-L106)

### particula.box_process.BoxProcess.advance

*method*

```python
def advance(aerosol, *, time, time_step, context)
```

Modify declared arrays in one box over the supplied interval [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L111-L113)

### particula.box_process.BoxProcess.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Publish a successful process substep atomically across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L115-L186)

### particula.box_process.BoxProcess.validate

*method*

```python
def validate(aerosol)
```

Check process requirements without modifying the supplied state.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L108-L109)

### particula.box_process.BoxProcess.__or__

*method*

```python
def __or__(other: RunnableABC) -> RunnableSequence
```

Inherited from: `particula.runnable.RunnableABC`

Chain this Runnable with another using the '|' operator.

This method enables an easy way to sequence processes, so they
can be executed in a defined order.

**Parameters**

- `other` (`RunnableABC`) — Another Runnable to append after this one.

**Returns**

- (`RunnableSequence`) — A sequence containing both processes.

**Examples**

```py title="Chaining two processes"
combined_process = process1 | process2
final_aerosol = combined_process.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L262-L283)

### particula.box_process.BoxProcess._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.box_process.BoxProcess._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.box_process.BoxProcess._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.box_process.BoxProcess._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.box_process.BoxProcess._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.box_process.BoxProcess._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.box_process.BoxProcess._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.box_process.BoxProcess._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.box_process.BoxProcess.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None, process_registry=None, time: float = 0.0) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Upload native state once and prepare supported resident GPU physics.

**Parameters**

- `aerosol` (`Aerosol`) — Initial native particle, gas and environment state.
- `time_step` (`float`) — Fixed duration of each complete process cycle [s].
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default; CPU execution uses execute().
- `device` (`str`) (default: `'cuda:0'`) — Explicit Warp device, default CUDA device zero.
- `root_seed` (`int`) (default: `0`) — Root seed for persistent independent process/box streams.
- `logical_box_ids` (default: `None`) — Optional stable identities in physical box order.
- `process_registry` (default: `None`) — Exact-type custom GPU adapter registrations.
- `time` (`float`) (default: `0.0`) — Absolute initial simulation time [s].

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L179-L224)

### particula.box_process.BoxProcess.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.box_process.BoxProcess.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None, process_registry=None) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.
- `process_registry` (default: `None`) — Compatible caller-supplied GPU implementations.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L226-L260)

## particula.box_process.FunctionProcess

*class*

```python
class FunctionProcess(BoxProcess)
```

Bases: `particula.box_process.BoxProcess`

Adapt a Python function to the public box-process contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L244-L266)

### particula.box_process.FunctionProcess.function

*attribute* · *instance attribute*

```python
function = function
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L254-L254)

### particula.box_process.FunctionProcess.validator

*attribute* · *instance attribute*

```python
validator = validate
```

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L255-L255)

### particula.box_process.FunctionProcess.extension_id

*attribute* · *instance attribute*

```python
extension_id = extension_id or name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L100-L100)

### particula.box_process.FunctionProcess.name

*attribute* · *instance attribute*

```python
name = name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L95-L95)

### particula.box_process.FunctionProcess.parameters

*attribute* · *instance attribute*

```python
parameters = parameters
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L105-L105)

### particula.box_process.FunctionProcess.reads

*attribute* · *instance attribute*

```python
reads = tuple(reads)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L96-L96)

### particula.box_process.FunctionProcess.representations

*attribute* · *instance attribute*

```python
representations = tuple(representations)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L99-L99)

### particula.box_process.FunctionProcess.stochastic

*attribute* · *instance attribute*

```python
stochastic = stochastic
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L98-L98)

### particula.box_process.FunctionProcess.version

*attribute* · *instance attribute*

```python
version = version
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L106-L106)

### particula.box_process.FunctionProcess.writes

*attribute* · *instance attribute*

```python
writes = tuple(writes)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L97-L97)

### particula.box_process.FunctionProcess.__init__

*method*

```python
def __init__(function, *, validate=None, **kwargs)
```

Retain advancement and optional validation callables.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L247-L255)

### particula.box_process.FunctionProcess.advance

*method*

```python
def advance(aerosol, *, time, time_step, context)
```

Delegate one box update without hiding callback exceptions.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L262-L266)

### particula.box_process.FunctionProcess.validate

*method*

```python
def validate(aerosol)
```

Invoke the optional requirement validator.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L257-L260)

### particula.box_process.FunctionProcess.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Inherited from: `particula.box_process.BoxProcess`

Publish a successful process substep atomically across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L115-L186)

### particula.box_process.FunctionProcess.__or__

*method*

```python
def __or__(other: RunnableABC) -> RunnableSequence
```

Inherited from: `particula.runnable.RunnableABC`

Chain this Runnable with another using the '|' operator.

This method enables an easy way to sequence processes, so they
can be executed in a defined order.

**Parameters**

- `other` (`RunnableABC`) — Another Runnable to append after this one.

**Returns**

- (`RunnableSequence`) — A sequence containing both processes.

**Examples**

```py title="Chaining two processes"
combined_process = process1 | process2
final_aerosol = combined_process.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L262-L283)

### particula.box_process.FunctionProcess._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.box_process.FunctionProcess._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.box_process.FunctionProcess._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.box_process.FunctionProcess._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.box_process.FunctionProcess._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.box_process.FunctionProcess._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.box_process.FunctionProcess._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.box_process.FunctionProcess._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.box_process.FunctionProcess.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None, process_registry=None, time: float = 0.0) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Upload native state once and prepare supported resident GPU physics.

**Parameters**

- `aerosol` (`Aerosol`) — Initial native particle, gas and environment state.
- `time_step` (`float`) — Fixed duration of each complete process cycle [s].
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default; CPU execution uses execute().
- `device` (`str`) (default: `'cuda:0'`) — Explicit Warp device, default CUDA device zero.
- `root_seed` (`int`) (default: `0`) — Root seed for persistent independent process/box streams.
- `logical_box_ids` (default: `None`) — Optional stable identities in physical box order.
- `process_registry` (default: `None`) — Exact-type custom GPU adapter registrations.
- `time` (`float`) (default: `0.0`) — Absolute initial simulation time [s].

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L179-L224)

### particula.box_process.FunctionProcess.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.box_process.FunctionProcess.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None, process_registry=None) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.
- `process_registry` (default: `None`) — Compatible caller-supplied GPU implementations.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L226-L260)

## particula.box_process._check_candidate

*function*

```python
def _check_candidate(candidate, before, retained, writes, containers)
```

Reject replacement, undeclared writes and invalid native state.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L189-L241)

## particula.box_process.validate_time

*function*

```python
def validate_time(time)
```

Validate absolute simulation time in seconds without advancing it.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L28-L34)
