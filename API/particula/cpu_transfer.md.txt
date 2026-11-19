# particula.cpu_transfer

*module*

Explicit host execution of a box process inside a prepared GPU sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L1-L1)

## particula.cpu_transfer.CPUTransfer

*class*

```python
class CPUTransfer(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Opt into a validated host round trip for a public BoxProcess.

Python callbacks may call arbitrary scientific libraries. Their declared
arrays publish only after all boxes validate. Prepared sequences containing
this wrapper cannot use CUDA capture. External callback side effects and
hidden mutable closure state are not part of checkpoint continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L9-L30)

### particula.cpu_transfer.CPUTransfer.process

*attribute* · *instance attribute*

```python
process = process
```

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L22-L22)

### particula.cpu_transfer.CPUTransfer.__init__

*method*

```python
def __init__(process: BoxProcess)
```

Retain the explicitly selected CPU process until preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L18-L22)

### particula.cpu_transfer.CPUTransfer.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Delegate ordinary CPU execution without introducing any transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L24-L30)

### particula.cpu_transfer.CPUTransfer.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L272-L293)

### particula.cpu_transfer.CPUTransfer._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.cpu_transfer.CPUTransfer._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.cpu_transfer.CPUTransfer._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.cpu_transfer.CPUTransfer._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.cpu_transfer.CPUTransfer._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.cpu_transfer.CPUTransfer._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.cpu_transfer.CPUTransfer._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.cpu_transfer.CPUTransfer._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.cpu_transfer.CPUTransfer.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None, process_registry=None, coagulation_work_config=None, coagulation_subcycling=None, time: float = 0.0) -> PreparedSimulation
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
- `coagulation_work_config` (default: `None`) — Optional coagulation resource budgets.
- `coagulation_subcycling` (default: `None`) — Optional bounded adaptive policy.
- `time` (`float`) (default: `0.0`) — Absolute initial simulation time [s].

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L179-L230)

### particula.cpu_transfer.CPUTransfer.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.cpu_transfer.CPUTransfer.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None, process_registry=None, retry_subcycling=None) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.
- `process_registry` (default: `None`) — Compatible caller-supplied GPU implementations.
- `retry_subcycling` (default: `None`) — Explicitly relax a saved adaptive coagulation policy while retaining the outer timestep and random streams.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L232-L270)

## particula.cpu_transfer._CPUTransferAdapter

*class*

```python
class _CPUTransferAdapter
```

Freeze an explicitly wrapped CPU implementation at preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L33-L64)

### particula.cpu_transfer._CPUTransferAdapter.plan

*method*

```python
def plan(wrapper, aerosol)
```

Check all callback requirements without advancing or drawing RNG.

[View source](https://github.com/uncscode/particula/blob/main/particula/cpu_transfer.py#L36-L64)
