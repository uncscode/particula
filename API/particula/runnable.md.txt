# particula.runnable

*module*

Runnable process utilities for modifying Aerosol instances.

This module provides abstract and concrete classes for processes
that can modify or transform an Aerosol object over a simulation
time step.

**Classes**

- `- Runnable` — Abstract base class requiring 'rate' and 'execute' methods.
- `- RunnableSequence` — Manages a sequence of Runnable processes.

**Examples**

```py title="Chaining processes"
process1 = SomeRunnableProcess()
process2 = AnotherRunnableProcess()
chained = process1 | process2
final_aerosol = chained.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L1-L1)

## particula.runnable.AVAILABLE_TQDM

*attribute* · *module attribute*

```python
AVAILABLE_TQDM = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L32-L32)

## particula.runnable.RunnableABC

*class*

```python
class RunnableABC
```

Abstract base class for processes modifying an Aerosol instance.

The default drivers stage and execute single-box hooks across a batch.
Implement ``_rate_box``, ``_execute_box``, and ``_writer_fields`` to use
this transaction boundary. Existing subclasses overriding public methods
retain their own execution and validation responsibilities.

**Functions**

- `- rate` — Calculate and return the rate of the process.
- `- execute` — Apply the process logic to the Aerosol over a specified
time.
- `- __or__` — Chain two processes using the '|' operator.

**Examples**

```py title="Defining a Custom Process"
class CustomProcess(Runnable):
    def rate(self, aerosol):
        return 42

    def execute(self, aerosol, time_step, sub_steps=1):
        # Modify aerosol here
        return aerosol
```

> **References**
>
> - No references available yet.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L43-L267)

### particula.runnable.RunnableABC.__or__

*method*

```python
def __or__(other: RunnableABC) -> RunnableSequence
```

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L246-L267)

### particula.runnable.RunnableABC._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L158)

### particula.runnable.RunnableABC._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L168-L170)

### particula.runnable.RunnableABC._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L152-L154)

### particula.runnable.RunnableABC._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L160-L162)

### particula.runnable.RunnableABC._rate_box

*method*

```python
def _rate_box(aerosol)
```

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L164-L166)

### particula.runnable.RunnableABC._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L145-L146)

### particula.runnable.RunnableABC._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L150)

### particula.runnable.RunnableABC.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L89-L143)

### particula.runnable.RunnableABC.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None) -> PreparedSimulation
```

Upload native state once and prepare supported resident GPU physics.

**Parameters**

- `aerosol` (`Aerosol`) — Initial native particle, gas and environment state.
- `time_step` (`float`) — Fixed duration of each complete process cycle [s].
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default; CPU execution uses execute().
- `device` (`str`) (default: `'cuda:0'`) — Explicit Warp device, default CUDA device zero.
- `root_seed` (`int`) (default: `0`) — Root seed for persistent independent process/box streams.
- `logical_box_ids` (default: `None`) — Optional stable identities in physical box order.

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L172-L211)

### particula.runnable.RunnableABC.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.runnable.RunnableABC.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None) -> PreparedSimulation
```

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L213-L244)

## particula.runnable.RunnableSequence

*class*

```python
class RunnableSequence
```

A sequence of Runnable processes executed in order.

This class maintains a list of processes to be applied sequentially
to an Aerosol. Each process modifies the Aerosol and passes it along
to the next in the sequence.

**Attributes**

- `-` (`processes`) — A list of Runnable objects forming the sequence.

**Functions**

- `- add_process` — Add a Runnable to the sequence.
- `- execute` — Apply each Runnable in the sequence to an Aerosol.
- `- __or__` — Chain a new Runnable into this sequence.

**Examples**

```py title="Building and running a RunnableSequence"
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
sequence.add_process(AnotherProcess())
final_aerosol = sequence.execute(aerosol, time_step=2.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L270-L460)

### particula.runnable.RunnableSequence.processes

*attribute* · *instance attribute*

```python
processes: list[RunnableABC] = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L299-L299)

### particula.runnable.RunnableSequence.__init__

*method*

```python
def __init__() -> None
```

Initialize the RunnableSequence.

Sets up an empty list to hold the sequence of Runnable processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L294-L299)

### particula.runnable.RunnableSequence.__or__

*method*

```python
def __or__(process: RunnableABC) -> RunnableSequence
```

Chain another Runnable into this sequence using the '|' operator.

**Parameters**

- `process` (`RunnableABC`) — The Runnable to add.

**Returns**

- (`RunnableSequence`) — This sequence with the new Runnable appended.

**Examples**

```py
sequence = RunnableSequence()
sequence |= CustomProcess()
# or
sequence = sequence | AnotherProcess()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L442-L460)

### particula.runnable.RunnableSequence.add_process

*method*

```python
def add_process(process: RunnableABC) -> None
```

Add a Runnable to the sequence.

**Parameters**

- `process` (`RunnableABC`) — The Runnable to add.

**Examples**

```py
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L301-L313)

### particula.runnable.RunnableSequence.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Execute all processes in the sequence on the given Aerosol.

Each Runnable in the sequence modifies the Aerosol and passes
it to the next Runnable until all have been executed. A full cycle is
performed over each sub-step of the time step.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance to be updated.
- `time_step` (`float`) — The time step size in seconds for each process.
- `sub_steps` (`int`) (default: `1`) — Number of sub-steps to subdivide the time step, default 1.

**Returns**

- (`Aerosol`) — - Aerosol : The resulting Aerosol after all processes run.

**Examples**

```py title="Executing a RunnableSequence"
sequence = RunnableSequence()
final_aerosol = sequence.execute(
    aerosol, time_step=1.0, sub_steps=4
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L315-L366)

### particula.runnable.RunnableSequence.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None) -> PreparedSimulation
```

Upload native state once and prepare supported resident GPU physics.

**Parameters**

- `aerosol` (`Aerosol`) — Initial native particle, gas and environment state.
- `time_step` (`float`) — Fixed duration of each complete process cycle [s].
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default; CPU execution uses execute().
- `device` (`str`) (default: `'cuda:0'`) — Explicit Warp device, default CUDA device zero.
- `root_seed` (`int`) (default: `0`) — Root seed for persistent independent process/box streams.
- `logical_box_ids` (default: `None`) — Optional stable identities in physical box order.

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L368-L407)

### particula.runnable.RunnableSequence.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None) -> PreparedSimulation
```

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L409-L440)
