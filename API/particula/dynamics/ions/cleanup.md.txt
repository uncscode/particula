# particula.dynamics.ions.cleanup

*module*

Export neutral products made only from reserved ion materials.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L1-L1)

## particula.dynamics.ions.cleanup.IonNeutralCleanup

*class* · *dataclass*

```python
class IonNeutralCleanup(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Export neutral ion-only products and reclaim their resolved slots.

The two material lanes must be reserved for ions. A neutral particle with
any mass in another lane is an aerosol and is never removed. Both reserved
lanes must contain positive mass. This operation is a modeled export of
recombination products, not an additional recombination reaction.

Prepared execution only; place this process at the desired refresh point
in the ordered runnable composition. Export occurs after every
``interval_steps`` calls; particles remain resident between exports.

**Parameters**

- `positive_species` (`int`) — Reserved positive-ion material index.
- `negative_species` (`int`) — Reserved negative-ion material index.
- `interval_steps` (`int`) (default: `1`) — Positive integer number of prepared steps per export. Default 1 exports every step; the first export follows one full interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L10-L58)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.interval_steps

*attribute* · *class attribute* · *instance attribute*

```python
interval_steps: int = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L33-L33)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.negative_species

*attribute* · *instance attribute*

```python
negative_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L32-L32)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.positive_species

*attribute* · *instance attribute*

```python
positive_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L31-L31)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.__init__

*method*

```python
def __init__(positive_species: int, negative_species: int, interval_steps: int = 1) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L0-L0)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Reject incompatible material maps and represented populations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L35-L52)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.execute

*method*

```python
def execute(*args, **kwargs)
```

Reject unsupported native CPU execution explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/cleanup.py#L54-L58)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L246-L267)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L158)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L168-L170)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L152-L154)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L160-L162)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L164-L166)

### particula.dynamics.ions.cleanup.IonNeutralCleanup._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L150)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None) -> PreparedSimulation
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

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L172-L211)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.ions.cleanup.IonNeutralCleanup.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

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
