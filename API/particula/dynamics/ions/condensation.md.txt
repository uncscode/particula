# particula.dynamics.ions.condensation

*module*

Explicit prepared condensation exclusion for reserved ion-only particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L1-L1)

## particula.dynamics.ions.condensation.IonCondensationExclusion

*class* · *dataclass*

```python
class IonCondensationExclusion(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Prevent vapor transfer to or from particles made only of ion materials.

Pair this configuration with prepared condensation. The two material lanes
must be reserved for nonvolatile ion mass. Aerosols containing any other
material remain eligible, including particles carrying attached ion mass.
Every condensation substep gates proposals before gas inventory limiting.
Native masses, represented counts and charges are never masked or replaced.

**Parameters**

- `positive_species` (`int`) — Reserved positive-ion material index.
- `negative_species` (`int`) — Reserved negative-ion material index.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L10-L49)

### particula.dynamics.ions.condensation.IonCondensationExclusion.negative_species

*attribute* · *instance attribute*

```python
negative_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L26-L26)

### particula.dynamics.ions.condensation.IonCondensationExclusion.positive_species

*attribute* · *instance attribute*

```python
positive_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L25-L25)

### particula.dynamics.ions.condensation.IonCondensationExclusion.__init__

*method*

```python
def __init__(positive_species: int, negative_species: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L0-L0)

### particula.dynamics.ions.condensation.IonCondensationExclusion._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Require resolved unit counts and distinct reserved ion lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L28-L43)

### particula.dynamics.ions.condensation.IonCondensationExclusion.execute

*method*

```python
def execute(*args, **kwargs)
```

Reject unsupported native CPU execution explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/condensation.py#L45-L49)

### particula.dynamics.ions.condensation.IonCondensationExclusion.__or__

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

### particula.dynamics.ions.condensation.IonCondensationExclusion._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L158)

### particula.dynamics.ions.condensation.IonCondensationExclusion._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L168-L170)

### particula.dynamics.ions.condensation.IonCondensationExclusion._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L152-L154)

### particula.dynamics.ions.condensation.IonCondensationExclusion._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L160-L162)

### particula.dynamics.ions.condensation.IonCondensationExclusion._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L164-L166)

### particula.dynamics.ions.condensation.IonCondensationExclusion._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L150)

### particula.dynamics.ions.condensation.IonCondensationExclusion.prepare

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

### particula.dynamics.ions.condensation.IonCondensationExclusion.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.ions.condensation.IonCondensationExclusion.resume

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
