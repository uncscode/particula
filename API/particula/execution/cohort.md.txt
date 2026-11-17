# particula.execution.cohort

*module*

Opt-in original-seed lineage observations for prepared resolved studies.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L1-L1)

## particula.execution.cohort.SeedCohort

*class* · *dataclass*

```python
class SeedCohort(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Track original seed survival to a diameter before background capture.

Initially, each unit-count particle containing seed material contributes
one lineage. Particles containing background material carry a persistent
background label. Seed-seed mergers add unresolved lineages; any background
merger captures unresolved lineages before evaluating the resulting size.
Successful lineages are counted once and remain successful thereafter.

This prepared-only observer runs after each physical process, independently
of its position in the composition. Events follow the configured process
order; substep crossing times are not interpolated. Unresolved lineages are
right-censored when the user ends the observation.

**Parameters**

- `seed_species` (`int`) — Initial seed-marker material index.
- `background_species` (`int`) — Initial background-marker material index.
- `target_diameter` (`float`) (default: `5e-08`) — First-passage diameter [m].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L10-L60)

### particula.execution.cohort.SeedCohort.background_species

*attribute* · *instance attribute*

```python
background_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L32-L32)

### particula.execution.cohort.SeedCohort.seed_species

*attribute* · *instance attribute*

```python
seed_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L31-L31)

### particula.execution.cohort.SeedCohort.target_diameter

*attribute* · *class attribute* · *instance attribute*

```python
target_diameter: float = 5e-08
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L33-L33)

### particula.execution.cohort.SeedCohort.__init__

*method*

```python
def __init__(seed_species: int, background_species: int, target_diameter: float = 5e-08) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L0-L0)

### particula.execution.cohort.SeedCohort._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Require an explicit material map and resolved unit counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L35-L56)

### particula.execution.cohort.SeedCohort.execute

*method*

```python
def execute(*args, **kwargs)
```

Reject unsupported native CPU execution explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cohort.py#L58-L60)

### particula.execution.cohort.SeedCohort.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L265-L286)

### particula.execution.cohort.SeedCohort._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.execution.cohort.SeedCohort._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.execution.cohort.SeedCohort._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.execution.cohort.SeedCohort._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.execution.cohort.SeedCohort._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.execution.cohort.SeedCohort._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.execution.cohort.SeedCohort._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.execution.cohort.SeedCohort.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None, process_registry=None, coagulation_work_config=None, time: float = 0.0) -> PreparedSimulation
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
- `time` (`float`) (default: `0.0`) — Absolute initial simulation time [s].

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L179-L227)

### particula.execution.cohort.SeedCohort.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.execution.cohort.SeedCohort.resume

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L229-L263)
