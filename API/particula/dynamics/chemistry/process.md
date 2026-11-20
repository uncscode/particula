# particula.dynamics.chemistry.process

*module*

Adaptive native gas chemistry over a prescribed-temperature interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L1-L1)

## particula.dynamics.chemistry.process._DEFAULT_SOLVER

*attribute* · *module attribute*

```python
_DEFAULT_SOLVER = ChemistrySolver()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L59-L59)

## particula.dynamics.chemistry.process.ChemistrySolver

*class* · *dataclass*

```python
class ChemistrySolver
```

Per-interval adaptive controls shared by NumPy and resident execution.

Absolute tolerance is in mol/m³ of air. A tuple follows normalized
mechanism species order. The trial budget counts all accepted and
rejected trials in one box during one process interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L14-L56)

### particula.dynamics.chemistry.process.ChemistrySolver.atol

*attribute* · *class attribute* · *instance attribute*

```python
atol: float | tuple[float, ...] = 1e-14
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L25-L25)

### particula.dynamics.chemistry.process.ChemistrySolver.first_step

*attribute* · *class attribute* · *instance attribute*

```python
first_step: float = 0.001
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L26-L26)

### particula.dynamics.chemistry.process.ChemistrySolver.max_step

*attribute* · *class attribute* · *instance attribute*

```python
max_step: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L28-L28)

### particula.dynamics.chemistry.process.ChemistrySolver.max_trials

*attribute* · *class attribute* · *instance attribute*

```python
max_trials: int = 100000
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L29-L29)

### particula.dynamics.chemistry.process.ChemistrySolver.method

*attribute* · *class attribute* · *instance attribute*

```python
method: str = 'rk23'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L23-L23)

### particula.dynamics.chemistry.process.ChemistrySolver.min_step

*attribute* · *class attribute* · *instance attribute*

```python
min_step: float = 1e-14
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L27-L27)

### particula.dynamics.chemistry.process.ChemistrySolver.rtol

*attribute* · *class attribute* · *instance attribute*

```python
rtol: float = 1e-07
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L24-L24)

### particula.dynamics.chemistry.process.ChemistrySolver.__init__

*method*

```python
def __init__(method: str = 'rk23', rtol: float = 1e-07, atol: float | tuple[float, ...] = 1e-14, first_step: float = 0.001, min_step: float = 1e-14, max_step: float = 1.0, max_trials: int = 100000) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L0-L0)

### particula.dynamics.chemistry.process.ChemistrySolver.__post_init__

*method*

```python
def __post_init__()
```

Validate tolerances and positive finite timestep limits.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L31-L56)

## particula.dynamics.chemistry.process.GasChemistry

*class*

```python
class GasChemistry(BoxProcess)
```

Bases: `particula.box_process.BoxProcess`

Advance bound gas concentrations in kg/m³ with staged box updates.

Temperature is prescribed and constant within each process interval.
Each interval starts a fresh controller, so only accepted concentrations
and the simulation clock carry forward to the next process boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L88-L177)

### particula.dynamics.chemistry.process.GasChemistry._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'gas_chemistry'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L96-L96)

### particula.dynamics.chemistry.process.GasChemistry.bound

*attribute* · *instance attribute*

```python
bound = bound
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L114-L114)

### particula.dynamics.chemistry.process.GasChemistry.solver

*attribute* · *instance attribute*

```python
solver = solver
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L115-L115)

### particula.dynamics.chemistry.process.GasChemistry.extension_id

*attribute* · *instance attribute*

```python
extension_id = extension_id or name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L100-L100)

### particula.dynamics.chemistry.process.GasChemistry.name

*attribute* · *instance attribute*

```python
name = name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L95-L95)

### particula.dynamics.chemistry.process.GasChemistry.parameters

*attribute* · *instance attribute*

```python
parameters = parameters
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L105-L105)

### particula.dynamics.chemistry.process.GasChemistry.reads

*attribute* · *instance attribute*

```python
reads = tuple(reads)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L96-L96)

### particula.dynamics.chemistry.process.GasChemistry.representations

*attribute* · *instance attribute*

```python
representations = tuple(representations)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L99-L99)

### particula.dynamics.chemistry.process.GasChemistry.stochastic

*attribute* · *instance attribute*

```python
stochastic = stochastic
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L98-L98)

### particula.dynamics.chemistry.process.GasChemistry.version

*attribute* · *instance attribute*

```python
version = version
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L106-L106)

### particula.dynamics.chemistry.process.GasChemistry.writes

*attribute* · *instance attribute*

```python
writes = tuple(writes)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L97-L97)

### particula.dynamics.chemistry.process.GasChemistry.__init__

*method*

```python
def __init__(bound: BoundMechanism, solver: ChemistrySolver = _DEFAULT_SOLVER, *, name='chemistry')
```

Retain the immutable mechanism, gas layout and solver controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L98-L122)

### particula.dynamics.chemistry.process.GasChemistry.advance

*method*

```python
def advance(aerosol, *, time, time_step, context)
```

Convert bound columns to molar units and publish accepted mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L164-L177)

### particula.dynamics.chemistry.process.GasChemistry.configuration

*method*

```python
def configuration()
```

Return native mechanism, binding and solver identity for restart.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L124-L136)

### particula.dynamics.chemistry.process.GasChemistry.integrate

*method*

```python
def integrate(state, temperature, duration)
```

Advance mol/m³ for one interval and return solver statistics.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L150-L162)

### particula.dynamics.chemistry.process.GasChemistry.validate

*method*

```python
def validate(aerosol)
```

Check bound layout, temperature and participating gas inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L138-L148)

### particula.dynamics.chemistry.process.GasChemistry.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Inherited from: `particula.box_process.BoxProcess`

Publish a successful process substep atomically across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L115-L186)

### particula.dynamics.chemistry.process.GasChemistry.__or__

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

### particula.dynamics.chemistry.process.GasChemistry._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.dynamics.chemistry.process.GasChemistry._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.dynamics.chemistry.process.GasChemistry._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.dynamics.chemistry.process.GasChemistry._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.dynamics.chemistry.process.GasChemistry._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.dynamics.chemistry.process.GasChemistry._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.dynamics.chemistry.process.GasChemistry._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.dynamics.chemistry.process.GasChemistry._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.dynamics.chemistry.process.GasChemistry.prepare

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

### particula.dynamics.chemistry.process.GasChemistry.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.chemistry.process.GasChemistry.resume

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

## particula.dynamics.chemistry.process._integrate_problem

*function*

```python
def _integrate_problem(problem, state, duration, settings, label)
```

Apply one trial budget across the full interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/process.py#L62-L85)
