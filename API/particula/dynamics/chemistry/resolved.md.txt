# particula.dynamics.chemistry.resolved

*module*

Closed gas, bulk particle and collision-limited uptake in resolved boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L1-L1)

## particula.dynamics.chemistry.resolved.ResolvedChemistry

*class*

```python
class ResolvedChemistry(BoxProcess)
```

Bases: `particula.box_process.BoxProcess`

Couple a version-2 mechanism across gas and resolved particle masses.

Bulk rates use mol/m³ of instantaneous particle material. Uptake uses
a spherical collision area, prescribed net uptake probability and mean
molecular speed. RK23 absolute tolerances use mol/m³ of air for gas
and mol/m³ of the initial particle volume for each process interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L119-L194)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'resolved_chemistry'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L128-L128)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.bound

*attribute* · *instance attribute*

```python
bound = bound
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L146-L146)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.solver

*attribute* · *instance attribute*

```python
solver = solver
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L147-L147)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.extension_id

*attribute* · *instance attribute*

```python
extension_id = extension_id or name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L100-L100)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.name

*attribute* · *instance attribute*

```python
name = name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L95-L95)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.parameters

*attribute* · *instance attribute*

```python
parameters = parameters
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L105-L105)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.reads

*attribute* · *instance attribute*

```python
reads = tuple(reads)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L96-L96)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.representations

*attribute* · *instance attribute*

```python
representations = tuple(representations)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L99-L99)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.stochastic

*attribute* · *instance attribute*

```python
stochastic = stochastic
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L98-L98)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.version

*attribute* · *instance attribute*

```python
version = version
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L106-L106)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.writes

*attribute* · *instance attribute*

```python
writes = tuple(writes)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L97-L97)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.__init__

*method*

```python
def __init__(bound: ResolvedBinding, solver=_DEFAULT_SOLVER, *, name='resolved_chemistry')
```

Select the immutable layout and per-box adaptive controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L130-L164)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.advance

*method*

```python
def advance(aerosol, *, time, time_step, context)
```

Integrate one simultaneous closed material balance within a box.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L184-L194)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.configuration

*method*

```python
def configuration()
```

Return restart identity including densities, mappings and solver.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L166-L178)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.validate

*method*

```python
def validate(aerosol)
```

Validate current representation, layout and inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L180-L182)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Inherited from: `particula.box_process.BoxProcess`

Publish a successful process substep atomically across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L115-L186)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.__or__

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

### particula.dynamics.chemistry.resolved.ResolvedChemistry._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.dynamics.chemistry.resolved.ResolvedChemistry._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.prepare

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

### particula.dynamics.chemistry.resolved.ResolvedChemistry.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.chemistry.resolved.ResolvedChemistry.resume

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

## particula.dynamics.chemistry.resolved._BoxKinetics

*class*

```python
class _BoxKinetics
```

Scale each particle's integrated moles by its initial volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L24-L116)

### particula.dynamics.chemistry.resolved._BoxKinetics.active

*attribute* · *instance attribute*

```python
active = particles.concentration[0] > 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L30-L30)

### particula.dynamics.chemistry.resolved._BoxKinetics.gm

*attribute* · *instance attribute*

```python
gm = np.asarray(bound.gas_masses)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L34-L34)

### particula.dynamics.chemistry.resolved._BoxKinetics.initial

*attribute* · *instance attribute*

```python
initial = np.concatenate((aerosol.gas.concentration[0] / self.gm, masses / self.pm / reference[:, None].ravel()))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L38-L43)

### particula.dynamics.chemistry.resolved._BoxKinetics.k

*attribute* · *instance attribute*

```python
k = a * temp / t_ref ** exponent * np.exp(-energy / GAS_CONSTANT * temp)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L46-L50)

### particula.dynamics.chemistry.resolved._BoxKinetics.kinds

*attribute* · *instance attribute*

```python
kinds = bound.kinds
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L53-L53)

### particula.dynamics.chemistry.resolved._BoxKinetics.net

*attribute* · *instance attribute*

```python
net = np.asarray(bound.net)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L52-L52)

### particula.dynamics.chemistry.resolved._BoxKinetics.ng

*attribute* · *instance attribute*

```python
ng = self.gm.size
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L35-L35)

### particula.dynamics.chemistry.resolved._BoxKinetics.orders

*attribute* · *instance attribute*

```python
orders = np.asarray(bound.orders)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L51-L51)

### particula.dynamics.chemistry.resolved._BoxKinetics.partial_volume

*attribute* · *instance attribute*

```python
partial_volume = self.pm / particles.density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L36-L36)

### particula.dynamics.chemistry.resolved._BoxKinetics.pm

*attribute* · *instance attribute*

```python
pm = np.asarray(bound.particle_masses)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L33-L33)

### particula.dynamics.chemistry.resolved._BoxKinetics.reference

*attribute* · *instance attribute*

```python
reference = masses / particles.density.sum(axis=1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L32-L32)

### particula.dynamics.chemistry.resolved._BoxKinetics.sources

*attribute* · *instance attribute*

```python
sources = bound.sources
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L54-L54)

### particula.dynamics.chemistry.resolved._BoxKinetics.speed

*attribute* · *instance attribute*

```python
speed = np.sqrt(8 * GAS_CONSTANT * temp / np.pi * self.gm)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L55-L55)

### particula.dynamics.chemistry.resolved._BoxKinetics.weights

*attribute* · *instance attribute*

```python
weights = particles.concentration[0] / particles.volume[0]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L31-L31)

### particula.dynamics.chemistry.resolved._BoxKinetics.__init__

*method*

```python
def __init__(bound, aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L27-L57)

### particula.dynamics.chemistry.resolved._BoxKinetics.admissible

*method*

```python
def admissible(state)
```

Reject a negative concentration or an emptied active slot.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L59-L66)

### particula.dynamics.chemistry.resolved._BoxKinetics.publish

*method*

```python
def publish(aerosol, final)
```

Publish changed lanes and preserve inactive and inert values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L105-L116)

### particula.dynamics.chemistry.resolved._BoxKinetics.rate

*method*

```python
def rate(time, state)
```

Return gas mol/m³-air/s and material mol/m³-initial-particle/s.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved.py#L68-L103)
