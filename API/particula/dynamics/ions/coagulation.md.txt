# particula.dynamics.ions.coagulation

*module*

Prepared collision laws for explicit ions and charged aerosols.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L1-L1)

## particula.dynamics.ions.coagulation.NEGATIVE_ION_MASS

*attribute* · *module attribute*

```python
NEGATIVE_ION_MASS = 90 * ATOMIC_MASS_UNIT
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L108-L108)

## particula.dynamics.ions.coagulation.POSITIVE_ION_MASS

*attribute* · *module attribute*

```python
POSITIVE_ION_MASS = 150 * ATOMIC_MASS_UNIT
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L107-L107)

## particula.dynamics.ions.coagulation.AtmosphericCoagulation

*class* · *dataclass*

```python
class AtmosphericCoagulation(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Combine charged aerosol coagulation, ion attachment and recombination.

Prepared Warp execution only, for unit represented particles. Aerosol pairs
use ``charged_model``; default ion-aerosol rates use conductive-air LF2013
tables at 298.15 K and 101325 Pa, with 150 u positive and 90 u negative
ions. Opposite free ions recombine; same-sign ions do not cluster. Neutral
ion-only products are inert until an explicit ``IonNeutralCleanup``.
Reserved material lanes must contain only ion mass.

**Parameters**

- `positive_species` (`int`) — Positive-ion material index.
- `negative_species` (`int`) — Negative-ion material index.
- `charged_model` (`str`) (default: `'hard_sphere'`) — Existing charged Brownian model name.
- `omitted` (`str`) (default: `'reject'`) — ``reject`` (default), ``lower`` or ``upper`` for repulsive attachment rates omitted below the published 1e-15 m^3/s cutoff. Bounds are sensitivity assumptions, not extrapolated fit values.
- `recombination` (`float`) (default: `1.7e-12`) — Explicit opposite-ion recombination coefficient [m^3/s], held constant with T/P; not predicted by the attachment model.
- `attachment_model` (`str`) (default: `'lf2013'`) — ``lf2013`` or opt-in ``mobility`` Coulomb transport approximation, supporting variable T/P. No image forces or hydration.
- `positive_mobility` (`float`) (default: `0.00014`) — Illustrative positive-ion mobility at 298.15 K, 101325 Pa [m^2/(V s)]; supply a measured value for a specific ion.
- `negative_mobility` (`float`) (default: `0.00019`) — Corresponding negative-ion reference mobility.
- `ion_radius` (`float`) (default: `2.5e-10`) — Ion contact radius [m], independent of mobility transport.

> **References**
>
> López-Yglesias and Flagan (2013), Aerosol Science and Technology,
> 47:688–704, https://doi.org/10.1080/02786826.2013.783684.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L12-L104)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.attachment_model

*attribute* · *class attribute* · *instance attribute*

```python
attachment_model: str = 'lf2013'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L50-L50)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.charged_model

*attribute* · *class attribute* · *instance attribute*

```python
charged_model: str = 'hard_sphere'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L47-L47)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.ion_radius

*attribute* · *class attribute* · *instance attribute*

```python
ion_radius: float = 2.5e-10
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L53-L53)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.negative_mobility

*attribute* · *class attribute* · *instance attribute*

```python
negative_mobility: float = 0.00019
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L52-L52)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.negative_species

*attribute* · *instance attribute*

```python
negative_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L46-L46)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.omitted

*attribute* · *class attribute* · *instance attribute*

```python
omitted: str = 'reject'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L48-L48)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.positive_mobility

*attribute* · *class attribute* · *instance attribute*

```python
positive_mobility: float = 0.00014
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L51-L51)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.positive_species

*attribute* · *instance attribute*

```python
positive_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L45-L45)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.recombination

*attribute* · *class attribute* · *instance attribute*

```python
recombination: float = 1.7e-12
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L49-L49)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.__init__

*method*

```python
def __init__(positive_species: int, negative_species: int, charged_model: str = 'hard_sphere', omitted: str = 'reject', recombination: float = 1.7e-12, attachment_model: str = 'lf2013', positive_mobility: float = 0.00014, negative_mobility: float = 0.00019, ion_radius: float = 2.5e-10) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L0-L0)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Reject unsupported model and physical state before GPU upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L55-L98)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.execute

*method*

```python
def execute(*args, **kwargs)
```

Reject native CPU execution without mutating the aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/coagulation.py#L100-L104)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.__or__

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

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.prepare

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

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.ions.coagulation.AtmosphericCoagulation.resume

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
