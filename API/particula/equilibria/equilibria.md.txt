# particula.equilibria.equilibria

*module*

Runnable wrapper for equilibria strategies.

Provides the ``Equilibria`` runnable to compose thermodynamic equilibrium
solvers with other runnables using the shared ``RunnableABC`` interface. The
runnable mirrors the sub-step semantics used by dynamics processes (e.g.,
condensation, coagulation, wall loss) while delegating equilibrium resolution
to a configured ``EquilibriaStrategy``.

**Examples**

Basic usage with a strategy::

    >>> import numpy as np
    >>> from particula.equilibria import (
    ...     Equilibria,
    ...     LiquidVaporPartitioningStrategy,
    ... )
    >>> strategy = LiquidVaporPartitioningStrategy(water_activity=0.75)
    >>> runnable = Equilibria(strategy=strategy)
    >>> aerosol = runnable.execute(aerosol, time_step=1.0)

Composing with other runnables::

    >>> pipeline = runnable | other_runnable
    >>> aerosol = pipeline.execute(aerosol, time_step=1.0, sub_steps=4)

> **Note**
>
> Equilibria solves are typically instantaneous, but the runnable still uses
> sub-step semantics to align with ``RunnableSequence`` composition. A future
> TODO is richer aerosol data mapping once broader data flow is finalized.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L1-L1)

## particula.equilibria.equilibria.Equilibria

*class*

```python
class Equilibria(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Execute an equilibria strategy within the runnable interface.

**Parameters**

- `strategy` (`EquilibriaStrategy`) — Concrete ``EquilibriaStrategy`` used to solve equilibrium.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L48-L223)

### particula.equilibria.equilibria.Equilibria.strategy

*attribute* · *instance attribute*

```python
strategy = strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L61-L61)

### particula.equilibria.equilibria.Equilibria.__init__

*method*

```python
def __init__(strategy: EquilibriaStrategy) -> None
```

Initialize with the provided equilibria strategy.

**Parameters**

- `strategy` (`EquilibriaStrategy`) — Concrete ``EquilibriaStrategy`` used during execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L55-L61)

### particula.equilibria.equilibria.Equilibria._apply_attribute_result

*method* · *staticmethod*

```python
def _apply_attribute_result(aerosol: Aerosol, result: Any) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L208-L215)

### particula.equilibria.equilibria.Equilibria._apply_equilibrium_result

*method*

```python
def _apply_equilibrium_result(aerosol: Aerosol, result: Any) -> Aerosol
```

Apply strategy results to the aerosol or attach for downstream use.

Accepts either an updated ``Aerosol`` instance, a mapping/struct with
recognizable concentration fields, or raises when ``None`` is returned.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L154-L188)

### particula.equilibria.equilibria.Equilibria._apply_mapping_result

*method* · *staticmethod*

```python
def _apply_mapping_result(aerosol: Aerosol, result: Mapping[str, Any]) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L197-L206)

### particula.equilibria.equilibria.Equilibria._attach_equilibrium_result_dataclass

*method* · *staticmethod*

```python
def _attach_equilibrium_result_dataclass(aerosol: Aerosol, result: EquilibriumResult) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L190-L195)

### particula.equilibria.equilibria.Equilibria._attach_phase_concentrations

*method* · *staticmethod*

```python
def _attach_phase_concentrations(aerosol: Aerosol, result: PhaseConcentrations) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L217-L223)

### particula.equilibria.equilibria.Equilibria._extract_partitioning_inputs

*method*

```python
def _extract_partitioning_inputs(aerosol: Aerosol) -> dict[str, Any]
```

Extract required inputs for partitioning strategies.

Looks for a ``partitioning_inputs`` mapping on the aerosol first. If
absent, falls back to attributes of the aerosol itself. Raises with a
clear message when a required field is unavailable to keep failures
explicit during testing.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L111-L152)

### particula.equilibria.equilibria.Equilibria.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Run equilibria over the provided time step.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol state passed to the strategy and updated in place.
- `time_step` (`float`) — Simulation interval in seconds.
- `sub_steps` (`int`) (default: `1`) — Number of sub-divisions of ``time_step``. Must be > 0.

**Returns**

- (`Aerosol`) — The updated aerosol after equilibrium application.

**Raises**

- `ValueError` — If ``sub_steps`` is not positive.
- `AttributeError` — When required partitioning inputs are missing.
- `TypeError` — If the strategy returns ``None``.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L67-L109)

### particula.equilibria.equilibria.Equilibria.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Return the strategy identifier for non-rate processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L63-L65)

### particula.equilibria.equilibria.Equilibria.__or__

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

### particula.equilibria.equilibria.Equilibria._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L158)

### particula.equilibria.equilibria.Equilibria._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L168-L170)

### particula.equilibria.equilibria.Equilibria._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L152-L154)

### particula.equilibria.equilibria.Equilibria._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L160-L162)

### particula.equilibria.equilibria.Equilibria._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L164-L166)

### particula.equilibria.equilibria.Equilibria._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L145-L146)

### particula.equilibria.equilibria.Equilibria._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L150)

### particula.equilibria.equilibria.Equilibria.prepare

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

### particula.equilibria.equilibria.Equilibria.resume

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
