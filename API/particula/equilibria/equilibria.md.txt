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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L48-L216)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L201-L208)

### particula.equilibria.equilibria.Equilibria._apply_equilibrium_result

*method*

```python
def _apply_equilibrium_result(aerosol: Aerosol, result: Any) -> Aerosol
```

Apply strategy results to the aerosol or attach for downstream use.

Accepts either an updated ``Aerosol`` instance, a mapping/struct with
recognizable concentration fields, or raises when ``None`` is returned.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L147-L181)

### particula.equilibria.equilibria.Equilibria._apply_mapping_result

*method* · *staticmethod*

```python
def _apply_mapping_result(aerosol: Aerosol, result: Mapping[str, Any]) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L190-L199)

### particula.equilibria.equilibria.Equilibria._attach_equilibrium_result_dataclass

*method* · *staticmethod*

```python
def _attach_equilibrium_result_dataclass(aerosol: Aerosol, result: EquilibriumResult) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L183-L188)

### particula.equilibria.equilibria.Equilibria._attach_phase_concentrations

*method* · *staticmethod*

```python
def _attach_phase_concentrations(aerosol: Aerosol, result: PhaseConcentrations) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L210-L216)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L104-L145)

### particula.equilibria.equilibria.Equilibria.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L67-L102)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)
