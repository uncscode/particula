# particula.dynamics.particle_process

*module*

Runnable particle-centric aerosol processes.

Includes condensation and evaporation, coagulation, wall loss, dilution, and
the CPU-only nucleation process.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L1-L1)

## particula.dynamics.particle_process._MAX_NUCLEATION_SUB_STEPS

*attribute* · *module attribute*

```python
_MAX_NUCLEATION_SUB_STEPS = 10000
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L60-L60)

## particula.dynamics.particle_process.Coagulation

*class*

```python
class Coagulation(_NativeBatchRunnable)
```

Bases: `particula.dynamics.particle_process._NativeBatchRunnable`

Implements a coagulation process for aerosol particles.

Apply a strategy to the held single-box native particle population.
Each equal substep reads the current environment and retains the particle
container. Non-native aggregates reject before strategy dispatch.

**Attributes**

- `coagulation_strategy` — The coagulation strategy used for particle collision calculations.

Methods:
- execute : Perform the coagulation step over a given time interval.
- rate : Calculate the coagulation rate for each particle.

**Examples**

```py title="Example Usage"
import particula as par
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=my_strategy
)
updated_aerosol = coagulation.execute(aerosol, time_step=0.5)
# updated_aerosol now reflects coalesced or aggregated particles
```

> **References**
>
> - [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
> - Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and
>   Physics: From Air Pollution to Climate Change," Wiley, 2016.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L638-L746)

### particula.dynamics.particle_process.Coagulation._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'coagulation'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L670-L670)

### particula.dynamics.particle_process.Coagulation._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute = 'coagulation_strategy'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L669-L669)

### particula.dynamics.particle_process.Coagulation.coagulation_strategy

*attribute* · *instance attribute*

```python
coagulation_strategy = coagulation_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L679-L679)

### particula.dynamics.particle_process.Coagulation.__init__

*method*

```python
def __init__(coagulation_strategy: CoagulationStrategyABC)
```

Initialize the Coagulation process.

**Parameters**

- `coagulation_strategy` (`CoagulationStrategyABC`) — The coagulation strategy to use, describing how particles collide and combine.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L672-L679)

### particula.dynamics.particle_process.Coagulation._execute_box

*method*

```python
def _execute_box(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
```

Perform the coagulation process over a given time step.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance to modify.
- `time_step` (`float`) — The total time interval for coagulation.
- `sub_steps` (`int`) (default: `1`) — Number of internal subdivisions for iterative calculation.

**Returns**

- `Aerosol` (`Aerosol`) — The updated aerosol object after the coagulation step.

**Examples**

```py title="Example Coagulation Execution"
updated_aerosol = coagulation.execute(
    aerosol, time_step=0.5, sub_steps=2
)
# The aerosol now reflects changes from particle collisions
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L681-L721)

### particula.dynamics.particle_process.Coagulation._rate_box

*method*

```python
def _rate_box(aerosol: Aerosol) -> Any
```

Compute the coagulation rate for each particle in the aerosol.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance containing particles.

**Returns**

- (`Any`) — np.ndarray: An array of coagulation rates for each particle, in units related to particle collisions per unit time.

**Examples**

```py title="Coagulation Rate Calculation Example"
rates = coagulation.rate(aerosol)
# rates might look like array([0.1, 0.05, ...])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L723-L746)

### particula.dynamics.particle_process.Coagulation._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process.Coagulation._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process.Coagulation._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process.Coagulation._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process.Coagulation._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process.Coagulation._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process.Coagulation.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

### particula.dynamics.particle_process.Coagulation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Inherited from: `particula.runnable.RunnableABC`

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L84-L138)

### particula.dynamics.particle_process.Coagulation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L67-L82)

## particula.dynamics.particle_process.Dilution

*class*

```python
class Dilution(_NativeBatchRunnable)
```

Bases: `particula.dynamics.particle_process._NativeBatchRunnable`

Apply a dilution strategy over one or more equal substeps.

Each substep delegates aerosol mutation to the configured strategy. The
runnable preserves the input aerosol identity and ignores a strategy's
return value. When configured with the public ``DilutionStrategy``, it
validates concrete aerosol state before its first delegated substep, so
malformed state cannot permit a partial multi-substep update. Compatible
custom strategies retain generic equal-substep delegation and own their
validation and atomicity.

**Parameters**

- `dilution_strategy` (`DilutionStrategyProtocol`) — Public ``DilutionStrategy`` or a compatible custom strategy that reports particle-number rates and applies aerosol dilution steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L845-L948)

### particula.dynamics.particle_process.Dilution._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'dilution'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L863-L863)

### particula.dynamics.particle_process.Dilution._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute = 'dilution_strategy'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L862-L862)

### particula.dynamics.particle_process.Dilution.dilution_strategy

*attribute* · *instance attribute*

```python
dilution_strategy = dilution_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L888-L888)

### particula.dynamics.particle_process.Dilution.__init__

*method*

```python
def __init__(dilution_strategy: DilutionStrategyProtocol)
```

Initialize the dilution runnable.

dilution_strategy: Public ``DilutionStrategy`` or compatible custom
    strategy that applies dilution to an aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L881-L888)

### particula.dynamics.particle_process.Dilution._execute_box

*method*

```python
def _execute_box(aerosol: Aerosol, time_step: float | np.number, sub_steps: int | np.integer = 1) -> Aerosol
```

Apply dilution as equal, sequential substeps over a total duration.

The runnable validates the total duration and substep count before
calling the strategy. A ``DilutionStrategy`` validates and executes one
preflight before the first equal native substep. Each substep is atomic;
prior successful steps persist if a later step fails.
Custom strategies are not inspected and retain equal-substep delegation
with responsibility for aerosol validation, mutation, and atomicity.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol to mutate in place.
- `time_step` (`float | np.number`) — Total elapsed time [s], finite and nonnegative.
- `sub_steps` (`int | np.integer`) (default: `1`) — Positive count of equal internal dilution steps.

**Returns**

- (`Aerosol`) — The identical, mutated aerosol instance.

**Raises**

- `ValueError` — If ``sub_steps`` is not a positive integer,
``time_step`` is nonfinite, negative, or nonscalar, or
supported concrete aerosol state is invalid.
- `TypeError` — If ``time_step`` is not numeric or a required supported
concrete aerosol value is not numeric.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L902-L948)

### particula.dynamics.particle_process.Dilution._rate_box

*method*

```python
def _rate_box(aerosol: Aerosol) -> float | NDArray[np.float64]
```

Delegate particle-number dilution-rate evaluation to the strategy.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol whose particle concentration is evaluated.

**Returns**

- (`float | NDArray[np.float64]`) — Particle-number concentration rate [1/(m³ s)] with the scalar or
- (`float | NDArray[np.float64]`) — array shape returned by the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L890-L900)

### particula.dynamics.particle_process.Dilution.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None)
```

Batch native dilution; retain legacy custom strategy delegation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L865-L873)

### particula.dynamics.particle_process.Dilution.rate

*method*

```python
def rate(aerosol)
```

Return native batched rates or the custom strategy's own result.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L875-L879)

### particula.dynamics.particle_process.Dilution._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process.Dilution._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process.Dilution._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process.Dilution._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process.Dilution._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process.Dilution._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process.Dilution.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

## particula.dynamics.particle_process.DilutionStrategyProtocol

*class*

```python
class DilutionStrategyProtocol(Protocol)
```

Bases: `Protocol`

Structural contract for strategies compatible with ``Dilution``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L102-L109)

### particula.dynamics.particle_process.DilutionStrategyProtocol.rate

*method*

```python
def rate(aerosol: Aerosol) -> float | NDArray[np.float64]
```

Return the particle-number concentration rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L105-L106)

### particula.dynamics.particle_process.DilutionStrategyProtocol.step

*method*

```python
def step(aerosol: Aerosol, time_step: float) -> Aerosol
```

Apply one dilution step to an aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L108-L109)

## particula.dynamics.particle_process.MassCondensation

*class*

```python
class MassCondensation(_NativeBatchRunnable)
```

Bases: `particula.dynamics.particle_process._NativeBatchRunnable`

Handles the mass condensation process for aerosols.

Apply a strategy to the currently held native particle and gas containers.
Each equal substep reads the current environment and preserves container
identity. Process-owned physics and species maps define the mass transfer;
non-native aggregates reject before strategy dispatch.

**Attributes**

- `condensation_strategy` — The condensation strategy used for mass transfer calculations.

Methods:
- execute : Perform the mass condensation over a specified time step.
- rate : Calculate the mass condensation rate for each particle.

**Examples**

```py title="Example Mass Condensation"
import particula as par
condensation = par.dynamics.MassCondensation(
    condensation_strategy=my_strategy
)
updated_aerosol = condensation.execute(aerosol, time_step=1.0)
# updated_aerosol now reflects condensed mass
```

References:
- [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
- Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and Physics:
  From Air Pollution to Climate Change," Wiley, 2016.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L520-L635)

### particula.dynamics.particle_process.MassCondensation._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'condensation'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L553-L553)

### particula.dynamics.particle_process.MassCondensation._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute = 'condensation_strategy'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L552-L552)

### particula.dynamics.particle_process.MassCondensation.condensation_strategy

*attribute* · *instance attribute*

```python
condensation_strategy = condensation_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L565-L565)

### particula.dynamics.particle_process.MassCondensation.__init__

*method*

```python
def __init__(condensation_strategy: CondensationStrategy)
```

Initialize the MassCondensation process.

**Parameters**

- `condensation_strategy` (`CondensationStrategy`) — The condensation strategy to use, responsible for calculating mass transfer.

**Returns**

- `result` (`None`) — No return value.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L555-L565)

### particula.dynamics.particle_process.MassCondensation._execute_box

*method*

```python
def _execute_box(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
```

Perform the mass condensation process over a given time step.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance to modify.
- `time_step` (`float`) — The total time interval for condensation.
- `sub_steps` (`int`) (default: `1`) — Number of subdivisions for iterative calculation.

**Returns**

- (`Aerosol`) — The updated aerosol object after condensation.

**Examples**

```py title="Example Condensation Execution"
updated_aerosol = condensation.execute(
    aerosol, time_step=1.0, sub_steps=2
)
# The aerosol now has reduced/increased particle/gas mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L567-L609)

### particula.dynamics.particle_process.MassCondensation._rate_box

*method*

```python
def _rate_box(aerosol: Aerosol) -> Any
```

Compute mass condensation rates for each particle.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance containing particles and gases.

**Returns**

- (`Any`) — An array of condensation rates for each particle, in units of mass per unit time.

**Examples**

```py title="Rate Calculation Example"
rates = condensation.rate(aerosol)
# rates may look like array([1.2e-12, 4.5e-12, ...])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L611-L635)

### particula.dynamics.particle_process.MassCondensation._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process.MassCondensation._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process.MassCondensation._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process.MassCondensation._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process.MassCondensation._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process.MassCondensation._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process.MassCondensation.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

### particula.dynamics.particle_process.MassCondensation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Inherited from: `particula.runnable.RunnableABC`

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L84-L138)

### particula.dynamics.particle_process.MassCondensation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L67-L82)

## particula.dynamics.particle_process.Nucleation

*class*

```python
class Nucleation(_NativeBatchRunnable)
```

Bases: `particula.dynamics.particle_process._NativeBatchRunnable`

Apply mapped nucleation independently to native resolved boxes.

Source rates and demand planning finalization use densities. The commit
converts event
density to represented counts using the current physical volume. Each
attempted substep is atomic; earlier successful substeps remain applied.
Gas and environment are read from the currently held aerosol every time.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L385-L517)

### particula.dynamics.particle_process.Nucleation._maximum_sub_steps

*attribute* · *class attribute* · *instance attribute*

```python
_maximum_sub_steps = _MAX_NUCLEATION_SUB_STEPS
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L397-L397)

### particula.dynamics.particle_process.Nucleation._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'nucleation'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L396-L396)

### particula.dynamics.particle_process.Nucleation._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L395-L395)

### particula.dynamics.particle_process.Nucleation.commit_config

*attribute* · *instance attribute*

```python
commit_config = commit_config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L412-L412)

### particula.dynamics.particle_process.Nucleation.source_config

*attribute* · *instance attribute*

```python
source_config = source_config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L411-L411)

### particula.dynamics.particle_process.Nucleation.__init__

*method*

```python
def __init__(source_config: NucleationSourceConfig, commit_config: NucleationCommitConfig) -> None
```

Bind immutable source selection and count-based commit controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L399-L412)

### particula.dynamics.particle_process.Nucleation._execute_box

*method*

```python
def _execute_box(aerosol: Aerosol, time_step: float | np.number, sub_steps: int | np.integer = 1) -> Aerosol
```

Commit equal sequential source substeps and retain all identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L477-L517)

### particula.dynamics.particle_process.Nucleation._rate

*method*

```python
def _rate(gas: GasData, environment: EnvironmentData) -> float
```

Evaluate the current precursor and optional saturation gate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L455-L470)

### particula.dynamics.particle_process.Nucleation._rate_box

*method*

```python
def _rate_box(aerosol: Aerosol) -> float
```

Return the current potential event density rate [m^-3 s^-1].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L472-L475)

### particula.dynamics.particle_process.Nucleation._state

*method*

```python
def _state(aerosol: Aerosol) -> tuple[ParticleData, GasData, EnvironmentData]
```

Validate current topology and configured injection alignment.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L414-L453)

### particula.dynamics.particle_process.Nucleation._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process.Nucleation._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process.Nucleation._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process.Nucleation._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process.Nucleation._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process.Nucleation._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process.Nucleation.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

### particula.dynamics.particle_process.Nucleation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Inherited from: `particula.runnable.RunnableABC`

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L84-L138)

### particula.dynamics.particle_process.Nucleation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L67-L82)

## particula.dynamics.particle_process.NucleationCommitConfig

*class* · *dataclass*

```python
class NucleationCommitConfig
```

Immutable per-box controls for CPU nucleation source commits.

The record mirrors the source transaction transaction controls while
retaining public runnable execution
ownership. Its rank-one sidecars are copied into read-only float64 arrays;
their required batch length is checked immediately before every nonzero
substep commit. The configuration is not a source transaction configuration
and does not
create a transaction spanning a complete ``Nucleation.execute`` call.

**Parameters**

- `maximum_slot_weight` (`float`) — Positive maximum represented events per slot: represented counts for native Nucleation.
- `source_charge` (`float`) — Finite dimensionless elementary-charge count assigned to activated source slots.
- `exhaustion_controls` (`ExhaustionControls`) — Immutable fixed-capacity policy controls.
- `requested_scale` (`NDArray[np.float64]`) — Requested representative-volume scale per box.
- `minimum_scale` (`NDArray[np.float64]`) — Minimum allowed representative-volume scale per box.
- `minimum_volume` (`NDArray[np.float64]`) — Minimum representative volume per box [m³].
- `radius_cubed_relative_error` (`float`) (default: `1.0`) — Allowed resampling radius-cubed error.
- `mean_radius_relative_error` (`float`) (default: `1.0`) — Allowed resampling mean-radius error.
- `surface_relative_error` (`float`) (default: `1.0`) — Allowed resampling surface-area error.
- `diversity_absolute_error` (`float`) (default: `1.0`) — Allowed resampling diversity error.

**Raises**

- `TypeError` — If a scalar control, tolerance, exhaustion control, or
sidecar type is invalid.
- `ValueError` — If a scalar or tolerance is outside its allowed domain, or
a sidecar is not rank one.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L154-L230)

### particula.dynamics.particle_process.NucleationCommitConfig.diversity_absolute_error

*attribute* · *class attribute* · *instance attribute*

```python
diversity_absolute_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L196-L196)

### particula.dynamics.particle_process.NucleationCommitConfig.exhaustion_controls

*attribute* · *instance attribute*

```python
exhaustion_controls: ExhaustionControls
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L189-L189)

### particula.dynamics.particle_process.NucleationCommitConfig.maximum_slot_weight

*attribute* · *instance attribute*

```python
maximum_slot_weight: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L187-L187)

### particula.dynamics.particle_process.NucleationCommitConfig.mean_radius_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
mean_radius_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L194-L194)

### particula.dynamics.particle_process.NucleationCommitConfig.minimum_scale

*attribute* · *instance attribute*

```python
minimum_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L191-L191)

### particula.dynamics.particle_process.NucleationCommitConfig.minimum_volume

*attribute* · *instance attribute*

```python
minimum_volume: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L192-L192)

### particula.dynamics.particle_process.NucleationCommitConfig.radius_cubed_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
radius_cubed_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L193-L193)

### particula.dynamics.particle_process.NucleationCommitConfig.requested_scale

*attribute* · *instance attribute*

```python
requested_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L190-L190)

### particula.dynamics.particle_process.NucleationCommitConfig.source_charge

*attribute* · *instance attribute*

```python
source_charge: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L188-L188)

### particula.dynamics.particle_process.NucleationCommitConfig.surface_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
surface_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L195-L195)

### particula.dynamics.particle_process.NucleationCommitConfig.__init__

*method*

```python
def __init__(maximum_slot_weight: float, source_charge: float, exhaustion_controls: ExhaustionControls, requested_scale: NDArray[np.float64], minimum_scale: NDArray[np.float64], minimum_volume: NDArray[np.float64], radius_cubed_relative_error: float = 1.0, mean_radius_relative_error: float = 1.0, surface_relative_error: float = 1.0, diversity_absolute_error: float = 1.0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L0-L0)

### particula.dynamics.particle_process.NucleationCommitConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate scalar controls and defensively own sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L198-L230)

## particula.dynamics.particle_process.WallLoss

*class*

```python
class WallLoss(_NativeBatchRunnable)
```

Bases: `particula.dynamics.particle_process._NativeBatchRunnable`

Apply wall loss strategy to aerosol particles.

Supports discrete, continuous PDF, and particle-resolved distributions via
the configured wall loss strategy. The total ``time_step`` is split across
``sub_steps``. Native strategies reject oversized steps before writing;
the runnable clamps concentrations after successful strategy delegation.

> **Example**
>
> >>> import particula as par
> >>> strategy = par.dynamics.SphericalWallLossStrategy(
> ...     wall_eddy_diffusivity=0.001,
> ...     chamber_radius=0.5,
> ...     distribution_type="discrete",
> ... )
> >>> wall_loss = par.dynamics.WallLoss(
> ...     wall_loss_strategy=strategy,
> ... )
> >>> _ = wall_loss.execute(aerosol, time_step=1.0, sub_steps=2)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L749-L842)

### particula.dynamics.particle_process.WallLoss._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'wall_loss'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L771-L771)

### particula.dynamics.particle_process.WallLoss._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute = 'wall_loss_strategy'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L770-L770)

### particula.dynamics.particle_process.WallLoss.wall_loss_strategy

*attribute* · *instance attribute*

```python
wall_loss_strategy = wall_loss_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L781-L781)

### particula.dynamics.particle_process.WallLoss.__init__

*method*

```python
def __init__(wall_loss_strategy: WallLossStrategy)
```

Create a wall loss runnable.

**Parameters**

- `wall_loss_strategy` (`WallLossStrategy`) — Strategy that provides wall loss rates and updates particle concentrations for the configured distribution type.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L773-L781)

### particula.dynamics.particle_process.WallLoss._clamp_non_negative

*method*

```python
def _clamp_non_negative(particle: ParticleData) -> None
```

Clamp particle concentrations to non-negative values.

**Parameters**

- `particle` (`ParticleData`) — Particle object whose concentration is clipped in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L783-L789)

### particula.dynamics.particle_process.WallLoss._execute_box

*method*

```python
def _execute_box(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
```

Apply wall loss over the provided time step.

Concentrations are clamped to remain non-negative after each
sub-step.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol instance to update.
- `time_step` (`float`) — Total simulation interval in seconds.
- `sub_steps` (`int`) (default: `1`) — Number of internal steps used to split ``time_step``.

**Returns**

- (`Aerosol`) — Aerosol with updated particle concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L791-L824)

### particula.dynamics.particle_process.WallLoss._rate_box

*method*

```python
def _rate_box(aerosol: Aerosol) -> Any
```

Return the wall loss rate for the aerosol particles.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol instance containing particles to evaluate.

**Returns**

- (`Any`) — Array of wall loss rates matching the particle representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L826-L842)

### particula.dynamics.particle_process.WallLoss._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process.WallLoss._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process.WallLoss._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process.WallLoss._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process.WallLoss._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process.WallLoss._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.dynamics.particle_process._NativeBatchRunnable`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process.WallLoss.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

### particula.dynamics.particle_process.WallLoss.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Inherited from: `particula.runnable.RunnableABC`

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L84-L138)

### particula.dynamics.particle_process.WallLoss.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L67-L82)

## particula.dynamics.particle_process._NativeBatchRunnable

*class*

```python
class _NativeBatchRunnable(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Bind native single-box strategies to the shared batch transaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L233-L382)

### particula.dynamics.particle_process._NativeBatchRunnable._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = ''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L237-L237)

### particula.dynamics.particle_process._NativeBatchRunnable._strategy_attribute

*attribute* · *class attribute* · *instance attribute*

```python
_strategy_attribute: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L236-L236)

### particula.dynamics.particle_process._NativeBatchRunnable._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L309-L337)

### particula.dynamics.particle_process._NativeBatchRunnable._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L299-L307)

### particula.dynamics.particle_process._NativeBatchRunnable._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L339-L382)

### particula.dynamics.particle_process._NativeBatchRunnable._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L259-L278)

### particula.dynamics.particle_process._NativeBatchRunnable._validate_gas_schema

*method*

```python
def _validate_gas_schema(gas)
```

Reject raw replacements before GasData copying can normalize them.

Only gas-coupled processes require these schemas. Condensation uses
float64 throughout; nucleation allows real numeric molar-mass metadata
but its writable gas inventory must remain float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L280-L297)

### particula.dynamics.particle_process._NativeBatchRunnable._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L239-L257)

### particula.dynamics.particle_process._NativeBatchRunnable.__or__

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L188)

### particula.dynamics.particle_process._NativeBatchRunnable._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.dynamics.particle_process._NativeBatchRunnable._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.dynamics.particle_process._NativeBatchRunnable.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1, *, context=None) -> Aerosol
```

Inherited from: `particula.runnable.RunnableABC`

Advance every box, committing atomically after each process substep.

Contexts own independent logical-box random streams. Completed substeps
remain committed if a later one fails. Custom subclasses overriding
this method retain responsibility for their own execution semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L84-L138)

### particula.dynamics.particle_process._NativeBatchRunnable.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L67-L82)

## particula.dynamics.particle_process._is_real_scalar

*function*

```python
def _is_real_scalar(value: object) -> bool
```

Return whether ``value`` is a non-Boolean real scalar.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L145-L151)

## particula.dynamics.particle_process._native_duration

*function*

```python
def _native_duration(time_step: object, sub_steps: object) -> float
```

Validate equal-substep controls before any native process mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L91-L99)

## particula.dynamics.particle_process._native_process_state

*function*

```python
def _native_process_state(aerosol: Aerosol) -> tuple[ParticleData, GasData, float, float]
```

Read and validate the currently held single-box native process state.

This boundary never creates a facade or caches aggregate references.
Scientific strategies additionally validate their own physical inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L63-L88)

## particula.dynamics.particle_process._readonly_float64_vector

*function*

```python
def _readonly_float64_vector(values: object, name: str) -> NDArray[np.float64]
```

Return an owned, read-only, rank-one float64 array.

**Parameters**

- `values` (`object`) — Array-compatible sidecar values.
- `name` (`str`) — Field name used in validation errors.

**Returns**

- (`NDArray[np.float64]`) — Fresh immutable float64 vector.

**Raises**

- `TypeError` — If the values cannot be converted to float64.
- `ValueError` — If the converted values are not rank one.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L112-L142)
