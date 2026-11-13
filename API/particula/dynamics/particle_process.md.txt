# particula.dynamics.particle_process

*module*

Runnable particle-centric aerosol processes.

Includes condensation and evaporation, coagulation, wall loss, dilution, and
the CPU-only single-box nucleation process.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L1-L1)

## particula.dynamics.particle_process._MAX_NUCLEATION_SUB_STEPS

*attribute* · *module attribute*

```python
_MAX_NUCLEATION_SUB_STEPS = 10000
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L59-L59)

## particula.dynamics.particle_process.Coagulation

*class*

```python
class Coagulation(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L478-L583)

### particula.dynamics.particle_process.Coagulation.coagulation_strategy

*attribute* · *instance attribute*

```python
coagulation_strategy = coagulation_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L516-L516)

### particula.dynamics.particle_process.Coagulation.__init__

*method*

```python
def __init__(coagulation_strategy: CoagulationStrategyABC)
```

Initialize the Coagulation process.

**Parameters**

- `coagulation_strategy` (`CoagulationStrategyABC`) — The coagulation strategy to use, describing how particles collide and combine.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L509-L516)

### particula.dynamics.particle_process.Coagulation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L518-L558)

### particula.dynamics.particle_process.Coagulation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L560-L583)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

## particula.dynamics.particle_process.Dilution

*class*

```python
class Dilution(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L679-L763)

### particula.dynamics.particle_process.Dilution.dilution_strategy

*attribute* · *instance attribute*

```python
dilution_strategy = dilution_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L703-L703)

### particula.dynamics.particle_process.Dilution.__init__

*method*

```python
def __init__(dilution_strategy: DilutionStrategyProtocol)
```

Initialize the dilution runnable.

dilution_strategy: Public ``DilutionStrategy`` or compatible custom
    strategy that applies dilution to an aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L696-L703)

### particula.dynamics.particle_process.Dilution.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float | np.number, sub_steps: int | np.integer = 1) -> Aerosol
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L717-L763)

### particula.dynamics.particle_process.Dilution.rate

*method*

```python
def rate(aerosol: Aerosol) -> float | NDArray[np.float64]
```

Delegate particle-number dilution-rate evaluation to the strategy.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol whose particle concentration is evaluated.

**Returns**

- (`float | NDArray[np.float64]`) — Particle-number concentration rate [1/(m³ s)] with the scalar or
- (`float | NDArray[np.float64]`) — array shape returned by the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L705-L715)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

## particula.dynamics.particle_process.DilutionStrategyProtocol

*class*

```python
class DilutionStrategyProtocol(Protocol)
```

Bases: `Protocol`

Structural contract for strategies compatible with ``Dilution``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L101-L108)

### particula.dynamics.particle_process.DilutionStrategyProtocol.rate

*method*

```python
def rate(aerosol: Aerosol) -> float | NDArray[np.float64]
```

Return the particle-number concentration rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L104-L105)

### particula.dynamics.particle_process.DilutionStrategyProtocol.step

*method*

```python
def step(aerosol: Aerosol, time_step: float) -> Aerosol
```

Apply one dilution step to an aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L107-L108)

## particula.dynamics.particle_process.MassCondensation

*class*

```python
class MassCondensation(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L363-L475)

### particula.dynamics.particle_process.MassCondensation.condensation_strategy

*attribute* · *instance attribute*

```python
condensation_strategy = condensation_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L405-L405)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L395-L405)

### particula.dynamics.particle_process.MassCondensation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L407-L449)

### particula.dynamics.particle_process.MassCondensation.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L451-L475)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

## particula.dynamics.particle_process.Nucleation

*class*

```python
class Nucleation(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Apply mapped single-box nucleation to a native resolved population.

Source rates and demand planning finalization use densities. The commit
converts event
density to represented counts using the current physical volume. Each
attempted substep is atomic; earlier successful substeps remain applied.
Gas and environment are read from the currently held aerosol every time.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L232-L360)

### particula.dynamics.particle_process.Nucleation.commit_config

*attribute* · *instance attribute*

```python
commit_config = commit_config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L255-L255)

### particula.dynamics.particle_process.Nucleation.source_config

*attribute* · *instance attribute*

```python
source_config = source_config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L254-L254)

### particula.dynamics.particle_process.Nucleation.__init__

*method*

```python
def __init__(source_config: NucleationSourceConfig, commit_config: NucleationCommitConfig) -> None
```

Bind immutable source selection and count-based commit controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L242-L255)

### particula.dynamics.particle_process.Nucleation._rate

*method*

```python
def _rate(gas: GasData, environment: EnvironmentData) -> float
```

Evaluate the current precursor and optional saturation gate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L298-L313)

### particula.dynamics.particle_process.Nucleation._state

*method*

```python
def _state(aerosol: Aerosol) -> tuple[ParticleData, GasData, EnvironmentData]
```

Validate current topology and configured injection alignment.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L257-L296)

### particula.dynamics.particle_process.Nucleation.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float | np.number, sub_steps: int | np.integer = 1) -> Aerosol
```

Commit equal sequential source substeps and retain all identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L320-L360)

### particula.dynamics.particle_process.Nucleation.rate

*method*

```python
def rate(aerosol: Aerosol) -> float
```

Return the current potential event density rate [m^-3 s^-1].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L315-L318)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

## particula.dynamics.particle_process.NucleationCommitConfig

*class* · *dataclass*

```python
class NucleationCommitConfig
```

Immutable one-box controls for CPU nucleation source commits.

The record mirrors the source transaction transaction controls while
retaining public runnable execution
ownership. Its rank-one sidecars are copied into read-only float64 arrays;
their required one-box length is checked immediately before every nonzero
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L153-L229)

### particula.dynamics.particle_process.NucleationCommitConfig.diversity_absolute_error

*attribute* · *class attribute* · *instance attribute*

```python
diversity_absolute_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L195-L195)

### particula.dynamics.particle_process.NucleationCommitConfig.exhaustion_controls

*attribute* · *instance attribute*

```python
exhaustion_controls: ExhaustionControls
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L188-L188)

### particula.dynamics.particle_process.NucleationCommitConfig.maximum_slot_weight

*attribute* · *instance attribute*

```python
maximum_slot_weight: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L186-L186)

### particula.dynamics.particle_process.NucleationCommitConfig.mean_radius_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
mean_radius_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L193-L193)

### particula.dynamics.particle_process.NucleationCommitConfig.minimum_scale

*attribute* · *instance attribute*

```python
minimum_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L190-L190)

### particula.dynamics.particle_process.NucleationCommitConfig.minimum_volume

*attribute* · *instance attribute*

```python
minimum_volume: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L191-L191)

### particula.dynamics.particle_process.NucleationCommitConfig.radius_cubed_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
radius_cubed_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L192-L192)

### particula.dynamics.particle_process.NucleationCommitConfig.requested_scale

*attribute* · *instance attribute*

```python
requested_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L189-L189)

### particula.dynamics.particle_process.NucleationCommitConfig.source_charge

*attribute* · *instance attribute*

```python
source_charge: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L187-L187)

### particula.dynamics.particle_process.NucleationCommitConfig.surface_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
surface_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L194-L194)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L197-L229)

## particula.dynamics.particle_process.WallLoss

*class*

```python
class WallLoss(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L586-L676)

### particula.dynamics.particle_process.WallLoss.wall_loss_strategy

*attribute* · *instance attribute*

```python
wall_loss_strategy = wall_loss_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L615-L615)

### particula.dynamics.particle_process.WallLoss.__init__

*method*

```python
def __init__(wall_loss_strategy: WallLossStrategy)
```

Create a wall loss runnable.

**Parameters**

- `wall_loss_strategy` (`WallLossStrategy`) — Strategy that provides wall loss rates and updates particle concentrations for the configured distribution type.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L607-L615)

### particula.dynamics.particle_process.WallLoss._clamp_non_negative

*method*

```python
def _clamp_non_negative(particle: ParticleData) -> None
```

Clamp particle concentrations to non-negative values.

**Parameters**

- `particle` (`ParticleData`) — Particle object whose concentration is clipped in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L617-L623)

### particula.dynamics.particle_process.WallLoss.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L625-L658)

### particula.dynamics.particle_process.WallLoss.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Return the wall loss rate for the aerosol particles.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol instance containing particles to evaluate.

**Returns**

- (`Any`) — Array of wall loss rates matching the particle representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L660-L676)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

## particula.dynamics.particle_process._is_real_scalar

*function*

```python
def _is_real_scalar(value: object) -> bool
```

Return whether ``value`` is a non-Boolean real scalar.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L144-L150)

## particula.dynamics.particle_process._native_duration

*function*

```python
def _native_duration(time_step: object, sub_steps: object) -> float
```

Validate equal-substep controls before any native process mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L90-L98)

## particula.dynamics.particle_process._native_process_state

*function*

```python
def _native_process_state(aerosol: Aerosol) -> tuple[ParticleData, GasData, float, float]
```

Read and validate the currently held single-box native process state.

This boundary never creates a facade or caches aggregate references.
Scientific strategies additionally validate their own physical inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L62-L87)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L111-L141)
