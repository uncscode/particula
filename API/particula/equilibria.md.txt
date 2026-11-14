# particula.equilibria

*module*

Thermodynamic and chemical equilibria utilities.

This module exposes the equilibria Strategy–Builder–Factory–Runnable
surface plus legacy helper functions for backward compatibility.
Recommended usage is via the strategy or factory interfaces; legacy
function shortcuts remain available but emit deprecation warnings.

**Classes**

- `EquilibriaStrategy` — Abstract base class for equilibria strategies.
- `LiquidVaporPartitioningStrategy` — Liquid-vapor partitioning strategy.
- `EquilibriumResult` — Result dataclass for equilibrium calculations.
- `PhaseConcentrations` — Dataclass for phase concentration data.
- `LiquidVaporPartitioningBuilder` — Builder for partitioning strategies.
- `EquilibriaFactory` — Factory for equilibria strategies.
- `Equilibria` — Runnable wrapper for equilibria strategies.

Functions (legacy):
    liquid_vapor_partitioning: Direct partitioning calculation (legacy).
    get_properties_for_liquid_vapor_partitioning: Legacy property helper.
    liquid_vapor_obj_function: Objective function for partitioning (legacy).

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an
> efficient reduced-complexity model. Atmospheric Chemistry and
> Physics, 19(19), 13383-13407.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/__init__.py#L1-L1)

## particula.equilibria.Equilibria

*class*

```python
class Equilibria(RunnableABC)
```

Bases: `particula.runnable.RunnableABC`

Re-exported from: `particula.equilibria.equilibria`

Execute an equilibria strategy within the runnable interface.

**Parameters**

- `strategy` (`EquilibriaStrategy`) — Concrete ``EquilibriaStrategy`` used to solve equilibrium.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L48-L223)

### particula.equilibria.Equilibria.strategy

*attribute* · *instance attribute*

```python
strategy = strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L61-L61)

### particula.equilibria.Equilibria.__init__

*method*

```python
def __init__(strategy: EquilibriaStrategy) -> None
```

Initialize with the provided equilibria strategy.

**Parameters**

- `strategy` (`EquilibriaStrategy`) — Concrete ``EquilibriaStrategy`` used during execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L55-L61)

### particula.equilibria.Equilibria._apply_attribute_result

*method* · *staticmethod*

```python
def _apply_attribute_result(aerosol: Aerosol, result: Any) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L208-L215)

### particula.equilibria.Equilibria._apply_equilibrium_result

*method*

```python
def _apply_equilibrium_result(aerosol: Aerosol, result: Any) -> Aerosol
```

Apply strategy results to the aerosol or attach for downstream use.

Accepts either an updated ``Aerosol`` instance, a mapping/struct with
recognizable concentration fields, or raises when ``None`` is returned.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L154-L188)

### particula.equilibria.Equilibria._apply_mapping_result

*method* · *staticmethod*

```python
def _apply_mapping_result(aerosol: Aerosol, result: Mapping[str, Any]) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L197-L206)

### particula.equilibria.Equilibria._attach_equilibrium_result_dataclass

*method* · *staticmethod*

```python
def _attach_equilibrium_result_dataclass(aerosol: Aerosol, result: EquilibriumResult) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L190-L195)

### particula.equilibria.Equilibria._attach_phase_concentrations

*method* · *staticmethod*

```python
def _attach_phase_concentrations(aerosol: Aerosol, result: PhaseConcentrations) -> Aerosol
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L217-L223)

### particula.equilibria.Equilibria._extract_partitioning_inputs

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

### particula.equilibria.Equilibria.execute

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

### particula.equilibria.Equilibria.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Return the strategy identifier for non-rate processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria.py#L63-L65)

### particula.equilibria.Equilibria.__or__

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

### particula.equilibria.Equilibria._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L158)

### particula.equilibria.Equilibria._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L168-L170)

### particula.equilibria.Equilibria._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L152-L154)

### particula.equilibria.Equilibria._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L160-L162)

### particula.equilibria.Equilibria._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L164-L166)

### particula.equilibria.Equilibria._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L145-L146)

### particula.equilibria.Equilibria._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L150)

### particula.equilibria.Equilibria.prepare

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

### particula.equilibria.Equilibria.resume

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

## particula.equilibria.EquilibriaFactory

*class*

```python
class EquilibriaFactory(StrategyFactoryABC[BuilderType, EquilibriaStrategy])
```

Bases: ``

Re-exported from: `particula.equilibria.equilibria_factories`

Factory for equilibria strategies.

Builds equilibria strategies using registered builders and the
:class:`StrategyFactoryABC` workflow.

> **Supported strategy types**
>
> - "liquid_vapor": ``LiquidVaporPartitioningStrategy`` for organic
>   aerosol liquid-vapor partitioning equilibrium.

**Examples**

```pycon
>>> factory = EquilibriaFactory()
>>> strategy = factory.get_strategy("liquid_vapor")
>>> strategy.water_activity
0.5
```

```pycon
>>> strategy = factory.get_strategy(
...     "liquid_vapor", parameters={"water_activity": 0.75}
... )
>>> strategy.water_activity
0.75
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L33-L106)

### particula.equilibria.EquilibriaFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, BuilderType]
```

Return available equilibria builders keyed by strategy name.

**Returns**

- (`Dict[str, BuilderType]`) — Dict[str, BuilderType]: Mapping of strategy type to builder.

**Examples**

```pycon
>>> factory = EquilibriaFactory()
>>> builders = factory.get_builders()
>>> "liquid_vapor" in builders
True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L56-L70)

### particula.equilibria.EquilibriaFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None) -> EquilibriaStrategy
```

Create an equilibria strategy using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Name of the strategy to build. Case-insensitive. Supported: "liquid_vapor".
- `parameters` (`Optional[Dict[str, Any]]`) (default: `None`) — Optional parameter mapping for the builder. For "liquid_vapor": {"water_activity": float}.

**Returns**

- (`EquilibriaStrategy`) — Built equilibria strategy instance.

**Raises**

- `ValueError` — If ``strategy_type`` is unknown or parameters are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L72-L106)

## particula.equilibria.EquilibriaStrategy

*class*

```python
class EquilibriaStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.equilibria.equilibria_strategies`

Abstract base class for equilibria strategies.

Equilibria strategies compute thermodynamic equilibrium states given
system conditions. Concrete implementations may solve liquid-vapor,
solid-liquid, or other phase equilibria problems.

**Examples**

```pycon
>>> import numpy as np
>>> class CustomEquilibria(EquilibriaStrategy):
...     def solve(self, *args, **kwargs):
...         return kwargs["equilibrium_result"]
>>> strategy = CustomEquilibria()
>>> strategy.get_name()
'CustomEquilibria'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L69-L113)

### particula.equilibria.EquilibriaStrategy.get_name

*method*

```python
def get_name() -> str
```

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L111-L113)

### particula.equilibria.EquilibriaStrategy.solve

*method* · *abstract*

```python
def solve(c_star_j_dry: NDArray[np.float64], concentration_organic_matter: NDArray[np.float64], molar_mass: NDArray[np.float64], oxygen2carbon: NDArray[np.float64], density: NDArray[np.float64], partition_coefficient_guess: Optional[NDArray[np.float64]] = None) -> EquilibriumResult
```

Solve for equilibrium state.

**Parameters**

- `c_star_j_dry` (`NDArray[np.float64]`) — Dry saturation concentrations [µg/m³].
- `concentration_organic_matter` (`NDArray[np.float64]`) — Organic mass concentrations [µg/m³].
- `molar_mass` (`NDArray[np.float64]`) — Species molar masses [g/mol].
- `oxygen2carbon` (`NDArray[np.float64]`) — Oxygen-to-carbon ratios [-].
- `density` (`NDArray[np.float64]`) — Species densities [kg/m³].
- `partition_coefficient_guess` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional initial guesses for partition coefficients [-].

**Returns**

- (`EquilibriumResult`) — Structured equilibrium result.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L86-L109)

## particula.equilibria.EquilibriumResult

*class* · *dataclass*

```python
class EquilibriumResult
```

Re-exported from: `particula.equilibria.equilibria_strategies`

Result of an equilibria calculation.

**Attributes**

- `alpha_phase` (`PhaseConcentrations`) — Concentrations in the alpha (water-rich) phase.
- `beta_phase` (`Optional[PhaseConcentrations]`) — Concentrations in the beta (organic-rich) phase. ``None`` when no beta phase is present.
- `partition_coefficients` (`NDArray[np.float64]`) — Species partition coefficients [-].
- `water_content` (`tuple[float, float]`) — Water content for (alpha, beta) phases [µg/m³].
- `error` (`float`) — Optimization convergence error.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L49-L66)

### particula.equilibria.EquilibriumResult.alpha_phase

*attribute* · *instance attribute*

```python
alpha_phase: PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L62-L62)

### particula.equilibria.EquilibriumResult.beta_phase

*attribute* · *instance attribute*

```python
beta_phase: Optional[PhaseConcentrations]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L63-L63)

### particula.equilibria.EquilibriumResult.error

*attribute* · *instance attribute*

```python
error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L66-L66)

### particula.equilibria.EquilibriumResult.partition_coefficients

*attribute* · *instance attribute*

```python
partition_coefficients: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L64-L64)

### particula.equilibria.EquilibriumResult.water_content

*attribute* · *instance attribute*

```python
water_content: tuple[float, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L65-L65)

### particula.equilibria.EquilibriumResult.__init__

*method*

```python
def __init__(alpha_phase: PhaseConcentrations, beta_phase: Optional[PhaseConcentrations], partition_coefficients: NDArray[np.float64], water_content: tuple[float, float], error: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)

## particula.equilibria.LiquidVaporPartitioningBuilder

*class*

```python
class LiquidVaporPartitioningBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.equilibria.equilibria_builders`

Builder for :class:`LiquidVaporPartitioningStrategy`.

Supports fluent configuration with validation for water activity and
deterministic parameter handling for factory usage.

**Examples**

```pycon
>>> builder = LiquidVaporPartitioningBuilder()
>>> strategy = builder.set_water_activity(0.8).build()
>>> strategy.water_activity
0.8
```

```pycon
>>> builder = LiquidVaporPartitioningBuilder()
>>> strategy = builder.set_parameters({"water_activity": 0.65}).build()
>>> strategy.water_activity
0.65
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L17-L100)

### particula.equilibria.LiquidVaporPartitioningBuilder._water_activity

*attribute* · *instance attribute*

```python
_water_activity: float | None = 0.5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L38-L38)

### particula.equilibria.LiquidVaporPartitioningBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.equilibria.LiquidVaporPartitioningBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize builder with default water activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L35-L38)

### particula.equilibria.LiquidVaporPartitioningBuilder.build

*method*

```python
def build() -> LiquidVaporPartitioningStrategy
```

Build the configured :class:`LiquidVaporPartitioningStrategy`.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L93-L100)

### particula.equilibria.LiquidVaporPartitioningBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Ensure builder is ready to build a strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L87-L91)

### particula.equilibria.LiquidVaporPartitioningBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> Self
```

Set parameters from a dictionary with strict key validation.

Only ``{"water_activity"}`` is accepted. Unknown keys, including
unit-suffixed variants, raise ``ValueError`` to keep behavior
deterministic for factories.

**Parameters**

- `parameters` (`dict[str, Any]`) — Parameter dictionary.

**Returns**

- (`Self`) — Self for method chaining.

**Raises**

- `ValueError` — If an unknown key is provided or value is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L58-L85)

### particula.equilibria.LiquidVaporPartitioningBuilder.set_water_activity

*method*

```python
def set_water_activity(value: float) -> Self
```

Set the target water activity.

**Parameters**

- `value` (`float`) — Water activity in range [0, 1].

**Returns**

- (`Self`) — Self for method chaining.

**Raises**

- `ValueError` — If ``value`` is outside the inclusive [0, 1] range.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L40-L56)

### particula.equilibria.LiquidVaporPartitioningBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

## particula.equilibria.LiquidVaporPartitioningStrategy

*class*

```python
class LiquidVaporPartitioningStrategy(EquilibriaStrategy)
```

Bases: `particula.equilibria.equilibria_strategies.EquilibriaStrategy`

Re-exported from: `particula.equilibria.equilibria_strategies`

Liquid-vapor partitioning equilibrium strategy.

Wraps the partitioning helpers to compute equilibrium between gas and
condensed phases while exposing structured results.

**Parameters**

- `water_activity` (`float`) (default: `0.5`) — Target water activity (0-1 range). Defaults to 0.5.

**Examples**

```pycon
>>> import numpy as np
>>> strategy = LiquidVaporPartitioningStrategy(water_activity=0.75)
>>> result = strategy.solve(
...     c_star_j_dry=np.array([1e-6, 1e-4, 1e-2]),
...     concentration_organic_matter=np.array([1.0, 5.0, 10.0]),
...     molar_mass=np.array([200.0, 200.0, 200.0]),
...     oxygen2carbon=np.array([0.2, 0.3, 0.5]),
...     density=np.array([1200.0, 1200.0, 1200.0]),
... )
>>> isinstance(result.partition_coefficients, np.ndarray)
True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L116-L261)

### particula.equilibria.LiquidVaporPartitioningStrategy.water_activity

*attribute* · *instance attribute*

```python
water_activity = float(water_activity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L145-L145)

### particula.equilibria.LiquidVaporPartitioningStrategy.__init__

*method*

```python
def __init__(water_activity: float = 0.5) -> None
```

Initialize the strategy with the requested water activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L139-L145)

### particula.equilibria.LiquidVaporPartitioningStrategy._convert_to_result

*method*

```python
def _convert_to_result(alpha: PhaseOutput, beta: Optional[PhaseOutput], system: SystemOutput, error_value: float) -> EquilibriumResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L189-L230)

### particula.equilibria.LiquidVaporPartitioningStrategy._extract_error

*method* · *staticmethod*

```python
def _extract_error(system: SystemOutput, fit_result: OptimizeResult) -> float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L252-L261)

### particula.equilibria.LiquidVaporPartitioningStrategy._is_valid_phase_output

*method* · *staticmethod*

```python
def _is_valid_phase_output(phase: Sequence[object]) -> bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L248-L250)

### particula.equilibria.LiquidVaporPartitioningStrategy._to_phase

*method*

```python
def _to_phase(phase: PhaseOutput) -> PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L232-L246)

### particula.equilibria.LiquidVaporPartitioningStrategy.solve

*method*

```python
def solve(c_star_j_dry: NDArray[np.float64], concentration_organic_matter: NDArray[np.float64], molar_mass: NDArray[np.float64], oxygen2carbon: NDArray[np.float64], density: NDArray[np.float64], partition_coefficient_guess: Optional[NDArray[np.float64]] = None) -> EquilibriumResult
```

Solve for equilibrium state using liquid-vapor partitioning.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L147-L187)

### particula.equilibria.LiquidVaporPartitioningStrategy.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriaStrategy`

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L111-L113)

## particula.equilibria.PhaseConcentrations

*class* · *dataclass*

```python
class PhaseConcentrations
```

Re-exported from: `particula.equilibria.equilibria_strategies`

Concentrations in a single phase.

**Attributes**

- `species_concentrations` (`NDArray[np.float64]`) — Species concentrations (organic + aqueous) in the phase [µg/m³].
- `water_concentration` (`float`) — Total aqueous concentration in the phase [µg/m³].
- `total_concentration` (`float`) — Total concentration (organic + water) [µg/m³].

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L33-L46)

### particula.equilibria.PhaseConcentrations.species_concentrations

*attribute* · *instance attribute*

```python
species_concentrations: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L44-L44)

### particula.equilibria.PhaseConcentrations.total_concentration

*attribute* · *instance attribute*

```python
total_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L46-L46)

### particula.equilibria.PhaseConcentrations.water_concentration

*attribute* · *instance attribute*

```python
water_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L45-L45)

### particula.equilibria.PhaseConcentrations.__init__

*method*

```python
def __init__(species_concentrations: NDArray[np.float64], water_concentration: float, total_concentration: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)

## particula.equilibria.get_properties_for_liquid_vapor_partitioning

*function*

```python
def get_properties_for_liquid_vapor_partitioning(*args, **kwargs)
```

Legacy property helper for liquid-vapor partitioning.

Emits a :class:`DeprecationWarning` and delegates to the partitioning
implementation. Prefer the strategy interfaces for new code.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/__init__.py#L64-L78)

## particula.equilibria.liquid_vapor_obj_function

*function*

```python
def liquid_vapor_obj_function(*args, **kwargs)
```

Legacy objective function for liquid-vapor partitioning.

Emits a :class:`DeprecationWarning` and delegates to the partitioning
implementation. Prefer the strategy interfaces for new code.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/__init__.py#L81-L93)

## particula.equilibria.liquid_vapor_partitioning

*function*

```python
def liquid_vapor_partitioning(*args, **kwargs)
```

Legacy shortcut for direct liquid-vapor partitioning.

Emits a :class:`DeprecationWarning` and delegates to the partitioning
implementation. Prefer :class:`LiquidVaporPartitioningStrategy` or
:class:`Equilibria` for new code.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/__init__.py#L48-L61)

**Modules**

- `particula.equilibria.equilibria`
- `particula.equilibria.equilibria_builders`
- `particula.equilibria.equilibria_factories`
- `particula.equilibria.equilibria_strategies`
- `particula.equilibria.partitioning`
