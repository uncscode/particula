# particula.equilibria.equilibria_strategies

*module*

Equilibria strategy abstractions and liquid-vapor implementation.

Provides a Strategy-pattern entry point for equilibria solvers plus
structured dataclasses for results. The initial implementation wraps the
existing liquid-vapor partitioning helpers.

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an
> efficient reduced-complexity model. Atmospheric Chemistry and Physics,
> 19(19), 13383-13407. https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L1-L1)

## particula.equilibria.equilibria_strategies.partitioning

*attribute* · *module attribute*

```python
partitioning = partitioning_module
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L36-L36)

## particula.equilibria.equilibria_strategies.EquilibriaStrategy

*class*

```python
class EquilibriaStrategy(ABC)
```

Bases: `ABC`

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L94-L117)

### particula.equilibria.equilibria_strategies.EquilibriaStrategy.get_name

*method*

```python
def get_name() -> str
```

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L115-L117)

### particula.equilibria.equilibria_strategies.EquilibriaStrategy.solve

*method* · *abstract*

```python
def solve(*args, **kwargs) -> Any
```

Solve for equilibrium with the inputs of the configured model.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L111-L113)

## particula.equilibria.equilibria_strategies.EquilibriumResult

*class* · *dataclass*

```python
class EquilibriumResult
```

Result of an equilibria calculation.

**Attributes**

- `alpha_phase` (`PhaseConcentrations`) — Concentrations in the alpha (water-rich) phase.
- `beta_phase` (`Optional[PhaseConcentrations]`) — Concentrations in the beta (organic-rich) phase. ``None`` when no beta phase is present.
- `partition_coefficients` (`NDArray[np.float64]`) — Species partition coefficients [-].
- `water_content` (`tuple[float, float]`) — Water content for (alpha, beta) phases [µg/m³].
- `error` (`float`) — Optimization convergence error.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L55-L72)

### particula.equilibria.equilibria_strategies.EquilibriumResult.alpha_phase

*attribute* · *instance attribute*

```python
alpha_phase: PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L68-L68)

### particula.equilibria.equilibria_strategies.EquilibriumResult.beta_phase

*attribute* · *instance attribute*

```python
beta_phase: Optional[PhaseConcentrations]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L69-L69)

### particula.equilibria.equilibria_strategies.EquilibriumResult.error

*attribute* · *instance attribute*

```python
error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L72-L72)

### particula.equilibria.equilibria_strategies.EquilibriumResult.partition_coefficients

*attribute* · *instance attribute*

```python
partition_coefficients: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L70-L70)

### particula.equilibria.equilibria_strategies.EquilibriumResult.water_content

*attribute* · *instance attribute*

```python
water_content: tuple[float, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L71-L71)

### particula.equilibria.equilibria_strategies.EquilibriumResult.__init__

*method*

```python
def __init__(alpha_phase: PhaseConcentrations, beta_phase: Optional[PhaseConcentrations], partition_coefficients: NDArray[np.float64], water_content: tuple[float, float], error: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)

## particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy

*class*

```python
class LiquidVaporPartitioningStrategy(EquilibriaStrategy)
```

Bases: `particula.equilibria.equilibria_strategies.EquilibriaStrategy`

Liquid-vapor partitioning equilibrium strategy.

Wraps the partitioning helpers to compute equilibrium between gas and
condensed phases while exposing structured results.

**Parameters**

- `water_activity` (`float | None`) (default: `None`) — Target water activity (0-1 range). Defaults to 0.5.

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L126-L394)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.activity_model

*attribute* · *instance attribute*

```python
activity_model = activity_model
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L165-L165)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.mixture_water_activity

*attribute* · *instance attribute*

```python
mixture_water_activity = water_activity
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L153-L153)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.water_activity

*attribute* · *instance attribute*

```python
water_activity = float(water_activity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L160-L160)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.__init__

*method*

```python
def __init__(water_activity: float | None = None, *, activity_model=None) -> None
```

Initialize the strategy with the requested water activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L149-L165)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._convert_to_result

*method*

```python
def _convert_to_result(alpha: PhaseOutput, beta: Optional[PhaseOutput], system: SystemOutput, error_value: float) -> EquilibriumResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L277-L318)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._extract_error

*method* · *staticmethod*

```python
def _extract_error(system: SystemOutput, fit_result: OptimizeResult) -> float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L385-L394)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._is_valid_phase_output

*method* · *staticmethod*

```python
def _is_valid_phase_output(phase: Sequence[object]) -> bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L381-L383)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._mixture_result

*method*

```python
def _mixture_result(bulk, inventories) -> MixtureEquilibriumResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L336-L379)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._to_phase

*method*

```python
def _to_phase(phase: PhaseOutput) -> PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L320-L334)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.solve

*method*

```python
def solve(c_star_j_dry: NDArray[np.float64] | None = None, concentration_organic_matter: NDArray[np.float64] | None = None, molar_mass: NDArray[np.float64] | None = None, oxygen2carbon: NDArray[np.float64] | None = None, density: NDArray[np.float64] | None = None, partition_coefficient_guess: Optional[NDArray[np.float64]] = None, **mixture_inputs) -> EquilibriumResult | SpeciesEquilibrium
```

Solve for equilibrium state using liquid-vapor partitioning.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L197-L275)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriaStrategy`

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L115-L117)

## particula.equilibria.equilibria_strategies.MixtureEquilibriumResult

*class* · *dataclass*

```python
class MixtureEquilibriumResult(EquilibriumResult)
```

Bases: `particula.equilibria.equilibria_strategies.EquilibriumResult`

Whole-mixture bulk equilibrium with material lanes in declaration order.

Concentrations and water exchange use µg/m³. Relative humidity is a ratio.
The liquids record holds the phase-resolved AIOMFAC chemistry, or None
when the condensed state is dry or the entire inventory is gaseous.
Inspect solid_liquid for crystal and residual ion masses.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L75-L91)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.gas_concentrations

*attribute* · *instance attribute*

```python
gas_concentrations: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L86-L86)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.liquids

*attribute* · *instance attribute*

```python
liquids: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L85-L85)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.phase_component_names

*attribute* · *instance attribute*

```python
phase_component_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L91-L91)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.relative_humidity

*attribute* · *instance attribute*

```python
relative_humidity: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L87-L87)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.solid_liquid

*attribute* · *instance attribute*

```python
solid_liquid: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L90-L90)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.vapor_residual

*attribute* · *instance attribute*

```python
vapor_residual: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L89-L89)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.water_exchanged

*attribute* · *instance attribute*

```python
water_exchanged: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L88-L88)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.alpha_phase

*attribute* · *instance attribute*

```python
alpha_phase: PhaseConcentrations
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriumResult`

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L68-L68)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.beta_phase

*attribute* · *instance attribute*

```python
beta_phase: Optional[PhaseConcentrations]
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriumResult`

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L69-L69)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.error

*attribute* · *instance attribute*

```python
error: float
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriumResult`

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L72-L72)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.partition_coefficients

*attribute* · *instance attribute*

```python
partition_coefficients: NDArray[np.float64]
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriumResult`

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L70-L70)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.water_content

*attribute* · *instance attribute*

```python
water_content: tuple[float, float]
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriumResult`

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L71-L71)

### particula.equilibria.equilibria_strategies.MixtureEquilibriumResult.__init__

*method*

```python
def __init__(alpha_phase: PhaseConcentrations, beta_phase: Optional[PhaseConcentrations], partition_coefficients: NDArray[np.float64], water_content: tuple[float, float], error: float, liquids: object | None, gas_concentrations: NDArray[np.float64], relative_humidity: float, water_exchanged: float, vapor_residual: float, solid_liquid: object | None, phase_component_names: tuple[str, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)

## particula.equilibria.equilibria_strategies.PhaseConcentrations

*class* · *dataclass*

```python
class PhaseConcentrations
```

Concentrations in a single phase.

**Attributes**

- `species_concentrations` (`NDArray[np.float64]`) — Species concentrations (organic + aqueous) in the phase [µg/m³].
- `water_concentration` (`float`) — Total aqueous concentration in the phase [µg/m³].
- `total_concentration` (`float`) — Total concentration (organic + water) [µg/m³].

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L39-L52)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.species_concentrations

*attribute* · *instance attribute*

```python
species_concentrations: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L50-L50)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.total_concentration

*attribute* · *instance attribute*

```python
total_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L52-L52)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.water_concentration

*attribute* · *instance attribute*

```python
water_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L51-L51)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.__init__

*method*

```python
def __init__(species_concentrations: NDArray[np.float64], water_concentration: float, total_concentration: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)

## particula.equilibria.equilibria_strategies._validate_closed_species_reservoir

*function*

```python
def _validate_closed_species_reservoir(strategy: EquilibriaStrategy) -> None
```

Physical species inventories close the gas and condensed water budget.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L120-L123)
