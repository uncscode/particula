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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L30-L30)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L69-L113)

### particula.equilibria.equilibria_strategies.EquilibriaStrategy.get_name

*method*

```python
def get_name() -> str
```

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L111-L113)

### particula.equilibria.equilibria_strategies.EquilibriaStrategy.solve

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L49-L66)

### particula.equilibria.equilibria_strategies.EquilibriumResult.alpha_phase

*attribute* · *instance attribute*

```python
alpha_phase: PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L62-L62)

### particula.equilibria.equilibria_strategies.EquilibriumResult.beta_phase

*attribute* · *instance attribute*

```python
beta_phase: Optional[PhaseConcentrations]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L63-L63)

### particula.equilibria.equilibria_strategies.EquilibriumResult.error

*attribute* · *instance attribute*

```python
error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L66-L66)

### particula.equilibria.equilibria_strategies.EquilibriumResult.partition_coefficients

*attribute* · *instance attribute*

```python
partition_coefficients: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L64-L64)

### particula.equilibria.equilibria_strategies.EquilibriumResult.water_content

*attribute* · *instance attribute*

```python
water_content: tuple[float, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L65-L65)

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

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.water_activity

*attribute* · *instance attribute*

```python
water_activity = float(water_activity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L145-L145)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.__init__

*method*

```python
def __init__(water_activity: float = 0.5) -> None
```

Initialize the strategy with the requested water activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L139-L145)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._convert_to_result

*method*

```python
def _convert_to_result(alpha: PhaseOutput, beta: Optional[PhaseOutput], system: SystemOutput, error_value: float) -> EquilibriumResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L189-L230)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._extract_error

*method* · *staticmethod*

```python
def _extract_error(system: SystemOutput, fit_result: OptimizeResult) -> float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L252-L261)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._is_valid_phase_output

*method* · *staticmethod*

```python
def _is_valid_phase_output(phase: Sequence[object]) -> bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L248-L250)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy._to_phase

*method*

```python
def _to_phase(phase: PhaseOutput) -> PhaseConcentrations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L232-L246)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.solve

*method*

```python
def solve(c_star_j_dry: NDArray[np.float64], concentration_organic_matter: NDArray[np.float64], molar_mass: NDArray[np.float64], oxygen2carbon: NDArray[np.float64], density: NDArray[np.float64], partition_coefficient_guess: Optional[NDArray[np.float64]] = None) -> EquilibriumResult
```

Solve for equilibrium state using liquid-vapor partitioning.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L147-L187)

### particula.equilibria.equilibria_strategies.LiquidVaporPartitioningStrategy.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.equilibria.equilibria_strategies.EquilibriaStrategy`

Return a stable strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L111-L113)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L33-L46)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.species_concentrations

*attribute* · *instance attribute*

```python
species_concentrations: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L44-L44)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.total_concentration

*attribute* · *instance attribute*

```python
total_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L46-L46)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.water_concentration

*attribute* · *instance attribute*

```python
water_concentration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L45-L45)

### particula.equilibria.equilibria_strategies.PhaseConcentrations.__init__

*method*

```python
def __init__(species_concentrations: NDArray[np.float64], water_concentration: float, total_concentration: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_strategies.py#L0-L0)
