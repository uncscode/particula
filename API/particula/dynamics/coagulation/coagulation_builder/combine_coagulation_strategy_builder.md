# particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder

*module*

Combine Coagulation Strategy Builder Module.

Provides a builder for creating `CombineCoagulationStrategy` objects,
which can merge multiple sub-strategies (e.g., Brownian, Turbulent)
into a single, combined coagulation approach. This allows flexible,
modular composition of different coagulation mechanisms.

Creates a combined coagulation strategy from multiple sub-strategies.

> **This builder follows standard usage**
>
> builder.set_strategies([...]).build()

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L26-L26)

## particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder

*class*

```python
class CombineCoagulationStrategyBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder for a combined coagulation strategy.

This class constructs a `CombineCoagulationStrategy` from multiple
sub-strategies (instances of `CoagulationStrategyABC`), enabling
advanced modeling scenarios where different coagulation mechanisms
act concurrently. Each sub-strategy's rate calculations are effectively
merged to act on the same particle population.

**Attributes**

- `strategies` — List of `CoagulationStrategyABC` objects to combine.

Methods:
- set_strategies : Set the list of coagulation strategies to combine.
- build : Create and return the combined coagulation strategy.

**Examples**

```py title="Combine Coagulation Strategy Example"
import particula as par
builder = par.dynamics.CombineCoagulationStrategyBuilder()
builder.set_strategies([brownian_strategy, turbulent_strategy])
combined_strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L29-L115)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.strategies

*attribute* · *instance attribute*

```python
strategies = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L67-L67)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.__init__

*method*

```python
def __init__()
```

Initialize the CombineCoagulationStrategyBuilder.

**Returns**

- `result` (`None`) — No return value.

> **Note**
>
> The only required parameter is 'strategies'. Attempting to
> build without setting it triggers an error. Use `set_strategies`
> before calling `build`.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L54-L67)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.build

*method*

```python
def build() -> CombineCoagulationStrategy
```

Builds and returns the combined coagulation strategy.

**Returns**

- `CombineCoagulationStrategy` (`CombineCoagulationStrategy`) — A strategy that combines all the previously added sub-strategies.

**Examples**

```py title="Build Example with CombineCoagulationStrategy"
combined_strategy = builder.build()
# Now you can use `combined_strategy.kernel(...)` to calculate
# combined coagulation effects from each sub-strategy.
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L99-L115)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.set_strategies

*method*

```python
def set_strategies(strategies: List[CoagulationStrategyABC], strategies_units: Optional[str] = None)
```

Sets a list of CoagulationStrategyABC objects to be combined.

**Parameters**

- `strategies` (`List[CoagulationStrategyABC]`) — A list of coagulation strategies to be combined.
- `strategies_units` (`Optional[str]`) (default: `None`) — For interface consistency, not used.

**Examples**

```py title="Set Strategies Example"
builder = CombineCoagulationStrategyBuilder()
builder.set_strategies([brownian_strategy, turbulent_strategy])
```

**Returns**

- `result` (`CombineCoagulationStrategyBuilder`) — This instance for method chaining. The builder instance, for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L69-L97)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder.CombineCoagulationStrategyBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)
