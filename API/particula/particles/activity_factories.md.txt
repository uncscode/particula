# particula.particles.activity_factories

*module*

Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L1-L1)

## particula.particles.activity_factories.ActivityFactory

*class*

```python
class ActivityFactory(StrategyFactoryABC[Union[ActivityIdealMassBuilder, ActivityIdealMolarBuilder, ActivityKappaParameterBuilder, ActivityNonIdealBinaryBuilder, _AIOMFACBuilder], ActivityStrategy])
```

Bases: ``

Factory for creating activity strategy builders for liquid mixtures.

This class supports strategies to compute activity and partial pressures
of species based on different thermodynamic models:

- **mass_ideal**: Raoult's Law based on mass fractions
- **molar_ideal**: Raoult's Law based on mole fractions
- **kappa_parameter** / **kappa**: Kappa hygroscopic parameter
  (Petters 2007)
- **non_ideal_binary** / **binary_non_ideal**: BAT model for non-ideal
  organic–water mixtures (Gorkowski 2019)

**Functions**

- `get_builders` — Provides a mapping from strategy type to builder.
- `get_strategy` — Validates inputs and returns a strategy instance for the
specified strategy type.

**Raises**

- `ValueError` — If the strategy type is unknown or if required parameters
are missing or invalid.

**Examples**

```py title="Factory Usage Example"
import particula as par
factory = par.particles.ActivityFactory()

# Ideal mass activity
strategy = factory.get_strategy("mass_ideal")

# Non-ideal binary (BAT model)
params = {
    "molar_mass": 0.200,
    "oxygen2carbon": 0.4,
    "density": 1400.0,
}
bat_strategy = factory.get_strategy("non_ideal_binary", params)
# bat_strategy.get_name() -> "ActivityNonIdealBinary"
```

> **References**
>
> - Petters, M. D., & Kreidenweis, S. M. (2007).
>   DOI:10.5194/acp-7-1961-2007
> - Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
>   DOI:10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L50-L140)

### particula.particles.activity_factories.ActivityFactory.get_builders

*method*

```python
def get_builders()
```

Return a mapping of strategy types to their corresponding builders.

**Returns**

- `result` (`dict`) — A dictionary mapping the activity strategy type
- `result` (`dict`) — to a builder instance.

> **Supported types**
>
> - "aiomfac": Optional direct CPU liquid-mixture provider
> - "mass_ideal": ActivityIdealMassBuilder (Raoult's Law, mass)
> - "molar_ideal": ActivityIdealMolarBuilder (Raoult's Law, mole)
> - "kappa_parameter": ActivityKappaParameterBuilder (kappa model)
> - "kappa": Alias for "kappa_parameter"
> - "non_ideal_binary": ActivityNonIdealBinaryBuilder (BAT model)
> - "binary_non_ideal": Alias for "non_ideal_binary"

**Examples**

```py title="Builders Retrieval Example"
factory = ActivityFactory()
builder_map = factory.get_builders()
non_ideal_builder = builder_map["non_ideal_binary"]
# non_ideal_builder -> ActivityNonIdealBinaryBuilder()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L108-L140)

## particula.particles.activity_factories._AIOMFACBuilder

*class*

```python
class _AIOMFACBuilder
```

Load the optional provider on explicit selection.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L19-L45)

### particula.particles.activity_factories._AIOMFACBuilder.parameters

*attribute* · *instance attribute*

```python
parameters = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L23-L23)

### particula.particles.activity_factories._AIOMFACBuilder.__init__

*method*

```python
def __init__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L22-L23)

### particula.particles.activity_factories._AIOMFACBuilder.build

*method*

```python
def build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L28-L45)

### particula.particles.activity_factories._AIOMFACBuilder.set_parameters

*method*

```python
def set_parameters(parameters)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L25-L26)
