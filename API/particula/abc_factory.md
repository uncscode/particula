# particula.abc_factory

*module*

Abstract base class for factories that use builders to create strategies.

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L1-L1)

## particula.abc_factory.BuilderT

*attribute* · *module attribute*

```python
BuilderT = TypeVar('BuilderT')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L10-L10)

## particula.abc_factory.StrategyT

*attribute* · *module attribute*

```python
StrategyT = TypeVar('StrategyT')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L11-L11)

## particula.abc_factory.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L7-L7)

## particula.abc_factory.StrategyFactoryABC

*class*

```python
class StrategyFactoryABC(ABC, Generic[BuilderT, StrategyT])
```

Bases: `ABC`, ``

Abstract base class for strategy factories.

This class provides a generic interface for creating strategy objects
using builder objects.

**Functions**

- `- get_builders` — Returns the mapping of strategy types to builder instances.
- `- get_strategy` — Gets the strategy instance for the specified strategy.

**Examples**

```py title="Simple Usage"
my_factory = SomeSpecificFactory()
strategy_instance = my_factory.get_strategy(
    "example_type", {"param": 123}
)
# strategy_instance is now built using "example_type"
```

References:
- "Factory Method Pattern,"
[Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L14-L110)

### particula.abc_factory.StrategyFactoryABC.get_builders

*method* · *abstract*

```python
def get_builders() -> Dict[str, BuilderT]
```

Retrieve a mapping of strategy types to builder instances.

**Returns**

- `dict` (`Dict[str, BuilderT]`) — A dictionary that maps strategy type names (str) to
- (`Dict[str, BuilderT]`) — builder instances.

**Examples**

```py title="Coagulation Factory Example"
from particula.coagulation_factory import CoagulationFactory

builders = CoagulationFactory().get_builders()
# Example result:
# {
#     "brownian": BrownianCoagulationBuilder(),
#     "charged": ChargedCoagulationBuilder(),
#     "turbulent_shear": TurbulentShearCoagulationBuilder(),
#     "turbulent_dns": TurbulentDNSCoagulationBuilder(),
#     "combine": CombineCoagulationStrategyBuilder(),
# }
```

> **References**
>
> - "Factory Method Pattern,"
> [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L40-L67)

### particula.abc_factory.StrategyFactoryABC.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None) -> StrategyT
```

Create a strategy instance using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Name of the strategy to build.
- `parameters` (`Dict[str, Any]`) (default: `None`) — Dictionary of parameters to configure the chosen builder.

**Returns**

- `StrategyT` (`StrategyT`) — The built strategy object corresponding to the specified type.

**Raises**

- `ValueError` — If the `strategy_type` is unknown, or if any required
parameter is invalid/missing for the chosen builder.

**Examples**

```py title="Strategy Creation Example"
my_factory = SomeStrategyFactory()
my_strategy = my_factory.get_strategy(
    "desired_strategy", {"param_x": 42}
)
# my_strategy is now an instance configured with param_x=42
```

> **References**
>
> - "Factory Method Pattern,"
> [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L69-L110)
