# particula.equilibria.equilibria_factories

*module*

Factory for creating equilibria strategies by name.

Provides a registry-based factory that returns equilibria strategies using
co-located builders. Keys are case-insensitive and follow the patterns used
across particula factories.

**Examples**

```pycon
>>> factory = EquilibriaFactory()
>>> strategy = factory.get_strategy("liquid_vapor")
>>> strategy.water_activity
0.5
```

```pycon
>>> strategy = factory.get_strategy(
...     "liquid_vapor", parameters={"water_activity": 0.8}
... )
>>> strategy.water_activity
0.8
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L1-L1)

## particula.equilibria.equilibria_factories.BuilderType

*attribute* · *module attribute*

```python
BuilderType = LiquidVaporPartitioningBuilder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L30-L30)

## particula.equilibria.equilibria_factories.EquilibriaFactory

*class*

```python
class EquilibriaFactory(StrategyFactoryABC[BuilderType, EquilibriaStrategy])
```

Bases: ``

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

### particula.equilibria.equilibria_factories.EquilibriaFactory.get_builders

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

### particula.equilibria.equilibria_factories.EquilibriaFactory.get_strategy

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
