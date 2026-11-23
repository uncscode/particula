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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L33-L33)

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
> - "liquid_vapor": ``LiquidVaporPartitioningStrategy`` for BAT or a
>   configured mixture activity model.

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L36-L110)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L59-L73)

### particula.equilibria.equilibria_factories.EquilibriaFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None) -> LiquidVaporPartitioningStrategy
```

Create an equilibria strategy using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Name of the strategy to build. Case-insensitive. Supported: "liquid_vapor".
- `parameters` (`Optional[Dict[str, Any]]`) (default: `None`) — Optional parameter mapping for the builder. Optional keys are water_activity and activity_model. With a mixture model, omitting water_activity closes the water budget.

**Returns**

- (`LiquidVaporPartitioningStrategy`) — Built equilibria strategy instance.

**Raises**

- `ValueError` — If ``strategy_type`` is unknown or parameters are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_factories.py#L75-L110)
