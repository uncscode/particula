# particula.gas.latent_heat_factories

*module*

Factories for latent heat strategies.

This module exposes :class:`LatentHeatFactory`, which builds constant,
linear, and power-law latent heat strategies via their respective builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L1-L1)

## particula.gas.latent_heat_factories.LatentHeatBuilderType

*attribute* · *module attribute*

```python
LatentHeatBuilderType = ConstantLatentHeatBuilder | LinearLatentHeatBuilder | PowerLawLatentHeatBuilder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L19-L23)

## particula.gas.latent_heat_factories.LatentHeatStrategyType

*attribute* · *module attribute*

```python
LatentHeatStrategyType = ConstantLatentHeat | LinearLatentHeat | PowerLawLatentHeat
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L25-L27)

## particula.gas.latent_heat_factories.LatentHeatFactory

*class*

```python
class LatentHeatFactory(StrategyFactoryABC[LatentHeatBuilderType, LatentHeatStrategyType])
```

Bases: ``

Factory for latent heat strategies.

The factory builds constant, linear, or power-law latent heat strategies
by delegating to the corresponding builder class.

**Examples**

```pycon
>>> from particula.gas import LatentHeatFactory
>>> factory = LatentHeatFactory()
>>> factory.get_strategy(
...     "constant",
...     {
...         "latent_heat_ref": 2.26e6,
...         "latent_heat_ref_units": "J/kg",
...     },
... ).latent_heat(300.0)
2260000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L30-L67)

### particula.gas.latent_heat_factories.LatentHeatFactory.get_builders

*method*

```python
def get_builders() -> dict[str, LatentHeatBuilderType]
```

Return latent heat builders keyed by strategy name.

**Returns**

- (`dict[str, LatentHeatBuilderType]`) — Builder instances keyed by strategy type.

**Examples**

```pycon
>>> from particula.gas import LatentHeatFactory
>>> builders = LatentHeatFactory().get_builders()
>>> isinstance(builders["constant"], ConstantLatentHeatBuilder)
True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L51-L67)
