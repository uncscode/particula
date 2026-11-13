# particula.gas.vapor_pressure_factories

*module*

Factory module to create a concrete VaporPressureStrategy object using
builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L1-L1)

## particula.gas.vapor_pressure_factories.VaporPressureFactory

*class*

```python
class VaporPressureFactory(StrategyFactoryABC[Union[ConstantVaporPressureBuilder, AntoineVaporPressureBuilder, ClausiusClapeyronVaporPressureBuilder, SaturationConcentrationVaporPressureBuilder, WaterBuckVaporPressureBuilder, TableVaporPressureBuilder], Union[ConstantVaporPressureStrategy, AntoineVaporPressureStrategy, ClausiusClapeyronStrategy, WaterBuckStrategy, TableVaporPressureStrategy]])
```

Bases: ``

Factory class to create vapor pressure strategy
builders.

This class provides a way to generate multiple vapor pressure calculation
strategies (e.g., constant, Antoine, Clausius-Clapeyron, or Water Buck) by
commissioning the appropriate builder. It is useful for scenarios requiring
a flexible way to switch or extend vapor pressure calculation methods.

- get_builders : Returns the mapping of strategy types to builder
  instances.
- get_strategy : Returns the selected vapor pressure strategy,
  given a strategy type and parameters.

**Examples**

```py title="Example VaporPressureFactory usage"
import particula as par

factory = par.gas.VaporPressureFactory()
# Create a constant vapor pressure strategy:
strategy = factory.get_strategy(
    "constant", {"constant_vapor_pressure": 101325.0}
)
# strategy is an instance of ConstantVaporPressureStrategy
```

> **References**
>
> - "Vapor Pressure,"
> [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L25-L104)

### particula.gas.vapor_pressure_factories.VaporPressureFactory.get_builders

*method*

```python
def get_builders()
```

Return a dictionary mapping strategy types to builder instances.

**Returns**

- `result` (`dict`) — Available builders. - "constant": ConstantVaporPressureBuilder - "antoine": AntoineVaporPressureBuilder - "clausius_clapeyron": ClausiusClapeyronVaporPressureBuilder - "saturation_concentration": SaturationConcentration   VaporPressureBuilder - "water_buck": WaterBuckVaporPressureBuilder

**Examples**

```py
import particula as par
builders_dict = par.gas.VaporPressureFactory().get_builders()
builder = builders_dict["constant"]
# builder is an instance of ConstantBuilder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L75-L104)
