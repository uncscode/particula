# Vapor Pressure Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Factories

> Auto-generated documentation for [particula.next.gas.vapor_pressure_factories](https://github.com/uncscode/particula/blob/main/particula/next/gas/vapor_pressure_factories.py) module.

## VaporPressureFactory

[Show source in vapor_pressure_factories.py:20](https://github.com/uncscode/particula/blob/main/particula/next/gas/vapor_pressure_factories.py#L20)

Factory class to create vapor pressure strategy builders

Factory class to create vapor pressure strategy builders for calculating
vapor pressure of gas species.

#### Methods

- `get_builders()` - Returns the mapping of strategy types to builder
instances.
- `get_strategy(strategy_type,` *parameters)* - Gets the strategy instance
for the specified strategy type.
    - `strategy_type` - Type of vapor pressure strategy to use, can be
    'constant', 'antoine', 'clausius_clapeyron', or 'water_buck'.
    parameters(Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - `-` *constant* - constant_vapor_pressure
        - `-` *antoine* - A, B, C
        - `-` *clausius_clapeyron* - A, B, C
        - `-` *water_buck* - No parameters are required.

#### Returns

- `VaporPressureStrategy` - An instance of the specified
    VaporPressureStrategy.

#### Raises

- `ValueError` - If an unknown strategy type is provided.
- `ValueError` - If any required key is missing during check_keys or
    pre_build_check, or if trying to set an invalid parameter.

#### Examples

```python
>>> strategy_is = VaporPressureFactory().get_strategy("constant")
```

#### Signature

```python
class VaporPressureFactory(
    StrategyFactory[
        Union[
            ConstantBuilder, AntoineBuilder, ClausiusClapeyronBuilder, WaterBuckBuilder
        ],
        Union[
            ConstantVaporPressureStrategy,
            AntoineVaporPressureStrategy,
            ClausiusClapeyronStrategy,
            WaterBuckStrategy,
        ],
    ]
): ...
```

#### See also

- [AntoineBuilder](./vapor_pressure_builders.md#antoinebuilder)
- [AntoineVaporPressureStrategy](./vapor_pressure_strategies.md#antoinevaporpressurestrategy)
- [ClausiusClapeyronBuilder](./vapor_pressure_builders.md#clausiusclapeyronbuilder)
- [ClausiusClapeyronStrategy](./vapor_pressure_strategies.md#clausiusclapeyronstrategy)
- [ConstantBuilder](./vapor_pressure_builders.md#constantbuilder)
- [ConstantVaporPressureStrategy](./vapor_pressure_strategies.md#constantvaporpressurestrategy)
- [WaterBuckBuilder](./vapor_pressure_builders.md#waterbuckbuilder)
- [WaterBuckStrategy](./vapor_pressure_strategies.md#waterbuckstrategy)

### VaporPressureFactory().get_builders

[Show source in vapor_pressure_factories.py:68](https://github.com/uncscode/particula/blob/main/particula/next/gas/vapor_pressure_factories.py#L68)

Returns the mapping of strategy types to builder instances.

#### Returns

A dictionary mapping strategy types to builder instances.
    - `-` *constant* - ConstantBuilder
    - `-` *antoine* - AntoineBuilder
    - `-` *clausius_clapeyron* - ClausiusClapeyronBuilder
    - `-` *water_buck* - WaterBuckBuilder

#### Signature

```python
def get_builders(self): ...
```
