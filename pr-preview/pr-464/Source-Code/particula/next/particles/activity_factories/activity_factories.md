# Activity Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Activity Factories

> Auto-generated documentation for [particula.next.particles.activity_factories](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_factories.py) module.

## ActivityFactory

[Show source in activity_factories.py:19](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_factories.py#L19)

Factory class to create activity strategy builders

Factory class to create activity strategy builders for calculating
activity and partial pressure of species in a mixture of liquids.

#### Methods

- `get_builders()` - Returns the mapping of strategy types to builder
instances.
- `get_strategy(strategy_type,` *parameters)* - Gets the strategy instance
for the specified strategy type.
    - `strategy_type` - Type of activity strategy to use, can be
    'mass_ideal' (default), 'molar_ideal', or 'kappa_parameter'.
    parameters(Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - `-` *mass_ideal* - No parameters are required.
        - `-` *molar_ideal* - molar_mass
        - kappa | kappa_parameter: kappa, density, molar_mass,
        water_index

#### Returns

- `ActivityStrategy` - An instance of the specified ActivityStrategy.

#### Raises

- `ValueError` - If an unknown strategy type is provided.
- `ValueError` - If any required key is missing during check_keys or
    pre_build_check, or if trying to set an invalid parameter.

#### Examples

```python
>>> strategy_is = ActivityFactory().get_strategy("mass_ideal")
```

#### Signature

```python
class ActivityFactory(
    StrategyFactory[
        Union[
            IdealActivityMassBuilder,
            IdealActivityMolarBuilder,
            KappaParameterActivityBuilder,
        ],
        Union[IdealActivityMass, IdealActivityMolar, KappaParameterActivity],
    ]
): ...
```

#### See also

- [IdealActivityMassBuilder](./activity_builders.md#idealactivitymassbuilder)
- [IdealActivityMass](./activity_strategies.md#idealactivitymass)
- [IdealActivityMolarBuilder](./activity_builders.md#idealactivitymolarbuilder)
- [IdealActivityMolar](./activity_strategies.md#idealactivitymolar)
- [KappaParameterActivityBuilder](./activity_builders.md#kappaparameteractivitybuilder)
- [KappaParameterActivity](./activity_strategies.md#kappaparameteractivity)

### ActivityFactory().get_builders

[Show source in activity_factories.py:63](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_factories.py#L63)

Returns the mapping of strategy types to builder instances.

#### Returns

- `Dict[str,` *Any]* - A dictionary mapping strategy types to builder
    instances.
    - `-` *mass_ideal* - IdealActivityMassBuilder
    - `-` *molar_ideal* - IdealActivityMolarBuilder
    - `-` *kappa_parameter* - KappaParameterActivityBuilder

#### Signature

```python
def get_builders(self): ...
```
