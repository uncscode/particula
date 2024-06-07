# Distribution Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Factories

> Auto-generated documentation for [particula.next.particles.distribution_factories](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_factories.py) module.

## DistributionFactory

[Show source in distribution_factories.py:15](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_factories.py#L15)

Factory class to create distribution strategy builders for
calculating particle distributions based on the specified
representation type.

Methods
-------
- get_builders(): Returns the mapping of strategy types to builder
instances.
- get_strategy(strategy_type, parameters): Gets the strategy instance
for the specified strategy type.
    - strategy_type: Type of distribution strategy to use, can be
    'mass_based_moving_bin', 'radii_based_moving_bin', or
    'speciated_mass_moving_bin'.
    - parameters(Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - mass_based_moving_bin: None
        - radii_based_moving_bin: None
        - speciated_mass_moving_bin: None

#### Returns

--------
- `-` *DistributionStrategy* - An instance of the specified DistributionStrategy.

#### Raises

-------
- `-` *ValueError* - If an unknown strategy type is provided.
- `-` *ValueError* - If any required key is missing during check_keys or
    pre_build_check, or if trying to set an invalid parameter.

#### Signature

```python
class DistributionFactory(
    StrategyFactory[
        Union[
            MassBasedMovingBinBuilder,
            RadiiBasedMovingBinBuilder,
            SpeciatedMassMovingBinBuilder,
        ],
        Union[MassBasedMovingBin, RadiiBasedMovingBin, SpeciatedMassMovingBin],
    ]
): ...
```

#### See also

- [MassBasedMovingBinBuilder](./distribution_builders.md#massbasedmovingbinbuilder)
- [MassBasedMovingBin](./distribution_strategies.md#massbasedmovingbin)
- [RadiiBasedMovingBinBuilder](./distribution_builders.md#radiibasedmovingbinbuilder)
- [RadiiBasedMovingBin](./distribution_strategies.md#radiibasedmovingbin)
- [SpeciatedMassMovingBinBuilder](./distribution_builders.md#speciatedmassmovingbinbuilder)
- [SpeciatedMassMovingBin](./distribution_strategies.md#speciatedmassmovingbin)

### DistributionFactory().get_builders

[Show source in distribution_factories.py:59](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_factories.py#L59)

Returns the mapping of strategy types to builder instances.

#### Returns

--------
- Dict[str, BuilderABC]: Mapping of strategy types to builder
instances.
    - `-` *'mass_based_moving_bin'* - MassBasedMovingBinBuilder
    - `-` *'radii_based_moving_bin'* - RadiiBasedMovingBinBuilder
    - `-` *'speciated_mass_moving_bin'* - SpeciatedMassMovingBinBuilder

#### Signature

```python
def get_builders(self): ...
```
