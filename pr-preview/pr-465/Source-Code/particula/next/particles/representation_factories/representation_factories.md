# Representation Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Factories

> Auto-generated documentation for [particula.next.particles.representation_factories](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_factories.py) module.

## ParticleRepresentationFactory

[Show source in representation_factories.py:15](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_factories.py#L15)

Factory class to create particle representation builders.

#### Methods

- `get_builders` *()* - Returns the mapping of strategy types to builder
instances.
get_strategy (strategy_type, parameters): Gets the strategy instance
for the specified strategy type.
    - `strategy_type` - Type of particle representation strategy to use,
    can be 'radius' (default) or 'mass'.
    - `parameters` - Parameters required for
    the builder

#### Returns

- `ParticleRepresentation` - An instance of the specified
ParticleRepresentation.

#### Raises

- `ValueError` - If an unknown strategy type is provided.
- `ValueError` - If any required key is missing during check_keys or
pre_build_check, or if trying to set an invalid parameter.

#### Signature

```python
class ParticleRepresentationFactory(
    StrategyFactory[
        Union[
            MassParticleRepresentationBuilder,
            RadiusParticleRepresentationBuilder,
            LimitedRadiusParticleBuilder,
        ],
        ParticleRepresentation,
    ]
): ...
```

#### See also

- [LimitedRadiusParticleBuilder](./representation_builders.md#limitedradiusparticlebuilder)
- [MassParticleRepresentationBuilder](./representation_builders.md#massparticlerepresentationbuilder)
- [ParticleRepresentation](./representation.md#particlerepresentation)
- [RadiusParticleRepresentationBuilder](./representation_builders.md#radiusparticlerepresentationbuilder)

### ParticleRepresentationFactory().get_builders

[Show source in representation_factories.py:47](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_factories.py#L47)

Returns the mapping of strategy types to builder instances.

#### Returns

- `dict[str,` *Any]* - A dictionary with the strategy types as keys and
the builder instances as values.
- `-` *'mass'* - MassParticleRepresentationBuilder
- `-` *'radius'* - RadiusParticleRepresentationBuilder

#### Signature

```python
def get_builders(self): ...
```
