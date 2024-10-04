# Representation Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Factories

> Auto-generated documentation for [particula.next.particles.representation_factories](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_factories.py) module.

## ParticleRepresentationFactory

[Show source in representation_factories.py:17](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_factories.py#L17)

Factory class to create particle representation builders.

#### Methods

- `get_builders` - Returns the mapping of strategy types to builder
    instances.
- `get_strategy` - Gets the strategy instance for the specified strategy.
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
            ParticleMassRepresentationBuilder,
            ParticleRadiusRepresentationBuilder,
            PresetParticleRadiusBuilder,
            ResolvedParticleMassRepresentationBuilder,
            PresetResolvedParticleMassBuilder,
        ],
        ParticleRepresentation,
    ]
): ...
```

#### See also

- [ParticleMassRepresentationBuilder](./representation_builders.md#particlemassrepresentationbuilder)
- [ParticleRadiusRepresentationBuilder](./representation_builders.md#particleradiusrepresentationbuilder)
- [ParticleRepresentation](./representation.md#particlerepresentation)
- [PresetParticleRadiusBuilder](./representation_builders.md#presetparticleradiusbuilder)
- [PresetResolvedParticleMassBuilder](./representation_builders.md#presetresolvedparticlemassbuilder)
- [ResolvedParticleMassRepresentationBuilder](./representation_builders.md#resolvedparticlemassrepresentationbuilder)

### ParticleRepresentationFactory().get_builders

[Show source in representation_factories.py:50](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_factories.py#L50)

Returns the mapping of strategy types to builder instances.

#### Returns

- `dict[str,` *Any]* - A dictionary with the strategy types as keys and
the builder instances as values.
- `-` *'mass'* - MassParticleRepresentationBuilder
- `-` *'radius'* - RadiusParticleRepresentationBuilder
- `-` *'preset_radius'* - LimitedRadiusParticleBuilder
- `-` *'resolved_mass'* - ResolvedMassParticleRepresentationBuilder
- `-` *'preset_resolved_mass'* - PresetResolvedMassParticleBuilder

#### Signature

```python
def get_builders(self): ...
```
