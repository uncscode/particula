# Surface Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Surface Factories

> Auto-generated documentation for [particula.next.particles.surface_factories](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_factories.py) module.

## SurfaceFactory

[Show source in surface_factories.py:17](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_factories.py#L17)

Factory class to call and create surface tension strategies.

Factory class to create surface tension strategy builders for
calculating surface tension and the Kelvin effect for species in
particulate phases.

#### Methods

- `get_builders()` - Returns the mapping of strategy types to builder
instances.
- `get_strategy(strategy_type,` *parameters)* - Gets the strategy instance
for the specified strategy type.
    - `strategy_type` - Type of surface tension strategy to use, can be
    'volume', 'mass', or 'molar'.
    parameters(Dict[str, Any], optional): Parameters required for the
    builder, dependent on the chosen strategy type.
        - `volume` - density, surface_tension
        - `mass` - density, surface_tension
        - `molar` - molar_mass, density, surface_tension

#### Returns

- `SurfaceStrategy` - An instance of the specified SurfaceStrategy.

#### Raises

- `ValueError` - If an unknown strategy type is provided.
- `ValueError` - If any required key is missing during check_keys or
    pre_build_check, or if trying to set an invalid parameter.

#### Signature

```python
class SurfaceFactory(
    StrategyFactory[
        Union[
            SurfaceStrategyVolumeBuilder,
            SurfaceStrategyMassBuilder,
            SurfaceStrategyMolarBuilder,
        ],
        Union[SurfaceStrategyVolume, SurfaceStrategyMass, SurfaceStrategyMolar],
    ]
): ...
```

#### See also

- [SurfaceStrategyMassBuilder](./surface_builders.md#surfacestrategymassbuilder)
- [SurfaceStrategyMass](./surface_strategies.md#surfacestrategymass)
- [SurfaceStrategyMolarBuilder](./surface_builders.md#surfacestrategymolarbuilder)
- [SurfaceStrategyMolar](./surface_strategies.md#surfacestrategymolar)
- [SurfaceStrategyVolumeBuilder](./surface_builders.md#surfacestrategyvolumebuilder)
- [SurfaceStrategyVolume](./surface_strategies.md#surfacestrategyvolume)

### SurfaceFactory().get_builders

[Show source in surface_factories.py:58](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_factories.py#L58)

Returns the mapping of strategy types to builder instances.

#### Returns

- `Dict[str,` *BuilderT]* - A dictionary mapping strategy types to
builder instances.
    - `volume` - SurfaceStrategyVolumeBuilder
    - `mass` - SurfaceStrategyMassBuilder
    - `molar` - SurfaceStrategyMolarBuilder

#### Signature

```python
def get_builders(self): ...
```
