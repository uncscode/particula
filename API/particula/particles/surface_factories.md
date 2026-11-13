# particula.particles.surface_factories

*module*

Surface tension strategies factory.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py#L1-L1)

## particula.particles.surface_factories.SurfaceFactory

*class*

```python
class SurfaceFactory(StrategyFactoryABC[Union[SurfaceStrategyVolumeBuilder, SurfaceStrategyMassBuilder, SurfaceStrategyMolarBuilder], Union[SurfaceStrategyVolume, SurfaceStrategyMass, SurfaceStrategyMolar]])
```

Bases: ``

Factory for creating surface tension strategy builders.

Creates builder instances for volume-, mass-, or molar-based
surface tension strategies. These strategies calculate surface
tension and the Kelvin effect for species in particulate phases.

- get_builders : Return a mapping of strategy types to builder
    instances.
- get_strategy : Return a strategy instance for the specified type
    ('volume', 'mass', or 'molar').

**Returns**

- `result` (`SurfaceStrategy`) — The instance of the requested surface strategy.

**Raises**

- `ValueError` — If an unknown strategy type is provided or if
required parameters are missing during check_keys/pre_build_check.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py#L18-L79)

### particula.particles.surface_factories.SurfaceFactory.get_builders

*method*

```python
def get_builders() -> dict[str, Union[SurfaceStrategyVolumeBuilder, SurfaceStrategyMassBuilder, SurfaceStrategyMolarBuilder]]
```

Return a mapping of strategy types to builder instances.

**Returns**

- (`dict[str, Union[SurfaceStrategyVolumeBuilder, SurfaceStrategyMassBuilder, SurfaceStrategyMolarBuilder]]`) — Keys are 'volume', 'mass', or 'molar', each mapped to the corresponding builder class.

**Examples**

```py title="SurfaceFactory Example"
import particula as par
factory = par.particles.SurfaceFactory()
builders = factory.get_builders()
volume_builder = builders["volume"]
mass_builder = builders["mass"]
molar_builder = builders["molar"]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py#L49-L79)
