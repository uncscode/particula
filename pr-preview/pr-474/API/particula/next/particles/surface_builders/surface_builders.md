# Surface Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Surface Builders

> Auto-generated documentation for [particula.next.particles.surface_builders](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py) module.

## SurfaceStrategyMassBuilder

[Show source in surface_builders.py:65](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L65)

Builder class for SurfaceStrategyMass objects.

#### Methods

- `set_surface_tension(surface_tension,` *surface_tension_units)* - Set the
    surface tension of the particle in N/m. Default units are 'N/m'.
- `set_density(density,` *density_units)* - Set the density of the particle in
    kg/m^3. Default units are 'kg/m^3'.
- `set_parameters(params)` - Set the parameters of the SurfaceStrategyMass
    object from a dictionary including optional units.
- `build()` - Validate and return the SurfaceStrategyMass object.

#### Signature

```python
class SurfaceStrategyMassBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyMassBuilder().build

[Show source in surface_builders.py:88](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L88)

Validate and return the SurfaceStrategyMass object.

#### Returns

- `SurfaceStrategyMass` - Instance of the SurfaceStrategyMass object.

#### Signature

```python
def build(self) -> SurfaceStrategyMass: ...
```

#### See also

- [SurfaceStrategyMass](./surface_strategies.md#surfacestrategymass)



## SurfaceStrategyMolarBuilder

[Show source in surface_builders.py:24](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L24)

Builder class for SurfaceStrategyMolar objects.

#### Methods

- `set_surface_tension(surface_tension,` *surface_tension_units)* - Set the
    surface tension of the particle in N/m. Default units are 'N/m'.
- `set_density(density,` *density_units)* - Set the density of the particle in
    kg/m^3. Default units are 'kg/m^3'.
- `set_molar_mass(molar_mass,` *molar_mass_units)* - Set the molar mass of the
    particle in kg/mol. Default units are 'kg/mol'.
- `set_parameters(params)` - Set the parameters of the SurfaceStrategyMolar
    object from a dictionary including optional units.
- `build()` - Validate and return the SurfaceStrategyMolar object.

#### Signature

```python
class SurfaceStrategyMolarBuilder(
    BuilderABC, BuilderDensityMixin, BuilderSurfaceTensionMixin, BuilderMolarMassMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyMolarBuilder().build

[Show source in surface_builders.py:51](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L51)

Validate and return the SurfaceStrategyMass object.

#### Returns

- `SurfaceStrategyMolar` - Instance of the SurfaceStrategyMolar object.

#### Signature

```python
def build(self) -> SurfaceStrategyMolar: ...
```

#### See also

- [SurfaceStrategyMolar](./surface_strategies.md#surfacestrategymolar)



## SurfaceStrategyVolumeBuilder

[Show source in surface_builders.py:101](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L101)

Builder class for SurfaceStrategyVolume objects.

#### Methods

- `set_surface_tension(surface_tension,` *surface_tension_units)* - Set the
    surface tension of the particle in N/m. Default units are 'N/m'.
- `set_density(density,` *density_units)* - Set the density of the particle in
    kg/m^3. Default units are 'kg/m^3'.
- `set_parameters(params)` - Set the parameters of the SurfaceStrategyVolume
    object from a dictionary including optional units.
- `build()` - Validate and return the SurfaceStrategyVolume object.

#### Signature

```python
class SurfaceStrategyVolumeBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyVolumeBuilder().build

[Show source in surface_builders.py:124](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_builders.py#L124)

Validate and return the SurfaceStrategyVolume object.

#### Returns

- `SurfaceStrategyVolume` - Instance of the SurfaceStrategyVolume
    object.

#### Signature

```python
def build(self) -> SurfaceStrategyVolume: ...
```

#### See also

- [SurfaceStrategyVolume](./surface_strategies.md#surfacestrategyvolume)
