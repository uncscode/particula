# Representation Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.next.particles.representation_builders](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py) module.

## ParticleMassRepresentationBuilder

[Show source in representation_builders.py:56](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L56)

General ParticleRepresentation objects with mass-based bins.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `mass` - Set the mass of the particles. Default units are 'kg'.
- `density` - Set the density of the particles. Default units are 'kg/m^3'.
- `concentration` - Set the concentration of the particles.
    Default units are '1/m^3'.
- `charge` - Set the number of charges.

#### Signature

```python
class ParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../builder_mixin.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../builder_mixin.md#builderdistributionstrategymixin)
- [BuilderMassMixin](../builder_mixin.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](../builder_mixin.md#buildersurfacestrategymixin)

### ParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:98](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L98)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## ParticleRadiusRepresentationBuilder

[Show source in representation_builders.py:116](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L116)

General ParticleRepresentation objects with radius-based bins.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `radius` - Set the radius of the particles. Default units are 'm'.
- `density` - Set the density of the particles. Default units are 'kg/m**3'.
- `concentration` - Set the concentration of the particles. Default units
    are '1/m^3'.
- `charge` - Set the number of charges.

#### Signature

```python
class ParticleRadiusRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../builder_mixin.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../builder_mixin.md#builderdistributionstrategymixin)
- [BuilderRadiusMixin](../builder_mixin.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](../builder_mixin.md#buildersurfacestrategymixin)

### ParticleRadiusRepresentationBuilder().build

[Show source in representation_builders.py:158](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L158)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## PresetParticleRadiusBuilder

[Show source in representation_builders.py:176](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L176)

General ParticleRepresentation objects with radius-based bins.

#### Attributes

- `mode` - Set the mode(s) of the distribution.
    Default is np.array([100e-9, 1e-6]) meters.
- `geometric_standard_deviation` - Set the geometric standard deviation(s)
    of the distribution. Default is np.array([1.2, 1.4]).
- `number_concentration` - Set the number concentration of the distribution.
    Default is np.array([1e4x1e6, 1e3x1e6]) particles/m^3.
- `radius_bins` - Set the radius bins of the distribution. Default is
    np.logspace(-9, -4, 250), meters.

#### Signature

```python
class PresetParticleRadiusBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../builder_mixin.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../builder_mixin.md#builderdistributionstrategymixin)
- [BuilderLognormalMixin](../builder_mixin.md#builderlognormalmixin)
- [BuilderRadiusMixin](../builder_mixin.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](../builder_mixin.md#buildersurfacestrategymixin)

### PresetParticleRadiusBuilder().build

[Show source in representation_builders.py:266](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L266)

Validate and return the ParticleRepresentation object.

This will build a distribution of particles with a lognormal size
distribution, before returning the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)

### PresetParticleRadiusBuilder().set_distribution_type

[Show source in representation_builders.py:247](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L247)

Set the distribution type for the particle representation.

#### Arguments

- `distribution_type` - The type of distribution to use.

#### Signature

```python
def set_distribution_type(
    self, distribution_type: str, distribution_type_units: Optional[str] = None
): ...
```

### PresetParticleRadiusBuilder().set_radius_bins

[Show source in representation_builders.py:230](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L230)

Set the radius bins for the distribution

#### Arguments

- `radius_bins` - The radius bins for the distribution.

#### Signature

```python
def set_radius_bins(
    self, radius_bins: NDArray[np.float64], radius_bins_units: str = "m"
): ...
```



## PresetResolvedParticleMassBuilder

[Show source in representation_builders.py:382](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L382)

General ParticleRepresentation objects with particle resolved masses.

This class has preset values for all the attributes, and allows you to
override them as needed. This is useful when you want to quickly
particle representation object with resolved masses.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `mass` - Set the mass of the particles Default
    units are 'kg'.
- `density` - Set the density of the particles.
    Default units are 'kg/m^3'.
- `charge` - Set the number of charges.
- `mode` - Set the mode(s) of the distribution.
    Default is np.array([100e-9, 1e-6]) meters.
- `geometric_standard_deviation` - Set the geometric standard
    deviation(s) of the distribution. Default is np.array([1.2, 1.4]).
- `number_concentration` - Set the number concentration of the
    distribution. Default is np.array([1e4 1e6, 1e3 1e6])
    particles/m^3.
- `particle_resolved_count` - Set the number of resolved particles.

#### Signature

```python
class PresetResolvedParticleMassBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
    BuilderVolumeMixin,
    BuilderParticleResolvedCountMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../builder_mixin.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../builder_mixin.md#builderdistributionstrategymixin)
- [BuilderLognormalMixin](../builder_mixin.md#builderlognormalmixin)
- [BuilderParticleResolvedCountMixin](../builder_mixin.md#builderparticleresolvedcountmixin)
- [BuilderSurfaceStrategyMixin](../builder_mixin.md#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../builder_mixin.md#buildervolumemixin)

### PresetResolvedParticleMassBuilder().build

[Show source in representation_builders.py:450](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L450)

Validate and return the ParticleRepresentation object.

This will build a distribution of particles with a lognormal size
distribution, before returning the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## ResolvedParticleMassRepresentationBuilder

[Show source in representation_builders.py:306](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L306)

Builder class for constructing ParticleRepresentation objects with
resolved masses.

This class allows you to set various attributes for a particle
representation, such as distribution strategy, mass, density, charge,
volume, and more. These attributes are validated and there a no presets.

#### Attributes

- `distribution_strategy` - Set the distribution strategy for particles.
- `activity_strategy` - Set the activity strategy for the particles.
- `surface_strategy` - Set the surface strategy for the particles.
- `mass` - Set the particle mass. Defaults to 'kg'.
- `density` - Set the particle density. Defaults to 'kg/m^3'.
- `charge` - Set the particle charge.
- `volume` - Set the particle volume. Defaults to 'm^3'.

#### Signature

```python
class ResolvedParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderVolumeMixin,
    BuilderMassMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../builder_mixin.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../builder_mixin.md#builderdistributionstrategymixin)
- [BuilderMassMixin](../builder_mixin.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](../builder_mixin.md#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../builder_mixin.md#buildervolumemixin)

### ResolvedParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:353](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation_builders.py#L353)

Validate and return a ParticleRepresentation object.

This method validates all the required attributes and builds a particle
representation with a lognormal size distribution.

#### Returns

- `ParticleRepresentation` - A validated particle representation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)
