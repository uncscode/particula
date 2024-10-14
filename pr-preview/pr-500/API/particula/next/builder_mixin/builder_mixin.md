# Builder Mixin

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Builder Mixin

> Auto-generated documentation for [particula.next.builder_mixin](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py) module.

## BuilderActivityStrategyMixin

[Show source in builder_mixin.py:373](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L373)

Mixin class for Builder classes to set activity_strategy.

#### Methods

- `set_activity_strategy` - Set the activity_strategy attribute.

#### Signature

```python
class BuilderActivityStrategyMixin:
    def __init__(self): ...
```

### BuilderActivityStrategyMixin().set_activity_strategy

[Show source in builder_mixin.py:383](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L383)

Set the activity strategy of the particle.

#### Arguments

- `activity_strategy` - Activity strategy of the particle.
- `activity_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_activity_strategy(
    self,
    activity_strategy: ActivityStrategy,
    activity_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [ActivityStrategy](particles/activity_strategies.md#activitystrategy)



## BuilderChargeMixin

[Show source in builder_mixin.py:152](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L152)

Mixin class for Builder classes to set charge and charge_units.

#### Methods

- `set_charge` - Set the charge attribute and units.

#### Signature

```python
class BuilderChargeMixin:
    def __init__(self): ...
```

### BuilderChargeMixin().set_charge

[Show source in builder_mixin.py:162](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L162)

Set the number of elemental charges on the particle.

#### Arguments

- `charge` - Charge of the particle [C].
- `charge_units` - Not used. (for interface consistency)

#### Signature

```python
def set_charge(
    self, charge: Union[float, NDArray[np.float64]], charge_units: Optional[str] = None
): ...
```



## BuilderConcentrationMixin

[Show source in builder_mixin.py:113](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L113)

Mixin class for Builder classes to set concentration and
concentration_units.

#### Arguments

- `default_units` - Default units of concentration. Default is *kg/m^3*.

#### Methods

- `set_concentration` - Set the concentration attribute and units.

#### Signature

```python
class BuilderConcentrationMixin:
    def __init__(self, default_units: Optional[str] = "kg/m^3"): ...
```

### BuilderConcentrationMixin().set_concentration

[Show source in builder_mixin.py:128](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L128)

Set the concentration.

#### Arguments

- `concentration` - Concentration in the mixture.
- `concentration_units` - Units of the concentration.
Default is *kg/m^3*.

#### Signature

```python
def set_concentration(
    self,
    concentration: Union[float, NDArray[np.float64]],
    concentration_units: Optional[str] = None,
): ...
```



## BuilderDensityMixin

[Show source in builder_mixin.py:20](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L20)

Mixin class for Builder classes to set density and density_units.

#### Methods

- `set_density` - Set the density attribute and units.

#### Signature

```python
class BuilderDensityMixin:
    def __init__(self): ...
```

### BuilderDensityMixin().set_density

[Show source in builder_mixin.py:30](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L30)

Set the density of the particle in kg/m^3.

#### Arguments

- `density` - Density of the particle.
- `density_units` - Units of the density. Default is *kg/m^3*

#### Signature

```python
def set_density(
    self,
    density: Union[float, NDArray[np.float64]],
    density_units: Optional[str] = "kg/m^3",
): ...
```



## BuilderDistributionStrategyMixin

[Show source in builder_mixin.py:400](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L400)

Mixin class for Builder classes to set distribution_strategy.

#### Methods

- `set_distribution_strategy` - Set the distribution_strategy attribute.

#### Signature

```python
class BuilderDistributionStrategyMixin:
    def __init__(self): ...
```

### BuilderDistributionStrategyMixin().set_distribution_strategy

[Show source in builder_mixin.py:410](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L410)

Set the distribution strategy of the particle.

#### Arguments

- `distribution_strategy` - Distribution strategy of the particle.
- `distribution_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_distribution_strategy(
    self,
    distribution_strategy: DistributionStrategy,
    distribution_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [DistributionStrategy](particles/distribution_strategies.md#distributionstrategy)



## BuilderLognormalMixin

[Show source in builder_mixin.py:429](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L429)

Mixin class for Builder classes to set lognormal distributions.

#### Methods

- `set_mode` - Set the mode attribute and units.
- `set_geometric_standard_deviation` - Set the geometric standard deviation
    attribute and units.
- `set_number_concentration` - Set the number concentration attribute and
    units.

#### Signature

```python
class BuilderLognormalMixin:
    def __init__(self): ...
```

### BuilderLognormalMixin().set_geometric_standard_deviation

[Show source in builder_mixin.py:466](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L466)

Set the geometric standard deviation for the distribution.

#### Arguments

- `geometric_standard_deviation` - The geometric standard deviation for
    the radius.
- `geometric_standard_deviation_units` - Optional, ignored units for
    geometric standard deviation [dimensionless].

#### Raises

- `ValueError` - If geometric standard deviation is negative.

#### Signature

```python
def set_geometric_standard_deviation(
    self,
    geometric_standard_deviation: NDArray[np.float64],
    geometric_standard_deviation_units: Optional[str] = None,
): ...
```

### BuilderLognormalMixin().set_mode

[Show source in builder_mixin.py:445](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L445)

Set the mode for distribution.

#### Arguments

- `mode` - The modes for the radius.
- `mode_units` - The units for the modes, default is 'm'.

#### Raises

- `ValueError` - If mode is negative.

#### Signature

```python
def set_mode(self, mode: NDArray[np.float64], mode_units: str = "m"): ...
```

### BuilderLognormalMixin().set_number_concentration

[Show source in builder_mixin.py:491](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L491)

Set the number concentration for the distribution.

#### Arguments

- `number_concentration` - The number concentration for the radius.
- `number_concentration_units` - The units for the number concentration,
    default is '1/m^3'.

#### Raises

- `ValueError` - If number concentration is negative.

#### Signature

```python
def set_number_concentration(
    self,
    number_concentration: NDArray[np.float64],
    number_concentration_units: str = "1/m^3",
): ...
```



## BuilderMassMixin

[Show source in builder_mixin.py:179](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L179)

Mixin class for Builder classes to set mass and mass_units.

#### Methods

- `set_mass` - Set the mass attribute and units.

#### Signature

```python
class BuilderMassMixin:
    def __init__(self): ...
```

### BuilderMassMixin().set_mass

[Show source in builder_mixin.py:189](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L189)

Set the mass of the particle in kg.

#### Arguments

- `mass` - Mass of the particle.
- `mass_units` - Units of the mass. Default is *kg*.

#### Raises

- `ValueError` - If mass is negative

#### Signature

```python
def set_mass(
    self, mass: Union[float, NDArray[np.float64]], mass_units: Optional[str] = "kg"
): ...
```



## BuilderMolarMassMixin

[Show source in builder_mixin.py:81](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L81)

Mixin class for Builder classes to set molar_mass and molar_mass_units.

#### Methods

- `set_molar_mass` - Set the molar_mass attribute and units.

#### Signature

```python
class BuilderMolarMassMixin:
    def __init__(self): ...
```

### BuilderMolarMassMixin().set_molar_mass

[Show source in builder_mixin.py:91](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L91)

Set the molar mass of the particle in kg/mol.

#### Arguments

-----
- `-` *molar_mass* - Molar mass of the particle.
- `-` *molar_mass_units* - Units of the molar mass. Default is *kg/mol*.

#### Signature

```python
def set_molar_mass(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    molar_mass_units: Optional[str] = "kg/mol",
): ...
```



## BuilderParticleResolvedCountMixin

[Show source in builder_mixin.py:516](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L516)

Mixin class for Builder classes to set particle_resolved_count.

#### Methods

- `set_particle_resolved_count` - Set the number of particles to resolve.

#### Signature

```python
class BuilderParticleResolvedCountMixin:
    def __init__(self): ...
```

### BuilderParticleResolvedCountMixin().set_particle_resolved_count

[Show source in builder_mixin.py:526](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L526)

Set the number of particles to resolve.

#### Arguments

- `particle_resolved_count` - The number of particles to resolve.
- `particle_resolved_count_units` - Ignored units for particle resolved.

#### Raises

- `ValueError` - If particle_resolved_count is negative.

#### Signature

```python
def set_particle_resolved_count(
    self,
    particle_resolved_count: int,
    particle_resolved_count_units: Optional[str] = None,
): ...
```



## BuilderPressureMixin

[Show source in builder_mixin.py:311](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L311)

Mixin class for AtmosphereBuilder to set total pressure.

#### Methods

- `set_pressure` - Set the total pressure attribute and units.

#### Signature

```python
class BuilderPressureMixin:
    def __init__(self): ...
```

### BuilderPressureMixin().set_pressure

[Show source in builder_mixin.py:321](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L321)

Set the total pressure of the atmosphere.

#### Arguments

- `total_pressure` - Total pressure of the gas mixture.
- `pressure_units` - Units of the pressure. Options include
    'Pa', 'kPa', 'MPa', 'psi', 'bar', 'atm'. Default is 'Pa'.

#### Returns

- `AtmosphereBuilderMixin` - This object instance with updated pressure.

#### Raises

- `ValueError` - If the total pressure is below zero.

#### Signature

```python
def set_pressure(
    self, pressure: Union[float, NDArray[np.float64]], pressure_units: str = "Pa"
): ...
```



## BuilderRadiusMixin

[Show source in builder_mixin.py:243](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L243)

Mixin class for Builder classes to set radius and radius_units.

#### Methods

- `set_radius` - Set the radius attribute and units.

#### Signature

```python
class BuilderRadiusMixin:
    def __init__(self): ...
```

### BuilderRadiusMixin().set_radius

[Show source in builder_mixin.py:253](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L253)

Set the radius of the particle in meters.

#### Arguments

- `radius` - Radius of the particle.
- `radius_units` - Units of the radius. Default is *m*.

#### Raises

- `ValueError` - If radius is negative

#### Signature

```python
def set_radius(
    self, radius: Union[float, NDArray[np.float64]], radius_units: Optional[str] = "m"
): ...
```



## BuilderSurfaceStrategyMixin

[Show source in builder_mixin.py:346](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L346)

Mixin class for Builder classes to set surface_strategy.

#### Methods

- `set_surface_strategy` - Set the surface_strategy attribute.

#### Signature

```python
class BuilderSurfaceStrategyMixin:
    def __init__(self): ...
```

### BuilderSurfaceStrategyMixin().set_surface_strategy

[Show source in builder_mixin.py:356](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L356)

Set the surface strategy of the particle.

#### Arguments

- `surface_strategy` - Surface strategy of the particle.
- `surface_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_surface_strategy(
    self, surface_strategy: SurfaceStrategy, surface_strategy_units: Optional[str] = None
): ...
```

#### See also

- [SurfaceStrategy](particles/surface_strategies.md#surfacestrategy)



## BuilderSurfaceTensionMixin

[Show source in builder_mixin.py:49](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L49)

Mixin class for Builder classes to set surface_tension.

#### Methods

-------
    - `set_surface_tension` - Set the surface_tension attribute and units.

#### Signature

```python
class BuilderSurfaceTensionMixin:
    def __init__(self): ...
```

### BuilderSurfaceTensionMixin().set_surface_tension

[Show source in builder_mixin.py:60](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L60)

Set the surface tension of the particle in N/m.

#### Arguments

- `surface_tension` - Surface tension of the particle.
- `surface_tension_units` - Surface tension units. Default is *N/m*.

#### Signature

```python
def set_surface_tension(
    self,
    surface_tension: Union[float, NDArray[np.float64]],
    surface_tension_units: Optional[str] = "N/m",
): ...
```



## BuilderTemperatureMixin

[Show source in builder_mixin.py:275](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L275)

Mixin class for AtmosphereBuilder to set temperature.

#### Methods

- `set_temperature` - Set the temperature attribute and units.

#### Signature

```python
class BuilderTemperatureMixin:
    def __init__(self): ...
```

### BuilderTemperatureMixin().set_temperature

[Show source in builder_mixin.py:285](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L285)

Set the temperature of the atmosphere.

#### Arguments

- `temperature` *float* - Temperature of the gas mixture.
- `temperature_units` *str* - Units of the temperature.
    Options include 'degC', 'degF', 'degR', 'K'. Default is 'K'.

#### Returns

- `AtmosphereBuilderMixin` - This object instance with updated
    temperature.

#### Raises

- `ValueError` - If the converted temperature is below absolute zero.

#### Signature

```python
def set_temperature(self, temperature: float, temperature_units: str = "K"): ...
```



## BuilderVolumeMixin

[Show source in builder_mixin.py:211](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L211)

Mixin class for Builder classes to set volume and volume_units.

#### Methods

- `set_volume` - Set the volume attribute and units.

#### Signature

```python
class BuilderVolumeMixin:
    def __init__(self): ...
```

### BuilderVolumeMixin().set_volume

[Show source in builder_mixin.py:221](https://github.com/uncscode/particula/blob/main/particula/next/builder_mixin.py#L221)

Set the volume in m^3.

#### Arguments

- `volume` - Volume.
- `volume_units` - Units of the volume. Default is *m^3*.

#### Raises

- `ValueError` - If volume is negative

#### Signature

```python
def set_volume(
    self, volume: Union[float, NDArray[np.float64]], volume_units: Optional[str] = "m^3"
): ...
```
