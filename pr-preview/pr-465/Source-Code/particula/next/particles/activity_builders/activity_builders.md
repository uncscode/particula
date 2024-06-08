# Activity Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Activity Builders

> Auto-generated documentation for [particula.next.particles.activity_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py) module.

## IdealActivityMassBuilder

[Show source in activity_builders.py:20](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L20)

Builder class for IdealActivityMass objects. No additional parameters.

#### Methods

- `build()` - Validate and return the IdealActivityMass object.

#### Signature

```python
class IdealActivityMassBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### IdealActivityMassBuilder().build

[Show source in activity_builders.py:31](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L31)

Validate and return the IdealActivityMass object.

#### Returns

- `IdealActivityMass` - The validated IdealActivityMass object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## IdealActivityMolarBuilder

[Show source in activity_builders.py:40](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L40)

Builder class for IdealActivityMolar objects.

#### Methods

- `set_molar_mass(molar_mass,` *molar_mass_units)* - Set the molar mass of the
    particle in kg/mol. Default units are 'kg/mol'.
- `set_parameters(params)` - Set the parameters of the IdealActivityMolar
    object from a dictionary including optional units.
- `build()` - Validate and return the IdealActivityMolar object.

#### Signature

```python
class IdealActivityMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### IdealActivityMolarBuilder().build

[Show source in activity_builders.py:59](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L59)

Validate and return the IdealActivityMolar object.

#### Returns

- `IdealActivityMolar` - The validated IdealActivityMolar object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## KappaParameterActivityBuilder

[Show source in activity_builders.py:69](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L69)

Builder class for KappaParameterActivity objects.

#### Methods

- `set_kappa(kappa)` - Set the kappa parameter for the activity calculation.
- `set_density(density,density_units)` - Set the density of the species in
    kg/m^3. Default units are 'kg/m^3'.
- `set_molar_mass(molar_mass,molar_mass_units)` - Set the molar mass of the
    species in kg/mol. Default units are 'kg/mol'.
- `set_water_index(water_index)` - Set the array index of the species.
- `set_parameters(dict)` - Set the parameters of the KappaParameterActivity
    object from a dictionary including optional units.
- `build()` - Validate and return the KappaParameterActivity object.

#### Signature

```python
class KappaParameterActivityBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### KappaParameterActivityBuilder().build

[Show source in activity_builders.py:137](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L137)

Validate and return the KappaParameterActivity object.

#### Returns

- `KappaParameterActivity` - The validated KappaParameterActivity object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)

### KappaParameterActivityBuilder().set_kappa

[Show source in activity_builders.py:97](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L97)

Set the kappa parameter for the activity calculation.

#### Arguments

- `kappa` - The kappa parameter for the activity calculation.
- `kappa_units` - Not used. (for interface consistency)

#### Signature

```python
def set_kappa(
    self, kappa: Union[float, NDArray[np.float_]], kappa_units: Optional[str] = None
): ...
```

### KappaParameterActivityBuilder().set_water_index

[Show source in activity_builders.py:117](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L117)

Set the array index of the species.

#### Arguments

- `water_index` - The array index of the species.
- `water_index_units` - Not used. (for interface consistency)

#### Signature

```python
def set_water_index(self, water_index: int, water_index_units: Optional[str] = None): ...
```
