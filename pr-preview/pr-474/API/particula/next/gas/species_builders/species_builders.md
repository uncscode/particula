# Species Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species Builders

> Auto-generated documentation for [particula.next.gas.species_builders](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py) module.

## GasSpeciesBuilder

[Show source in species_builders.py:26](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L26)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Attributes

- `name` - The name of the gas species.
- `molar_mass` - The molar mass of the gas species in kg/mol.
- `vapor_pressure_strategy` - The vapor pressure strategy for the
    gas species.
- `condensable` - Whether the gas species is condensable.
- `concentration` - The concentration of the gas species in the
    mixture, in kg/m^3.

#### Methods

- `set_name` - Set the name of the gas species.
- `set_molar_mass` - Set the molar mass of the gas species in kg/mol.
- `set_vapor_pressure_strategy` - Set the vapor pressure strategy
    for the gas species.
- `set_condensable` - Set the condensable bool of the gas species.
- `set_concentration` - Set the concentration of the gas species in the
    mixture, in kg/m^3.
- `set_parameters` - Set the parameters of the GasSpecies object from
    a dictionary including optional units.

#### Raises

- `ValueError` - If any required key is missing. During check_keys and
    pre_build_check. Or if trying to set an invalid parameter.
- `Warning` - If using default units for any parameter.

#### Signature

```python
class GasSpeciesBuilder(BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)

### GasSpeciesBuilder().build

[Show source in species_builders.py:94](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L94)

Validate and return the GasSpecies object.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### GasSpeciesBuilder().set_condensable

[Show source in species_builders.py:86](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L86)

Set the condensable bool of the gas species.

#### Signature

```python
def set_condensable(self, condensable: Union[bool, NDArray[np.bool_]]): ...
```

### GasSpeciesBuilder().set_name

[Show source in species_builders.py:73](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L73)

Set the name of the gas species.

#### Signature

```python
def set_name(self, name: Union[str, NDArray[np.str_]]): ...
```

### GasSpeciesBuilder().set_vapor_pressure_strategy

[Show source in species_builders.py:78](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L78)

Set the vapor pressure strategy for the gas species.

#### Signature

```python
def set_vapor_pressure_strategy(
    self, strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
): ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)



## PresetGasSpeciesBuilder

[Show source in species_builders.py:107](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L107)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Signature

```python
class PresetGasSpeciesBuilder(GasSpeciesBuilder):
    def __init__(self): ...
```

#### See also

- [GasSpeciesBuilder](#gasspeciesbuilder)

### PresetGasSpeciesBuilder().build

[Show source in species_builders.py:125](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_builders.py#L125)

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)
