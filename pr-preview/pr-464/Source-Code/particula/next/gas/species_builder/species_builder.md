# Species Builder

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species Builder

> Auto-generated documentation for [particula.next.gas.species_builder](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py) module.

## GasSpeciesBuilder

[Show source in species_builder.py:24](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py#L24)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Attributes

----------
- name (str): The name of the gas species.
- molar_mass (float): The molar mass of the gas species in kg/mol.
- vapor_pressure_strategy (VaporPressureStrategy): The vapor pressure
    strategy for the gas species.
- condensable (bool): Whether the gas species is condensable.
- concentration (float): The concentration of the gas species in the
    mixture, in kg/m^3.

#### Methods

-------
- `-` *set_name(name)* - Set the name of the gas species.
- set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
    gas species in kg/mol.
- `-` *set_vapor_pressure_strategy(strategy)* - Set the vapor pressure strategy
    for the gas species.
- `-` *set_condensable(condensable)* - Set the condensable bool of the gas
    species.
- set_concentration(concentration, concentration_units): Set the
    concentration of the gas species in the mixture, in kg/m^3.
- `-` *set_parameters(params)* - Set the parameters of the GasSpecies object from
    a dictionary including optional units.
- `-` *build()* - Validate and return the GasSpecies object.

#### Raises

------
- `-` *ValueError* - If any required key is missing. During check_keys and
    pre_build_check. Or if trying to set an invalid parameter.
- `-` *Warning* - If using default units for any parameter.

#### Signature

```python
class GasSpeciesBuilder(BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderConcentrationMixin](../abc_builder.md#builderconcentrationmixin)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### GasSpeciesBuilder().build

[Show source in species_builder.py:98](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py#L98)

Validate and return the GasSpecies object.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### GasSpeciesBuilder().set_condensable

[Show source in species_builder.py:90](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py#L90)

Set the condensable bool of the gas species.

#### Signature

```python
def set_condensable(self, condensable: Union[bool, NDArray[np.bool_]]): ...
```

### GasSpeciesBuilder().set_name

[Show source in species_builder.py:77](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py#L77)

Set the name of the gas species.

#### Signature

```python
def set_name(self, name: Union[str, NDArray[np.str_]]): ...
```

### GasSpeciesBuilder().set_vapor_pressure_strategy

[Show source in species_builder.py:82](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builder.py#L82)

Set the vapor pressure strategy for the gas species.

#### Signature

```python
def set_vapor_pressure_strategy(
    self, strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
): ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)
