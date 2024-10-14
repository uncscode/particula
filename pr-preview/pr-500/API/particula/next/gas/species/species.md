# Species

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species

> Auto-generated documentation for [particula.next.gas.species](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py) module.

## GasSpecies

[Show source in species.py:17](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L17)

GasSpecies represents an individual or array of gas species with
properties like name, molar mass, vapor pressure, and condensability.

#### Attributes

------------
- name (str): The name of the gas species.
- molar_mass (float): The molar mass of the gas species.
- pure_vapor_pressure_strategy (VaporPressureStrategy): The strategy for
    calculating the pure vapor pressure of the gas species. Can be a single
    strategy or a list of strategies. Default is a constant vapor pressure
    strategy with a vapor pressure of 0.0 Pa.
- condensable (bool): Indicates whether the gas species is condensable.
    Default is True.
- concentration (float): The concentration of the gas species in the
    mixture. Default is 0.0 kg/m^3.

#### Methods

--------
- `-` *get_molar_mass* - Get the molar mass of the gas species.
- `-` *get_condensable* - Check if the gas species is condensable.
- `-` *get_concentration* - Get the concentration of the gas species in the
    mixture.
- `-` *get_pure_vapor_pressure* - Calculate the pure vapor pressure of the gas
    species at a given temperature.
- `-` *get_partial_pressure* - Calculate the partial pressure of the gas species.
- `-` *get_saturation_ratio* - Calculate the saturation ratio of the gas species.
- `-` *get_saturation_concentration* - Calculate the saturation concentration of
    the gas species.
- `-` *add_concentration* - Add concentration to the gas species.

#### Signature

```python
class GasSpecies:
    def __init__(
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
        ] = ConstantVaporPressureStrategy(0.0),
        condensable: Union[bool, NDArray[np.bool_]] = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ) -> None: ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)

### GasSpecies().__len__

[Show source in species.py:70](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L70)

Return the number of gas species.

#### Signature

```python
def __len__(self): ...
```

### GasSpecies().__str__

[Show source in species.py:66](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L66)

Return a string representation of the GasSpecies object.

#### Signature

```python
def __str__(self): ...
```

### GasSpecies().add_concentration

[Show source in species.py:272](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L272)

Add concentration to the gas species.

#### Arguments

- added_concentration (float): The concentration to add to the gas
    species.

#### Signature

```python
def add_concentration(self, added_concentration: Union[float, NDArray[np.float64]]): ...
```

### GasSpecies().get_concentration

[Show source in species.py:101](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L101)

Get the concentration of the gas species in the mixture, in kg/m^3.

#### Returns

- concentration (float or NDArray[np.float64]): The concentration of
    the gas species in the mixture.

#### Signature

```python
def get_concentration(self) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_condensable

[Show source in species.py:93](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L93)

Check if the gas species is condensable or not.

#### Returns

- condensable (bool): True if the gas species is condensable, False
    otherwise.

#### Signature

```python
def get_condensable(self) -> Union[bool, NDArray[np.bool_]]: ...
```

### GasSpecies().get_molar_mass

[Show source in species.py:85](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L85)

Get the molar mass of the gas species in kg/mol.

#### Returns

- molar_mass (float or NDArray[np.float64]): The molar mass of the gas
    species, in kg/mol.

#### Signature

```python
def get_molar_mass(self) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_name

[Show source in species.py:78](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L78)

Get the name of the gas species.

#### Returns

- name (str or NDArray[np.str_]): The name of the gas species.

#### Signature

```python
def get_name(self) -> Union[str, NDArray[np.str_]]: ...
```

### GasSpecies().get_partial_pressure

[Show source in species.py:146](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L146)

Calculate the partial pressure of the gas based on the vapor
pressure strategy. This method accounts for multiple strategies if
assigned and calculates partial pressure for each strategy based on
the corresponding concentration and molar mass.

#### Arguments

- temperature (float or NDArray[np.float64]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- partial_pressure (float or NDArray[np.float64]): Partial pressure
of the gas in Pascals.

#### Raises

- `-` *ValueError* - If the vapor pressure strategy is not set.

#### Signature

```python
def get_partial_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_pure_vapor_pressure

[Show source in species.py:110](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L110)

Calculate the pure vapor pressure of the gas species at a given
temperature in Kelvin.

This method supports both a single strategy or a list of strategies
for calculating vapor pressure.

#### Arguments

- temperature (float or NDArray[np.float64]): The temperature in
Kelvin at which to calculate vapor pressure.

#### Returns

- vapor_pressure (float or NDArray[np.float64]): The calculated pure
vapor pressure in Pascals.

#### Raises

- `ValueError` - If no vapor pressure strategy is set.

#### Signature

```python
def get_pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_saturation_concentration

[Show source in species.py:232](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L232)

Calculate the saturation concentration of the gas based on the vapor
pressure strategy. This method accounts for multiple strategies if
assigned and calculates saturation concentration for each strategy
based on the molar mass.

#### Arguments

- temperature (float or NDArray[np.float64]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- saturation_concentration (float or NDArray[np.float64]): The
saturation concentration of the gas

#### Raises

- `-` *ValueError* - If the vapor pressure strategy is not set.

#### Signature

```python
def get_saturation_concentration(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_saturation_ratio

[Show source in species.py:189](https://github.com/uncscode/particula/blob/main/particula/next/gas/species.py#L189)

Calculate the saturation ratio of the gas based on the vapor
pressure strategy. This method accounts for multiple strategies if
assigned and calculates saturation ratio for each strategy based on
the corresponding concentration and molar mass.

#### Arguments

- temperature (float or NDArray[np.float64]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- saturation_ratio (float or NDArray[np.float64]): The saturation ratio
of the gas

#### Raises

- `-` *ValueError* - If the vapor pressure strategy is not set.

#### Signature

```python
def get_saturation_ratio(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```
