# Convert Mass Concentration

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Util](../index.md#util) / [Converting](./index.md#converting) / Convert Mass Concentration

> Auto-generated documentation for [particula.util.converting.convert_mass_concentration](https://github.com/uncscode/particula/blob/main/particula/util/converting/convert_mass_concentration.py) module.

## to_mass_fraction

[Show source in convert_mass_concentration.py:97](https://github.com/uncscode/particula/blob/main/particula/util/converting/convert_mass_concentration.py#L97)

Convert mass concentrations to mass fractions for N components.

If inputs are one-dimensional or float, the summation is done over the
entire array. If mass_concentration is a 2D array, the summation is done
row-wise.

#### Arguments

- `mass_concentrations` - A list or ndarray of mass concentrations
    (SI, kg/m^3).

#### Returns

An ndarray of mass fractions.

Reference:
    The mass fraction of a component is calculated by dividing the mass
    concentration of that component by the total mass concentration of
    all components.
    - https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)

#### Signature

```python
def to_mass_fraction(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## to_mole_fraction

[Show source in convert_mass_concentration.py:7](https://github.com/uncscode/particula/blob/main/particula/util/converting/convert_mass_concentration.py#L7)

Convert mass concentrations to mole fractions for N components.

If the input mass_concentrations is 1D, the summation is performed over the
entire array. If mass_concentrations is 2D, the summation is done row-wise.

#### Arguments

- `mass_concentrations` - A list or ndarray of mass concentrations
    (SI, kg/m^3).
- `molar_masses` - A list or ndarray of molecular weights (SI, kg/mol).

#### Returns

An ndarray of mole fractions.

Reference:
    The mole fraction of a component is given by the ratio of its molar
    concentration to the total molar concentration of all components.
    - https://en.wikipedia.org/wiki/Mole_fraction

#### Signature

```python
def to_mole_fraction(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## to_volume_fraction

[Show source in convert_mass_concentration.py:51](https://github.com/uncscode/particula/blob/main/particula/util/converting/convert_mass_concentration.py#L51)

Convert mass concentrations to volume fractions for N components.

If inputs are the one dimensional or float, the summation is done over the
the whole array. It mass_concentration is a 2D array, the summation is done
row-wise.

#### Arguments

- `mass_concentrations` - A list or ndarray of mass concentrations
    (SI, kg/m^3).
- `densities` - A list or ndarray of densities of each component
    (SI, kg/m^3).

#### Returns

An ndarray of volume fractions.

Reference:
    The volume fraction of a component is calculated by dividing the volume
    of that component (derived from mass concentration and density) by the
    total volume of all components.
    - https://en.wikipedia.org/wiki/Volume_fraction

#### Signature

```python
def to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```
