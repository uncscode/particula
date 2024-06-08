# Mass Concentration

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Util](../index.md#util) / [Converting](./index.md#converting) / Mass Concentration

> Auto-generated documentation for [particula.util.converting.mass_concentration](https://github.com/Gorkowski/particula/blob/main/particula/util/converting/mass_concentration.py) module.

## to_mole_fraction

[Show source in mass_concentration.py:7](https://github.com/Gorkowski/particula/blob/main/particula/util/converting/mass_concentration.py#L7)

Convert mass concentrations to mole fractions for N components.

#### Arguments

-----------
- `-` *mass_concentrations* - A list or ndarray of mass concentrations
(SI, kg/m^3).
- `-` *molar_masses* - A list or ndarray of molecular weights (SI, kg/mol).

#### Returns

--------
- An ndarray of mole fractions.

Reference:
----------
The mole fraction of a component is given by the ratio of its molar
concentration to the total molar concentration of all components.
- https://en.wikipedia.org/wiki/Mole_fraction

#### Signature

```python
def to_mole_fraction(
    mass_concentrations: NDArray[np.float_], molar_masses: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```



## to_volume_fraction

[Show source in mass_concentration.py:41](https://github.com/Gorkowski/particula/blob/main/particula/util/converting/mass_concentration.py#L41)

Convert mass concentrations to volume fractions for N components.

#### Arguments

-----------
- `-` *mass_concentrations* - A list or ndarray of mass concentrations
(SI, kg/m^3).
- `-` *densities* - A list or ndarray of densities of each component
(SI, kg/m^3).

#### Returns

--------
- An ndarray of volume fractions.

Reference:
----------
The volume fraction of a component is calculated by dividing the volume
of that component (derived from mass concentration and density) by the
total volume of all components.
- https://en.wikipedia.org/wiki/Volume_fraction

#### Signature

```python
def to_volume_fraction(
    mass_concentrations: NDArray[np.float_], densities: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```
