# Species Density

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Species Density

> Auto-generated documentation for [particula.activity.species_density](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py) module.

## organic_array

[Show source in species_density.py:93](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py#L93)

Get densities for an array.

#### Signature

```python
def organic_array(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
): ...
```



## organic_density_estimate

[Show source in species_density.py:12](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py#L12)

Function to estimate the density of organic compounds based on the simple
model by Girolami (1994). The input parameters include molar mass, O:C
and H:C ratios. If the H:C ratio is unknown at input, enter a negative
value. The actual H:C will then be estimated based on an initial assumption
of H:C = 2. The model also estimates the number of carbon atoms per
molecular structure based on molar mass, O:C, and H:C.
The density is then approximated by the formula of Girolami.

Reference:
Girolami, G. S.: A Simple 'Back of the Envelope' Method for Estimating
the Densities and Molecular Volumes of Liquids and Solids,
J. Chem. Educ., 71(11), 962, doi:10.1021/ed071p962, 1994.

#### Arguments

- `molar_mass(float)` - Molar mass.
- `oxygen2carbon` *float* - O:C ratio.
- `hydrogen2carbon` *float* - H:C ratio. If unknown, provide a negative
    value.
- `nitrogen2carbon` *float, optional* - N:C ratio. Defaults to None.

#### Returns

- `densityEst` *float* - Estimated density in g/cm^3.

#### Signature

```python
def organic_density_estimate(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
): ...
```
