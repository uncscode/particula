# Species Properties

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Species Properties

> Auto-generated documentation for [particula.util.species_properties](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py) module.

#### Attributes

- `species_properties` - maybe this should be loaded from a file, or for an option of the user to
  add their own species or load files with species properties: {'generic': {'molecular_weight': 200.0 * u.g / u.mol, 'surface_tension': 0.072 * u.N / u.m, 'density': 1000.0 * u.kg / u.m ** 3, 'vapor_radius': 1e-14 * u.Pa, 'vapor_attachment': 1.0 * u.dimensionless, 'saturation_pressure': 1.0 * u.Pa, 'latent_heat': 1.0 * u.J / u.g, 'heat_vaporization': 1.0 * u.J / u.kg, 'kappa': 0.2 * u.dimensionless}, 'water': {'molecular_weight': 18.01528 * u.g / u.mol, 'surface_tension': 0.072 * u.N / u.m, 'density': 1000.0 * u.kg / u.m ** 3, 'vapor_radius': 1.6e-09 * u.m, 'vapor_attachment': 1.0 * u.dimensionless, 'saturation_pressure': water_buck_psat, 'latent_heat': lambda T,: 2500.8 - 2.36 * T.m_as('degC') + 0.0016 * T.m_as('degC') ** 2 - 6e-05 * T.m_as('degC') ** 3 * u.J / u.g, 'heat_vaporization': 2257000.0 * u.J / u.kg, 'kappa': 0.2 * u.dimensionless}, 'ammonium sulfate': {'molecular_weight': 132.14 * u.g / u.mol, 'surface_tension': 0.072 * u.N / u.m, 'density': 1770.0 * u.kg / u.m ** 3, 'vapor_radius': 1.6e-09 * u.m, 'vapor_attachment': 1.0 * u.dimensionless, 'saturation_pressure': 1e-14 * u.Pa, 'latent_heat': 0.0 * u.J / u.g, 'heat_vaporization': 0.0 * u.J / u.kg, 'kappa': 0.53 * u.dimensionless}}


## MaterialProperties

[Show source in species_properties.py:17](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L17)

#### Signature

```python
class MaterialProperties:
    def __init__(self, species_properties): ...
```

### MaterialProperties().latent_heat

[Show source in species_properties.py:34](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L34)

#### Signature

```python
def latent_heat(self, temperature, species): ...
```

### MaterialProperties().saturation_pressure

[Show source in species_properties.py:24](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L24)

#### Signature

```python
def saturation_pressure(self, temperature, species): ...
```



## clausius_clapeyron

[Show source in species_properties.py:63](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L63)

Calculates the vapor pressure of a substance at a given temperature
using the Clausius-Clapeyron equation.

Args
----------
    temperature (float): Temperature reference in Kelvin
    vapor_pressure (float): Vapor pressure reference in Pa
    temperature_new (float): Temperature new in Kelvin
    heat_vaporization (float): Heat of vaporization in J/kg

Returns
-------
    vapor_pressure_new: Vapor pressure in Pa

#### Signature

```python
def clausius_clapeyron(
    temperature, vapor_pressure, temperature_new, heat_vaporization
): ...
```



## material_properties

[Show source in species_properties.py:140](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L140)

Return the material properties for a given species.

Args
----------
property : str
    Property to return. Options are: 'all', 'molecular_weight',
    'surface_tension', 'density', 'vapor_radius', 'vapor_attachment',
    'kappa'.
species : str
    Species for which to return the material properties.
temperature : K
    Temperature of the material.

Returns
-------
material_properties : value
    The material property for the given species.

#### Signature

```python
def material_properties(property, species="water", temperature=298.15 * u.K): ...
```



## vapor_concentration

[Show source in species_properties.py:177](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L177)

Convert saturation ratio to mass concentration at a given temperature.

Args
----------
sat_ratio : float
    saturation ratio.
temperature : K
    Air temperature.
species : str, optional
    Species for which to calculate the saturation vapor pressure.
    Default is "water".

Returns
-------
concentration : concentration units
    Concentration vapor.

TODO: check values for different species

#### Signature

```python
def vapor_concentration(saturation_ratio, temperature=298.15 * u.K, species="water"): ...
```



## water_buck_psat

[Show source in species_properties.py:47](https://github.com/uncscode/particula/blob/main/particula/util/species_properties.py#L47)

 Buck equation for water vapor pressure
https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def water_buck_psat(temperature): ...
```
