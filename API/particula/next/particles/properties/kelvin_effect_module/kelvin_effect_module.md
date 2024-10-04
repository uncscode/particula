# Kelvin Effect Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Kelvin Effect Module

> Auto-generated documentation for [particula.next.particles.properties.kelvin_effect_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/kelvin_effect_module.py) module.

## kelvin_radius

[Show source in kelvin_effect_module.py:10](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/kelvin_effect_module.py#L10)

Calculate the Kelvin radius which determines the curvature effect on
vapor pressure.

#### Arguments

-----
- surface_tension (float or NDArray[float]): Surface tension of the
mixture [N/m].
- molar_mass (float or NDArray[float]): Molar mass of the species
[kg/mol].
- mass_concentration (float or NDArray[float]): Concentration of the
species [kg/m^3].
- temperature (float): Temperature of the system [K].

#### Returns

--------
- float or NDArray[float]: Kelvin radius [m].

#### References

-----------
- Based on Neil Donahue's approach to the Kelvin equation:
r = 2 * surface_tension * molar_mass / (R * T * density)
- `See` *more* - https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
def kelvin_radius(
    effective_surface_tension: Union[float, NDArray[np.float64]],
    effective_density: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## kelvin_term

[Show source in kelvin_effect_module.py:45](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/kelvin_effect_module.py#L45)

Calculate the Kelvin term, which quantifies the effect of particle
curvature on vapor pressure.

#### Arguments

-----
- radius (float or NDArray[float]): Radius of the particle [m].
- kelvin_radius (float or NDArray[float]): Kelvin radius [m].

#### Returns

--------
- float or NDArray[float]: The exponential factor adjusting vapor
pressure due to curvature.

#### References

Based on Neil Donahue's collection of terms in the Kelvin equation:
exp(kelvin_radius / particle_radius)
- `See` *more* - https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
def kelvin_term(
    radius: Union[float, NDArray[np.float64]],
    kelvin_radius_value: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
