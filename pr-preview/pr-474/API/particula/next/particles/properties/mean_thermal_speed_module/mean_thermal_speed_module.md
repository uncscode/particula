# Mean Thermal Speed Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Mean Thermal Speed Module

> Auto-generated documentation for [particula.next.particles.properties.mean_thermal_speed_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/mean_thermal_speed_module.py) module.

## mean_thermal_speed

[Show source in mean_thermal_speed_module.py:11](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/mean_thermal_speed_module.py#L11)

Returns the particles mean thermal speed. Due to the the impact
of air molecules on the particles, the particles will have a mean
thermal speed.

Args
----
mass : The per particle mass of the particles [kg].
temperature : The temperature of the air [K].

Returns
-------
The mean thermal speed of the particles [m/s].

References
----------
Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 9.5.3 Mean Free Path of an Aerosol Particle Equation 9.87.

#### Signature

```python
def mean_thermal_speed(
    mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
