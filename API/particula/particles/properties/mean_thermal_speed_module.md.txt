# particula.particles.properties.mean_thermal_speed_module

*module*

Module contains the function for calculating the mean thermal speed
of particles in a fluid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mean_thermal_speed_module.py#L1-L1)

## particula.particles.properties.mean_thermal_speed_module.get_mean_thermal_speed

*function*

```python
def get_mean_thermal_speed(particle_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the mean thermal speed of a particle in a fluid.

The mean thermal speed (v) is derived from kinetic theory and is given by:

- v = √( (8 × k_B × T) / (π × m) )
    - v is the mean thermal speed in m/s,
    - k_B is the Boltzmann constant in J/K,
    - T is the temperature in Kelvin (K),
    - m is the particle mass in kilograms (kg).

**Parameters**

- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particle(s) in kg.
- `temperature` (`Union[float, NDArray[np.float64]]`) — The temperature of the system in Kelvin (K).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean thermal speed in m/s, as either a float or an NDArray[np.float64].

**Examples**

``` py title="Example"
import particula as par
par.particles.get_mean_thermal_speed(1e-17, 298)
# Output: ...
```

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
>   Physics, Section 9.5.3 Mean Free Path of an Aerosol Particle,
>   Equation 9.87.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mean_thermal_speed_module.py#L14-L56)
