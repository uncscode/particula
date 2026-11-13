# particula.particles.properties.friction_factor_module

*module*

Module for friction factor of a particle in a fluid.

Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E., &
Hogan, C. J. (2012). Determination of the Scalar Friction Factor for
Nonspherical Particles and Aggregates Across the Entire Knudsen Number Range
by Direct Simulation Monte Carlo (DSMC). Aerosol Science and Technology,
46(10), 1065-1078. https://doi.org/10.1080/02786826.2012.690543

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/friction_factor_module.py#L1-L1)

## particula.particles.properties.friction_factor_module.get_friction_factor

*function*

```python
def get_friction_factor(particle_radius: Union[float, NDArray[np.float64]], dynamic_viscosity: float, slip_correction: Union[float, NDArray[np.float64]])
```

Calculate the friction factor for a particle in a fluid.

This friction factor (f) is the proportionality constant between
the fluid velocity and the resulting drag force on the particle.
The formula used is:

- f = (6πμ r) / C
    - f is the friction factor (N·s/m),
    - μ is the dynamic viscosity of the fluid (Pa·s),
    - r is the radius of the particle (m),
    - C is the slip correction factor (dimensionless).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particle in meters (m).
- `dynamic_viscosity` (`float`) — Dynamic viscosity of the fluid in Pa·s.
- `slip_correction` (`Union[float, NDArray[np.float64]]`) — Slip correction factor (dimensionless).

**Returns**

- `result` (`float | NDArray[np.float64]`) — The friction factor of the particle in N·s/m.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_friction_factor(
    particle_radius=1e-7,
    dynamic_viscosity=1.8e-5,
    slip_correction=1.1
)
# Output: ...
```

> **References**
>
> - Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E.,
>   & Hogan, C. J. (2012). "Determination of the Scalar Friction Factor
>   for Nonspherical Particles and Aggregates Across the Entire Knudsen
>   Number Range by Direct Simulation Monte Carlo (DSMC)."
>   Aerosol Science and Technology, 46(10), 1065-1078.
>   https://doi.org/10.1080/02786826.2012.690543

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/friction_factor_module.py#L18-L70)
