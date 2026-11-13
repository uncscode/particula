# particula.particles.properties.reynolds_number

*module*

Calculate the particle Reynolds number.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/reynolds_number.py#L1-L1)

## particula.particles.properties.reynolds_number.get_particle_reynolds_number

*function*

```python
def get_particle_reynolds_number(particle_radius: Union[float, NDArray[np.float64]], particle_velocity: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the Reynolds number (Reₚ) of a particle in a fluid.

This dimensionless quantity characterizes the flow regime:

- Reₚ = (2 × a × vₚ) / ν
    - a is the particle radius in meters (m).
    - vₚ is the particle velocity in meters/second (m/s).
    - ν is the kinematic viscosity in square meters/second (m²/s).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius (m).
- `particle_velocity` (`Union[float, NDArray[np.float64]]`) — Particle velocity relative to the fluid (m/s).
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid (m²/s).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Dimensionless Reynolds number (float or NDArray[np.float64]).

**Examples**

```py title="Example"
import particula as par
par.particles.get_particle_reynolds_number(
    particle_radius=1e-6,
    particle_velocity=0.1,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

> **References**
>
> - [Reynolds number, Wikipedia](https://en.wikipedia.org/wiki/Reynolds_number)
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
>     Physics,
> - **Stokes Flow (Viscous Dominated, Re_p < 1)**:
>     - Particles follow the fluid closely (e.g., aerosols).
> - **Transitional Flow (1 < Re_p < 1000)**:
>     - Both **viscous and inertial forces** contribute to flow behavior.
>     - Intermediate drag corrections apply.
> - **Turbulent Flow (Re_p > 1000)**:
>     - **Inertial forces dominate**, resulting in vortex shedding and
>         wake formation.
>     - Applies to **large, fast-moving particles**
>         (e.g., raindrops, large sediment).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/reynolds_number.py#L11-L66)
