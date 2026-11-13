# particula.particles.properties.inertia_time

*module*

Particle inertia time calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/inertia_time.py#L1-L1)

## particula.particles.properties.inertia_time.get_particle_inertia_time

*function*

```python
def get_particle_inertia_time(particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], fluid_density: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the particle inertia time (τ_p).

The particle inertia time represents the response time of a particle to
changes in fluid velocity, given by:

- τ_p = (2 / 9) × (ρ_p / ρ_f) × (r² / ν)
    - τ_p is the particle inertia time in seconds (s).
    - ρ_p is the particle density (kg/m³).
    - ρ_f is the surrounding fluid density (kg/m³).
    - r is the particle radius (m).
    - ν is the kinematic viscosity of the fluid (m²/s).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters (m).
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Density of the particle in kg/m³.
- `fluid_density` (`Union[float, NDArray[np.float64]]`) — Density of the fluid in kg/m³.
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid in m²/s.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The particle inertia time in seconds (s). Returned as either a float or NDArray[np.float64].

**Examples**

```py title="Example"
import particula as par
par.particles.get_particle_inertia_time(
    particle_radius=1e-6,
    particle_density=1000.0,
    fluid_density=1.225,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). "Effects of turbulence on
>   the geometric collision rate of sedimenting droplets. Part 2. Theory
>   and parameterization." New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/inertia_time.py#L11-L69)
