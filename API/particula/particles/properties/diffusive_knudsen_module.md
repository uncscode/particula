# particula.particles.properties.diffusive_knudsen_module

*module*

Module for the diffusive knudsen number.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusive_knudsen_module.py#L1-L1)

## particula.particles.properties.diffusive_knudsen_module.KINETIC_ENHANCE_MIN

*attribute* · *module attribute*

```python
KINETIC_ENHANCE_MIN = 1e-80
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusive_knudsen_module.py#L15-L15)

## particula.particles.properties.diffusive_knudsen_module.get_diffusive_knudsen_number

*function*

```python
def get_diffusive_knudsen_number(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], friction_factor: Union[float, NDArray[np.float64]], coulomb_potential_ratio: Union[float, NDArray[np.float64]] = 0.0, temperature: float = 298.15) -> Union[float, NDArray[np.float64]]
```

Compute the diffusive Knudsen number for particle-particle interactions.

The *diffusive* Knudsen number (Kn_d) differs from the standard Knudsen
number. It represents the ratio of the mean particle persistence
distance to the effective Coulombic interaction scale. Mathematically:

- Kn_d = [ √(k_B × T × μ_red) / f_red ] / [ (rᵢ + rⱼ) × (Γ_c / Γ_k) ]
    - k_B is the Boltzmann constant (J/K).
    - T is the temperature (K).
    - μ_red is the reduced mass of particles (kg).
    - f_red is the reduced friction factor (dimensionless).
    - rᵢ + rⱼ is the sum of radii for the interacting particles (m).
    - Γ_c is the continuum-limit Coulomb enhancement factor(dimensionless).
    - Γ_k is the kinetic-limit Coulomb enhancement factor (dimensionless).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particle(s) in meters (m).
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — Mass of the particle(s) in kilograms (kg).
- `friction_factor` (`Union[float, NDArray[np.float64]]`) — Friction factor(s) (dimensionless).
- `coulomb_potential_ratio` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Coulomb potential ratio (dimensionless), zero if no charge.
- `temperature` (`float`) (default: `298.15`) — Temperature of the system in Kelvin (K).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number, either a float or NDArray[np.float64].

**Examples**

```py title="Single Particle Example"
import numpy as np
import particula as par
par.particles.get_diffusive_knudsen_number(
    particle_radius=1e-7,
    particle_mass=1e-17,
    friction_factor=0.8,
    coulomb_potential_ratio=0.3,
    temperature=300
)
# Output: 0.12...
```
```py title="Multiple Particles Example"
import numpy as np
import particula as par
# Multiple particles example
radius_arr = np.array([1e-7, 2e-7])
mass_arr = np.array([1e-17, 2e-17])
friction_arr = np.array([0.8, 1.1])
potential_arr = np.array([0.3, 0.5])
par.particles.par.get_diffusive_knudsen_number(
    radius_arr, mass_arr, friction_arr, potential_arr
)
# Output: array([...])
```

> **References**
>
> - Chahl, H. S., & Gopalakrishnan, R. (2019). "High potential, near free
>   molecular regime Coulombic collisions in aerosols and dusty plasmas."
>   Aerosol Science and Technology, 53(8), 933-957.
>   https://doi.org/10.1080/02786826.2019.1614522
> - Gopalakrishnan, R., & Hogan, C. J. (2012). "Coulomb-influenced
>   collisions in aerosols and dusty plasmas." Physical Review E, 85(2).
>   https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusive_knudsen_module.py#L18-L137)
