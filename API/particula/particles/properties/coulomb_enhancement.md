# particula.particles.properties.coulomb_enhancement

*module*

Module for coulomb-related enhancements.

References:
Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L1-L1)

## particula.particles.properties.coulomb_enhancement.get_coulomb_continuum_limit

*function*

```python
def get_coulomb_continuum_limit(coulomb_potential: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the continuum-limit Coulomb enhancement factor, Γ_c.

The continuum-limit factor is computed by:

- Γ_c =
    ϕ_E / [1 - exp(-ϕ_E)]    if ϕ_E ≠ 0
    1                        if ϕ_E = 0

where ϕ_E is the Coulomb potential ratio (dimensionless).

**Parameters**

- `coulomb_potential` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio ϕ_E (dimensionless).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Coulomb enhancement factor in the continuum limit (dimensionless).

**Examples**

``` py title="Example"
import numpy as np
import particula as par
potential = np.array([-0.5, 0.0, 0.5])
par.particles.get_coulomb_continuum_limit(potential)
print(gamma_cont)
# Output: array([...])
```

> **References**
>
> - Equation (6b): Gopalakrishnan, R., & Hogan, C. J. (2012).
>   Coulomb-influenced collisions in aerosols and dusty plasmas.
>   Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L143-L186)

## particula.particles.properties.coulomb_enhancement.get_coulomb_enhancement_ratio

*function*

```python
def get_coulomb_enhancement_ratio(particle_radius: Union[float, NDArray[np.float64]], charge: Union[int, NDArray[np.float64]] = 0, temperature: float = 298.15, ratio_lower_limit: float = -200) -> Union[float, NDArray[np.float64]]
```

Calculate the Coulomb potential ratio, ϕ_E, for particle-particle
interactions.

The potential ratio is computed using:

- ϕ_E = - (qᵢ × qⱼ × e²) / [4π ε₀ (rᵢ + rⱼ) k_B T]
    - ϕ_E is the Coulomb potential ratio (dimensionless).
    - qᵢ, qⱼ are the charges (dimensionless, e.g. the number of electrons).
    - e is the elementary charge in coulombs (C).
    - ε₀ is the electric permittivity of free space (F·m⁻¹).
    - rᵢ, rⱼ are the particle radii (m).
    - k_B is the Boltzmann constant (J·K⁻¹).
    - T is the temperature (K).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particles (m).
- `charge` (`Union[int, NDArray[np.float64]]`) (default: `0`) — Number of integer charges on the particles (dimensionless).
- `temperature` (`float`) (default: `298.15`) — System temperature (K).
- `ratio_lower_limit` (`float`) (default: `-200`) — Lower limit to clip the potential ratio for very large negative (repulsive) values.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio (dimensionless).

**Examples**

``` py title="Example"
import numpy as np
import particula as par
par.particles.get_coulomb_enhancement_ratio(
    radius=np.array([1e-7, 2e-7]),
    charge=np.array([1, 2]),
    temperature=298.15,
    ratio_lower_limit=-200
)
# Output: array([...])
```

> **References**
>
> - Equation (7): Gopalakrishnan, R., & Hogan, C. J. (2012).
>   Coulomb-influenced collisions in aerosols and dusty plasmas.
>   Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L24-L97)

## particula.particles.properties.coulomb_enhancement.get_coulomb_kinetic_limit

*function*

```python
def get_coulomb_kinetic_limit(coulomb_potential: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the kinetic-limit Coulomb enhancement factor, Γₖᵢₙ.

The kinetic-limit factor is computed by:

- Γₖᵢₙ =
    1 + ϕ_E      if ϕ_E ≥ 0
    exp(ϕ_E)     if ϕ_E < 0

where ϕ_E is the Coulomb potential ratio (dimensionless).

**Parameters**

- `coulomb_potential` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio ϕ_E (dimensionless).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Coulomb enhancement factor in the kinetic limit (dimensionless).

**Examples**

``` py title="Example"
import numpy as np
import particula as par
potential = np.array([-0.5, 0.0, 0.5])
par.particles.get_coulomb_kinetic_limit(potential)
# Output: array([...])
```

> **References**
>
> - Equations (6d) and (6e): Gopalakrishnan, R., & Hogan, C. J. (2012).
>   Coulomb-influenced collisions in aerosols and dusty plasmas.
>   Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L100-L140)
