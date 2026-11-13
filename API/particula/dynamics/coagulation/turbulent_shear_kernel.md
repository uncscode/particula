# particula.dynamics.coagulation.turbulent_shear_kernel

*module*

Turbulent shear kernel for coagulation (Saffman & Turner, 1956).

This module computes the collision kernel attributed to turbulent shear
in a fluid. The formula stems from Equation 13A.2 in Seinfeld & Pandis (2016),
and was originally derived by Saffman & Turner (1956). Particles gain relative
velocities due to turbulent velocity gradients, increasing collision likelihood.

> **Equation**
>
> - K(D₁, D₂) = √(π × eₖ / (120 × ν)) × (D₁ + D₂)³
>   - eₖ : Turbulent kinetic energy dissipation rate [m²/s³],
>   - ν : Kinematic viscosity [m²/s],
>   - D₁, D₂ : particle diameters [m].

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
>   in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
>   https://doi.org/10.1017/S0022112056000020
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.2.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py#L1-L1)

## particula.dynamics.coagulation.turbulent_shear_kernel.get_turbulent_shear_kernel_st1956

*function*

```python
def get_turbulent_shear_kernel_st1956(particle_radius: NDArray[np.float64], turbulent_dissipation: float, kinematic_viscosity: float) -> NDArray[np.float64]
```

Calculate the turbulent shear kernel (Equation 13A.2, Saffman & Turner,
1956).

This function implements the formula for collisions induced by turbulent
shear. The turbulent dissipation rate and kinematic viscosity determine
how rapidly eddies drive particle collisions.

Equation:
- K(D₁, D₂) = √(π × eₖ / (120 × ν)) × (D₁ + D₂)³
    - eₖ : Turbulent kinetic energy dissipation rate [m²/s³],
    - ν : Kinematic viscosity [m²/s],
    - D₁, D₂ : diameters of particles [m].

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `turbulent_dissipation` (`float`) — Turbulent energy dissipation rate [m²/s³].
- `kinematic_viscosity` (`float`) — Kinematic viscosity [m²/s].

**Returns**

- (`NDArray[np.float64]`) — Turbulent shear kernel matrix [m³/s], shape (n, n).

**Examples**

```py title="Example"
import numpy as np

r = np.array([1e-7, 2e-7])
k_matrix = get_turbulent_shear_kernel_st1956(
    particle_radius=r,
    turbulent_dissipation=1e-3,
    kinematic_viscosity=1.5e-5
)
print(k_matrix)
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
>   in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.2.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py#L30-L81)

## particula.dynamics.coagulation.turbulent_shear_kernel.get_turbulent_shear_kernel_st1956_via_system_state

*function*

```python
def get_turbulent_shear_kernel_st1956_via_system_state(particle_radius: NDArray[np.float64], turbulent_dissipation: float, temperature: float, fluid_density: float) -> NDArray[np.float64]
```

Calculate the turbulent shear kernel using system state data.

This version derives the kinematic viscosity from the temperature and
fluid density, then uses get_turbulent_shear_kernel_st1956 for the
Saffman & Turner (1956) formula:

> **Equation**
>
> - K(D₁, D₂) = √(π × eₖ / (120 × ν)) × (D₁ + D₂)³
>   - eₖ : Turbulent dissipation rate [m²/s³],
>   - ν : Kinematic viscosity [m²/s],
>   - D₁, D₂ : particle diameters [m].

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `turbulent_dissipation` (`float`) — Turbulent dissipation rate [m²/s³].
- `temperature` (`float`) — Temperature [K].
- `fluid_density` (`float`) — Fluid density [kg/m³].

**Returns**

- (`NDArray[np.float64]`) — Turbulent shear kernel matrix [m³/s], shape (n, n).

**Examples**

```py title="Example"
import numpy as np

r = np.array([1e-7, 2e-7])
kernel_matrix = get_turbulent_shear_kernel_st1956_via_system_state(
    particle_radius=r,
    turbulent_dissipation=1e-3,
    temperature=300,
    fluid_density=1.2
)
print(kernel_matrix)
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
>   in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.2.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py#L84-L138)
