# particula.dynamics.coagulation.sedimentation_kernel

*module*

Sedimentation kernel for aerosol particles.

Used for calculating collisions due to gravitational settling, where heavier
particles fall faster and may collide with slower ones. The kernel follows
Equation 13A.4 from Seinfeld and Pandis (2016), proportional to the difference
in settling velocities and the combined projected area of two particles.

> **Equation**
>
> - K(i, j) = (π / 4) × (Dᵢ + Dⱼ)² × |vᵢ - vⱼ| × Eᵢⱼ
>   - Dᵢ, Dⱼ : diameters of particle i and j [m],
>   - vᵢ, vⱼ : settling velocities [m/s],
>   - Eᵢⱼ : collision efficiency (dimensionless).

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
>   and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L1-L1)

## particula.dynamics.coagulation.sedimentation_kernel.calculate_collision_efficiency_function

*function*

```python
def calculate_collision_efficiency_function(radius1: float, radius2: float) -> float
```

Calculate the collision efficiency between two particles (placeholder).

This function calculates the collision efficiency E for two particles
of radii radius1 and radius2, which can depend on additional factors
(e.g., fluid flow or electrostatic forces). Currently not implemented.

**Parameters**

- `radius1` (`float`) — The radius of the first particle [m].
- `radius2` (`float`) — The radius of the second particle [m].

**Returns**

- (`float`) — Collision efficiency [dimensionless].

**Examples**

```py
# Not implemented
calculate_collision_efficiency_function(1e-7, 2e-7)
# Raises NotImplementedError
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
>   in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L28-L58)

## particula.dynamics.coagulation.sedimentation_kernel.get_sedimentation_kernel_sp2016

*function*

```python
def get_sedimentation_kernel_sp2016(particle_radius: NDArray[np.float64], settling_velocities: NDArray[np.float64], calculate_collision_efficiency: bool = True) -> NDArray[np.float64]
```

Calculate sedimentation kernel for aerosol particles (Eq 13A.4).

This function computes the coagulation kernel due to gravitational
settling, where larger particles settle faster and overtake smaller
ones. The kernel is based on the combined diameters, the settling
velocity difference, and the collision efficiency.

Equation:
- K(i, j) = (π / 4) × (Dᵢ + Dⱼ)² × |vᵢ - vⱼ| × Eᵢⱼ
    - Dᵢ, Dⱼ : diameters of particle i and j [m],
    - vᵢ, vⱼ : settling velocities [m/s],
    - Eᵢⱼ : collision efficiency (dimensionless).

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `settling_velocities` (`NDArray[np.float64]`) — Array of particle settling velocities [m/s].
- `calculate_collision_efficiency` (`bool`) (default: `True`) — Whether to calculate collision efficiency or use 1. Defaults to True.

**Returns**

- (`NDArray[np.float64]`) — Sedimentation kernel matrix [m³/s], shape (n, n).

**Examples**

```py title="Example"
import numpy as np

rads = np.array([1e-7, 2e-7])
vels = np.array([5e-3, 1e-2])
kernel = get_sedimentation_kernel_sp2016(rads, vels)
print(kernel)
```

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
>   and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L61-L122)

## particula.dynamics.coagulation.sedimentation_kernel.get_sedimentation_kernel_sp2016_via_system_state

*function*

```python
def get_sedimentation_kernel_sp2016_via_system_state(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float, calculate_collision_efficiency: bool = True) -> NDArray[np.float64]
```

Calculate the sedimentation kernel (Equation 13A.4) via system state.

This function first derives settling velocities using the system state
(particle radius, density, temperature, pressure), then calls
get_sedimentation_kernel_sp2016.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Array of particle densities [kg/m³].
- `temperature` (`float`) — Temperature [K].
- `pressure` (`float`) — Pressure [Pa].
- `calculate_collision_efficiency` (`bool`) (default: `True`) — Whether to calculate collision efficiency or use 1. Defaults to True.

**Returns**

- (`NDArray[np.float64]`) — Sedimentation kernel matrix [m³/s], shape (n, n).

**Examples**

```py title="Example"
import numpy as np
rads = np.array([1e-7, 2e-7])
dens = np.array([1000, 1200])
kernel = get_sedimentation_kernel_sp2016_via_system_state(
    particle_radius=rads,
    particle_density=dens,
    temperature=298,
    pressure=101325
)
print(kernel)
```

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
>   and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L125-L179)
