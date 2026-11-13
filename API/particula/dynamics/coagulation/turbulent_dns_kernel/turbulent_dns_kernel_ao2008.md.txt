# particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008

*module*

Calculate the geometric collision kernel Γ₁₂ (or K₁₂) based on turbulent
DNS simulations.

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008.get_turbulent_dns_kernel_ao2008

*function*

```python
def get_turbulent_dns_kernel_ao2008(particle_radius: Union[float, NDArray[np.float64]], velocity_dispersion: Union[float, NDArray[np.float64]], particle_inertia_time: Union[float, NDArray[np.float64]], stokes_number: Union[float, NDArray[np.float64]], kolmogorov_length_scale: float, reynolds_lambda: float, normalized_accel_variance: float, kolmogorov_velocity: float, kolmogorov_time: float) -> Union[float, NDArray[np.float64]]
```

Compute the geometric collision kernel Γ₁₂ from DNS simulations.

Where the equation is

- Γ₁₂ = 2π R² ⟨|wᵣ|⟩ g₁₂
    - Γ₁₂ is collision kernel [m³/s].
    - R is collision radius [m].
    - wᵣ is radial relative velocity [m/s].
    - g₁₂ is radial distribution function [dimensionless].

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius [m].
- `velocity_dispersion` (`Union[float, NDArray[np.float64]]`) — Velocity dispersion [m/s].
- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Particle inertia time [s].
- `stokes_number` (`Union[float, NDArray[np.float64]]`) — Stokes number [-].
- `kolmogorov_length_scale` (`float`) — Kolmogorov length scale [m].
- `reynolds_lambda` (`float`) — Reynolds number [-].
- `normalized_accel_variance` (`float`) — Normalized acceleration variance [-].
- `kolmogorov_velocity` (`float`) — Kolmogorov velocity [m/s].
- `kolmogorov_time` (`float`) — Kolmogorov time [s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Collision kernel Γ₁₂ [m³/s].

References:
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py#L23-L100)

## particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008.get_turbulent_dns_kernel_ao2008_via_system_state

*function*

```python
def get_turbulent_dns_kernel_ao2008_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], fluid_density: float, temperature: float, re_lambda: float, relative_velocity: Union[float, NDArray[np.float64]], turbulent_dissipation: float) -> Union[float, NDArray[np.float64]]
```

Compute the geometric collision kernel Γ₁₂ using AO2008 for system.

This function orchestrates the calculation of the geometric collision
kernel by deriving necessary fluid, turbulence, and particle parameters
from the provided system state. The returned value (or array) represents
the collision kernel, Γ₁₂ [m³/s], which describes collision frequency
under turbulence.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particles [m]. If an array is given, it is assumed to represent multiple particle sizes.
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Density of the particles [kg/m³]. Must match the dimensionality of `particle_radius` if both are arrays.
- `fluid_density` (`float`) — Density of the surrounding fluid [kg/m³].
- `temperature` (`float`) — Temperature of the fluid [K].
- `re_lambda` (`float`) — Turbulent Reynolds number based on the Taylor microscale.
- `relative_velocity` (`Union[float, NDArray[np.float64]]`) — Mean relative velocity between the particle and fluid [m/s]. Can be a single value or an array of the same dimensionality as `particle_radius`.
- `turbulent_dissipation` (`float`) — Turbulent kinetic energy dissipation rate [m²/s³].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Collision kernel Γ₁₂ [m³/s].

**Examples**

```py
kernel_via_state = get_turbulent_dns_kernel_ao2008_via_system_state(
    particle_radius=np.array([1e-6, 2e-6]),
    particle_density=1000.0,
    fluid_density=1.225,
    temperature=298.15,
    re_lambda=100.0,
    relative_velocity=0.1,
    turbulent_dissipation=0.01,
)
# Output: array([...])
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>   the geometric collision rate of sedimenting droplets. Part 2.
>   Theory and parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py#L103-L273)
