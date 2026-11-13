# particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008

*module*

Calculate the radial distribution function g_{12} for particles in a
turbulent flow.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._calculate_c1

*function*

```python
def _calculate_c1(stokes_number: NDArray[np.float64], reynolds_lambda: float, kolmogorov_velocity: float, kolmogorov_time: float) -> NDArray[np.float64]
```

Compute C_1 based on Stokes number and turbulence properties.

C_1 is given by:

    C_1 = y(St) / (|g| / (v_k / τ_k))^f_3(R_λ)

- y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St
- f_3(R_λ) = 0.1886 * exp(20.306 / R_λ)
- |g| : Gravitational acceleration [m/s²]
- v_k : Kolmogorov velocity scale [m/s]
- τ_k : Kolmogorov timescale [s]

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L120-L144)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._calculate_rc

*function*

```python
def _calculate_rc(stokes_diff_matrix: NDArray[np.float64], kolmogorov_length_scale: float, normalized_accel_variance: float, reynolds_lambda: float, kolmogorov_velocity: float, kolmogorov_time: float) -> NDArray[np.float64]
```

Compute r_c, the turbulence-driven correction to the collision kernel.

The equation is:

(r_c / η)^2 = |St_2 - St_1| * F(a_Og, R_λ)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L171-L194)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._compute_a_og

*function*

```python
def _compute_a_og(normalized_accel_variance: float, kolmogorov_velocity: float, kolmogorov_time: float) -> float
```

Compute aOg, which accounts for the effect of gravity on
turbulence-driven clustering.

- a_Og = a_o + (π / 8) * (|g| / (v_k / τ_k))^2

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L197-L210)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._compute_f

*function*

```python
def _compute_f(a_og: float, reynolds_lambda: float) -> float
```

Compute F(aOg, R_lambda), an empirical scaling factor for
turbulence effects.

- F(a_Og, R_λ) = 20.115 * (a_Og / R_λ)^0.5

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L213-L222)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._compute_f3_lambda

*function*

```python
def _compute_f3_lambda(reynolds_lambda: float) -> float
```

Compute f_3(R_lambda), an empirical turbulence factor.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L165-L168)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008._compute_y_stokes

*function*

```python
def _compute_y_stokes(stokes_number: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute y(St), ensuring values remain non-negative.

y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St

Ensures y(St) ≥ 0 (if negative, sets to 0).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L147-L162)

## particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008.get_g12_radial_distribution_ao2008

*function*

```python
def get_g12_radial_distribution_ao2008(particle_radius: NDArray[np.float64], stokes_number: NDArray[np.float64], kolmogorov_length_scale: float, reynolds_lambda: float, normalized_accel_variance: float, kolmogorov_velocity: float, kolmogorov_time: float) -> NDArray[np.float64]
```

Compute the radial distribution function g₁₂ for particles in a
turbulent flow.

This function describes the clustering of particles in a turbulent flow.
The equation is:

- g₁₂ = ((η² + r_c²) / (R² + r_c²))^(C₁/2)
    - g₁₂ is the radial distribution function (dimensionless),
    - η is the Kolmogorov length scale (m),
    - r_c is the turbulence-driven correction length (m),
    - R is the collision radius (sum of the two particle radii) (m),
    - C₁ is a dimensionless function dependent on the Stokes numbers,
      Reynolds number, etc.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii in meters.
- `stokes_number` (`NDArray[np.float64]`) — Array of particle Stokes numbers (dimensionless).
- `kolmogorov_length_scale` (`float`) — Kolmogorov length scale in meters.
- `reynolds_lambda` (`float`) — Taylor-microscale Reynolds number (dimensionless).
- `normalized_accel_variance` (`float`) — Normalized acceleration variance (dimensionless).
- `kolmogorov_velocity` (`float`) — Kolmogorov velocity scale in m/s.
- `kolmogorov_time` (`float`) — Kolmogorov timescale in seconds.

**Returns**

- (`NDArray[np.float64]`) — The radial distribution function g₁₂ (dimensionless).

**Examples**

```py title="Example Usage"
import numpy as np
from particula.dynamics.coagulation.turbulent_dns_kernel
    .g12_radial_distribution_ao2008 import (
        get_g12_radial_distribution_ao2008,
    )

radii = np.array([1e-7, 1e-6])
stks = np.array([0.1, 0.2])
result = get_g12_radial_distribution_ao2008(
    particle_radius=radii,
    stokes_number=stks,
    kolmogorov_length_scale=1e-4,
    reynolds_lambda=100,
    normalized_accel_variance=0.5,
    kolmogorov_velocity=0.1,
    kolmogorov_time=0.001,
)
print(result)
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>   the geometric collision rate of sedimenting droplets. Part 2.
>   Theory and parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L13-L117)
