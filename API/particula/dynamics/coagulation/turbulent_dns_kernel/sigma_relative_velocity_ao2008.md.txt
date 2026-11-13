# particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008

*module*

Compute RMS fluctuation velocities and cross-correlations.

This module provides functions to compute the square of the RMS fluctuation
velocity and the cross-correlation of the fluctuating velocities for colliding
droplets, based on the theory of turbulent DNS kernels.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms

*class*

```python
class VelocityCorrelationTerms(NamedTuple)
```

Bases: `NamedTuple`

Parameters from computing velocity correlation terms.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L34-L44)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.b1

*attribute* · *instance attribute*

```python
b1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L37-L37)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.b2

*attribute* · *instance attribute*

```python
b2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L38-L38)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.c1

*attribute* · *instance attribute*

```python
c1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L41-L41)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.c2

*attribute* · *instance attribute*

```python
c2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L42-L42)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.d1

*attribute* · *instance attribute*

```python
d1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L39-L39)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.d2

*attribute* · *instance attribute*

```python
d2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L40-L40)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.e1

*attribute* · *instance attribute*

```python
e1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L43-L43)

### particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.VelocityCorrelationTerms.e2

*attribute* · *instance attribute*

```python
e2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L44-L44)

## particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008._compute_cross_correlation_velocity

*function*

```python
def _compute_cross_correlation_velocity(fluid_rms_velocity: float, collisional_radius: Union[float, NDArray[np.float64]], particle_inertia_time: Union[float, NDArray[np.float64]], particle_velocity: Union[float, NDArray[np.float64]], taylor_microscale: float, eulerian_integral_length: float, velocity_correlation_terms: VelocityCorrelationTerms) -> Union[float, NDArray[np.float64]]
```

Compute cross-correlation of fluctuating velocities for two droplets.

This function calculates the cross-correlation of the fluctuating
velocities of two droplets using the following equation:

Where the equation is

- ⟨v'¹ v'²⟩ = (u'² f₂(R) / (τ_p1 τ_p2)) *
    [b₁ d₁ Φ(c₁, e₁) - b₁ d₂ Φ(c₁, e₂) - b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)]
    - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.
    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_p1, τ_p2 : Inertia timescales of droplets 1 and 2 [s].
    - f₂(R) : Longitudinal velocity correlation function.
    - Φ(c, e) : Function Φ computed using `get_phi_ao2008`.

**Parameters**

- `fluid_rms_velocity` (`float`) — Fluid RMS fluctuation velocity [m/s].
- `collisional_radius` (`Union[float, NDArray[np.float64]]`) — Distance between two colliding droplets [m].
- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Inertia timescale of droplet 1 [s].
- `particle_velocity` (`Union[float, NDArray[np.float64]]`) — Droplet velocity [m/s].
- `taylor_microscale` (`float`) — Taylor microscale [m].
- `eulerian_integral_length` (`float`) — Eulerian integral length scale [m].
- `velocity_correlation_terms` (`VelocityCorrelationTerms`) — Velocity correlation coefficients [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Cross-correlation velocity [m²/s²].

**Examples**

```py
import numpy as np
ccv = _compute_cross_correlation_velocity(
    fluid_rms_velocity=0.3,
    collisional_radius=np.array([1e-4, 2e-4]),
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    taylor_microscale=0.01,
    eulerian_integral_length=0.1,
    velocity_correlation_terms=VelocityCorrelationTerms(
        b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
    )
)
# Output: array([...])
```

> **References**
>
> - Ayala, O. et al. (2008). Effects of turbulence on the geometric
>   collision rate of sedimenting droplets. Part 2. Theory and
>   parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L260-L395)

## particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008._compute_rms_fluctuation_velocity

*function*

```python
def _compute_rms_fluctuation_velocity(fluid_rms_velocity: float, particle_inertia_time: Union[float, NDArray[np.float64]], particle_velocity: Union[float, NDArray[np.float64]], velocity_correlation_terms: VelocityCorrelationTerms) -> Union[float, NDArray[np.float64]]
```

Compute RMS fluctuation velocity for the k-th droplet.

This function calculates the square of the RMS fluctuation velocity for
the k-th droplet using the following equation:

Where the equation is:

- ⟨(v'ᵏ)²⟩ = (u'² / τ_pk) * [b₁ d₁ Ψ(c₁, e₁) - b₁ d₂ Ψ(c₁, e₂)
  - b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)]
    - v'ᵏ is the fluctuating velocity for droplet k.
    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_pk (particle_inertia_time) : Inertia timescale of the droplet k
        [s].
    - Ψ(c, e) : Function Ψ computed using `get_psi_ao2008`.

**Parameters**

- `fluid_rms_velocity` (`float`) — Fluid RMS fluctuation velocity [m/s].
- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Inertia timescale of the droplet k [s].
- `particle_velocity` (`Union[float, NDArray[np.float64]]`) — Droplet velocity [m/s].
- `velocity_correlation_terms` (`VelocityCorrelationTerms`) — Velocity correlation coefficients [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — RMS fluctuation velocity [m²/s²].

**Examples**

```py
import numpy as np
rms_fluct = _compute_rms_fluctuation_velocity(
    fluid_rms_velocity=0.3,
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    velocity_correlation_terms=VelocityCorrelationTerms(
        b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
    )
)
# Output: array([...])
```

> **References**
>
> - Ayala, O. et al. (2008). Effects of turbulence on the geometric
>   collision rate of sedimenting droplets. Part 2. Theory and
>   parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L155-L257)

## particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008.get_relative_velocity_variance

*function*

```python
def get_relative_velocity_variance(fluid_rms_velocity: float, collisional_radius: NDArray[np.float64], particle_inertia_time: NDArray[np.float64], particle_velocity: NDArray[np.float64], taylor_microscale: float, eulerian_integral_length: float, lagrangian_integral_time: float, lagrangian_taylor_microscale_time: float) -> Union[float, NDArray[np.float64]]
```

Compute the variance of particle relative-velocity fluctuations.

This function calculates the variance of particle relative-velocity
fluctuations using the following equation:

Where the equation is:

- σ² = ⟨(v'²)⟩₁ + ⟨(v'²)⟩₂ - 2⟨v'¹ v'²⟩
    - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.

**Parameters**

- `fluid_rms_velocity` (`float`) — Fluid RMS fluctuation velocity [m/s].
- `collisional_radius` (`NDArray[np.float64]`) — Distance between two colliding droplets [m].
- `particle_inertia_time` (`NDArray[np.float64]`) — Inertia timescale of droplet 1 [s].
- `particle_velocity` (`NDArray[np.float64]`) — Droplet velocity [m/s].
- `taylor_microscale` (`float`) — Taylor microscale [m].
- `eulerian_integral_length` (`float`) — Eulerian integral length scale [m].
- `lagrangian_integral_time` (`float`) — Lagrangian integral time scale [s].

**Returns**

- `σ²` (`Union[float, NDArray[np.float64]]`) — Variance of the particle relative-velocity fluctuation [m²/s²].

**Examples**

```py
import numpy as np
sigma_sq = get_relative_velocity_variance(
    fluid_rms_velocity=0.3,
    collisional_radius=np.array([1e-4, 2e-4]),
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    taylor_microscale=0.01,
    eulerian_integral_length=0.1,
    lagrangian_integral_time=0.5,
    lagrangian_taylor_microscale_time=0.05
)
# Output: array([...])
```

> **References**
>
> - Ayala, O. et al. (2008). Effects of turbulence on the geometric
>   collision rate of sedimenting droplets. Part 2. Theory and
>   parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L47-L152)
