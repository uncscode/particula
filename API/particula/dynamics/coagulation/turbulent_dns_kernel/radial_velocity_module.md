# particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module

*module*

Radial relative velocity calculation module.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module.get_radial_relative_velocity_ao2008

*function*

```python
def get_radial_relative_velocity_ao2008(velocity_dispersion: Union[float, NDArray[np.float64]], particle_inertia_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the radial relative velocity based on Ayala et al. (2008).

This function estimates the radial relative velocity between pairs of
particles considering both turbulent velocity dispersion and gravitational
acceleration. The conceptual form is:

- ⟨|wᵣ|⟩ = √(2/π) × √(σ² + (π/8) × (τₚ₁ - τₚ₂)² × g²)
    - wᵣ is the radial relative velocity in m/s,
    - σ is the turbulence velocity dispersion in m/s,
    - τₚ₁, τₚ₂ are the inertia timescales (s),
    - g is the gravitational acceleration (m/s²).

**Parameters**

- `velocity_dispersion` (`Union[float, NDArray[np.float64]]`) — Turbulence velocity dispersion (σ) in m/s.
- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Inertia timescale(s) of particles [s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The radial relative velocity ⟨|wᵣ|⟩ in m/s.

**Examples**

```py
import numpy as np
import particula as par

# Example usage (currently raises NotImplementedError)
try:
    rv = par.dynamics.get_radial_relative_velocity_ao2008(
        1.0, np.array([0.05, 0.1])
    )
except NotImplementedError as e:
    print(e)
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>   the geometric collision rate of sedimenting droplets. Part 2. Theory
>   and parameterization. New Journal of Physics, 10.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py#L75-L132)

## particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module.get_radial_relative_velocity_dz2002

*function*

```python
def get_radial_relative_velocity_dz2002(velocity_dispersion: Union[float, NDArray[np.float64]], particle_inertia_time: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Compute the radial relative velocity based on Dodin and Elperin (2002).

This function calculates the radial relative velocity between pairs of
particles under turbulent conditions, capturing the effects of different
inertia timescales. The equation is:

- ⟨|wᵣ|⟩ = √(2/π) × σ × f(b)
    - wᵣ is the radial relative velocity in m/s,
    - σ is the turbulence velocity dispersion in m/s,
    - b = (g × |τₚᵢ - τₚⱼ|) / (√2 × σ),
    - f(b) = ½√π (b + 0.5 / b) erf(b) + ½ exp(-b²).

**Parameters**

- `velocity_dispersion` (`Union[float, NDArray[np.float64]]`) — Turbulence velocity dispersion (σ) in m/s.
- `particle_inertia_time` (`NDArray[np.float64]`) — Inertia timescale(s) (τₚ) in seconds.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The radial relative velocity ⟨|wᵣ|⟩ in m/s.

**Examples**

```py
import numpy as np
import particula as par

# Example with an array of inertia times
result = par.dynamics.get_radial_relative_velocity_dz2002(
    1.0, np.array([0.1, 0.2, 0.3])
)
print(result)
```

> **References**
>
> - Dodin, Z., & Elperin, T. (2002). Phys. Fluids, 14, 2921–2924.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py#L13-L72)
