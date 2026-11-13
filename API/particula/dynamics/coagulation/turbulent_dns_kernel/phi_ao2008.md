# particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008

*module*

Compute the function Φ(α, φ) for the given particle properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms

*class*

```python
class PhiComputeTerms(NamedTuple)
```

Bases: `NamedTuple`

Parameters for computing Φ function terms.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L11-L19)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.alpha

*attribute* · *instance attribute*

```python
alpha: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L18-L18)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.phi

*attribute* · *instance attribute*

```python
phi: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L19-L19)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.tau1

*attribute* · *instance attribute*

```python
tau1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L16-L16)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.tau2

*attribute* · *instance attribute*

```python
tau2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L17-L17)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.v1

*attribute* · *instance attribute*

```python
v1: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L14-L14)

### particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.PhiComputeTerms.v2

*attribute* · *instance attribute*

```python
v2: Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L15-L15)

## particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008._compute_phi_term1

*function*

```python
def _compute_phi_term1(terms: PhiComputeTerms) -> Union[float, NDArray[np.float64]]
```

Compute the first term of the Φ function.

 term_1 = {
    1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )
}        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L131-L153)

## particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008._compute_phi_term2

*function*

```python
def _compute_phi_term2(terms: PhiComputeTerms) -> Union[float, NDArray[np.float64]]
```

Compute the second term of the Φ function.

term₂ =
{
  4 / [ (v₂ / φ)² − ( (1 / τ₂) + (1 / α) )² ]       − 1 / [ (v₂ / φ) + (1 / τ₂) + (1 / α) ]²       − 1 / [ (v₂ / φ) − (1 / τ₂) − (1 / α) ]²     }
× v₂ /       [ 2 φ ( (1 / τ₁) − (1 / α) + ( (1 / τ₂) + (1 / α) ) × (v₁ / v₂) ) ]

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L156-L188)

## particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008._compute_phi_term3

*function*

```python
def _compute_phi_term3(terms: PhiComputeTerms) -> Union[float, NDArray[np.float64]]
```

Compute the third term of the Φ function.

term_3 =
{
    2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )
}
    ×  1 / ( 2φ ( (vₚ₁ - vₚ₂/φ ) + (1 / τₚ₁) + (1 / τₚ₂) ) )

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L191-L221)

## particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008.get_phi_ao2008

*function*

```python
def get_phi_ao2008(alpha: Union[float, NDArray[np.float64]], phi: Union[float, NDArray[np.float64]], particle_inertia_time: Union[float, NDArray[np.float64]], particle_velocity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the function Φ(α, φ) for the given particle properties using
Ayala et al. (2008).

This function calculates Φ(α, φ) when vₚ₁ > vₚ₂ by considering the
velocities (vₚ₁, vₚ₂) and inertia times (τₚ₁, τₚ₂). The equation is:

Φ(α, φ), for vₚ₁ > vₚ₂ =
    {  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) ) }        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )

    + {  4 / ( (vₚ₂ / φ)² - ( (1 / τₚ₂) + (1 / α) )² )        -  1 / ( (vₚ₂ / φ) + (1 / τₚ₂) + (1 / α) )²        -  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )²  }        ×  ( vₚ₂ / ( 2 φ ( (1 / τₚ₁) - (1 / α)         + ( (1 / τₚ₂) + (1 / α) ) (vₚ₁ / vₚ₂) ) ) )

    + {  2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )  }        ×  1 / ( 2φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) ) )

  - v₁ and v₂: Velocities of particles 1 and 2 in m/s.
  - τ₁ and τ₂: Inertia timescales of particles 1 and 2 in s.
  - α: Turbulent interaction parameter (dimensionless).
  - φ: Characteristic velocity (m/s).

**Parameters**

- `alpha` (`Union[float, NDArray[np.float64]]`) — Turbulence/droplet interaction parameter (dimensionless).
- `phi` (`Union[float, NDArray[np.float64]]`) — Characteristic velocity parameter (m/s).
- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Inertia timescales τₚ₁ and τₚ₂ (s).
- `particle_velocity` (`Union[float, NDArray[np.float64]]`) — Velocities vₚ₁ and vₚ₂ (m/s).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The computed Φ(α, φ) (dimensionless).

**Examples**

```py
import numpy as np
from particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008
    import get_phi_ao2008

alpha_val = 0.3
phi_val = 0.1
inertia_times = np.array([0.05, 0.06])
velocities = np.array([0.2, 0.18])
result = get_phi_ao2008(alpha_val, phi_val, inertia_times, velocities)
print(result)
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>   the geometric collision rate of sedimenting droplets. Part 2.
>   Theory and parameterization. New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L22-L128)
