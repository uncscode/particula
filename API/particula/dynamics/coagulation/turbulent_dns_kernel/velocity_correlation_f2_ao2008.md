# particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_f2_ao2008

*module*

Velocity correlation terms for the two-point velocity correlation function
f₂(R) from Ayala et al. (2008).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_f2_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_f2_ao2008.get_f2_longitudinal_velocity_correlation

*function*

```python
def get_f2_longitudinal_velocity_correlation(collisional_radius: Union[float, NDArray[np.float64]], taylor_microscale: Union[float, NDArray[np.float64]], eulerian_integral_length: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the longitudinal velocity correlation function f₂(R) from
Ayala et al. (2008).

This function describes the correlation of velocity fluctuations as a
function of collisional radius R between two colliding droplets.

Where the equation is:

- f₂(R) = 1 / (2√(1 - 2β²)) × {
    (1 + √(1 - 2β²)) exp[-2R / ((1 + √(1 - 2β²)) L_e)]
    - (1 - √(1 - 2β²)) exp[-2R / ((1 - √(1 - 2β²)) L_e)]
  }
    - f₂(R) is the longitudinal velocity correlation function [-].
    - R is the collisional radius [m].
    - β = (√2 * λ) / L_e
    - λ (taylor_microscale) : Taylor microscale [m].
    - L_e (eulerian_integral_length) : Eulerian integral length scale
      [m].

**Parameters**

- `collisional_radius` (`Union[float, NDArray[np.float64]]`) — Distance between two colliding droplets [m].
- `taylor_microscale` (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m].
- `eulerian_integral_length` (`Union[float, NDArray[np.float64]]`) — Eulerian integral length scale [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — f₂(R) value [dimensionless].

**Examples**

```py
import numpy as np
example_f2 = get_f2_longitudinal_velocity_correlation(
    collisional_radius=np.array([1e-4, 2e-4]),
    taylor_microscale=1e-3,
    eulerian_integral_length=1e-2,
)
# Output: array([...])
```

References:
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2. Theory
  and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_f2_ao2008.py#L15-L87)
