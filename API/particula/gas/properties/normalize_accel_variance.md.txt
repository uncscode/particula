# particula.gas.properties.normalize_accel_variance

*module*

Calculates the normalized acceleration variance in isotropic turbulence.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/normalize_accel_variance.py#L1-L1)

## particula.gas.properties.normalize_accel_variance.get_normalized_accel_variance_ao2008

*function*

```python
def get_normalized_accel_variance_ao2008(re_lambda: Union[float, NDArray[np.float64]], numerical_stability_epsilon: float = 1e-14) -> Union[float, NDArray[np.float64]]
```

Calculate the normalized acceleration variance in isotropic turbulence.

This coefficient describes the statistical behavior of acceleration
fluctuations in turbulent flows.

- a_o = (11 + 7 R_λ) / (205 + R_λ)
    - a_o is Normalized acceleration variance in isotropic turbulence [-].
    - R_λ is Taylor-microscale Reynolds number [-].

> **Where**
>
> - a_o (accel_variance) is Normalized acceleration variance in isotropic
>   turbulence [-].
> - R_λ (re_lambda) is Taylor-microscale Reynolds number [-].
> - ε (numerical_stability_epsilon) is Small number added to R_λ
>   for numerical stability.

**Parameters**

- `re_lambda` (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [-]

**Returns**

- `accel_variance` (`Union[float, NDArray[np.float64]]`) — Normalized acceleration variance [-]

**Examples**

```py title="Example Usage"
import particula as par
par.gas.get_normalized_accel_variance_ao2008(500.0)
# Output: ~0.05
```

References:
- The equivalent numerically stable version used is this.
    (7 + 11 / (R_λ + ε)) / (1 + 205 / (R_λ + ε))
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2. Theory
    and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/normalize_accel_variance.py#L11-L55)
