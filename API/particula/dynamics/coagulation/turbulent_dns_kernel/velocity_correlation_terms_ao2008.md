# particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008

*module*

Velocity correlation terms for the DNS kernel of the turbulent coagulation
model by Ayala 2008.

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L1-L1)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_b1

*function*

```python
def compute_b1(z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute b₁, a dimensionless parameter in the Ayala 2008 model.

- b₁ = (1 + √(1 - 2z²)) / (2 √(1 - 2z²))
    - z is τ_T / T_L.

**Parameters**

- `z` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — b₁ value [dimensionless].

**Examples**

```py
b1_val = compute_b1(0.5)
# Output: 0.866
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L98-L126)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_b2

*function*

```python
def compute_b2(z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute b₂, a dimensionless parameter in the Ayala 2008 model.

- b₂ = (1 - √(1 - 2z²)) / (2 √(1 - 2z²))
    - z is τ_T / T_L.

**Parameters**

- `z` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — b₂ value [dimensionless].

**Examples**

```py
b2_val = compute_b2(0.5)
# Output: 0.134
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L129-L157)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_beta

*function*

```python
def compute_beta(taylor_microscale: Union[float, NDArray[np.float64]], eulerian_integral_length: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute β, the ratio of microscale to integral length scale.

- β = (√2 × λ) / L_e
    - λ is Taylor microscale [m].
    - L_e is Eulerian integral length scale [m].

**Parameters**

- `taylor_microscale` (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m].
- `eulerian_integral_length` (`Union[float, NDArray[np.float64]]`) — Eulerian integral length scale [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — β value [dimensionless].

**Examples**

```py
beta_val = compute_beta(0.001, 0.1)
# Output: 0.01414
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L60-L95)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_c1

*function*

```python
def compute_c1(z: Union[float, NDArray[np.float64]], lagrangian_integral_scale: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute c₁, a dimensionless timescale factor in the Ayala 2008 model.

- c₁ = ((1 + √(1 - 2z²)) × T_L) / 2
    - z is τ_T / T_L.
    - T_L is the Lagrangian integral timescale [s].

**Parameters**

- `z` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].
- `lagrangian_integral_scale` (`Union[float, NDArray[np.float64]]`) — Lagrangian integral timescale [s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — c₁ value [dimensionless].

**Examples**

```py
c1_val = compute_c1(0.5, 1.0)
# Output: 0.933
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L160-L190)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_c2

*function*

```python
def compute_c2(z: Union[float, NDArray[np.float64]], lagrangian_integral_scale: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute c₂, a dimensionless timescale factor in the Ayala 2008 model.

- c₂ = ((1 - √(1 - 2z²)) × T_L) / 2
    - z is τ_T / T_L.
    - T_L is the Lagrangian integral timescale [s].

**Parameters**

- `z` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].
- `lagrangian_integral_scale` (`Union[float, NDArray[np.float64]]`) — Lagrangian integral timescale [s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — c₂ value [dimensionless].

**Examples**

```py
c2_val = compute_c2(0.5, 1.0)
# Output: 0.067
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L193-L223)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_d1

*function*

```python
def compute_d1(beta: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute d₁, another dimensionless coefficient from Ayala 2008.

- d₁ = (1 + √(1 - 2β²)) / (2 √(1 - 2β²))
    - β is defined as β = (√2 × λ) / L_e.

**Parameters**

- `beta` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — d₁ value [dimensionless].

**Examples**

```py
d1_val = compute_d1(0.5)
# Output: 0.866
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L226-L254)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_d2

*function*

```python
def compute_d2(beta: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute d₂, another dimensionless coefficient from Ayala 2008.

- d₂ = (1 - √(1 - 2β²)) / (2 √(1 - 2β²))
    - β is defined as β = (√2 × λ) / L_e.

**Parameters**

- `beta` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — d₂ value [dimensionless].

**Examples**

```py
d2_val = compute_d2(0.5)
# Output: 0.134
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L257-L285)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_e1

*function*

```python
def compute_e1(beta: Union[float, NDArray[np.float64]], eulerian_integral_length: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute e₁, which scales the integral length by a factor in Ayala 2008.

- e₁ = ((1 + √(1 - 2β²)) × L_e) / 2
    - β is defined as β = (√2 × λ) / L_e.
    - L_e is the Eulerian integral length scale [m].

**Parameters**

- `beta` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].
- `eulerian_integral_length` (`Union[float, NDArray[np.float64]]`) — Eulerian integral length scale [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — e₁ value [dimensionless].

**Examples**

```py
e1_val = compute_e1(0.5, 0.1)
# Output: 0.0866
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L288-L318)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_e2

*function*

```python
def compute_e2(beta: Union[float, NDArray[np.float64]], eulerian_integral_length: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute e₂, which scales the integral length by a factor in Ayala 2008.

- e₂ = ((1 - √(1 - 2β²)) × L_e) / 2
    - β is defined as β = (√2 × λ) / L_e.
    - L_e is the Eulerian integral length scale [m].

**Parameters**

- `beta` (`Union[float, NDArray[np.float64]]`) — A dimensionless parameter related to turbulence [-].
- `eulerian_integral_length` (`Union[float, NDArray[np.float64]]`) — Eulerian integral length scale [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — e₂ value [dimensionless].

**Examples**

```py
e2_val = compute_e2(0.5, 0.1)
# Output: 0.0134
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L321-L351)

## particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008.compute_z

*function*

```python
def compute_z(lagrangian_taylor_microscale_time: Union[float, NDArray[np.float64]], lagrangian_integral_scale: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute z, the ratio of Taylor microscale time to Lagrangian timescale.

Where the equation is
- z = τ_T / T_L
    - τ_T (lagrangian_taylor_microscale_time) is the Lagrangian Taylor
        microscale time [s].
    - T_L (lagrangian_integral_scale) is the Lagrangian integral
        timescale [s].

**Parameters**

- `lagrangian_taylor_microscale_time` (`Union[float, NDArray[np.float64]]`) — Lagrangian Taylor microscale time [s].
- `lagrangian_integral_scale` (`Union[float, NDArray[np.float64]]`) — Lagrangian integral timescale [s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — z value [dimensionless].

**Examples**

```py
example_z = compute_z(0.5, 1.0)
# Output: 0.5
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
>     the geometric collision rate of sedimenting droplets. Part 2.
>     Theory and parameterization. New Journal of Physics, 10.
>     https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L18-L57)
