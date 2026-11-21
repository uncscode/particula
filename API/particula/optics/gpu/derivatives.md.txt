# particula.optics.gpu.derivatives

*module*

Analytic derivatives of the smooth Mie cross-section channels.

These local derivatives use the same evaluated coefficients as the forward
Mie calculation. They avoid differentiating through convergence-controlled
loops and can seed a Warp tape for upstream size and composition physics.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/derivatives.py#L1-L1)

## particula.optics.gpu.derivatives._coefficient_slopes

*function*

```python
def _coefficient_slopes(d: wp.vec2d, dm: wp.vec2d, dx: wp.vec2d, a: wp.vec2d, psi: wp.float64, psi_derivative: wp.float64, xi: wp.vec2d, xi_derivative: wp.vec2d, previous_xi: wp.vec2d, previous_xi_derivative: wp.vec2d, previous_psi_derivative: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/derivatives.py#L20-L50)

## particula.optics.gpu.derivatives._slopes

*function*

```python
def _slopes(a: wp.vec2d, index_slope: wp.vec2d, size_slope: wp.vec2d)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/derivatives.py#L53-L62)

## particula.optics.gpu.derivatives.mie_cross_section_sums_jacobian

*function*

```python
def mie_cross_section_sums_jacobian(relative_index: wp.array(dtype=wp.vec2d), size_parameter: wp.array(dtype=wp.float64), derivatives: wp.array2d(dtype=wp.vec2d), electric: wp.array2d(dtype=wp.vec2d), magnetic: wp.array2d(dtype=wp.vec2d), term_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), extinction: wp.array(dtype=wp.vec3d), scattering: wp.array(dtype=wp.vec3d))
```

Differentiate the dimensionless Mie sums before the 2/x² factor.

Each output vector is ordered (relative n, relative k, x). Physical cross
sections multiply the sums by 2*pi*(lambda/(2*pi*n_medium))². At k=0
the index derivative is the right derivative of the unrounded solution.
Inputs require successful coefficients and positive x in the qualified
gradient domain. A failed case retains or sets nonzero status.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/derivatives.py#L65-L178)
