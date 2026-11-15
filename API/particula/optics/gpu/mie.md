# particula.optics.gpu.mie

*module*

Direct float64 Warp Mie kernels with caller-owned buffers.

Equations and SI/index conventions match particula.optics.mie. Inputs here
are flattened independent cases of relative n+ik and dimensionless x.
Status codes: 0 success, 1 invalid input/capacity, 2 convergence failure,
3 nonfinite results, 4 unphysical efficiencies beyond roundoff. Failed
results must not be consumed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L1-L1)

## particula.optics.gpu.mie._CF_TINY

*attribute* · *module attribute*

```python
_CF_TINY = wp.constant(wp.float64(1e-300))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L22-L22)

## particula.optics.gpu.mie._CF_TOLERANCE_SQUARED

*attribute* · *module attribute*

```python
_CF_TOLERANCE_SQUARED = wp.constant(wp.float64(4e-28))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L23-L23)

## particula.optics.gpu.mie._ORDER_GROWTH

*attribute* · *module attribute*

```python
_ORDER_GROWTH = wp.constant(wp.float64(4.05))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L25-L25)

## particula.optics.gpu.mie._ROUNDOFF

*attribute* · *module attribute*

```python
_ROUNDOFF = wp.constant(wp.float64(64 * 2.220446049250313e-16))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L26-L26)

## particula.optics.gpu.mie._SMALL_TOLERANCE

*attribute* · *module attribute*

```python
_SMALL_TOLERANCE = wp.constant(wp.float64(2e-16))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L24-L24)

## particula.optics.gpu.mie.logarithmic_derivative

*function*

```python
def logarithmic_derivative(z: wp.vec2d, count: int)
```

Return the continued-fraction derivative and convergence flag.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L45-L70)

## particula.optics.gpu.mie.mie_coefficients_kernel

*function*

```python
def mie_coefficients_kernel(relative_index: wp.array(dtype=wp.vec2d), size_parameter: wp.array(dtype=wp.float64), derivatives: wp.array2d(dtype=wp.vec2d), electric: wp.array2d(dtype=wp.vec2d), magnetic: wp.array2d(dtype=wp.vec2d), term_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32))
```

Evaluate one sphere per thread; overwrite counts/status every call.

Workspace derivatives has coefficient width+1 columns. Electric and
magnetic arrays share shape (cases, capacity). Trailing coefficients are
zeroed. No host transfers, allocations or synchronization occur here.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L73-L172)

## particula.optics.gpu.mie.mie_efficiencies_kernel

*function*

```python
def mie_efficiencies_kernel(relative_index: wp.array(dtype=wp.vec2d), size_parameter: wp.array(dtype=wp.float64), electric: wp.array2d(dtype=wp.vec2d), magnetic: wp.array2d(dtype=wp.vec2d), term_count: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), efficiencies: wp.array2d(dtype=wp.float64))
```

Write Qext,Qsca,Qabs,g,Qback,Qpr and a scattering-valid flag.

Shape is (cases,7). Failed upstream cases produce zero outputs but retain
nonzero status; status must be checked before results are used.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L175-L262)

## particula.optics.gpu.mie.regular_small

*function*

```python
def regular_small(x: wp.float64, order: int) -> wp.float64
```

Evaluate x*j_order(x) using the convergent small-argument series.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/mie.py#L29-L42)
