# particula.optics.gpu.angular

*module*

Direct device angular scattering and polarized optical operations.

Conventions match the CPU optics modules: S1 perpendicular, S2 parallel,
Stokes I,Q,U,V with V=-2 Im(Eparallel Eperpendicular*). Every output is
caller-owned; kernels allocate and transfer no arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/angular.py#L1-L1)

## particula.optics.gpu.angular.apply_mueller_kernel

*function*

```python
def apply_mueller_kernel(matrix: wp.array4d(dtype=wp.float64), matrix_status: wp.array2d(dtype=wp.int32), incident: wp.array2d(dtype=wp.vec4d), input_rotation: wp.array2d(dtype=wp.float64), output_rotation: wp.array2d(dtype=wp.float64), outgoing: wp.array2d(dtype=wp.vec4d), status: wp.array2d(dtype=wp.int32))
```

Apply R(output) M R(input) on-device to each (case,angle).

Status 1 marks nonfinite vector/rotation/matrix, 2 upstream failure,
3 numerical failure.
No distance scaling or intensity normalization is applied.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/angular.py#L146-L196)

## particula.optics.gpu.angular.mueller_kernel

*function*

```python
def mueller_kernel(perpendicular: wp.array2d(dtype=wp.vec2d), parallel: wp.array2d(dtype=wp.vec2d), amplitude_status: wp.array2d(dtype=wp.int32), scale: wp.array(dtype=wp.float64), matrix: wp.array4d(dtype=wp.float64), status: wp.array2d(dtype=wp.int32))
```

Construct full Mueller matrices with an explicit per-case scale.

Use scale=1 for raw amplitudes; (lambda/(2*pi*n_medium))² for m²/sr;
divide that factor by scattering cross section for normalized phase.
Zero scattering uses scale=0 and an externally retained validity mask.
Shape is (cases,angles,4,4); status 1 invalid scale, 2 failed amplitudes,
3 numerical failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/angular.py#L79-L130)

## particula.optics.gpu.angular.rotated_stokes

*function*

```python
def rotated_stokes(vector: wp.vec4d, angle: wp.float64) -> wp.vec4d
```

Rotate basis axes by angle, with Q'=cos(2a)Q+sin(2a)U.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/angular.py#L133-L143)

## particula.optics.gpu.angular.scattering_amplitudes_kernel

*function*

```python
def scattering_amplitudes_kernel(electric: wp.array2d(dtype=wp.vec2d), magnetic: wp.array2d(dtype=wp.vec2d), term_count: wp.array(dtype=wp.int32), coefficient_status: wp.array(dtype=wp.int32), angles: wp.array(dtype=wp.float64), perpendicular: wp.array2d(dtype=wp.vec2d), parallel: wp.array2d(dtype=wp.vec2d), status: wp.array2d(dtype=wp.int32))
```

Evaluate one (case,angle) per thread using cached coefficients.

Outputs have shape (cases,angles). Status 0 is success, 1 invalid angle,
2 failed coefficients, 3 numerical failure. Zero outputs on failure
are not valid results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/angular.py#L15-L76)
