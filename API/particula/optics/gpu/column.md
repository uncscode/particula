# particula.optics.gpu.column

*module*

Device optical depth, direct-beam transmission and mass opacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/column.py#L1-L1)

## particula.optics.gpu.column.column_optics_kernel

*function*

```python
def column_optics_kernel(extinction: wp.array2d(dtype=wp.float64), path_length: wp.array2d(dtype=wp.float64), input_status: wp.array2d(dtype=wp.int32), layer_depth: wp.array2d(dtype=wp.float64), total_depth: wp.array(dtype=wp.float64), transmission: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Integrate (columns,layers) coefficients along explicit lengths.

Extinction [m⁻¹], lengths [m], and outputs [1]. Status codes 1 invalid,
2 upstream failure, 3 overflow; failed partial results are not usable.
Empty columns have zero depth and unit transmission.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/column.py#L8-L53)

## particula.optics.gpu.column.mass_opacity_kernel

*function*

```python
def mass_opacity_kernel(coefficient: wp.array(dtype=wp.float64), mass_density: wp.array(dtype=wp.float64), input_status: wp.array(dtype=wp.int32), opacity: wp.array(dtype=wp.float64), valid: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32))
```

Compute mass-specific opacity [m²/kg] from m⁻¹ and kg/m³.

Zero/zero returns zero with valid=0; nonzero/zero is invalid. Status codes
1 invalid input, 2 upstream failure, 3 nonfinite result.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/column.py#L56-L95)
