# particula.optics.gpu.population

*module*

Device population weighting and incoherent optical reductions.

Caller-provided arrays retain stable shapes. Status 0 means success, 1 invalid
input, 2 upstream failure, 3 nonfinite result. Failed output must not be used.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/population.py#L1-L1)

## particula.optics.gpu.population.finite_output

*function*

```python
def finite_output(output: wp.array2d(dtype=wp.float64), group: int) -> bool
```

Check a nine-column optical result for finite values.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/population.py#L67-L73)

## particula.optics.gpu.population.population_mueller_kernel

*function*

```python
def population_mueller_kernel(matrices: wp.array4d(dtype=wp.float64), weights: wp.array2d(dtype=wp.float64), input_status: wp.array3d(dtype=wp.int32), output: wp.array3d(dtype=wp.float64), status: wp.array3d(dtype=wp.int32))
```

Reduce (groups,particles,angles,16) differential Mueller matrices.

Flatten the final 4x4 in row-major order. Output is (groups,angles,16)
[m⁻¹/sr]; signed polarized elements are preserved. Status is per element.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/population.py#L131-L162)

## particula.optics.gpu.population.population_optics_kernel

*function*

```python
def population_optics_kernel(cross_sections: wp.array3d(dtype=wp.float64), asymmetry: wp.array2d(dtype=wp.float64), weights: wp.array2d(dtype=wp.float64), input_status: wp.array2d(dtype=wp.int32), output: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Reduce (groups,particles,5) cross sections to (groups,9) outputs.

Groups can flatten box/wavelength axes. Cross-section columns are
extinction, scattering, absorption, backscatter, radiation pressure [m²].
Outputs are those five coefficients [m⁻¹], SSA, g, extinction-valid,
scattering-valid. input_status combines optical and weight admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/population.py#L76-L128)

## particula.optics.gpu.population.population_weights_kernel

*function*

```python
def population_weights_kernel(concentration: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), radius_grid: wp.array(dtype=wp.float64), measure: int, weights: wp.array2d(dtype=wp.float64), status: wp.array2d(dtype=wp.int32))
```

Produce (boxes,particles) number weights [m⁻³].

Measure 0 is discrete density, 1 represented count divided by box volume,
2 radius PDF times trapezoidal widths. PDF grid length matches particles
and must be at least two. Volume/grid arrays are only read when needed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/population.py#L12-L64)
