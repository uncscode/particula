# particula.gpu.kernels.ion_coagulation

*module*

Validate explicit ions and cache attachment rates before collision draws.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L1-L1)

## particula.gpu.kernels.ion_coagulation._REFERENCE_PRESSURE

*attribute* · *module attribute*

```python
_REFERENCE_PRESSURE = wp.constant(wp.float64(101325.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L14-L14)

## particula.gpu.kernels.ion_coagulation._REFERENCE_TEMPERATURE

*attribute* · *module attribute*

```python
_REFERENCE_TEMPERATURE = wp.constant(wp.float64(298.15))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L13-L13)

## particula.gpu.kernels.ion_coagulation.cache_attachment_rates

*function*

```python
def cache_attachment_rates(kinds: wp.array2d(dtype=wp.int32), presence: wp.array2d(dtype=wp.int32), radii: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), omitted: wp.int32, tables: wp.array3d(dtype=wp.float64), rates: wp.array3d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Cache only needed polarities; reject invalid rates before sampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L72-L96)

## particula.gpu.kernels.ion_coagulation.cache_mobility_rates

*function*

```python
def cache_mobility_rates(kinds: wp.array2d(dtype=wp.int32), presence: wp.array2d(dtype=wp.int32), radii: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), masses: wp.array3d(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), positive_mass: wp.float64, negative_mass: wp.float64, positive_mobility: wp.float64, negative_mobility: wp.float64, ion_radius: wp.float64, rates: wp.array3d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Refresh variable-T/P rates from current wet aerosol radii and charge.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L99-L145)

## particula.gpu.kernels.ion_coagulation.classify_ions

*function*

```python
def classify_ions(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), positive_species: wp.int32, negative_species: wp.int32, positive_mass: wp.float64, negative_mass: wp.float64, kinds: wp.array2d(dtype=wp.int32), presence: wp.array2d(dtype=wp.int32), failure: wp.array(dtype=wp.int32), attachment_model: wp.int32)
```

Validate unit particles and the fixed LF2013 air reference per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_coagulation.py#L17-L69)
