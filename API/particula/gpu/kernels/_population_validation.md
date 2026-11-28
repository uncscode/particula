# particula.gpu.kernels._population_validation

*module*

Read-only metadata, schema and storage checks for native population users.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L1-L1)

## particula.gpu.kernels._population_validation._scan_population

*function*

```python
def _scan_population(masses: Any, counts: Any, charge: Any, density: Any, volume: Any, code: int, invalid: Any) -> None
```

Check representation values on device without a caller-array readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L76-L113)

## particula.gpu.kernels._population_validation.array_field

*function*

```python
def array_field(container: Any, name: str, shape: tuple[int, ...], dtype: Any, device: Any) -> Any
```

Require a contiguous same-device field without reading its payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L24-L39)

## particula.gpu.kernels._population_validation.memory_range

*function*

```python
def memory_range(array: Any) -> tuple[int, int]
```

Return a checked contiguous array interval for all supported dtypes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L42-L59)

## particula.gpu.kernels._population_validation.native_slot_density

*function*

```python
def native_slot_density(value: wp.float64, box_volume: wp.float64, resolved: bool) -> wp.float64
```

Read an admitted discrete density or resolved count as m⁻³.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L14-L21)

## particula.gpu.kernels._population_validation.reject_writer_aliases

*function*

```python
def reject_writer_aliases(writers: tuple[Any, ...], protected: tuple[Any, ...]) -> None
```

Reject writers overlapping other writers or protected state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L62-L73)

## particula.gpu.kernels._population_validation.validate_population

*function*

```python
def validate_population(particles: Any) -> tuple[Any, ...]
```

Validate native PMF/resolved representation and return primary fields.

This establishes storage interpretation only, not algorithm admission or
active-slot geometry. Callers retain those process-specific checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L116-L149)
