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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L66-L103)

## particula.gpu.kernels._population_validation.array_field

*function*

```python
def array_field(container: Any, name: str, shape: tuple[int, ...], dtype: Any, device: Any) -> Any
```

Require a contiguous same-device field without reading its payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L14-L29)

## particula.gpu.kernels._population_validation.memory_range

*function*

```python
def memory_range(array: Any) -> tuple[int, int]
```

Return a checked contiguous array interval for all supported dtypes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L32-L49)

## particula.gpu.kernels._population_validation.reject_writer_aliases

*function*

```python
def reject_writer_aliases(writers: tuple[Any, ...], protected: tuple[Any, ...]) -> None
```

Reject writers overlapping other writers or protected state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L52-L63)

## particula.gpu.kernels._population_validation.validate_population

*function*

```python
def validate_population(particles: Any) -> tuple[Any, ...]
```

Validate native PMF/resolved representation and return primary fields.

This establishes storage interpretation only, not algorithm admission or
active-slot geometry. Callers retain those process-specific checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_population_validation.py#L106-L139)
