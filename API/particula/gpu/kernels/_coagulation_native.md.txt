# particula.gpu.kernels._coagulation_native

*module*

Native resolved coagulation preflight; no caller-owned storage is written.

Population counts retain their physical volume authority. Equal positive
counts are necessary for the existing donor/recipient merge. Validation uses
private device flags only; preparation may synchronize, enqueue must not.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L1-L1)

## particula.gpu.kernels._coagulation_native._memory_range

*function*

```python
def _memory_range(array: Any) -> tuple[int, int]
```

Check contiguous supported storage and return its byte range.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L122-L138)

## particula.gpu.kernels._coagulation_native._validate_volume_and_derived

*function*

```python
def _validate_volume_and_derived(masses: Any, concentration: Any, density: Any, volume: Any, requested_volume: Any, invalid: Any) -> None
```

Check volume authority and finite derived population quantities.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L39-L67)

## particula.gpu.kernels._coagulation_native.validate_ownership

*function*

```python
def validate_ownership(particles: Any, pairs: Any, counts: Any, states: Any, protected_inputs: tuple[Any, ...]) -> None
```

Reject every native writer overlapping another writer or read source.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L141-L177)

## particula.gpu.kernels._coagulation_native.validate_physics

*function*

```python
def validate_physics(particles: Any, requested_volume: Any) -> None
```

Admit equal active counts and physical state before any state writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L70-L119)

## particula.gpu.kernels._coagulation_native.validate_schema

*function*

```python
def validate_schema(particles: Any) -> None
```

Require native fp64 primary storage before payload access.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_native.py#L14-L36)
