# particula.gpu.kernels._wall_loss_native

*module*

Read-only native resolved wall-loss preflight before caller-owned writers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_wall_loss_native.py#L1-L1)

## particula.gpu.kernels._wall_loss_native._memory_range

*function*

```python
def _memory_range(array: Any) -> tuple[int, int]
```

Return a contiguous array's byte range without payload access.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_wall_loss_native.py#L80-L97)

## particula.gpu.kernels._wall_loss_native._scan_population

*function*

```python
def _scan_population(masses: Any, concentration: Any, density: Any, volume: Any, invalid: Any) -> None
```

Validate count density and representable material geometry per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_wall_loss_native.py#L14-L50)

## particula.gpu.kernels._wall_loss_native.validate_ownership

*function*

```python
def validate_ownership(particles: Any, states: Any, protected_inputs: tuple[Any, ...]) -> None
```

Reject writer overlap with another writer or protected input storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_wall_loss_native.py#L100-L129)

## particula.gpu.kernels._wall_loss_native.validate_physics

*function*

```python
def validate_physics(particles: Any) -> None
```

Admit current raw-count physics without imposing equal weights.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_wall_loss_native.py#L53-L77)
