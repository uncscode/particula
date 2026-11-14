# particula.execution.prepared_diagnostics

*module*

Bounded, deterministic particle reductions for CPU observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_diagnostics.py#L1-L1)

## particula.execution.prepared_diagnostics._finish_partials

*function*

```python
def _finish_partials(partials: wp.array3d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), resolved: wp.int32, output: wp.array2d(dtype=wp.float64))
```

Normalize counts by volume; retain density and active-slot count.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_diagnostics.py#L35-L50)

## particula.execution.prepared_diagnostics._particle_partials

*function*

```python
def _particle_partials(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), partials: wp.array3d(dtype=wp.float64))
```

Reduce fixed 256-slot chunks in a reproducible ascending order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_diagnostics.py#L11-L32)

## particula.execution.prepared_diagnostics.setup_diagnostics

*function*

```python
def setup_diagnostics(owner: Any)
```

Allocate reduction storage once and return a retained observer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_diagnostics.py#L53-L111)
