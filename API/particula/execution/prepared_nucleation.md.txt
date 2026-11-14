# particula.execution.prepared_nucleation

*module*

Translation of the existing unit-count, neutral GPU nucleation model.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation.py#L1-L1)

## particula.execution.prepared_nucleation._BOUND_NAMES

*attribute* · *module attribute*

```python
_BOUND_NAMES = ('radius_cubed_relative_error', 'mean_radius_relative_error', 'surface_relative_error', 'diversity_absolute_error')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation.py#L12-L17)

## particula.execution.prepared_nucleation.compile_nucleation

*function*

```python
def compile_nucleation(plan: _ProcessPlan, owner: Any)
```

Retain all source/exhaustion buffers and explicit status observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation.py#L101-L197)

## particula.execution.prepared_nucleation.plan_nucleation

*function*

```python
def plan_nucleation(process, aerosol) -> _ProcessPlan
```

Freeze full-gas provenance, source physics and exhaustion controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation.py#L20-L98)
