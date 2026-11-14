# particula.execution.prepared_condensation

*module*

Host-side condensation capability checks and retained GPU compilation.

The direct solver has four equal substeps, one activity-corrected material,
and a mixture Kelvin density. Admit only CPU configuration that can be
represented by those controls; do not discard unsupported strategy choices.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L1-L1)

## particula.execution.prepared_condensation._activity_plan

*function*

```python
def _activity_plan(activity, particles, gas, pairs, vapor)
```

Require one participating volatile material lane.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L116-L190)

## particula.execution.prepared_condensation._surface_plan

*function*

```python
def _surface_plan(surface, particles)
```

Require an exact static or single-phase volume-weighted surface law.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L70-L113)

## particula.execution.prepared_condensation._vapor_plan

*function*

```python
def _vapor_plan(strategy, width)
```

Freeze constant/Buck definitions in complete gas order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L34-L67)

## particula.execution.prepared_condensation._vector

*function*

```python
def _vector(value, width, name, *, positive=False)
```

Normalize finite nonnegative scalar/vector metadata without aliasing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L19-L31)

## particula.execution.prepared_condensation.compile_condensation

*function*

```python
def compile_condensation(plan: _ProcessPlan, owner: Any)
```

Allocate retained thermodynamics, species maps and transport sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L270-L367)

## particula.execution.prepared_condensation.plan_condensation

*function*

```python
def plan_condensation(process, aerosol) -> _ProcessPlan
```

Freeze representable physical parameters before any upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L193-L267)
