# particula.execution.prepared_condensation

*module*

Host-side condensation capability checks and retained GPU compilation.

The direct solver has four equal substeps, material activity corrections,
and a mixture Kelvin density. Admit only CPU configuration that can be
represented by those controls; do not discard unsupported strategy choices.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L1-L1)

## particula.execution.prepared_condensation._activity_plan

*function*

```python
def _activity_plan(activity, particles, gas, pairs, vapor)
```

Translate material activities to the direct solver configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L125-L211)

## particula.execution.prepared_condensation._surface_plan

*function*

```python
def _surface_plan(surface, particles)
```

Require an exact static or single-phase volume-weighted surface law.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L77-L122)

## particula.execution.prepared_condensation._vapor_plan

*function*

```python
def _vapor_plan(strategy, width, *, allow_zero=False)
```

Freeze constant/Buck definitions in complete gas order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L35-L74)

## particula.execution.prepared_condensation._vector

*function*

```python
def _vector(value, width, name, *, positive=False)
```

Normalize finite nonnegative scalar/vector metadata without aliasing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L20-L32)

## particula.execution.prepared_condensation.compile_condensation

*function*

```python
def compile_condensation(plan: _ProcessPlan, context: BuiltinPreparationContext)
```

Allocate retained thermodynamics, species maps and transport sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L303-L421)

## particula.execution.prepared_condensation.plan_condensation

*function*

```python
def plan_condensation(process, aerosol, *, allow_zero_vapor=False) -> _ProcessPlan
```

Freeze representable physical parameters before any upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_condensation.py#L214-L300)
