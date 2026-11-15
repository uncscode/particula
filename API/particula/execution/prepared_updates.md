# particula.execution.prepared_updates

*module*

Explicit host forcing and thermodynamic freshness for prepared sessions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_updates.py#L1-L1)

## particula.execution.prepared_updates.apply_forcing

*function*

```python
def apply_forcing(owner: Any, updates: tuple[tuple[Any, np.ndarray], ...])
```

Observe prior work and finish an explicit, validated upload boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_updates.py#L28-L47)

## particula.execution.prepared_updates.compose_operations

*function*

```python
def compose_operations(context: BuiltinPreparationContext, plans, operations)
```

Refresh saturation before declared consumers and after complete steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_updates.py#L50-L123)

## particula.execution.prepared_updates.forcing_array

*function*

```python
def forcing_array(value, shape, name, *, positive)
```

Own finite physical host values with scalar/species/box broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_updates.py#L12-L25)

## particula.execution.prepared_updates.validate_extension_freshness

*function*

```python
def validate_extension_freshness(plans)
```

Reject stale saturation consumers if no refresh provider is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_updates.py#L126-L157)
