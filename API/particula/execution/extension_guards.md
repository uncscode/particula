# particula.execution.extension_guards

*module*

Prepared device-only validation of declared custom process destinations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L1-L1)

## particula.execution.extension_guards._float_launch

*function*

```python
def _float_launch(array, lower, upper, strict, failure)
```

Retain flattened views and typed launch inputs during setup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L64-L71)

## particula.execution.extension_guards._float_values

*function*

```python
def _float_values(values: wp.array(dtype=wp.float64), lower: wp.float64, upper: wp.float64, strict_lower: int, failure: wp.array(dtype=wp.int32))
```

Latch finite/bounds failures before dependent kernels can run.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L10-L29)

## particula.execution.extension_guards._int32_values

*function*

```python
def _int32_values(values: wp.array(dtype=wp.int32), lower: wp.int64, upper: wp.int64, failure: wp.array(dtype=wp.int32))
```

Compare integer bounds without losing precision through float casts.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L32-L45)

## particula.execution.extension_guards._int64_values

*function*

```python
def _int64_values(values: wp.array(dtype=wp.int64), lower: wp.int64, upper: wp.int64, failure: wp.array(dtype=wp.int32))
```

Retain all 64 integer bits in field-bound validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L48-L61)

## particula.execution.extension_guards.prepare_validation

*function*

```python
def prepare_validation(configuration, arrays, state, owner)
```

Prepare bounds once; validation enqueues no allocation or readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_guards.py#L74-L146)
