# particula.util.validate_inputs

*module*

Tools to validate function inputs, ensuring they meet various constraints.

This module provides decorators and helper functions to check if
arguments are positive, negative, nonzero, finite, etc.

**Examples**

``` py
from particula.util.validate_inputs import validate_inputs

@validate_inputs({"radius": "positive", "concentration": "nonnegative"})
def example_function(radius, concentration):
    return radius * concentration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L1-L1)

## particula.util.validate_inputs.validate_finite

*function*

```python
def validate_finite(value, name)
```

Validate that a numeric array or scalar has no infinities or NaNs.

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is inf or NaN.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L92-L103)

## particula.util.validate_inputs.validate_inputs

*function*

```python
def validate_inputs(dict_args)
```

A decorator to validate function inputs against specified constraints.

The constraints are defined by a dictionary of argument names and their
validation types (e.g., "positive", "negative", "nonnegative", etc.). If
any argument violates its constraint, a ValueError is raised. Arguments
explicitly passed as ``None`` are ignored and no validation is performed
on them.

**Parameters**

- `dict_args` (`dict[str, str]`) — Map argument names to constraint types. The constraint type is one of: - "positive" : Must be strictly > 0. - "negative" : Must be strictly < 0. - "nonpositive" : Must be <= 0. - "nonnegative" : Must be >= 0. - "nonzero" : Must be != 0. - "finite" : Must not contain inf or NaN.

**Returns**

- `result` (`Callable`) — A decorator that applies the specified input validations.

**Examples**

``` py
from particula.util.validate_inputs import validate_inputs

@validate_inputs({"mass": "positive", "temperature": "nonnegative"})
def some_function(mass, temperature):
    return mass * temperature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L106-L181)

## particula.util.validate_inputs.validate_negative

*function*

```python
def validate_negative(value, name)
```

Validate that a numeric array or scalar is strictly negative.

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is >= 0.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L36-L47)

## particula.util.validate_inputs.validate_nonnegative

*function*

```python
def validate_nonnegative(value, name)
```

Validate that a numeric array or scalar is nonnegative (>= 0).

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is < 0.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L64-L75)

## particula.util.validate_inputs.validate_nonpositive

*function*

```python
def validate_nonpositive(value, name)
```

Validate that a numeric array or scalar is nonpositive (<= 0).

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is > 0.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L50-L61)

## particula.util.validate_inputs.validate_nonzero

*function*

```python
def validate_nonzero(value, name)
```

Validate that a numeric array or scalar is nonzero.

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is 0.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L78-L89)

## particula.util.validate_inputs.validate_positive

*function*

```python
def validate_positive(value, name)
```

Validate that a numeric array or scalar is strictly positive.

**Parameters**

- `value` (`ArrayLike`) — Array-like numeric values to check.
- `name` (`str`) — The argument name, used in the error message.

**Raises**

- `ValueError` — If any element is <= 0.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L22-L33)
