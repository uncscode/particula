"""Tools to validate function inputs, ensuring they meet various constraints.

This module provides decorators and helper functions to check if
arguments are positive, negative, nonzero, finite, etc.

Examples:
    ``` py
    from particula.util.validate_inputs import validate_inputs

    @validate_inputs({"radius": "positive", "concentration": "nonnegative"})
    def example_function(radius, concentration):
        return radius * concentration
    ```
"""

import inspect
from functools import wraps

import numpy as np


def validate_positive(value, name):
    """Validate that a numeric array or scalar is strictly positive.

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is <= 0.
    """
    if np.any(value <= 0):
        raise ValueError(f"Argument '{name}' must be positive.")


def validate_negative(value, name):
    """Validate that a numeric array or scalar is strictly negative.

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is >= 0.
    """
    if np.any(value >= 0):
        raise ValueError(f"Argument '{name}' must be negative.")


def validate_nonpositive(value, name):
    """Validate that a numeric array or scalar is nonpositive (<= 0).

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is > 0.
    """
    if np.any(value > 0):
        raise ValueError(f"Argument '{name}' must be nonpositive.")


def validate_nonnegative(value, name):
    """Validate that a numeric array or scalar is nonnegative (>= 0).

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is < 0.
    """
    if np.any(value < 0):
        raise ValueError(f"Argument '{name}' must be nonnegative.")


def validate_nonzero(value, name):
    """Validate that a numeric array or scalar is nonzero.

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is 0.
    """
    if np.any(value == 0):
        raise ValueError(f"Argument '{name}' must be nonzero.")


def validate_finite(value, name):
    """Validate that a numeric array or scalar has no infinities or NaNs.

    Arguments:
        - value : Array-like numeric values to check.
        - name : The argument name, used in the error message.

    Raises:
        - ValueError : If any element is inf or NaN.
    """
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Argument '{name}' must be finite (no inf or NaN).")


def validate_inputs(dict_args):  # noqa: C901
    """A decorator to validate function inputs against specified constraints.

    The constraints are defined by a dictionary of argument names and their
    validation types (e.g., "positive", "negative", "nonnegative", etc.). If
    any argument violates its constraint, a ValueError is raised. Arguments
    explicitly passed as ``None`` are ignored and no validation is performed
    on them.

    Arguments:
        - dict_args : Dictionary {argument_name: constraint_type}, where the
            constraint_type is one of:
            - "positive" : Must be strictly > 0.
            - "negative" : Must be strictly < 0.
            - "nonpositive" : Must be <= 0.
            - "nonnegative" : Must be >= 0.
            - "nonzero" : Must be != 0.
            - "finite" : Must not contain inf or NaN.

    Returns:
        - A decorator that applies the specified input validations.

    Examples:
        ``` py
        from particula.util.validate_inputs import validate_inputs

        @validate_inputs({"mass": "positive", "temperature": "nonnegative"})
        def some_function(mass, temperature):
            return mass * temperature
        ```
    """

    def decorator(func):  # noqa: C901
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Map argument names to values using cached signature
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for name, comp in dict_args.items():
                if name not in bound.arguments:
                    raise TypeError(
                        f"Argument '{name}' is not provided and has no default."
                    )
                if bound.arguments[name] is None:
                    continue
                value = np.asarray(bound.arguments[name])
                if comp == "positive":
                    validate_positive(value, name)
                    validate_finite(value, name)
                elif comp == "negative":
                    validate_negative(value, name)
                    validate_finite(value, name)
                elif comp == "nonpositive":
                    validate_nonpositive(value, name)
                    validate_finite(value, name)
                elif comp == "nonnegative":
                    validate_nonnegative(value, name)
                    validate_finite(value, name)
                elif comp == "nonzero":
                    validate_nonzero(value, name)
                    validate_finite(value, name)
                elif comp == "finite":
                    validate_finite(value, name)
                else:
                    raise ValueError(
                        f"Unknown validation '{comp}' for argument '{name}'."
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator
