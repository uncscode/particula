""" validate inputs
"""

import inspect
from functools import wraps

import numpy as np


def validate_positive(value, name):
    """ validate positive """
    if np.any(value <= 0):
        raise ValueError(f"Argument '{name}' must be positive.")


def validate_negative(value, name):
    """ validate negative """
    if np.any(value >= 0):
        raise ValueError(f"Argument '{name}' must be negative.")


def validate_nonpositive(value, name):
    """ validate nonpositive """
    if np.any(value > 0):
        raise ValueError(f"Argument '{name}' must be nonpositive.")


def validate_nonnegative(value, name):
    """ validate nonnegative """
    if np.any(value < 0):
        raise ValueError(f"Argument '{name}' must be nonnegative.")


def validate_nonzero(value, name):
    """ validate nonzero """
    if np.any(value == 0):
        raise ValueError(f"Argument '{name}' must be nonzero.")


def validate_inputs(dict_args):
    """
    A decorator to validate that specified arguments meet certain constraints.

    Arguments:
        dict_args: Dictionary of argument names and their constraints.

    Returns:
        A decorator for input validation.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Map argument names to values
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            for name, comp in dict_args.items():
                value = kwargs.get(
                    name, args[params.index(name)]
                    if name in params and params.index(name) < len(args)
                    else None
                )
                if comp == "positive":
                    validate_positive(value, name)
                elif comp == "negative":
                    validate_negative(value, name)
                elif comp == "nonpositive":
                    validate_nonpositive(value, name)
                elif comp == "nonnegative":
                    validate_nonnegative(value, name)
                elif comp == "nonzero":
                    validate_nonzero(value, name)
                else:
                    raise ValueError(
                        f"Unknown validation '{comp}' for argument '{name}'."
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
