"""Machine max or min overflow protection."""

from numpy.typing import ArrayLike
import numpy as np


MIN_POSITIVE_VALUE = np.nextafter(0, 1, dtype=np.float64)
MAX_POSITIVE_VALUE = np.finfo(np.float64).max
MAX_NEGATIVE_VALUE = np.finfo(np.float64).min


def get_safe_exp(value: ArrayLike) -> np.ndarray:
    """
    Compute the exponential of each element in the input array, with overflow
    protection.

    The exponential is calculated using:
        - y = exp(x), where x is clipped to avoid exceeding machine limits.

    Arguments:
        - value : Array-like of values to exponentiate.

    Returns:
        - np.ndarray of exponentiated values, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        from particula.util.machine_limit import safe_exp
        import numpy as np

        arr = np.array([0, 10, 1000])
        print(safe_exp(arr))
        # Output: [1.00000000e+000 2.20264658e+004 1.79769313e+308]
        ```

    References:
        - "Floating Point Arithmetic," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    max_exp_input = np.log(np.finfo(value.dtype).max)
    return np.exp(np.clip(value, None, max_exp_input))


def get_safe_log(value: ArrayLike) -> np.ndarray:
    """
    Compute the natural logarithm of each element in the input array, with
    underflow protection.

    The natural log is calculated using:
        - y = ln(x), where x is clipped away from zero to maintain positivity.

    Arguments:
        - value : Array-like of values for logarithm calculation.

    Returns:
        - np.ndarray of natural logarithms, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        from particula.util.machine_limit import safe_log
        import numpy as np

        arr = np.array([1e-320, 1.0, 10.0])
        print(safe_log(arr))
        # Output: [-7.40545337e+02  0.00000000e+00  2.30258509e+00]
        ```

    References:
        - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log(np.clip(value, min_positive_value, None))


def get_safe_log10(value: ArrayLike) -> np.ndarray:
    """
    Compute the base-10 logarithm of each element in the input array, with
    underflow protection.

    The base-10 log is calculated using:
        - y = log10(x), where x is clipped away from zero to maintain positivity.

    Arguments:
        - value : Array-like of values for base-10 logarithm calculation.

    Returns:
        - np.ndarray of base-10 logarithms, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        from particula.util.machine_limit import safe_log10
        import numpy as np

        arr = np.array([1e-320, 1.0, 1000.0])
        print(safe_log10(arr))
        # Output: [-320.           0.           3.        ]
        ```

    References:
        - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log10(np.clip(value, min_positive_value, None))
