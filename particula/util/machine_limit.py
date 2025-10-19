"""Machine max or min overflow protection."""

import numpy as np
from numpy.typing import ArrayLike

from particula.util.validate_inputs import validate_inputs

MIN_POSITIVE_VALUE = np.nextafter(0, 1, dtype=np.float64)
MAX_POSITIVE_VALUE = np.finfo(np.float64).max
MAX_NEGATIVE_VALUE = np.finfo(np.float64).min


def get_safe_exp(value: ArrayLike) -> np.ndarray:
    """Compute the exponential of each element in the input array, with overflow
    protection.

    The exponential is calculated using:
        - y = exp(x), where x is clipped to avoid exceeding machine limits.

    Arguments:
        - value : Array-like of values to exponentiate.

    Returns:
        - np.ndarray of exponentiated values, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        import particula as par

        arr = np.array([0, 10, 1000])
        print(par.get_safe_exp(arr))
        # Output: [1.00000000e+000 2.20264658e+004 1.79769313e+308]
        ```

    References:
        - "Floating Point Arithmetic," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    max_exp_input = np.log(np.finfo(value.dtype).max)
    return np.exp(np.clip(value, None, max_exp_input))


def get_safe_log(value: ArrayLike) -> np.ndarray:
    """Compute the natural logarithm of each element in the input array, with
    underflow protection.

    The natural log is calculated using:
        - y = ln(x), where x is clipped away from zero to maintain positivity.

    Arguments:
        - value : Array-like of values for logarithm calculation.

    Returns:
        - np.ndarray of natural logarithms, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        import particula as par

        arr = np.array([1e-320, 1.0, 10.0])
        print(get_safe_log(arr))
        # Output: [-7.40545337e+02  0.00000000e+00  2.30258509e+00]
        ```

    References:
        - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log(np.clip(value, min_positive_value, None))


def get_safe_log10(value: ArrayLike) -> np.ndarray:
    """Compute the base-10 logarithm of each element in the input array, with
    underflow protection.

    The base-10 log is calculated using:
        - y = log10(x), x clipped from zero to maintain positivity.

    Arguments:
        - value : Array-like of values for base-10 logarithm calculation.

    Returns:
        - np.ndarray of base-10 logarithms, with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        import particula as par

        arr = np.array([1e-320, 1.0, 1000.0])
        print(par.get_safe_log10(arr))
        # Output: [-320.           0.           3.        ]
        ```

    References:
        - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log10(np.clip(value, min_positive_value, None))


@validate_inputs(
    {
        "base": "positive",
    }
)
def get_safe_power(base: ArrayLike, exponent: ArrayLike) -> np.ndarray:
    """Compute the power (base ** exponent) with overflow protection.

    The power is computed as: result = exp(exponent * log(base))
    where the intermediate value is clipped to avoid overflow beyond the
    machine limits. This function assumes that `base` contains positive values.
    The behavior for non-positive bases is undefined.

    Arguments:
        - base : Array-like of positive base values.
        - exponent : Array-like of exponents.

    Returns:
        - np.ndarray of power values, computed with machine-level clipping.

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        import particula as par

        base = np.array([1, 2, 3])
        exponent = np.array([1, 2, 3])
        print(par.get_safe_power(base, exponent))
        # Output: [ 1.  4. 27.]
        ```

    References:
        - "Floating Point Arithmetic," NumPy Documentation, NumPy.org.
    """
    base = np.asarray(base, dtype=np.float64)
    exponent = np.asarray(exponent, dtype=np.float64)

    # Compute the intermediate value using logarithm.
    intermediate = exponent * np.log(base)

    # Compute the maximum safe value for the exponent in np.exp.
    max_exp_input = np.log(MAX_POSITIVE_VALUE)

    # Clip the intermediate result to prevent overflow.
    intermediate_clipped = np.clip(intermediate, None, max_exp_input)

    return np.exp(intermediate_clipped)
