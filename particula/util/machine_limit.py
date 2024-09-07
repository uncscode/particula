"""Machine max or min overflow protection."""

from numpy.typing import ArrayLike
import numpy as np


MIN_POSITIVE_VALUE = np.nextafter(0, 1, dtype=np.float64)
MAX_POSITIVE_VALUE = np.finfo(np.float64).max
MAX_NEGATIVE_VALUE = np.finfo(np.float64).min


def safe_exp(value: ArrayLike) -> np.ndarray:
    """
    Compute the exponential of each element in the input array, with limits
    to prevent overflow based on machine precision.

    Args:
        value (ArrayLike): Input array.

    Returns:
        np.ndarray: Exponential of the input array with overflow protection.
    """
    value = np.asarray(value, dtype=np.float64)
    max_exp_input = np.log(np.finfo(value.dtype).max)
    return np.exp(np.clip(value, None, max_exp_input))


def safe_log(value: ArrayLike) -> np.ndarray:
    """
    Compute the natural logarithm of each element in the input array, with
    limits to prevent underflow based on machine precision.

    Args:
        value (ArrayLike): Input array.

    Returns:
        np.ndarray: Natural logarithm of the input array with underflow
        protection.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log(np.clip(value, min_positive_value, None))


def safe_log10(value: ArrayLike) -> np.ndarray:
    """
    Compute the base 10 logarithm of each element in the input array, with
    limits to prevent underflow based on machine precision.

    Args:
        value (ArrayLike): Input array.

    Returns:
        np.ndarray: Base 10 logarithm of the input array with underflow
        protection.
    """
    value = np.asarray(value, dtype=np.float64)
    min_positive_value = np.nextafter(0, 1, dtype=value.dtype)
    return np.log10(np.clip(value, min_positive_value, None))
