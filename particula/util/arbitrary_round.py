"""Rounding function that allows for arbitrary bases and rounding modes.

To be removed, likely particula_beta only. -kyle
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.convert_dtypes import get_coerced_type


def get_arbitrary_round(
    values: Union[float, list[float], np.ndarray],
    base: Union[float, np.float64] = 1.0,
    mode: str = "round",
    nonzero_edge: bool = False,
) -> Union[float, NDArray[np.float64]]:
    """Round values to the nearest multiple of a specified base.

    The function supports "round", "floor", or "ceil" modes, and can retain
    original nonzero values if rounding returns zero.

    Arguments:
        - values : The values to be rounded.
        - base : Positive float indicating the rounding interval.
        - mode : Rounding mode, one of ['round', 'floor', 'ceil'].
        - nonzero_edge : If True, zeros after rounding are replaced with the
            original values.

    Returns:
        - The input values rounded according to the specified base and mode.

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        import particula as par

        arr = np.array([1.2, 2.5, 3.7, 4.0])
        print(par.get_arbitrary_round(arr, base=1.0, mode='round'))
        # Output: [1.  2.  4.  4.]

        print(par.get_arbitrary_round(arr, base=0.5, mode='floor'))
        # Output: [1.  2.  3.5 4. ]

        print(par.get_arbitrary_round(2.5, base=1.0, mode='round'))
        # Output: 2.0
        ```

    References:
        - "Rounding," Python Documentation, docs.python.org.
        - "NumPy Rounding," NumPy Documentation, NumPy.org.
    """
    # Check if values is a NumPy array
    working_values = get_coerced_type(values, np.ndarray)
    base = get_coerced_type(base, float)

    # Validate base parameter
    if not isinstance(base, float) or base <= 0:
        raise ValueError("base must be a positive float")
    # Validate mode parameter
    if mode not in ["round", "floor", "ceil"]:
        raise ValueError("mode must be one of ['round', 'floor', 'ceil']")

    # Calculate rounding factors
    factor = np.array([-0.5, 0, 0.5])

    # Compute rounded values
    rounded = base * np.round(
        working_values / base
        + factor[np.array(["floor", "round", "ceil"]).tolist().index(mode)]
    )

    # Apply round_nonzero mode
    if nonzero_edge:
        rounded = np.where(rounded != 0, rounded, working_values)

    return float(rounded) if isinstance(values, float) else rounded
