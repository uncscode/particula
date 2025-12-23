"""Utilities for coercing data types, converting lists to dictionaries,
retrieving dictionary values by keys, and validating or reshaping data
arrays with matching time steps and headers.

References:
    - NumPy Documentation: https://numpy.org/doc/

To be removed, likely particula_beta only. -kyle
"""

from collections.abc import Sequence
from typing import Any, Dict, List

import numpy as np


def get_coerced_type(data, dtype):
    """Coerce the given data to the specified dtype if it is not already of that
    type.

    Arguments:
        - data : The data to be coerced (any type).
        - dtype : The desired data type, e.g. float, int, or np.ndarray.

    Returns:
        - The data converted to the specified type.

    Raises:
        - ValueError : If the data cannot be coerced to the desired dtype.

    Examples:
        ``` py title="Coerce integer to float"
        import particula as par
        x = par.get_coerced_type(1, float)
        print(x)
        # 1.0
        ```

        ``` py title="Coerce list to numpy array"
        import numpy as np
        import particula as par
        arr = par.get_coerced_type([1, 2, 3], np.ndarray)
        print(arr)
        # [1 2 3]
        ```

    References:
        - NumPy Documentation: https://numpy.org/doc/
    """
    if not isinstance(data, dtype):
        try:
            return np.array(data) if dtype == np.ndarray else dtype(data)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not coerce {data} to {dtype}") from exc
    return data


def get_dict_from_list(list_of_str: list) -> dict:
    """Convert a list of strings into a dictionary mapping each string to its
    index.

    Arguments:
        - list_of_str : A non-empty list of strings.

    Returns:
        - A dict where keys are the strings and values are their indices.

    Raises:
        - TypeError : If the list is empty or contains non-string items.

    Examples:
        ``` py title="Convert list of strings to dictionary"
        import particula as par

        str_list = ["alpha", "beta", "gamma"]
        mapping = par.get_dict_from_list(str_list)
        print(mapping)
        # {'alpha': 0, 'beta': 1, 'gamma': 2}
        ```
    """
    # basic type / emptiness check
    if not isinstance(list_of_str, Sequence) or not list_of_str:
        raise TypeError("list_of_str must be a non-empty sequence of strings.")

    # one pass: ensure every element is a non-empty string
    if any(not isinstance(item, str) or item == "" for item in list_of_str):
        raise TypeError(
            "All elements in list_of_str must be non-empty strings."
        )

    # Create a dictionary from the list of strings using a dictionary
    # comprehension
    return {str_val: i for i, str_val in enumerate(list_of_str)}


def get_values_of_dict(
    key_list: List[str], dict_to_check: Dict[str, Any]
) -> List[Any]:
    """Retrieve a list of index values from a dictionary for the specified keys.

    Arguments:
        - key_list : The keys to look up in the dictionary.
        - dict_to_check : The dictionary from which values are retrieved.

    Returns:
        - A list of values corresponding to the given keys.

    Raises:
        - KeyError : If any key in key_list is not found in dict_to_check.

    Examples:
        ``` py
        import particula as par
        my_dict = {'a': 1, 'b': 2, 'c': 3}
        vals = par.get_values_of_dict(['a', 'c'], my_dict)
        print(vals)
        # [1, 3]
        ```
    """
    values = []
    for key in key_list:
        if key in dict_to_check:
            values.append(dict_to_check[key])
        else:
            raise KeyError(
                f"Key '{key}' not found in the dictionary. Available keys:"
                + f"{list(dict_to_check.keys())}"
            )
    return values


def get_shape_check(
    time: np.ndarray,
    data: np.ndarray,
    header: list,
) -> np.ndarray:
    """Validate or reshape data array for compatibility with time array
    and header list.

    If data is 2D, the function attempts to align the time dimension with one
    of the axes. If data is 1D, the header list must have exactly one entry.

    Arguments:
        - time : 1D array of time values.
        - data : 1D or 2D array of data values.
        - header : List of headers corresponding to the data dimensions.

    Returns:
        - A possibly reshaped data array ensuring alignment with time and
          header constraints.

    Raises:
        - ValueError : If the header length does not match the data shape,
          or if data is 1D but header has more than one entry.

    Examples:
        ``` py
        import numpy as np
        import particula as par
        time_array = np.arange(0, 10)
        data_2d = np.random.rand(10, 5)
        headers = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
        reshaped_data = par.get_shape_check(time_array, data_2d, headers)
        print(reshaped_data.shape)
        # Should be (10, 5)
        ```
    """
    # Check if data_new is 2D or 1D
    if len(data.shape) == 2:
        # Check if time matches the dimensions of data
        if len(time) == data.shape[0] and len(time) == data.shape[1]:
            concatenate_axis_new = 1  # Default to the axis=1
        else:
            # Find the axis that doesn't match the length of time
            indices = np.argwhere(np.array(data.shape) != len(time)).flatten()
            # NumPy 2.0 requires a Python int for axis values
            concatenate_axis_new = int(indices[0].item())
        # Reshape new data so the concatenate axis is axis=1
        data = np.moveaxis(data, concatenate_axis_new, 1)

        # check header list length matches data_new shape
        if len(header) != data.shape[1]:
            print(
                f"header len: {len(header)} vs. data.shape: \
                  {data.shape}"
            )
            print(header)
            raise ValueError(
                "Header list length must match the second \
                              dimension of data_new."
            )
    elif len(header) == 1:
        # Reshape new data so the concatenate axis is axis=1
        data = np.expand_dims(data, 1)

    else:
        raise ValueError(
            "Header list must be a single entry if data_new \
                              is 1D."
        )
    return data
