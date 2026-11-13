# particula.util.convert_dtypes

*module*

Utilities for coercing data types, converting lists to dictionaries,
retrieving dictionary values by keys, and validating or reshaping data
arrays with matching time steps and headers.

> **References**
>
> - NumPy Documentation: https://numpy.org/doc/

To be removed, likely particula_beta only. -kyle

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L1-L1)

## particula.util.convert_dtypes.get_coerced_type

*function*

```python
def get_coerced_type(data, dtype)
```

Coerce the given data to the specified dtype if it is not already of that
type.

**Parameters**

- `data` (`Any`) — The data to be coerced (any type).
- `dtype` (`type`) — The desired data type, e.g. float, int, or np.ndarray.

**Returns**

- `result` (`Any`) — The data converted to the specified type.

**Raises**

- `ValueError` — If the data cannot be coerced to the desired dtype.

**Examples**

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

> **References**
>
> - NumPy Documentation: https://numpy.org/doc/

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L17-L55)

## particula.util.convert_dtypes.get_dict_from_list

*function*

```python
def get_dict_from_list(list_of_str: list) -> dict
```

Convert a list of strings into a dictionary mapping each string to its
index.

**Parameters**

- `list_of_str` (`list`) — A non-empty list of strings.

**Returns**

- (`dict`) — A dict where keys are the strings and values are their indices.

**Raises**

- `TypeError` — If the list is empty or contains non-string items.

**Examples**

``` py title="Convert list of strings to dictionary"
import particula as par

str_list = ["alpha", "beta", "gamma"]
mapping = par.get_dict_from_list(str_list)
print(mapping)
# {'alpha': 0, 'beta': 1, 'gamma': 2}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L58-L93)

## particula.util.convert_dtypes.get_shape_check

*function*

```python
def get_shape_check(time: np.ndarray, data: np.ndarray, header: list) -> np.ndarray
```

Validate or reshape data array for compatibility with time array
and header list.

If data is 2D, the function attempts to align the time dimension with one
of the axes. If data is 1D, the header list must have exactly one entry.

**Parameters**

- `time` (`np.ndarray`) — 1D array of time values.
- `data` (`Any`) — 1D or 2D array of data values.
- `header` (`list`) — List of headers corresponding to the data dimensions.

**Returns**

- (`np.ndarray`) — A possibly reshaped data array ensuring alignment with time and header constraints.

**Raises**

- `ValueError` — If the header length does not match the data shape,
or if data is 1D but header has more than one entry.

**Examples**

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

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L132-L201)

## particula.util.convert_dtypes.get_values_of_dict

*function*

```python
def get_values_of_dict(key_list: List[str], dict_to_check: Dict[str, Any]) -> List[Any]
```

Retrieve a list of index values from a dictionary for the specified keys.

**Parameters**

- `key_list` (`List[str]`) — The keys to look up in the dictionary.
- `dict_to_check` (`Dict[str, Any]`) — The dictionary from which values are retrieved.

**Returns**

- (`List[Any]`) — A list of values corresponding to the given keys.

**Raises**

- `KeyError` — If any key in key_list is not found in dict_to_check.

**Examples**

``` py
import particula as par
my_dict = {'a': 1, 'b': 2, 'c': 3}
vals = par.get_values_of_dict(['a', 'c'], my_dict)
print(vals)
# [1, 3]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L96-L129)
