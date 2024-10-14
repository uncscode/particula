# Stream

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Stream

> Auto-generated documentation for [particula.data.stream](https://github.com/uncscode/particula/blob/main/particula/data/stream.py) module.

## Stream

[Show source in stream.py:13](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L13)

Consistent format for storing data.

Represents a consistent format for storing and managing data streams
within a list. Similar to pandas but with tighter control over the
data allowed and expected format.

#### Attributes

- `header` - Headers of the data stream, each a string.
- `data` - 2D numpy array where rows are timepoints and columns
    correspond to headers.
- `time` - 1D numpy array representing the time points of the data stream.
- `files` - List of filenames that contain the data stream.

#### Methods

- `validate_inputs` - Validates the types of class inputs.
- `__getitem__(index)` - Returns the data at the specified index.
- `__setitem__(index,` *value)* - Sets or updates data at the specified index.
- `__len__()` - Returns the length of the time stream.
- `datetime64` - Converts time stream to numpy datetime64 array for plots.
- `header_dict` - Provides a dictionary mapping of header indices to names.
- `header_float` - Converts header names to a numpy array of floats.

#### Signature

```python
class Stream: ...
```

### Stream().__getitem__

[Show source in stream.py:54](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L54)

Gets data at a specified index or header name.

Allows indexing of the data stream using an integer index or a string
corresponding to the header. If a string is used, the header index is
retrieved and used to return the data array. Only one str
argument is allowed. A list of int is allowed.

#### Arguments

- `index` - The index or name of the data column to
    retrieve.

#### Returns

- `np.ndarray` - The data array at the specified index.

#### Signature

```python
def __getitem__(self, index: Union[int, str]) -> NDArray[np.float64]: ...
```

### Stream().__len__

[Show source in stream.py:99](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L99)

Returns the number of time points in the data stream.

#### Returns

- `int` - Length of the time stream.

#### Signature

```python
def __len__(self) -> int: ...
```

### Stream().__pop__

[Show source in stream.py:108](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L108)

Removes data at a specified index or header name.

Allows indexing of the data stream using an integer index or a string
corresponding to the header. If a string is used, the header index is
retrieved and used to return the data array. Only one str
argument is allowed. A list of int is allowed.

#### Arguments

- `index` - The index or name of the data column to
    retrieve.

#### Signature

```python
def __pop__(self, index: Union[int, str]) -> None: ...
```

### Stream().__setitem__

[Show source in stream.py:73](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L73)

Sets or adds data at a specified index.

If index is a string and not in headers, it is added. This is used
to add new data columns to the stream.

#### Arguments

- `index` - The index or name of the data column to set.
- `value` - The data to set at the specified index.

#### Notes

Support setting multiple rows by accepting a list of values.

#### Signature

```python
def __setitem__(self, index: Union[int, str], value: NDArray[np.float64]): ...
```

### Stream().datetime64

[Show source in stream.py:125](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L125)

Converts the epoch time array to a datetime64 for plotting.

This method converts the time array to a datetime64 array, which
can be used for plotting time series data. This generally assumes
that the time array is in seconds since the epoch.

#### Returns

- `np.ndarray` - Datetime64 array representing the time stream.

#### Signature

```python
@property
def datetime64(self) -> NDArray[np.float64]: ...
```

### Stream().header_dict

[Show source in stream.py:138](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L138)

Provides a dictionary mapping from index to header names.

#### Returns

- `dict` - Dictionary with indices as keys and header names as values.

#### Signature

```python
@property
def header_dict(self) -> dict[int, str]: ...
```

### Stream().header_float

[Show source in stream.py:147](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L147)

Attempts to convert header names to a float array, where possible.

#### Returns

- `np.ndarray` - Array of header names converted to floats.

#### Signature

```python
@property
def header_float(self) -> NDArray[np.float64]: ...
```

### Stream().validate_inputs

[Show source in stream.py:45](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L45)

Validates that header is a list.

#### Raises

- `TypeError` - If [header](#stream) is not a list.

#### Signature

```python
def validate_inputs(self): ...
```



## StreamAveraged

[Show source in stream.py:158](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L158)

Stream Class with Averaged Data and Standard Deviation.

Extends the Stream class with functionalities specific to handling
averaged data streams. Mainly adding standard deviation to the data
stream.

#### Attributes

- `average_interval` - The interval in units (e.g., seconds, minutes) over
    which data is averaged.
- `start_time` - The start time from which data begins to be averaged.
- `stop_time` - The time at which data ceases to be averaged.
- `standard_deviation` - A numpy array storing the standard deviation of
    data streams.

#### Signature

```python
class StreamAveraged(Stream): ...
```

#### See also

- [Stream](#stream)

### StreamAveraged().get_std

[Show source in stream.py:210](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L210)

Retrieves the standard deviation

In the averaged data stream, the standard deviation of the data is
stored in a separate array that mirrors the same indices as the data
stream. This method allows retrieval of the standard deviation at a
specified index.

#### Arguments

- `index` - The index or header name of the data stream
for which standard deviation is needed.

#### Returns

- `np.ndarray` - The standard deviation values at the specified index.

#### Raises

- `ValueError` - If the specified index does not exist in the header.

#### Signature

```python
def get_std(self, index: Union[int, str]) -> NDArray[np.float64]: ...
```

### StreamAveraged().validate_averaging_params

[Show source in stream.py:185](https://github.com/uncscode/particula/blob/main/particula/data/stream.py#L185)

Ensures that averaging parameters are valid.

#### Raises

- `ValueError` - If [average_interval](#streamaveraged) is not a positive number.
- `ValueError` - If [start_time](#streamaveraged) or [stop_time](#streamaveraged) are not numerical or if
    [start_time](#streamaveraged) is greater than or equal to [stop_time](#streamaveraged).

#### Signature

```python
def validate_averaging_params(self): ...
```
