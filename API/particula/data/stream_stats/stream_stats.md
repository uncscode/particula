# Stream Stats

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Stream Stats

> Auto-generated documentation for [particula.data.stream_stats](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py) module.

## average_std

[Show source in stream_stats.py:34](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L34)

Calculate the average and standard deviation of data within a given
'stream' object over specified intervals.

This function takes a 'stream' object, which should contain time-series
data, and computes the average and standard deviation of the data at
intervals specified by 'average_interval'. If data.time is in seconds
then the units of the interval are seconds (hour in hours etc). The
results are returned as a new 'StreamAveraged' object containing the
processed data.

#### Arguments

- stream (object): The input stream object containing 'time' and 'data'
    arrays along with other associated metadata.
- average_interval (float|int, optional): The time interval over which the
    averaging is to be performed.
- new_time_array (np.ndarray, optional): An optional array of time points
    at which the average and standard deviation are computed.
    If not provided, a new time array is generated based on the start and
    end times within the 'stream.time' object.

#### Returns

- StreamAveraged (object): An object of type 'StreamAveraged' containing
    the averaged data, time array, start and stop times, the standard
    deviation of the averaged data, and other metadata from the original
    'stream' object.

The function checks for an existing 'new_time_array' and generates one if
needed. It then calculates the average and standard deviation for each
interval and constructs a 'StreamAveraged' object with the results and
metadata from the original 'stream' object.

#### Signature

```python
def average_std(
    stream: Stream,
    average_interval: Union[float, int] = 60,
    new_time_array: Optional[np.ndarray] = None,
) -> StreamAveraged: ...
```

#### See also

- [StreamAveraged](./stream.md#streamaveraged)
- [Stream](./stream.md#stream)



## drop_masked

[Show source in stream_stats.py:14](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L14)

Drop rows where mask is false, and return data stream.

Args
----------
stream : object
    data stream object
mask : np.ndarray
    mask to apply to data stream

Returns
-------
object
    stream object

#### Signature

```python
def drop_masked(stream: Stream, mask: ignore) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## filtering

[Show source in stream_stats.py:104](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L104)

Filters the data of the given 'stream' object based on the specified
bounds or specific value. The filtered data can be either dropped or
replaced with a specified value.  Note, not all parameters need to be
specified, but at least one must be provided (top, bottom, value)

#### Arguments

- stream (Stream): The input stream object containing 'data' and 'time'
    attributes.
- bottom (float, optional): The lower bound for filtering data. Defaults
    to None.
- top (float, optional): The upper bound for filtering data.
    Defaults to None.
- value (float, optional): Specific value to filter from data.
    Defaults to None.
- invert (bool): If True, inverts the filter criteria.
    Defaults to False.
- clone (bool): If True, returns a copy of the 'stream' object, with
    filtered data. If False, modifies the 'stream' object in-place.
    Defaults to True.
- replace_with (float|int, optional): Value to replace filtered-out data.
    Defaults to None.
- drop (bool, optional): If True, filtered-out data points are dropped
    from the dataset. Defaults to False.
- header (list, optional): The header of the data to filter on. This can
    same as calling Stream['header']
    Defaults to None.

#### Returns

- `-` *Stream* - The 'stream' object with data filtered as specified.

If 'drop' is True, 'replace_with' is ignored and filtered data points are
removed from the 'stream' object. Otherwise, filtered data points are
replaced with 'replace_with' value.

add specific data row to filter on

#### Signature

```python
def filtering(
    stream: Stream,
    bottom: Optional[float] = None,
    top: Optional[float] = None,
    value: Optional[float] = None,
    invert: Optional[bool] = False,
    clone: Optional[bool] = True,
    replace_with: Optional[Union[float, int]] = None,
    drop: Optional[bool] = False,
    header: Optional[Union[list, int, str]] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## remove_time_window

[Show source in stream_stats.py:177](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L177)

Remove a time window from a stream object.

#### Arguments

- `stream` - The input stream object containing 'data' and 'time'
    attributes.
- `epoch_start` - The start time of the time window to be
    removed.
- `epoch_end` - The end time of the time window to be
    removed. If not provided, the time window is the closest time
    point to 'epoch_start'.

#### Returns

- `Stream` - The 'stream' object with the specified time window removed.

#### Signature

```python
def remove_time_window(
    stream: Stream,
    epoch_start: Union[float, int],
    epoch_end: Optional[Union[float, int]] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## select_time_window

[Show source in stream_stats.py:212](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L212)

Keep only a specified time window in a stream object and remove all other
data.

#### Arguments

- `stream` - The input stream object containing 'data' and 'time'
    attributes.
- `epoch_start` - The start time of the time window to be kept.
- `epoch_end` - The end time of the time window to be kept. If not provided,
    only the closest time point to 'epoch_start' will be kept.

#### Returns

- `Stream` - The stream object with only the specified time window retained.

#### Signature

```python
def select_time_window(
    stream: Stream,
    epoch_start: Union[float, int],
    epoch_end: Optional[Union[float, int]] = None,
    clone: Optional[bool] = True,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## time_derivative_of_stream

[Show source in stream_stats.py:251](https://github.com/uncscode/particula/blob/main/particula/data/stream_stats.py#L251)

Calculate the rate of change of the concentration PMF over time and
return a new stream.

Uses a linear regression model to fit the slope over a time window.
The edge cases are handled by using a smaller window size.

#### Arguments

- `pmf_fitted_stream` - Stream object containing the fitted concentration
    PMF data.
- `window_size` - Size of the time window for fitting the slope.

#### Returns

- `rate_of_change_stream` - Stream object containing the rate of
    change of the concentration PMF.

#### Signature

```python
def time_derivative_of_stream(
    stream: Stream, liner_slope_window_size: int = 12
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)
