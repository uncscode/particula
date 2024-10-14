# Stats

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Stats

> Auto-generated documentation for [particula.util.stats](https://github.com/uncscode/particula/blob/main/particula/util/stats.py) module.

## average_to_interval

[Show source in stats.py:101](https://github.com/uncscode/particula/blob/main/particula/util/stats.py#L101)

Calculate the average of the data stream over the specified time intervals.

This function calculates the average of the data stream over a series of
time intervals specified by `average_interval_array`. The average and
standard
deviation of the data are calculated for each interval, and the results
are returned as two arrays.

#### Arguments

----------
    - `time_raw` *np.ndarray* - An array of timestamps, sorted in ascending
        order.
    - `average_interval` *float* - The length of each time interval in seconds.
    - `average_interval_array` *np.ndarray* - An array of timestamps
        representing
        the start times of each time interval.
    - `data_raw` *np.ndarray* - An array of data points corresponding to the
        timestamps in `time_raw`.
    - `average_data` *np.ndarray* - An empty array of shape
        (num_channels, num_intervals)that will be filled with the
        average data for each time interval.
    - `average_data_std` *np.ndarray* - An empty array of shape
        (num_channels, num_intervals) that will be filled with the standard
        deviation of the data for each time interval.

#### Returns

-------
    - `Tuple[np.ndarray,` *np.ndarray]* - A tuple containing the average data
        and the standard deviation of the data, both as arrays of shape
        (num_channels, num_intervals).

#### Signature

```python
def average_to_interval(
    time_raw: np.ndarray,
    data_raw: np.ndarray,
    average_interval: float,
    average_interval_array: np.ndarray,
    average_data: np.ndarray,
    average_data_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## distribution_integration

[Show source in stats.py:272](https://github.com/uncscode/particula/blob/main/particula/util/stats.py#L272)

Performs either PDF integration or PMS integration based on the input.
This function supports broadcasting where x_array has shape (m,) and
distribution has shape (n, m).

#### Arguments

-----
    - `distribution` - The distribution array to integrate.
        It should have a shape of (n, m).
    - `x_array` - The x-values array for PDF
        integration. It should have a shape of (m,).
        If None, PMS integration is performed. Defaults to None.
    - `axis` - The axis along which to perform the integration
        for PDF or the sum for PMS.
        Defaults to 0.

#### Returns

-------
    - `np.ndarray` - The result of the integration. If PDF integration is
    performed, the result will have a shape of (n,) if axis=0 or (m,)
    if axis=1. If PMS integration is performed, the result will be a
    single value if axis=None, or an array with reduced dimensionality
    otherwise.

#### Signature

```python
def distribution_integration(
    distribution: np.ndarray, x_array: Optional[np.ndarray] = None, axis: int = 0
) -> np.ndarray: ...
```



## mask_outliers

[Show source in stats.py:219](https://github.com/uncscode/particula/blob/main/particula/util/stats.py#L219)

Create a boolean mask for outliers in a data array. Outliers are defined as
values that are either above or below a specified threshold, or that are
equal to a specified value. Not all parameters need to be specified. If
`invert` is True, the mask will be inverted. The mask will be True for
False for outliers and True for non-outliers.

#### Arguments

----------
    - `data` *np.ndarray* - The data array to be masked.
    - `bottom` *float* - The lower threshold for outliers.
    - `top` *float* - The upper threshold for outliers.
    - `value` *float* - The value to be masked.
    - `invert` *bool* - If True, the mask will be inverted.

#### Returns

-------
    - `np.ndarray` - A boolean mask for the outliers in the data array. Mask is
        True for non-outliers and False for outliers, and the same shape as
        the data array.

#### Signature

```python
def mask_outliers(
    data: np.ndarray,
    bottom: Optional[float] = None,
    top: Optional[float] = None,
    value: Optional[float] = None,
    invert: Optional[bool] = False,
) -> np.ndarray: ...
```



## merge_formatting

[Show source in stats.py:10](https://github.com/uncscode/particula/blob/main/particula/util/stats.py#L10)

Formats two data arrays and their headers so that the data new can be
subsiqently added to data current.

#### Arguments

- `data_current` *np.ndarray* - First data array to merge.
- `header_current` *list* - Header for the first data array.
- `data_new` *np.ndarray* - Second data array to merge.
- `header_new` *list* - Header for the second data array.

#### Returns

(np.ndarray, list, np.array, list): A tuple formatted data
and headers.

#### Raises

- `ValueError` - If the data arrays are not the same shape.
- `ValueError` - If the headers are not the same length.

#### Signature

```python
def merge_formatting(
    data_current: np.ndarray,
    header_current: list,
    data_new: np.ndarray,
    header_new: list,
) -> Tuple[np.ndarray, list, np.ndarray, list]: ...
```
