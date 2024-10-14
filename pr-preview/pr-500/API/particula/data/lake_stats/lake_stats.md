# Lake Stats

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Lake Stats

> Auto-generated documentation for [particula.data.lake_stats](https://github.com/uncscode/particula/blob/main/particula/data/lake_stats.py) module.

## average_std

[Show source in lake_stats.py:11](https://github.com/uncscode/particula/blob/main/particula/data/lake_stats.py#L11)

Averages the data in each stream within a 'Lake' object.

If 'clone' is True, a new 'Lake' instance is created and the averaged
data is stored there. If 'clone' is False, the original 'Lake' instance
is modified. The averaged output also includes the standard deviation of
the data.

#### Examples

```python
# Example lake with two streams, each containing numerical data
lake_data = Lake({'stream1': [1, 2, 3], 'stream2': [4, 5, 6]})
# Average over a 60-second interval without creating a new lake.
averaged_lake = average_std(lake_data, 60, clone=False)
print(averaged_lake)
Lake({'stream1': [2], 'stream2': [5]})
```

#### Arguments

- `lake` - The lake data structure containing multiple streams.
- `average_interval` - The interval over which to average the data.
    Default is 60.
- `new_time_array` - A new array of time points at which to compute the
    averages.
- `clone` - Indicates whether to modify the original lake or return a new
    one. Default is True.

#### Returns

- `Lake` - A lake instance with averaged data.

#### Signature

```python
def average_std(
    lake: Lake,
    average_interval: Union[float, int] = 60,
    new_time_array: Optional[NDArray[np.float64]] = None,
    clone: bool = True,
) -> Lake: ...
```

#### See also

- [Lake](./lake.md#lake)
