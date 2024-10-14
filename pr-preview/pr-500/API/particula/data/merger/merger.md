# Merger

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Merger

> Auto-generated documentation for [particula.data.merger](https://github.com/uncscode/particula/blob/main/particula/data/merger.py) module.

## combine_data

[Show source in merger.py:15](https://github.com/uncscode/particula/blob/main/particula/data/merger.py#L15)

 "
Merge or adds processed data together. Accounts for data shape
miss matches and duplicate timestamps. If the data is a different shape
than
the existing data, it will be reshaped to match the existing data.

#### Arguments

-----------
data : np.array
    Existing data stream.
time : np.array
    Time array for the existing data.
header_list : List[str]
    List of headers for the existing data.
data_new : np.array
    Processed data to add to the data stream.
time_new : np.array
    Time array for the new data.
header_new : List[str]
    List of headers for the new data.

#### Returns

--------
Tuple[np.array, List[str], Dict[str, int]]
    A tuple containing the updated data stream, the updated header list,
    and
    a dictionary mapping the header names to their corresponding indices in
    the data stream.

#### Signature

```python
def combine_data(
    data: np.ndarray,
    time: np.ndarray,
    header_list: list,
    data_new: np.ndarray,
    time_new: np.ndarray,
    header_new: list,
) -> Tuple[np.ndarray, list]: ...
```



## stream_add_data

[Show source in merger.py:99](https://github.com/uncscode/particula/blob/main/particula/data/merger.py#L99)

Adds a new data stream and corresponding time stream to the
existing data.

Args
----------
stream : object
    A Stream object, containing the existing data.
new_time : np.ndarray (m,)
    An array of time values for the new data stream.
new_data : np.ndarray
    An array of data values for the new data stream.
header_check : bool, optional
    If True, checks whether the header in the new data matches the
    header in the existing data. Defaults to False.
new_header : list of str, optional
    A list of header names for the new data stream. Required if
    header_check is True.

Returns
-------
stream : object
    A Stream object, containing the updated data.

Raises
------
ValueError
    If header_check is True and header is not provided or
    header does not match the existing header.

Notes
-----

If header_check is True, the method checks whether the header in the
new data matches the header in the existing data. If they do not match,
the method attempts to merge the headers and updates the header
dictionary.

If header_check is False or the headers match, the new data is
appended to the existing data.

The function also checks whether the time stream is increasing, and if
not, sorts the time stream and corresponding data.

#### Signature

```python
def stream_add_data(
    stream: Stream,
    time_new: np.ndarray,
    data_new: np.ndarray,
    header_check: Optional[bool] = False,
    header_new: Optional[list] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)
