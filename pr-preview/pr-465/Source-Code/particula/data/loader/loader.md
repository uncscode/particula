# Loader

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader

> Auto-generated documentation for [particula.data.loader](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py) module.

## data_format_checks

[Show source in loader.py:94](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L94)

Check if the data is in the correct format.

#### Arguments

- `data` *List[str]* - A list of strings containing the raw data.
- `data_checks` *dict* - Dictionary containing the format checks.

#### Returns

- `List[str]` - A list of strings containing the formatted data.

#### Raises

- `TypeError` - If data is not a list.

#### Examples

```python
>>> data = ['row 1', 'row 2', 'row 3']
>>> data_checks = {
...     "characters": [0, 10],
...     "char_counts": {",": 2, "/": 0, ":": 0},
...     "skip_rows": 0,
...     "skip_end": 0
... }
>>> formatted_data = data_format_checks(data, data_checks)
>>> print(formatted_data)
['row 2']
```

#### Signature

```python
def data_format_checks(data: List[str], data_checks: dict) -> List[str]: ...
```



## data_raw_loader

[Show source in loader.py:21](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L21)

Load raw data from a file at the specified file path and return it as a
list of strings.

#### Arguments

- `file_path` *str* - The file path of the file to read.

#### Returns

- `list` - The raw data read from the file as a list of strings.

#### Examples

```python
>>> data = data_raw_loader('my_file.txt')
Loading data from: my_file.txt
>>> print(data)
['line 1', 'line 2', 'line 3']
```

#### Signature

```python
def data_raw_loader(file_path: str) -> list: ...
```



## filter_list

[Show source in loader.py:47](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L47)

A pass filter of rows from a list of strings.
Each row must contain a specified number of characters to pass the filter.
The number of characters to count is specified in the char_counts
dictionary. The keys are the characters to count, and the values are the
exact count required for each character in each row.

#### Arguments

----------
    - `data` *List[str]* - A list of strings to filter.
        A list of strings to filter.
    - `char_counts` *dict* - A dictionary of character counts to select by.
        The keys are the characters to count, and the values are the
        count required for each character.

#### Returns

----------
    - `List[str]` - A new list of strings containing only the rows that meet the
    character count requirements.

#### Raises

----------
    - `UserWarning` - If more than 90% of the rows are filtered out, and it
        includes the character(s) used in the filter.

#### Examples

----------

```python
>>> data = ['apple,banana,orange', 'pear,kiwi,plum',
            'grapefruit,lemon']
>>> char_counts = {',': 2}
>>> filtered_data = filter_rows_by_count(data, char_counts)
>>> print(filtered_data)
['apple,banana,orange', 'pear,kiwi,plum']
```

#### Signature

```python
def filter_list(data: List[str], char_counts: dict) -> List[str]: ...
```



## general_data_formatter

[Show source in loader.py:348](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L348)

Formats and samples the data to get the time and data streams.

#### Arguments

----------
data : list
    The list of strings containing the data.
data_checks : dict
    A dictionary of data format checks to apply to the data.
data_column : list
    The list of indices of the columns containing the data.
time_column : Union[int, List[int]]
    The index or indices of the column(s) containing the time information.
time_format : str
    The format of the time information, e.g. '%Y-%m-%d %H:%M:%S'.
delimiter : str, default=','
    The delimiter used to separate columns in the data.
date_offset : str, default=None
    A fixed date offset to add to the timestamp in front.
seconds_shift : int, default=0
    A number of seconds to add to the timestamp.

#### Returns

-------
Tuple[np.array, np.array]
    A tuple containing two np.array objects: the first contains the
    epoch times, and the second contains the data.

#### Signature

```python
def general_data_formatter(
    data: list,
    data_checks: dict,
    data_column: list,
    time_column: Union[int, List[int]],
    time_format: str,
    delimiter: str = ",",
    header_row: int = 0,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = "UTC",
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## get_files_in_folder_with_size

[Show source in loader.py:573](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L573)

Returns a list of files in the specified folder and subfolder that
match the given filename pattern and have a size greater than the
specified minimum size.

#### Arguments

----------
path : str
    The path to the parent folder.
subfolder : str
    The name of the subfolder containing the files.
filename_regex : str
    A regular expression pattern for matching the filenames.
min_size : int, optional
    The minimum file size in bytes (default is 10).

#### Returns

-------
Tuple[List[str], List[str], List[int]]
    A tuple containing three lists:
    - The filenames that match the pattern and size criteria
    - The full paths to the files
    - The file sizes in bytes

#### Signature

```python
def get_files_in_folder_with_size(
    path: str, subfolder: str, filename_regex: str, min_size: int = 10
) -> Tuple[List[str], List[str], List[int]]: ...
```



## keyword_to_index

[Show source in loader.py:414](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L414)

Convert a keyword indicating a position in the header to its index.

This function takes a keyword which can be either an integer index or
a string representing the column name. If the keyword is an integer,
it's assumed to directly represent the index. If it's a string, the
function searches for the keyword in the header list and returns its index.

#### Arguments

keyword (Union[str, int]):
    The keyword representing the column's position in the header.
    It can be an integer index or a string for the column name.
- `header` *List[str]* - The list of column names (header) of the data.

#### Returns

- `int` - The index of the column in the header.

#### Raises

ValueError:
    If the keyword is a string and is not found in the header,
    or if the keyword is an integer but out of range of the header.

#### Signature

```python
def keyword_to_index(keyword: Union[str, int], header: List[str]) -> int: ...
```



## load_lake

[Show source in loader.py:822](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L822)

Load datalake object from a pickle file.

Args
----------
path : str
    Path to load pickle file.

Returns
-------
data_lake : DataLake
    Loaded DataLake object.

#### Signature

```python
def load_lake(path: str, sufix_name: Optional[str] = None) -> Lake: ...
```

#### See also

- [Lake](./lake.md#lake)



## load_stream

[Show source in loader.py:736](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L736)

Load stream object from a pickle file.

Args
----------
path : str
    Path to load pickle file.
sufix_name : str, optional
    Suffix to add to pickle file name. The default is None.
folder : str, optional
    Folder to load pickle file from. The default is 'output'.

Returns
-------
Stream
    Loaded Stream object.

#### Signature

```python
def load_stream(
    path: str, sufix_name: Optional[str] = None, folder: Optional[str] = "output"
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## netcdf_data_1d_load

[Show source in loader.py:882](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L882)

Given a netCDF file path and settings, returns a tuple containing the
epoch time, header, and data as a numpy array. We do apply the mask to the
data, and fill the masked values with nan.

#### Arguments

----------
    - `file_path` *str* - The path to the netCDF file.
    - `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

-------
    Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
    header, and data as a numpy array.

Errors:
------
    - `KeyError` - If the settings dictionary does not contain 'data_1d'.

#### Signature

```python
def netcdf_data_1d_load(
    file_path: str, settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]: ...
```



## netcdf_data_2d_load

[Show source in loader.py:941](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L941)

Given a netCDF file path and settings, returns a tuple containing the
epoch time, header, and data as a numpy array. We do apply the mask to the
data, and fill the masked values with nan.

#### Arguments

----------
    - `file_path` *str* - The path to the netCDF file.
    - `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

-------
    Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
    header, and data as a numpy array.

Errors:
------
    - `KeyError` - If the settings dictionary does not contain 'data_2d'.

#### Signature

```python
def netcdf_data_2d_load(
    file_path: str, settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]: ...
```



## netcdf_get_epoch_time

[Show source in loader.py:851](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L851)

Given a netCDF file path and settings, returns an array of epoch times in
seconds as a float.

Currently only uses ARM 1.2 netCDF files (base_time + time_offset)

#### Arguments

----------
    - `file_path` *str* - The path to the netCDF file.
    - `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

-------
    - `np.ndarray` - An array of epoch times, in seconds as a float.

#### Signature

```python
def netcdf_get_epoch_time(file_path: str, settings: dict) -> np.ndarray: ...
```



## netcdf_info_print

[Show source in loader.py:995](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L995)

Prints information about a netCDF file. Useful for generating settings
dictionaries.

#### Arguments

----------
    - `file_path` *str* - The path to the netCDF file.
    - `file_return` *bool* - If True, returns the netCDF file object.
        Defaults to False.

#### Returns

-------
    - `nc_file` *netCDF4.Dataset* - The netCDF file object.

#### Signature

```python
def netcdf_info_print(file_path, file_return=False): ...
```



## non_standard_date_location

[Show source in loader.py:535](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L535)

Extracts the date from a non-standard location in the data.

#### Arguments

----------
data : list
    A list of strings representing the data.
date_location : dict
    A dictionary specifying the method for extracting the date from the
    data.
    Supported methods include:
        - `-` *'file_header_block'* - The date is located in the file header
            block, and its position is specified by the 'row',
            'delimiter', and 'index' keys.

#### Returns

-------
str
    The date extracted from the specified location in the data.

#### Raises

------
ValueError
    If an unsupported or invalid method is specified in date_location.

#### Signature

```python
def non_standard_date_location(data: list, date_location: dict) -> str: ...
```



## parse_time_column

[Show source in loader.py:162](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L162)

Parses the time column of a data line and returns it as a timestamp.

#### Arguments

----------
time_column : Union[int, List[int]]
    The index or indices of the column(s) containing the time information.
time_format : str
    The format of the time information, e.g. '%Y-%m-%d %H:%M:%S'.
line : str
    The data line to parse.
date_offset : Optional[str], default=None
    A fixed date offset to add to the timestamp in front.
seconds_shift : int, default=0
    A number of seconds to add to the timestamp.

#### Returns

-------
float
    The timestamp corresponding to the time information in the data line,
    in seconds since the epoch.

#### Raises

------
ValueError
    If an invalid time column or format is specified.

#### Signature

```python
def parse_time_column(
    time_column: Union[int, List[int]],
    time_format: str,
    line: np.ndarray,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = "UTC",
) -> float: ...
```



## sample_data

[Show source in loader.py:224](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L224)

Samples the data to get the time and data streams.

#### Arguments

-----------
data : List[str]
    The input data in the form of a list of strings.
time_column : int
    The index of the column that contains the time values.
time_format : str
    The format string that specifies the time format.
data_columns : List[int]
    The indices of the columns that contain the data values.
delimiter : str
    The delimiter character used to separate columns in the input data.
date_offset : str, optional
    A string that represents an offset in the date, in the format
    'days:hours:minutes:seconds'. Defaults to None.
seconds_shift : int, optional
    An integer that represents a time shift in seconds. Defaults to 0.
timezone_identifier : str, optional
    What timezone the data is in. Defaults to 'UTC'.

#### Returns

--------
Tuple[np.ndarray, np.ndarray]
    A tuple of two numpy arrays - epoch_time and data_array:
    - epoch_time : np.ndarray
        A 1-D numpy array of epoch times.
    - data_array : np.ndarray
        A 2-D numpy array of data values.

#### Raises

-------
ValueError:
    - If the data value is not in the correct format.
    - If no match for data value is found.

#### Signature

```python
def sample_data(
    data: List[str],
    time_column: Union[int, List[int]],
    time_format: str,
    data_columns: List[int],
    delimiter: str,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = "UTC",
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## save_lake

[Show source in loader.py:774](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L774)

Save lake object as a pickle file.

Args
----------
data_lake : DataLake
    DataLake object to be saved.
path : str
    Path to save pickle file.
sufix_name : str, optional
    Suffix to add to pickle file name. The default is None.

#### Signature

```python
def save_lake(
    path: str,
    lake: Lake,
    sufix_name: Optional[str] = None,
    folder: Optional[str] = "output",
) -> None: ...
```

#### See also

- [Lake](./lake.md#lake)



## save_stream

[Show source in loader.py:688](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L688)

Save stream object as a pickle file.

Args
----------
stream : Stream
    Stream object to be saved.
path : str
    Path to save pickle file.
sufix_name : str, optional
    Suffix to add to pickle file name. The default is None.

#### Signature

```python
def save_stream(
    path: str,
    stream: Stream,
    sufix_name: Optional[str] = None,
    folder: Optional[str] = "output",
) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)



## save_stream_to_csv

[Show source in loader.py:624](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L624)

Save stream object as a CSV file, with an option to include formatted time.

#### Arguments

----------
stream : Stream
    Stream object to be saved.
path : str
    Path where the CSV file will be saved.
suffix_name : str, optional
    Suffix to add to CSV file name. The default is None.
folder : str, optional
    Subfolder within path to save the CSV file. The default is 'output'.
include_time : bool, optional
    Whether to include time data in the first column. The default is True.

#### Signature

```python
def save_stream_to_csv(
    stream: Stream,
    path: str,
    suffix_name: Optional[str] = None,
    folder: Optional[str] = "output",
    include_time: bool = True,
) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)



## sizer_data_formatter

[Show source in loader.py:448](https://github.com/Gorkowski/particula/blob/main/particula/data/loader.py#L448)

Formats data from a particle sizer.

Args
----------
data : List[str]
    The data to be formatted.
data_checks : Dict[str, Any]
    Dictionary specifying the formatting requirements for the data.
data_sizer_reader : Dict[str, str]
    Dictionary containing information about the sizer data format.
time_column : int
    The index of the time column in the data.
time_format : str
    The format of the time information.
delimiter : str, default=','
    The delimiter used in the data.
date_offset : str, default=None
    The date offset to add to the timestamp.
seconds_shift : int, default=0
    The number of seconds to add to the timestamp.
timezone_identifier : str, default='UTC'
    The timezone identifier for the data.

Returns
-------
Tuple[np.ndarray, List(str) np.ndarray, np.ndarray]
    A tuple containing the epoch time, the Dp header, and the data arrays.

#### Signature

```python
def sizer_data_formatter(
    data: List[str],
    data_checks: Dict[str, Any],
    data_sizer_reader: Dict[str, str],
    time_column: Union[int, List[int]],
    time_format: str,
    delimiter: str = ",",
    header_row: int = 0,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = "UTC",
) -> Tuple[np.ndarray, np.ndarray, list]: ...
```
