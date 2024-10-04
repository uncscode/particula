# Loader

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader

> Auto-generated documentation for [particula.data.loader](https://github.com/uncscode/particula/blob/main/particula/data/loader.py) module.

## data_format_checks

[Show source in loader.py:154](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L154)

Validate and format raw data according to specified checks.

#### Arguments

- `data` - List of strings containing the raw data to be checked.
- `data_checks` - Dictionary specifying the format checks to apply,
    such as character limits, character counts, and rows to skip.

#### Returns

A list of strings containing the validated and formatted data.

#### Raises

- `TypeError` - If `data` is not provided as a list.

#### Examples

``` py title="Validate line based on counts"
data = ['row 1', 'row 2', 'row 3']
data_checks = {
    "characters": [0, 10],
    "char_counts": {",": 2, "/": 0, ":": 0},
    "skip_rows": 0,
    "skip_end": 0
}
formatted_data = data_format_checks(data, data_checks)
print(formatted_data)
['row 2']
```

#### Signature

```python
def data_format_checks(data: List[str], data_checks: dict) -> List[str]: ...
```



## data_raw_loader

[Show source in loader.py:23](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L23)

Loads raw data from file.

Load raw data from a file at the specified file path and return it as a
list of strings. Attempts to handle UTF-8, UTF-16, and UTF-32 encodings.
Defaults to UTF-8 if no byte order mark (BOM) is found.

#### Arguments

- `file_path` *str* - The file path of the file to read.

#### Returns

- `list` - The raw data read from the file as a list of strings.

#### Examples

``` py title="Load my_file.txt"
data = data_raw_loader('my_file.txt')
Loading data from: my_file.txt
print(data)
['line 1', 'line 2', 'line 3']
```

#### Signature

```python
def data_raw_loader(file_path: str) -> list: ...
```



## filter_list

[Show source in loader.py:72](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L72)

Filter rows from a list of strings based on character counts.

Each row must contain a specified number of certain characters to pass
the filter. The `char_counts` dictionary specifies the characters to count
and the exact count required for each character in each row.

#### Arguments

- `data` - A list of strings to be filtered.
- `char_counts` - A dictionary specifying character counts for filtering.
    The keys are the characters to count, and the values are the
    required counts for each character in a row.

#### Returns

A new list of strings containing only the rows that meet the
character count requirements.

#### Raises

- `UserWarning` - If more than 90% of the rows are filtered out, indicating
    that the filter may be too strict based on the specified
    character(s).

#### Examples

``` py title="Filter rows based on comma counts"
data = ['apple,banana,orange',
         'pear,kiwi,plum', 'grapefruit,lemon']
char_counts = {',': 2}
filtered_data = filter_list(data, char_counts)
print(filtered_data)
['apple,banana,orange', 'pear,kiwi,plum']
```

#### Signature

```python
def filter_list(data: List[str], char_counts: dict) -> List[str]: ...
```



## general_data_formatter

[Show source in loader.py:454](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L454)

Format and sample data to extract time and data streams.

#### Arguments

- `data` - List of strings containing the raw data.
- `data_checks` - Dictionary specifying validation rules for the data.
- `data_column` - List of indices identifying the columns containing the
    data.
- `time_column` - Index or list of indices identifying the column(s)
    containing the time information.
- `time_format` - String specifying the format of the time information,
    e.g., '%Y-%m-%d %H:%M:%S'.
- `delimiter` - String used to separate columns in the data. Default is ','.
- `header_row` - Index of the row containing column names. Default is 0.
- `date_offset` - Optional string to add as a fixed offset to the timestamp.
    Default is None.
- `seconds_shift` - Number of seconds to add to the timestamp. Default is 0.
- `timezone_identifier` - Timezone identifier for the timestamps.
    Default is 'UTC'.

#### Returns

Tuple (np.ndarray, np.ndarray):
    - The first array contains the epoch times.
    - The second array contains the corresponding data values.

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

[Show source in loader.py:675](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L675)

Returns a list of files in the specified folder and subfolder that
match the given filename pattern and have a size greater than the
specified minimum size.

#### Arguments

path : str
    The path to the parent folder.
subfolder : str
    The name of the subfolder containing the files.
filename_regex : str
    A regular expression pattern for matching the filenames.
min_size : int, optional
    The minimum file size in bytes (default is 10).

#### Returns

Tuple(List[str], List[str], List[int]):
- `-` *file_list* - The filenames that match the pattern and size criteria.
- `-` *full_path* - The full paths to the files.
- `-` *file_size* - The file sizes in bytes.

#### Signature

```python
def get_files_in_folder_with_size(
    path: str, subfolder: str, filename_regex: str, min_size: int = 10
) -> Tuple[List[str], List[str], List[int]]: ...
```



## keyword_to_index

[Show source in loader.py:521](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L521)

Convert a keyword representing a column position in the header to
its index.

This function processes a keyword that can either be an integer index
or a string corresponding to a column name. If the keyword is an integer,
it is treated as the direct index of the column. If the keyword is a
string, the function searches the header list for the column name
and returns its index.

#### Arguments

- `keyword` - The keyword representing the column's position in the header.
    It can be an integer index or a string specifying the column name.
- `header` - A list of column names (header) in the data.

#### Returns

The index of the column in the header.

#### Raises

- `ValueError` - If the keyword is a string and is not found in the header,
    or if the keyword is an integer but is out of the header's
    index range.

#### Signature

```python
def keyword_to_index(keyword: Union[str, int], header: List[str]) -> int: ...
```



## load_lake

[Show source in loader.py:956](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L956)

Load a lake object by loading individual streams from separate pickle files.

#### Arguments

- `path` - Path to load pickle files.
- `suffix_name` - Suffix to add to pickle file names. The default is None.
- `folder` - Folder to load pickle files from. The default is 'output'.

#### Returns

- `Lake` - Reconstructed Lake object.

#### Signature

```python
def load_lake(
    path: str, suffix_name: Optional[str] = None, folder: str = "output"
) -> Lake: ...
```

#### See also

- [Lake](./lake.md#lake)



## load_stream

[Show source in loader.py:845](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L845)

Load stream object from a pickle file.

Args
----------
path : str
    Path to load pickle file.
suffix_name : str, optional
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
    path: str, suffix_name: Optional[str] = None, folder: str = "output"
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## netcdf_data_1d_load

[Show source in loader.py:1067](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L1067)

Given a netCDF file path and settings, returns a tuple containing the
epoch time, header, and data as a numpy array. We do apply the mask to the
data, and fill the masked values with nan.

#### Arguments

- `file_path` *str* - The path to the netCDF file.
- `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
header, and data as a numpy array.

Errors:
    - `KeyError` - If the settings dictionary does not contain 'data_1d'.

#### Signature

```python
def netcdf_data_1d_load(
    file_path: str, settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]: ...
```



## netcdf_data_2d_load

[Show source in loader.py:1123](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L1123)

Given a netCDF file path and settings, returns a tuple containing the
epoch time, header, and data as a numpy array. We do apply the mask to the
data, and fill the masked values with nan.

#### Arguments

- `file_path` *str* - The path to the netCDF file.
- `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
header, and data as a numpy array.

Errors:
    - `KeyError` - If the settings dictionary does not contain 'data_2d'.

#### Signature

```python
def netcdf_data_2d_load(
    file_path: str, settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]: ...
```



## netcdf_get_epoch_time

[Show source in loader.py:1041](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L1041)

Given a netCDF file path and settings, returns an array of epoch times in
seconds as a float.

Currently only uses ARM 1.2 netCDF files (base_time + time_offset)

#### Arguments

- `file_path` *str* - The path to the netCDF file.
- `settings` *dict* - A dictionary containing settings for the instrument.

#### Returns

- `np.ndarray` - An array of epoch times, in seconds as a float.

#### Signature

```python
def netcdf_get_epoch_time(file_path: str, settings: dict) -> np.ndarray: ...
```



## netcdf_info_print

[Show source in loader.py:1172](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L1172)

Prints information about a netCDF file. Useful for generating settings
dictionaries.

#### Arguments

- `file_path` *str* - The path to the netCDF file.
- `file_return` *bool* - If True, returns the netCDF file object.
    Defaults to False.

#### Returns

- `nc_file` *netCDF4.Dataset* - The netCDF file object.

#### Signature

```python
def netcdf_info_print(file_path, file_return=False): ...
```



## non_standard_date_location

[Show source in loader.py:647](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L647)

Extracts the date from a non-standard location in the data.

#### Arguments

- `data` - A list of strings representing the data.
- `date_location` - A dictionary specifying the method for extracting the
    date from the data.
        - `-` *'file_header_block'* - The date is located in the file header
            block, and its position is specified by the 'row',
            'delimiter', and 'index' keys.

#### Returns

- `str` - The date extracted from the specified location in the data.

#### Raises

- `ValueError` - If an unsupported or invalid method is specified in
    date_location.

#### Signature

```python
def non_standard_date_location(data: list, date_location: dict) -> str: ...
```



## parse_time_column

[Show source in loader.py:223](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L223)

Parse the time column(s) from a data line and return the timestamp.

#### Arguments

- `time_column` - Index or list of indices identifying the column(s)
    containing the time information.
- `time_format` - String specifying the format of the time information,
    e.g., '%Y-%m-%d %H:%M:%S'.
- `line` - A numpy array representing the data line to parse.
- `date_offset` - Optional string representing a fixed offset to add
    to the timestamp. Default is None.
- `seconds_shift` - Number of seconds to add to the timestamp. Default is 0.
- `timezone_identifier` - Timezone identifier for the timestamp.
    Default is 'UTC'.

#### Returns

A float representing the timestamp in seconds since the epoch.

#### Raises

- `ValueError` - If the specified time column or format is invalid.

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



## replace_list

[Show source in loader.py:119](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L119)

Replace characters in each string of a list based on a replacement
dictionary.

Each character specified in the `replace_dict` will be replaced with the
corresponding value in every string in the input list.

#### Arguments

- `data` - A list of strings in which the characters will be replaced.
- `replace_dict` - A dictionary specifying character replacements.
    The keys are the characters to be replaced, and the values are the
    replacement characters or strings.

#### Returns

A new list of strings with the replacements applied.

#### Examples

``` py title="Replace characters in a list of strings"
data = ['apple[banana]orange', '[pear] kiwi plum']
replace_dict = {'[': '', ']': ''}
replaced_data = replace_list(data, replace_dict)
print(replaced_data)
['applebananaorange', 'pear kiwi plum']
```

#### Signature

```python
def replace_list(data: List[str], replace_dict: Dict[str, str]) -> List[str]: ...
```



## sample_data

[Show source in loader.py:291](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L291)

Extract time and data streams from input data.

#### Arguments

- `data` - List of strings containing the input data.
- `time_column` - Index or list of indices indicating the column(s)
    containing the time values.
- `time_format` - Format string specifying the time format, e.g.,
    '%Y-%m-%d %H:%M:%S'.
- `data_columns` - List of indices identifying the columns containing
    the data values.
- `delimiter` - Character used to separate columns in the input data.
- `date_offset` - Optional string representing an offset to apply to
    the date, in the format 'days:hours:minutes:seconds'.
    Default is None.
- `seconds_shift` - Number of seconds to shift the timestamps. Default is 0.
- `timezone_identifier` - Timezone of the data. Default is 'UTC'.

#### Returns

Tuple (np.ndarray, np.ndarray):
    - `-` *`epoch_time`* - A 1-D numpy array of epoch times.
    - `-` *`data_array`* - A 2-D numpy array of data values.

#### Raises

- `ValueError` - If the data is not in the expected format or
    if no matching data value is found.

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

[Show source in loader.py:884](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L884)

Save each stream in the lake as separate pickle files.

#### Arguments

- `path` - Path to save pickle files.
- `lake` - Lake object to be saved.
- `suffix_name` - Suffix to add to pickle file names. The default is None.
- `folder` - Folder to save pickle files. The default is 'output'.

#### Signature

```python
def save_lake(
    path: str, lake: Lake, suffix_name: Optional[str] = None, folder: str = "output"
) -> None: ...
```

#### See also

- [Lake](./lake.md#lake)



## save_stream

[Show source in loader.py:796](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L796)

Save stream object as a pickle file.

Args
----------
stream : Stream
    Stream object to be saved.
path : str
    Path to save pickle file.
suffix_name : str, optional
    Suffix to add to pickle file name. The default is None.

#### Signature

```python
def save_stream(
    path: str, stream: Stream, suffix_name: Optional[str] = None, folder: str = "output"
) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)



## save_stream_to_csv

[Show source in loader.py:723](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L723)

Save stream object as a CSV file, with an option to include formatted time.

#### Arguments

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
include_iso_datatime : bool, optional
    Whether to include ISO formatted datetime in the second column.
    The default is True. The format is ISO 8601,
    '2021-01-01T00:00:00Z'.

#### Signature

```python
def save_stream_to_csv(
    stream: Stream,
    path: str,
    suffix_name: Optional[str] = None,
    folder: str = "output",
    include_time: bool = True,
) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)



## sizer_data_formatter

[Show source in loader.py:558](https://github.com/uncscode/particula/blob/main/particula/data/loader.py#L558)

Format data from a particle sizer into structured arrays.

#### Arguments

- `data` - List of raw data strings to be formatted.
- `data_checks` - Dictionary specifying validation rules for the data.
- `data_sizer_reader` - Dictionary containing mappings for interpreting
    the sizer data format.
- `time_column` - Index or list of indices indicating the position of
    the time column(s) in the data.
- `time_format` - Format string for parsing time information in the data.
- `delimiter` - Delimiter used to separate values in the data.
    Default is ','.
- `header_row` - Row index of the header containing column names.
    Default is 0.
- `date_offset` - Optional string representing an offset to add to
    timestamps. Default is None.
- `seconds_shift` - Number of seconds to shift the timestamps.
    Default is 0.
- `timezone_identifier` - Timezone identifier for the data timestamps.
    Default is 'UTC'.

#### Returns

Tuple(np.ndarray, np.ndarray, list):
    - A numpy array of epoch times.
    - A numpy array of Dp header values.
    - A list of numpy arrays representing the data.

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
