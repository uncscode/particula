# Mixin

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Mixin

> Auto-generated documentation for [particula.data.mixin](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py) module.

## ChecksCharCountsMixin

[Show source in mixin.py:430](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L430)

Mixin class for setting the character counts for data checks.

#### Signature

```python
class ChecksCharCountsMixin:
    def __init__(self): ...
```

### ChecksCharCountsMixin().set_char_counts

[Show source in mixin.py:436](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L436)

Set the required character counts for the data checks. This is
the number of times a character should appear in a line of the data
file, for it to be considered valid, and proceed with data parsing.

#### Arguments

- `char_counts` - Dictionary of characters and their required counts
    for the data checks. The keys are the characters, and the
    values are the required counts. e.g. {",": 4, ":": 0}.

#### Examples

``` py title="Set number of commas"
char_counts = {",": 4}
# valid line: '1,2,3,4'
# invalid line removed: '1,2,3'
```

``` py title="Filter out specific words"
char_counts = {"Temp1 Error": 0}
# valid line: '23.4, 0.1, 0.2, no error'
# invalid line removed: '23.4, 0.1, 0.2, Temp1 Error'
```

#### Signature

```python
def set_char_counts(self, char_counts: dict[str, int]): ...
```



## ChecksCharactersMixin

[Show source in mixin.py:395](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L395)

Mixin class for setting the character length range for data checks.

#### Signature

```python
class ChecksCharactersMixin:
    def __init__(self): ...
```

### ChecksCharactersMixin().set_characters

[Show source in mixin.py:401](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L401)

Set the character length range for the data checks. This is
how many characters are expected a line of the data file, for it to
be considered valid, and proceed with data parsing.

#### Arguments

- `characters` - List of one (or two) integers for the minimum (and
    maximum) number of characters expected in a line of the data
    file. e.g. [10, 100] for 10 to 100 characters. or [10] for
    10 or more characters.

#### Examples

``` py title="Set minimum characters"
characters = [5]
# valid line: '1,2,3,4,5'
# invalid line: '1,2'
```

``` py title="Set range of characters"
characters = [5, 10]
# valid line: '1,2,3,4,5'
# invalid line: '1,2,3,4,5,6,7,8,9,10,11'
# invalid line: '1,2'
```

#### Signature

```python
def set_characters(self, characters: list[int]): ...
```



## ChecksReplaceCharsMixin

[Show source in mixin.py:508](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L508)

Mixin class for setting the characters to replace in the data lines.

#### Signature

```python
class ChecksReplaceCharsMixin:
    def __init__(self): ...
```

### ChecksReplaceCharsMixin().set_replace_chars

[Show source in mixin.py:514](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L514)

Set the characters to replace in the data lines.

This is useful to replace unwanted characters from the data lines
before converting the data to the required format. Each key in the
replace_dict represents the character to replace, and the corresponding
value is the replacement target.

#### Arguments

- `replace_dict` *dict* - Dictionary with keys as characters to replace
    and values as the replacement targets.

#### Examples

``` py title="Replace brackets with empty string"
replace_dict = {"[": "", "]": ""}
# data: '[1], [2], [3]' -> '1, 2, 3'
```

``` py title="Replace spaces with underscores"
replace_dict = {" ": "_"}
# data: '1, 2, 3' -> '1,_2,_3'
```

``` py title="Replace multiple characters"
replace_dict = {"[": "", "]": "", "
": " "}
# data: '[1]
[2]
[3]' -> '1 2 3'
```

#### Returns

- `self` - The instance of the class to allow for method chaining.

#### References

[Python str.replace](https://docs.python.org/3/library/stdtypes.html#str.replace)

#### Signature

```python
def set_replace_chars(self, replace_chars: dict[str, str]): ...
```



## ChecksSkipEndMixin

[Show source in mixin.py:486](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L486)

Mixin class for setting the number of rows to skip at the end.

#### Signature

```python
class ChecksSkipEndMixin:
    def __init__(self): ...
```

### ChecksSkipEndMixin().set_skip_end

[Show source in mixin.py:492](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L492)

Set the number of rows to skip at the end of the file.

#### Arguments

- `skip_end` *int* - Number of rows to skip at the end of the file.

#### Examples

``` py title="Skip last row"
skip_end = 10
# Skip the last 10 row of the file.
```

#### Signature

```python
def set_skip_end(self, skip_end: int = 0): ...
```



## ChecksSkipRowsMixin

[Show source in mixin.py:463](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L463)

Mixin class for setting the number of rows to skip at the beginning.

#### Signature

```python
class ChecksSkipRowsMixin:
    def __init__(self): ...
```

### ChecksSkipRowsMixin().set_skip_rows

[Show source in mixin.py:469](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L469)

Set the number of rows to skip at the beginning of the file.

#### Arguments

- `skip_rows` *int* - Number of rows to skip at the beginning of the
    file.

#### Examples

``` py title="Skip the first 2 rows"
skip_rows = 2
# Skip the first 2 rows of the file.
```

#### Signature

```python
def set_skip_rows(self, skip_rows: int = 0): ...
```



## DataChecksMixin

[Show source in mixin.py:120](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L120)

Mixin class for setting the data checks.

#### Signature

```python
class DataChecksMixin:
    def __init__(self): ...
```

### DataChecksMixin().set_data_checks

[Show source in mixin.py:126](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L126)

Dictionary of data checks to perform on the loaded data.

#### Arguments

- `checks` *dict* - Dictionary of data checks to perform on the loaded
    data. The keys are the names of the checks, and the values are
    the parameters for the checks.

#### Signature

```python
def set_data_checks(self, data_checks: Dict[str, Any]): ...
```



## DataColumnMixin

[Show source in mixin.py:138](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L138)

Mixin class for setting the data column.

#### Signature

```python
class DataColumnMixin:
    def __init__(self): ...
```

### DataColumnMixin().set_data_column

[Show source in mixin.py:144](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L144)

The data columns for the data files to load. Build with
`DataChecksBuilder`.

#### Arguments

- `data_columns` - List of column numbers or names for the data columns
    to load from the data files. The columns are indexed from 0.
    e.g. [3, 5] or ['data 1', 'data 3'].

#### Examples

``` py title="Single data column, index"
data_columns = [3]
# header: 'Time, Temp, data 1, data 2, data 3'
# line: '2021-01-01T12:00:00, 25.8, 1.2, 3.4' # load 1.2
```

``` py title="Single data column, name"
data_columns = ['data 1']
# header: 'Time, Temp, data 1, data 3, data 5'
# line: '2021-01-01T12:00:00, 25.8, 1.2, 3.4' # load 25.8
```

``` py title="Multiple data columns, index"
data_columns = [1, 3]
# header: 'Time, Temp, data 1, data 3, data 5'
# line: '2021-01-01T12:00:00, 25.8, 1.2, 3.4' # load 25.8, 3.4
```

``` py title="Multiple data columns, name"
data_columns = ['Temp', 'data 3']
# header: 'Time, Temp, data 1, data 3, data 5'
# line: '2021-01-01T12:00:00, 25.8, 1.2, 3.4' # load 25.8, 3.4
```

#### Signature

```python
def set_data_column(self, data_columns: Union[List[str], List[int]]): ...
```



## DataHeaderMixin

[Show source in mixin.py:182](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L182)

Mixin class for setting the data header for the Stream.

#### Signature

```python
class DataHeaderMixin:
    def __init__(self): ...
```

### DataHeaderMixin().set_data_header

[Show source in mixin.py:188](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L188)

Set the Stream headers corresponding to the data columns. This is
to improve the readability of the Stream data. The headers should be
in the same order as the data columns. These are also the same headers
that will be written to the output file or csv.

#### Arguments

- `headers` - List of headers corresponding to the data
    columns to load. e.g. ['data-1[m/s]', 'data_3[L]'].

#### Examples

``` py title="Single header"
headers = ['data-1[m/s]']
# Name the only data column as 'data-1[m/s]'.
```

``` py title="Multiple headers"
headers = ['data-1[m/s]', 'data-3[L]']
# Name the data columns as 'data-1[m/s]' and 'data-3[L]'.
```

#### Signature

```python
def set_data_header(self, headers: List[str]): ...
```



## DelimiterMixin

[Show source in mixin.py:296](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L296)

Mixin class for setting the delimiter.

#### Signature

```python
class DelimiterMixin:
    def __init__(self): ...
```

### DelimiterMixin().set_delimiter

[Show source in mixin.py:302](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L302)

Set the delimiter for the data files to load.

#### Arguments

- `delimiter` *str* - Delimiter for the data columns in the data files.
    e.g. ',' for CSV files or '	' for tab-separated files.

#### Examples

``` py title="CSV delimiter"
delimiter = ","
# CSV file with columns separated by commas.
```

``` py title="Tab delimiter"
delimiter = "	"
# Tab-separated file with columns separated by tabs.
```

``` py title="Space delimiter"
delimiter = " "
# Space-separated file with columns separated by spaces.
```

#### Signature

```python
def set_delimiter(self, delimiter: str): ...
```



## FileMinSizeBytesMixin

[Show source in mixin.py:74](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L74)

Mixin class for setting the minimum file size in bytes.

#### Signature

```python
class FileMinSizeBytesMixin:
    def __init__(self): ...
```

### FileMinSizeBytesMixin().set_file_min_size_bytes

[Show source in mixin.py:80](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L80)

Set the minimum file size in bytes for the data files to load.

#### Arguments

- `size` *int* - Minimum file size in bytes. Default is 10000 bytes.

#### Signature

```python
def set_file_min_size_bytes(self, size: int = 10000): ...
```



## FilenameRegexMixin

[Show source in mixin.py:37](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L37)

Mixin class for setting the filename regex.

#### Signature

```python
class FilenameRegexMixin:
    def __init__(self): ...
```

### FilenameRegexMixin().set_filename_regex

[Show source in mixin.py:43](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L43)

Set the filename regex for the data files to load.

#### Arguments

- `regex` *str* - Regular expression for the filenames, e.g.
    'data_*.csv'.

#### Examples

``` py title="Match all files"
regex = ".*"
# Match all files in the folder.
```

``` py title="Match CSV files"
regex = ".*.csv"
# Match all CSV files in the folder.
```

``` py title="Match specific files"
regex = "data_*.csv"
# Match files starting with 'data_' and ending with '.csv'.
```

#### References

[Explore Regex](https://regex101.com/)
[Python Regex Doc](https://docs.python.org/3/library/re.html)

#### Signature

```python
def set_filename_regex(self, regex: str): ...
```



## HeaderRowMixin

[Show source in mixin.py:90](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L90)

Mixin class for setting the header row.

#### Signature

```python
class HeaderRowMixin:
    def __init__(self): ...
```

### HeaderRowMixin().set_header_row

[Show source in mixin.py:96](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L96)

Set the header row for the data files to load.

#### Arguments

- `row` *int* - Row number for the header row in the data file, indexed
    from 0.

#### Examples

``` py title="Header row at the top"
row = 0
# line 0: 'Time, Temp, data 1, data 2, data 3'
```

``` py title="Header is third row"
row = 2
# line 0: "Experiment 1"
# line 1: "Date: 2021-01-01"
# line 2: 'Time, Temp, data 1, data 2, data 3'
```

#### Signature

```python
def set_header_row(self, row: int): ...
```



## RelativeFolderMixin

[Show source in mixin.py:8](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L8)

Mixin class for setting the relative data folder.

#### Signature

```python
class RelativeFolderMixin:
    def __init__(self): ...
```

### RelativeFolderMixin().set_relative_data_folder

[Show source in mixin.py:14](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L14)

Set the relative data folder for the folder with the data loading.

#### Arguments

- `folder` *str* - Relative path to the data folder.
    e.g. 'data_folder'. Where the data folder is located in
    project_path/data_folder.

#### Examples

``` py title="Set data folder"
folder = "data_folder"
# Set the data folder to 'data_folder'.
```

``` py title="Set a subfolder"
folder = "subfolder/data_folder"
# Set the data folder to 'subfolder/data_folder'.
```

#### Signature

```python
def set_relative_data_folder(self, folder: str): ...
```



## SizerConcentrationConvertFromMixin

[Show source in mixin.py:617](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L617)

Mixin class for setting to convert the sizer concentration to
a different scale.

#### Signature

```python
class SizerConcentrationConvertFromMixin:
    def __init__(self): ...
```

### SizerConcentrationConvertFromMixin().set_sizer_concentration_convert_from

[Show source in mixin.py:624](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L624)

Set to convert the sizer concentration from dw or (pmf) scale to
dN/dlogDp scale.

#### Arguments

- `convert_from` - Conversion flag to convert the sizer concentration
    from dw or (pmf) scale to dN/dlogDp scale. The option is only
    "dw" all other values are ignored.

#### Examples

``` py title="Convert from dw scale"
convert_from = "dw"
# Convert the sizer concentration from dw scale to dN/dlogDp scale.
```

``` py title="Convert Ignored"
convert_from = "pmf"
# Ignored, no conversion is performed, when loading the sizer data.
```

#### Signature

```python
def set_sizer_concentration_convert_from(self, convert_from: Optional[str] = None): ...
```



## SizerDataReaderMixin

[Show source in mixin.py:651](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L651)

Mixin class for the dictionary of the sizer data reader settings.

#### Signature

```python
class SizerDataReaderMixin:
    def __init__(self): ...
```

### SizerDataReaderMixin().set_data_sizer_reader

[Show source in mixin.py:657](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L657)

Dictionary of the sizer data reader settings for the data files.
Build with `SizerDataReaderBuilder`.

#### Arguments

- `data_sizer_reader` - Dictionary of the sizer data reader settings
    for the data files. The keys are the names of the settings,
    and the values are the parameters for the settings.

#### Signature

```python
def set_data_sizer_reader(self, data_sizer_reader: Dict[str, Any]): ...
```



## SizerEndKeywordMixin

[Show source in mixin.py:586](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L586)

Mixin class for setting the end key for the sizer data.

#### Signature

```python
class SizerEndKeywordMixin:
    def __init__(self): ...
```

### SizerEndKeywordMixin().set_sizer_end_keyword

[Show source in mixin.py:592](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L592)

Set the end keyword for the sizer data, to identify the end of
the sizer data block in the data files. This can be a string or an
integer (column index) to identify the end of the sizer data block.

#### Arguments

- `end_keyword` - End key for the sizer data in the data files.
    e.g. '789.3' or -3 for the 3rd column from the end.

#### Examples

``` py title="End key as a string"
end_key = "789.3"
# header: '... 689.1, 750.2, 789.3, Total Conc, Comments'
```

``` py title="End key as a column index"
end_key = -3
# header: '... 689.1, 750.2, 789.3, Total Conc, Comments'
```

#### Signature

```python
def set_sizer_end_keyword(self, end_key: Union[str, int]): ...
```



## SizerStartKeywordMixin

[Show source in mixin.py:555](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L555)

Mixin class for setting the start key for the sizer data.

#### Signature

```python
class SizerStartKeywordMixin:
    def __init__(self): ...
```

### SizerStartKeywordMixin().set_sizer_start_keyword

[Show source in mixin.py:561](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L561)

Set the start keyword for the sizer data, to identify the start of
the sizer data block in the data files. This can be a string or an
integer (column index) to identify the start of the sizer data block.

#### Arguments

- `start_keyword` - Start key for the sizer data in the data files.
    e.g. '25.8' or 3 for the 4th column

#### Examples

``` py title="Start key as a string"
start_key = "35.8"
# header: 'Time, Temp, 35.8, 36.0, 36.2, ...'
```

``` py title="Start key as a column index"
start_key = 2
# header: 'Time, Temp, 35.8, 36.0, 36.2, ...'
```

#### Signature

```python
def set_sizer_start_keyword(self, start_key: Union[str, int]): ...
```



## TimeColumnMixin

[Show source in mixin.py:213](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L213)

Mixin class for setting the time column.

#### Signature

```python
class TimeColumnMixin:
    def __init__(self): ...
```

### TimeColumnMixin().set_time_column

[Show source in mixin.py:219](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L219)

The time column for the data files to load. The time column is
used to convert the time data to an Unix-Epoch timestamp.

#### Arguments

- `columns` - List of column indexes for the time columns to
    load from the data files. The columns are indexed from 0.
    e.g. [0] or [1, 2] to combine 1 and 2 columns.

#### Examples

``` py title="Single time column"
columns = [0]
# Load the time data from the first column.
# line: '2021-01-01T12:00:00, 1.2, 3.4'
```

``` py title="Multiple time columns"
columns = [1, 2]
# Load the time data from the second and third columns.
# line: '1.2, 2021-01-01, 12:00:00'
```

#### Signature

```python
def set_time_column(self, columns: List[int]): ...
```



## TimeFormatMixin

[Show source in mixin.py:245](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L245)

Mixin class for setting the time format.

#### Signature

```python
class TimeFormatMixin:
    def __init__(self): ...
```

### TimeFormatMixin().set_time_format

[Show source in mixin.py:251](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L251)

Set the time format for the time data in the data files.

#### Arguments

- `time_format_str` *str* - Time format string for the time data in the
    data files. Default is ISO "%Y-%m-%dT%H:%M:%S", list "epoch"
    if the time data is in Unix-Epoch format. Use the Python time
    format codes otherwise,
    e.g. "%Y-%m-%dT%H:%M:%S" for '2021-01-01T12:00:00'.

#### Examples

``` py title="USA date format"
time_format_str = "%m/%d/%Y %H:%M:%S"
# e.g. '01/01/2021 12:00:00'
```

``` py title="European date format"
time_format_str = "%d/%m/%Y %H:%M:%S"
# e.g. '01/01/2021 12:00:00'
```

``` py title="ISO date format"
time_format_str = "%Y-%m-%dT%H:%M:%S"
# e.g. '2021-01-01T12:00:00'
```

``` py title="AM/PM time format"
time_format_str = "%Y-%m-%d %I:%M:%S %p"
# e.g. '2021-01-01 12:00:00 PM'
```

``` py title="Fractional seconds"
time_format_str = "%Y-%m-%dT%H:%M:%S.%f"
# e.g. '2021-01-01T12:00:00.123456'
```

#### References

- [Python Docs](
https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
- [Python Time Format](https://strftime.org/)

#### Signature

```python
def set_time_format(self, time_format_str: str = "%Y-%m-%dT%H:%M:%S"): ...
```



## TimeShiftSecondsMixin

[Show source in mixin.py:329](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L329)

Mixin class for setting the time shift in seconds.

#### Signature

```python
class TimeShiftSecondsMixin:
    def __init__(self): ...
```

### TimeShiftSecondsMixin().set_time_shift_seconds

[Show source in mixin.py:335](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L335)

Set the time shift in seconds for the time data in the data files.
This is helpful to match the time stamps of two data folders. This
shift is applied to all files loaded with this builder.

#### Arguments

- `shift` *int* - Time shift in seconds for the time data in the data
    files. Default is 0 seconds.

#### Examples

``` py title="Shift by 1 hour"
shift = 3600
# Shift the time data by 1 hour (3600 seconds).
```

``` py title="Shift by 1 day"
shift = 86400
# Shift the time data by 1 day (86400 seconds).
```

#### Signature

```python
def set_time_shift_seconds(self, shift: int = 0): ...
```



## TimezoneIdentifierMixin

[Show source in mixin.py:359](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L359)

Mixin class for setting the timezone identifier.

#### Signature

```python
class TimezoneIdentifierMixin:
    def __init__(self): ...
```

### TimezoneIdentifierMixin().set_timezone_identifier

[Show source in mixin.py:365](https://github.com/uncscode/particula/blob/main/particula/data/mixin.py#L365)

Set the timezone identifier for the time data in the data files.
The timezone shift is handled by the pytz library.

#### Arguments

- `timezone` *str* - Timezone identifier for the time data in the data
    files. Default is 'UTC'.

#### Examples

``` py title="List of Timezones"
timezone = "Europe/London"  # or "GMT"
```

``` py title="Mountain Timezone"
timezone = "America/Denver"  # or "MST7MDT"
```

``` py title="ETH Zurich Timezone"
timezone = "Europe/Zurich"  # or "CET"
```

#### References

[List of Timezones](
https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

#### Signature

```python
def set_timezone_identifier(self, timezone: str = "UTC"): ...
```
