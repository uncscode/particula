# Settings Generator

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Settings Generator

> Auto-generated documentation for [particula.data.settings_generator](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py) module.

## for_general_1d_load

[Show source in settings_generator.py:14](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L14)

Generate a settings dictionary for loading and checking 1D data from CSV
files.

#### Arguments

------------
- relative_data_folder (str): The folder path relative to the main script
    where data files are located. Default is 'instrument_data'.
- filename_regex (str): Regular expression pattern to match filenames in
    the data folder. Default is '*.csv'.
- file_min_size_bytes (int): Minimum size in bytes for files to be
    considered valid. Default is 10.
- header_row (int): The index of the row containing column headers
    (0-indexed). Default is 0.
- data_checks (Optional[dict]): A dictionary containing data quality
    checks such as character length, required character counts, rows to
    skip at the beginning or end. Defaults to basic checks if None.
- data_column (list of int): List of indices for columns containing data
    points to be loaded. Default is [3, 5].
- data_header (List[str]): List of strings representing the header names
    for data columns. Default is ['data 1', 'data 3'].
- time_column (List[int]): List of indices for columns containing time
    information. Default is [0, 1].
- time_format (str): String format for parsing time columns, using
    strftime conventions. Default is '%Y-%m-%d %H:%M:%S.%f'.
- delimiter (str): Character used to separate values in the file.
    Default is ','.
- time_shift_seconds (int): Number of seconds by which to shift time data
    (positive or negative). Default is 0.
- timezone_identifier (str): Timezone identifier for time conversion.
    Default is 'UTC'.

#### Returns

- `-` *dict* - A dictionary with settings for data loading procedures including
    file paths, size requirements, header information, and data check
    parameters.

#### Signature

```python
def for_general_1d_load(
    relative_data_folder: str = "instrument_data",
    filename_regex: str = "*.csv",
    file_min_size_bytes: int = 10,
    header_row: int = 0,
    data_checks: Optional[dict] = None,
    data_column: list = [3, 5],
    data_header: List[str] = ["data 1", "data 3"],
    time_column: List[int] = [0, 1],
    time_format: str = "%Y-%m-%d %H:%M:%S.%f",
    delimiter: str = ",",
    time_shift_seconds: int = 0,
    timezone_identifier: str = "UTC",
) -> dict: ...
```



## for_general_sizer_1d_2d_load

[Show source in settings_generator.py:89](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L89)

Generate settings for the 1D general file loader and the 2D general sizer
    file loader.

#### Arguments

- relative_data_folder (str): Path to the folder containing data files,
    relative to the script's location.
- filename_regex (str): Regex pattern to match filenames for loading.
- file_min_size_bytes (int): Minimum file size in bytes for a file to be
    considered valid for loading.
- header_row (int): Row index for the header (0-based) in the data files.
- data_checks (dict, optional): Specifications for data integrity checks
    to apply when loading data.
- data_1d_column (list of int): Column indices for 1D data extraction.
- data_1d_header (list of str): Header names corresponding to the
    `data_1d_column` indices.
- data_2d_dp_start_keyword (str): Keyword indicating the start of 2D data
    points in a file.
- data_2d_dp_end_keyword (str): Keyword indicating the end of 2D data
    points in a file.
- data_2d_convert_concentration_from (str, optional): Unit to convert from
    if concentration scaling is needed for 2D data.
- time_column (list of int): Column indices for time data extraction.
- time_format (str): Format string for parsing time data.
- delimiter (str): Delimiter character for splitting data in the file.
- time_shift_seconds (int): Seconds to shift the time data by.
- timezone_identifier (str): Timezone ID for time data interpretation.

#### Returns

- tuple of (dict, dict): A tuple containing two dictionaries with settings
    for the 1D and 2D data loaders.

The function defaults `data_checks` to basic validation criteria if not
    provided. It returns separate dictionaries for settings applicable to
    1D and 2D data loaders, which include file paths, size checks, and
    data parsing rules.

#### Signature

```python
def for_general_sizer_1d_2d_load(
    relative_data_folder: str = "instrument_data",
    filename_regex: str = "*.csv",
    file_min_size_bytes: int = 10,
    header_row: int = 0,
    data_checks: Optional[dict] = None,
    data_1d_column: list = [3, 5],
    data_1d_header: List[str] = ["data 1", "data 3"],
    data_2d_dp_start_keyword: str = "Date Time",
    data_2d_dp_end_keyword: str = "Total Conc",
    data_2d_convert_concentration_from: str = "dw/dlogdp",
    time_column: List[int] = [0, 1],
    time_format: str = "%Y-%m-%d %H:%M:%S.%f",
    delimiter: str = ",",
    time_shift_seconds: int = 0,
    timezone_identifier: str = "UTC",
) -> tuple: ...
```



## load_settings_for_lake

[Show source in settings_generator.py:260](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L260)

Load settings for Lake data from a JSON file. The settings file is
a dictionary of stream settings dictionaries.

Given a path and subfolder, this function searches for a JSON file
named 'lake_settings' with an optional suffix. It returns the settings
as a dictionary. If no file is found, or multiple files are found,
appropriate errors or warnings are raised.

#### Arguments

- `-` *path* - The path where the subfolder is located.
- `-` *subfolder* - The subfolder where the settings file is expected.
- `-` *settings_suffix* - An optional suffix for the settings
    file name. Default is an empty string.

#### Returns

- `-` *dict* - A dictionary of settings loaded from the file.

#### Raises

- `-` *FileNotFoundError* - If no settings file is found.
- `-` *Warning* - If more than one settings file is found.

#### Signature

```python
def load_settings_for_lake(
    path: str, subfolder: str = "", settings_suffix: str = ""
) -> dict: ...
```



## load_settings_for_stream

[Show source in settings_generator.py:186](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L186)

Load settings for Stream data from a JSON file.

Given a path and subfolder, this function searches for a JSON file
named 'stream_settings' with an optional suffix. It returns the settings
as a dictionary. If no file is found, or multiple files are found,
appropriate errors or warnings are raised.

#### Arguments

- `-` *path* - The path where the subfolder is located.
- `-` *subfolder* - The subfolder where the settings file is expected.
- `-` *settings_suffix* - An optional suffix for the settings
    file name. Default is an empty string.

#### Returns

- `-` *dict* - A dictionary of settings loaded from the file.

#### Raises

- `-` *FileNotFoundError* - If no settings file is found.
- `-` *Warning* - If more than one settings file is found.

#### Signature

```python
def load_settings_for_stream(
    path: str, subfolder: str, settings_suffix: str = ""
) -> dict: ...
```



## save_settings_for_lake

[Show source in settings_generator.py:307](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L307)

Save settings for lake data to a JSON file.

Given a dictionary of settings, this function saves it to a JSON file
named 'lake_settings' with an optional suffix in the specified filename.
The JSON file is formatted with a 4-space indentation.

#### Arguments

- `-` *settings* - The settings dictionary to be saved.
- `-` *path* - The path where the subfolder is located.
- `-` *subfolder* - The subfolder where the settings file will be saved.
- `-` *settings_suffix* - An optional suffix for the settings
    file name. Default is an empty string.

#### Returns

- None

#### Signature

```python
def save_settings_for_lake(
    settings: dict, path: str, subfolder: str = "", settings_suffix: str = ""
) -> None: ...
```



## save_settings_for_stream

[Show source in settings_generator.py:232](https://github.com/uncscode/particula/blob/main/particula/data/settings_generator.py#L232)

Save settings for lake data to a JSON file.

Given a dictionary of settings, this function saves it to a JSON file
named 'stream_settings' with an optional suffix in the specified filename.
The JSON file is formatted with a 4-space indentation.

#### Arguments

- `-` *settings* - The settings dictionary to be saved.
- `-` *path* - The path where the subfolder is located.
- `-` *subfolder* - The subfolder where the settings file will be saved.
- `-` *settings_suffix* - An optional suffix for the settings
    file name. Default is an empty string.

#### Returns

- None

#### Signature

```python
def save_settings_for_stream(
    settings: dict, path: str, subfolder: str, settings_suffix: str = ""
) -> None: ...
```
