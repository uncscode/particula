# Loader Interface

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader Interface

> Auto-generated documentation for [particula.data.loader_interface](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py) module.

## get_1d_stream

[Show source in loader_interface.py:226](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py#L226)

Loads and formats a 1D data stream from a file and initializes or updates
a Stream object.

#### Arguments

----------
file_path : str
    The path of the file to load data from.
first_pass : bool
    Whether this is the first time data is being loaded. If True, the
    stream is initialized.
    If False, raises an error as only one file can be loaded.
settings : dict
    A dictionary containing data formatting settings such as data checks,
    column names,
    time format, delimiter, and timezone information.
stream : Stream, optional
    An instance of Stream class to be updated with loaded data. Defaults
    to a new Stream object.

#### Returns

-------
Stream
    The Stream object updated with the loaded data and corresponding time
    information.

#### Raises

------
ValueError
    If `first_pass` is False, indicating data has already been loaded.
TypeError
    If `settings` is not a dictionary.
FileNotFoundError
    If the file specified by `file_path` does not exist.
KeyError
    If any required keys are missing in the `settings` dictionary.

#### Signature

```python
def get_1d_stream(
    file_path: str,
    settings: dict,
    first_pass: bool = True,
    stream: Optional[Stream] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## get_2d_stream

[Show source in loader_interface.py:338](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py#L338)

Initializes a 2D stream using the settings in the DataLake object.

#### Arguments

----------
    - `key` *str* - The key of the stream to initialise.
    - `path` *str* - The path of the file to load data from.
    - `first_pass` *bool* - Whether this is the first time loading data.

#### Returns

----------
    None.

#### Signature

```python
def get_2d_stream(
    file_path: str,
    settings: dict,
    first_pass: bool = True,
    stream: Optional[Stream] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## get_new_files

[Show source in loader_interface.py:13](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py#L13)

Scan a directory for new files based on import settings and stream status.

This function looks for files in a specified path using import settings.
It compares the new list of files with a pre-loaded list in the stream
object to determine which files are new. The comparison is made based on
file names and sizes. It returns a tuple with the paths of new files, a
boolean indicating if this was the first pass, and a list of file
information for new files.

#### Arguments

----------
path : str
    The top-level directory path to scan for files.
import_settings : dict
    A dictionary with 'relative_data_folder', 'filename_regex',
    and 'MIN_SIZE_BYTES' as keys
    used to specify the subfolder path and the regex pattern for filtering
    file names. It should also include 'min_size' key to specify the
    minimum size of the files to be considered.
loaded_list : list of lists
    A list of lists with file names and sizes that have already been
    loaded. The default is None. If None, it will be assumed that no
    files have been loaded.

#### Returns

-------
tuple of (list, bool, list)
    A tuple containing a list of full paths of new files, a boolean
    indicating if no previous files were loaded (True if it's the first
    pass), and a list of lists with new file names and sizes.

#### Raises

------
YourErrorType
    Explanation of when and why your error is raised and what it means.

#### Signature

```python
def get_new_files(
    path: str, import_settings: dict, loaded_list: Optional[list] = None
) -> tuple: ...
```



## load_files_interface

[Show source in loader_interface.py:109](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py#L109)

Load files into a stream object based on settings.

#### Arguments

----------
path : str
    The top-level directory path to scan for folders of data.
folder_settings : dict
    A dictionary with keys corresponding to the stream names and values
    corresponding to the settings for each stream. The settings can
    be generated using the settings_generator function.
stream : Stream, optional
    An instance of Stream class to be updated with loaded data. Defaults
    to a new Stream object.
- `sub_sample` - int, optional
    sub-sample only the first n files. Defaults to None.

#### Returns

-------
Stream
    The Stream object updated with the loaded data.

#### Signature

```python
def load_files_interface(
    path: str,
    settings: dict,
    stream: Optional[Stream] = None,
    sub_sample: Optional[int] = None,
) -> Stream: ...
```

#### See also

- [Stream](./stream.md#stream)



## load_folders_interface

[Show source in loader_interface.py:185](https://github.com/uncscode/particula/blob/main/particula/data/loader_interface.py#L185)

Load files into a lake object based on settings.

#### Arguments

----------
path : str
    The top-level directory path to scan for folders of data.
folder_settings : dict
    A dictionary with keys corresponding to the stream names and values
    corresponding to the settings for each stream. The settings can
    be generated using the settings_generator function.
lake : Lake, optional
    An instance of Lake class to be updated with loaded data. Defaults
    to a new Lake object.

#### Returns

-------
Lake
    The Lake object updated with the loaded data streams.

#### Signature

```python
def load_folders_interface(
    path: str, folder_settings: dict, lake: Optional[Lake] = None
) -> Lake: ...
```

#### See also

- [Lake](./lake.md#lake)
