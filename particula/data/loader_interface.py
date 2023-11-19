"""interface to import data to a data stream"""


from typing import Optional
import os
import numpy as np
from particula.data import loader, merger
from particula.data.stream import Stream
from particula.data.lake import Lake
from particula.util import convert


def get_new_files(
        path: str,
        import_settings: dict,
        loaded_list: Optional[list] = None,
) -> tuple:
    """
    Scan a directory for new files based on import settings and stream status.

    This function looks for files in a specified path using import settings.
    It compares the new list of files with a pre-loaded list in the stream
    object to determine which files are new. The comparison is made based on
    file names and sizes. It returns a tuple with the paths of new files, a
    boolean indicating if this was the first pass, and a list of file
    information for new files.

    Args:
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

    Returns:
    -------
    tuple of (list, bool, list)
        A tuple containing a list of full paths of new files, a boolean
        indicating if no previous files were loaded (True if it's the first
        pass), and a list of lists with new file names and sizes.

    Returns:
    Raises:
    ------
    YourErrorType
        Explanation of when and why your error is raised and what it means.
    """
    # Validate the path and settings
    if not os.path.isdir(path):
        raise ValueError(
            f"The provided path does not exist or is not a directory: {path}")
    # Settings Verification
    required_keys = [
        'relative_data_folder',
        'filename_regex',
        'MIN_SIZE_BYTES']
    if any(key not in import_settings for key in required_keys):
        raise KeyError(
            f"import_settings must contain the following keys: {required_keys}"
        )
    # Loaded files verification
    if loaded_list is not None:
        if not all(isinstance(item, list) for item in loaded_list):
            raise TypeError(
                f"loaded_list must be a list of lists. It is currently a \
                list of {type(loaded_list[0])}"
            )
        if any(len(item) != 2 for item in loaded_list):
            raise ValueError(
                f"loaded_list must be a list of lists with 2 items. It is \
                currently a list of lists with {len(loaded_list[0])} items"
            )

    # import data based on the settings what type
    file_names, full_paths, file_sizes = loader.get_files_in_folder_with_size(
        path=path,
        subfolder=import_settings['relative_data_folder'],
        filename_regex=import_settings['filename_regex'],
        min_size=import_settings['MIN_SIZE_BYTES'])
    # combined the file names and sizes into a list of lists
    file_info = [[name, size]
                 for name, size in zip(file_names, file_sizes)]
    # check and compare the previous file list with the new list.
    if not loaded_list:
        first_pass = True
    else:
        first_pass = False
        # check if the files are the same
        new_full_paths = []
        new_file_info = []
        for i, comparison_list in enumerate(file_info):
            if comparison_list not in loaded_list:
                # keep the new files
                new_full_paths.append(full_paths[i])
                new_file_info.append(comparison_list)
        full_paths = new_full_paths  # replace to return only new files
        file_info = new_file_info
    return full_paths, first_pass, file_info


def load_files_interface(
        path: str,
        settings: dict,
        stream: Optional[Stream] = None,
) -> Stream:
    """
    Load files into a stream object based on settings.

    Args:
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

    Returns:
    -------
    Stream
        The Stream object updated with the loaded data.
    """
    if stream is None:
        stream = Stream(
            header=[],
            data=np.array([]),
            time=np.array([]),
            files=[])
    # get the files to load
    full_paths, first_pass, file_info = get_new_files(
        path=path,
        import_settings=settings,
        loaded_list=stream.files
    )

    # load the data type
    for file_i, file_path in enumerate(full_paths):
        print('  Loading file:', file_info[file_i][0])

        if settings['data_loading_function'] == 'general_1d_load':
            stream = get_1d_stream(
                file_path=file_path,
                first_pass=first_pass,
                settings=settings,
                stream=stream
            )

        elif settings['data_loading_function'] == 'general_2d_load':
            stream = get_2d_stream(
                file_path=file_path,
                first_pass=first_pass,
                settings=settings,
                stream=stream
            )

        # elif (self.settings[key]['data_loading_function'] ==
        #         'netcdf_load'):
        #     self.initialise_netcdf_stream(key, path, first_pass)
        #     first_pass = False
        else:
            raise ValueError('Data loading function not recognized',
                             settings['data_loading_function'])

        stream.files.append(file_info[file_i])  # add file info as loaded
        first_pass = False  # set first pass to false after first file
    return stream


def load_folders_interface(
        path: str,
        folder_settings: dict,
        lake: Optional[Lake] = None,
) -> Lake:
    """
    Load files into a lake object based on settings.

    Args:
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

    Returns:
    -------
    Lake
        The Lake object updated with the loaded data streams.
    """
    if lake is None:
        lake = Lake()

    # loop through the folders
    for key, file_settings in folder_settings.items():
        stream = lake[key] if key in lake.streams.keys() else None
        print('Folder Settings:', key)
        # call the load files interface
        lake[key] = load_files_interface(
            path=path,
            settings=file_settings,
            stream=stream
        )
    return lake


def get_1d_stream(
    file_path: str,
    settings: dict,
    first_pass: bool = True,
    stream: Optional[Stream] = None,
) -> Stream:
    """
    Loads and formats a 1D data stream from a file and initializes or updates
    a Stream object.

    Args:
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

    Returns:
    -------
    Stream
        The Stream object updated with the loaded data and corresponding time
        information.

    Raises:
    ------
    ValueError
        If `first_pass` is False, indicating data has already been loaded.
    TypeError
        If `settings` is not a dictionary.
    FileNotFoundError
        If the file specified by `file_path` does not exist.
    KeyError
        If any required keys are missing in the `settings` dictionary.
    """
    if stream is None:
        stream = Stream(
            header=[],
            data=np.array([]),
            time=np.array([]),
            files=[])
    # Input validation, should it be abstracted?
    if not isinstance(settings, dict):
        raise TypeError("The setting parameters must be in a dictionary.")

    required_keys = ['data_checks', 'data_column', 'time_column',
                     'time_format', 'delimiter', 'time_shift_seconds',
                     'timezone_identifier', 'data_header', 'header_row']
    if any(key not in settings for key in required_keys):
        missing_key = [key for key in required_keys if key not in settings]
        raise KeyError(
            f"The settings dictionary is missing required keys: {missing_key}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file path specified does not exist: \
                                {file_path}")

    if not isinstance(first_pass, bool):
        raise TypeError("The first_pass parameter must be a boolean.")

    # should should consolidate and abstract this
    data = loader.data_raw_loader(file_path=file_path)

    if 'date_location' in settings:
        date_offset = loader.non_standard_date_location(
            data=data,
            date_location=settings['date_location']
        )
    else:
        date_offset = None

    epoch_time, data = loader.general_data_formatter(
        data=data,
        data_checks=settings['data_checks'],
        data_column=settings['data_column'],
        time_column=settings['time_column'],
        time_format=settings['time_format'],
        delimiter=settings['delimiter'],
        header_row=settings['header_row'],
        date_offset=date_offset,
        seconds_shift=settings['time_shift_seconds'],
        timezone_identifier=settings['timezone_identifier']
    )

    # check data shape
    data = convert.data_shape_check(
        time=epoch_time,
        data=data,
        header=settings['data_header'])
    if first_pass:
        stream.header = settings['data_header']
        stream.data = data
        stream.time = epoch_time
    else:
        stream = merger.stream_add_data(
            stream=stream,
            time_new=epoch_time,
            data_new=data,
            header_check=True,
            header_new=settings['data_header']
        )
    return stream


def get_2d_stream(
    file_path: str,
    settings: dict,
    first_pass: bool = True,
    stream: Optional[Stream] = None,
) -> Stream:
    """
    Initializes a 2D stream using the settings in the DataLake object.

    Args:
    ----------
        key (str): The key of the stream to initialise.
        path (str): The path of the file to load data from.
        first_pass (bool): Whether this is the first time loading data.

    Returns:
    ----------
        None.
    """
    if stream is None:
        stream = Stream(
            header=[],
            data=np.array([]),
            time=np.array([]),
            files=[])
    # Input validation
    if not isinstance(settings, dict):
        raise TypeError("The setting parameters must be in a dictionary.")

    required_keys = ['data_checks', 'time_column',
                     'time_format', 'delimiter', 'time_shift_seconds',
                     'timezone_identifier', 'data_sizer_reader',
                     'header_row']
    if any(key not in settings for key in required_keys):
        missing_key = [key for key in required_keys if key not in settings]
        raise KeyError(
            f"The settings dictionary is missing required keys: {missing_key}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file path specified does not exist: \
                                {file_path}")

    if not isinstance(first_pass, bool):
        raise TypeError("The first_pass parameter must be a boolean.")

    # should should consolidate and abstract this
    data = loader.data_raw_loader(file_path=file_path)
    if 'date_location' in settings:
        date_offset = loader.non_standard_date_location(
            data=data,
            date_location=settings['date_location']
        )
    else:
        date_offset = None

    epoch_time, data, header = loader.sizer_data_formatter(
        data=data,
        data_checks=settings['data_checks'],
        data_sizer_reader=settings['data_sizer_reader'],
        time_column=settings['time_column'],
        time_format=settings['time_format'],
        delimiter=settings['delimiter'],
        header_row=settings['header_row'],
        date_offset=date_offset,
        seconds_shift=settings['time_shift_seconds'],
        timezone_identifier=settings['timezone_identifier']
    )

    # check data shape
    data = convert.data_shape_check(
        time=epoch_time,
        data=data,
        header=header)
    if first_pass:
        stream.header = header
        stream.data = data
        stream.time = epoch_time
    else:
        stream = merger.stream_add_data(
            stream=stream,
            time_new=epoch_time,
            data_new=data,
            header_check=True,
            header_new=header
        )
    return stream


# def initialise_netcdf_stream(
#     self,
#     key: str,
#     path: str,
#     first_pass: bool
# ) -> None:
#     """
#     Initialise a netcdf stream using the settings in the DataLake
#     object. This can load either 1D or 2D data, as specified in the
#     settings.
#     Args:
#     ----------
#         key (str): The key of the stream to initialise.
#         path (str): The path of the file to load data from.
#         first_pass (bool): Whether this is the first time loading data.

#     Returns:
#     ----------
#         None.
#     """
#     # ValueKey error if netcdf_reader not in settings
#     if 'netcdf_reader' not in self.settings[key]:
#         raise ValueError('netcdf_reader not in settings')

#     # Load the data 1d data
#     if 'data_1d' in self.settings[key]['netcdf_reader']:
#         epoch_time, header_1d, data_1d = loader.netcdf_data_1d_load(
#             file_path=path,
#             settings=self.settings[key])

#         if first_pass:  # create the stream
#             self.streams[
#                 self.settings[key]['data_stream_name'][0]
#             ] = stream(
#                 header_list=header_1d,
#                 average_times=[600],
#                 average_base=self.settings[key]['base_interval_sec']
#             )

#         self.streams[
#             self.settings[key]['data_stream_name'][0]
#         ].add_data(
#             time_stream=epoch_time,
#             data_stream=data_1d,
#         )

#     if 'data_2d' in self.settings[key]['netcdf_reader']:
#         epoch_time, header_2d, data_2d = loader.netcdf_data_2d_load(
#             file_path=path,
#             settings=self.settings[key])

#         if first_pass:  # create the stream
#             self.streams[
#                 self.settings[key]['data_stream_name'][1]
#             ] = stream(
#                 header_list=header_2d,
#                 average_times=[600],
#                 average_base=self.settings[key]['base_interval_sec']
#             )

#         self.streams[
#             self.settings[key]['data_stream_name'][1]
#         ].add_data(
#             time_stream=epoch_time,
#             data_stream=data_2d,
#             header_check=True,
#             header=header_2d
#         )
