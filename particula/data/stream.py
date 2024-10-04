"""A module for the Stream and StreamAveraged(Stream) classes."""


from typing import List, Union
from dataclasses import dataclass, field
import numpy as np
from particula.util import time_manage


@dataclass
class Stream:
    """A class for consistent data storage and format.

    Attributes:
    ---------
    header : List[str]
        A list of strings representing the header of the data stream.
    data : np.ndarray
        A numpy array representing the data stream. The first dimension
        represents time and the second dimension represents the header.
    time : np.ndarray
        A numpy array representing the time stream.
    files : List[str]
        A list of strings representing the files containing the data stream.

    Methods:
    -------
    validate_inputs
        Validates the inputs to the Stream class.
    datetime64 -> np.ndarray
        Returns an array of datetime64 objects representing the time stream.
        Useful for plotting, with matplotlib.dates.
    return_header_dict -> dict
        Returns the header as a dictionary with keys as header elements and
        values as their indices.
    """

    # Initialize other fields as empty arrays
    header: List[str] = field(default_factory=list)
<<<<<<< HEAD
    data: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
=======
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
>>>>>>> upstream/main
    files: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate_inputs()

    def validate_inputs(self):
        """
        Validates the inputs for the DataStream object.
        Raises:
            TypeError: If header is not a list.
        # this might be why I can't call Stream without inputs
        """
<<<<<<< HEAD
        if not isinstance(self.header, list):  # type: ignore
            raise TypeError("header must be a list of strings")

    def __getitem__(self, index: Union[int, str]) -> NDArray[np.float64]:
        """Gets data at a specified index or header name.

        Allows indexing of the data stream using an integer index or a string
        corresponding to the header. If a string is used, the header index is
        retrieved and used to return the data array. Only one str
        argument is allowed. A list of int is allowed.
=======
        if not isinstance(self.header, list):
            raise TypeError("header_list must be a list")
>>>>>>> upstream/main

    def __getitem__(self, index: Union[int, str]):
        """Allows for indexing of the data stream.
        Args:
        ----------
        index : int or str
            The index of the data stream to return.
        Returns:
        -------
        np.ndarray
            The data stream at the specified index."""
        if isinstance(index, str):
            index = self.header.index(index)
        return self.data[:, index]

<<<<<<< HEAD
    def __setitem__(self, index: Union[int, str], value: NDArray[np.float64]):
        """Sets or adds data at a specified index.

        If index is a string and not in headers, it is added. This is used
        to add new data columns to the stream.

=======
    def __setitem__(self, index: Union[int, str], value):
        """Allows for setting or adding of a row of data in the stream.
>>>>>>> upstream/main
        Args:
        index : The index of the data stream to set.
        value : The data to set at the specified index.

        future work maybe add a list option and iterate through the list"""
        if isinstance(index, str):
            if index not in self.header:
                self.header.append(index)  # add new header element
                self.data = np.hstack((self.data, value))
            index = self.header.index(index)
        # if index is an int, set the data at that index
        self.data[:, index] = value

    def __len__(self):
        """Returns the length of the time stream."""
        return len(self.time)

    def __pop__(self, index: Union[int, str]) -> None:
        """Removes data at a specified index or header name.

        Allows indexing of the data stream using an integer index or a string
        corresponding to the header. If a string is used, the header index is
        retrieved and used to return the data array. Only one str
        argument is allowed. A list of int is allowed.

        Args:
            index: The index or name of the data column to
                retrieve.
        """
        if isinstance(index, str):
            index = self.header.index(index)
        self.data = np.delete(self.data, index, axis=1)
        self.header.pop(index)

    @property
<<<<<<< HEAD
    def datetime64(self) -> NDArray[np.float64]:
        """Converts the epoch time array to a datetime64 for plotting.

        This method converts the time array to a datetime64 array, which
        can be used for plotting time series data. This generally assumes
        that the time array is in seconds since the epoch.

        Returns:
            np.ndarray: Datetime64 array representing the time stream.
=======
    def datetime64(self) -> np.ndarray:
>>>>>>> upstream/main
        """
        Returns an array of datetime64 objects representing the time stream.
        Useful for plotting, with matplotlib.dates.
        """
        return time_manage.datetime64_from_epoch_array(self.time)

    @property
    def header_dict(self) -> dict:
        """Returns the header as a dictionary with index (0, 1) as the keys
        and the names as values."""
        return dict(enumerate(self.header))

    @property
<<<<<<< HEAD
    def header_float(self) -> NDArray[np.float64]:
        """Attempts to convert header names to a float array, where possible.

        Returns:
            np.ndarray: Array of header names converted to floats.
        """
=======
    def header_float(self) -> np.ndarray:
        """Returns the header as a numpy array of floats."""
>>>>>>> upstream/main
        return np.array(self.header, dtype=float)


@dataclass
class StreamAveraged(Stream):
    """A subclass of Stream with additional parameters related to averaging.

    Attributes:
        average_interval (float): The size of the window used for averaging.
        start_time (float): The start time for averaging.
        stop_time (float): The stop time for averaging.
        standard_deviation (float): The standard deviation of the data.
    """

    average_interval: float = field(default_factory=float)
    start_time: float = field(default_factory=float)
    stop_time: float = field(default_factory=float)
    standard_deviation: np.ndarray = field(
        default_factory=lambda: np.array([]))

    def __post_init__(self):
        super().__post_init__()
        self.validate_averaging_params()

    def validate_averaging_params(self):
        """
        Validates the averaging parameters for the stream.

        Raises:
            ValueError: If average_window is not a positive number or if
            start_time and stop_time are not numbers or if start_time is
            greater than or equal to stop_time.
        """
        if not isinstance(self.average_interval, (int, float)
                          ) or self.average_interval <= 0:
            raise ValueError("average_window must be a positive number")
        if not isinstance(self.start_time, (int, float)) or \
                not isinstance(self.stop_time, (int, float)) or \
                self.start_time >= self.stop_time:
            raise ValueError(
                "start_time must be less than stop_time and numerical")

<<<<<<< HEAD
    def get_std(self, index: Union[int, str]) -> NDArray[np.float64]:
        """Retrieves the standard deviation

        In the averaged data stream, the standard deviation of the data is
        stored in a separate array that mirrors the same indices as the data
        stream. This method allows retrieval of the standard deviation at a
        specified index.

        Args:
            index: The index or header name of the data stream
            for which standard deviation is needed.

        Returns:
            np.ndarray: The standard deviation values at the specified index.

        Raises:
            ValueError: If the specified index does not exist in the header.
        """
=======
    def get_std(self, index) -> np.ndarray:
        """Returns the standard deviation of the data."""
>>>>>>> upstream/main
        if isinstance(index, str):
            index = self.header.index(index)
        return self.standard_deviation[:, index]
