"""A module for the Stream and StreamAveraged(Stream) classes."""

from typing import List, Union
import logging
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("particula")


@dataclass
class Stream:
    """Consistent format for storing data.

    Represents a consistent format for storing and managing data streams
    within a list. Similar to pandas but with tighter control over the
    data allowed and expected format.

    Attributes:
        header: Headers of the data stream, each a string.
        data: 2D numpy array where rows are timepoints and columns
            correspond to headers.
        time: 1D numpy array representing the time points of the data stream.
        files: List of filenames that contain the data stream.

    Methods:
        validate_inputs: Validates the types of class inputs.
        __getitem__(index): Returns the data at the specified index.
        __setitem__(index, value): Sets or updates data at the specified index.
        __len__(): Returns the length of the time stream.
        datetime64: Converts time stream to numpy datetime64 array for plots.
        header_dict: Provides a dictionary mapping of header indices to names.
        header_float: Converts header names to a numpy array of floats.
    """

    header: List[str] = field(default_factory=list)
    data: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    files: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate_inputs()

    def validate_inputs(self):
        """Validates that header is a list.

        Raises:
            TypeError: If `header` is not a list.
        """
        if not isinstance(self.header, list):  # type: ignore
            raise TypeError("header must be a list of strings")

    def __getitem__(self, index: Union[int, str]) -> NDArray[np.float64]:
        """Gets data at a specified index or header name.

        Allows indexing of the data stream using an integer index or a string
        corresponding to the header. If a string is used, the header index is
        retrieved and used to return the data array. Only one str
        argument is allowed. A list of int is allowed.

        Args:
            index: The index or name of the data column to
                retrieve.

        Returns:
            np.ndarray: The data array at the specified index.
        """
        if isinstance(index, str):
            index = self.header.index(index)
        return self.data[:, index]

    def __setitem__(self, index: Union[int, str], value: NDArray[np.float64]):
        """Sets or adds data at a specified index.

        If index is a string and not in headers, it is added. This is used
        to add new data columns to the stream.

        Args:
            index: The index or name of the data column to set.
            value: The data to set at the specified index.

        Notes:
            Support setting multiple rows by accepting a list of values.
        """
        if isinstance(index, str):
            if index not in self.header:
                self.header.append(index)  # add new header element
                if value.ndim == 1:
                    zeros_array = np.zeros_like(value) * np.nan
                    zeros_array = zeros_array[:, np.newaxis]  # add dimension
                self.data = np.hstack((self.data, zeros_array))
                # self.data = np.hstack((self.data, value))
                # self.header.append(index)
                # self.data = np.hstack((self.data, np.atleast_2d(value).T))
            index = self.header.index(index)
        self.data[:, index] = value

    def __len__(self) -> int:
        """
        Returns the number of time points in the data stream.

        Returns:
            int: Length of the time stream.
        """
        return len(self.time)

    @property
    def datetime64(self) -> NDArray[np.float64]:
        """Converts the epoch time array to a datetime64 for plotting.

        This method converts the time array to a datetime64 array, which
        can be used for plotting time series data. This generally assumes
        that the time array is in seconds since the epoch.

        Returns:
            np.ndarray: Datetime64 array representing the time stream.
        """
        return np.array(self.time, dtype="datetime64[s]")

    @property
    def header_dict(self) -> dict[int, str]:
        """Provides a dictionary mapping from index to header names.

        Returns:
            dict: Dictionary with indices as keys and header names as values.
        """
        return dict(enumerate(self.header))

    @property
    def header_float(self) -> NDArray[np.float64]:
        """Attempts to convert header names to a float array, where possible.

        Returns:
            np.ndarray: Array of header names converted to floats.
        """
        return np.array(self.header, dtype=float)


@dataclass
class StreamAveraged(Stream):
    """Stream Class with Averaged Data and Standard Deviation.

    Extends the Stream class with functionalities specific to handling
    averaged data streams. Mainly adding standard deviation to the data
    stream.

    Attributes:
        average_interval: The interval in units (e.g., seconds, minutes) over
            which data is averaged.
        start_time: The start time from which data begins to be averaged.
        stop_time: The time at which data ceases to be averaged.
        standard_deviation: A numpy array storing the standard deviation of
            data streams.
    """

    average_interval: float = field(default_factory=lambda: 0.0)
    start_time: float = field(default_factory=lambda: 0.0)
    stop_time: float = field(default_factory=lambda: 0.0)
    standard_deviation: np.ndarray = field(  # type: ignore
        default_factory=lambda: np.array([])
    )

    def __post_init__(self):
        super().__post_init__()  # Fixed to correct method call
        self.validate_averaging_params()

    def validate_averaging_params(self):
        """Ensures that averaging parameters are valid.

        Raises:
            ValueError: If `average_interval` is not a positive number.
            ValueError: If `start_time` or `stop_time` are not numerical or if
                `start_time` is greater than or equal to `stop_time`.
        """
        if (
            not isinstance(self.average_interval, (int, float))  # type: ignore
            or self.average_interval <= 0
        ):
            raise ValueError("average_interval must be a positive number")
        if (
            not isinstance(self.start_time, (int, float))  # type: ignore
            or not isinstance(self.stop_time, (int, float))  # type: ignore
            or self.start_time >= self.stop_time
        ):
            message = (
                "start_time must be less than"
                "stop_time and both must be numerical"
            )
            logger.error(message)
            raise ValueError(message)

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
        if isinstance(index, str):
            if index not in self.header:
                message = f"Header '{index}' not found."
                logger.error(message)
                raise ValueError(message)
            index = self.header.index(index)
        return self.standard_deviation[:, index]  # type: ignore
