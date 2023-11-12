"""A module for the Stream and StreamAveraged(Stream) classes."""
# pytype: skip-file

from typing import List
from dataclasses import dataclass, field
import numpy as np
from particula.util import convert


@dataclass
class Stream:
    """A class for consistent data storage and format.

    Attributes:
    ---------
    header : List[str]
        A list of strings representing the header of the data stream.
    data : np.ndarray
        A numpy array representing the data stream.
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
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    files: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate_inputs()

    def validate_inputs(self):
        """
        Validates the inputs for the DataStream object.

        Raises:
            TypeError: If header is not a list.
        """
        if not isinstance(self.header, list):
            raise TypeError("header_list must be a list")

    @property
    def datetime64(self) -> np.ndarray:
        """
        Returns an array of datetime64 objects representing the time stream.
        Useful for plotting, with matplotlib.dates.
        """
        return convert.datetime64_from_epoch_array(self.time)

    @property
    def return_header_dict(self) -> dict:
        """Returns the header as a dictionary with index (0, 1) as the keys
        and the names as values."""
        return dict(enumerate(self.header))


@dataclass
class StreamAveraged(Stream):
    """A subclass of Stream with additional parameters related to averaging.

    Attributes:
        average_window (float): The size of the window used for averaging.
        start_time (float): The start time for averaging.
        stop_time (float): The stop time for averaging.
        standard_deviation (float): The standard deviation of the data.
    """

    average_window: float = field(default_factory=float)
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
        if not isinstance(self.average_window, (int, float)
                          ) or self.average_window <= 0:
            raise ValueError("average_window must be a positive number")
        if not isinstance(self.start_time, (int, float)) or \
                not isinstance(self.stop_time, (int, float)) or \
                self.start_time >= self.stop_time:
            raise ValueError(
                "start_time must be less than stop_time and numerical")
