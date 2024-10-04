"""Time management utilities."""

import copy
from datetime import datetime
import numpy as np
import pytz
from particula.util import input_handling


def time_str_to_epoch(
        time: str,
        time_format: str,
        timezone_identifier: str = 'UTC',
) -> float:
    """Convert to UTC (epoch) timezone from all inputs. Using pytz library,
    which implements the Olson time zone database. tz identifiers are strings
    from the database.
    See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    for a list of time zones.

    Args:
    time : float (single value no arrays)
        Epoch time in seconds.
    time_format : str
        The format of the time string. See
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        for a list of format codes.
    timezone_identifier : str
        The time zone identifier for the current time zone.

    Returns:
    new_time : float
        The float time in the new time zone.

    Example: Date Time Format Codes
    - '2019-01-01 00:00:00' is '%Y-%m-%d %H:%M:%S'
    - '10/01/2019 00:00:00' is '%d/%m/%Y %H:%M:%S'
    - '2019-01-01 00:00:00.000000' is '%Y-%m-%d %H:%M:%S.%f'
    - '5/1/2019 1:00:00 PM' is '%m/%d/%Y %I:%M:%S %p'
    - %Y: Year with century as a decimal number.
    - %m: Month as a zero-padded decimal number.
    - %d: Day of the month as a zero-padded decimal number.
    - %H: Hour (24-hour clock) as a zero-padded decimal number.
    - %M: Minute as a zero-padded decimal number.
    - %S: Second as a zero-padded decimal number.
    - %f: Microsecond as a decimal number, zero-padded on the left.
    - %p: Locales equivalent of either AM or PM.
    # """
    time_zone = pytz.timezone(timezone_identifier)
    time_obj = datetime.strptime(time, time_format)

    time_obj = time_zone.localize(time_obj)

    return time_obj.timestamp()


def datetime64_from_epoch_array(
        epoch_array: np.ndarray,
        delta: int = 0) -> np.ndarray:
    """
    Converts an array of epoch times to a numpy array of datetime64 objects.

    Args:
    -----------
        epoch_array (np.ndarray): Array of epoch times (in seconds since
            the Unix epoch).
        delta (int): An optional offset (in seconds) to add to the epoch times
            before converting to datetime64 objects.

    Returns:
    --------
        np.ndarray: Array of datetime64 objects corresponding to the input
            epoch times.
    """
    assert len(epoch_array) > 0, "Input epoch_array must not be empty."
    # assert np.issubdtype(epoch_array.dtype, np.integer), \
    #     "Input epoch_array must be an array of integers."

    # Convert epoch times to datetime64 objects with an optional offset
    return np.array(
        [np.datetime64(int(epoch + delta), 's') for epoch in epoch_array]
    )


def relative_time(
        epoch_array: np.ndarray,
        units: str = 'hours',
) -> np.ndarray:
    """Cacluates the relative time from the start of the epoch
    array in the specified units.

    Args:
    -epoch_array (np.ndarray): Array of epoch times (in seconds since
        the Unix epoch).
    -units (str): The units of the relative time. Default is hours.

    Returns:
    -np.ndarray: Array of relative times in the specified units.
    """
    epoch_array = copy.copy(epoch_array)
    assert len(epoch_array) > 0, "Input epoch_array must not be empty."

    epoch_array -= epoch_array[0]
    epoch_array *= input_handling.convert_units('seconds', units)
    return epoch_array
