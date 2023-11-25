"""Functions to operate on stream objects."""

from typing import Optional, Union
import copy
import numpy as np

from particula.util import stats
from particula.data.stream import StreamAveraged, Stream


def drop_masked(stream: Stream, mask: np.ndarray) -> Stream:
    """Drop rows where mask is false, and return data stream.

    Parameters
    ----------
    stream : object
        data stream object
    mask : np.ndarray
        mask to apply to data stream

    Returns
    -------
    object
        stream object
    """
    stream.data = stream.data[mask, :]
    stream.time = stream.time[mask]
    return stream


def average_std(
        stream: Stream,
        average_interval: Union[float, int] = 60,
        new_time_array: Optional[np.ndarray] = None,
) -> StreamAveraged:
    """
    Calculate the average and standard deviation of data within a given
    'stream' object over specified intervals.

    This function takes a 'stream' object, which should contain time-series
    data, and computes the average and standard deviation of the data at
    intervals specified by 'average_interval'. If data.time is in seconds
    then the units of the interval are seconds (hour in hours etc). The
    results are returned as a new 'StreamAveraged' object containing the
    processed data.

    Args:
    - stream (object): The input stream object containing 'time' and 'data'
        arrays along with other associated metadata.
    - average_interval (float|int, optional): The time interval over which the
        averaging is to be performed.
    - new_time_array (np.ndarray, optional): An optional array of time points
        at which the average and standard deviation are computed.
        If not provided, a new time array is generated based on the start and
        end times within the 'stream.time' object.

    Returns:
    - StreamAveraged (object): An object of type 'StreamAveraged' containing
        the averaged data, time array, start and stop times, the standard
        deviation of the averaged data, and other metadata from the original
        'stream' object.

    The function checks for an existing 'new_time_array' and generates one if
    needed. It then calculates the average and standard deviation for each
    interval and constructs a 'StreamAveraged' object with the results and
    metadata from the original 'stream' object.
    """
    # check for new time array
    if new_time_array is None:
        # generate new time array from start and end times
        new_time_array = np.arange(
            start=stream.time[0],
            stop=stream.time[-1],
            step=average_interval
        )
    # generate empty arrays for averaged data and std to be filled in
    average = np.zeros([len(new_time_array), len(stream.header)])
    std = np.zeros_like(average)

    # average data
    average, std = stats.average_to_interval(
        time_raw=stream.time,
        data_raw=stream.data,
        average_interval=average_interval,
        average_interval_array=new_time_array,
        average_data=average,
        average_data_std=std,
    )
    # write to new StreamAveraged object and return
    return StreamAveraged(
        header=stream.header,
        data=average,
        time=new_time_array,
        files=stream.files,
        average_interval=average_interval,
        start_time=new_time_array[0],
        stop_time=new_time_array[-1],
        standard_deviation=std,
    )


# pylint: disable=too-many-arguments
def filtering(
    stream: Stream,
    bottom: Optional[float] = None,
    top: Optional[float] = None,
    value: Optional[float] = None,
    invert: Optional[bool] = False,
    clone: Optional[bool] = True,
    replace_with: Optional[Union[float, int]] = None,
    drop: Optional[bool] = True,
) -> Stream:
    """
    Filters the data of the given 'stream' object based on the specified
    bounds or specific value. The filtered data can be either dropped or
    replaced with a specified value.  Note, not all parameters need to be
    specified, but at least one must be provided (top, bottom, value)

    Args:
    - stream (Stream): The input stream object containing 'data' and 'time'
        attributes.
    - bottom (float, optional): The lower bound for filtering data. Defaults
        to None.
    - top (float, optional): The upper bound for filtering data.
        Defaults to None.
    - value (float, optional): Specific value to filter from data.
        Defaults to None.
    - invert (bool): If True, inverts the filter criteria.
        Defaults to False.
    - clone (bool): If True, returns a copy of the 'stream' object, with
        filtered data. If False, modifies the 'stream' object in-place.
        Defaults to True.
    - replace_with (float|int, optional): Value to replace filtered-out data.
        Defaults to None.
    - drop (bool, optional): If True, filtered-out data points are dropped
        from the dataset. Defaults to False.

    Returns:
    - Stream: The 'stream' object with data filtered as specified.

    If 'drop' is True, 'replace_with' is ignored and filtered data points are
    removed from the 'stream' object. Otherwise, filtered data points are
    replaced with 'replace_with' value.

    add specific data row to filter on
    """
    # copy of stream object to avoid modifying original
    if clone:
        stream = copy.copy(stream)
    # Create a mask for the data that should be retained or replaced
    mask = stats.mask_outliers(
        data=stream.data,
        bottom=bottom,
        top=top,
        value=value,
        invert=invert
    )
    if drop and replace_with is None:
        # Apply mask to data and time, dropping filtered values
        # if any rows are then drop that whole column
        mask_sum = np.invert(np.sum(np.invert(mask), axis=1) > 0)
        stream = drop_masked(stream, mask_sum)
    elif replace_with is not None:
        stream.data = np.where(mask, stream.data, replace_with)
        # No need to modify 'stream.time' as it remains consistent with
        # 'stream.data'
    return stream


def remove_time_window(
        stream: Stream,
        epoch_start: Union[float, int],
        epoch_end: Optional[Union[float, int]] = None,
) -> Stream:
    """
    Remove a time window from a stream object.

    Args:
    - stream: The input stream object containing 'data' and 'time'
        attributes.
    - epoch_start: The start time of the time window to be
        removed.
    - epoch_end: The end time of the time window to be
        removed. If not provided, the time window is the closest time point to
        'epoch_start'.

    Returns:
    - Stream: The 'stream' object with the specified time window removed.
    """
    # get index of start time
    index_start = np.argmin(np.abs(stream.time - epoch_start))
    if epoch_end is None:
        # if no end time provided, remove the closest time point
        stream.time = np.delete(stream.time, index_start)
        stream.data = np.delete(stream.data, index_start, axis=0)
        return stream
    # get index of end time
    index_end = np.argmin(np.abs(stream.time - epoch_end)) + 1
    # remove time and data between start and end times
    stream.time = np.delete(stream.time, slice(index_start, index_end))
    stream.data = np.delete(stream.data, slice(index_start, index_end), axis=0)
    return stream
