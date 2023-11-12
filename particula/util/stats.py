"""stats module for datacula"""
# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

from typing import Union, Tuple
import numpy as np


def drop_zeros(datastream_object: object, zero_keys: list) -> object:
    """Drop rows where zero keys not zero, and return data stream

    Parameters
    ----------
    datastream_object : object
        data stream object
    zero_keys : list
        list of keys to check for zeros

    Returns
    -------
    object
        data stream object
    """
    # get zero keys
    zeros = np.sum(
            datastream_object.return_data(
                keys=zero_keys,
                raw=True
            ),
            axis=0
        ) == 0
    datastream_object.data_stream = datastream_object.data_stream[:, zeros]
    datastream_object.time_stream = datastream_object.time_stream[zeros]
    datastream_object.reaverage()
    return datastream_object


def merge_formatting(
        data_current: np.array,
        header_current: list,
        data_new: np.array,
        header_new: list
        ) -> Union[np.array, list, np.array, list]:
    """
    Formats two data arrays and their headers so that the data new can be
    subsiqently added to data current.

    Parameters:
        data_current (np.ndarray): First data array to merge.
        header_current (list): Header for the first data array.
        data_new (np.ndarray): Second data array to merge.
        header_new (list): Header for the second data array.

    Returns:
        (np.ndarray, list, np.array, list): A tuple formatted data
        and headers.

    Raises:
        ValueError: If the data arrays are not the same shape.
        ValueError: If the headers are not the same length.
    """
    # elements in header_new that are not in header_current
    header_new_not_listed = [
            x for x in header_new if x not in header_current
        ]
    header_list_not_new = [
            x for x in header_current if x not in header_new
        ]

    # expand the data array to include the new columns if there are any
    if bool(header_new_not_listed):
        header_current.extend(
                header_new_not_listed
            )
        # add other rows to the data array
        data_current = np.concatenate(
                (
                    data_current,
                    np.full((
                        len(header_new_not_listed),
                        data_current.shape[1]
                        ), np.nan)
                ),
                axis=0
            )

    if bool(header_list_not_new):
        header_new.extend(
                    header_list_not_new
                )
        data_new = np.concatenate(
                (
                    data_new,
                    np.full((
                        len(header_list_not_new),
                        data_new.shape[1]
                        ), np.nan)
                ),
                axis=0
            )

    # check that the shapes are the same
    if data_current.shape[0] != data_new.shape[0]:
        raise ValueError(
                'data_current  ',
                data_current .shape,
                ' and data_new ',
                data_new.shape,
                ' are not the same shape, check the data formatting'
            )
    if len(header_current) != len(header_new):
        raise ValueError(
                'header_current ',
                header_current.shape,
                ' and header_new ',
                header_new.shape,
                ' are not the same shape, check the data formatting'
            )

    # check if all the headers are numbers by checking the first element
    list_of_numbers = np.all([x[0].isnumeric() for x in header_current])

    if list_of_numbers:  # make it a sorted list of numbers
        # convert the headers to floats
        header_list_numeric = np.array(header_current).astype(float)
        header_new_numeric = np.array(header_new).astype(float)
        # find the indices of the headers
        header_stored_indices = np.argsort(header_list_numeric)
        header_new_indices = np.argsort(header_new_numeric)
        # sort the header and keep it a list
        header_current = [header_current[i] for i in header_stored_indices]
        header_new = [header_new[i] for i in header_new_indices]

        # sort the data
        data_current = data_current[header_stored_indices, :]
        data_new = data_new[header_new_indices, :]
    else:
        # match header_new to header_current and sort the data
        header_new_indices = [
                header_new.index(x) for x in header_current
            ]
        header_new = [header_new[i] for i in header_new_indices]
        data_new = data_new[header_new_indices, :]
    return data_current, header_current, data_new, header_new


def average_to_interval(
            time_stream: np.ndarray,
            average_base_sec: float,
            average_base_time: np.ndarray,
            data_stream: np.ndarray,
            average_base_data: np.ndarray,
            average_base_data_std: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the average of the data stream over the specified time intervals.

    This function calculates the average of the data stream over a series of
    time intervals specified by `average_base_time`. The average and standard
    deviation of the data are calculated for each interval, and the results
    are returned as two arrays.

    Parameters:
    ----------
        time_stream (np.ndarray): An array of timestamps, sorted in ascending
            order.
        average_base_sec (float): The length of each time interval in seconds.
        average_base_time (np.ndarray): An array of timestamps representing
            the start times of each time interval.
        data_stream (np.ndarray): An array of data points corresponding to the
            timestamps in `time_stream`.
        average_base_data (np.ndarray): An empty array of shape
            (num_channels, num_intervals)that will be filled with the
            average data for each time interval.
        average_base_data_std (np.ndarray): An empty array of shape
            (num_channels, num_intervals) that will be filled with the standard
            deviation of the data for each time interval.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: A tuple containing the average data
            and the standard deviation of the data, both as arrays of shape
            (num_channels, num_intervals).

    TODO: add custom average starting interval
    """

    # average the data in the time interval initialization
    # find closet index in time_stream to the start time
    start_index = np.argmin(np.abs(average_base_time[0] - time_stream))
    # start_index = 0
    stop_index = 0
    interval_look_buffer_multiple = 2
    start_time = time_stream[0]

    # estimating how much of the time_stream we would need to look at for a
    # given average interval.
    if len(time_stream) > 100:
        time_lookup_span = round(
                average_base_sec
                * interval_look_buffer_multiple
                / np.nanmean(np.diff(time_stream[0:100]))
            )
    else:
        time_lookup_span = 100

    # loop through the average time intervals
    for i, time_i in enumerate(average_base_time, start=1):
        if (stop_index < len(time_stream)) and (start_time < time_i):

            # trying to only look at the time_stream in the average time
            # interval, assumes that the time stream is sorted
            if start_index+time_lookup_span < len(time_stream):
                compare_bool = np.nonzero(
                        time_stream[start_index:start_index+time_lookup_span]
                        >= time_i
                    )
                if len(compare_bool[0]) > 0:
                    stop_index = start_index + compare_bool[0][0]
                else:
                    # used it all for this iteration
                    compare_bool = np.nonzero(
                            time_stream[start_index:]
                            >= time_i
                        )
                    stop_index = start_index+compare_bool[0][0]
                    # re-calculate time look up span,
                    # as timesteps have changed
                    if len(time_stream[start_index:]) > 100:
                        time_lookup_span = round(
                                average_base_sec
                                * interval_look_buffer_multiple
                                / np.nanmean(
                                    np.diff(
                                        time_stream[
                                            start_index:start_index+100
                                        ]
                                    )
                                )
                            )
                    else:
                        time_lookup_span = 100
            else:
                compare_bool = np.nonzero(
                            time_stream[
                                    start_index:start_index + time_lookup_span
                                ] >= time_i
                        )
                if len(compare_bool[0]) > 0:
                    stop_index = start_index + compare_bool[0][0]
                else:
                    stop_index = len(time_stream)

            if start_index < stop_index:
                # average the data in the time interval
                average_base_data[:, i-1] = np.nanmean(
                    data_stream[:, start_index:stop_index], axis=1
                    )  # the actual averaging of data is here
                average_base_data_std[:, i-1] = np.nanstd(
                    data_stream[:, start_index:stop_index], axis=1
                    )  # the actual std data is here
            else:
                start_time = time_stream[stop_index]
            start_index = stop_index

    return average_base_data, average_base_data_std


def mask_outliers(
        data: np.ndarray,
        bottom: float=None,
        top: float=None,
        value: float=None,
        invert: bool=False
        ) -> np.ndarray:
    """
    Creat a boolean mask for outliers in a data array. Outliers are defined as
    values that are either above or below a specified threshold, or that are
    equal to a specified value. Not all parameters need to be specified. If 
    `invert` is True, the mask will be inverted. The mask will be True for
    False for outliers and True for non-outliers.

    Parameters:
    ----------
        data (np.ndarray): The data array to be masked.
        bottom (float): The lower threshold for outliers.
        top (float): The upper threshold for outliers.
        value (float): The value to be masked.
        invert (bool): If True, the mask will be inverted.

    Returns:
    -------
        np.ndarray: A boolean mask for the outliers in the data array.
    """

    # initialize the mask
    mask = np.zeros(data.shape, dtype=bool)

    # mask values below the bottom threshold
    if bottom is not None:
        mask = np.logical_or(mask, data < bottom)

    # mask values above the top threshold
    if top is not None:
        mask = np.logical_or(mask, data > top)

    # mask values equal to the specified value
    if value is not None:
        mask = np.logical_or(mask, data == value)

    # if you want true = outliers
    if invert:
        return mask

    # invert the mask to get true= non-outliers and false=outliers
    mask = np.logical_not(mask)
    return mask


def drop_mask(datastream_object: object, mask: np.ndarray) -> object:
    """Drop rows where mask is false, and return data stream.

    Parameters
    ----------
    datastream_object : object
        data stream object
    mask : np.ndarray
        mask to apply to data stream

    Returns
    -------
    object
        data stream object
    """
    datastream_object.data_stream = datastream_object.data_stream[:, mask[0, :]]
    datastream_object.time_stream = datastream_object.time_stream[mask[0, :]]
    datastream_object.reaverage()
    return datastream_object