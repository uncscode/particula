"""stats module for datacula"""
# linting disabled until reformatting of this file


from typing import Tuple, Optional
import numpy as np


def merge_formatting(
        data_current: np.ndarray,
        header_current: list,
        data_new: np.ndarray,
        header_new: list
) -> Tuple[np.ndarray, list, np.ndarray, list]:
    """
    Formats two data arrays and their headers so that the data new can be
    subsiqently added to data current.

    Args:
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
                        data_current.shape[0],
                        len(header_new_not_listed)
                        ), np.nan)
            ),
            axis=1
        )

    if bool(header_list_not_new):
        header_new.extend(
            header_list_not_new
        )
        data_new = np.concatenate(
            (
                data_new,
                np.full((
                        data_new.shape[0],
                        len(header_list_not_new)
                        ), np.nan)
            ),
            axis=1
        )

    # check that the shapes are the same
    if data_current.shape[1] != data_new.shape[1]:
        raise ValueError(
            'data_current  ',
            data_current.shape,
            ' and data_new ',
            data_new.shape,
            ' are not the same shape, check the data formatting'
        )
    if len(header_current) != len(header_new):
        raise ValueError(
            'header_current ',
            len(header_current),
            ' and header_new ',
            len(header_new),
            ' are not the same shape, check the data formatting'
        )
    # check if the headers are numeric
    if np.all([x[0].isnumeric() for x in header_current]):
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
        data_current = data_current[:, header_stored_indices]
    else:
        # match header_new to header_current and sort the data
        header_new_indices = [
            header_new.index(x) for x in header_current
        ]
        header_new = [header_new[i] for i in header_new_indices]
    data_new = data_new[:, header_new_indices]
    return data_current, header_current, data_new, header_new


def average_to_interval(
    time_raw: np.ndarray,
    data_raw: np.ndarray,
    average_interval: float,
    average_interval_array: np.ndarray,
    average_data: np.ndarray,
    average_data_std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # pylint: disable=too-many-arguments, too-many-branches
    """
    Calculate the average of the data stream over the specified time intervals.

    This function calculates the average of the data stream over a series of
    time intervals specified by `average_interval_array`. The average and
    standard
    deviation of the data are calculated for each interval, and the results
    are returned as two arrays.

    Args:
    ----------
        time_raw (np.ndarray): An array of timestamps, sorted in ascending
            order.
        average_interval (float): The length of each time interval in seconds.
        average_interval_array (np.ndarray): An array of timestamps
            representing
            the start times of each time interval.
        data_raw (np.ndarray): An array of data points corresponding to the
            timestamps in `time_raw`.
        average_data (np.ndarray): An empty array of shape
            (num_channels, num_intervals)that will be filled with the
            average data for each time interval.
        average_data_std (np.ndarray): An empty array of shape
            (num_channels, num_intervals) that will be filled with the standard
            deviation of the data for each time interval.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: A tuple containing the average data
            and the standard deviation of the data, both as arrays of shape
            (num_channels, num_intervals).
    """
    # average the data in the time interval initialization
    # find closet index in time_raw to the start time
    start_index = np.argmin(np.abs(average_interval_array[0] - time_raw))
    # start_index = 0
    stop_index = 0
    interval_look_buffer_multiple = 2
    start_time = time_raw[0]

    # estimating how much of the time_raw we would need to look at for a
    # given average interval.
    if len(time_raw) > 100:
        time_lookup_span = round(
            (
                average_interval
                * interval_look_buffer_multiple
                / np.nanmean(np.diff(time_raw[:100]))
            )
        )
    else:
        time_lookup_span = 100

    # loop through the average time intervals
    for i, time_i in enumerate(average_interval_array, start=1):
        if (stop_index < len(time_raw)) and (start_time < time_i):

            # trying to only look at the time_raw in the average time
            # interval, assumes that the time stream is sorted
            if start_index + time_lookup_span < len(time_raw):
                compare_bool = np.nonzero(
                    time_raw[start_index:start_index + time_lookup_span]
                    >= time_i
                )
                if len(compare_bool[0]) > 0:
                    stop_index = start_index + compare_bool[0][0]
                else:
                    # used it all for this iteration
                    compare_bool = np.nonzero(
                        time_raw[start_index:]
                        >= time_i
                    )
                    stop_index = start_index + compare_bool[0][0]
                    # re-calculate time look up span,
                    # as timesteps have changed
                    if len(time_raw[start_index:]) > 100:
                        time_lookup_span = round(
                            average_interval
                            * interval_look_buffer_multiple
                            / np.nanmean(
                                np.diff(
                                    time_raw[
                                        start_index:start_index + 100
                                    ]
                                )
                            )
                        )
                    else:
                        time_lookup_span = 100
            else:
                compare_bool = np.nonzero(
                    time_raw[
                        start_index:start_index + time_lookup_span
                    ] >= time_i
                )
                if len(compare_bool[0]) > 0:
                    stop_index = start_index + compare_bool[0][0]
                else:
                    stop_index = len(time_raw)

            if start_index < stop_index:
                # average the data in the time interval
                average_data[i - 1, :] = np.nanmean(
                    data_raw[start_index:stop_index, :], axis=0
                )  # the actual averaging of data is here
                average_data_std[i - 1, :] = np.nanstd(
                    data_raw[start_index:stop_index, :], axis=0
                )  # the actual std data is here
            else:
                start_time = time_raw[stop_index]
            start_index = stop_index

    return average_data, average_data_std


def mask_outliers(
        data: np.ndarray,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
        value: Optional[float] = None,
        invert: Optional[bool] = False
) -> np.ndarray:
    """
    Create a boolean mask for outliers in a data array. Outliers are defined as
    values that are either above or below a specified threshold, or that are
    equal to a specified value. Not all parameters need to be specified. If
    `invert` is True, the mask will be inverted. The mask will be True for
    False for outliers and True for non-outliers.

    Args:
    ----------
        data (np.ndarray): The data array to be masked.
        bottom (float): The lower threshold for outliers.
        top (float): The upper threshold for outliers.
        value (float): The value to be masked.
        invert (bool): If True, the mask will be inverted.

    Returns:
    -------
        np.ndarray: A boolean mask for the outliers in the data array. Mask is
            True for non-outliers and False for outliers, and the same shape as
            the data array.
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
