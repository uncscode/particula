"""
Merge or adds processed data to the data stream. Accounts for data shape mis
matches and duplicate timestamps. If the data is a different shape than the
data stream, it will interpolate the data to the data stream's time array.
If the data has duplicate timestamps, it will remove the duplicates and
interpolate the data to the data stream's time array.
"""

from typing import Tuple, Optional
import numpy as np
from particula.util import convert, stats
from particula.data.stream import Stream


def combine_data(
    data: np.ndarray,
    time: np.ndarray,
    header_list: list,
    data_new: np.ndarray,
    time_new: np.ndarray,
    header_new: list,
) -> Tuple[np.ndarray, list]:
    # pylint: disable=too-many-arguments
    """"
    Merge or adds processed data together. Accounts for data shape
    miss matches and duplicate timestamps. If the data is a different shape
    than
    the existing data, it will be reshaped to match the existing data.

    Args:
    -----------
    data : np.array
        Existing data stream.
    time : np.array
        Time array for the existing data.
    header_list : List[str]
        List of headers for the existing data.
    data_new : np.array
        Processed data to add to the data stream.
    time_new : np.array
        Time array for the new data.
    header_new : List[str]
        List of headers for the new data.

    Returns:
    --------
    Tuple[np.array, List[str], Dict[str, int]]
        A tuple containing the updated data stream, the updated header list,
        and
        a dictionary mapping the header names to their corresponding indices in
        the data stream.
    """

    data_new = convert.data_shape_check(
        time=time_new,
        data=data_new,
        header=header_new)

    # Check if time_new matches the dimensions of data_new
    if np.array_equal(time, time_new):
        # no need to interpolate the data_new before adding
        # it to the data
        header_updated = np.append(header_list, header_new)
        data_updated = np.concatenate(
            (
                data,
                data_new,
            ),
            axis=1,
        )
    else:  # interpolate the data_new before adding it to the data_stream
        data_interp = np.empty((len(time), data_new.shape[1]))
        for i in range(data_new.shape[1]):
            mask = ~np.isnan(data_new[:, i])
            if not mask.any():
                data_interp[:, i] = np.nan
            else:
                left_value = data_new[mask, i][0]
                right_value = data_new[mask, i][-1]
                data_interp[:, i] = np.interp(
                    time,
                    time_new[mask],
                    data_new[mask, i],
                    left=left_value,
                    right=right_value,
                )
        # update the data array
        data_updated = np.concatenate(
            (
                data,
                data_interp,
            ),
            axis=1,
        )

    header_updated = list(np.append(header_list, header_new))
    return data_updated, header_updated


def stream_add_data(
    stream: Stream,
    time_new: np.ndarray,
    data_new: np.ndarray,
    header_check: Optional[bool] = False,
    header_new: Optional[list] = None
) -> Stream:
    """
    Adds a new data stream and corresponding time stream to the
    existing data.

    Parameters
    ----------
    stream : object
        A Stream object, containing the existing data.
    new_time : np.ndarray (m,)
        An array of time values for the new data stream.
    new_data : np.ndarray
        An array of data values for the new data stream.
    header_check : bool, optional
        If True, checks whether the header in the new data matches the
        header in the existing data. Defaults to False.
    new_header : list of str, optional
        A list of header names for the new data stream. Required if
        header_check is True.

    Returns
    -------
    stream : object
        A Stream object, containing the updated data.

    Raises
    ------
    ValueError
        If header_check is True and header is not provided or
        header does not match the existing header.

    Notes
    -----

    If header_check is True, the method checks whether the header in the
    new data matches the header in the existing data. If they do not match,
    the method attempts to merge the headers and updates the header
    dictionary.

    If header_check is False or the headers match, the new data is
    appended to the existing data.

    The function also checks whether the time stream is increasing, and if
    not, sorts the time stream and corresponding data.
    """

    if stream.data.size == 0:
        stream.data = data_new
        stream.time = time_new
    elif header_check:
        header_new = stream.header if header_new is None else header_new

        stream.data, stream.header, data_new, header_new = \
            stats.merge_formatting(
                data_current=stream.data,
                header_current=stream.header,
                data_new=data_new,
                header_new=header_new
            )
        # updates stream
        stream.data = np.vstack((stream.data, data_new))
        stream.time = np.concatenate((stream.time, time_new))
    else:
        stream.data = np.vstack((stream.data, data_new))
        stream.time = np.concatenate((stream.time, time_new))

    # check if the time stream added is increasing
    increasing_time = np.all(
        stream.time[1:] >= stream.time[:-1],
        axis=0
    )

    if not increasing_time:
        # sort the time stream
        sorted_time_index = np.argsort(stream.time)
        stream.time = stream.time[sorted_time_index]
        stream.data = stream.data[:, sorted_time_index]
    return stream
