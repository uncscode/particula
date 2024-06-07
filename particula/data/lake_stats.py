"""Functions to operate on stream objects."""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from particula.data.lake import Lake
from particula.data import stream_stats


def average_std(
    lake: Lake,
    average_interval: Union[float, int] = 60,
    new_time_array: Optional[NDArray[np.float_]] = None,
    clone: bool = True,
) -> Lake:
    """Averages the data in each stream within a 'Lake' object.

    If 'clone' is True, a new 'Lake' instance is created and the averaged
    data is stored there. If 'clone' is False, the original 'Lake' instance
    is modified. The averaged output also includes the standard deviation of
    the data.

    Example:
        ```python
        # Example lake with two streams, each containing numerical data
        lake_data = Lake({'stream1': [1, 2, 3], 'stream2': [4, 5, 6]})
        # Average over a 60-second interval without creating a new lake.
        averaged_lake = average_std(lake_data, 60, clone=False)
        print(averaged_lake)
        Lake({'stream1': [2], 'stream2': [5]})
        ```

    Args:
        lake: The lake data structure containing multiple streams.
        average_interval: The interval over which to average the data.
            Default is 60.
        new_time_array: A new array of time points at which to compute the
            averages.
        clone: Indicates whether to modify the original lake or return a new
            one. Default is True.

    Returns:
        Lake: A lake instance with averaged data.
    """
    lake_averaged = Lake() if clone else lake
    for name, stream in lake.items():
        lake_averaged[name] = stream_stats.average_std(
            stream=stream,
            average_interval=average_interval,
            new_time_array=new_time_array,
        )
    return lake_averaged
