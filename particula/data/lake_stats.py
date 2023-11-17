"""Functions to operate on stream objects."""

from typing import Optional, Union
import numpy as np

from particula.data.lake import Lake
from particula.data import stream_stats


def average_std(
        lake: Lake,
        average_interval: Union[float, int] = 60,
        new_time_array: Optional[np.ndarray] = None,
        clone: bool = True,
) -> Lake:
    """"
    Averages the data in a lake over a specified time interval.
    """
    # nice line of code
    lake_averaged = Lake() if clone else lake

    # iterate over streams in lake
    for name, stream in lake.items():
        lake_averaged[name] = stream_stats.average_std(
            stream=stream,
            average_interval=average_interval,
            new_time_array=new_time_array,
        )
    return lake_averaged
