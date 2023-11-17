"""test the initialization of the Lake class."""


import numpy as np
from particula.data.lake import Lake
from particula.data.stream import Stream


def test_lake_initialization():
    """Test the initialization of the Lake class."""

    header = ['header11', 'header12']
    data = np.array([1, 2, 3])
    time = np.array([1.0, 2.0, 3.0])
    files = ['file1', 'file2']
    stream1 = Stream(header=header, data=data, time=time, files=files)

    header = ['header21', 'header22']
    data = np.array([1, 2, 3])
    time = np.array([1.0, 2.0, 3.0])
    files = ['file1', 'file2']
    stream2 = Stream(header=header, data=data, time=time, files=files)

    lake = Lake()
    lake.add_stream(stream1, name='BlueRiver')
    lake.add_stream(stream2, name='GreenRiver')

    assert lake.BlueRiver.header == stream1.header
    assert lake.GreenRiver.header == stream2.header
