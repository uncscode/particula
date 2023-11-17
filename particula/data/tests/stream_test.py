"""Test the Stream class."""


import numpy as np
from particula.data.stream import Stream, StreamAveraged


def test_stream_initialization():
    """Test the initialization of the Stream class."""
    # Arrange
    header = ['header1', 'header2']
    data = np.array([1, 2, 3])
    time = np.array([1.0, 2.0, 3.0])
    files = ['file1', 'file2']

    # Act
    # stream = Stream(header=header)
    stream = Stream(header=header, data=data, time=time, files=files)

    # Assert
    assert stream.header == header
    assert np.array_equal(stream.data, data)
    assert np.array_equal(stream.time, time)
    assert stream.files == files


def test_stream_property_datetime64():
    """Test the datetime64 property of the Stream class."""
    # Arrange
    header = ['header1', 'header2']
    data = np.array([1, 2, 3])
    time = np.array([1.0, 2.0, 3.0])
    files = ['file1', 'file2']
    stream = Stream(header=header, data=data, time=time, files=files)

    # Act
    datetime64 = stream.datetime64

    # Assert
    assert np.array_equal(
        datetime64, np.array(['1970-01-01T00:00:01.000000000',
                              '1970-01-01T00:00:02.000000000',
                              '1970-01-01T00:00:03.000000000'],
                             dtype='datetime64[ns]'))


def test_stream_averaged_initialization():
    """Test the initialization of the StreamAveraged class."""
    # Arrange
    header = ['header1', 'header2']
    data = np.array([1, 2, 3])
    time = np.array([1.0, 2.0, 3.0])
    files = ['file1', 'file2']
    average_interval = 1.0
    start_time = 0.0
    stop_time = 2.0
    standard_deviation = np.array([0.1, 0.2, 0.3])

    # Act
    stream_averaged = StreamAveraged(
        header=header,
        data=data,
        time=time,
        files=files,
        average_interval=average_interval,
        start_time=start_time,
        stop_time=stop_time,
        standard_deviation=standard_deviation)

    # Assert
    assert stream_averaged.header == header
    assert np.array_equal(stream_averaged.data, data)
    assert np.array_equal(stream_averaged.time, time)
    assert stream_averaged.files == files
    assert stream_averaged.average_interval == average_interval
    assert stream_averaged.start_time == start_time
    assert stream_averaged.stop_time == stop_time
    assert np.array_equal(stream_averaged.standard_deviation,
                          standard_deviation)
