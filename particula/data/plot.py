# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

import numpy as np


def timeseries(
            plot_ax,
            datalake,
            datastream_key,
            data_key,
            label,
            color=None,
            line_kwargs=None,
            shade_kwargs=None,
            shade=True,
            raw=False,
):  # sourcery skip
    """
    Plot a datastream from the datalake.

    Parameters
    ----------
    plot_ax : matplotlib.axes._subplots.AxesSubplot
        The axis to plot on.
    datalake : DataLake
        The datalake to plot from.
    datastream_key : str
        The key of the datastream to plot.
    data_key : str
        The key of the data to plot.
    label : str
        The label for the plot.
    color : str, optional
        The color of the plot, by default None
    line_kwargs : dict
        The keyword arguments for the line plot.
    shade_kwargs : dict
        The keyword arguments for the shaded area plot.
    """
    if line_kwargs is None:
        line_kwargs = {}
    if shade_kwargs is None:
        shade_kwargs = {'alpha': 0.25}
    if color is None:
        color = plot_ax._get_lines.get_next_color()
    
    time = datalake.datastreams[datastream_key].return_time(
        datetime64=True,
        raw=raw)
    data = datalake.datastreams[datastream_key].return_data(
        keys=[data_key],
        raw=raw)[0]
    
    if shade and not raw:
        std = datalake.datastreams[datastream_key].return_std(keys=[data_key])[0]
        # plot shaded area
        plot_ax.fill_between(
            time,
            data *(1-std/data),
            data *(1+std/data),
            color=color,
            **shade_kwargs
        )
    # plot line
    plot_ax.plot(
        time,
        data,
        color=color,
        label=label,
        **line_kwargs
    )


def histogram(
            plot_ax,
            datalake,
            datastream_key,
            data_key,
            label,
            bins,
            range,
            color=None,
            raw=False,
            kwargs=None,
):  # sourcery skip
    """
    Plot a datastream from the datalake.

    Parameters
    ----------
    plot_ax : matplotlib.axes._subplots.AxesSubplot
        The axis to plot on.
    datalake : DataLake
        The datalake to plot from.
    datastream_key : str
        The key of the datastream to plot.
    data_key : str
        The key of the data to plot.
    label : str
        The label for the plot.
    bins : int
        The number of bins for the histogram.
    range : tuple
        The range of the histogram.
    color : str, optional
        The color of the plot, by default None
    raw : bool, optional
        Whether to use the un-averaged data, by default False
    kwargs : dict
        The keyword arguments for the hist plot.
    
    """
    if kwargs is None:
        kwargs = {}
    if color is None:
        color = plot_ax._get_lines.get_next_color()

    data = datalake.datastreams[datastream_key].return_data(
        keys=[data_key],
        raw=raw)[0]

    plot_ax.hist(
        data,
        bins=bins,
        range=range,
        label=label,
        color=color,
        **kwargs)
