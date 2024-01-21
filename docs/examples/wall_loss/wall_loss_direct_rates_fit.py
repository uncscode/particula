# %% # Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from datetime import datetime

from particula import u
from particula.data import loader_interface, settings_generator
from particula.data.tests.example_data.get_example_data import get_data_folder

from particula.data import stream_stats
from particula.util import convert, time_manage

from particula.dynamics import Solver
from particula import particle

from particula.util.convert import distribution_convert_pdf_pms


# import data
# Set the parent directory of the data folders
path = get_data_folder()
print('Path to data folder:')
print(path.rsplit('particula')[-1])

# Load the 2D data
smps_2d_stream_settings = settings_generator.load_settings_for_stream(
    path=path,
    subfolder='chamber_data',
    settings_suffix='_smps_2d',
)
stream_smps_2d = loader_interface.load_files_interface(
    path=path,
    settings=smps_2d_stream_settings
)

# Convert from dn/dlogDp
stream_smps_2d.data = convert.convert_sizer_dn(
    diameter=np.array(stream_smps_2d.header, dtype=float),
    dn_dlogdp=stream_smps_2d.data,
)


# Select the time window
bad_window_start_epoch = time_manage.time_str_to_epoch(
    time='09-25-2023 19:00:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)
bad_window_end_epoch = time_manage.time_str_to_epoch(
    time='09-25-2023 19:45:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)

print(f"Length of stream before crop: {len(stream_smps_2d)}")
# Remove the bad data
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)

# Crop start
experiment_start_epoch = time_manage.time_str_to_epoch(
    time='09-25-2023 15:25:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=stream_smps_2d.time[0],
    epoch_end=experiment_start_epoch,
)

# Crop the end
experiment_end_epoch = time_manage.time_str_to_epoch(
    time='09-26-2023 07:00:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=experiment_end_epoch,
    epoch_end=stream_smps_2d.time[-1],
)

# Average to 15 minute intervals
average_interval = 60 * 15  # seconds
stream_averaged_2d = stream_stats.average_std(
    stream=stream_smps_2d,
    average_interval=average_interval,
)

# get the time in hours
experiment_time = time_manage.relative_time(
    epoch_array=stream_averaged_2d.time.copy(),
    units='hours',
)

# %% plot
# plot the 1d data
fig, ax = plt.subplots()
ax.plot(
    experiment_time,
    stream_averaged_2d.data[:, 100],
    label='Concentration',
    marker='.',
)
plt.xticks(rotation=45)
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel('Particle concentration (#/cc)')
plt.show()

# %% known rates
# Chamber flow rates
chamber_push = 1.27  # L/min
chamber_dillution = 1.27  # L/min

# Dilution correction
dilution_correction = (chamber_push + chamber_dillution) / chamber_push
# Scale the concentrations for dilution
stream_smps_2d.data *= dilution_correction

# Convert diameter to radius in meters and adjust concentration units
radius_bins = stream_smps_2d.header_float / 2 * 1e-9  # convert to m and radius
stream_smps_2d.data *= 1e6  # convert to number per m3

# Chamber volume and rate constants
CHAMBER_VOLUME = 908.2  # L
k_rate_chamber_min = chamber_push / CHAMBER_VOLUME
k_rate_chamber_hr = k_rate_chamber_min * 60

# %%