# %%

# all the imports
import numpy as np
import matplotlib.pyplot as plt

from particula import u
from particula.data import loader_interface, settings_generator
from particula.data.tests.example_data.get_example_data import get_data_folder

from particula.data import stream_stats
from particula.util import convert, time_manage

from particula.dynamics import Solver
from particula import particle

#%%
# set the parent directory of the data folders
path = get_data_folder()
print('Path to data folder:')
print(path.rsplit('particula')[-1])

# load the 1d data
smps_1d_stream_settings = settings_generator.load_settings_for_stream(
    path=path,
    subfolder='chamber_data',
    settings_suffix='_smps_1d',
)
stream_smps_1d = loader_interface.load_files_interface(
    path=path,
    settings=smps_1d_stream_settings
)

# load the 2d data
smps_2d_stream_settings = settings_generator.load_settings_for_stream(
    path=path,
    subfolder='chamber_data',
    settings_suffix='_smps_2d',
)
stream_smps_2d = loader_interface.load_files_interface(
    path=path,
    settings=smps_2d_stream_settings
)

# 1 convert to dn/dlogDp
stream_smps_2d.data = convert.convert_sizer_dn(
    diameter=np.array(stream_smps_2d.header, dtype=float),
    dn_dlogdp=stream_smps_2d.data,
)

#%%

# Dilution correction
dilution_correction = 2

# scale the concentrations
stream_smps_2d.data *= dilution_correction
stream_smps_1d['Total_Conc_(#/cc)'] *= dilution_correction


# select the time window
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

print(f"Length of stream before crop: {len(stream_smps_1d)}")
# remove the bad data
stream_smps_1d = stream_stats.remove_time_window(
    stream=stream_smps_1d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)

stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)

# crop start
experiment_start_epoch = time_manage.time_str_to_epoch(
    time='09-25-2023 15:25:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)

stream_smps_1d = stream_stats.remove_time_window(
    stream=stream_smps_1d,
    epoch_start=stream_smps_1d.time[0],
    epoch_end=experiment_start_epoch,
)
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=stream_smps_2d.time[0],
    epoch_end=experiment_start_epoch,
)

# crop the end
experiment_end_epoch = time_manage.time_str_to_epoch(
    time='09-26-2023 07:00:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)

stream_smps_1d = stream_stats.remove_time_window(
    stream=stream_smps_1d,
    epoch_start=experiment_end_epoch,
    epoch_end=stream_smps_1d.time[-1],
)
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=experiment_end_epoch,
    epoch_end=stream_smps_2d.time[-1],
)

#%%
# plot the 1d data
experiment_time = time_manage.relative_time(
    epoch_array=stream_smps_1d.time,
    units='hours',
)

fig, ax = plt.subplots()
ax.plot(
    experiment_time,
    stream_smps_1d['Total_Conc_(#/cc)'],
    label='Concentration',
    marker='.',
)
plt.xticks(rotation=45)
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel('Particle concentration (#/cc)')
plt.show()

# %%


