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

# 1 convert from dn/dlogDp
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

time_pairs = [5, 30]
start_number = stream_smps_2d.data[time_pairs[0], :]
end_number = stream_smps_2d.data[time_pairs[1], :]
time_span = stream_smps_2d.time[time_pairs[1]] \
    - stream_smps_2d.time[time_pairs[0]]

radius_bins = stream_smps_2d.header_float * 1e-9  # convert to m

# chamber flow rates
chamber_push = 1.2  # L/min
chamber_dillution = 1.2  # L/min

CHAMBER_VOLUME = 908.2  # L

k_rate_chamber_min = chamber_push / CHAMBER_VOLUME
k_rate_chamber_hr = k_rate_chamber_min * 60

#%%
# Define initial simulation parameters
simple_dic_kwargs = {
    "particle_radius": radius_bins,  # Number of bins for size distribution
    "particle_number": start_number,  # Total number of particles
    "particle_density": 1.8e3,  # Density of particles in kg/m^3
    "particle_charge": 0,  # Charge of particles in elementary charges
    "volume": 1e-6,  # Volume occupied by particles in cubic meters (1 cc)
    "dilution_rate_coefficient": k_rate_chamber_hr * u.hour**-1,  # Rate of particle dilution
    "wall_loss_approximation": "rectangle",  # Method for approximating wall loss
    # Dimensions of the chamber in meters
    "chamber_dimension": [0.739, 0.739, 1.663] * u.m,
    "chamber_ktp_value": 2 * u.s**-1,  # Rate of wall eddy diffusivity
}

# Create particle distribution using the defined parameters
particle_dist = particle.Particle(**simple_dic_kwargs)
kernel = particle_dist.coagulation()
# Define the time array for simulation, simulating 1 hour in 100 steps
time_array = np.linspace(0, time_span, 10)

# Define additional parameters for dynamics simulation
rates_kwargs = {
    "particle": particle_dist,  # pass it the particle distribution
}

# Initialize and solve the dynamics with the specified conditions
solution_coag = Solver(
    time_span=time_array,  # Time over which to solve the dynamics
    do_coagulation=True,  # Enable coagulation process
    do_condensation=False,  # Disable condensation process
    do_nucleation=False,  # Disable nucleation process
    do_dilution=True,  # Disable dilution process
    do_wall_loss=True,  # Disable wall loss process
    **rates_kwargs  # Additional parameters for the solver
).solution(method='odeint')  # Specify the method for solving the ODEs

# Initialize and solve the dynamics with the specified conditions
solution_coagOff = Solver(
    time_span=time_array,  # Time over which to solve the dynamics
    do_coagulation=False,  # Enable coagulation process
    do_condensation=False,  # Disable condensation process
    do_nucleation=False,  # Disable nucleation process
    do_dilution=True,  # Disable dilution process
    do_wall_loss=True,  # Disable wall loss process
    **rates_kwargs  # Additional parameters for the solver
).solution(method='odeint')  # Specify the method for solving the ODEs



# Plotting the simulation results
# Adjusting the figure size for better clarity
fig, ax = plt.subplots(1, 1, figsize=[8, 6])

# Retrieving the radius and distribution data
radius = particle_dist.particle_radius  # Particle radii in meters
# Initial particle distribution
initial_distribution = particle_dist.particle_distribution().m

# Plotting simulation
ax.semilogx(radius.m, (initial_distribution*radius).m, '-b',
            label='Start Simulation')  # Initial distribution
ax.semilogx(radius.m, (solution_coag.m[-1, :]*radius).m,
            '-r', label='t=End Simulation')  # Final distribution
ax.semilogx(radius.m, (solution_coagOff.m[-1, :]*radius).m,
            '.m', label='t=End Simulation NoCoag')  # Final distribution no coag
ax.semilogx(radius_bins, start_number*1e6, '--k',
            label='Start Experiment')  # Initial measured
ax.semilogx(radius_bins, end_number*1e6, '--g',
            label='t=End Experiment')  # Final measured

# Enhancing the plot with labels, title, grid, and legend
# X-axis label with units
ax.set_xlabel(f"Radius ({(particle_dist.particle_radius*radius).u})")
# Y-axis label with units
ax.set_ylabel(f"Number ({particle_dist.particle_distribution().u})")
ax.set_title("Particle Size Distribution Over Time")  # Title of the plot
ax.grid(True, which="both", linestyle='--', linewidth=0.5,
        alpha=0.7)  # Grid for better readability
ax.legend()  # Legend to identify the lines

fig.tight_layout()  # Adjusting the layout to prevent clipping of labels
plt.show()  # Displaying the plot
# %%
