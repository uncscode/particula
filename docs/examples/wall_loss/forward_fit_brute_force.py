# %%
# flake8: noqa

# all the imports
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

# %% convert concentrations and correct for dilution

# chamber flow rates
chamber_push = 1.27  # L/min
chamber_dillution = 1.27  # L/min

# Dilution correction
dilution_correction = (chamber_push + chamber_dillution) / chamber_push
# scale the concentrations for dilution
stream_smps_2d.data *= dilution_correction
stream_smps_1d['Total_Conc_(#/cc)'] *= dilution_correction

radius_bins = stream_smps_2d.header_float/2 * 1e-9  # convert to m and radius
stream_smps_2d.data *= 1e6 # convert to number per m3

CHAMBER_VOLUME = 908.2  # L
k_rate_chamber_min = chamber_push / CHAMBER_VOLUME
k_rate_chamber_hr = k_rate_chamber_min * 60

# %% evaluate distribution

def simulate_chamber_interval(
    particle_kwargs: dict,
    time_array: np.ndarray,
    concentration_m3: np.ndarray,
    radius_bins: np.ndarray,
):
    """
    Simulate the aerosol chamber for a given time interval.

    This function simulates the dynamics of aerosol particles in a chamber
    over a specified time interval. It considers processes like
    coagulation, condensation, nucleation, dilution, and wall loss.

    Args:
    -----
        particle_kwargs (dict): Keyword arguments for the Particle object,
            including particle properties and chamber characteristics.
        time_array (np.ndarray): Array of time points over which to
            simulate the chamber dynamics.
        concentration_m3 (np.ndarray): Concentration of particles per
            cubic meter for each radius bin.
        radius_bins (np.ndarray): Array of radius bins for the particles.

    Returns:
    --------
        tuple: A tuple containing two elements:
            - simulation_pdf (np.ndarray): The particle number density
                distribution over time as a probability density function.
            - concentration_pdf (np.ndarray): The concentration distribution
                over time as a probability density function.

    Note:
        - The function modifies 'particle_kwargs' to include
            particle number and radius.
        - It assumes that the time array is increasing sequence of time points.
    """

    # Convert size distribution to PDF
    concentration_pdf = distribution_convert_pdf_pms(
        x_array=radius_bins,
        distribution=concentration_m3,
        to_pdf=True
    ) * radius_bins

    # Set number and radius in particle_kwargs
    particle_kwargs["particle_number"] = concentration_pdf[0, :]
    particle_kwargs["particle_radius"] = radius_bins
    particle_kwargs["volume"] = 1  # volume of concentration in cubic meters

    # Rebase time to start from zero
    rebased_time = time_array - time_array[0]

    # Create particle distribution
    particle_dist = particle.Particle(**particle_kwargs)

    # Pack the distribution into the solver
    rates_kwargs = {
        "particle": particle_dist,
    }

    # Initialize and solve the dynamics
    solution_coag = Solver(
        time_span=rebased_time,
        do_coagulation=True,
        do_condensation=False,
        do_nucleation=False,
        do_dilution=True,
        do_wall_loss=True,
        **rates_kwargs
    ).solution(method='odeint')

    # Process the solution
    simulation_pdf = solution_coag[:, :] * particle_dist.particle_radius
    concentration_pdf = concentration_pdf * simulation_pdf.u
    return simulation_pdf, concentration_pdf


# calculate the error

def simulation_error(simulation_pdf, concentration_pdf):
    """
    Calculate the Mean Absolute Error (MAE) between simulation PDF and concentration PDF.

    The function computes the MAE as the average of absolute differences between 
    the two provided probability density functions (PDFs). This metric is useful 
    for quantifying the accuracy of a simulation against observed data.

    Args:
        simulation_pdf (np.ndarray): The simulated particle number density distribution as a PDF.
        concentration_pdf (np.ndarray): The observed concentration distribution as a PDF.

    Returns:
        float: The Mean Absolute Error between the two PDFs.
    
    Raises:
        ValueError: If the input arrays do not have the same shape.
    """

    if simulation_pdf.shape != concentration_pdf.shape:
        raise ValueError("The shapes of simulation_pdf and concentration_pdf must be the same.")

    # Calculate the mean absolute error
    mae = np.mean(np.abs(simulation_pdf.m - concentration_pdf.m))
    return mae


def chamber_ktp_objective_funciton(
        chamber_ktp_value: float,
        particle_kwargs: dict,
        time_array: np.ndarray,
        concentration_m3: np.ndarray,
        radius_bins: np.ndarray,
) -> float:
    """objective funciton to compare simualiton of chamber to fit ktp values
    """
    
    particle_kwargs['chamber_ktp_value'] = chamber_ktp_value * u.s**-1

    simulation_pdf, concentration_pdf = simulate_chamber_interval(
        particle_kwargs=particle_kwargs,
        time_array=time_array,
        concentration_m3=concentration_m3,
        radius_bins=radius_bins,
    )

    error_out = simulation_error(simulation_pdf, concentration_pdf)

    return error_out


def optimize_ktp_value(
        particle_kwargs: dict,
        time_array: np.ndarray,
        concentration_m3: np.ndarray,
        radius_bins: np.ndarray,
        guess_ktp_value=1,
        display_fitting=True,
) -> float:
    """ optimized for ktp value"""

    bounds = Bounds(lb=0.01, ub=10)

    problem = {
        'fun': lambda x: chamber_ktp_objective_funciton(
            chamber_ktp_value=x,
            particle_kwargs=particle_kwargs,
            time_array=time_array,
            concentration_m3=concentration_m3,
            radius_bins=radius_bins),
        'x0': guess_ktp_value,
        'bounds': bounds,
        'tol': 1e-4,
        'options': {'disp': display_fitting, 'maxiter' : 10},
    }

    fit_result = minimize(**problem)
    return fit_result

#%% run fit
# inputs
index_span = [5, 15]
concentration_m3= stream_smps_2d.data[index_span[0]:index_span[1], :]
time_array = stream_smps_2d.time[index_span[0]:index_span[1]]

particle_kwargs = {
    "particle_density": 1.8e3,  # Density of particles in kg/m^3
    "particle_charge": 0,  # Charge of particles in elementary charges
    # Rate of particle dilution
    "dilution_rate_coefficient": k_rate_chamber_hr * u.hour**-1,
    "wall_loss_approximation": "rectangle",  # Method for approximating wall loss
    # Dimensions of the chamber in meters
    "chamber_dimension": [0.739, 0.739, 1.663] * u.m,
    "chamber_ktp_value": 0.5 * u.s**-1,  # Rate of wall eddy diffusivity
}

fit_return = optimize_ktp_value(
    particle_kwargs=particle_kwargs,
    time_array=time_array,
    concentration_m3=concentration_m3,
    radius_bins=radius_bins,
)

print(f"Fit ktp value: {fit_return.x[0]}")



particle_kwargs['chamber_ktp_value'] = fit_return.x * u.s**-1
simulation_pdf, concentration_pdf = simulate_chamber_interval(
    particle_kwargs=particle_kwargs,
    time_array=time_array,
    concentration_m3=concentration_m3,
    radius_bins=radius_bins,
)



# %%


# Plotting the simulation results
# Adjusting the figure size for better clarity
fig, ax = plt.subplots(1, 1, figsize=[8, 6])

# Plotting simulation
ax.semilogx(radius_bins, (simulation_pdf.m[0, :]), '-b',
            label='Start Simulation')  # Initial distribution
ax.semilogx(radius_bins, (simulation_pdf.m[-1, :]),
            '-r', label='t=End Simulation')  # Final distribution
ax.semilogx(radius_bins, concentration_pdf[0,: ], '--k',
            label='Start Experiment')  # Initial measured
ax.semilogx(radius_bins, concentration_pdf[-1, :], '--g',
            label='t=End Experiment')  # Final measured

# Enhancing the plot with labels, title, grid, and legend
# X-axis label with units
ax.set_xlabel(f"Radius (meter)")
# Y-axis label with units
ax.set_ylabel(f"Number ({simulation_pdf.u})")
ax.set_title("Particle Size Distribution Over Time")  # Title of the plot
ax.grid(True, which="both", linestyle='--', linewidth=0.5,
        alpha=0.7)  # Grid for better readability
ax.legend()  # Legend to identify the lines
# ax.set_ylim(1e5,1e20)

fig.tight_layout()  # Adjusting the layout to prevent clipping of labels
plt.show()  # Displaying the plot
# %% loop over the full data set

simulation_window = 20
step_size = 10
index_steps = np.arange(0, len(stream_smps_2d)-simulation_window, step_size)

fitted_ktp_values = np.zeros_like(index_steps, dtype=float)
error_mae = np.zeros_like(index_steps, dtype=float)
for i, index_start in enumerate(index_steps):
    print(
        f"Fit Percent: {i/len(index_steps) * 100:.2f}%, "
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )

    index_end = index_start + simulation_window
    concentration_m3 = stream_smps_2d.data[index_start:index_end, :]
    time_array = stream_smps_2d.time[index_start:index_end]

    if i == 0:
        guess_ktp_value = 1
    else:
        guess_ktp_value = fitted_ktp_values[i-1]
    fit_return = optimize_ktp_value(
        particle_kwargs=particle_kwargs,
        time_array=time_array,
        concentration_m3=concentration_m3,
        radius_bins=radius_bins,
        guess_ktp_value=guess_ktp_value * u.s**-1,
        display_fitting=False,
    )
    print(f"Fit ktp value: {fit_return.x[0]}")
    fitted_ktp_values[i] = fit_return.x[0]
    error_mae[i] = fit_return.fun

# %% export the fits to csv using numpy

np.savetxt(
    fname='ktp_fits.csv',
    X=fitted_ktp_values,
    delimiter=',',
    header='ktp_fits',
    comments='',
)


