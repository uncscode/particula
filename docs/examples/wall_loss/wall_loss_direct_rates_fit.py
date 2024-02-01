# %% # Import necessary libraries
from scipy.signal import savgol_filter, wiener
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit

from particula import u
from particula.data import loader_interface, settings_generator
from particula.data.tests.example_data.get_example_data import get_data_folder

from particula.data import stream_stats
from particula.data.process import size_distribution
from particula.util import convert, time_manage


from particula.rates import Rates
from particula import particle

from particula.util.convert import distribution_convert_pdf_pms




# %% calculate the slop of dn/dt for each radius


def sizer_smooth_and_slope(
    data: np.ndarray,
    time: np.ndarray,
    smoothing_factor: float = 1e4,
    k_degree: int = 5,
    spline_kwargs: dict = {},
    sg_window_length: int = 11,
    sg_poly_order: int = 3,
    sg_radius_window_length: int = 23,
    sg_radius_poly_order: int = 3,
) -> tuple:
    """
    Apply a Savitzky-Golay smoothing filter and then a smoothing spline to
    each size bin of the 2D sizer data and calculate the first
    derivative (slope).

    The function iterates over each column (size bin) in the data, applies
    a Savitzky-Golay filter, fits a smoothing spline, and computes the
    first derivative.

    Args:
    -----
        data: 2D array where each column represents sizer data for a specific
            size bin.
        time: 1D array representing time points corresponding to the data.
        smoothing_factor: Positive smoothing factor used to control the
            trade-off between fidelity to the data and roughness of the spline
            fit. The number of knots will be increased until the smoothing
            condition is satisfied. Default is 1e4.
        k_degree: Degree of the smoothing spline. Must be in the
            range 1 <= k <= 5. Default is 5 (quintic spline).
        spline_kwargs: Additional keyword arguments to pass to the 
            UnivariateSpline constructor.
        sg_window_length: The length of the filter window
            (i.e., the number of coefficients).
            `sg_window_length` must be a positive odd integer.
        sg_poly_order: The order of the polynomial used to fit the samples.
            `sg_poly_order` must be less than `sg_window_length`.

    Returns:
    -------
        spline_values: 2D array of the smoothed values.
        spline_derivative: 2D array of the first derivative of the smoothed
            values, same shape as the input data.

    Raises:
    ------
        ValueError: If the input constraints are violated
        (e.g., data and time dimension mismatch, invalid k_degree).
    """

    # Validate inputs
    if not isinstance(data, np.ndarray) or not isinstance(time, np.ndarray):
        raise ValueError("Data and time must be numpy arrays.")

    if data.shape[0] != time.shape[0]:
        raise ValueError("Time array must match the number of rows in data.")

    if not (1 <= k_degree <= 5):
        raise ValueError("k_degree must be between 1 and 5, inclusive.")

    if sg_window_length % 2 == 0 or sg_window_length <= 1:
        raise ValueError("sg_window_length must be a positive odd integer.")

    if not (0 <= sg_poly_order < sg_window_length):
        raise ValueError("sg_poly_order must be less than sg_window_length.")

    # Prepare arrays for results
    spline_values = np.zeros_like(data)
    spline_derivative = np.zeros_like(data)

    # # smooth across the rows first
    # for i in range(data.shape[0]):
    #     # Smooth the data using the Savitzky-Golay filter
    #     data[i, :] = savgol_filter(
    #         data[i, :], sg_radius_window_length, sg_radius_poly_order)

    # Smooth across the rows
    for i in range(data.shape[0]):
        # Apply the Wiener filter
        data[i, :] = wiener(data[i, :], mysize=sg_radius_window_length)

    # Apply Savitzky-Golay filter and fit a spline for each size bin (column
    # in data)
    for i in range(data.shape[1]):
        # # Smooth the data using the Savitzky-Golay filter
        # smoothed_data = savgol_filter(
        #     data[:, i], sg_window_length, sg_poly_order)
        smoothed_data = wiener(data[:, i], mysize=sg_window_length)

        # Fit a spline to the smoothed data
        spline = UnivariateSpline(
            time, smoothed_data,
            s=smoothing_factor, k=k_degree, **spline_kwargs)
        spline_values[:, i] = spline(time)
        spline_derivative[:, i] = spline.derivative()(time)

    return spline_values, spline_derivative


def size_distribution_rates(
    particle_kwargs: dict,
    concentration_pdf: np.ndarray,
    radius_bins: np.ndarray,
):
    """
    Create the object for the aerosol rates at a given size distrubiton.

    This function can calculate the rates for a given process.

    Args:
    -----
        particle_kwargs (dict): Keyword arguments for the Particle object,
            including particle properties and chamber characteristics.
        concentration_m3 (np.ndarray): Concentration of particles per
            cubic meter for each radius bin.
        radius_bins (np.ndarray): Array of radius bins for the particles.

    Returns:
    --------
        rates: rates object

    Note:
        - The function modifies 'particle_kwargs' to include
            particle number and radius.
        - It assumes that the time array is increasing sequence of time points.
    """

    # Set number and radius in particle_kwargs
    particle_kwargs["particle_number"] = concentration_pdf
    particle_kwargs["particle_radius"] = radius_bins
    particle_kwargs["volume"] = 1  # volume of concentration in cubic meters

    # Create particle distribution
    particle_dist = particle.Particle(**particle_kwargs)

    # Initialize the Rates
    particle_rates = Rates(
        particle=particle_dist,
    )
    return particle_rates


def optimize_ktp_value(
    particle_kwargs: dict,
    concentration_pdf: np.ndarray,
    measured_rate: np.ndarray,
    radius_bins: np.ndarray,
    guess_ktp_value: float = 1.0,
    display_fitting: bool = False
) -> dict:
    """
    Perform least squares optimization to find the best ktp value that
    fits the data.
    
    Parameters:
    - particle_kwargs (dict): Keyword arguments for particle properties.
    - concentration_pdf (np.ndarray): Concentration probability density
        function.
    - measured_rate (np.ndarray): Measured rates.
    - radius_bins (np.ndarray): Radius bins used in the calculation.
    - guess_ktp_value (float): Initial guess for the ktp value.
    - display_fitting (bool): Flag to control the display of fitting process.
    
    Returns:
    - dict: Result dictionary containing the optimized ktp value,
        standard error, 95% confidence error, and R-squared value.
    """

    rates_i = size_distribution_rates(
        particle_kwargs=particle_kwargs,
        concentration_pdf=concentration_pdf,
        radius_bins=radius_bins,
    )

    # Fixed rates at this concentration
    coagulation_rate = rates_i.coagulation_rate()
    dilution_rate = rates_i.dilution_rate()

    def residuals(ktp_value):
        """Calculate residuals between predicted and measured rates."""
        rates_i.particle.chamber_ktp_value = ktp_value * u.s**-1
        total_rate = coagulation_rate + dilution_rate + rates_i.wall_loss_rate()
        return (total_rate.m * radius_bins) - measured_rate
    
    # Set up bounds for ktp_value
    bounds = (0.01, 10)

    # Call least_squares optimization
    result = least_squares(
        fun=residuals,
        x0=guess_ktp_value,
        bounds=bounds,
        verbose=2 if display_fitting else 0,
        loss='soft_l1',
        ftol=1e-12,
    )

    # Estimate covariance matrix (J.T @ J is the approximate Hessian)
    J = result.jac
    cov = np.linalg.inv(J.T @ J)
    
    # Standard error (square root of diagonal elements of the covariance matrix)
    standard_error = np.sqrt(np.diag(cov))
    
    # R-squared value
    ss_res = np.sum(result.fun ** 2)
    ss_tot = np.sum((measured_rate - np.mean(measured_rate)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Construct result dictionary
    result_dict = {
        'ktp_value': result.x[0],
        'standard_error': standard_error[0],
        'error_95': 1.96 * standard_error[0],
        'r_squared': r_squared
    }

    return result_dict


def loop_for_ktp_optimization(
    particle_kwargs: dict,
    concentration_pdf: np.ndarray,
    measured_rate: np.ndarray,
    radius_bins: np.ndarray
) -> tuple:
    """
    Perform ktp optimization for each timestep to find the best ktp value.

    This function loops through each timestep, performing optimization to determine the ktp value that best fits the data for that particular timestep.

    Parameters:
    - particle_kwargs (dict): Keyword arguments for particle properties.
    - concentration_pdf (np.ndarray): Concentration probability density function for each timestep.
    - measured_rate (np.ndarray): Measured rates for each timestep.
    - radius_bins (np.ndarray): Radius bins used in the calculation.

    Returns:
    - tuple: Tuple containing arrays of ktp values, r_squared values, 1-sigma errors, and 95% confidence errors for each timestep.
    """

    # Initialize arrays to store fit results
    ktp_values = np.zeros(measured_rate.shape[0])
    r_squared_values = np.zeros_like(ktp_values)
    error_1sigma_values = np.zeros_like(ktp_values)
    error_95_values = np.zeros_like(ktp_values)

    # Loop through each timestep
    for i in range(measured_rate.shape[0]):
        # Perform optimization for the current timestep
        result = optimize_ktp_value(
            particle_kwargs=particle_kwargs,
            concentration_pdf=concentration_pdf[i, :],
            measured_rate=measured_rate[i, :],
            radius_bins=radius_bins,
            display_fitting=False
        )

        # Extract and store the results from optimization
        ktp_values[i] = result['ktp_value']
        r_squared_values[i] = result['r_squared']
        error_1sigma_values[i] = result['standard_error']
        error_95_values[i] = result['error_95']

    return (ktp_values, r_squared_values, error_1sigma_values, error_95_values)


# %% import data
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

# delete the first 10 bins
stream_smps_2d.header = stream_smps_2d.header[15:]
stream_smps_2d.data = stream_smps_2d.data[:, 15:]

# Remove the bad data based on time window, can comment this out
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
stream_smps_2d = stream_stats.remove_time_window(
    stream=stream_smps_2d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)

# Crop start to experiment start
experiment_start_epoch = time_manage.time_str_to_epoch(
    time='09-25-2023 15:25:00',
    time_format='%m-%d-%Y %H:%M:%S',
    timezone_identifier='UTC'
)
stream_smps_2d = stream_stats.remove_time_window(  # remove data
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
stream_smps_2d = stream_stats.remove_time_window(  # remove data
    stream=stream_smps_2d,
    epoch_start=experiment_end_epoch,
    epoch_end=stream_smps_2d.time[-1],
)


# Average to minute intervals
average_interval = 60 * 10  # seconds
stream_averaged_2d = stream_stats.average_std(
    stream=stream_smps_2d,
    average_interval=average_interval,
)

# filter out zeros
stream_averaged_2d = stream_stats.filtering(
    stream=stream_averaged_2d,
    value=0,
    drop=True,
    header=[100]
)

# get the time in hours
experiment_time = time_manage.relative_time(
    epoch_array=stream_averaged_2d.time.copy(),
    units='hours',
)

# %% Correction for dilution and convert units for simulation comparison
##############################################################
# Chamber flow rates
chamber_push = 1.27  # L/min
chamber_dilution = 1.27  # L/min

# Dilution correction
dilution_correction = (chamber_push + chamber_dilution) / chamber_push
# convert to dN from dn/dlogdp
stream_averaged_2d.data = convert.convert_sizer_dn(
    diameter=stream_averaged_2d.header_float,
    dn_dlogdp=stream_averaged_2d.data,
)
# Scale the concentrations for dilution
stream_averaged_2d.data *= dilution_correction

# Convert diameter to radius in meters and adjust concentration units
radius_bins = stream_averaged_2d.header_float / \
    2 * 1e-9  # convert to m and radius

# Chamber volume and rate constants
CHAMBER_VOLUME = 908.2  # L
k_rate_chamber_min = chamber_push / CHAMBER_VOLUME
k_rate_chamber_hr = k_rate_chamber_min * 60

# Convert size distribution to PDF
concentration_pdf = distribution_convert_pdf_pms(
    x_array=radius_bins,
    distribution=stream_averaged_2d.data * 1e6,  # convert to number per m3
    to_pdf=True
) * radius_bins

# calculate slope and smoothed data
time_sec = time_manage.relative_time(
    epoch_array=stream_averaged_2d.time.copy(),
    units='sec',
)
values_smooth, slope_smooth = sizer_smooth_and_slope(
    data=concentration_pdf,
    time=time_sec,
    smoothing_factor=0,
    k_degree=3,
    sg_window_length=7,
    sg_poly_order=3,
    sg_radius_window_length=25,
    sg_radius_poly_order=3,
)




# %% Bulk properties
# convert back to dn/dlogdp for bulk properties
stream_averaged_2d.data = convert.convert_sizer_dn(
    diameter=stream_averaged_2d.header_float,
    dn_dlogdp=stream_averaged_2d.data,
    inverse=True
)
# calcualte bulk propeties
sizer_bulk = size_distribution.sizer_mean_properties(
    stream=stream_averaged_2d,
    density=1.5,  # g/cm3
    diameter_units='nm'
)

# %% calculate the ktp rate vs time

# Set the particle properties and chamber characteristics
particle_kwargs = {
    "particle_density": 1.8e3,  # Density of particles in kg/m^3
    "particle_charge": 0,  # Charge of particles in elementary charges
    "dilution_rate_coefficient": k_rate_chamber_hr * u.hour**-1,
    "wall_loss_approximation": "rectangle",  # Method for wall loss
    # Dimensions of the chamber in meters
    "chamber_dimension": [0.739, 0.739, 1.663] * u.m,
    "chamber_ktp_value": 1 * u.s**-1,  # Rate of wall eddy diffusivity
}

# run the optimization for each time step
(ktp_values, r_squared_values, error_1sigma_values, error_95_values) \
    = loop_for_ktp_optimization(
        particle_kwargs=particle_kwargs,
        concentration_pdf=values_smooth,
        measured_rate=slope_smooth,
        radius_bins=radius_bins
    )


# %% the bulk and fit information

fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), sharex=True)

# Plot 1: Particle concentration vs. Experiment time
ax1.plot(
    experiment_time,
    sizer_bulk['Total_Conc_(#/cc)'],
    label='Concentration',
    marker='.',
    color='b'
)
ax1.set_ylabel('Chamber concentration (#/cc)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')
ax1.grid(True)

# Twin axis for Plot 1: Particle mass vs. Experiment time
ax2 = ax1.twinx()
ax2.set_ylabel('Chamber mass (ug/m3)', color='r')
ax2.plot(
    experiment_time,
    sizer_bulk['Mass_(ug/m3)'],
    label='Mass',
    color='r'
)
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='upper right')

# Plot 2: ktp values vs. Experiment time, colored by R-squared values
scatter = ax3.scatter(
    experiment_time,
    ktp_values,
    c=r_squared_values,
    label='ktp values',
    vmin=0,
    vmax=1,
    cmap='viridis'
)
# Color bar for Plot 2, placed horizontally above ax3
cbar = plt.colorbar(scatter, ax=ax3, orientation='horizontal', pad=0.2)
cbar.set_label('R-squared fit')
ax3.set_xlabel('Experiment time (hours)')
ax3.set_ylabel('Wall loss parameter ktp (1/sec)')
ax3.set_ylim([0, 3])
ax3.grid(True)

# Set x-ticks rotation
plt.xticks(rotation=45)

# Adjust layout
fig.tight_layout()
plt.show()


# %% plot rates at 1 hour mark

time_index = 50

particle_kwargs["chamber_ktp_value"] = ktp_values[time_index] * u.s**-1

rates_i = size_distribution_rates(
    particle_kwargs=particle_kwargs,
    concentration_pdf=values_smooth[time_index, :],
    radius_bins=radius_bins,
)

# Fixed rates at this concentration
coagulation_rate = rates_i.coagulation_rate()
dilution_rate = rates_i.dilution_rate()
wall_loss_rate = rates_i.wall_loss_rate()
total_rate = coagulation_rate + dilution_rate + wall_loss_rate
x_axis = stream_averaged_2d.header_float

fig, (axtop, ax) = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), sharex=True)

axtop.plot(
    x_axis,
    concentration_pdf[time_index, :],
    label='Distribuiton',
    color='gray'
)
axtop.plot(
    x_axis,
    values_smooth[time_index, :],
    label='Smoothed Distribuiton',
    color='black'
)
axtop.set_ylabel('Concentration (#/m^3)')

ax.plot(
    x_axis,
    slope_smooth[time_index, :],
    label='Measured Rate',
    marker='.',
    color='black'
)
ax.plot(
    x_axis,
    coagulation_rate*radius_bins,
    label='Coagulation Rate',
)
ax.plot(
    x_axis,
    dilution_rate*radius_bins,
    label='Dilution Rate',
)
ax.plot(
    x_axis,
    wall_loss_rate*radius_bins,
    label='Wall loss Rate',
)
ax.plot(
    x_axis,
    total_rate*radius_bins,
    label='total Rate',
    linewidth=3,
    color='red'
)

ax.set_xlabel('Particle Diameter (nm)')
ax.set_ylabel('Rates dN/dt (#/m^3 sec)')
ax.legend()
# ax.set_yscale('log')
ax.set_xscale('log')

# %%




# #%%
# fig, ax1 = plt.subplots()

# index = 100
# data = concentration_pdf[:, index]
# # Plot the data and the spline on the primary y-axis
# ax1.plot(time, data, 'g-', label='Data')
# ax1.plot(time, values_smooth[:, index], 'b.', label='Spline')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Values and Spline', color='b')
# ax1.tick_params('y', colors='b')
# ax1.legend(loc='upper left')

# # Create a second y-axis
# ax2 = ax1.twinx()

# # Plot the slope on the secondary y-axis
# ax2.plot(time, slope_smooth[:, index], 'r-', label='Slope')
# ax2.set_ylabel('Slope', color='r')
# ax2.tick_params('y', colors='r')
# ax2.legend(loc='upper right')
# # %%

# %%
