# %% # Import necessary libraries
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, approx_fprime, least_squares
from datetime import datetime

from particula import u
from particula.data import loader_interface, settings_generator
from particula.data.tests.example_data.get_example_data import get_data_folder

from particula.data import stream_stats
from particula.data.process import size_distribution
from particula.util import convert, time_manage


from particula.dynamics import Solver
from particula.rates import Rates
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

# delete the first 10 bins
stream_smps_2d.header = stream_smps_2d.header[15:]
stream_smps_2d.data = stream_smps_2d.data[:, 15:]

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

# calcualte bulk propeties
sizer_bulk = size_distribution.sizer_mean_properties(
    stream=stream_averaged_2d,
    density=1.5,  # g/cm3
    diameter_units='nm'
)

# %% plot
# plot the 1d data
fig, ax = plt.subplots()
ax.plot(
    experiment_time,
    sizer_bulk['Total_Conc_(#/cc)'],
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
chamber_dilution = 1.27  # L/min

# Dilution correction
dilution_correction = (chamber_push + chamber_dilution) / chamber_push
# Scale the concentrations for dilution
stream_averaged_2d.data *= dilution_correction

# Convert diameter to radius in meters and adjust concentration units
radius_bins = stream_averaged_2d.header_float / 2 * 1e-9  # convert to m and radius
stream_averaged_2d.data *= 1e6  # convert to number per m3

# Chamber volume and rate constants
CHAMBER_VOLUME = 908.2  # L
k_rate_chamber_min = chamber_push / CHAMBER_VOLUME
k_rate_chamber_hr = k_rate_chamber_min * 60

# %% calculate the slop of dn/dt for each radius


def sizer_smooth_and_slope(
    data: np.ndarray,
    time: np.ndarray,
    smoothing_factor: float = 1e4,
    k_degree: int = 5,
    spline_kwargs: dict = {},
    sg_window_length: int = 11,
    sg_poly_order: int = 3,
) -> tuple:
    """
    Apply a Savitzky-Golay smoothing filter and then a smoothing spline to each size bin of the 2D sizer data and
    calculate the first derivative (slope).

    The function iterates over each column (size bin) in the data, applies a Savitzky-Golay filter,
    fits a smoothing spline, and computes the first derivative.

    Args:
    -----
        data: 2D array where each column represents sizer data for a specific
            size bin.
        time: 1D array representing time points corresponding to the data.
        smoothing_factor: Positive smoothing factor used to control the
            trade-off between fidelity to the data and roughness of the spline
            fit. The number of knots will be increased until the smoothing
            condition is satisfied. Default is 1e4.
        k_degree: Degree of the smoothing spline. Must be in the range 1 <= k <= 5.
            Default is 5 (quintic spline).
        spline_kwargs: Additional keyword arguments to pass to the UnivariateSpline
            constructor.
        sg_window_length: The length of the filter window (i.e., the number of coefficients).
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

    # Apply Savitzky-Golay filter and fit a spline for each size bin (column
    # in data)
    for i in range(data.shape[1]):
        # Smooth the data using the Savitzky-Golay filter
        smoothed_data = savgol_filter(
            data[:, i], sg_window_length, sg_poly_order)

        # Fit a spline to the smoothed data
        spline = UnivariateSpline(
            time, smoothed_data, s=smoothing_factor, k=k_degree, **spline_kwargs)
        spline_values[:, i] = spline(time)
        spline_derivative[:, i] = spline.derivative()(time)

    return spline_values, spline_derivative




# %% match rates

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


# %%


particle_kwargs = {
    "particle_density": 1.8e3,  # Density of particles in kg/m^3
    "particle_charge": 0,  # Charge of particles in elementary charges
    "dilution_rate_coefficient": k_rate_chamber_hr * u.hour**-1,
    "wall_loss_approximation": "rectangle",  # Method for approximating wall loss
    # Dimensions of the chamber in meters
    "chamber_dimension": [0.739, 0.739, 1.663] * u.m,
    "chamber_ktp_value": 1 * u.s**-1,  # Rate of wall eddy diffusivity
}

# Convert size distribution to PDF
concentration_pdf = distribution_convert_pdf_pms(
    x_array=radius_bins,
    distribution=stream_averaged_2d.data,
    to_pdf=True
) * radius_bins

time = time_manage.relative_time(
    epoch_array=stream_averaged_2d.time.copy(),
    units='sec',
)

values_smooth, slope_smooth = sizer_smooth_and_slope(
    data=concentration_pdf,
    time=time,
    smoothing_factor=0,
    k_degree=5,
    sg_window_length=7,
    sg_poly_order=3,
)

time_index = 20

rates_i = size_distribution_rates(
    particle_kwargs=particle_kwargs,
    concentration_pdf=values_smooth[time_index, :],
    radius_bins=radius_bins,
)

coagulation = rates_i.coagulation_rate()
dilution_rate = rates_i.dilution_rate()
wall_loss = rates_i.wall_loss_rate()
rates_i.particle.chamber_ktp_value = 5 * u.s**-1  # how to vary the ktp value
wall_loss2 = rates_i.wall_loss_rate()
total = coagulation + dilution_rate + wall_loss

# %% objective function


def ktp_rate_objective_function(
        chamber_ktp_value: float,
        rates_i: object,
        coagulation_rate: np.ndarray,
        dilution_rate: np.ndarray,
        measured_rate: np.ndarray,
        radius_bins: np.ndarray,
) -> float:
    """objective funciton to compare simualiton of chamber to fit ktp values
    """

    rates_i.particle.chamber_ktp_value = chamber_ktp_value * u.s**-1

    # calculate total rate
    total_rate = coagulation_rate + dilution_rate + rates_i.wall_loss_rate()
    return np.mean((total_rate.m*radius_bins - measured_rate)**2)


def hessian(f, x, epsilon=1.e-4):
    """Calculate the Hessian matrix with finite differences"""
    return approx_fprime(x, lambda x: approx_fprime(x, f, epsilon), epsilon)


def optimize_ktp_value(
        particle_kwargs: dict,
        concentration_pdf: np.ndarray,
        measured_rate: np.ndarray,
        radius_bins: np.ndarray,
        guess_ktp_value=1,
        display_fitting=False,
) -> float:
    """ optimized for ktp value"""

    rates_i = size_distribution_rates(
        particle_kwargs=particle_kwargs,
        concentration_pdf=concentration_pdf,
        radius_bins=radius_bins,
    )

    # fixed rates at this concentration
    coagulation_rate = rates_i.coagulation_rate()
    dilution_rate = rates_i.dilution_rate()

    # optimize for the wall loss
    bounds = Bounds(lb=0.01, ub=10)
    problem = {
        'fun': lambda x: ktp_rate_objective_function(
            chamber_ktp_value=x,
            rates_i=rates_i,
            coagulation_rate=coagulation_rate,
            dilution_rate=dilution_rate,
            measured_rate=measured_rate,
            radius_bins=radius_bins,),
        'x0': guess_ktp_value,
        'bounds': bounds,
        'tol': 1e-4,
        'options': {'disp': display_fitting, 'maxiter': 50},
    }

    result = minimize(**problem)

    # calculate hessian at optimum
    H = hessian(lambda x: ktp_rate_objective_function(
            chamber_ktp_value=x,
            rates_i=rates_i,
            coagulation_rate=coagulation_rate,
            dilution_rate=dilution_rate,
            measured_rate=measured_rate,
            radius_bins=radius_bins,),
            result.x[0])

    try:
        covariance_matrix = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # Handle non-invertible Hessian
        covariance_matrix = np.full(H.shape, np.inf)

    standard_errors = np.sqrt(np.diag(covariance_matrix))
    result['standard_errors'] = standard_errors
    result['95_errors'] = 1.96 * standard_errors
    result['hessian'] = H
    return result

# %% run example
particle_kwargs = {
    "particle_density": 1.8e3,  # Density of particles in kg/m^3
    "particle_charge": 0,  # Charge of particles in elementary charges
    "dilution_rate_coefficient": k_rate_chamber_hr * u.hour**-1,
    "wall_loss_approximation": "rectangle",  # Method for approximating wall loss
    # Dimensions of the chamber in meters
    "chamber_dimension": [0.739, 0.739, 1.663] * u.m,
    "chamber_ktp_value": 1 * u.s**-1,  # Rate of wall eddy diffusivity
}

# run optimizaiton
time_index=20

result = optimize_ktp_value(
    particle_kwargs=particle_kwargs,
    concentration_pdf=values_smooth[time_index, :],
    measured_rate=slope_smooth[time_index, :],
    radius_bins=radius_bins,
    guess_ktp_value=1,
    display_fitting=True,
)

print(f"Optimized ktp value: {result.x[0]}")
print(f"Fit error: {result.fun}")
print(f"Hessian: {result.hessian}")

# %% least squares fit

def optimize_ktp_value(
        particle_kwargs: dict,
        concentration_pdf: np.ndarray,
        measured_rate: np.ndarray,
        radius_bins: np.ndarray,
        guess_ktp_value=1,
        display_fitting=False,
) -> float:
    """ Optimized for ktp value using least squares. """
    
    rates_i = size_distribution_rates(
        particle_kwargs=particle_kwargs,
        concentration_pdf=concentration_pdf,
        radius_bins=radius_bins,
        )

    # Fixed rates at this concentration
    coagulation_rate = rates_i.coagulation_rate()
    dilution_rate = rates_i.dilution_rate()

    def residuals(ktp_value):
        # Calculate the model rates
        rates_i.particle.chamber_ktp_value = ktp_value * u.s**-1
        # calculate total rate
        total_rate = coagulation_rate + dilution_rate + rates_i.wall_loss_rate()
        return total_rate.m*radius_bins - measured_rate
    
    # Set up bounds for ktp_value
    bounds = (0.01, 10)

    # Call least_squares optimization
    result = least_squares(
        fun=residuals,
        x0=guess_ktp_value,
        bounds=bounds,
        verbose=2 if display_fitting else 0,
        loss='linear',
    )

    # If you want to calculate and return the standard error as well
    # Jacobian matrix at the solution
    J = result.jac
    # Covariance matrix
    cov = np.linalg.inv(J.T @ J)
    # Standard error (square root of diagonal elements of the covariance matrix)
    standard_error = np.sqrt(np.diag(cov))
    result['standard_error'] = standard_error
    result['95_error'] = 1.96 * standard_error
    return result

# Call the function
result = optimize_ktp_value(
    particle_kwargs=particle_kwargs,
    concentration_pdf=values_smooth[time_index, :],
    measured_rate=slope_smooth[time_index, :],
    radius_bins=radius_bins,
    display_fitting=False)

print(f"Optimized ktp value: {result.x[0]}")
print(f"Fit error: {result.standard_error}")
print(f"95% confidence interval: {result['95_error']}")


# %% plot rates

fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    slope_smooth[time_index, :],
    label='Measurement Slopes',
    marker='.',
)
ax.plot(
    radius_bins,
    coagulation*radius_bins,
    label='Coagulation Rate',
)
ax.plot(
    radius_bins,
    dilution_rate*radius_bins,
    label='Dilution Rate',
)
ax.plot(
    radius_bins,
    wall_loss*radius_bins,
    label='Wall loss Rate',
)
ax.plot(
    radius_bins,
    total*radius_bins,
    label='total Rate',
    linestyle='dashed',
    linewidth=3
)

ax.set_xlabel('radius')
ax.set_ylabel('rates dn/dt')
ax.legend()
# ax.set_yscale('log')
ax.set_xscale('log')





#%%
fig, ax1 = plt.subplots()

index = 100
data = concentration_pdf[:, index]
# Plot the data and the spline on the primary y-axis
ax1.plot(time, data, 'g-', label='Data')
ax1.plot(time, values_smooth[:, index], 'b.', label='Spline')
ax1.set_xlabel('Time')
ax1.set_ylabel('Values and Spline', color='b')
ax1.tick_params('y', colors='b')
ax1.legend(loc='upper left')

# Create a second y-axis
ax2 = ax1.twinx()

# Plot the slope on the secondary y-axis
ax2.plot(time, slope_smooth[:, index], 'r-', label='Slope')
ax2.set_ylabel('Slope', color='r')
ax2.tick_params('y', colors='r')
ax2.legend(loc='upper right')
# %%
