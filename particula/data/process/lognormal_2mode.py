"""Fit the lognormal 2-mode distribution to the concentration PDF."""

import logging
import warnings
from typing import Tuple, Optional, List, Any
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from particula.next.particles.properties import (
    lognormal_pdf_distribution,
    lognormal_pmf_distribution,
)
from particula.data.stream import Stream
from particula.data.process.ml_analysis import generate_and_train_2mode_sizer

# Set up logging
logger = logging.getLogger("particula")

# Suppress all warnings
warnings.filterwarnings("ignore")


def cost_function(
    params: NDArray[np.float64],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> float:
    """
    Cost function for the lognormal distribution with 2 modes.

    Arguments:
        params: Combined array of mode_values, geometric_standard_deviation,
            and number_of_particles.
        x_values: The x-values (particle sizes).
        concentration_pdf: The actual concentration PDF to fit.

    Returns:
        The mean squared error between the actual and guessed concentration
            PDF.
    """
    # Unpack the parameters
    num_modes = 2
    mode_values = params[:num_modes]
    geometric_standard_deviation = params[num_modes: (2 * num_modes)]
    number_of_particles = params[(2 * num_modes):]

    # Generate the guessed concentration PDF
    concentration_pdf_guess = lognormal_pdf_distribution(
        x_values=x_values,
        mode=mode_values,
        geometric_standard_deviation=geometric_standard_deviation,
        number_of_particles=number_of_particles,
    )
    if np.any(np.isnan(concentration_pdf_guess)):
        print("Nan in concentration_pdf_guess")
        return 1e10
    # The mean squared error
    number_dist_error = mean_squared_error(  # type: ignore
        concentration_pdf, concentration_pdf_guess
    )
    # The volume distribution error
    total_number_dist_error = (
        np.trapz(np.abs(concentration_pdf - concentration_pdf_guess), x_values)
        ** 2
    )

    return float(number_dist_error + total_number_dist_error)


def get_bounds() -> List[Tuple[float, Any]]:
    """Provide default bounds for optimization parameters."""
    return [
        (1e-10, None),  # Mode values bounds
        (1e-10, None),
        (1.0001, None),  # Geometric standard deviation bounds
        (1.0001, None),
        (0, None),  # Number of particles bounds
        (0, None),
    ]


def get_initial_guesses(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine initial guesses into a single array."""
    return np.hstack(
        [
            mode_guess,
            geometric_standard_deviation_guess,
            number_of_particles_in_mode_guess,
        ]
    )


def run_optimization(
    method: str,
    initial_guess: NDArray[np.float64],
    bounds: List[Tuple[float, Any]],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Optional[dict[str, Any]]:
    """Perform the optimization using the specified method."""
    try:
        result = minimize(  # type: ignore
            fun=cost_function,
            x0=initial_guess,
            args=(x_values, concentration_pdf),
            method=method,
            bounds=bounds,
        )
        result["method"] = method
        return result  # type: ignore
    except (
        OverflowError,
        FloatingPointError,
        ValueError,
        RuntimeWarning,
        UserWarning,
    ) as e:
        logger.warning(
            "Method %s failed with %s: %s", method, type(e).__name__, e
        )
    return None


def evaluate_fit(
    best_result: dict[str, Any],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float
]:
    """Evaluate the best fit and calculate R² score."""
    optimized_params = np.array(best_result["x"], dtype=np.float64)
    optimized_mode_values = optimized_params[:2]
    optimized_gsd = optimized_params[2:4]
    optimized_number_of_particles = optimized_params[4:]

    concentration_pdf_optimized = lognormal_pdf_distribution(
        x_values=logspace_x,
        mode=optimized_mode_values,
        geometric_standard_deviation=optimized_gsd,
        number_of_particles=optimized_number_of_particles,
    )
    r2 = float(
        r2_score(  # type: ignore
            concentration_pdf, concentration_pdf_optimized
        )
    )

    return (
        optimized_mode_values,
        optimized_gsd,
        optimized_number_of_particles,
        r2,
    )


# pylint: disable=too-many-positional-arguments, too-many-arguments
def optimize_fit(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
    bounds: Optional[List[Tuple[float, Any]]] = None,
    list_of_methods: Optional[List[str]] = None,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    dict[str, Any],
]:
    """
    Optimize the lognormal 2-mode distribution parameters using multiple
    optimization methods.
    """
    if bounds is None:
        bounds = get_bounds()

    if list_of_methods is None:
        list_of_methods = [
            "Nelder-Mead",
            "Powell",
            "L-BFGS-B",
            "TNC",
            "SLSQP",
            "trust-constr",
        ]

    initial_guess = get_initial_guesses(
        mode_guess,
        geometric_standard_deviation_guess,
        number_of_particles_in_mode_guess,
    )

    best_result = {"fun": None}

    for method in list_of_methods:
        result = run_optimization(
            method, initial_guess, bounds, logspace_x, concentration_pdf
        )
        if result and (
            best_result["fun"] is None or result["fun"] < best_result["fun"]
        ):
            best_result = result

    if not best_result["fun"]:
        logger.error("All optimization methods failed")
        raise ValueError("All optimization methods failed")

    optimized_mode_values, optimized_gsd, optimized_number_of_particles, r2 = (
        evaluate_fit(best_result, logspace_x, concentration_pdf)
    )

    best_result["r2"] = r2  # type: ignore
    return (
        optimized_mode_values,
        optimized_gsd,
        optimized_number_of_particles,
        r2,
        best_result,
    )


def optimize_fit_looped(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
    bounds: Optional[List[Tuple[float, Any]]] = None,
    list_of_methods: Optional[List[str]] = None,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Loop through the concentration PDFs to get the best optimization.

    Arguments:
        mode_guess: Array of mode values.
        geometric_standard_deviation_guess: Array of geometric standard
            deviations.
        number_of_particles_in_mode_guess: Array of number of particles.
        x_values: Array of x-values (particle sizes).
        concentration_pdf: Matrix of concentration PDF values.
        bounds: List of bounds for optimization.
        list_of_methods: List of optimization methods.

    Returns:
        optimized_mode_values: Optimized mode values.
        optimized_gsd: Optimized geometric standard deviations.
        optimized_number_of_particles: Optimized number of particles.
        r2: R² score.
        optimization_results: Dictionary of optimization results.
    """
    n_rows = concentration_pdf.shape[0]
    optimized_mode_values = np.zeros([n_rows, 2], dtype=np.float64)
    optimized_gsd = np.zeros([n_rows, 2], dtype=np.float64)
    optimized_number_of_particles = np.zeros([n_rows, 2], dtype=np.float64)
    r2 = np.zeros(n_rows, dtype=np.float64)

    for row in tqdm(range(n_rows), desc="Lognormal 2-mode", total=n_rows):
        (
            optimized_mode_values[row],
            optimized_gsd[row],
            optimized_number_of_particles[row],
            r2[row],
            _,
        ) = optimize_fit(
            mode_guess=mode_guess[row],
            geometric_standard_deviation_guess=(
                geometric_standard_deviation_guess[row]
            ),
            number_of_particles_in_mode_guess=(
                number_of_particles_in_mode_guess[row]
            ),
            logspace_x=logspace_x,
            concentration_pdf=concentration_pdf[row],
            bounds=bounds,
            list_of_methods=list_of_methods,
        )

    return (
        optimized_mode_values,
        optimized_gsd,
        optimized_number_of_particles,
        r2,
    )


def guess_and_optimize_looped(
    experiment_time: NDArray[np.float64],
    radius_m: NDArray[np.float64],
    concentration_m3_pdf: NDArray[np.float64],
) -> Stream:
    """
    Generate initial guesses using a machine learning model, optimize them,
    and return a Stream object with the results.

    Arguments:
        experiment_time: Array of experiment time points.
        radius_m: Array of particle radii in meters.
        concentration_m3_pdf: 2D array of concentration PDFs for each
            time point.

    Returns:
        fitted_stream: A Stream object containing the initial guesses,
            optimized values, and R² scores.
    """
    # Get the initial guess with the ML model
    (
        mode_values_guess,
        geometric_standard_deviation_guess,
        number_of_particles_guess,
    ) = generate_and_train_2mode_sizer.looped_lognormal_2mode_ml_guess(
        logspace_x=radius_m,
        concentration_pdf=concentration_m3_pdf,
    )

    # Get the optimized values
    (
        mode_values_optimized,
        gsd_optimized,
        number_of_particles_optimized,
        r2_optimized,
    ) = optimize_fit_looped(
        mode_guess=mode_values_guess,
        geometric_standard_deviation_guess=geometric_standard_deviation_guess,
        number_of_particles_in_mode_guess=number_of_particles_guess,
        logspace_x=radius_m,
        concentration_pdf=concentration_m3_pdf,
    )

    # Create and populate the Stream object
    fitted_stream = Stream()
    fitted_stream.time = experiment_time
    fitted_stream.header = [
        "ML_Mode_1",
        "ML_Mode_2",
        "ML_GSD_1",
        "ML_GSD_2",
        "ML_N_1",
        "ML_N_2",
        "Opt_Mode_1",
        "Opt_Mode_2",
        "Opt_GSD_1",
        "Opt_GSD_2",
        "Opt_N_1",
        "Opt_N_2",
        "R2",
    ]
    fitted_stream.data = np.array(
        [
            mode_values_guess[:, 0],
            mode_values_guess[:, 1],
            geometric_standard_deviation_guess[:, 0],
            geometric_standard_deviation_guess[:, 1],
            number_of_particles_guess[:, 0],
            number_of_particles_guess[:, 1],
            mode_values_optimized[:, 0],
            mode_values_optimized[:, 1],
            gsd_optimized[:, 0],
            gsd_optimized[:, 1],
            number_of_particles_optimized[:, 0],
            number_of_particles_optimized[:, 1],
            r2_optimized,
        ]
    ).T  # Transpose to match the shape expected by the Stream

    return fitted_stream


def create_lognormal_2mode_from_fit(
    parameters_stream: Stream,
    radius_min: float = 1e-9,
    radius_max: float = 1e-6,
    num_radius_bins: int = 250,
) -> Tuple[Stream, NDArray[np.float64]]:
    """
    Create a fitted PMF stream and concentration matrix based on
    optimized parameters.

    Arguments:
        parameters_stream: Stream object containing the optimized parameters.
        radius_min: Log10 of the minimum radius value in meters (default: -9).
        radius_max: Log10 of the maximum radius value in meters (default: -6).
        num_radius_bins: Number of radius bins to create between radius_min
            and radius_max.

    Returns:
        fitted_pmf_stream: A Stream object containing the time and fitted
            concentration PMF data.
        fitted_concentration_pmf: A numpy array with the fitted
            concentration PMF values.
    """
    # Define the radius values
    radius_m_values = np.logspace(
        start=np.log10(radius_min),
        stop=np.log10(radius_max),
        num=num_radius_bins,
        dtype=np.float64,
    )

    # Initialize the concentration matrix
    fitted_concentration_pmf = np.zeros(
        (len(parameters_stream.time), len(radius_m_values))
    )
    mode_1 = parameters_stream["Opt_Mode_1"]  # Opt_Mode_1
    mode_2 = parameters_stream["Opt_Mode_2"]  # Opt_Mode_2
    gsd_1 = parameters_stream["Opt_GSD_1"]  # Opt_GSD_1
    gsd_2 = parameters_stream["Opt_GSD_2"]  # Opt_GSD_2
    n_1 = parameters_stream["Opt_N_1"]  # Opt_N_1
    n_2 = parameters_stream["Opt_N_2"]  # Opt_N_2

    # Calculate the fitted PMF for each set of optimized parameters
    for i, m1 in enumerate(mode_1):
        fitted_concentration_pmf[i] = lognormal_pmf_distribution(
            x_values=radius_m_values,
            mode=np.array([m1, mode_2[i]]),
            geometric_standard_deviation=np.array([gsd_1[i], gsd_2[i]]),
            number_of_particles=np.array([n_1[i], n_2[i]]),
        )

    # Create and populate the Stream object
    fitted_pmf_stream = Stream()
    fitted_pmf_stream.time = parameters_stream.time
    fitted_pmf_stream.header = np.array(radius_m_values, dtype=str).tolist()
    fitted_pmf_stream.data = fitted_concentration_pmf

    return fitted_pmf_stream, fitted_concentration_pmf
