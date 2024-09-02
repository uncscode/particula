"""
Functions for fitting the chamber rates to the observed rates.
"""

from typing import Tuple, List, Optional, Callable
import copy
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import r2_score, mean_squared_error  # type: ignore
from tqdm import tqdm

from particula.data.stream import Stream
from particula.next.dynamics import dilution, wall_loss, coagulation


# pylint: disable=too-many-arguments, too-many-locals
def calculate_pmf_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    alpha_collision_efficiency: float = 1,
    volume: float = 1,  # m^3
    input_flow_rate: float = 0.16e-6,  # m^3/s
    wall_eddy_diffusivity: float = 0.1,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),  # m
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate the coagulation, dilution, and wall loss rates,
    and return the net rate.

    Arguments:
        radius_bins: Array of particle radii.
        concentration_pmf: Array of particle concentration
            probability mass function.
        temperature: Temperature in Kelvin.
        pressure: Pressure in Pascals.
        particle_density: Density of the particles in kg/m^3.
        alpha_collision_efficiency: Collision efficiency factor.
        volume: Volume of the chamber in m^3.
        input_flow_rate: Input flow rate in m^3/s.
        wall_eddy_diffusivity: Eddy diffusivity for wall loss in m^2/s.
        chamber_dimensions: Dimensions of the chamber
            (length, width, height) in meters.

    Returns:
        coagulation_loss: Loss rate due to coagulation.
        coagulation_gain: Gain rate due to coagulation.
        dilution_loss: Loss rate due to dilution.
        wall_loss_rate: Loss rate due to wall deposition.
        net_rate: Net rate considering all effects.
    """
    # Mass of the particles in kg
    mass_particle = 4 / 3 * np.pi * radius_bins**3 * particle_density

    # Coagulation kernel
    kernel = coagulation.brownian_coagulation_kernel_via_system_state(
        radius_particle=radius_bins,
        mass_particle=mass_particle,
        temperature=temperature,
        pressure=pressure,
        alpha_collision_efficiency=alpha_collision_efficiency,
    )

    # Coagulation loss and gain
    coagulation_loss = coagulation.discrete_loss(
        concentration=concentration_pmf,
        kernel=kernel,  # type: ignore
    )
    coagulation_gain = coagulation.discrete_gain(
        radius=radius_bins,
        concentration=concentration_pmf,
        kernel=kernel,  # type: ignore
    )
    coagulation_net = coagulation_gain - coagulation_loss

    # Dilution loss rate
    dilution_coefficient = dilution.volume_dilution_coefficient(
        volume=volume, input_flow_rate=input_flow_rate
    )
    dilution_loss = dilution.dilution_rate(
        coefficient=dilution_coefficient,
        concentration=concentration_pmf,
    )

    # Wall loss rate
    wall_loss_rate = wall_loss.rectangle_wall_loss_rate(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=radius_bins,
        particle_density=particle_density,
        particle_concentration=concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=chamber_dimensions,
    )

    # Net rate considering coagulation, dilution, and wall loss
    total_rate = coagulation_net + dilution_loss + wall_loss_rate

    return (  # type: ignore
        coagulation_loss,
        coagulation_gain,
        dilution_loss,
        wall_loss_rate,
        total_rate,
    )


# pylint: disable=too-many-arguments
def coagulation_rates_cost_function(
    parameters: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    time_derivative_concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    volume: float = 1,  # m^3
    input_flow_rate: float = 0.16e-6,  # m^3/s
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),  # m
) -> float:
    """Cost function for the optimization of the eddy diffusivity
    and alpha collision efficiency."""

    # Unpack the parameters
    wall_eddy_diffusivity = parameters[0]
    alpha_collision_efficiency = parameters[1]

    # Calculate the rates
    _, _, _, _, net_rate = calculate_pmf_rates(
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        particle_density=particle_density,
        alpha_collision_efficiency=alpha_collision_efficiency,
        volume=volume,
        input_flow_rate=input_flow_rate,
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        chamber_dimensions=chamber_dimensions,
    )

    # Calculate the cost
    number_cost = mean_squared_error(  # type: ignore
        time_derivative_concentration_pmf, net_rate)

    # total_volume comparison
    total_volume_cost = np.power(
        net_rate.sum() - time_derivative_concentration_pmf.sum(),
        2,
        dtype=np.float64,
    )

    if np.isnan(number_cost):  # type: ignore
        return 1e34

    return number_cost + total_volume_cost


@dataclass
class ChamberParameters:
    """Data class for the chamber parameters."""
    temperature: float = 293.15
    pressure: float = 101325
    particle_density: float = 1000
    volume: float = 1
    input_flow_rate_m3_sec: float = 1e-6
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1)


def create_guess_and_bounds(
    guess_eddy_diffusivity: float,
    guess_alpha_collision_efficiency: float,
    bounds_eddy_diffusivity: Tuple[float, float],
    bounds_alpha_collision_efficiency: Tuple[float, float],
) -> Tuple[NDArray[np.float64], List[Tuple[float, float]]]:
    """
    Create the initial guess array and bounds list for the optimization.

    Arguments:
        guess_eddy_diffusivity: Initial guess for eddy diffusivity.
        guess_alpha_collision_efficiency: Initial guess for alpha collision
            efficiency.
        bounds_eddy_diffusivity: Bounds for eddy diffusivity.
        bounds_alpha_collision_efficiency: Bounds for alpha collision
            efficiency.

    Returns:
        initial_guess: Numpy array of the initial guess values.
        bounds: List of tuples representing the bounds for each parameter.
    """
    initial_guess = np.array(
        [guess_eddy_diffusivity, guess_alpha_collision_efficiency],
        dtype=np.float64,
    )
    bounds = [bounds_eddy_diffusivity, bounds_alpha_collision_efficiency]

    return initial_guess, bounds


def optimize_parameters(
    cost_function: Callable,  # type: ignore
    initial_guess: NDArray[np.float64],
    bounds: List[Tuple[float, float]],
    method: str,
) -> Tuple[float, float]:
    """Get the optimized parameters using the given cost function."""
    result = minimize(  # type: ignore
        fun=cost_function,
        x0=initial_guess,
        method=method,
        bounds=bounds,
    )
    return result.x[0], result.x[1]  # type: ignore


# pylint: disable=too-many-arguments
def optimize_chamber_parameters(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    time_derivative_concentration_pmf: NDArray[np.float64],
    chamber_parameters: ChamberParameters,
    fit_guess: NDArray[np.float64],
    fit_bounds: List[Tuple[float, float]],
    minimize_method: str = "L-BFGS-B",
) -> Tuple[float, float]:
    """
    Optimize the eddy diffusivity and alpha collision efficiency parameters
    for a given particle size distribution and its time derivative.

    Arguments:
        radius_bins: Array of particle size bins in meters.
        concentration_pmf: Array of particle mass fractions (PMF)
            concentrations at each radius bin.
        time_derivative_concentration_pmf: Array of time derivatives of
            the PMF concentrations, representing the rate of change
            in concentration over time.
        chamber_params: ChamberParameters object containing the physical
            properties of the chamber, including temperature, pressure,
            particle density, volume, input flow rate, and chamber dimensions.
        fit_guess: Initial guess for the optimization parameters
            (eddy diffusivity and alpha collision efficiency).
        fit_bounds: List of tuples specifying the bounds for the
            optimization parameters (lower and upper bounds
            for each parameter).
        minimize_method: Optimization method to be used. Default is "L-BFGS-B".
            The following methods from `scipy.optimize.minimize` accept bounds,
            "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr".

    Returns:
        wall_eddy_diffusivity_optimized: Optimized value of the wall eddy
            diffusivity (in 1/s).
        alpha_collision_efficiency_optimized: Optimized value of the alpha
            collision efficiency (dimensionless).
    """
    # Partial evaluation of the cost function
    partial_cost_function = partial(
        coagulation_rates_cost_function,
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        time_derivative_concentration_pmf=time_derivative_concentration_pmf,
        temperature=chamber_parameters.temperature,
        pressure=chamber_parameters.pressure,
        particle_density=chamber_parameters.particle_density,
        volume=chamber_parameters.volume,
        input_flow_rate=chamber_parameters.input_flow_rate_m3_sec,
        chamber_dimensions=chamber_parameters.chamber_dimensions,
    )

    # Optimize the parameters
    return optimize_parameters(
        cost_function=partial_cost_function,
        initial_guess=fit_guess,
        bounds=fit_bounds,
        method=minimize_method,
    )


# pylint: disable=too-many-arguments
def calculate_optimized_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    wall_eddy_diffusivity: float,
    alpha_collision_efficiency: float,
    chamber_parameters: ChamberParameters,
    time_derivative_concentration_pmf: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate the coagulation rates using the optimized parameters and return
    the rates and R2 score.

    Arguments:
        radius_bins: Array of particle radii in meters.
        concentration_pmf: 2D array of concentration PMF values.
        wall_eddy_diffusivity: Optimized wall eddy diffusivity.
        alpha_collision_efficiency: Optimized alpha collision efficiency.
        chamber_params: ChamberParameters object containing chamber-related
            parameters.
        time_derivative_concentration_pmf: Array of observed rate of change
            of the concentration PMF (optional).

    Returns:
        coagulation_loss: Loss rate due to coagulation.
        coagulation_gain: Gain rate due to coagulation.
        dilution_loss: Loss rate due to dilution.
        wall_loss_rate: Loss rate due to wall deposition.
        net_rate: Net rate considering all effects.
        r2_value: R2 score between the net rate and the observed rate.
    """
    # Calculate the rates
    (
        coagulation_loss,
        coagulation_gain,
        dilution_loss,
        wall_loss_rate,
        net_rate,
    ) = calculate_pmf_rates(
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        temperature=chamber_parameters.temperature,
        pressure=chamber_parameters.pressure,
        particle_density=chamber_parameters.particle_density,
        alpha_collision_efficiency=alpha_collision_efficiency,
        volume=chamber_parameters.volume,
        input_flow_rate=chamber_parameters.input_flow_rate_m3_sec,
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        chamber_dimensions=chamber_parameters.chamber_dimensions,
    )

    coagulation_net = coagulation_gain - coagulation_loss

    r2_value = (  # type: ignore
        r2_score(time_derivative_concentration_pmf, net_rate)
        if time_derivative_concentration_pmf is not None
        else None
    )

    return (  # type: ignore
        coagulation_loss,
        coagulation_gain,
        coagulation_net,
        dilution_loss,
        wall_loss_rate,
        r2_value,
    )


# pylint: disable=too-many-locals
def optimize_and_calculate_rates_looped(
    pmf_stream: Stream,
    pmf_derivative_stream: Stream,
    chamber_parameters: ChamberParameters,
    fit_guess: NDArray[np.float64],
    fit_bounds: List[Tuple[float, float]],
) -> Tuple[Stream, Stream, Stream, Stream, Stream, Stream, Stream]:
    """
    Perform optimization and calculate rates for each time point in the stream.

    Arguments:
        pmf_stream: Stream object containing the fitted PMF data.
        pmf_derivative_stream: Stream object containing the derivative of the
            PMF data.
        chamber_parameters: ChamberParameters object containing
            chamber-related parameters.
        fit_guess: Initial guess for the optimization.
        fit_bounds: Bounds for the optimization parameters.

    Returns:
        result_stream: Stream containing the optimization results for
            each time point.
        coagulation_loss_stream: Stream containing coagulation loss rates.
        coagulation_gain_stream: Stream containing coagulation gain rates.
        coagulation_net_stream: Stream containing net coagulation rates.
        dilution_loss_stream: Stream containing dilution loss rates.
        wall_loss_rate_stream: Stream containing wall loss rates.
        total_rate_stream: Stream containing total rates.
    """
    fit_length = len(pmf_stream.time)

    # Prepare result storage
    wall_eddy_diffusivity = np.zeros(fit_length)
    alpha_collision_efficiency = np.zeros(fit_length)
    r2_value = np.zeros(fit_length)
    coagulation_loss = np.zeros_like(pmf_stream.data)
    coagulation_gain = np.zeros_like(pmf_stream.data)
    coagulation_net = np.zeros_like(pmf_stream.data)
    dilution_loss = np.zeros_like(pmf_stream.data)
    wall_loss_rate = np.zeros_like(pmf_stream.data)
    total_rate = np.zeros_like(pmf_stream.data)

    # Loop through each index and perform optimization and rate calculation
    for index in tqdm(
        range(fit_length), desc="Chamber rates", total=fit_length
    ):
        # Optimize chamber parameters
        wall_eddy_diffusivity[index], alpha_collision_efficiency[index] = (
            optimize_chamber_parameters(
                radius_bins=pmf_stream.header_float,
                concentration_pmf=pmf_stream.data[index, :],
                time_derivative_concentration_pmf=(
                    pmf_derivative_stream.data[index, :]),
                chamber_parameters=chamber_parameters,
                fit_guess=fit_guess,
                fit_bounds=fit_bounds,
            )
        )

        # Calculate the rates
        (
            coagulation_loss[index, :],
            coagulation_gain[index, :],
            coagulation_net[index, :],
            dilution_loss[index, :],
            wall_loss_rate[index, :],
            r2_value[index],
        ) = calculate_optimized_rates(
            radius_bins=pmf_stream.header_float,
            concentration_pmf=pmf_stream.data[index, :],
            wall_eddy_diffusivity=wall_eddy_diffusivity[index],
            alpha_collision_efficiency=alpha_collision_efficiency[index],
            chamber_parameters=chamber_parameters,
            time_derivative_concentration_pmf=(
                pmf_derivative_stream.data[index, :]),
        )

        # Store the total
        total_rate[index, :] = (
            coagulation_net[index, :]
            + dilution_loss[index, :]
            + wall_loss_rate[index, :]
        )

    # Create the result stream
    result_stream = Stream()
    result_stream.time = pmf_stream.time
    result_stream.header = [
        "wall_eddy_diffusivity_[1/s]",
        "alpha_collision_efficiency_[-]",
        "r2_value",
    ]
    result_stream.data = np.column_stack(
        [wall_eddy_diffusivity, alpha_collision_efficiency, r2_value]
    )

    # Add derived rates to the result stream
    result_stream["coagulation_loss_[1/m3s]"] = coagulation_loss.sum(axis=1)
    result_stream["coagulation_gain_[1/m3s]"] = coagulation_gain.sum(axis=1)
    result_stream["coagulation_net_[1/m3s]"] = coagulation_net.sum(axis=1)
    result_stream["dilution_loss_[1/m3s]"] = dilution_loss.sum(axis=1)
    result_stream["wall_loss_rate_[1/m3s]"] = wall_loss_rate.sum(axis=1)
    result_stream["total_rate_[1/m3s]"] = total_rate.sum(axis=1)

    # Add fractions of the total rate
    total_rate_sum = total_rate.sum(axis=1)
    result_stream["coagulation_loss_fraction"] = (
        coagulation_loss.sum(axis=1) / total_rate_sum
    )
    result_stream["coagulation_gain_fraction"] = (
        coagulation_gain.sum(axis=1) / total_rate_sum
    )
    result_stream["coagulation_net_fraction"] = (
        coagulation_net.sum(axis=1) / total_rate_sum
    )
    result_stream["dilution_loss_fraction"] = (
        dilution_loss.sum(axis=1) / total_rate_sum
    )
    result_stream["wall_loss_rate_fraction"] = (
        wall_loss_rate.sum(axis=1) / total_rate_sum
    )

    # Create and return additional streams
    coagulation_loss_stream = copy.deepcopy(pmf_stream)
    coagulation_loss_stream.data = coagulation_loss

    coagulation_gain_stream = copy.deepcopy(pmf_stream)
    coagulation_gain_stream.data = coagulation_gain

    coagulation_net_stream = copy.deepcopy(pmf_stream)
    coagulation_net_stream.data = coagulation_net

    dilution_loss_stream = copy.deepcopy(pmf_stream)
    dilution_loss_stream.data = dilution_loss

    wall_loss_rate_stream = copy.deepcopy(pmf_stream)
    wall_loss_rate_stream.data = wall_loss_rate

    total_rate_stream = copy.deepcopy(pmf_stream)
    total_rate_stream.data = total_rate

    return (
        result_stream,
        coagulation_loss_stream,
        coagulation_gain_stream,
        coagulation_net_stream,
        dilution_loss_stream,
        wall_loss_rate_stream,
        total_rate_stream,
    )
