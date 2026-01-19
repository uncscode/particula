"""Equilibrium calculations for the particula thermodynamic model.

This module provides utilities to solve liquid-vapor partitioning using
activity coefficients and water activity. The implementation follows
Google-style documentation, explicit type hints, and small helpers to
improve readability and testability.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, minimize

from particula.activity.water_activity import fixed_water_activity
from particula.util.validate_inputs import validate_inputs

EPSILON = 1e-16

PhaseOutput = tuple[NDArray[np.float64], NDArray[np.float64], float, float]
SystemOutput = tuple[float, float, NDArray[np.float64], float]


def _calculate_phase(
    c_j_liquid: NDArray[np.float64],
    q_phase: NDArray[np.float64],
    mass_fraction_water_phase: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    NDArray[np.float64],
]:
    """Compute phase concentrations, aqueous content, and totals.

    Args:
        c_j_liquid: Liquid phase concentrations per species [µg/m³].
        q_phase: Phase fraction for the target phase (q_α or q_β) [-].
        mass_fraction_water_phase: Mass fraction of water per species in the
            target phase [-].

    Returns:
        Tuple containing:
        - c_j_phase: Organic concentrations per species in the phase [µg/m³].
        - c_j_aq_phase: Aqueous concentrations per species in the phase
            [µg/m³].
        - c_phase_total: Total concentration (organic + water) in the phase
            [µg/m³].
        - c_aq_total: Total aqueous concentration in the phase [µg/m³].
        - denominator: Stabilized denominator ``(1 - mass_fraction_water)``
            inverse with zero-fill when the mass fraction is one.
    """
    q_phase = np.asarray(q_phase, dtype=float)
    mass_fraction_water_phase = np.asarray(
        mass_fraction_water_phase, dtype=float
    )

    denominator = np.where(
        mass_fraction_water_phase >= 1.0,
        0.0,
        1.0 / (1.0 - mass_fraction_water_phase + EPSILON),
    )
    c_j_phase = c_j_liquid * q_phase
    c_j_aq_phase = c_j_phase * mass_fraction_water_phase * denominator
    c_phase_total = float(np.sum(c_j_phase) + np.sum(c_j_aq_phase))
    c_aq_total = float(np.sum(c_j_aq_phase))
    return c_j_phase, c_j_aq_phase, c_phase_total, c_aq_total, denominator


def _calculate_alpha_phase(
    c_j_liquid: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    NDArray[np.float64],
]:
    """Wrapper to compute alpha-phase concentrations."""
    return _calculate_phase(
        c_j_liquid, q_ab[:, 0], mass_fraction_water_ab[:, 0]
    )


def _calculate_beta_phase(
    c_j_liquid: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    NDArray[np.float64],
]:
    """Wrapper to compute beta-phase concentrations."""
    return _calculate_phase(
        c_j_liquid, q_ab[:, 1], mass_fraction_water_ab[:, 1]
    )


def _calculate_cstar(
    c_star_j_dry: NDArray[np.float64],
    gamma_organic_phase: NDArray[np.float64],
    q_phase: NDArray[np.float64],
    c_liquid_total: float,
    molar_mass: NDArray[np.float64],
    mass_weighted_molar_mass: float,
) -> NDArray[np.float64]:
    r"""Calculate phase-specific C* with zero protection.

    Implements:

    .. math::

        C^*_j = C^*_{j,\mathrm{dry}} \cdot \gamma_{j,\mathrm{phase}} \cdot
        q_{\mathrm{phase}} \cdot \frac{C_{\mathrm{liq}}}
        {M_j \cdot \overline{M}_{\mathrm{phase}}}

    Args:
        c_star_j_dry: Dry saturation concentrations [µg/m³].
        gamma_organic_phase: Activity coefficients for the phase [-].
        q_phase: Phase fraction for the target phase [-].
        c_liquid_total: Total liquid concentration across phases [µg/m³].
        molar_mass: Species molar masses [g/mol].
        mass_weighted_molar_mass: Mass-weighted mean molar mass for the phase
            [g/mol].

    Returns:
        Phase-specific C* array [µg/m³]; zeros if the mass-weighted molar mass
        is zero to maintain numerical stability.
    """
    if mass_weighted_molar_mass <= 0:
        return np.zeros_like(c_star_j_dry, dtype=float)

    return (
        c_star_j_dry
        * gamma_organic_phase
        * q_phase
        * c_liquid_total
        / (molar_mass * mass_weighted_molar_mass)
    )


@overload
def liquid_vapor_obj_function(
    e_j_partition_guess: NDArray[np.float64],
    c_star_j_dry: NDArray[np.float64],
    concentration_organic_matter: NDArray[np.float64],
    gamma_organic_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    *,
    error_only: Literal[True] = True,
) -> float: ...


@overload
def liquid_vapor_obj_function(
    e_j_partition_guess: NDArray[np.float64],
    c_star_j_dry: NDArray[np.float64],
    concentration_organic_matter: NDArray[np.float64],
    gamma_organic_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    *,
    error_only: Literal[False],
) -> tuple[PhaseOutput, PhaseOutput, SystemOutput]: ...


def liquid_vapor_obj_function(
    e_j_partition_guess: NDArray[np.float64],
    c_star_j_dry: NDArray[np.float64],
    concentration_organic_matter: NDArray[np.float64],
    gamma_organic_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    *,
    error_only: bool = True,
) -> float | tuple[PhaseOutput, PhaseOutput, SystemOutput]:
    """Objective function for liquid-vapor partitioning.

    Args:
        e_j_partition_guess: Partition coefficient guess per species [-].
        c_star_j_dry: Dry saturation concentrations [µg/m³].
        concentration_organic_matter: Organic aerosol mass concentration
            [µg/m³].
        gamma_organic_ab: Activity coefficients (shape: (n, 2)) [-].
        mass_fraction_water_ab: Water mass fractions (shape: (n, 2)) [-].
        q_ab: Phase fractions for alpha/beta (shape: (n, 2)) [-].
        molar_mass: Species molar masses [g/mol].
        error_only: Whether to return only the scalar error (True) or detailed
            phase/system outputs (False).

    Returns:
        If ``error_only`` is True, a scalar error. Otherwise, a tuple:
        ``(alpha_phase_output, beta_phase_output, system_output)`` where each
        phase output is ``(c_j_liquid, c_j_aq, c_liquid_total, c_aq_total)`` and
        ``system_output`` is ``(c_liquid_total, c_liquid_total_water,
        e_j_partition_new, error_out)``.

    Raises:
        ValueError: If input shapes are incompatible.

    Examples:
        >>> import numpy as np
        >>> from particula.equilibria import partitioning
        >>> guess = np.full(3, 0.5)
        >>> c_star = np.array([1e-3, 1e-2, 1e-1])
        >>> c_om = np.array([1.0, 2.0, 3.0])
        >>> gamma = np.ones((3, 2))
        >>> mf_water = np.zeros((3, 2))
        >>> q = np.full((3, 2), 0.5)
        >>> molar_mass = np.full(3, 200.0)
        >>> partitioning.liquid_vapor_obj_function(
        ...     guess, c_star, c_om, gamma, mf_water, q, molar_mass
        ... ) > 0
        True
    """
    species_count = len(c_star_j_dry)
    if (
        q_ab.shape != (species_count, 2)
        or gamma_organic_ab.shape
        != (
            species_count,
            2,
        )
        or mass_fraction_water_ab.shape != (species_count, 2)
    ):
        raise ValueError(
            "q_ab, gamma_organic_ab, and mass_fraction_water_ab "
            "must have shape (n, 2)"
        )

    c_j_liquid = e_j_partition_guess * concentration_organic_matter
    c_liquid_guess = float(np.sum(c_j_liquid))

    (
        c_j_alpha,
        c_j_aq_alpha_arr,
        c_alpha_total,
        c_aq_alpha,
        denominator_alpha,
    ) = _calculate_alpha_phase(c_j_liquid, q_ab, mass_fraction_water_ab)
    c_j_beta, c_j_aq_beta_arr, c_beta_total, c_aq_beta, denominator_beta = (
        _calculate_beta_phase(c_j_liquid, q_ab, mass_fraction_water_ab)
    )

    c_liquid_total = c_alpha_total + c_beta_total

    mass_weighted_molar_mass_alpha = float(
        np.sum(c_j_alpha / molar_mass) + c_aq_alpha / 18.015
    )
    mass_weighted_molar_mass_beta = float(
        np.sum(c_j_beta / molar_mass) + c_aq_beta / 18.015
    )

    c_star_j_via_alpha = _calculate_cstar(
        c_star_j_dry,
        gamma_organic_ab[:, 0],
        q_ab[:, 0],
        c_liquid_total,
        molar_mass,
        mass_weighted_molar_mass_alpha,
    )
    c_star_j_via_beta = _calculate_cstar(
        c_star_j_dry,
        gamma_organic_ab[:, 1],
        q_ab[:, 1],
        c_liquid_total,
        molar_mass,
        mass_weighted_molar_mass_beta,
    )

    c_star_j_new = (
        c_star_j_via_alpha * q_ab[:, 0] + c_star_j_via_beta * q_ab[:, 1]
    )
    e_j_partition_new = 1.0 / (1.0 + c_star_j_new / (c_liquid_total + EPSILON))

    c_j_liquid_new = e_j_partition_new * concentration_organic_matter
    c_liquid_total_new = float(np.sum(c_j_liquid_new))

    error_out = float(
        np.sum((e_j_partition_guess - e_j_partition_new) ** 2)
        + (c_liquid_guess - c_liquid_total_new) ** 2
    )
    if error_only:
        return error_out

    (
        c_j_liquid_new_alpha,
        c_j_aq_new_alpha,
        c_liquid_new_alpha,
        c_aq_new_alpha,
        _,
    ) = _calculate_alpha_phase(c_j_liquid_new, q_ab, mass_fraction_water_ab)
    (
        c_j_liquid_new_beta,
        c_j_aq_new_beta,
        c_liquid_new_beta,
        c_aq_new_beta,
        _,
    ) = _calculate_beta_phase(c_j_liquid_new, q_ab, mass_fraction_water_ab)

    c_liquid_total_water_new = c_aq_new_alpha + c_aq_new_beta

    alpha_phase_output = (
        c_j_liquid_new_alpha,
        c_j_aq_new_alpha,
        c_liquid_new_alpha,
        c_aq_new_alpha,
    )
    beta_phase_output = (
        c_j_liquid_new_beta,
        c_j_aq_new_beta,
        c_liquid_new_beta,
        c_aq_new_beta,
    )
    system_output = (
        c_liquid_total_new,
        c_liquid_total_water_new,
        e_j_partition_new,
        error_out,
    )
    return alpha_phase_output, beta_phase_output, system_output


@validate_inputs(
    {
        "c_star_j_dry": "nonnegative",
        "concentration_organic_matter": "nonnegative",
    }
)
def liquid_vapor_partitioning(
    c_star_j_dry: NDArray[np.float64],
    concentration_organic_matter: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    gamma_organic_ab: NDArray[np.float64],
    mass_fraction_water_ab: NDArray[np.float64],
    q_ab: NDArray[np.float64],
    partition_coefficient_guess: NDArray[np.float64] | None = None,
) -> tuple[PhaseOutput, PhaseOutput, SystemOutput, OptimizeResult]:  # pylint: disable=too-many-arguments, too-many-locals
    """Thermodynamic equilibrium between liquid and vapor phases.

    Args:
        c_star_j_dry: Dry saturation concentrations [µg/m³], shape (n,).
        concentration_organic_matter: Organic aerosol mass concentration
            [µg/m³], shape (n,).

        molar_mass: Species molar masses [g/mol], shape (n,).
        gamma_organic_ab: Activity coefficients, shape (n, 2) [-].
        mass_fraction_water_ab: Water mass fractions, shape (n, 2) [-].
        q_ab: Phase fractions for alpha/beta, shape (n, 2) [-].
        partition_coefficient_guess: Optional initial partition coefficients
            (bounds [0, 1]); defaults to 0.5 for each species when None.

    Returns:
        Tuple ``(alpha, beta, system, fit_result)`` where ``alpha`` and ``beta``
        are phase outputs from :func:`liquid_vapor_obj_function`, ``system``
        contains aggregate system values, and ``fit_result`` is the SciPy
        optimization result.

    Raises:
        ValueError: If array lengths are inconsistent or guess length mismatches
        the number of species.
    """
    c_star_j_dry = np.asarray(c_star_j_dry, dtype=float)
    concentration_organic_matter = np.asarray(
        concentration_organic_matter, dtype=float
    )
    molar_mass = np.asarray(molar_mass, dtype=float)
    gamma_organic_ab = np.nan_to_num(np.asarray(gamma_organic_ab, dtype=float))
    mass_fraction_water_ab = np.nan_to_num(
        np.asarray(mass_fraction_water_ab, dtype=float)
    )
    q_ab = np.nan_to_num(np.asarray(q_ab, dtype=float))

    species_count = len(c_star_j_dry)
    expected_length = {
        len(concentration_organic_matter),
        len(molar_mass),
        gamma_organic_ab.shape[0],
        mass_fraction_water_ab.shape[0],
        q_ab.shape[0],
    }
    if len(expected_length) != 1:
        raise ValueError(
            "All input arrays must share the same length for species dimension."
        )
    # Check that arrays are 2D before accessing shape[1]
    if (
        gamma_organic_ab.ndim != 2
        or mass_fraction_water_ab.ndim != 2
        or q_ab.ndim != 2
    ):
        raise ValueError(
            "gamma_organic_ab, mass_fraction_water_ab, and q_ab must be "
            "2D arrays."
        )
    if (
        gamma_organic_ab.shape[1] != 2
        or mass_fraction_water_ab.shape[1] != 2
        or q_ab.shape[1] != 2
    ):
        raise ValueError(
            "gamma_organic_ab, mass_fraction_water_ab, and q_ab must have two "
            "columns."
        )

    if partition_coefficient_guess is None:
        partition_coefficient_guess = np.full(species_count, 0.5, dtype=float)
    else:
        partition_coefficient_guess = np.asarray(
            partition_coefficient_guess, dtype=float
        )

    if len(partition_coefficient_guess) != species_count:
        raise ValueError(
            "partition_coefficient_guess length must match c_star_j_dry length."
        )

    bounds = Bounds(lb=0, ub=1)

    problem = {
        "fun": lambda x: liquid_vapor_obj_function(
            e_j_partition_guess=x,
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            molar_mass=molar_mass,
            error_only=True,
        ),
        "x0": partition_coefficient_guess,
        "bounds": bounds,
    }

    fit_result = minimize(**problem)

    alpha, beta, system = liquid_vapor_obj_function(
        e_j_partition_guess=fit_result.x,
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
        molar_mass=molar_mass,
        error_only=False,
    )
    return alpha, beta, system, fit_result


def get_properties_for_liquid_vapor_partitioning(
    water_activity_desired: NDArray[np.float64] | float,
    molar_mass: NDArray[np.float64],
    oxygen2carbon: NDArray[np.float64],
    density: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Get activity and phase properties for liquid-vapor partitioning.

    Args:
        water_activity_desired: Target water activity [-]; scalar or array of
            length n for species broadcast.
        molar_mass: Species molar masses [g/mol], shape (n,).
        oxygen2carbon: Oxygen-to-carbon ratios [-], shape (n,).
        density: Species densities [kg/m³], shape (n,).

    Returns:
        Tuple ``(gamma_organic_ab, mass_fraction_water_ab, q_ab)`` each shaped
        (n, 2). ``gamma_organic_ab[:, 1]`` and ``mass_fraction_water_ab[:, 1]``
        are zero-filled when the beta phase is absent.

    Raises:
        ValueError: If input lengths are inconsistent or water activity cannot
        be broadcast to the species dimension.
    """
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=float)
    molar_mass = np.asarray(molar_mass, dtype=float)
    density = np.asarray(density, dtype=float)
    water_activity_desired_array = np.atleast_1d(
        np.asarray(water_activity_desired, dtype=float)
    )

    species_count = len(oxygen2carbon)
    if len(molar_mass) != species_count or len(density) != species_count:
        raise ValueError(
            "molar_mass, oxygen2carbon, and density must share length."
        )

    if water_activity_desired_array.size not in {1, species_count}:
        raise ValueError(
            "water_activity_desired must be scalar or match species length."
        )
    if water_activity_desired_array.size == 1:
        water_activity_desired_array = np.full(
            species_count, float(water_activity_desired_array.item())
        )

    gamma_organic_ab = np.empty([species_count, 2], dtype=float)
    mass_fraction_water_ab = np.empty([species_count, 2], dtype=float)
    q_ab = np.empty([species_count, 2], dtype=float)

    molar_mass_ratio = 18.015 / molar_mass

    for i, oxy in enumerate(oxygen2carbon):
        water_activity_scalar = float(water_activity_desired_array[i])
        water_activity = np.atleast_1d(water_activity_scalar)
        alpha_raw, beta_raw, q_alpha_raw = fixed_water_activity(
            water_activity=water_activity,
            molar_mass_ratio=molar_mass_ratio[i],
            oxygen2carbon=oxy,
            density=density[i],
        )

        alpha_activity_raw = np.atleast_1d(
            np.asarray(cast(Sequence[float], alpha_raw)[-1], dtype=float)
        )
        alpha_water_fraction_raw = np.atleast_1d(
            np.asarray(cast(Sequence[float], alpha_raw)[2], dtype=float)
        )

        gamma_organic_ab[i, 0] = float(alpha_activity_raw.ravel()[0])
        mass_fraction_water_ab[i, 0] = float(
            alpha_water_fraction_raw.ravel()[0]
        )

        if beta_raw is None:
            gamma_organic_ab[i, 1] = 0.0
            mass_fraction_water_ab[i, 1] = 0.0
        else:
            beta_activity_raw = np.atleast_1d(
                np.asarray(cast(Sequence[float], beta_raw)[-1], dtype=float)
            )
            beta_water_fraction_raw = np.atleast_1d(
                np.asarray(cast(Sequence[float], beta_raw)[2], dtype=float)
            )
            gamma_organic_ab[i, 1] = float(beta_activity_raw.ravel()[0])
            mass_fraction_water_ab[i, 1] = float(
                beta_water_fraction_raw.ravel()[0]
            )

        q_alpha_array = np.atleast_1d(np.asarray(q_alpha_raw, dtype=float))
        q_value = float(q_alpha_array.ravel()[0])
        q_ab[i, 0] = q_value
        q_ab[i, 1] = 1 - q_value

    return gamma_organic_ab, mass_fraction_water_ab, q_ab
