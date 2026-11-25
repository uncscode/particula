"""Empirical finite-reservoir scenarios for the two measured recipes.

All concentrations are mol m-3 of wet solution. Both input sizes are radii.
Thermodynamic surface excesses of the ternary mixtures are unmeasured.
"""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAS_CONSTANT = 8.314462618
WATER_MOLAR_MASS = 0.01801528
WATER_DENSITY = 1000.0
DRY_DENSITY = 1665.0  # Anhydrous acid proxy for the acid-dominated dry mixture.
ACID_MOLAR_MASS = 0.192124
SURF_MOLAR_MASS = {"octyl": 0.23227, "dodecyl": 0.28838}
RATIO = {"octyl": 100.0, "dodecyl": 1000.0}
WATER_SIGMA = 72.2  # mN/m, Fig. 5 water endpoints at 295 K
LAB_RADIUS = 3.75e-6  # midpoint of reported 5–10 µm diameter range
SCENARIOS = (
    "central",
    "low",
    "high",
    "growth-low",
    "growth-high",
    "dilute-low",
    "dilute-high",
    "lab-small",
    "lab-large",
    "error-low",
    "error-high",
    "response-low",
    "response-high",
)


def read_rows(file):
    """Read a bounded study input table next to the model."""
    with (HERE / file).open() as stream:
        return list(csv.DictReader(stream))


OBS = read_rows("figure5.csv")
GROWTH = [
    r
    for r in read_rows("growth_envelope.csv")
    if int(r["rh_pct"]) % 5 == 0 or int(r["rh_pct"]) == 94
]


@lru_cache(None)
def growth_knots(scenario):
    """Derive acid uptake from Figure 6b with a dilute κ extension."""
    rh = np.array([float(row["rh_pct"]) / 100 for row in GROWTH])
    low = np.array([float(row["gf_min"]) for row in GROWTH])
    high = np.array([float(row["gf_max"]) for row in GROWTH])
    factor = {"growth-low": low, "growth-high": high}.get(
        scenario, (high + low) / 2
    )
    # Water/dry volume at 10% RH is assumed. The figure gives ratios only.
    q10 = {"growth-low": 0.01, "growth-high": 0.05}.get(scenario, 0.03)
    q = (1 + q10) * (factor / factor[0]) ** 3 - 1
    if np.any(np.diff(q) <= 0):
        raise ValueError("Growth envelope does not increase with humidity")
    return np.r_[0, q], np.r_[0, rh], q[-1] * (1 - rh[-1]) / rh[-1]


def acid_activity(q, scenario):
    """Interpolate the acid activity at a given water/acid volume ratio."""
    q_grid, rh, dilute_kappa = growth_knots(scenario)
    return np.where(
        q <= q_grid[-1], np.interp(q, q_grid, rh), q / (q + dilute_kappa)
    )


def properties(recipe, scenario):
    """Return assumed saturation excess (mol/m²) and midpoint (mol/m³)."""
    if recipe == "acid":
        return 0.0, 1.0
    # Independent molecular-area scale ~0.42 nm² (SDS, air/water).
    gamma = 3.8e-6
    half = {"octyl": 10.0, "dodecyl": 1.0}[recipe]
    # Low/high stress the two recipes in opposing directions. Stronger binding
    # produces a larger reference surface excess at the measured tension.
    if scenario in ("low", "high"):
        stronger = (scenario == "low") == (recipe == "dodecyl")
        gamma *= 1.35 if stronger else 0.65
        half *= 0.3 if stronger else 3.0
    return gamma, half


def lab_radius(scenario):
    """Return the assumed wet radius of a trapped drop in metres."""
    return {"lab-small": 2.5e-6, "lab-large": 5e-6}.get(scenario, LAB_RADIUS)


def free_concentration(total, capacity, half):
    """Solve the Langmuir inventory for nonnegative dissolved concentration."""
    b = half + capacity - total
    root = np.sqrt(b**2 + 4 * total * half)
    return np.where(
        b >= 0, 2 * total * half / (root + np.abs(b)), (root - b) / 2
    )


@lru_cache(None)
def series(recipe, scenario):
    """Return the measured tension series and instrument reference radii."""
    rows = sorted(
        (r for r in OBS if r["recipe"] == recipe),
        key=lambda r: float(r["rh_percent"]),
    )
    rh = np.array([float(r["rh_percent"]) / 100 for r in rows])
    tensions = np.array([float(r["sigma_mN_m"]) for r in rows])
    if scenario in ("error-low", "error-high") and recipe != "acid":
        sign = 1 if (scenario == "error-low") == (recipe == "dodecyl") else -1
        tensions += sign * np.array(
            [float(r["reported_error_mN_m"] or 0) for r in rows]
        )
    reciprocal_radius = np.array(
        [
            1 / lab_radius(scenario) if r["method"] == "trap" else 0.0
            for r in rows
        ]
    )
    if rh[-1] < 1:
        rh = np.r_[rh, 1.0]
        tensions = np.r_[tensions, WATER_SIGMA]
        reciprocal_radius = np.r_[reciprocal_radius, 0.0]
    return rh, tensions, reciprocal_radius


def reference_tension(aw, recipe, scenario):
    """Interpolate tension in mN/m and inverse reference radius in m⁻¹."""
    rh, tensions, inverse_radius = series(recipe, scenario)
    sigma = np.interp(aw, rh, tensions)
    if scenario in ("dilute-low", "dilute-high") and recipe != "acid":
        slow_return = (scenario == "dilute-high") == (recipe == "dodecyl")
        shape = 0.33 if slow_return else 3.0
        t = np.clip((1 - aw) / (1 - rh[-2]), 0.0, 1.0)
        weight = t / (shape + (1 - shape) * t)
        sigma = np.where(
            aw >= rh[-2],
            WATER_SIGMA + (tensions[-2] - WATER_SIGMA) * weight,
            sigma,
        )
    return sigma, np.interp(aw, rh, inverse_radius)


def state(
    dry_radius_m, wet_radius_m, recipe, model="finite", scenario="central"
):
    """Return activity, tension, concentrations and surface excesses."""
    if recipe not in ("acid", "octyl", "dodecyl") or model not in (
        "finite",
        "bulk",
    ):
        raise ValueError((recipe, model))
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    wet = np.asarray(wet_radius_m)
    dry_fraction = (np.asarray(dry_radius_m) / wet) ** 3
    if np.any(dry_fraction > 1 + 1e-10) or np.any(dry_fraction <= 0):
        raise ValueError("Require wet radius >= positive dry radius")
    water_fraction = np.maximum(1 - dry_fraction, 0)
    if recipe == "acid":
        aw = acid_activity(water_fraction / dry_fraction, scenario)
        sigma, _ = reference_tension(aw, recipe, scenario)
        zero = np.zeros_like(aw)
        return aw, sigma, zero, zero, zero, zero
    mixture_mass = RATIO[recipe] * ACID_MOLAR_MASS + SURF_MOLAR_MASS[recipe]
    total = dry_fraction * DRY_DENSITY / mixture_mass
    acid_mass_fraction = RATIO[recipe] * ACID_MOLAR_MASS / mixture_mass
    acid_aw = acid_activity(
        water_fraction / (dry_fraction * acid_mass_fraction), scenario
    )
    gamma, half = properties(recipe, scenario)
    dissolved = (
        free_concentration(total, 3 * gamma / wet, half)
        if model == "finite"
        else total
    )
    water = water_fraction * WATER_DENSITY / WATER_MOLAR_MASS
    # Ideal two-osmole increment from dissolved surfactant salt on the acid
    # baseline.
    # The adsorbed salt is treated as neutral ion pairs without a separate
    # double layer.
    aw = acid_aw * water / (water + 2 * dissolved)
    sigma_acid, _ = reference_tension(acid_aw, "acid", scenario)
    sigma_path, inverse_radius = reference_tension(aw, recipe, scenario)
    free_ref = free_concentration(total, 3 * gamma * inverse_radius, half)
    gamma_ref = gamma * free_ref / (half + free_ref)
    gamma_here = gamma * dissolved / (half + dissolved)
    # Empirical pressure proportional to fractional coverage at fixed acid
    # dilution.
    # Calibrate the coefficient along the Fig. 5 path at each reference radius.
    relative_pressure = gamma_here / gamma_ref
    if scenario in ("response-low", "response-high"):
        # A curved surface pressure response preserves the same laboratory
        # anchor
        # and a linear low-coverage limit; the data do not identify its shape.
        efficient = (scenario == "response-high") == (recipe == "dodecyl")
        curvature = 0.2 if efficient else 3.0
        theta = gamma_here / gamma
        theta_ref = gamma_ref / gamma
        relative_pressure *= (curvature + (1 - curvature) * theta_ref) / (
            curvature + (1 - curvature) * theta
        )
    sigma = (
        sigma_acid - (sigma_acid - sigma_path) * relative_pressure
        if model == "finite"
        else sigma_path
    )
    return aw, sigma, total, dissolved, gamma_here, gamma_ref


def equilibrium_ratio(
    dry_radius_m,
    wet_radius_m,
    temperature_k,
    recipe,
    model="finite",
    scenario="central",
):
    """Return the equilibrium saturation ratio including Kelvin curvature."""
    aw, sigma, *_ = state(dry_radius_m, wet_radius_m, recipe, model, scenario)
    kelvin = (
        2
        * WATER_MOLAR_MASS
        * sigma
        * 1e-3
        / (
            GAS_CONSTANT
            * temperature_k
            * WATER_DENSITY
            * np.asarray(wet_radius_m)
        )
    )
    return aw * np.exp(kelvin)
