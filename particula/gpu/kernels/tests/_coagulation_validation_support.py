"""Independent fp64 fixtures and reference equations for coagulation probes.

This module deliberately contains no Warp import.  It is the expected-value
side of ``coagulation_validation_test``; device observations live there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particula.util import constants


@dataclass(frozen=True)
class MechanismRow:
    """Literal fixed-mask metadata used by the validation matrix."""

    mask: int
    mechanisms: tuple[str, ...]
    enabled: tuple[bool, bool, bool, bool]


EXECUTABLE_ROWS = (
    MechanismRow(1, ("brownian",), (True, False, False, False)),
    MechanismRow(2, ("charged_hard_sphere",), (False, True, False, False)),
    MechanismRow(
        3, ("brownian", "charged_hard_sphere"), (True, True, False, False)
    ),
    MechanismRow(4, ("sedimentation_sp2016",), (False, False, True, False)),
    MechanismRow(
        5, ("brownian", "sedimentation_sp2016"), (True, False, True, False)
    ),
    MechanismRow(
        6,
        ("charged_hard_sphere", "sedimentation_sp2016"),
        (False, True, True, False),
    ),
    MechanismRow(8, ("turbulent_shear_st1956",), (False, False, False, True)),
    MechanismRow(
        9, ("brownian", "turbulent_shear_st1956"), (True, False, False, True)
    ),
    MechanismRow(
        10,
        ("charged_hard_sphere", "turbulent_shear_st1956"),
        (False, True, False, True),
    ),
    MechanismRow(
        12,
        ("sedimentation_sp2016", "turbulent_shear_st1956"),
        (False, False, True, True),
    ),
    MechanismRow(
        15,
        (
            "brownian",
            "charged_hard_sphere",
            "sedimentation_sp2016",
            "turbulent_shear_st1956",
        ),
        (True, True, True, True),
    ),
)
DEFERRED_ROWS = (
    MechanismRow(
        7,
        ("brownian", "charged_hard_sphere", "sedimentation_sp2016"),
        (True, True, True, False),
    ),
    MechanismRow(
        11,
        ("brownian", "charged_hard_sphere", "turbulent_shear_st1956"),
        (True, True, False, True),
    ),
    MechanismRow(
        13,
        ("brownian", "sedimentation_sp2016", "turbulent_shear_st1956"),
        (True, False, True, True),
    ),
    MechanismRow(
        14,
        (
            "charged_hard_sphere",
            "sedimentation_sp2016",
            "turbulent_shear_st1956",
        ),
        (False, True, True, True),
    ),
)


@dataclass(frozen=True)
class Fixture:
    """Test-owned particle state with explicit compact active indices."""

    name: str
    radii: np.ndarray
    density: np.ndarray
    charges: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    dissipation: np.ndarray
    fluid_density: np.ndarray
    active: tuple[tuple[int, ...], ...]

    @property
    def masses(self) -> np.ndarray:
        """Return homogeneous-sphere fp64 masses for every fixture slot."""
        return (4.0 * np.pi / 3.0 * self.radii**3 * self.density).astype(
            np.float64
        )


def _fixture(
    name: str,
    radii: list[list[float]],
    charges: list[list[float]],
    active: tuple[tuple[int, ...], ...],
    temperature: tuple[float, ...] = (298.15,),
    pressure: tuple[float, ...] = (101325.0,),
    dissipation: tuple[float, ...] = (2.0e-4,),
) -> Fixture:
    """Create explicit float64 fixture data without randomization."""
    radii_array = np.asarray(radii, dtype=np.float64)
    return Fixture(
        name=name,
        radii=radii_array,
        density=np.full_like(radii_array, 1000.0),
        charges=np.asarray(charges, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        pressure=np.asarray(pressure, dtype=np.float64),
        dissipation=np.asarray(dissipation, dtype=np.float64),
        fluid_density=np.full(len(temperature), 1.2, dtype=np.float64),
        active=active,
    )


# Inactive slot 1 is deliberately larger than every active radius.
FIXTURES = {
    "normal": _fixture(
        "normal", [[8e-8, 9e-5, 2e-7, 3e-6]], [[1, 8, -1, 0]], ((0, 2, 3),)
    ),
    "two_box": _fixture(
        "two_box",
        [[5e-8, 9e-5, 2e-7, 4e-6], [8e-8, 8e-5, 7e-7, 6e-6]],
        [[1, 20, -1, 2], [-2, 20, 0, 3]],
        ((0, 2, 3), (0, 2, 3)),
        (285.0, 320.0),
        (90000.0, 105000.0),
        (2e-4, 4e-4),
    ),
    "repulsive": _fixture(
        "repulsive", [[1e-7, 9e-5, 2e-7, 3e-6]], [[1e8, 0, 1e8, 0]], ((0, 2),)
    ),
    "equal_velocity": _fixture(
        "equal_velocity", [[1e-6, 9e-5, 1e-6, 3e-6]], [[0, 0, 0, 0]], ((0, 2),)
    ),
    "zero_dissipation": _fixture(
        "zero_dissipation",
        [[1e-7, 9e-5, 2e-7, 3e-6]],
        [[0, 0, 0, 0]],
        ((0, 2, 3),),
        dissipation=(0.0,),
    ),
    "zero_active": _fixture(
        "zero_active", [[1e-7, 9e-5, 2e-7, 3e-6]], [[0, 0, 0, 0]], ((),)
    ),
    "one_active": _fixture(
        "one_active", [[1e-7, 9e-5, 2e-7, 3e-6]], [[0, 0, 0, 0]], ((2,),)
    ),
}

# Keep this table explicit so the required edge-case policy cannot silently
# change when fixed-mask behavior is extended.
FIXTURE_NAMES_BY_MASK = {
    1: ("normal", "two_box"),
    2: ("normal", "two_box", "repulsive"),
    3: ("normal", "two_box", "repulsive"),
    4: ("normal", "two_box", "equal_velocity"),
    5: ("normal", "two_box", "equal_velocity"),
    6: ("normal", "two_box", "repulsive", "equal_velocity"),
    8: ("normal", "zero_dissipation"),
    9: ("normal", "two_box", "zero_dissipation"),
    10: ("normal", "two_box", "repulsive", "zero_dissipation"),
    12: ("normal", "two_box", "equal_velocity", "zero_dissipation"),
    15: (
        "normal",
        "two_box",
        "repulsive",
        "equal_velocity",
        "zero_dissipation",
    ),
}


def applicable_fixture_names(mask: int) -> tuple[str, ...]:
    """Return the literal fixture policy for one executable row."""
    return FIXTURE_NAMES_BY_MASK[mask]


def _transport(temperature: float, pressure: float) -> tuple[float, float]:
    viscosity = (
        constants.REF_VISCOSITY_AIR_STP
        * (temperature / constants.REF_TEMPERATURE_STP) ** 1.5
        * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
        / (temperature + constants.SUTHERLAND_CONSTANT)
    )
    mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
        8.0
        * constants.MOLECULAR_WEIGHT_AIR
        / (np.pi * constants.GAS_CONSTANT * temperature)
    )
    return float(viscosity), float(mean_free_path)


def properties(fixture: Fixture, box: int) -> dict[str, np.ndarray | float]:
    """Independently derive all device pair-rate input properties in fp64."""
    radius = fixture.radii[box]
    mass = fixture.masses[box]
    mu, mfp = _transport(
        float(fixture.temperature[box]), float(fixture.pressure[box])
    )
    kn = mfp / radius
    slip = 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
    diffusivity = (
        constants.BOLTZMANN_CONSTANT
        * fixture.temperature[box]
        * slip
        / (6.0 * np.pi * mu * radius)
    )
    speed = np.sqrt(
        8.0
        * constants.BOLTZMANN_CONSTANT
        * fixture.temperature[box]
        / (np.pi * mass)
    )
    particle_mfp = 8.0 * diffusivity / (np.pi * speed)
    g_term = (
        (2.0 * radius + particle_mfp) ** 3
        - (4.0 * radius**2 + particle_mfp**2) ** 1.5
    ) / (6.0 * radius * particle_mfp) - 2.0 * radius
    settling = (
        2.0
        * radius**2
        * fixture.density[box]
        * slip
        * constants.STANDARD_GRAVITY
        / (9.0 * mu)
    )
    return {
        "diffusivity": diffusivity,
        "speed": speed,
        "g_term": g_term,
        "settling": settling,
        "nu": mu / fixture.fluid_density[box],
    }


def _charged(
    radius_i: float,
    radius_j: float,
    mass_i: float,
    mass_j: float,
    charge_i: float,
    charge_j: float,
    temperature: float,
    pressure: float,
) -> float:
    """Return the guarded charged hard-sphere probe oracle equation."""
    mu, mfp = _transport(temperature, pressure)
    slip_i = 1.0 + (mfp / radius_i) * (
        1.257 + 0.4 * np.exp(-1.1 / (mfp / radius_i))
    )
    slip_j = 1.0 + (mfp / radius_j) * (
        1.257 + 0.4 * np.exp(-1.1 / (mfp / radius_j))
    )
    friction_i, friction_j = (
        6 * np.pi * mu * radius_i / slip_i,
        6 * np.pi * mu * radius_j / slip_j,
    )
    potential = -(
        charge_i * charge_j * constants.ELEMENTARY_CHARGE_VALUE**2
    ) / (
        4
        * np.pi
        * constants.ELECTRIC_PERMITTIVITY
        * (radius_i + radius_j)
        * constants.BOLTZMANN_CONSTANT
        * temperature
    )
    potential = float(np.clip(potential, -200.0, 200.0))
    kinetic = 1.0 + potential if potential >= 0 else np.exp(potential)
    if kinetic < 1e-80:
        return 0.0
    continuum = 1.0 if potential == 0 else potential / (-np.expm1(-potential))
    reduced_mass = mass_i * mass_j / (mass_i + mass_j)
    reduced_friction = friction_i * friction_j / (friction_i + friction_j)
    diffusive_knudsen = (
        np.sqrt(constants.BOLTZMANN_CONSTANT * temperature * reduced_mass)
        / reduced_friction
    ) / ((radius_i + radius_j) * continuum / kinetic)
    numerator = (
        4 * np.pi * diffusive_knudsen**2
        + 25.836 * diffusive_knudsen**3
        + np.sqrt(8 * np.pi) * 11.211 * diffusive_knudsen**4
    )
    denominator = (
        1
        + 3.502 * diffusive_knudsen
        + 7.211 * diffusive_knudsen**2
        + 11.211 * diffusive_knudsen**3
    )
    dimensionless = numerator / denominator
    result = (
        dimensionless
        * reduced_friction
        * (radius_i + radius_j) ** 3
        * kinetic**2
        / (reduced_mass * continuum)
    )
    return float(result) if np.isfinite(result) and result > 0 else 0.0


def pair_rate(fixture: Fixture, box: int, i: int, j: int, mask: int) -> float:
    """Return the independent enabled-component sum with safe zeros."""
    p = properties(fixture, box)
    r, m = fixture.radii[box], fixture.masses[box]
    total = 0.0
    if mask & 1:
        total += brownian_rate_from_properties(fixture, box, i, j)
    if mask & 2:
        total += _charged(
            r[i],
            r[j],
            m[i],
            m[j],
            fixture.charges[box, i],
            fixture.charges[box, j],
            float(fixture.temperature[box]),
            float(fixture.pressure[box]),
        )
    if mask & 4:
        total += float(
            np.pi
            * (r[i] + r[j]) ** 2
            * abs(p["settling"][i] - p["settling"][j])
        )
    if mask & 8 and fixture.dissipation[box] != 0.0:
        total += float(
            np.sqrt(np.pi * fixture.dissipation[box] / (120.0 * p["nu"]))
            * (2.0 * (r[i] + r[j])) ** 3
        )
    return total if np.isfinite(total) and total > 0 else 0.0


def brownian_rate_from_properties(
    fixture: Fixture,
    box: int,
    i: int,
    j: int,
) -> float:
    """Calculate Brownian pair rate from independently derived inputs."""
    derived = properties(fixture, box)
    radius = fixture.radii[box]
    diffusivity = np.asarray(derived["diffusivity"], dtype=np.float64)
    g_term = np.asarray(derived["g_term"], dtype=np.float64)
    speed = np.asarray(derived["speed"], dtype=np.float64)
    sum_radius = radius[i] + radius[j]
    sum_diffusivity = diffusivity[i] + diffusivity[j]
    denominator = sum_radius / (
        sum_radius + np.hypot(g_term[i], g_term[j])
    ) + 4.0 * sum_diffusivity / (sum_radius * np.hypot(speed[i], speed[j]))
    return float(4.0 * np.pi * sum_diffusivity * sum_radius / denominator)


def selector_majorant(fixture: Fixture, box: int, mask: int) -> float:
    """Return the independent selector majorant, not a component sum."""
    active = fixture.active[box]
    if len(active) < 2:
        return 0.0
    if mask == 8:
        first, second = sorted((fixture.radii[box, index] for index in active))[
            -2:
        ]
        return (
            pair_rate(fixture, box, active[0], active[1], mask)
            if fixture.dissipation[box] == 0
            else float(
                np.sqrt(
                    np.pi
                    * fixture.dissipation[box]
                    / (120.0 * properties(fixture, box)["nu"])
                )
                * (2.0 * (first + second)) ** 3
            )
        )
    return max(
        pair_rate(fixture, box, i, j, mask)
        for position, i in enumerate(active)
        for j in active[position + 1 :]
    )
