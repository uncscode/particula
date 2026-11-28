"""Equilibrium CCN spectra of dry material in discrete or resolved particles."""

from dataclasses import dataclass
from operator import index

import numpy as np

from particula.particles.properties.kappa_kohler import (
    get_kappa_kohler_critical_dry_diameter,
)
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
)


def _values(value, name, lower=0.0, strict=False):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} requires real values")
    data = np.asarray(value, dtype=np.float64)
    if data.ndim > 1 or not data.size or not np.all(np.isfinite(data)):
        raise ValueError(f"{name} requires finite scalar or vector values")
    if np.any(data < lower) or (strict and np.any(data == lower)):
        raise ValueError(f"{name} outside its physical domain")
    return tuple(map(float, data.reshape(-1)))


@dataclass(frozen=True, kw_only=True)
class CCNConfig:
    """Define counter settings and the dry material model.

    saturation_ratios are dimensionless and above one. Supply either kappa
    (one value or one per box) or material_kappa in particle material order,
    with zero in the water lane. Supply water_index explicitly, using None
    for an all-dry material list. kelvin_diameter is A in metres, scalar or
    per box. If omitted, use the live box temperature (K), surface tension
    (N/m), water density (kg/m³) and water molar mass (kg/mol).
    """

    saturation_ratios: tuple[float, ...]
    water_index: int | None
    kappa: float | tuple[float, ...] | None = None
    material_kappa: tuple[float, ...] | None = None
    kelvin_diameter: float | tuple[float, ...] | None = None
    surface_tension: float = 0.072
    water_density: float = 997.0
    water_molar_mass: float = 0.01801528

    def __post_init__(self):
        """Freeze input coordinates and reject conflicting material models."""
        if (self.kappa is None) == (self.material_kappa is None):
            raise ValueError("Set exactly one of kappa and material_kappa")
        for key, lower, strict in (
            ("saturation_ratios", 1, True),
            ("kappa", 0, False),
            ("material_kappa", 0, False),
            ("kelvin_diameter", 0, True),
        ):
            val = getattr(self, key)
            if val is not None:
                object.__setattr__(self, key, _values(val, key, lower, strict))
        if self.water_index is not None:
            if (
                isinstance(self.water_index, (bool, np.bool_))
                or index(self.water_index) < 0
            ):
                raise ValueError(
                    "water_index must be a nonnegative material index"
                )
            object.__setattr__(self, "water_index", index(self.water_index))
        for key in ("surface_tension", "water_density", "water_molar_mass"):
            val = _values(getattr(self, key), key, strict=True)
            if len(val) != 1:
                raise ValueError(f"{key} requires a scalar")
            object.__setattr__(self, key, val[0])

    @property
    def kelvin_temperature_product(self):
        """Return A times T in m K for constant water properties."""
        return 2 * get_kelvin_radius(
            self.surface_tension, self.water_density, self.water_molar_mass, 1.0
        )

    def bind(self, boxes, materials, distribution):
        """Validate population and material axes and expand per-box inputs."""
        if distribution not in ("discrete", "particle_resolved"):
            raise ValueError(
                "CCN sampling requires discrete or resolved counts"
            )
        water = self.water_index
        if water is not None and water >= materials:
            raise ValueError("water_index outside particle material axis")
        if self.material_kappa is not None:
            if len(self.material_kappa) != materials or (
                water is not None and self.material_kappa[water] != 0
            ):
                raise ValueError(
                    "material_kappa must match material order and exclude water"
                )
        common = (
            None
            if self.kappa is None
            else np.broadcast_to(self.kappa, (boxes,))
        )
        length = (
            None
            if self.kelvin_diameter is None
            else np.broadcast_to(self.kelvin_diameter, (boxes,))
        )
        return common, length


@dataclass(frozen=True)
class CCNResult:
    """Number density (box, setting) and critical dry diameters [m].

    Common κ yields diameters (box, setting). Mixed material yields individual
    diameters (box, slot, setting), with NaN for slots without a dry seed.
    """

    equilibrium_ccn_number_concentration: np.ndarray
    critical_dry_diameter: np.ndarray


def _evaluate(
    masses,
    population,
    volume,
    density,
    temperature,
    distribution,
    config,
    fixed=None,
):
    boxes, slots, materials = masses.shape
    common, length = config.bind(boxes, materials, distribution)
    if length is None:
        length = config.kelvin_temperature_product / np.asarray(temperature)
    volumes = masses / density
    if config.water_index is not None:
        volumes[..., config.water_index] = 0
    dry = volumes.sum(axis=-1)
    seed = (population > 0) & (dry > 0)
    diameter = np.cbrt(dry * (6.0 / np.pi))
    ratios = np.asarray(config.saturation_ratios)
    if fixed is not None:
        critical = fixed
        threshold = fixed[:, None, :]
    elif common is not None:
        critical = get_kappa_kohler_critical_dry_diameter(
            common[:, None], ratios, length[:, None]
        )
        threshold = np.asarray(critical)[:, None, :]
    else:
        fractions = np.divide(
            volumes,
            dry[..., None],
            out=np.zeros_like(volumes),
            where=dry[..., None] > 0,
        )
        kappa = fractions @ np.asarray(config.material_kappa)
        threshold = np.full((boxes, slots, len(ratios)), np.nan)
        threshold[seed] = get_kappa_kohler_critical_dry_diameter(
            kappa[seed][:, None], ratios, length[np.nonzero(seed)[0], None]
        )
        critical = threshold
    # Admit ties within 32 float64 epsilons to accommodate backend rounding
    # of the dry-volume cube root and stationary-point solve.
    eligible = seed[..., None] & (
        diameter[..., None] >= threshold * (1 - 32 * np.finfo(float).eps)
    )
    number = np.sum(eligible * population[..., None], axis=1) / volume[:, None]
    if not np.all(np.isfinite(number)):
        raise FloatingPointError(
            "CCN number concentration outside float64 range"
        )
    return CCNResult(number, np.asarray(critical))


def evaluate_ccn(state, config: CCNConfig) -> CCNResult:
    """Evaluate CCN number [m⁻³] and dry thresholds [m] on a snapshot."""
    return _sample_state(state, config, None)


def _sample_state(state, config, fixed):
    particles = state.particles
    particles.validate_representation()
    return _evaluate(
        particles.masses,
        particles.population,
        particles.volume,
        particles.density,
        state.environment.temperature,
        particles.distribution_type,
        config,
        fixed,
    )


def evaluate_ccn_history(history, config: CCNConfig) -> CCNResult:
    """Evaluate a saved native history on a new saturation grid.

    Requires slot masses, populations, sampled volume and material densities.
    Temperature is needed for a temperature-derived Kelvin length. Legacy
    discrete concentration histories use their per-volume convention.
    """
    data = history.arrays
    science = history.metadata["science"]
    distribution = science["distribution_type"]
    required = {"particle_masses", "volume"}
    if config.kelvin_diameter is None:
        required.add("temperature")
    if missing := required.difference(data):
        raise ValueError(f"CCN history requires {sorted(missing)}")
    if "particle_population" in data:
        pop = data["particle_population"]
    elif "particle_concentration" in data:
        pop = data["particle_concentration"]
        if distribution == "discrete":
            pop = pop * data["volume"][..., None]
    else:
        raise ValueError("CCN history requires particle_population")
    density = np.asarray(science["material_density_kg_m3"])
    rows = [
        _evaluate(
            mass,
            pop[i],
            data["volume"][i],
            density,
            data["temperature"][i] if "temperature" in data else None,
            distribution,
            config,
        )
        for i, mass in enumerate(data["particle_masses"])
    ]
    return CCNResult(
        np.stack([row.equilibrium_ccn_number_concentration for row in rows]),
        np.stack([row.critical_dry_diameter for row in rows]),
    )
