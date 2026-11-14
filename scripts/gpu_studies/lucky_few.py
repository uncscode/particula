"""Controlled vapor-fed seed-survival study with explicit inventory accounting.

Parameters are demonstration assumptions, not a fitted atmospheric chemistry
model. One original seed per independent box makes the finite-horizon success
indicator a Bernoulli observation; the library still handles seed coalescence
for studies with multiple initial seeds. Production ion cases require an
explicit, scientifically justified LF2013 omission policy.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from particula import Aerosol
from particula.dynamics import MassCondensation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.ions.cleanup import IonNeutralCleanup
from particula.dynamics.ions.coagulation import (
    NEGATIVE_ION_MASS,
    POSITIVE_ION_MASS,
    AtmosphericCoagulation,
)
from particula.dynamics.ions.condensation import IonCondensationExclusion
from particula.dynamics.ions.source import IonPairSource
from particula.execution.cohort import SeedCohort
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMolar
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import AVOGADRO_NUMBER
from scipy.stats import beta


@dataclass(frozen=True)
class SurvivalConfig:
    """Physical and numerical parameters in SI units unless explicitly named."""

    boxes: int = 64
    capacity: int = 128
    background_particles: int = 32
    box_volume: float = 3.2e-10
    seed_diameter: float = 3e-9
    background_diameter: float = 100e-9
    target_diameter: float = 50e-9
    material_density: float = 1400.0
    vapor_molar_mass: float = 0.2
    vapor_diffusion: float = 5e-6
    vapor_pressure: float = 1e-12
    vapor_concentration: float = 2e-10
    vapor_refresh_seconds: float = 10.0
    duration: float = 20000.0
    time_step: float = 1.0
    cleanup_seconds: float = 10.0
    ion_pairs_cm3_s: float = 0.0
    omitted: str = "reject"
    root_seed: int = 317
    coagulation_first: bool = False

    def validate(self):
        """Keep whole physical intervals and immutable population geometry."""
        for name in ("boxes", "capacity", "background_particles", "root_seed"):
            value = getattr(self, name)
            minimum = 0 if name in ("background_particles", "root_seed") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.coagulation_first) is not bool:
            raise ValueError("coagulation_first must be a boolean")
        if self.capacity < self.background_particles + 1:
            raise ValueError(
                "capacity must hold the initial background and seed"
            )
        for name in (
            "box_volume",
            "seed_diameter",
            "background_diameter",
            "target_diameter",
            "material_density",
            "vapor_molar_mass",
            "vapor_diffusion",
            "vapor_pressure",
            "vapor_concentration",
            "vapor_refresh_seconds",
            "duration",
            "time_step",
            "cleanup_seconds",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.ion_pairs_cm3_s) or self.ion_pairs_cm3_s < 0:
            raise ValueError("ion production must be finite and nonnegative")
        if self.omitted not in ("reject", "lower", "upper"):
            raise ValueError("omitted must be reject, lower or upper")
        whole_steps(self.duration, self.vapor_refresh_seconds)
        whole_steps(self.vapor_refresh_seconds, self.time_step)
        whole_steps(self.cleanup_seconds, self.time_step)


def whole_steps(interval, step):
    """Require integral subdivision without silently rounding physical time."""
    if (
        not np.isfinite(interval)
        or not np.isfinite(step)
        or min(interval, step) <= 0
    ):
        raise ValueError("time intervals must be finite and positive")
    count = round(interval / step)
    if count < 1 or not np.isclose(count * step, interval, rtol=1e-12, atol=0):
        raise ValueError("time intervals must contain whole positive steps")
    return count


def build_survival_case(config):
    """Construct five material lanes and one explicitly mapped vapor lane.

    Material order is condensed vapor, original seed, original background,
    positive ion and negative ion. The extremely low volatility vapor uses a
    finite reference pressure [Pa] and unit accommodation. Periodic gas refresh
    is an external forcing operation accounted separately by the study runner.
    """
    config.validate()
    b, n = config.boxes, config.capacity
    masses = np.zeros((b, n, 5))
    masses[:, 0, 1] = (
        config.material_density * np.pi / 6 * config.seed_diameter**3
    )
    masses[:, 1 : config.background_particles + 1, 2] = (
        config.material_density * np.pi / 6 * config.background_diameter**3
    )
    counts = np.zeros((b, n))
    counts[:, : config.background_particles + 1] = 1
    ion_volume = np.pi / 6 * (0.5e-9) ** 3
    density = np.array(
        [
            config.material_density,
            config.material_density,
            config.material_density,
            POSITIVE_ION_MASS / ion_volume,
            NEGATIVE_ION_MASS / ion_volume,
        ]
    )
    aerosol = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            np.zeros((b, n)),
            density,
            np.full(b, config.box_volume),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["controlled vapor"],
            np.array([config.vapor_molar_mass]),
            np.full((b, 1), config.vapor_concentration),
            np.array([True]),
        ),
        environment=EnvironmentData(
            np.full(b, 298.15),
            np.full(b, 101325.0),
            np.zeros((b, 1)),
        ),
    )
    condensation = MassCondensation(
        CondensationIsothermal(
            molar_mass=aerosol.gas.molar_mass,
            diffusion_coefficient=config.vapor_diffusion,
            accommodation_coefficient=1.0,
            activity_strategy=ActivityIdealMolar(
                np.array(
                    [
                        config.vapor_molar_mass,
                        config.vapor_molar_mass,
                        config.vapor_molar_mass,
                        POSITIVE_ION_MASS * AVOGADRO_NUMBER,
                        NEGATIVE_ION_MASS * AVOGADRO_NUMBER,
                    ]
                )
            ),
            surface_strategy=SurfaceStrategyVolume(
                density=config.material_density,
                surface_tension=0.03,
            ),
            vapor_pressure_strategy=ConstantVaporPressureStrategy(
                config.vapor_pressure
            ),
            species_map=CondensationSpeciesMap(
                tuple(aerosol.gas.name), ((0, 0),)
            ),
        )
    )
    coagulation = AtmosphericCoagulation(3, 4, omitted=config.omitted)
    first, second = (
        (coagulation, condensation)
        if config.coagulation_first
        else (condensation, coagulation)
    )
    process = (
        IonPairSource(
            config.ion_pairs_cm3_s * 1e6,
            3,
            4,
            POSITIVE_ION_MASS,
            NEGATIVE_ION_MASS,
        )
        | first
        | second
        | IonNeutralCleanup(
            3,
            4,
            interval_steps=whole_steps(
                config.cleanup_seconds, config.time_step
            ),
        )
        | IonCondensationExclusion(3, 4)
        | SeedCohort(1, 2, config.target_diameter)
    )
    return aerosol, process


def _confidence_tail(confidence):
    """Validate coverage and return the equal-tail probability."""
    if (
        isinstance(confidence, (bool, np.bool_))
        or not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and between zero and one")
    return (1 - confidence) / 2


def success_interval(reached, total, *, confidence=0.95):
    """Exact binomial interval and a one-sided zero-event upper bound.

    Confidence describes coverage of each interval separately. One initial
    seed per independent box is required for the binomial interpretation.
    Uses Clopper and Pearson (1934), doi:10.1093/biomet/26.4.404.
    """
    if type(reached) is not int or type(total) is not int:
        raise ValueError("event counts must be integers")
    if total < 1 or not 0 <= reached <= total:
        raise ValueError("event counts require 0 <= reached <= total")
    tail = _confidence_tail(confidence)
    low = 0.0 if reached == 0 else beta.ppf(tail, reached, total - reached + 1)
    high = (
        1.0
        if reached == total
        else beta.ppf(1 - tail, reached + 1, total - reached)
    )
    zero_upper = (
        -np.expm1(np.log(1 - confidence) / total) if reached == 0 else None
    )
    return float(low), float(high), zero_upper


def difference_interval(
    first_reached, first_total, second_reached, second_total, *, confidence=0.95
):
    """Conservative interval for the second success probability minus first.

    Two exact marginal intervals at (1 + confidence)/2 give at least the
    requested joint coverage by the union bound. This remains valid when the
    two runs share random streams; independence between runs is not assumed.
    A convergence claim requires the entire difference interval, rather than
    only its point estimate, to lie inside the predeclared tolerance.
    """
    joint = 1 - _confidence_tail(confidence)
    low_a, high_a, _ = success_interval(
        first_reached, first_total, confidence=joint
    )
    low_b, high_b, _ = success_interval(
        second_reached, second_total, confidence=joint
    )
    return low_b - high_a, high_b - low_a


def run_survival(config, *, device="cuda:0", capture=True):
    """Advance physical steps between explicit vapor refreshes and reductions.

    Default execution requires CUDA and never falls back. ``device='cpu'``
    with ``capture=False`` is a separately requested regression target.
    Return compact cohort time series and budgets, not particle-state dumps.
    """
    initial, process = build_survival_case(config)
    segments = whole_steps(config.duration, config.vapor_refresh_seconds)
    steps = whole_steps(config.vapor_refresh_seconds, config.time_step)
    supplied = initial.particles.masses.sum(axis=1)
    supplied[:, 0] += config.vapor_concentration * config.box_volume
    gas_target = initial.gas.concentration.copy()
    times, outcomes = [], []
    compute_seconds, forcing_seconds = 0.0, 0.0
    max_budget_error, max_occupancy = 0.0, 0
    start = perf_counter()
    with process.prepare(
        initial,
        device=device,
        time_step=config.time_step,
        root_seed=config.root_seed,
    ) as simulation:
        if capture:
            simulation.capture()
        simulation.synchronize()
        preparation_seconds = perf_counter() - start
        for segment in range(segments + 1):
            if segment:
                start = perf_counter()
                simulation.run(steps)
                simulation.synchronize()
                compute_seconds += perf_counter() - start
            data = simulation.diagnostics()
            source = simulation.ion_source_diagnostics()
            cleanup = simulation.ion_cleanup_diagnostics()
            cohort = simulation.cohort_diagnostics()
            particle_mass = data.particle_mass_concentration * config.box_volume
            accounted = particle_mass + cleanup["exported_mass"]
            accounted[:, 0] += data.gas_concentration[:, 0] * config.box_volume
            expected = supplied + source["injected_mass"]
            np.testing.assert_allclose(accounted, expected, rtol=2e-11, atol=0)
            occupied = expected > 0
            max_budget_error = max(
                max_budget_error,
                np.max(np.abs(accounted[occupied] / expected[occupied] - 1)),
            )
            np.testing.assert_array_equal(data.charge_concentration, 0)
            np.testing.assert_array_equal(
                cohort["reached"] + cohort["captured"] + cohort["unresolved"],
                np.ones(config.boxes),
            )
            times.append(data.simulated_time)
            outcomes.append(
                [
                    int(cohort[key].sum())
                    for key in ("reached", "captured", "unresolved")
                ]
            )
            max_occupancy = max(max_occupancy, int(data.active_slots.max()))
            if segment < segments:
                start = perf_counter()
                change = gas_target - data.gas_concentration
                simulation.update_gas(gas_target)
                supplied[:, 0] += change[:, 0] * config.box_volume
                forcing_seconds += perf_counter() - start
    counts = np.asarray(outcomes, dtype=np.int64)
    return {
        "config": asdict(config),
        "device": device,
        "time": np.asarray(times),
        "outcomes": counts,
        "success_interval95": success_interval(
            int(counts[-1, 0]), config.boxes
        ),
        "maximum_mass_residual": max_budget_error,
        "maximum_observed_slots": max_occupancy,
        "preparation_seconds": preparation_seconds,
        "synchronized_compute_seconds": compute_seconds,
        "forcing_seconds": forcing_seconds,
    }
