"""Closed charged-aerosol water-growth sweep with independent box budgets.

All physical parameters are illustrative. Vapor diffusivity is held fixed at
2e-5 m2/s: the prepared API accepts species constants, not per-box transport
laws. Temperature and pressure still enter the supported collision and
condensation formulas. These boxes are controlled environments, not altitudes.
"""

from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter

import numpy as np
from particula import Aerosol
from particula.dynamics import (
    ChargedCoagulationStrategy,
    Coagulation,
    HardSphereKernelStrategy,
    MassCondensation,
)
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
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityKappaParameter
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from scripts.gpu_studies.population_diagnostics import (
    DIAMETER_EDGES,
    diagnostic_steps,
    population_summary,
)


@dataclass(frozen=True)
class EnvironmentSweep:
    """Sixteen conditions; default layout starts with 2,097,152 particles."""

    boxes_per_condition: int = 1024
    particles_per_box: int = 128
    time_step: float = 1e-6
    duration: float = 0.003
    seed: int = 411
    coagulation: bool = True
    ions: bool = False
    ion_pairs_per_box: int = 16
    ion_pair_source: float = 1e14

    def validate(self):  # noqa: C901
        """Reject ambiguous layouts and fractional timesteps."""
        for key in ("boxes_per_condition", "particles_per_box", "seed"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"invalid {key}")
        if self.particles_per_box % 4:
            raise ValueError("particles_per_box must be divisible by four")
        if type(self.coagulation) is not bool:
            raise ValueError("coagulation must be boolean")
        if type(self.ions) is not bool or (self.ions and not self.coagulation):
            raise ValueError("ions requires coagulation")
        if (
            type(self.ion_pairs_per_box) is not int
            or self.ion_pairs_per_box <= 0
        ):
            raise ValueError("ion_pairs_per_box must be positive integer")
        if not np.isfinite(self.ion_pair_source) or self.ion_pair_source < 0:
            raise ValueError("ion_pair_source must be finite and nonnegative")
        for value in (self.time_step, self.duration):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "positive finite duration and timestep required"
                )
        steps = round(self.duration / self.time_step)
        if (
            steps < 10
            or steps % 10
            or not np.isclose(
                steps * self.time_step, self.duration, rtol=1e-12, atol=0
            )
        ):
            raise ValueError("duration must contain a multiple of ten steps")


def conditions():
    """Return temperature [K], pressure [Pa] and initial relative humidity."""
    return np.array(
        list(
            product((278.15, 298.15), (70000.0, 101325.0), (0.4, 0.6, 0.8, 0.9))
        )
    )


def saturation_pressure(temperature):
    """Independent Buck expression for liquid-water saturation pressure [Pa]."""
    c = np.asarray(temperature) - 273.15
    return 611.21 * np.exp((18.678 - c / 234.5) * c / (257.14 + c))


def build_environment_sweep(config):
    """Construct matched populations and a closed two-material model."""
    config.validate()
    grid = np.repeat(conditions(), config.boxes_per_condition, axis=0)
    b, n = len(grid), config.particles_per_box
    rng = np.random.default_rng(config.seed)
    # Match initial particle populations across all sixteen conditions.
    dry_radius = np.tile(
        np.clip(
            rng.lognormal(
                np.log(35e-9), np.log(1.6), (config.boxes_per_condition, n)
            ),
            15e-9,
            150e-9,
        ),
        (16, 1),
    )
    dry_volume = 4 * np.pi / 3 * dry_radius**3
    masses = np.stack((0.2 * dry_volume * 1000, dry_volume * 1400), axis=-1)
    charge = np.tile(np.array([-1.0, 0.0, 0.0, 1.0]), (b, n // 4))
    volume = np.full(b, n / 5e13)
    density = np.array([1000.0, 1400.0])
    vapor = (
        grid[:, 2]
        * saturation_pressure(grid[:, 0])
        * 0.018
        / (GAS_CONSTANT * grid[:, 0])
    )
    counts = np.ones((b, n))
    if config.ions:
        ni = config.ion_pairs_per_box
        # Fixed capacity includes room for source draws; never resize on device.
        slots = n + 2 * ni + 32
        expanded = np.zeros((b, slots, 4))
        expanded[:, :n, :2] = masses
        expanded[:, n : n + ni, 2] = POSITIVE_ION_MASS
        expanded[:, n + ni : n + 2 * ni, 3] = NEGATIVE_ION_MASS
        masses = expanded
        charge = np.pad(charge, ((0, 0), (0, slots - n)))
        charge[:, n : n + ni] = 1
        charge[:, n + ni : n + 2 * ni] = -1
        counts = np.zeros((b, slots))
        counts[:, : n + 2 * ni] = 1
        ion_volume = np.pi / 6 * (0.5e-9) ** 3
        density = np.array(
            [
                1000.0,
                1400.0,
                POSITIVE_ION_MASS / ion_volume,
                NEGATIVE_ION_MASS / ion_volume,
            ]
        )
    initial = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            charge,
            density,
            volume,
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["water"], np.array([0.018]), vapor[:, None], np.array([True])
        ),
        environment=EnvironmentData(grid[:, 0], grid[:, 1], np.ones((b, 1))),
    )
    condensation = MassCondensation(
        CondensationIsothermal(
            molar_mass=np.array([0.018]),
            diffusion_coefficient=2e-5,
            accommodation_coefficient=1.0,
            activity_strategy=ActivityKappaParameter(
                kappa=np.array(
                    [0.0, 0.3, 0.0, 0.0] if config.ions else [0.0, 0.3]
                ),
                density=density,
                molar_mass=np.array(
                    [0.018, 0.2, 0.150, 0.090] if config.ions else [0.018, 0.2]
                ),
                water_index=0,
            ),
            surface_strategy=SurfaceStrategyVolume(
                surface_tension=0.072, density=1000.0
            ),
            vapor_pressure_strategy=WaterBuckStrategy(),
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
    )
    process = condensation
    if config.ions:
        process = (
            IonPairSource(
                config.ion_pair_source,
                2,
                3,
                POSITIVE_ION_MASS,
                NEGATIVE_ION_MASS,
            )
            | condensation
            | AtmosphericCoagulation(2, 3, attachment_model="mobility")
            | IonNeutralCleanup(2, 3, interval_steps=10)
            | IonCondensationExclusion(2, 3)
        )
    elif config.coagulation:
        process = condensation | Coagulation(
            ChargedCoagulationStrategy(
                "particle_resolved", HardSphereKernelStrategy()
            )
        )
    return initial, process


def run_environment_sweep(config, *, device="cuda:0", capture=True):
    """Sample early growth and full-duration dynamics; check every box.

    CPU execution is an explicit test option, never an automatic fallback.
    The main workload keeps all sixteen environments resident simultaneously.
    """
    import warp as wp

    if str(device).startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    initial, process = build_environment_sweep(config)
    p = initial.particles
    expected = p.masses.sum(axis=1)
    expected[:, 0] += initial.gas.concentration[:, 0] * p.volume
    grid = np.repeat(conditions(), config.boxes_per_condition, axis=0)
    rows = []
    max_error = 0.0
    compute_seconds = 0.0
    start = perf_counter()
    with process.prepare(
        initial,
        device=device,
        time_step=config.time_step,
        root_seed=config.seed,
    ) as sim:
        if capture:
            sim.capture()
        sim.synchronize()
        preparation_seconds = perf_counter() - start
        previous_step = 0
        for step in diagnostic_steps(config.duration, config.time_step):
            if step:
                start = perf_counter()
                sim.run(step - previous_step)
                sim.synchronize()
                compute_seconds += perf_counter() - start
            previous_step = step
            data = sim.diagnostics()
            inventory = data.particle_mass_concentration * p.volume[:, None]
            inventory[:, 0] += data.gas_concentration[:, 0] * p.volume
            target = expected
            if config.ions:
                source = sim.ion_source_diagnostics()
                cleanup = sim.ion_cleanup_diagnostics()
                inventory += cleanup["exported_mass"]
                target = expected + source["injected_mass"]
            np.testing.assert_allclose(inventory, target, rtol=2e-11, atol=0)
            np.testing.assert_array_equal(data.charge_concentration, 0)
            np.testing.assert_equal(np.all(data.gas_concentration >= 0), True)
            max_error = max(
                max_error, float(np.max(np.abs(inventory / target - 1)))
            )
            rh = (
                data.gas_concentration[:, 0]
                * GAS_CONSTANT
                * grid[:, 0]
                / (0.018 * saturation_pressure(grid[:, 0]))
            )
            water = data.particle_mass_concentration[:, 0] * p.volume
            counts = data.active_slots
            final = sim.snapshot()
            population = population_summary(
                final, 16, material=1 if config.ions else None
            )
            ion_history = {}
            if config.ions:
                fp = final.particles
                aerosol_mask = fp.masses[..., 1] > 0
                free_plus = (
                    (~aerosol_mask) & (fp.charge == 1) & (fp.concentration > 0)
                )
                free_minus = (
                    (~aerosol_mask) & (fp.charge == -1) & (fp.concentration > 0)
                )
                neutral = (
                    (~aerosol_mask) & (fp.charge == 0) & (fp.concentration > 0)
                )
                recombined = neutral.sum(axis=1) + cleanup["exported_particles"]
                injected = source["injected_pairs"]
                # Ion balances identify attachments independently of mergers.
                attached_plus = (
                    config.ion_pairs_per_box
                    + injected
                    - free_plus.sum(axis=1)
                    - recombined
                )
                attached_minus = (
                    config.ion_pairs_per_box
                    + injected
                    - free_minus.sum(axis=1)
                    - recombined
                )
                for lane, mass, attached in [
                    (2, POSITIVE_ION_MASS, attached_plus),
                    (3, NEGATIVE_ION_MASS, attached_minus),
                ]:
                    np.testing.assert_allclose(
                        (fp.masses[..., lane] * aerosol_mask).sum(axis=1)
                        / mass,
                        attached,
                        atol=1e-10,
                        rtol=1e-12,
                    )
                for key, values in {
                    "free_positive": free_plus.sum(axis=1),
                    "free_negative": free_minus.sum(axis=1),
                    "recombination": recombined,
                    "attachment_positive": attached_plus,
                    "attachment_negative": attached_minus,
                    "injected_pairs": injected,
                }.items():
                    ion_history[key] = (
                        values.reshape(16, -1).sum(axis=1).tolist()
                    )
            else:
                np.testing.assert_array_equal(
                    population["number"], counts.reshape(16, -1).sum(axis=1)
                )
                np.testing.assert_array_equal(population["net_charge"], 0)
            rows.append(
                {
                    "time": float(data.simulated_time),
                    **population,
                    **ion_history,
                    "count": population["number"],
                    "water_mass": water.reshape(16, -1).sum(axis=1).tolist(),
                    "rh": rh.reshape(16, -1).mean(axis=1).tolist(),
                }
            )
    np.testing.assert_equal(np.all(np.isfinite(final.particles.masses)), True)
    np.testing.assert_equal(np.all(final.particles.masses >= 0), True)
    report = {
        "config": asdict(config),
        "conditions": conditions().tolist(),
        "initial_particles": int(p.concentration.sum()),
        "history": rows,
        "diameter_edges": DIAMETER_EDGES.tolist(),
        "maximum_mass_relative_error": max_error,
        "device": wp.get_device(device).name,
        "preparation_seconds": preparation_seconds,
        "compute_seconds": compute_seconds,
    }
    return report, final
