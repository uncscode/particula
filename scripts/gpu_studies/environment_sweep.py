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
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityKappaParameter
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT


@dataclass(frozen=True)
class EnvironmentSweep:
    """Sixteen conditions; default layout starts with 2,097,152 particles."""

    boxes_per_condition: int = 1024
    particles_per_box: int = 128
    time_step: float = 1e-6
    duration: float = 0.001
    seed: int = 411
    coagulation: bool = True

    def validate(self):
        """Reject ambiguous layouts and fractional timesteps."""
        for key in ("boxes_per_condition", "particles_per_box", "seed"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"invalid {key}")
        if self.particles_per_box % 4:
            raise ValueError("particles_per_box must be divisible by four")
        if type(self.coagulation) is not bool:
            raise ValueError("coagulation must be boolean")
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
    initial = Aerosol(
        particles=ParticleData(
            masses,
            np.ones((b, n)),
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
                kappa=np.array([0.0, 0.3]),
                density=density,
                molar_mass=np.array([0.018, 0.2]),
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
    if config.coagulation:
        process = condensation | Coagulation(
            ChargedCoagulationStrategy(
                "particle_resolved", HardSphereKernelStrategy()
            )
        )
    return initial, process


def run_environment_sweep(config, *, device="cuda:0", capture=True):
    """Run ten diagnostic intervals and check budgets in every box.

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
        for interval in range(11):
            if interval:
                start = perf_counter()
                sim.run(round(config.duration / config.time_step) // 10)
                sim.synchronize()
                compute_seconds += perf_counter() - start
            data = sim.diagnostics()
            inventory = data.particle_mass_concentration * p.volume[:, None]
            inventory[:, 0] += data.gas_concentration[:, 0] * p.volume
            np.testing.assert_allclose(inventory, expected, rtol=2e-11, atol=0)
            np.testing.assert_array_equal(data.charge_concentration, 0)
            np.testing.assert_equal(np.all(data.gas_concentration >= 0), True)
            max_error = max(
                max_error, float(np.max(np.abs(inventory / expected - 1)))
            )
            rh = (
                data.gas_concentration[:, 0]
                * GAS_CONSTANT
                * grid[:, 0]
                / (0.018 * saturation_pressure(grid[:, 0]))
            )
            water = data.particle_mass_concentration[:, 0] * p.volume
            counts = data.active_slots
            rows.append(
                {
                    "time": float(data.simulated_time),
                    "count": counts.reshape(16, -1).sum(axis=1).tolist(),
                    "water_mass": water.reshape(16, -1).sum(axis=1).tolist(),
                    "rh": rh.reshape(16, -1).mean(axis=1).tolist(),
                }
            )
        final = sim.snapshot()
    np.testing.assert_equal(np.all(np.isfinite(final.particles.masses)), True)
    np.testing.assert_equal(np.all(final.particles.masses >= 0), True)
    report = {
        "config": asdict(config),
        "conditions": conditions().tolist(),
        "initial_particles": int(p.concentration.sum()),
        "history": rows,
        "maximum_mass_relative_error": max_error,
        "device": wp.get_device(device).name,
        "preparation_seconds": preparation_seconds,
        "compute_seconds": compute_seconds,
    }
    return report, final
