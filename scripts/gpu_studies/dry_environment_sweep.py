"""Controlled dry charged coagulation from surface to mesopause pressures."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from particula import Aerosol
from particula.dynamics import (
    ChargedCoagulationStrategy,
    Coagulation,
    HardSphereKernelStrategy,
)
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scripts.gpu_studies.population_diagnostics import (
    DIAMETER_EDGES,
    population_summary,
)

ALTITUDES_KM = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 86.0]
)


def standard_atmosphere(altitude_km):
    """US76 layer approximation: temperature [K] and pressure [Pa].

    Input is geometric altitude, 0–86 km. Constant mean molar mass gives the
    molecular-scale temperature; it approximates kinetic temperature within
    0.1 K at the 86 km boundary. This is not a local weather observation.
    See NASA-TM-X-74335 and NASA/TM-2005-213659, Appendix A.
    """
    z = np.asarray(altitude_km, dtype=float)
    if not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 86)):
        raise ValueError("geometric altitude must be finite and within 0–86 km")
    height = 6356.766 * z / (6356.766 + z)
    boundaries = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.8521]
    lapse = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]
    temperatures, pressures = [], []
    for h in height.flat:
        t, p = 288.15, 101325.0
        for low, high, gradient in zip(
            boundaries[:-1], boundaries[1:], lapse, strict=True
        ):
            delta = min(max(h - low, 0), high - low)
            next_t = t + gradient * delta
            factor = 1000 * 9.80665 * 0.0289644 / 8.31432
            p *= (
                (next_t / t) ** (-factor / gradient)
                if gradient
                else np.exp(-factor * delta / t)
            )
            t = next_t
            if h <= high:
                break
        temperatures.append(t)
        pressures.append(p)
    return np.array(temperatures).reshape(z.shape), np.array(pressures).reshape(
        z.shape
    )


def dry_conditions():
    """Return sixteen altitudes followed by 2/5/10-atm chambers."""
    t, p = standard_atmosphere(ALTITUDES_KM)
    return np.concatenate((t, [288.15] * 3)), np.concatenate(
        (p, 101325 * np.array([2.0, 5.0, 10.0]))
    )


def dry_labels():
    """Label altitude and chamber cases in numerical order."""
    return [f"{z:g} km" for z in ALTITUDES_KM] + [
        "2 atm chamber",
        "5 atm chamber",
        "10 atm chamber",
    ]


@dataclass(frozen=True)
class DryEnvironmentSweep:
    """Fixed number density isolates environmental effects; parameters in SI."""

    boxes_per_condition: int = 1024
    particles_per_box: int = 128
    number_concentration: float = 1e11
    duration: float = 1000.0
    time_step: float = 0.5
    seed: int = 512

    def validate(self):
        """Reject invalid counts, density and diagnostic schedules."""
        for name in ("boxes_per_condition", "particles_per_box", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        if self.particles_per_box % 4:
            raise ValueError("particles_per_box must be divisible by four")
        for value in (self.number_concentration, self.duration, self.time_step):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "density and times must be positive and finite"
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


def build_dry_sweep(config):
    """Construct matched dry size/charge populations at fixed number density."""
    config.validate()
    t, p = dry_conditions()
    groups = len(t)
    b, n = groups * config.boxes_per_condition, config.particles_per_box
    rng = np.random.default_rng(config.seed)
    radius = np.tile(
        np.clip(
            rng.lognormal(
                np.log(35e-9), np.log(1.6), (config.boxes_per_condition, n)
            ),
            15e-9,
            150e-9,
        ),
        (groups, 1),
    )
    initial = Aerosol(
        particles=ParticleData(
            (1400 * 4 * np.pi / 3 * radius**3)[..., None],
            np.ones((b, n)),
            np.tile([-1.0, 0.0, 0.0, 1.0], (b, n // 4)),
            np.array([1400.0]),
            np.full(b, n / config.number_concentration),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert"],
            np.array([0.0289644]),
            np.zeros((b, 1)),
            np.array([False]),
        ),
        environment=EnvironmentData(
            np.repeat(t, config.boxes_per_condition),
            np.repeat(p, config.boxes_per_condition),
            np.ones((b, 1)),
        ),
    )
    return initial, Coagulation(
        ChargedCoagulationStrategy(
            "particle_resolved", HardSphereKernelStrategy()
        )
    )


def run_dry_sweep(config, *, device="cuda:0", capture=True):
    """Collect distributions and budgets with all conditions resident."""
    import warp as wp

    if str(device).startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    initial, process = build_dry_sweep(config)
    groups = len(dry_labels())
    expected = initial.particles.masses.sum(axis=1)
    rows, compute_seconds, max_error = [], 0.0, 0.0
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
            inventory = (
                data.particle_mass_concentration
                * initial.particles.volume[:, None]
            )
            np.testing.assert_allclose(inventory, expected, rtol=2e-11, atol=0)
            np.testing.assert_array_equal(data.charge_concentration, 0)
            max_error = max(
                max_error, float(np.max(np.abs(inventory / expected - 1)))
            )
            final = sim.snapshot()
            population = population_summary(final, groups)
            np.testing.assert_array_equal(population["net_charge"], 0)
            np.testing.assert_array_equal(
                population["number"],
                data.active_slots.reshape(groups, -1).sum(axis=1),
            )
            rows.append({"time": float(data.simulated_time), **population})
    t, p = dry_conditions()
    return {
        "config": asdict(config),
        "labels": dry_labels(),
        "temperature": t.tolist(),
        "pressure": p.tolist(),
        "altitude_km": ALTITUDES_KM.tolist(),
        "history": rows,
        "diameter_edges": DIAMETER_EDGES.tolist(),
        "initial_particles": int(initial.particles.concentration.sum()),
        "maximum_mass_relative_error": max_error,
        "device": wp.get_device(device).name,
        "preparation_seconds": preparation_seconds,
        "compute_seconds": compute_seconds,
    }, final
