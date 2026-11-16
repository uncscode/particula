"""Measure native adaptive accuracy, estimator behavior and CUDA control cost.

Uses the established growth/equilibrium/depletion cases and independent radius
references. CPU RK23/ROS2 and actual CUDA RK23 are compared over the same
physical intervals. Timings distinguish setup, accepted-step work (including
rejections and scalar readbacks), and full observation downloads. This is a
bounded assessment, not a general speedup or measured peak-memory claim.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import warp as wp
from particula import _cpu_batch
from particula.aerosol import Aerosol
from particula.dynamics import CondensationIsothermal
from particula.execution.condensation_ode import CondensationODE
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scipy.integrate import solve_ivp
from scripts import assess_native_condensation as model

VISCOSITY = (
    1.716e-5
    * (model.TEMPERATURE / 273.15) ** 1.5
    * (273.15 + 110.4)
    / (model.TEMPERATURE + 110.4)
)


def initial(equilibrium, boxes=1):
    """Use the native reference case, with heterogeneous gas if batched."""
    particle, gas, strategy = model.initial(CondensationIsothermal, equilibrium)
    if boxes > 1:
        volumes = np.linspace(0.1, 0.9, boxes)
        particle = ParticleData(
            masses=np.repeat(particle.masses, boxes, axis=0),
            concentration=np.column_stack(
                (model.NUMBER * volumes, np.zeros(boxes))
            ),
            charge=np.zeros((boxes, 2)),
            density=particle.density.copy(),
            volume=volumes,
            distribution_type="particle_resolved",
        )
        gas = GasData(
            name=gas.name.copy(),
            molar_mass=gas.molar_mass.copy(),
            concentration=np.repeat(gas.concentration, boxes, axis=0),
            partitioning=gas.partitioning.copy(),
        )
        gas.concentration[:, 1] *= np.geomspace(0.1, 10, boxes)
    environment = EnvironmentData(
        temperature=np.full(boxes, model.TEMPERATURE),
        pressure=np.full(boxes, model.PRESSURE),
        saturation_ratio=np.zeros((boxes, 2)),
    )
    return Aerosol(
        particles=particle, gas=gas, environment=environment
    ), strategy


def local_reference(old_radius, old_gas, duration, equilibrium):
    """Cross-check one-step radius flow from the numerical pre-step state."""
    mean_free_path = (2 * VISCOSITY / model.PRESSURE) / np.sqrt(
        8 * model.MOLAR_MASS / (np.pi * model.GAS_CONSTANT * model.TEMPERATURE)
    )
    total = old_gas + model.NUMBER * model.MASS * old_radius**3

    def rate(time, radius):
        x = radius[0]
        kn = mean_free_path / (model.RADIUS * x)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        vapor = total - model.NUMBER * model.MASS * x**3
        surface = equilibrium * np.exp(
            2
            * 0.072
            * model.MOLAR_MASS
            / (
                model.GAS_CONSTANT
                * model.TEMPERATURE
                * model.DENSITY
                * model.RADIUS
                * x
            )
        )
        return [
            model.DIFFUSION
            * correction
            * (vapor - surface)
            / (model.DENSITY * model.RADIUS**2 * x)
        ]

    solutions = []
    for method in ("DOP853", "Radau"):
        result = solve_ivp(
            rate,
            (0, duration),
            [old_radius],
            method=method,
            rtol=2e-12,
            atol=2e-14,
        )
        if not result.success:
            raise RuntimeError(result.message)
        radius = result.y[0, -1]
        mass = model.MASS * radius**3
        solutions.append(np.array([mass, total - model.NUMBER * mass]))
    return solutions[0], np.abs(solutions[0] - solutions[1])


def measure(kind, duration, equilibrium, tolerance, device):
    """Retain every accepted point and sampled local-error comparisons."""
    state, strategy = initial(equilibrium)
    options = dict(
        rtol=tolerance,
        particle_atol=model.MASS * tolerance * 1e-3,
        gas_atol=model.VAPOR * tolerance * 1e-3,
        first_step=0.1,
        max_step=duration,
    )
    start = perf_counter()
    owner = (
        GPUCondensationODE(state, strategy, device=device, **options)
        if kind == "cuda-rk23"
        else CondensationODE(state, strategy, method=kind, **options)
    )
    setup_seconds = perf_counter() - start
    counter = [0]
    if kind != "cuda-rk23":
        raw_rate = owner._rate

        def counted_rate(time, values):
            counter[0] += 1
            return raw_rate(time, values)

        owner._rate = counted_rate
        owner.problem = replace(owner.problem, rate=counted_rate)
        owner.solver.problem = owner.problem
    times, radii, vapors, estimates = [0.0], [1.0], [model.VAPOR], []
    step_seconds, observation_seconds = 0.0, 0.0
    while owner.time < duration:
        if len(times) > 50000 or step_seconds > 180:
            raise RuntimeError("assessment work limit exceeded")
        start = perf_counter()
        owner.step(duration)
        step_seconds += perf_counter() - start
        start = perf_counter()
        current = owner.snapshot()
        observation_seconds += perf_counter() - start
        times.append(owner.time)
        radii.append(current.particles.radii[0, 0] / model.RADIUS)
        vapors.append(current.gas.concentration[0, 1])
        solver = owner if kind == "cuda-rk23" else owner.solver
        estimates.append(solver.last_error)
    expected, reference_difference = model.reference(
        duration, equilibrium, np.asarray(times), viscosity=VISCOSITY
    )
    local_norms, oracle_norms, ratios = [], [], []
    for index in np.unique(
        np.linspace(1, len(times) - 1, min(16, len(times) - 1)).astype(int)
    ):
        exact, uncertainty = local_reference(
            radii[index - 1],
            vapors[index - 1],
            times[index] - times[index - 1],
            equilibrium,
        )
        before = np.array(
            [model.MASS * radii[index - 1] ** 3, vapors[index - 1]]
        )
        actual = np.array([model.MASS * radii[index] ** 3, vapors[index]])
        weights = np.array(
            [options["particle_atol"], options["gas_atol"]]
        ) + tolerance * np.maximum(np.abs(before), np.abs(actual))
        norm = float(np.max(np.abs(actual - exact) / weights))
        oracle_norms.append(float(np.max(uncertainty / weights)))
        local_norms.append(norm)
        if estimates[index - 1] > 1e-10:
            ratios.append(norm / estimates[index - 1])
    return {
        "method": kind,
        "rtol": tolerance,
        "setup_seconds": setup_seconds,
        "step_seconds": step_seconds,
        "observation_seconds": observation_seconds,
        "reference_difference": reference_difference,
        "work": asdict(solver.statistics),
        "physical_rate_evaluations": (
            solver.statistics.rate_calls if kind == "cuda-rk23" else counter[0]
        ),
        "control_scalar_readbacks": getattr(owner, "control_readbacks", 0),
        "allocated_device_bytes": getattr(owner, "device_storage_bytes", None),
        "sampled_local_true_error_max": max(local_norms),
        "sampled_local_reference_uncertainty_max": max(oracle_norms),
        "sampled_true_to_estimated_error_max": max(ratios) if ratios else None,
        "local_samples": len(local_norms),
        **model.metrics(
            np.asarray(radii), np.asarray(vapors), expected, equilibrium
        ),
    }


def ensemble(boxes, device):
    """Compare one shared controller with independent native box controllers."""
    state, strategy = initial(0.0, boxes)
    reference = CondensationODE(
        state, strategy, rtol=1e-10, particle_atol=1e-28, gas_atol=1e-18
    )
    while reference.time < 0.2:
        reference.step(0.2)
    expected = reference.snapshot()
    rows = []
    for mode in ("shared", "independent"):
        start = perf_counter()
        owners = (
            [GPUCondensationODE(state, strategy, device=device, rtol=1e-6)]
            if mode == "shared"
            else [
                GPUCondensationODE(
                    _cpu_batch.box_view(state, box),
                    strategy,
                    device=device,
                    rtol=1e-6,
                )
                for box in range(boxes)
            ]
        )
        setup_seconds = perf_counter() - start
        start = perf_counter()
        for owner in owners:
            while owner.time < 0.2:
                owner.step(0.2)
        step_seconds = perf_counter() - start
        start = perf_counter()
        outputs = [owner.snapshot() for owner in owners]
        observation_seconds = perf_counter() - start
        masses = np.concatenate([output.particles.masses for output in outputs])
        gases = np.concatenate([output.gas.concentration for output in outputs])
        rows.append(
            {
                "mode": mode,
                "boxes": boxes,
                "setup_seconds": setup_seconds,
                "step_seconds": step_seconds,
                "observation_seconds": observation_seconds,
                "accepted_per_controller": [
                    owner.statistics.accepted for owner in owners
                ],
                "rejected_per_controller": [
                    owner.statistics.rejected for owner in owners
                ],
                "scalar_readbacks": sum(
                    owner.control_readbacks for owner in owners
                ),
                "allocated_device_bytes": sum(
                    owner.device_storage_bytes for owner in owners
                ),
                "maximum_particle_mass_relative_difference": float(
                    np.max(
                        np.abs(
                            masses[:, 0] / expected.particles.masses[:, 0] - 1
                        )
                    )
                ),
                "maximum_vapor_relative_difference": float(
                    np.max(
                        np.abs(
                            gases[:, 1] / expected.gas.concentration[:, 1] - 1
                        )
                    )
                ),
            }
        )
    return rows


def main():
    """Require actual CUDA and save successes and failures without filtering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boxes", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("this assessment requires actual CUDA")
    # Compile all kernels before recording warm step timings.
    state, strategy = initial(0.0)
    GPUCondensationODE(state, strategy, device=device).step(0.001)
    rows = []
    for name, duration, equilibrium in (
        ("smooth_growth", 1.4, 0.0),
        ("equilibrium", 20.0, 1e-6),
        ("near_depletion", 20.0, 0.0),
    ):
        for kind in ("rk23", "ros2", "cuda-rk23"):
            for tolerance in (1e-3, 1e-5, 1e-7):
                row = dict(case=name, method=kind, rtol=tolerance)
                try:
                    row.update(
                        measure(kind, duration, equilibrium, tolerance, device)
                    )
                    row["status"] = "success"
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", reason=str(error))
                rows.append(row)
                print(name, kind, tolerance, row["status"], flush=True)
    # Qualify settings against predeclared *global* observable targets rather
    # than treating the local tolerance as a global guarantee. The baseline
    # grid above deliberately retains the loose-tolerance failures of accuracy.
    targets = []
    for name, equilibrium in (("equilibrium", 1e-6), ("near_depletion", 0.0)):
        for kind in ("rk23", "cuda-rk23"):
            for tolerance, target in ((1e-4, 1e-3), (1e-8, 1e-7)):
                row = dict(
                    case=name, method=kind, rtol=tolerance, target=target
                )
                try:
                    row.update(
                        measure(kind, 20.0, equilibrium, tolerance, device)
                    )
                    row["status"] = "success"
                    row["meets_observable_target"] = (
                        max(
                            row["radius_error"], row["vapor_error_over_initial"]
                        )
                        <= target
                    )
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", reason=str(error))
                targets.append(row)
                print(
                    name, kind, tolerance, "target", row["status"], flush=True
                )
    sources = [
        Path(__file__),
        Path(model.__file__),
        *[
            Path("particula/execution") / name
            for name in (
                "gpu_condensation_ode.py",
                "_condensation_ode_kernels.py",
                "condensation_ode.py",
                "ode.py",
                "prepared_condensation.py",
            )
        ],
        Path("particula/gpu/kernels/condensation.py"),
    ]
    report = {
        "runs": rows,
        "target_checks": targets,
        "ensemble": ensemble(args.boxes, device),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "device": {"alias": device.alias, "name": device.name},
        "python": platform.python_version(),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "viscosity_Pa_s": VISCOSITY,
        "source_sha256": {
            str(
                path.relative_to(Path.cwd()) if path.is_absolute() else path
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "timing_scope": (
            "one warm sample; setup and full observations separate; "
            "step includes rejection and scalar transfers"
        ),
        "memory_scope": (
            "allocated retained device arrays only, "
            "not compiler/context/registers or measured peak memory"
        ),
        "ensemble_scope": (
            "shared CUDA controller vs sequential independent CUDA owners; "
            "independent per-box CUDA kernels are separate prototype evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
