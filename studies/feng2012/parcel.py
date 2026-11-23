"""Phase disappearance on an ascending parcel with a finite water reservoir."""

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from particula.dynamics.multivapor_parcel import (
    CP,
    RD,
    MultiVaporParcelInitialState,
    ParcelVapor,
    cpu_multivapor_reference,
)
from particula.dynamics.parcel import ParcelROS3
from particula.particles import ActivityFactory
from particula.util.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula_aiomfac import MixtureSpec

TIMES = np.array([0.0, 0.005, 0.015, 0.035, 0.075])
UPDRAFT = 2.0


def initial_state():
    """Make one 2 µm wet class; an unoccupied slot tests masking."""
    groups = {
        "water": {16: 1},
        "alcohol": {141: 2, 151: 1, 153: 1},
        "salt": {204: 2, 261: 1},
    }
    strategy = ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(groups),
            "material_mapping": dict(enumerate(groups)),
            "phase_mode": "stable",
        },
    )
    water = 0.515
    fractions = np.array([water, (1 - water) * 0.7, (1 - water) * 0.3])
    mass = (4 * np.pi / 3) * (2e-6) ** 3 * 1000 * fractions
    pressure = 101325.0
    p_water = 0.865 * 4246.0
    mixing_ratio = (
        0.01801528 / MOLECULAR_WEIGHT_AIR * p_water / (pressure - p_water)
    )
    return MultiVaporParcelInitialState(
        masses=np.stack([mass, np.zeros(3)]),
        number=np.array([1e8, 0.0]),
        vapor=np.array([mixing_ratio]),
        material_names=tuple(groups),
        density=np.full(3, 1000.0),
        vapors=(
            ParcelVapor("water", 0, 0.01801528, 4246.0, 303.15, 2.5e6, 2.11e-5),
        ),
        activity_strategy=strategy,
        pressure=pressure,
        temperature=303.15,
    )


def summarize(state, masses, temperatures, pressures, vapors, device):
    """Compute phase status, common observables and a separate water balance."""
    masses = np.stack(masses)
    np.testing.assert_array_equal(masses[:, 1], np.zeros((len(TIMES), 3)))
    np.testing.assert_array_equal(
        masses[:, 0, 1:], np.tile(state.masses[0, 1:], (len(TIMES), 1))
    )
    wet = masses[:, 0, :]
    vapor = np.asarray(vapors).ravel()
    inventory = state.number[0] * wet[:, 0] / state.dry_air_mass + vapor
    phases = [
        state.activity_strategy.phase_equilibrium(m, t, execution=device)
        for m, t in zip(wet, temperatures, strict=True)
    ]
    shares = [
        0.0 if p.beta_masses is None else p.beta_masses.sum() / m.sum()
        for p, m in zip(phases, wet, strict=True)
    ]
    np.testing.assert_array_equal(
        np.array(shares) > 0, [True, True, False, False, False]
    )
    np.testing.assert_allclose(
        inventory / inventory[0], 1.0, rtol=0, atol=1e-10
    )
    temperatures, pressures = np.array(temperatures), np.array(pressures)
    rw = GAS_CONSTANT / state.vapors[0].molar_mass
    volume = (RD + rw * vapor) * temperatures / pressures
    work = np.r_[
        0.0, np.cumsum((volume[:-1] + volume[1:]) * np.diff(pressures) / 2)
    ]
    latent = (
        (wet[:, 0] - wet[0, 0])
        * state.number[0]
        / state.dry_air_mass
        * state.vapors[0].latent_heat
    )
    energy_error = temperatures - temperatures[0] - (work + latent) / CP
    np.testing.assert_allclose(energy_error, 0.0, atol=1e-6, rtol=0)
    return {
        "time_s": TIMES.tolist(),
        "max_first_law_error_k": float(abs(energy_error).max()),
        "water_fraction": (wet[:, 0] / wet.sum(axis=1)).tolist(),
        "beta_mass_share": shares,
        "height_m": (UPDRAFT * TIMES).tolist(),
        "pressure_pa": list(pressures),
        "temperature_k": list(temperatures),
        "vapor_kg_per_kg_dry_air": vapor.tolist(),
        "max_relative_water_inventory_error": float(
            abs(inventory / inventory[0] - 1).max()
        ),
    }


def cpu(state):
    """Run the independent Radau trajectory at two tolerances."""
    out = {"processor": platform.machine()}
    for tol in [2e-5, 2e-6]:
        start = perf_counter()
        result = cpu_multivapor_reference(state, TIMES, UPDRAFT, rtol=tol)
        elapsed = perf_counter() - start
        meta = summarize(
            state,
            result.masses,
            result.temperature,
            result.pressure,
            result.vapor,
            "cpu",
        )
        out[str(tol)] = dict(meta, trajectory_wall_seconds=elapsed)
    return out


def gpu(state):
    """Replay device ROS3 and serialize an intermediate boundary."""
    import warp as wp
    from particula.execution.multivapor_parcel import (
        ParcelReplayCheckpoint,
        ResidentMultiVaporParcel,
    )

    def owner(tol):
        return ResidentMultiVaporParcel(
            state,
            device="cuda:0",
            adaptive=ParcelROS3(rtol=tol),
            timestep=0.001,
            updraft=UPDRAFT,
        )

    device = wp.get_device("cuda:0")
    out = {"device": device.name, "warp_version": wp.__version__}
    for tol in [2e-5, 2e-6]:
        start = perf_counter()
        run = owner(tol)
        wp.synchronize_device(device)
        prepare = perf_counter() - start
        states = [
            (state.pressure, state.temperature, state.vapor, state.masses)
        ]
        elapsed = 0.0
        for k, t in enumerate(TIMES[1:], start=1):
            start = perf_counter()
            states.append(run.advance(t))
            elapsed += perf_counter() - start
            if k == 1:
                boundary = run.checkpoint()
        endpoint = run.checkpoint()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "parcel.zip"
            boundary.save(path)
            resumed = owner(tol)
            resumed.restore(ParcelReplayCheckpoint.load(path))
            for t in TIMES[2:]:
                resumed.advance(t)
            after = resumed.checkpoint()
        np.testing.assert_array_equal(after.coordinates, endpoint.coordinates)
        np.testing.assert_array_equal(after.counters, endpoint.counters)
        p, temp, q, m = zip(*states, strict=True)
        meta = summarize(state, m, temp, p, q, "cuda")
        out[str(tol)] = dict(
            meta,
            trajectory_wall_seconds=elapsed,
            preparation_wall_seconds=prepare,
            bitwise_restart=True,
            accepted_steps=endpoint.counters[0],
            rejected_steps=endpoint.counters[1],
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cpu-reference",
        type=Path,
        help="Compare the fine CUDA trajectory with saved Radau results.",
    )
    args = parser.parse_args()
    if args.cpu_reference and args.device != "cuda":
        parser.error("--cpu-reference is for the CUDA comparison")
    result = (
        cpu(initial_state()) if args.device == "cpu" else gpu(initial_state())
    )
    fine = result["2e-06"]
    np.testing.assert_allclose(
        result["2e-05"]["water_fraction"],
        fine["water_fraction"],
        atol=5e-5,
        rtol=0.0,
    )
    if args.cpu_reference:
        reference = json.loads(args.cpu_reference.read_text())["2e-06"]
        errors = {}
        for key, tolerance in (
            ("water_fraction", 5e-5),
            ("temperature_k", 1e-5),
            ("beta_mass_share", 1e-3),
        ):
            error = np.asarray(fine[key]) - reference[key]
            np.testing.assert_allclose(error, 0.0, rtol=0.0, atol=tolerance)
            errors[key] = float(np.abs(error).max())
        result["max_absolute_difference_from_cpu"] = errors
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
