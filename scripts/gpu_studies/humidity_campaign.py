"""Benchmark the two-leg BAT humidity pulse at matched observation times."""

import argparse
import json
import os
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter

import numpy as np
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scripts.docs_examples import import_study

pulse = import_study("bat-humidity-pulse")


def chamber(boxes, factors=None):
    """Use one computational slot per independently varied concentration."""
    initial, process = pulse.chamber()
    if factors is None:
        factors = np.geomspace(0.5, 2.0, boxes) if boxes > 1 else np.ones(1)
    particles = initial.particles
    gas = initial.gas
    return Aerosol(
        particles=ParticleData(
            masses=np.repeat(particles.masses, boxes, axis=0),
            concentration=particles.concentration * factors[:, None],
            volume=np.repeat(particles.volume, boxes),
            density=particles.density.copy(),
            charge=np.zeros((boxes, 1)),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            gas.name,
            gas.molar_mass,
            np.repeat(gas.concentration, boxes, axis=0),
            gas.partitioning,
        ),
        environment=EnvironmentData(
            np.full(boxes, pulse.TEMPERATURE),
            np.full(boxes, 101325.0),
            np.full((boxes, 1), 0.90),
        ),
    ), process


def run(  # noqa: C901 -- timed protocol with explicit observation boundaries
    boxes, mode, steps, output, restart=False, factors=None, checkpoints=None
):
    """Record time-resolved histories and synchronized workflow costs."""
    costs = {}
    owner = None
    device = None
    if mode != "cpu":
        import warp as wp

        wp.init()
        device = wp.get_device("cuda:0")
        if not device.is_cuda:
            raise RuntimeError("An actual CUDA device is required")

    def measure(name, action, *args):
        start = perf_counter()
        result = action(*args)
        if device:
            wp.synchronize_device(device)
        costs[name] = costs.get(name, 0.0) + perf_counter() - start
        return result

    start = perf_counter()
    state, process = measure("construct", lambda: chamber(boxes, factors))
    dt = pulse.SAMPLE_INTERVAL / steps
    dry_mass = state.particles.masses[:, :, 1].copy()
    if device:
        owner = measure(
            "prepare_upload",
            lambda: process.prepare(state, time_step=dt, device="cuda:0"),
        )
        if mode == "captured":
            measure("capture", owner.capture)
    rows = []

    def observe():
        snapshot = owner.snapshot() if owner else state
        np.testing.assert_array_equal(
            snapshot.particles.masses[:, :, 1], dry_mass
        )
        return np.stack(
            (
                snapshot.particles.radii[:, 0],
                snapshot.gas.concentration[:, 0],
                snapshot.particles.species_mass_density[:, 0],
            ),
            axis=-1,
        )

    def advance(count):
        if owner:
            owner.run(count)
        else:
            for _ in range(count):
                process.execute(state, time_step=dt, sub_steps=4)

    try:
        for leg in range(2):
            if leg:
                if checkpoints is not None:
                    checkpoints.append(owner.checkpoint())
                if restart:
                    saved = owner.checkpoint()
                    owner.close()
                    owner = process.resume(saved)
                    if mode == "captured":
                        owner.capture()
                replacement = np.full((boxes, 1), pulse.gas_water(0.35))
                if owner:
                    measure("reset", owner.update_gas, replacement)
                else:
                    state.gas.concentration[:] = replacement
            if leg and checkpoints is not None:
                checkpoints.append(owner.checkpoint())
            rows.append(measure("observations", observe))
            for sample in range(32):
                if mode == "captured" and leg == sample == 0:
                    measure("first_replay", lambda: advance(1))
                    measure("step", lambda: advance(steps - 1))
                else:
                    measure("step", lambda: advance(steps))
                rows.append(measure("observations", observe))
        if checkpoints is not None:
            checkpoints.append(owner.checkpoint())
        if device:
            costs["device_pool_peak_bytes"] = wp.get_mempool_used_mem_high(
                device
            )
    finally:
        if owner:
            measure("close", owner.close)
    data = np.asarray(rows)
    for leg in np.split(data, 2):
        changes = leg - leg[0]
        np.testing.assert_allclose(
            changes[:, :, 1], -changes[:, :, 2], rtol=1e-9, atol=1e-15
        )
    np.testing.assert_array_equal(data[32, :, [0, 2]], data[33, :, [0, 2]])
    np.testing.assert_allclose(
        data[33, :, 1], pulse.gas_water(0.35), rtol=0, atol=0
    )

    def save():
        with output.open("wb") as stream:
            np.save(stream, data)
            stream.flush()
            os.fsync(stream.fileno())

    measure("save", save)
    costs["total"] = perf_counter() - start
    costs["peak_rss_kib"] = getrusage(RUSAGE_SELF).ru_maxrss
    return costs


def main():
    """Run one configuration in an isolated process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "prepared", "captured"])
    parser.add_argument("boxes", type=int)
    parser.add_argument("steps", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for rep in range(args.repetitions):
        stem = f"{args.mode}-{args.boxes}-{args.steps}"
        timing = run(
            args.boxes, args.mode, args.steps, args.output / f"{stem}.npy"
        )
        print(json.dumps(dict(repetition=rep, **timing)), flush=True)
        (args.output / f"{stem}-{rep}.json").write_text(json.dumps(timing))


if __name__ == "__main__":
    main()
