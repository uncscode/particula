"""Bounded, deterministic particle reductions for CPU observation."""

from typing import Any, no_type_check

import numpy as np
import warp as wp

from particula.execution.prepared_types import SimulationDiagnostics


@no_type_check
@wp.kernel
def _particle_partials(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    partials: wp.array3d(dtype=wp.float64),
):
    """Reduce fixed 256-slot chunks in a reproducible ascending order."""
    box, chunk, lane = wp.tid()
    total = wp.float64(0.0)
    for slot in range(chunk * 256, wp.min((chunk + 1) * 256, counts.shape[1])):
        count = counts[box, slot]
        if lane < masses.shape[2]:
            total += masses[box, slot, lane] * count
        elif lane == masses.shape[2]:
            total += count
        elif lane == masses.shape[2] + 1:
            total += charge[box, slot] * count
        elif count > 0.0:
            total += wp.float64(1.0)
    partials[box, chunk, lane] = total


@no_type_check
@wp.kernel
def _finish_partials(
    partials: wp.array3d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    resolved: wp.int32,
    output: wp.array2d(dtype=wp.float64),
):
    """Normalize counts by volume; retain density and active-slot count."""
    box, lane = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(partials.shape[1]):
        total += partials[box, chunk, lane]
    if resolved != 0 and lane < partials.shape[2] - 1:
        total /= volume[box]
    output[box, lane] = total


def setup_diagnostics(owner: Any):
    """Allocate reduction storage once and return a retained observer."""
    particles, gas, environment = (
        owner._session.particles,
        owner._session.gas,
        owner._session.environment,
    )
    boxes, slots, species = particles.masses.shape
    partials = wp.empty(
        (boxes, (slots + 255) // 256, species + 3),
        dtype=wp.float64,
        device=owner.device.native,
    )
    output = wp.empty(
        (boxes, species + 3), dtype=wp.float64, device=owner.device.native
    )
    resolved = owner._session.metadata.distribution_type == "particle_resolved"
    condensation = owner._bindings.get("condensation")
    energy = (
        None if condensation is None else condensation.prepared.energy_transfer
    )

    def observe():
        wp.launch(
            _particle_partials,
            dim=partials.shape,
            inputs=[
                particles.masses,
                particles.concentration,
                particles.charge,
                partials,
            ],
            device=owner.device.native,
        )
        wp.launch(
            _finish_partials,
            dim=output.shape,
            inputs=[partials, particles.volume, int(resolved), output],
            device=owner.device.native,
        )
        owner._wait()
        totals = output.numpy().copy()
        return SimulationDiagnostics(
            owner.completed_steps,
            owner.simulated_time,
            owner.gas_names,
            totals[:, species + 2].astype(np.int64),
            totals[:, species].copy(),
            totals[:, species + 1].copy(),
            totals[:, :species].copy(),
            gas.concentration.numpy().copy(),
            environment.temperature.numpy().copy(),
            environment.pressure.numpy().copy(),
            environment.saturation_ratio.numpy().copy(),
            particles.volume.numpy().copy(),
            None if energy is None else energy.numpy().copy(),
        )

    return observe
