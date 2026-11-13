"""Deterministic chunk reductions for large native condensation populations.

No atomics or particle-pair storage are used. Grouping changes roundoff relative
to the serial route; representable transfer reconciliation still uses the same
four downward corrections followed by suppression of excessive uptake. All
scratch is allocated at preparation and overwritten before each use.
"""

from dataclasses import dataclass
from typing import Any, no_type_check

import warp as wp

from particula.gpu.kernels._condensation_native import _density_weight

CHUNK_SIZE = 256
MIN_SLOTS = 4096


@no_type_check
@wp.kernel
def _candidate_chunks(
    candidate: Any,
    concentration: Any,
    masses: Any,
    mapping: Any,
    volume: Any,
    kind: wp.int32,
    uptake: Any,
    release: Any,
):
    box, gas, chunk = wp.tid()
    material = mapping[gas]
    positive = wp.float64(0.0)
    negative = wp.float64(0.0)
    if material >= 0:
        start = chunk * 256
        stop = wp.min(start + 256, candidate.shape[1])
        for slot in range(start, stop):
            transfer = candidate[box, slot, material]
            if transfer < wp.float64(0.0):
                mass = masses[box, slot, material]
                transfer = (mass + transfer) - mass
            weighted = transfer * _density_weight(
                concentration, volume, kind, box, slot
            )
            if transfer > wp.float64(0.0):
                positive += weighted
            else:
                negative -= weighted
    uptake[box, gas, chunk] = positive
    release[box, gas, chunk] = negative


@no_type_check
@wp.kernel
def _finish_candidates(positive: Any, negative: Any, demand: Any, release: Any):
    box, gas = wp.tid()
    uptake = wp.float64(0.0)
    evaporation = wp.float64(0.0)
    for chunk in range(positive.shape[2]):
        uptake += positive[box, gas, chunk]
        evaporation += negative[box, gas, chunk]
    demand[box, gas] = uptake
    release[box, gas] = evaporation


@no_type_check
@wp.kernel
def _transfer_chunks(
    transfer: Any,
    concentration: Any,
    mapping: Any,
    volume: Any,
    kind: wp.int32,
    partial: Any,
):
    box, gas, chunk = wp.tid()
    material = mapping[gas]
    total = wp.float64(0.0)
    if material >= 0:
        start = chunk * 256
        stop = wp.min(start + 256, transfer.shape[1])
        for slot in range(start, stop):
            total += transfer[box, slot, material] * _density_weight(
                concentration, volume, kind, box, slot
            )
    partial[box, gas, chunk] = total


@no_type_check
@wp.kernel
def _finish_transfer(partial: Any, gas: Any, delta: Any, status: Any):
    box, species = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(partial.shape[2]):
        total += partial[box, species, chunk]
    delta[box, species] = total
    flag = wp.int32(0)
    if not wp.isfinite(total):
        flag = wp.int32(1)
    elif total > gas[box, species]:
        flag = wp.int32(2)
    status[box, species] = flag


@no_type_check
@wp.kernel
def _correct_transfer(
    finalized: Any, masses: Any, mapping: Any, status: Any, attempt: wp.int32
):
    box, slot = wp.tid()
    for material in range(finalized.shape[2]):
        gas = mapping[material]
        if gas >= 0:
            flag = status[box, gas]
            transfer = finalized[box, slot, material]
            if flag == wp.int32(1):
                finalized[box, slot, material] = wp.float64(0.0)
            elif flag == wp.int32(2) and transfer > wp.float64(0.0):
                mass = masses[box, slot, material]
                updated = mass + transfer
                decrement = wp.max(
                    updated * wp.float64(2.220446049250313e-16),
                    wp.float64(4.9406564584124654e-324),
                )
                updated = wp.max(mass, updated - decrement)
                if attempt == wp.int32(4):
                    updated = mass
                finalized[box, slot, material] = updated - mass


@no_type_check
@wp.kernel
def _energy_chunks(
    total: Any,
    concentration: Any,
    mapping: Any,
    volume: Any,
    kind: wp.int32,
    partial: Any,
):
    box, gas, chunk = wp.tid()
    material = mapping[gas]
    energy_mass = wp.float64(0.0)
    if material >= 0:
        start = chunk * 256
        stop = wp.min(start + 256, total.shape[1])
        for slot in range(start, stop):
            count = concentration[box, slot]
            if kind == wp.int32(1):
                count *= volume[box]
            energy_mass += total[box, slot, material] * count
    partial[box, gas, chunk] = energy_mass


@no_type_check
@wp.kernel
def _finish_energy(partial: Any, latent: Any, energy: Any):
    box, gas = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(partial.shape[2]):
        total += partial[box, gas, chunk]
    energy[box, gas] = total * latent[gas]


@dataclass(frozen=True)
class _ReductionWork:
    """Private scratch, retained by the prepared call and its captured graph."""

    positive: Any
    negative: Any
    partial: Any
    status: Any


def prepare(boxes, slots, gases, device):
    """Allocate O(B G ceil(N/256)) storage once, outside stepping."""
    shape = (boxes, gases, (slots + CHUNK_SIZE - 1) // CHUNK_SIZE)
    return _ReductionWork(
        *(wp.zeros(shape, dtype=wp.float64, device=device) for _ in range(3)),
        wp.zeros((boxes, gases), dtype=wp.int32, device=device),
    )


def candidates(p):
    """Reduce proposals into the same full-gas demand/release outputs."""
    work = p.parallel_reductions
    wp.launch(
        _candidate_chunks,
        dim=work.partial.shape,
        inputs=[
            p.candidate,
            p.particle_concentration,
            p.particle_masses,
            p.gas_to_particle,
            p.particle_volume,
            wp.int32(p.distribution_kind),
            work.positive,
            work.negative,
        ],
        device=p.device,
    )
    wp.launch(
        _finish_candidates,
        dim=p.gas_concentration.shape,
        inputs=[work.positive, work.negative, p.demand, p.release],
        device=p.device,
    )


def finalized(p):
    """Reconcile rounded uptake with a fixed capture-safe sequence."""
    work = p.parallel_reductions
    for attempt in range(6):
        wp.launch(
            _transfer_chunks,
            dim=work.partial.shape,
            inputs=[
                p.finalized,
                p.particle_concentration,
                p.gas_to_particle,
                p.particle_volume,
                wp.int32(p.distribution_kind),
                work.partial,
            ],
            device=p.device,
        )
        wp.launch(
            _finish_transfer,
            dim=p.gas_concentration.shape,
            inputs=[
                work.partial,
                p.gas_concentration,
                p.gas_delta,
                work.status,
            ],
            device=p.device,
        )
        if attempt < 5:
            wp.launch(
                _correct_transfer,
                dim=p.dimensions[:2],
                inputs=[
                    p.finalized,
                    p.particle_masses,
                    p.particle_to_gas,
                    work.status,
                    wp.int32(attempt),
                ],
                device=p.device,
            )


def energy(p):
    """Reduce accumulated transfer to extensive energy in full gas order."""
    work = p.parallel_reductions
    wp.launch(
        _energy_chunks,
        dim=work.partial.shape,
        inputs=[
            p.total_mass_transfer,
            p.particle_concentration,
            p.gas_to_particle,
            p.particle_volume,
            wp.int32(p.distribution_kind),
            work.partial,
        ],
        device=p.device,
    )
    wp.launch(
        _finish_energy,
        dim=p.gas_concentration.shape,
        inputs=[work.partial, p.latent_heat, p.energy_transfer],
        device=p.device,
    )
