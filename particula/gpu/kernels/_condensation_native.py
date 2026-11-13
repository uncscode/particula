"""Private mapped condensation reductions and read-only native admission.

Particle transfer arrays remain per-particle kg in particle-material order.
Gas reductions use kg/m^3 in complete gas order. Explicit device maps contain
-1 for unmapped/skipped lanes; no species identity is inferred from widths.
"""

from typing import Any, no_type_check

import warp as wp


@wp.kernel
def _validate_primary(  # noqa: C901
    masses: Any,
    concentration: Any,
    charge: Any,
    density: Any,
    volume: Any,
    gas_concentration: Any,
    molar_mass: Any,
    kind: wp.int32,
    invalid: Any,
) -> None:
    """Check physical native values without writing caller storage."""
    box, lane = wp.tid()  # type: ignore[misc]
    if lane == 0:
        v = volume[box]
        if not wp.isfinite(v) or v <= wp.float64(0.0):
            wp.atomic_add(invalid, 0, 1)
        if kind == wp.int32(1) and v != wp.float64(1.0):
            wp.atomic_add(invalid, 0, 1)
    if lane < masses.shape[1]:
        count = concentration[box, lane]
        if not wp.isfinite(count) or count < wp.float64(0.0):
            wp.atomic_add(invalid, 0, 1)
        if kind == wp.int32(3):
            v = volume[box]
            if v > wp.float64(0.0) and wp.isfinite(v):
                if not wp.isfinite(count / v):
                    wp.atomic_add(invalid, 0, 1)
        if not wp.isfinite(charge[box, lane]):
            wp.atomic_add(invalid, 0, 1)
        for species in range(masses.shape[2]):
            m = masses[box, lane, species]
            if not wp.isfinite(m) or m < wp.float64(0.0):
                wp.atomic_add(invalid, 0, 1)
    if lane < density.shape[0]:
        rho = density[lane]
        if not wp.isfinite(rho) or rho <= wp.float64(0.0):
            wp.atomic_add(invalid, 0, 1)
    if lane < gas_concentration.shape[1]:
        c = gas_concentration[box, lane]
        m = molar_mass[lane]
        if not wp.isfinite(c) or c < wp.float64(0.0):
            wp.atomic_add(invalid, 0, 1)
        if not wp.isfinite(m) or m <= wp.float64(0.0):
            wp.atomic_add(invalid, 0, 1)


def validate_primary_state(particles: Any, gas: Any, kind: int) -> None:
    """Read back only a private validation flag before any state writer."""
    boxes, slots, materials = particles.masses.shape
    invalid = wp.zeros(1, dtype=wp.int32, device=particles.masses.device)
    wp.launch(
        _validate_primary,
        dim=(boxes, max(1, slots, materials, gas.concentration.shape[1])),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            gas.concentration,
            gas.molar_mass,
            wp.int32(kind),
            invalid,
        ],
        device=particles.masses.device,
    )
    if invalid.numpy()[0]:
        raise ValueError("invalid native condensation physical state or volume")


@wp.func
def _density_weight(
    concentration: Any,
    volume: Any,
    kind: wp.int32,
    box: int,
    slot: int,
) -> wp.float64:
    weight = concentration[box, slot]
    if kind == wp.int32(3):
        weight = weight / volume[box]
    return weight


@wp.kernel
def gate_transfer(
    partitioning: Any,
    concentration: Any,
    proposal: Any,
    particle_to_gas: Any,
) -> None:
    """Zero disabled full-gas lanes, absent links and inactive particles."""
    box, slot = wp.tid()  # type: ignore[misc]
    for p in range(proposal.shape[2]):
        g = particle_to_gas[p]
        enabled = False
        if g >= 0:
            enabled = partitioning[box, g] == wp.int32(1)
        if not enabled or concentration[box, slot] == wp.float64(0.0):
            proposal[box, slot, p] = wp.float64(0.0)


@wp.kernel
def reduce_candidates(
    candidate: Any,
    concentration: Any,
    demand: Any,
    release: Any,
    gas_to_particle: Any,
    volume: Any,
    kind: wp.int32,
    masses: Any,
) -> None:
    """Reduce bounded particle proposals into full-gas density inventory."""
    box, g = wp.tid()  # type: ignore[misc]
    p = gas_to_particle[g]
    uptake = wp.float64(0.0)
    evaporation = wp.float64(0.0)
    if p >= 0:
        for slot in range(candidate.shape[1]):
            transfer = candidate[box, slot, p]
            if transfer < wp.float64(0.0):
                mass = masses[box, slot, p]
                transfer = (mass + transfer) - mass
            weighted = transfer * _density_weight(
                concentration, volume, kind, box, slot
            )
            if transfer > wp.float64(0.0):
                uptake += weighted
            else:
                evaporation -= weighted
    demand[box, g] = uptake
    release[box, g] = evaporation


@wp.kernel
def finalize_transfer(
    masses: Any,
    candidate: Any,
    scale: Any,
    finalized: Any,
    particle_to_gas: Any,
) -> None:
    """Retain the representable particle change after mapped limiting."""
    box, slot = wp.tid()  # type: ignore[misc]
    for p in range(masses.shape[2]):
        g = particle_to_gas[p]
        transfer = wp.float64(0.0)
        if g >= 0:
            transfer = candidate[box, slot, p]
            if transfer > wp.float64(0.0):
                transfer *= scale[box, g]
            mass = masses[box, slot, p]
            transfer = (mass + transfer) - mass
        finalized[box, slot, p] = transfer


@wp.kernel
def reduce_finalized(
    finalized: Any,
    concentration: Any,
    gas_delta: Any,
    gas_to_particle: Any,
    volume: Any,
    kind: wp.int32,
    masses: Any,
    gas_concentration: Any,
) -> None:
    """Reconcile rounded uptake before either primary inventory is updated.

    One lane owns each mapped species, so corrections and the weighted sum
    use the same deterministic order. A downward relative-epsilon step moves
    gaining masses by at least one representable value (Warp has no nextafter).
    If four corrections cannot fit the budget, suppress uptake for that species.
    Nonfinite reductions suppress all transfers for that species.
    """
    box, g = wp.tid()  # type: ignore[misc]
    p = gas_to_particle[g]
    delta = wp.float64(0.0)
    if p >= 0:
        for attempt in range(5):
            delta = wp.float64(0.0)
            for slot in range(finalized.shape[1]):
                delta += finalized[box, slot, p] * _density_weight(
                    concentration, volume, kind, box, slot
                )
            if wp.isfinite(delta) and delta <= gas_concentration[box, g]:
                break
            for slot in range(finalized.shape[1]):
                transfer = finalized[box, slot, p]
                if not wp.isfinite(delta):
                    finalized[box, slot, p] = wp.float64(0.0)
                elif transfer > wp.float64(0.0):
                    mass = masses[box, slot, p]
                    updated = mass + transfer
                    decrement = wp.max(
                        updated * wp.float64(2.220446049250313e-16),
                        wp.float64(4.9406564584124654e-324),
                    )
                    updated = wp.max(mass, updated - decrement)
                    if attempt == 4:
                        updated = mass
                    finalized[box, slot, p] = updated - mass
        # Recompute after the last correction, including the bounded fallback.
        delta = wp.float64(0.0)
        for slot in range(finalized.shape[1]):
            delta += finalized[box, slot, p] * _density_weight(
                concentration, volume, kind, box, slot
            )
    gas_delta[box, g] = delta


@wp.kernel
def energy_transfer(
    total: Any,
    latent_heat: Any,
    energy: Any,
    gas_to_particle: Any,
    concentration: Any,
    volume: Any,
    kind: wp.int32,
) -> None:
    """Write signed extensive energy in J in full gas order."""
    box, g = wp.tid()  # type: ignore[misc]
    p = gas_to_particle[g]
    delta = wp.float64(0.0)
    if p >= 0:
        for slot in range(total.shape[1]):
            count = concentration[box, slot]
            if kind == wp.int32(1):
                count *= volume[box]
            delta += total[box, slot, p] * count
    energy[box, g] = delta * latent_heat[g]


@no_type_check
@wp.func
def retire_evaporated_slots_body(
    masses: Any,
    concentration: Any,
    charge: Any,
    box: wp.int32,
    slot: wp.int32,
) -> None:
    """Clear fully evaporated resolved slots after gas/energy accounting."""
    empty = bool(True)
    for species in range(masses.shape[2]):
        if masses[box, slot, species] != wp.float64(0.0):
            empty = False
    if empty and concentration[box, slot] > wp.float64(0.0):
        concentration[box, slot] = wp.float64(0.0)
        charge[box, slot] = wp.float64(0.0)


@no_type_check
@wp.kernel
def retire_evaporated_slots(
    masses: Any, concentration: Any, charge: Any
) -> None:
    """Run the shared device operation through the direct kernel API."""
    box, slot = wp.tid()
    retire_evaporated_slots_body(masses, concentration, charge, box, slot)


@no_type_check
@wp.kernel
def retire_evaporated_slots_guarded(
    masses: Any,
    concentration: Any,
    charge: Any,
    failure: wp.array(dtype=wp.int32),
) -> None:
    """Skip this resident writer after a latched simulation failure."""
    if failure[0] != 0:
        return
    box, slot = wp.tid()
    retire_evaporated_slots_body(masses, concentration, charge, box, slot)
