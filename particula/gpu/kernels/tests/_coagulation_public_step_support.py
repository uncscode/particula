# ruff: noqa: S101
"""Lazy Warp-backed public coagulation-step test support.

This private module owns P2/P3 public-step materialization and invariants.  It
does not import Warp until a selected device observation requests it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.kernels._coagulation_config import CoagulationMechanismConfig


def _require_warp() -> Any:
    """Import Warp only for a selected runtime observation test."""
    try:
        import warp as wp
    except ImportError:
        pytest.skip("Warp not installed")
    return wp


def _materialize_public_particles(
    fixture: Any,
    *,
    n_species: int,
    active_by_box: tuple[tuple[int, ...], ...] | None = None,
    volume: np.ndarray | None = None,
    concentration: np.ndarray | None = None,
) -> Any:
    """Build CPU particle data with explicit active and inactive sentinels."""
    from particula.particles.particle_data import ParticleData

    active_by_box = active_by_box or fixture.active
    n_boxes, n_particles = fixture.radii.shape
    if volume is not None and np.asarray(volume).shape != (n_boxes,):
        raise ValueError("volume override must have shape (n_boxes,).")
    if concentration is not None and np.asarray(concentration).shape != (
        n_boxes,
        n_particles,
    ):
        raise ValueError(
            "concentration override must have shape (n_boxes, n_particles)."
        )
    masses = np.full((n_boxes, n_particles, n_species), 7.0e-31)
    particle_concentration = np.full((n_boxes, n_particles), 17.0)
    charge = np.full((n_boxes, n_particles), 29.0)
    fractions = np.arange(1, n_species + 1, dtype=np.float64)
    fractions /= fractions.sum()
    for box, active in enumerate(active_by_box):
        particle_concentration[box, :] = 0.0
        for slot in active:
            masses[box, slot] = fixture.masses[box, slot] * fractions
            particle_concentration[box, slot] = (
                1.0 if concentration is None else concentration[box, slot]
            )
            charge[box, slot] = fixture.charges[box, slot]
    return ParticleData(
        masses=masses,
        concentration=particle_concentration,
        charge=charge,
        density=np.full(n_species, 1000.0),
        volume=(
            np.linspace(1.0e-18, 2.0e-18, n_boxes)
            if volume is None
            else np.asarray(volume, dtype=np.float64)
        ),
    )


def _public_snapshot(
    particles: Any, pairs: Any, counts: Any, states: Any
) -> dict[str, np.ndarray]:
    """Synchronize mutable public state used by one public-step call."""
    return {
        "masses": particles.masses.numpy().copy(),
        "concentration": particles.concentration.numpy().copy(),
        "charge": particles.charge.numpy().copy(),
        "pairs": pairs.numpy().copy(),
        "counts": counts.numpy().copy(),
        "states": states.numpy().copy(),
    }


def _run_on_warp_devices(wp: Any) -> list[str]:
    """Return available Warp devices after lazy Warp import."""
    from particula.gpu.tests.cuda_availability import warp_devices

    return warp_devices(wp)


def _assert_public_invariants(
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    active_by_box: tuple[tuple[int, ...], ...],
    *,
    charge_transfers: bool,
) -> None:
    """Assert collision ownership, inventory, charge, and inactive state."""
    initial_inventory = np.sum(
        initial["masses"] * initial["concentration"][..., None], axis=1
    )
    final_inventory = np.sum(
        final["masses"] * final["concentration"][..., None], axis=1
    )
    npt.assert_allclose(
        final_inventory, initial_inventory, rtol=1e-12, atol=1e-30
    )
    if charge_transfers:
        npt.assert_array_equal(
            final["charge"].sum(axis=1), initial["charge"].sum(axis=1)
        )
    for box, active in enumerate(active_by_box):
        count = int(final["counts"][box])
        pairs = final["pairs"][box, :count]
        assert 0 <= count <= final["pairs"].shape[1]
        assert np.all(pairs[:, 0] < pairs[:, 1]) if count else True
        if count > 1:
            assert np.all(
                np.lexsort((pairs[:, 1], pairs[:, 0])) == np.arange(count)
            )
        used = [int(slot) for pair in pairs for slot in pair]
        assert len(used) == len(set(used))
        assert set(used) <= set(active)
        inactive = sorted(set(range(final["charge"].shape[1])) - set(active))
        for key in ("masses", "concentration", "charge"):
            npt.assert_array_equal(
                final[key][box, inactive], initial[key][box, inactive]
            )
        for recipient, donor in pairs:
            npt.assert_array_equal(final["masses"][box, donor], 0.0)
            assert final["concentration"][box, donor] == 0.0
            assert final["charge"][box, donor] == 0.0
            npt.assert_allclose(
                final["masses"][box, recipient],
                initial["masses"][box, recipient]
                + initial["masses"][box, donor],
                rtol=1e-12,
                atol=1e-30,
            )
            assert final["charge"][box, recipient] == (
                initial["charge"][box, recipient]
                + initial["charge"][box, donor]
            )


def _run_public_case(  # noqa: PLR0913
    row: Any,
    fixture: Any,
    *,
    n_species: int,
    max_collisions: int,
    device: str,
    time_step: float = 1.0,
    turbulent_arrays: bool = False,
    active_by_box: tuple[tuple[int, ...], ...] | None = None,
    volume: np.ndarray | None = None,
    concentration: np.ndarray | None = None,
    seed: int = 41,
    turbulent_dissipation: float | np.ndarray | None = None,
    fluid_density: float | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Any, Any, Any, Any]:
    """Execute one fresh caller-owned public-step case and return all state."""
    wp = _require_warp()
    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    particles = to_warp_particle_data(
        _materialize_public_particles(
            fixture,
            n_species=n_species,
            active_by_box=active_by_box,
            volume=volume,
            concentration=concentration,
        ),
        device=device,
    )
    active_device = particles.masses.device
    n_boxes, _ = fixture.radii.shape
    pairs = wp.full(
        (n_boxes, max_collisions, 2), -31, dtype=wp.int32, device=active_device
    )
    counts = wp.full((n_boxes,), -17, dtype=wp.int32, device=active_device)
    states = wp.full((n_boxes,), seed, dtype=wp.uint32, device=active_device)
    initial = _public_snapshot(particles, pairs, counts, states)
    kwargs: dict[str, Any] = {}
    if row.mask & 8:
        dissipation = (
            2.0e-4 if turbulent_dissipation is None else turbulent_dissipation
        )
        density = 1.2 if fluid_density is None else fluid_density
        if turbulent_arrays:
            kwargs["turbulent_dissipation"] = wp.array(
                np.broadcast_to(dissipation, (n_boxes,)),
                dtype=wp.float64,
                device=active_device,
            )
            kwargs["fluid_density"] = wp.array(
                np.broadcast_to(density, (n_boxes,)),
                dtype=wp.float64,
                device=active_device,
            )
        else:
            kwargs["turbulent_dissipation"] = dissipation
            kwargs["fluid_density"] = density
    result, returned_pairs, returned_counts = coagulation_step_gpu(
        particles,
        temperature=wp.array(
            fixture.temperature, dtype=wp.float64, device=active_device
        ),
        pressure=wp.array(
            fixture.pressure, dtype=wp.float64, device=active_device
        ),
        time_step=time_step,
        volume=particles.volume,
        max_collisions=max_collisions,
        rng_seed=seed,
        collision_pairs=pairs,
        n_collisions=counts,
        rng_states=states,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(mechanisms=row.mechanisms),
        **kwargs,
    )
    assert result is particles
    assert returned_pairs is pairs
    assert returned_counts is counts
    return (
        initial,
        _public_snapshot(particles, pairs, counts, states),
        particles,
        pairs,
        counts,
        states,
    )
