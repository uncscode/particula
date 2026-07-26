"""Focused contract checks for direct fixed-capacity GPU resampling."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


def _warp():
    """Import Warp only while executing a Warp-targeted test."""
    return pytest.importorskip("warp")


def _state(device: str = "cpu"):
    """Create a sparse two-species fixed-capacity particle state and buffers."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import ResamplingBuffers

    particles = SimpleNamespace(
        masses=wp.array(
            np.array([[[1.0, 0.0], [2.0, 1.0], [4.0, 0.0], [0.0, 0.0]]]),
            dtype=wp.float64,
            device=device,
        ),
        concentration=wp.array(
            [[2.0, 3.0, 5.0, 0.0]], dtype=wp.float64, device=device
        ),
        charge=wp.array(
            [[1.0, 2.0, 3.0, 0.0]], dtype=wp.float64, device=device
        ),
        density=wp.array([1.0, 1.0], dtype=wp.float64, device=device),
        volume=wp.array([1.0], dtype=wp.float64, device=device),
    )
    buffers = ResamplingBuffers(
        retained_counts=wp.zeros(1, dtype=wp.int32, device=device),
        released_counts=wp.zeros(1, dtype=wp.int32, device=device),
        retained_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        released_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        sorted_indices=wp.zeros((1, 4), dtype=wp.int32, device=device),
        replacement_masses=wp.zeros((1, 4, 2), dtype=wp.float64, device=device),
        replacement_concentration=wp.zeros(
            (1, 4), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros((1, 4), dtype=wp.float64, device=device),
        source_radii=wp.zeros((1, 4), dtype=wp.float64, device=device),
        radius_cubed_relative_error=wp.zeros(
            1, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(1, dtype=wp.float64, device=device),
        surface_relative_error=wp.zeros(1, dtype=wp.float64, device=device),
        diversity_absolute_error=wp.zeros(1, dtype=wp.float64, device=device),
        planning_status=wp.zeros(1, dtype=wp.int32, device=device),
    )
    counts = wp.array([1], dtype=wp.int32, device=device)
    return particles, counts, buffers


def _custom_state(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    density: np.ndarray,
    counts: np.ndarray,
    device: str = "cpu",
):
    """Create a custom Warp state for targeted parity and validation cases."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import ResamplingBuffers

    box_count, particle_count, species_count = masses.shape
    particles = SimpleNamespace(
        masses=wp.array(masses, dtype=wp.float64, device=device),
        concentration=wp.array(concentration, dtype=wp.float64, device=device),
        charge=wp.array(charge, dtype=wp.float64, device=device),
        density=wp.array(density, dtype=wp.float64, device=device),
        volume=wp.array(np.ones(box_count), dtype=wp.float64, device=device),
    )
    buffers = ResamplingBuffers(
        retained_counts=wp.zeros(box_count, dtype=wp.int32, device=device),
        released_counts=wp.zeros(box_count, dtype=wp.int32, device=device),
        retained_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        released_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        sorted_indices=wp.zeros(
            (box_count, particle_count), dtype=wp.int32, device=device
        ),
        replacement_masses=wp.zeros(
            (box_count, particle_count, species_count),
            dtype=wp.float64,
            device=device,
        ),
        replacement_concentration=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        source_radii=wp.zeros(
            (box_count, particle_count), dtype=wp.float64, device=device
        ),
        radius_cubed_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        surface_relative_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        diversity_absolute_error=wp.zeros(
            box_count, dtype=wp.float64, device=device
        ),
        planning_status=wp.zeros(box_count, dtype=wp.int32, device=device),
    )
    return particles, wp.array(counts, dtype=wp.int32, device=device), buffers


def _single_active_state():
    """Create a valid state with one active slot and reusable buffers."""
    wp = _warp()
    particles, _, buffers = _state()
    particles.masses = wp.array(
        np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        dtype=wp.float64,
        device="cpu",
    )
    particles.concentration = wp.array(
        [[2.0, 0.0, 0.0, 0.0]], dtype=wp.float64, device="cpu"
    )
    particles.charge = wp.zeros((1, 4), dtype=wp.float64, device="cpu")
    return particles, buffers


def _snapshot(*containers: object) -> list[np.ndarray]:
    """Copy all Warp fields in containers for exact non-mutation checks."""
    return [
        value.numpy().copy()
        for container in containers
        for value in vars(container).values()
    ]


def _assert_snapshot(snapshot: list[np.ndarray], *containers: object) -> None:
    """Assert all Warp fields in containers retain their copied values."""
    values = [
        value for container in containers for value in vars(container).values()
    ]
    for value, expected in zip(values, snapshot, strict=True):
        npt.assert_array_equal(value.numpy(), expected)


def _numpy_diversity(masses: np.ndarray, concentration: np.ndarray) -> float:
    """Calculate the independent Riemer diversity for a dense source set."""
    represented = concentration * np.sum(masses, axis=1)
    total = np.sum(represented)
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    entropy = 0.0
    for particle, particle_mass in enumerate(np.sum(masses, axis=1)):
        if represented[particle] <= 0.0 or particle_mass <= 0.0:
            continue
        fractions = masses[particle] / particle_mass
        positive_fractions = fractions[fractions > 0.0]
        entropy -= (
            represented[particle]
            / total
            * np.sum(positive_fractions * np.log(positive_fractions))
        )
    return float(np.exp(entropy))


def _numpy_remapping_oracle(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    density: np.ndarray,
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Independently implement the P2 equal-weight remapping contract."""
    box_count, particle_count, species_count = masses.shape
    result = {
        "masses": masses.copy(),
        "concentration": concentration.copy(),
        "charge": charge.copy(),
        "retained_counts": np.zeros(box_count, dtype=np.int32),
        "released_counts": np.zeros(box_count, dtype=np.int32),
        "retained_indices": np.full((box_count, particle_count), -1, np.int32),
        "released_indices": np.full((box_count, particle_count), -1, np.int32),
        "sorted_indices": np.full((box_count, particle_count), -1, np.int32),
        "replacement_masses": np.zeros_like(masses),
        "replacement_concentration": np.zeros_like(concentration),
        "replacement_charge": np.zeros_like(charge),
        "source_radii": np.zeros_like(concentration),
        "radius_error": np.zeros(box_count),
        "mean_error": np.zeros(box_count),
        "surface_error": np.zeros(box_count),
        "diversity_error": np.zeros(box_count),
    }
    for box in range(box_count):
        if counts[box] == 0:
            continue
        active = np.flatnonzero(concentration[box] > 0.0)
        radii = np.cbrt(
            3.0 * np.sum(masses[box] / density, axis=1) / (4.0 * np.pi)
        )
        result["source_radii"][box, active] = radii[active]

        source_records = []
        for index in active.tolist():
            total = np.sum(masses[box, index])
            fractions = tuple((masses[box, index] / total).tolist())
            source_records.append(
                (radii[index], *fractions, charge[box, index], index)
            )
        source_records.sort()
        ordered = np.array(
            [record[-1] for record in source_records], dtype=np.int32
        )
        kept = len(active) - int(counts[box])
        equal = np.sum(concentration[box, ordered]) / kept
        result["retained_counts"][box] = kept
        result["released_counts"][box] = counts[box]
        result["sorted_indices"][box, : len(ordered)] = ordered
        result["retained_indices"][box, :kept] = active[:kept]
        result["released_indices"][box, : counts[box]] = active[kept:]

        source = 0
        source_end = concentration[box, ordered[0]]
        for target in range(kept):
            target_start = target * equal
            target_end = (target + 1) * equal
            while source < len(ordered) and source_end <= target_start:
                source += 1
                if source < len(ordered):
                    source_end += concentration[box, ordered[source]]
            cursor = target_start
            while cursor < target_end and source < len(ordered):
                overlap_end = min(target_end, source_end)
                overlap = overlap_end - cursor
                particle = ordered[source]
                result["replacement_masses"][box, target] += (
                    overlap * masses[box, particle] / equal
                )
                result["replacement_charge"][box, target] += (
                    overlap * charge[box, particle] / equal
                )
                cursor = overlap_end
                if cursor >= source_end:
                    source += 1
                    if source < len(ordered):
                        source_end += concentration[box, ordered[source]]
            result["replacement_concentration"][box, target] = equal

        source_weight = concentration[box, ordered]
        source_radii = radii[ordered]
        replacement = result["replacement_masses"][box, :kept]
        replacement_radii = np.cbrt(
            3.0 * np.sum(replacement / density, axis=1) / (4.0 * np.pi)
        )
        source_moment = np.sum(source_weight * source_radii**3)
        source_mean = np.sum(source_weight * source_radii)
        source_surface = np.sum(source_weight * 4.0 * np.pi * source_radii**2)
        result["radius_error"][box] = abs(
            np.sum(equal * replacement_radii**3) - source_moment
        ) / abs(source_moment)
        result["mean_error"][box] = abs(
            np.sum(equal * replacement_radii) - source_mean
        ) / abs(source_mean)
        result["surface_error"][box] = abs(
            np.sum(equal * 4.0 * np.pi * replacement_radii**2) - source_surface
        ) / abs(source_surface)
        result["diversity_error"][box] = abs(
            _numpy_diversity(replacement, np.full(kept, equal))
            - _numpy_diversity(masses[box, ordered], source_weight)
        )
        retained = active[:kept]
        released = active[kept:]
        result["masses"][box, retained] = replacement
        result["concentration"][box, retained] = equal
        result["charge"][box, retained] = result["replacement_charge"][
            box, :kept
        ]
        result["masses"][box, released] = 0.0
        result["concentration"][box, released] = 0.0
        result["charge"][box, released] = 0.0
    return result


def test_resampling_remaps_retained_original_slots_and_conserves_inventory() -> (
    None
):
    """A valid P2 call commits equal rows and clears released original slots."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    before_mass = np.sum(
        particles.masses.numpy() * particles.concentration.numpy()[..., None],
        axis=1,
    )
    before_charge = np.sum(
        particles.charge.numpy() * particles.concentration.numpy(), axis=1
    )
    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.retained_indices.numpy()[0, :2].tolist() == [0, 1]
    assert buffers.released_indices.numpy()[0, :1].tolist() == [2]
    assert buffers.retained_indices.numpy()[0, 2:].tolist() == [-1, -1]
    assert particles.concentration.numpy()[0, 2] == 0.0
    assert np.all(particles.masses.numpy()[0, 2] == 0.0)
    after_mass = np.sum(
        particles.masses.numpy() * particles.concentration.numpy()[..., None],
        axis=1,
    )
    after_charge = np.sum(
        particles.charge.numpy() * particles.concentration.numpy(), axis=1
    )
    npt.assert_allclose(after_mass, before_mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(after_charge, before_charge, rtol=1e-12, atol=1e-30)


def test_resampling_zero_demand_is_write_free() -> None:
    """All-zero demand preserves particles and every caller-owned buffer."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    counts = _warp().zeros(1, dtype=_warp().int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)
    assert resampling_step_gpu(particles, counts, buffers) is particles
    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_empty_box_with_zero_demand_is_write_free() -> None:
    """An empty valid box and zero demand leave all caller storage untouched."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, buffers = _single_active_state()
    wp = _warp()
    particles.masses = wp.zeros((1, 4, 2), dtype=wp.float64, device="cpu")
    particles.concentration = wp.zeros((1, 4), dtype=wp.float64, device="cpu")
    counts = wp.zeros(1, dtype=wp.int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    assert resampling_step_gpu(particles, counts, buffers) is particles

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_empty_batch_validates_density_before_write_free_return() -> (
    None
):
    """An empty batch validates global density and preserves valid caller data."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.empty((0, 1, 1), dtype=np.float64)
    concentration = np.empty((0, 1), dtype=np.float64)
    charge = np.empty((0, 1), dtype=np.float64)
    particles, counts, buffers = _custom_state(
        masses,
        concentration,
        charge,
        np.array([1.0], dtype=np.float64),
        np.empty(0, dtype=np.int32),
    )
    snapshots = _snapshot(particles, SimpleNamespace(counts=counts), buffers)
    assert resampling_step_gpu(particles, counts, buffers) is particles
    _assert_snapshot(
        snapshots, particles, SimpleNamespace(counts=counts), buffers
    )

    particles, counts, buffers = _custom_state(
        masses,
        concentration,
        charge,
        np.array([np.nan], dtype=np.float64),
        np.empty(0, dtype=np.int32),
    )
    snapshots = _snapshot(particles, SimpleNamespace(counts=counts), buffers)
    with pytest.raises(ValueError, match="invalid values"):
        resampling_step_gpu(particles, counts, buffers)
    _assert_snapshot(
        snapshots, particles, SimpleNamespace(counts=counts), buffers
    )


def test_resampling_accepts_maximum_valid_release_count() -> None:
    """Releasing active-count minus one leaves exactly one retained slot."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, _, buffers = _state()
    wp = _warp()
    counts = wp.array([2], dtype=wp.int32, device="cpu")

    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.retained_counts.numpy().tolist() == [1]
    assert buffers.released_counts.numpy().tolist() == [2]
    assert np.count_nonzero(particles.concentration.numpy()) == 1


def test_resampling_rejects_release_from_single_active_slot_without_mutation() -> (
    None
):
    """A nonzero demand cannot release the only active slot."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, buffers = _single_active_state()
    wp = _warp()
    counts = wp.array([1], dtype=wp.int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_failed_diagnostics_do_not_commit_particles() -> None:
    """A planning diagnostic failure leaves the caller particle state intact."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    snapshots = [value.numpy().copy() for value in vars(particles).values()]

    with pytest.raises(ValueError, match="resampling diagnostic exceeds bound"):
        resampling_step_gpu(
            particles,
            counts,
            buffers,
            radius_cubed_relative_error=0.0,
            mean_radius_relative_error=0.0,
            surface_relative_error=0.0,
            diversity_absolute_error=0.0,
        )

    for value, expected in zip(
        vars(particles).values(), snapshots, strict=True
    ):
        npt.assert_array_equal(value.numpy(), expected)


def test_resampling_orders_sources_by_cpu_key_stably() -> None:
    """The direct planner sorts active sources by the documented CPU key."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.array([[[5.0], [1.0], [3.0], [2.0], [4.0]]], dtype=np.float64)
    concentration = np.ones((1, 5), dtype=np.float64)
    charge = np.zeros((1, 5), dtype=np.float64)
    density = np.array([1.0], dtype=np.float64)
    counts = np.array([2], dtype=np.int32)
    particles, counts, buffers = _custom_state(
        masses, concentration, charge, density, counts
    )

    assert resampling_step_gpu(particles, counts, buffers) is particles
    assert buffers.sorted_indices.numpy()[0].tolist() == [1, 3, 2, 4, 0]
    assert buffers.retained_indices.numpy()[0, :3].tolist() == [0, 1, 2]
    assert buffers.released_indices.numpy()[0, :2].tolist() == [3, 4]


def test_resampling_rejects_invalid_density_without_mutation() -> None:
    """Density validation rejects nonpositive species values before planning."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    wp = _warp()
    particles.density = wp.array([1.0, 0.0], dtype=wp.float64, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_rejects_diversity_bound_failure_without_commit() -> None:
    """The planning pass surfaces diversity violations before commit."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _custom_state(
        masses=np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
        concentration=np.array([[1.0, 1.0]], dtype=np.float64),
        charge=np.zeros((1, 2), dtype=np.float64),
        density=np.array([1.0, 1.0], dtype=np.float64),
        counts=np.array([1], dtype=np.int32),
    )
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container)

    with pytest.raises(ValueError, match="resampling diagnostic exceeds bound"):
        resampling_step_gpu(
            particles,
            counts,
            buffers,
            diversity_absolute_error=0.0,
        )

    _assert_snapshot(snapshots, particles, count_container)
    assert buffers.planning_status.numpy().tolist() == [5]


def test_resampling_pure_species_zero_diversity_bound_preserves_inventory() -> (
    None
):
    """A one-species source has exact diversity despite nonzero remapping error."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _custom_state(
        masses=np.array([[[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]]]),
        concentration=np.array([[1.0, 2.0, 4.0]]),
        charge=np.array([[1.0, -2.0, 3.0]]),
        density=np.array([1.0, 1.0]),
        counts=np.array([1], dtype=np.int32),
    )
    before_mass = np.sum(
        particles.masses.numpy() * particles.concentration.numpy()[..., None],
        axis=1,
    )
    before_charge = np.sum(
        particles.charge.numpy() * particles.concentration.numpy(), axis=1
    )

    assert (
        resampling_step_gpu(
            particles, counts, buffers, diversity_absolute_error=0.0
        )
        is particles
    )

    assert buffers.planning_status.numpy().tolist() == [0]
    assert buffers.diversity_absolute_error.numpy().tolist() == [0.0]
    npt.assert_allclose(
        np.sum(
            particles.masses.numpy()
            * particles.concentration.numpy()[..., None],
            axis=1,
        ),
        before_mass,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        np.sum(
            particles.charge.numpy() * particles.concentration.numpy(), axis=1
        ),
        before_charge,
        rtol=1e-12,
        atol=1e-30,
    )


def test_resampling_nonfinite_planning_rejects_all_box_commits() -> None:
    """An overflowing box prevents a valid sibling box from committing."""
    from particula.gpu.kernels.exhaustion import (
        PLANNING_NONFINITE_FAILURE,
        resampling_step_gpu,
    )

    particles, counts, buffers = _custom_state(
        masses=np.array(
            [
                [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [3.0, 1.0], [8.0, 0.0]],
            ]
        ),
        concentration=np.array([[1e308, 1e308, 0.0], [1.0, 2.0, 3.0]]),
        charge=np.zeros((2, 3)),
        density=np.array([1.0, 1.0]),
        counts=np.array([1, 1], dtype=np.int32),
    )
    count_container = SimpleNamespace(counts=counts)
    particle_snapshot = _snapshot(particles, count_container)

    with pytest.raises(ValueError, match="resampling diagnostic exceeds bound"):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(particle_snapshot, particles, count_container)
    assert buffers.planning_status.numpy().tolist() == [
        PLANNING_NONFINITE_FAILURE,
        0,
    ]


def test_resampling_rejects_duplicate_and_protected_alias_without_mutation() -> (
    None
):
    """Duplicate buffers and count-buffer aliasing fail before any caller write."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    for invalid_buffers in (
        replace(buffers, released_counts=buffers.retained_counts),
        replace(buffers, planning_status=counts),
    ):
        count_container = SimpleNamespace(counts=counts)
        snapshots = _snapshot(particles, count_container, buffers)
        with pytest.raises(
            ValueError, match="resampling arrays must not overlap"
        ):
            resampling_step_gpu(particles, counts, invalid_buffers)
        _assert_snapshot(snapshots, particles, count_container, buffers)


def test_resampling_rejects_strided_masses_without_mutation() -> None:
    """A strided Warp mass view is rejected before values or buffers are read."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _custom_state(
        masses=np.array([[[1.0, 0.0], [3.0, 1.0]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.zeros((1, 2)),
        density=np.array([1.0, 1.0]),
        counts=np.array([1], dtype=np.int32),
    )
    backing = wp.zeros((1, 4, 2), dtype=wp.float64, device="cpu")
    particles.masses = wp.array(
        ptr=backing.ptr,
        capacity=backing.capacity,
        dtype=wp.float64,
        shape=(1, 2, 2),
        strides=(64, 32, 8),
        device="cpu",
        copy=False,
    )
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)

    with pytest.raises(ValueError, match="masses must be a contiguous"):
        resampling_step_gpu(particles, counts, buffers)

    _assert_snapshot(snapshots, particles, count_container, buffers)


@pytest.mark.parametrize("count", [-1, 3])
def test_resampling_rejects_invalid_release_counts_without_mutation(
    count: int,
) -> None:
    """Release counts outside the active-capacity range are preflight errors."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, _, buffers = _state()
    wp = _warp()
    counts = wp.array([count], dtype=wp.int32, device="cpu")
    snapshots = [value.numpy().copy() for value in vars(particles).values()]
    snapshots.extend(value.numpy().copy() for value in vars(buffers).values())

    with pytest.raises(
        ValueError,
        match="particles or required_release_counts have invalid values",
    ):
        resampling_step_gpu(particles, counts, buffers)

    values = list(vars(particles).values()) + list(vars(buffers).values())
    for value, expected in zip(values, snapshots, strict=True):
        npt.assert_array_equal(value.numpy(), expected)


@pytest.mark.parametrize("bound", [0, np.float64(0.0), -1.0, float("nan")])
def test_resampling_rejects_non_exact_or_invalid_bounds(bound: Any) -> None:
    """Diagnostic bounds accept only finite nonnegative exact Python floats."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    count_container = SimpleNamespace(counts=counts)
    snapshots = _snapshot(particles, count_container, buffers)
    with pytest.raises(
        (TypeError, ValueError), match="radius_cubed_relative_error"
    ):
        resampling_step_gpu(  # type: ignore[arg-type]
            particles, counts, buffers, radius_cubed_relative_error=bound
        )
    _assert_snapshot(snapshots, particles, count_container, buffers)


def _assert_multi_box_oracle_parity(device: str) -> None:
    """Compare one multi-box P2 call with the independent NumPy oracle."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.array(
        [
            [[4.0, 0.0], [1.0, 2.0], [9.0, 3.0], [0.0, 0.0]],
            [[2.0, 1.0], [8.0, 0.0], [3.0, 4.0], [5.0, 1.0]],
        ],
        dtype=np.float64,
    )
    concentration = np.array(
        [[1.0, 2.0, 4.0, 0.0], [3.0, 1.0, 2.0, 5.0]], dtype=np.float64
    )
    charge = np.array(
        [[3.0, -1.0, 2.0, 0.0], [1.0, 4.0, -2.0, 5.0]], dtype=np.float64
    )
    density = np.array([1.0, 2.0], dtype=np.float64)
    counts = np.array([1, 2], dtype=np.int32)
    expected = _numpy_remapping_oracle(
        masses, concentration, charge, density, counts
    )
    particles, warp_counts, buffers = _custom_state(
        masses, concentration, charge, density, counts, device=device
    )
    before_inventory = np.sum(masses * concentration[..., None], axis=1)

    assert resampling_step_gpu(particles, warp_counts, buffers) is particles

    npt.assert_allclose(
        particles.masses.numpy(), expected["masses"], rtol=1e-12, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        expected["concentration"],
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        particles.charge.numpy(), expected["charge"], rtol=1e-12, atol=0.0
    )
    for name, expected_name in (
        ("retained_counts", "retained_counts"),
        ("released_counts", "released_counts"),
        ("retained_indices", "retained_indices"),
        ("released_indices", "released_indices"),
        ("sorted_indices", "sorted_indices"),
    ):
        npt.assert_array_equal(
            getattr(buffers, name).numpy(), expected[expected_name]
        )
    for name, expected_name in (
        ("replacement_masses", "replacement_masses"),
        ("replacement_concentration", "replacement_concentration"),
        ("replacement_charge", "replacement_charge"),
        ("source_radii", "source_radii"),
        ("radius_cubed_relative_error", "radius_error"),
        ("mean_radius_relative_error", "mean_error"),
        ("surface_relative_error", "surface_error"),
        ("diversity_absolute_error", "diversity_error"),
    ):
        npt.assert_allclose(
            getattr(buffers, name).numpy(),
            expected[expected_name],
            rtol=1e-12,
            atol=1e-15,
        )
    npt.assert_array_equal(
        buffers.planning_status.numpy(), np.zeros(2, np.int32)
    )
    # Conservation is intentionally separate from deterministic CPU-oracle parity.
    npt.assert_allclose(
        np.sum(
            particles.masses.numpy()
            * particles.concentration.numpy()[..., None],
            axis=1,
        ),
        before_inventory,
        rtol=1e-12,
        atol=1e-30,
    )


def test_resampling_multi_box_multi_species_matches_independent_numpy_oracle() -> (
    None
):
    """Warp CPU P2 outputs and diagnostics match the independent oracle."""
    _assert_multi_box_oracle_parity(device="cpu")


@pytest.mark.cuda
def test_resampling_cuda_multi_box_multi_species_matches_numpy_oracle() -> None:
    """CUDA P2 parity is optional and cleanly skipped on hosts without CUDA."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    _assert_multi_box_oracle_parity(device="cuda:0")


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.performance
def test_resampling_benchmark_smoke_uses_opt_in_marker() -> None:
    """Keep allocation-sensitive timing evidence behind the benchmark opt-in."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, counts, buffers = _state()
    assert resampling_step_gpu(particles, counts, buffers) is particles


def test_gpu_representative_volume_scaling_matches_numpy_oracle() -> None:
    """Concrete P4 scales only selected Warp rows and emits diagnostics."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    masses = np.array(
        [[[1.0, 0.0], [2.0, 1.0]], [[3.0, 0.0], [4.0, 2.0]]],
        dtype=np.float64,
    )
    concentration = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    particles = SimpleNamespace(
        masses=wp.array(masses, dtype=wp.float64, device="cpu"),
        concentration=wp.array(concentration, dtype=wp.float64, device="cpu"),
        charge=wp.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=wp.float64, device="cpu"
        ),
        density=wp.array([1.0, 1.0], dtype=wp.float64, device="cpu"),
        volume=wp.array([4.0, 6.0], dtype=wp.float64, device="cpu"),
    )
    demand = wp.array([3.0, 2.0], dtype=wp.float64, device="cpu")
    flags = wp.array([1, 0], dtype=wp.int32, device="cpu")
    requested = wp.array([0.5, 0.75], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25, 0.5], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([1.0, 1.0], dtype=wp.float64, device="cpu")
    resolved = wp.zeros(2, dtype=wp.float64, device="cpu")
    original_mass = particles.masses.numpy().copy()
    original_charge = particles.charge.numpy().copy()

    returned = representative_volume_scaling_step_gpu(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert returned[0] is particles
    assert returned[1] is demand
    assert returned[2] is resolved
    wp.synchronize_device("cpu")
    npt.assert_allclose(particles.volume.numpy(), [2.0, 6.0])
    npt.assert_allclose(
        particles.concentration.numpy(), [[1.0, 1.5], [5.0, 7.0]]
    )
    npt.assert_allclose(demand.numpy(), [1.5, 2.0])
    npt.assert_allclose(resolved.numpy(), [0.5, 1.0])
    npt.assert_array_equal(particles.masses.numpy(), original_mass)
    npt.assert_array_equal(particles.charge.numpy(), original_charge)


def test_gpu_representative_volume_scaling_rejects_late_invalid_row() -> None:
    """A later P4 bound failure leaves particles and every sidecar unchanged."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.array([[[1.0]], [[2.0]]]),
        np.array([[1.0], [2.0]]),
        np.zeros((2, 1)),
        np.array([1.0]),
        np.zeros(2, dtype=np.int32),
    )
    particles.volume = wp.array([2.0, 1.0], dtype=wp.float64, device="cpu")
    demand = wp.array([1.0, 1.0], dtype=wp.float64, device="cpu")
    flags = wp.array([1, 1], dtype=wp.int32, device="cpu")
    requested = wp.array([0.5, 0.5], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25, 0.25], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([0.5, 1.0], dtype=wp.float64, device="cpu")
    resolved = wp.array([9.0, 9.0], dtype=wp.float64, device="cpu")
    sidecars = SimpleNamespace(
        demand=demand,
        flags=flags,
        requested=requested,
        minimum=minimum,
        minimum_volume=minimum_volume,
        resolved=resolved,
    )
    snapshot = _snapshot(particles, sidecars)
    with pytest.raises(ValueError, match="representative scaling sidecars"):
        representative_volume_scaling_step_gpu(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    _assert_snapshot(snapshot, particles, sidecars)


def test_gpu_representative_volume_scaling_rejects_zeroing_active_slot() -> (
    None
):
    """P4 rejects an active concentration that would underflow to zero."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.array([[[1.0]]]),
        np.array([[np.nextafter(0.0, 1.0)]]),
        np.zeros((1, 1)),
        np.array([1.0]),
        np.zeros(1, dtype=np.int32),
    )
    demand = wp.array([1.0], dtype=wp.float64, device="cpu")
    flags = wp.array([1], dtype=wp.int32, device="cpu")
    requested = wp.array([0.5], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([0.1], dtype=wp.float64, device="cpu")
    resolved = wp.array([9.0], dtype=wp.float64, device="cpu")
    sidecars = SimpleNamespace(
        demand=demand,
        flags=flags,
        requested=requested,
        minimum=minimum,
        minimum_volume=minimum_volume,
        resolved=resolved,
    )
    snapshot = _snapshot(particles, sidecars)

    with pytest.raises(ValueError, match="representative scaling sidecars"):
        representative_volume_scaling_step_gpu(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )

    _assert_snapshot(snapshot, particles, sidecars)


def test_gpu_representative_volume_scaling_preflights_every_active_slot() -> (
    None
):
    """P4 rejects a later active lane that would underflow on scaling."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.array([[[1.0], [1.0]]]),
        np.array([[1.0, np.nextafter(0.0, 1.0)]]),
        np.zeros((1, 2)),
        np.array([1.0]),
        np.zeros(1, dtype=np.int32),
    )
    demand = wp.array([1.0], dtype=wp.float64, device="cpu")
    flags = wp.array([1], dtype=wp.int32, device="cpu")
    requested = wp.array([0.5], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([0.1], dtype=wp.float64, device="cpu")
    resolved = wp.array([9.0], dtype=wp.float64, device="cpu")
    sidecars = SimpleNamespace(
        demand=demand,
        flags=flags,
        requested=requested,
        minimum=minimum,
        minimum_volume=minimum_volume,
        resolved=resolved,
    )
    snapshot = _snapshot(particles, sidecars)

    with pytest.raises(ValueError, match="representative scaling sidecars"):
        representative_volume_scaling_step_gpu(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )

    _assert_snapshot(snapshot, particles, sidecars)


def test_gpu_representative_volume_scaling_zero_demand_writes_diagnostic_only() -> (
    None
):
    """P4 leaves non-diagnostic Warp storage untouched for zero demand."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.array([[[1.0]], [[2.0]]]),
        np.array([[1.0], [2.0]]),
        np.zeros((2, 1)),
        np.array([1.0]),
        np.zeros(2, dtype=np.int32),
    )
    demand = wp.zeros(2, dtype=wp.float64, device="cpu")
    flags = wp.array([1, 1], dtype=wp.int32, device="cpu")
    requested = wp.array([0.5, 0.5], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25, 0.25], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([0.1, 0.1], dtype=wp.float64, device="cpu")
    resolved = wp.zeros(2, dtype=wp.float64, device="cpu")
    sidecars = SimpleNamespace(
        demand=demand,
        flags=flags,
        requested=requested,
        minimum=minimum,
        minimum_volume=minimum_volume,
    )
    snapshot = _snapshot(particles, sidecars)

    returned = representative_volume_scaling_step_gpu(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert returned == (particles, demand, resolved)
    wp.synchronize_device("cpu")
    _assert_snapshot(snapshot, particles, sidecars)
    npt.assert_array_equal(resolved.numpy(), np.ones(2))


def test_gpu_representative_volume_scaling_false_flags_write_diagnostic_only() -> (
    None
):
    """False P4 flags leave positive-demand Warp rows unchanged and report one."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.array([[[1.0]], [[2.0]]]),
        np.array([[1.0], [2.0]]),
        np.zeros((2, 1)),
        np.array([1.0]),
        np.zeros(2, dtype=np.int32),
    )
    demand = wp.array([1.0, 2.0], dtype=wp.float64, device="cpu")
    flags = wp.zeros(2, dtype=wp.int32, device="cpu")
    requested = wp.array([0.5, 0.5], dtype=wp.float64, device="cpu")
    minimum = wp.array([0.25, 0.25], dtype=wp.float64, device="cpu")
    minimum_volume = wp.array([0.1, 0.1], dtype=wp.float64, device="cpu")
    resolved = wp.zeros(2, dtype=wp.float64, device="cpu")
    sidecars = SimpleNamespace(
        demand=demand,
        flags=flags,
        requested=requested,
        minimum=minimum,
        minimum_volume=minimum_volume,
    )
    snapshot = _snapshot(particles, sidecars)

    returned = representative_volume_scaling_step_gpu(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )

    assert returned[0] is particles
    assert returned[1] is demand
    assert returned[2] is resolved
    wp.synchronize_device("cpu")
    _assert_snapshot(snapshot, particles, sidecars)
    npt.assert_array_equal(resolved.numpy(), np.ones(2))


def _assert_p4_subnormal_demand_scaling(device: str) -> None:
    """Assert an immutable P4 selection scales every selected row lane."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles, _, _ = _custom_state(
        np.ones((1, 3, 1), dtype=np.float64),
        np.array([[2.0, 4.0, 8.0]], dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.zeros(1, dtype=np.int32),
        device=device,
    )
    particles.volume = wp.array([1.0], dtype=wp.float64, device=device)
    demand = wp.array(
        [np.nextafter(np.float64(0.0), np.float64(1.0))],
        dtype=wp.float64,
        device=device,
    )
    flags = wp.array([1], dtype=wp.int32, device=device)
    requested = wp.array([0.5], dtype=wp.float64, device=device)
    minimum = wp.array([0.25], dtype=wp.float64, device=device)
    minimum_volume = wp.array([0.1], dtype=wp.float64, device=device)
    resolved = wp.zeros(1, dtype=wp.float64, device=device)

    representative_volume_scaling_step_gpu(
        particles, demand, flags, requested, minimum, minimum_volume, resolved
    )
    wp.synchronize_device(device)
    npt.assert_array_equal(particles.concentration.numpy(), [[1.0, 2.0, 4.0]])
    npt.assert_array_equal(particles.volume.numpy(), [0.5])
    npt.assert_array_equal(demand.numpy(), [0.0])
    npt.assert_array_equal(resolved.numpy(), [0.5])


def test_gpu_representative_volume_scaling_subnormal_demand_is_atomic() -> None:
    """Warp CPU P4 selection remains immutable when demand underflows on write."""
    _assert_p4_subnormal_demand_scaling(device="cpu")


@pytest.mark.cuda
def test_gpu_representative_volume_scaling_cuda_matches_subnormal_oracle() -> (
    None
):
    """CUDA P4 subnormal-demand parity is optional and cleanly guarded."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    _assert_p4_subnormal_demand_scaling(device="cuda:0")


def test_gpu_representative_volume_scaling_empty_invalid_density_rejects() -> (
    None
):
    """Empty P4 inputs still reject invalid global density before output writes."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    particles = SimpleNamespace(
        masses=wp.zeros((0, 1, 1), dtype=wp.float64, device="cpu"),
        concentration=wp.zeros((0, 1), dtype=wp.float64, device="cpu"),
        charge=wp.zeros((0, 1), dtype=wp.float64, device="cpu"),
        density=wp.array([-1.0], dtype=wp.float64, device="cpu"),
        volume=wp.zeros(0, dtype=wp.float64, device="cpu"),
    )
    demand = wp.zeros(0, dtype=wp.float64, device="cpu")
    flags = wp.zeros(0, dtype=wp.int32, device="cpu")
    requested = wp.zeros(0, dtype=wp.float64, device="cpu")
    minimum = wp.zeros(0, dtype=wp.float64, device="cpu")
    minimum_volume = wp.zeros(0, dtype=wp.float64, device="cpu")
    resolved = wp.zeros(0, dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="representative scaling sidecars"):
        representative_volume_scaling_step_gpu(
            particles,
            demand,
            flags,
            requested,
            minimum,
            minimum_volume,
            resolved,
        )
    npt.assert_array_equal(resolved.numpy(), np.empty(0, dtype=np.float64))


def _direct_particle_ledger(
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    volume: np.ndarray,
    *,
    include_volume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently reduce per-box number, species mass, and charge."""
    number = np.asarray(
        np.sum(concentration, axis=1, dtype=np.float64), dtype=np.float64
    )
    mass = np.einsum(
        "bn,bns->bs", concentration, masses, dtype=np.float64, optimize=True
    )
    signed_charge = np.asarray(
        np.sum(concentration * charge, axis=1, dtype=np.float64),
        dtype=np.float64,
    )
    if include_volume:
        number *= volume
        mass *= volume[:, None]
        signed_charge *= volume
    return number, mass, signed_charge


def _assert_direct_ledger(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Keep tight conservation assertions separate from field parity."""
    for actual_values, expected_values in zip(actual, expected, strict=True):
        npt.assert_allclose(
            actual_values, expected_values, rtol=1e-12, atol=1e-30
        )


def _commit_fixed_source(
    particles: Any,
    indices: np.ndarray,
    source_masses: np.ndarray,
    source_concentration: np.ndarray,
    source_charge: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Commit fixed external fixture values only into available P2/P4 slots."""
    wp = _warp()
    wp.synchronize_device(device)
    masses = particles.masses.numpy()
    concentration = particles.concentration.numpy()
    charge = particles.charge.numpy()
    boxes, _, species = masses.shape
    if (
        indices.shape != (boxes,)
        or source_masses.shape != (boxes, species)
        or source_concentration.shape != (boxes,)
        or source_charge.shape != (boxes,)
    ):
        raise ValueError("invalid test source fixture shape")
    source_mass_state = np.zeros_like(masses)
    source_concentration_state = np.zeros_like(concentration)
    source_charge_state = np.zeros_like(charge)
    for box, index in enumerate(indices):
        if concentration[box, index] != 0.0:
            raise ValueError("test source fixture requires an available slot")
        source_mass_state[box, index] = source_masses[box]
        source_concentration_state[box, index] = source_concentration[box]
        source_charge_state[box, index] = source_charge[box]
        masses[box, index] = source_masses[box]
        concentration[box, index] = source_concentration[box]
        charge[box, index] = source_charge[box]
    wp.copy(particles.masses, wp.array(masses, dtype=wp.float64, device=device))
    wp.copy(
        particles.concentration,
        wp.array(concentration, dtype=wp.float64, device=device),
    )
    wp.copy(particles.charge, wp.array(charge, dtype=wp.float64, device=device))
    return _direct_particle_ledger(
        source_mass_state,
        source_concentration_state,
        source_charge_state,
        np.ones(boxes, dtype=np.float64),
        include_volume=False,
    )


def _assert_source_and_resampling_matrix(device: str) -> None:
    """Exercise sparse, zero-demand sibling, and repeated P2 ledger evidence."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.array(
        [
            [[1e-21, 2e-18], [3e-15, 4e-12], [5e-9, 6e-6], [0.0, 0.0]],
            [[7e-3, 8.0], [9.0, 1e-12], [2e-18, 3e-15], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    concentration = np.array([[1.0, 2.0, 4.0, 0.0], [3.0, 5.0, 7.0, 0.0]])
    charge = np.array([[1.0, -2.0, 3.0, 0.0], [-4.0, 5.0, -6.0, 0.0]])
    counts = np.array([1, 0], dtype=np.int32)
    particles, warp_counts, buffers = _custom_state(
        masses,
        concentration,
        charge,
        np.array([1000.0, 2000.0]),
        counts,
        device,
    )
    before = _direct_particle_ledger(
        masses, concentration, charge, np.ones(2), include_volume=False
    )
    identities = (id(particles), id(buffers), id(warp_counts))
    assert resampling_step_gpu(particles, warp_counts, buffers) is particles
    wp.synchronize_device(device)
    source = _commit_fixed_source(
        particles,
        np.array([2, 3], dtype=np.intp),
        np.array([[1e-24, 2e-20], [3e-16, 4e-12]], dtype=np.float64),
        np.array([11.0, 0.25], dtype=np.float64),
        np.array([2.0, -3.0], dtype=np.float64),
        device,
    )
    wp.synchronize_device(device)
    _assert_direct_ledger(
        _direct_particle_ledger(
            particles.masses.numpy(),
            particles.concentration.numpy(),
            particles.charge.numpy(),
            particles.volume.numpy(),
            include_volume=False,
        ),
        tuple(pre + added for pre, added in zip(before, source, strict=True)),
    )
    after_source = _direct_particle_ledger(
        particles.masses.numpy(),
        particles.concentration.numpy(),
        particles.charge.numpy(),
        particles.volume.numpy(),
        include_volume=False,
    )
    assert resampling_step_gpu(particles, warp_counts, buffers) is particles
    wp.synchronize_device(device)
    _assert_direct_ledger(
        _direct_particle_ledger(
            particles.masses.numpy(),
            particles.concentration.numpy(),
            particles.charge.numpy(),
            particles.volume.numpy(),
            include_volume=False,
        ),
        after_source,
    )
    assert identities == (id(particles), id(buffers), id(warp_counts))
    npt.assert_array_equal(buffers.retained_indices.numpy()[0, :2], [0, 1])
    npt.assert_array_equal(buffers.released_indices.numpy()[0, :1], [2])
    assert particles.masses.shape == (2, 4, 2)


def test_resampling_source_fixture_conserves_each_box_and_species() -> None:
    """Warp CPU P2 ledger includes only already-admitted external source state."""
    _assert_source_and_resampling_matrix("cpu")


@pytest.mark.cuda
def test_resampling_cuda_source_fixture_conserves_each_box_and_species() -> (
    None
):
    """CUDA source-plus-particle P2 evidence remains optional."""
    if not _warp().is_cuda_available():
        pytest.skip("CUDA is unavailable")
    _assert_source_and_resampling_matrix("cuda:0")


def test_resampling_active_count_rejection_preserves_count_and_buffers() -> (
    None
):
    """Count equal to active capacity is a write-free preflight failure."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    particles, _, buffers = _state()
    wp = _warp()
    counts = wp.array([3], dtype=wp.int32, device="cpu")
    count_container = SimpleNamespace(counts=counts)
    snapshot = _snapshot(particles, count_container, buffers)
    with pytest.raises(ValueError, match="invalid values"):
        resampling_step_gpu(particles, counts, buffers)
    _assert_snapshot(snapshot, particles, count_container, buffers)


def test_resampling_full_capacity_fixture_conserves_particle_ledger() -> None:
    """A five-slot, two-species full row preserves each particle ledger lane."""
    from particula.gpu.kernels.exhaustion import resampling_step_gpu

    masses = np.array(
        [
            [
                [1.0e-21, 2.0e-18],
                [3.0e-15, 4.0e-12],
                [5.0e-9, 6.0e-6],
                [7.0e-3, 8.0],
                [9.0, 1.0e-12],
            ]
        ],
        dtype=np.float64,
    )
    concentration = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    charge = np.array([[1.0, -2.0, 3.0, -4.0, 5.0]])
    particles, counts, buffers = _custom_state(
        masses,
        concentration,
        charge,
        np.array([1000.0, 2000.0]),
        np.array([2], dtype=np.int32),
    )
    before = _direct_particle_ledger(
        masses,
        concentration,
        charge,
        np.ones(1, dtype=np.float64),
        include_volume=False,
    )
    assert resampling_step_gpu(particles, counts, buffers) is particles
    _assert_direct_ledger(
        _direct_particle_ledger(
            particles.masses.numpy(),
            particles.concentration.numpy(),
            particles.charge.numpy(),
            particles.volume.numpy(),
            include_volume=False,
        ),
        before,
    )
    npt.assert_array_equal(buffers.released_indices.numpy()[0, :2], [3, 4])


def _assert_p4_source_ledger(device: str) -> None:
    """Check selected P4 volume accounting before fixed source insertion."""
    wp = _warp()
    from particula.gpu.kernels.exhaustion import (
        representative_volume_scaling_step_gpu,
    )

    masses = np.array(
        [
            [[1e-21, 2e-18], [3e-15, 4e-12], [0.0, 0.0]],
            [[5e-9, 6e-6], [7e-3, 8.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    concentration = np.array([[2.0, 3.0, 0.0], [5.0, 7.0, 0.0]])
    charge = np.array([[1.0, -2.0, 0.0], [3.0, -4.0, 0.0]])
    particles, _, _ = _custom_state(
        masses,
        concentration,
        charge,
        np.array([1000.0, 2000.0]),
        np.zeros(2, dtype=np.int32),
        device,
    )
    particles.volume = wp.array([4.0, 9.0], dtype=wp.float64, device=device)
    before = _direct_particle_ledger(
        masses, concentration, charge, np.array([4.0, 9.0]), include_volume=True
    )
    untouched = tuple(
        value.numpy()[1].copy() for value in vars(particles).values()
    )
    demand = wp.array([3.0, 2.0], dtype=wp.float64, device=device)
    resolved = wp.zeros(2, dtype=wp.float64, device=device)
    returned = representative_volume_scaling_step_gpu(
        particles,
        demand,
        wp.array([1, 0], dtype=wp.int32, device=device),
        wp.array([0.5, 0.8], dtype=wp.float64, device=device),
        wp.array([0.25, 0.5], dtype=wp.float64, device=device),
        wp.array([1.0, 1.0], dtype=wp.float64, device=device),
        resolved,
    )
    assert returned == (particles, demand, resolved)
    source = _commit_fixed_source(
        particles,
        np.array([2, 2], dtype=np.intp),
        np.array([[1e-24, 2e-20], [0.0, 0.0]], dtype=np.float64),
        np.array([6.0, 0.0], dtype=np.float64),
        np.array([2.0, 0.0]),
        device,
    )
    wp.synchronize_device(device)
    source_volume = tuple(
        values * particles.volume.numpy()[:, None]
        if values.ndim == 2
        else values * particles.volume.numpy()
        for values in source
    )
    scale = np.array([0.25, 1.0])
    expected = tuple(
        values * scale[:, None] if values.ndim == 2 else values * scale
        for values in before
    )
    _assert_direct_ledger(
        _direct_particle_ledger(
            particles.masses.numpy(),
            particles.concentration.numpy(),
            particles.charge.numpy(),
            particles.volume.numpy(),
            include_volume=True,
        ),
        tuple(
            base + added
            for base, added in zip(expected, source_volume, strict=True)
        ),
    )
    npt.assert_allclose(demand.numpy(), [1.5, 2.0])
    npt.assert_allclose(resolved.numpy(), [0.5, 1.0])
    for value, expected_value in zip(
        vars(particles).values(), untouched, strict=True
    ):
        npt.assert_array_equal(value.numpy()[1], expected_value)


def test_gpu_representative_scaling_source_fixture_uses_volume_ledger() -> None:
    """Warp CPU P4 selected rows preserve explicit volume-ledger accounting."""
    _assert_p4_source_ledger("cpu")


@pytest.mark.cuda
def test_gpu_representative_scaling_cuda_source_fixture_uses_volume_ledger() -> (
    None
):
    """CUDA P4 source-ledger evidence is optional and guarded."""
    if not _warp().is_cuda_available():
        pytest.skip("CUDA is unavailable")
    _assert_p4_source_ledger("cuda:0")
