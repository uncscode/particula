"""Tests for direct Warp particle-slot diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from particula.particles.particle_data import ParticleData
from particula.particles.slot_management import get_slot_diagnostics

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


def _warp():
    """Import Warp only when a marked test executes."""
    return pytest.importorskip("warp")


def _particles(masses, concentration, charge, device="cpu"):
    """Build a minimal Warp particle container with protected sentinels."""
    wp = _warp()
    return SimpleNamespace(
        masses=wp.array(masses, dtype=wp.float64, device=device),
        concentration=wp.array(concentration, dtype=wp.float64, device=device),
        charge=wp.array(charge, dtype=wp.float64, device=device),
        density="not a Warp array",
        volume="not a Warp array",
    )


def _sidecars(n_boxes, n_particles, device="cpu"):
    """Build stale diagnostic sidecars to prove complete overwrite."""
    wp = _warp()
    return (
        wp.full((n_boxes, n_particles), 77, dtype=wp.int32, device=device),
        wp.full(n_boxes, 77, dtype=wp.int32, device=device),
        wp.full(n_boxes, 77, dtype=wp.int32, device=device),
    )


def _diagnostics(particles, free_indices, active_counts, free_counts):
    """Call the concrete-only direct GPU diagnostic boundary."""
    from particula.gpu.kernels.slot_management import get_slot_diagnostics_gpu

    return get_slot_diagnostics_gpu(
        particles, free_indices, active_counts, free_counts
    )


def _cpu_particles(masses, concentration, charge):
    """Build equivalent CPU particle data for diagnostic-reference checks."""
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=np.ones(masses.shape[2], dtype=np.float64),
        volume=np.ones(masses.shape[0], dtype=np.float64),
    )


def test_slot_diagnostics_matches_cpu_oracle_and_preserves_particles():
    """Write exact ascending diagnostics without reading protected fields."""
    masses = np.array(
        [
            [[1.0, 2.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 1.0], [0.0, 0.0], [4.0, 0.0]],
        ],
        dtype=np.float64,
    )
    concentration = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 3.0]])
    charge = np.array([[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    particles = _particles(masses, concentration, charge)
    sidecars = _sidecars(2, 4)
    particle_values = tuple(
        field.numpy().copy()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )

    returned = _diagnostics(particles, *sidecars)

    assert returned[0] is sidecars[0]
    assert returned[1] is sidecars[1]
    assert returned[2] is sidecars[2]
    expected = get_slot_diagnostics(
        _cpu_particles(masses, concentration, charge)
    )
    for sidecar, reference in zip(sidecars, expected, strict=True):
        npt.assert_array_equal(sidecar.numpy(), reference)
    for field, expected in zip(
        (particles.masses, particles.concentration, particles.charge),
        particle_values,
        strict=True,
    ):
        npt.assert_array_equal(field.numpy(), expected)
    assert particles.density == "not a Warp array"
    assert particles.volume == "not a Warp array"


@pytest.mark.parametrize(
    "masses, concentration, charge",
    [
        (
            np.array([[[1.0], [0.0]], [[0.0], [2.0]]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, -1.0]]),
        ),
        (
            np.empty((1, 2, 0)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        ),
    ],
)
def test_slot_diagnostics_matches_cpu_oracle_at_classification_boundaries(
    masses, concentration, charge
):
    """Match CPU diagnostics for mixed rows and zero-species free slots."""
    particles = _particles(masses, concentration, charge)
    sidecars = _sidecars(masses.shape[0], masses.shape[1])

    _diagnostics(particles, *sidecars)

    expected = get_slot_diagnostics(
        _cpu_particles(masses, concentration, charge)
    )
    for sidecar, reference in zip(sidecars, expected, strict=True):
        npt.assert_array_equal(sidecar.numpy(), reference)


def test_slot_diagnostics_all_free_and_all_active_rows():
    """Overwrite stale tails for free and full active rows."""
    particles = _particles(
        np.array([[[0.0], [0.0]], [[1.0], [2.0]]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.zeros((2, 2)),
    )
    sidecars = _sidecars(2, 2)

    _diagnostics(particles, *sidecars)

    npt.assert_array_equal(sidecars[0].numpy(), [[0, 1], [-1, -1]])
    npt.assert_array_equal(sidecars[1].numpy(), [0, 2])
    npt.assert_array_equal(sidecars[2].numpy(), [2, 0])


@pytest.mark.parametrize(
    "masses, concentration, charge",
    [
        (np.array([[[1.0]]]), np.array([[0.0]]), np.array([[0.0]])),
        (np.array([[[0.0]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[-1.0]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[np.nan]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[np.inf]]]), np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[-1.0]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[np.nan]]), np.array([[0.0]])),
        (np.array([[[0.0]]]), np.array([[0.0]]), np.array([[1.0]])),
        (np.array([[[1.0]]]), np.array([[np.inf]]), np.array([[0.0]])),
        (np.array([[[1.0]]]), np.array([[1.0]]), np.array([[np.nan]])),
        (np.array([[[1.0]]]), np.array([[1.0]]), np.array([[np.inf]])),
        (
            np.array([[[np.finfo(np.float64).max, np.finfo(np.float64).max]]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
        ),
    ],
)
def test_slot_diagnostics_rejects_invalid_state_without_output_mutation(
    masses, concentration, charge
):
    """Reject every non-active, non-free representative before writer launch."""
    cpu_particles = _cpu_particles(masses, concentration, charge)
    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(cpu_particles)
    particles = _particles(masses, concentration, charge)
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


def test_slot_diagnostics_rejects_schema_before_output_mutation():
    """Reject an invalid sidecar schema while retaining stale diagnostics."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    free_indices, active_counts, free_counts = _sidecars(1, 1)
    original = free_indices.numpy().copy()
    wp = _warp()
    wrong_counts = wp.zeros(1, dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError):
        _diagnostics(particles, free_indices, wrong_counts, free_counts)

    npt.assert_array_equal(free_indices.numpy(), original)
    assert active_counts.numpy()[0] == 77
    assert free_counts.numpy()[0] == 77


@pytest.mark.parametrize(
    "name, replacement",
    [
        ("free_indices", "not a Warp array"),
        ("active_counts", "not a Warp array"),
        ("free_counts", "not a Warp array"),
    ],
)
def test_slot_diagnostics_rejects_non_warp_sidecars_without_writes(
    name, replacement
):
    """Reject every non-Warp diagnostic sidecar before any output write."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    original_sidecars = _sidecars(1, 1)
    sidecars = list(original_sidecars)
    original = tuple(sidecar.numpy().copy() for sidecar in original_sidecars)
    sidecars[("free_indices", "active_counts", "free_counts").index(name)] = (
        replacement
    )

    with pytest.raises(ValueError, match=rf"^{name} must be a Warp array[.]$"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(original_sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


def test_slot_diagnostics_converts_inaccessible_fields_to_value_error():
    """Convert hostile field access into a stable preflight error."""

    class InaccessibleParticles:
        """Particle-like object with a failing mass-field accessor."""

        @property
        def masses(self):
            """Raise an exception that must not escape the API boundary."""
            raise RuntimeError("unavailable")

    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError, match="^particles[.]masses must be"):
        _diagnostics(InaccessibleParticles(), *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


def test_slot_diagnostics_converts_warp_like_detection_errors_to_value_error(
    monkeypatch,
):
    """Treat internal Warp-like detection failures as stable validation errors."""
    import particula.gpu.kernels.slot_management as slot_management

    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    def _boom(_values):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(slot_management, "_is_warp_array_like", _boom)

    with pytest.raises(ValueError, match="^particles[.]masses must be"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


def test_slot_diagnostics_converts_masses_metadata_errors_to_value_error(
    monkeypatch,
):
    """Treat masses metadata failures as stable validation errors."""
    import particula.gpu.kernels.slot_management as slot_management

    class BrokenMasses:
        """Mass field stub whose schema metadata access fails."""

        dtype = _warp().float64
        ndim = 3

        @property
        def shape(self):
            """Raise from the post-Warp schema path."""
            raise RuntimeError("broken shape")

        @property
        def device(self):
            """Return a device placeholder if metadata access continues."""
            return "cpu"

    def _return_broken(_values, name, dtype, ndim, shape=None, device=None):
        return BrokenMasses()

    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    monkeypatch.setattr(
        slot_management, "_validate_array_schema", _return_broken
    )

    with pytest.raises(ValueError, match="^particles[.]masses has invalid"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


@pytest.mark.parametrize("missing_field", ["masses", "concentration", "charge"])
def test_slot_diagnostics_rejects_missing_particle_field_without_writes(
    missing_field,
):
    """Reject missing classification fields before changing diagnostics."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    delattr(particles, missing_field)
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError, match=rf"^particles[.]{missing_field}"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("masses", np.zeros((1, 1, 1))),
        ("concentration", np.zeros((1, 1))),
        ("charge", np.zeros((1, 1))),
    ],
)
def test_slot_diagnostics_rejects_non_warp_particle_field_without_writes(
    field, replacement
):
    """Reject non-Warp classification fields before changing diagnostics."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    setattr(particles, field, replacement)
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError, match=rf"^particles[.]{field}"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


@pytest.mark.parametrize(
    "field, replacement_factory",
    [
        ("masses", lambda wp: wp.zeros((1, 1, 1), dtype=wp.float32)),
        ("masses", lambda wp: wp.zeros((1, 1), dtype=wp.float64)),
        ("concentration", lambda wp: wp.zeros((1, 1), dtype=wp.float32)),
        ("concentration", lambda wp: wp.zeros(1, dtype=wp.float64)),
        ("concentration", lambda wp: wp.zeros((1, 2), dtype=wp.float64)),
        ("charge", lambda wp: wp.zeros((1, 1), dtype=wp.float32)),
        ("charge", lambda wp: wp.zeros(1, dtype=wp.float64)),
        ("charge", lambda wp: wp.zeros((1, 2), dtype=wp.float64)),
    ],
)
def test_slot_diagnostics_rejects_particle_schema_errors_without_writes(
    field, replacement_factory
):
    """Reject every classification-field schema error before diagnostics write."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    setattr(particles, field, replacement_factory(_warp()))
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


@pytest.mark.parametrize(
    "sidecar_index, replacement_factory",
    [
        (0, lambda wp: wp.zeros((1, 1), dtype=wp.float64)),
        (0, lambda wp: wp.zeros(1, dtype=wp.int32)),
        (0, lambda wp: wp.zeros((1, 2), dtype=wp.int32)),
        (1, lambda wp: wp.zeros(1, dtype=wp.float64)),
        (1, lambda wp: wp.zeros((1, 1), dtype=wp.int32)),
        (1, lambda wp: wp.zeros(2, dtype=wp.int32)),
        (2, lambda wp: wp.zeros(1, dtype=wp.float64)),
        (2, lambda wp: wp.zeros((1, 1), dtype=wp.int32)),
        (2, lambda wp: wp.zeros(2, dtype=wp.int32)),
    ],
)
def test_slot_diagnostics_rejects_sidecar_schema_errors_without_writes(
    sidecar_index, replacement_factory
):
    """Reject malformed output schemas before changing valid stale sidecars."""
    particles = _particles(
        np.array([[[1.0]]]), np.array([[1.0]]), np.array([[0.0]])
    )
    original_sidecars = _sidecars(1, 1)
    sidecars = list(original_sidecars)
    sidecars[sidecar_index] = replacement_factory(_warp())
    original = tuple(sidecar.numpy().copy() for sidecar in original_sidecars)

    with pytest.raises(ValueError):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(original_sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


def test_slot_diagnostics_zero_particles_overwrites_counts():
    """Handle an empty particle extent without a classification launch."""
    particles = _particles(
        np.empty((2, 0, 1)), np.empty((2, 0)), np.empty((2, 0))
    )
    sidecars = _sidecars(2, 0)

    _diagnostics(particles, *sidecars)

    npt.assert_array_equal(sidecars[1].numpy(), [0, 0])
    npt.assert_array_equal(sidecars[2].numpy(), [0, 0])


def test_slot_diagnostics_zero_boxes_returns_without_writes():
    """Accept an empty box extent without launching a diagnostic kernel."""
    particles = _particles(
        np.empty((0, 2, 1)), np.empty((0, 2)), np.empty((0, 2))
    )
    sidecars = _sidecars(0, 2)

    returned = _diagnostics(particles, *sidecars)

    assert returned[0] is sidecars[0]
    assert returned[1] is sidecars[1]
    assert returned[2] is sidecars[2]


def test_slot_diagnostics_zero_species_all_zero_slots_are_free():
    """Treat zero-species all-zero records as free rather than active."""
    particles = _particles(
        np.empty((1, 2, 0)), np.zeros((1, 2)), np.zeros((1, 2))
    )
    sidecars = _sidecars(1, 2)

    _diagnostics(particles, *sidecars)

    npt.assert_array_equal(sidecars[0].numpy(), [[0, 1]])
    npt.assert_array_equal(sidecars[1].numpy(), [0])
    npt.assert_array_equal(sidecars[2].numpy(), [2])


@pytest.mark.parametrize(
    "concentration, charge",
    [
        (np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[0.0]]), np.array([[1.0]])),
    ],
)
def test_slot_diagnostics_zero_species_rejects_nonfree_slots(
    concentration, charge
):
    """Reject zero-species records that cannot satisfy the free predicate."""
    cpu_particles = _cpu_particles(np.empty((1, 1, 0)), concentration, charge)
    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        get_slot_diagnostics(cpu_particles)
    particles = _particles(np.empty((1, 1, 0)), concentration, charge)
    sidecars = _sidecars(1, 1)
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError, match="^Invalid particle slot state[.]$"):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)


@pytest.mark.cuda
def test_slot_diagnostics_cuda_skips_cleanly_when_unavailable():
    """Execute the same direct contract on CUDA when the device is available."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    particles = _particles(
        np.array([[[1.0], [0.0]]]),
        np.array([[1.0, 0.0]]),
        np.zeros((1, 2)),
        device="cuda:0",
    )
    sidecars = _sidecars(1, 2, device="cuda:0")

    _diagnostics(particles, *sidecars)

    npt.assert_array_equal(sidecars[0].numpy(), [[1, -1]])
    npt.assert_array_equal(sidecars[1].numpy(), [1])
    npt.assert_array_equal(sidecars[2].numpy(), [1])


@pytest.mark.cuda
@pytest.mark.parametrize("sidecar_index", [0, 1, 2])
def test_slot_diagnostics_rejects_cross_device_sidecars_without_writes(
    sidecar_index,
):
    """Reject a CPU sidecar before changing CUDA particle or output storage."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    particles = _particles(
        np.array([[[1.0], [0.0]]]),
        np.array([[1.0, 0.0]]),
        np.zeros((1, 2)),
        device="cuda:0",
    )
    original_sidecars = _sidecars(1, 2, device="cuda:0")
    sidecars = list(original_sidecars)
    sidecars[sidecar_index] = _sidecars(1, 2, device="cpu")[sidecar_index]
    particle_values = tuple(
        field.numpy().copy()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
        )
    )
    original = tuple(sidecar.numpy().copy() for sidecar in sidecars)

    with pytest.raises(ValueError):
        _diagnostics(particles, *sidecars)

    for sidecar, expected in zip(sidecars, original, strict=True):
        npt.assert_array_equal(sidecar.numpy(), expected)
    for field, expected in zip(
        (particles.masses, particles.concentration, particles.charge),
        particle_values,
        strict=True,
    ):
        npt.assert_array_equal(field.numpy(), expected)


def test_slot_diagnostics_is_not_kernel_package_export():
    """Keep the direct diagnostic boundary concrete-module-only."""
    import particula.gpu.kernels as kernels

    assert "get_slot_diagnostics_gpu" not in kernels.__all__
