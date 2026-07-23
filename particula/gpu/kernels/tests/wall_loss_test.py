"""Tests for fixed-shape GPU neutral and charged wall-loss execution."""

from __future__ import annotations

import re
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp


def _warp():
    """Import Warp at test runtime so marker deselection stays safe."""
    return pytest.importorskip("warp")


def _particles(device: str = "cpu"):
    """Create valid explicit float64 particle storage for P3 preflight."""
    wp = _warp()
    from particula.gpu import WarpParticleData

    particles = WarpParticleData()
    particles.masses = wp.array(
        np.array([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [5.0, 6.0]]]),
        dtype=wp.float64,
        device=device,
    )
    particles.concentration = wp.array(
        [[1.0, 0.0], [2.0, 3.0]], dtype=wp.float64, device=device
    )
    particles.charge = wp.array(
        [[0.0, 2.0], [1.0, 0.0]], dtype=wp.float64, device=device
    )
    particles.density = wp.array(
        [1000.0, 1200.0], dtype=wp.float64, device=device
    )
    particles.volume = wp.array([1.0, 2.0], dtype=wp.float64, device=device)
    return particles


def _config(geometry: str = "spherical"):
    """Build one valid neutral configuration for the selected geometry."""
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    if geometry == "spherical":
        return NeutralWallLossConfig("spherical", 0.01, chamber_radius=1.0)
    return NeutralWallLossConfig(
        "rectangular", 0.01, chamber_dimensions=(1.0, 2.0, 3.0)
    )


def _charged_config(geometry: str = "spherical", device: str = "cpu"):
    """Build one valid charged configuration with caller-owned field state."""
    wp = _warp()
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    if geometry == "spherical":
        return NeutralWallLossConfig(
            "spherical",
            0.01,
            chamber_radius=1.0,
            mode="charged",
            wall_potential=-12.0,
            wall_electric_field=3.0,
        )
    return NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_potential=-12.0,
        wall_electric_field=wp.array(
            [-1.0, 2.0, 3.0], dtype=wp.float64, device=device
        ),
    )


def _snapshot(particles, rng_states=None):
    """Snapshot all mutable particle and optional sidecar values."""
    snapshot = {
        name: (
            getattr(particles, name),
            getattr(particles, name).numpy().copy(),
        )
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    if rng_states is not None:
        snapshot["rng_states"] = (rng_states, rng_states.numpy().copy())
    return snapshot


def _assert_snapshot_unchanged(particles, snapshot, rng_states=None) -> None:
    """Assert preflight retained field identity and exact caller values."""
    for name in ("masses", "concentration", "charge", "density", "volume"):
        original, expected = snapshot[name]
        assert getattr(particles, name) is original
        npt.assert_array_equal(getattr(particles, name).numpy(), expected)
    if rng_states is not None:
        original, expected = snapshot["rng_states"]
        assert rng_states is original
        npt.assert_array_equal(rng_states.numpy(), expected)


def _assert_slots_preserved_or_cleared(particles, snapshot) -> None:
    """Assert P5 only clears complete active slots and preserves protected data."""
    original_masses = snapshot["masses"][1]
    original_concentration = snapshot["concentration"][1]
    original_charge = snapshot["charge"][1]
    for box, particle in np.ndindex(original_concentration.shape):
        current_mass = particles.masses.numpy()[box, particle]
        current_concentration = particles.concentration.numpy()[box, particle]
        current_charge = particles.charge.numpy()[box, particle]
        if (
            original_concentration[box, particle] > 0.0
            and np.sum(original_masses[box, particle]) > 0.0
            and current_concentration == 0.0
            and np.all(current_mass == 0.0)
        ):
            assert current_charge == 0.0
        else:
            npt.assert_array_equal(current_mass, original_masses[box, particle])
            assert (
                current_concentration == original_concentration[box, particle]
            )
            assert current_charge == original_charge[box, particle]
    npt.assert_array_equal(particles.density.numpy(), snapshot["density"][1])
    npt.assert_array_equal(particles.volume.numpy(), snapshot["volume"][1])


def test_public_step_is_lazy_and_config_is_concrete_only() -> None:
    """Expose only the direct step from the lazy kernel package namespace."""
    from particula.gpu import kernels
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import (
        NeutralWallLossConfig,
    )
    from particula.gpu.kernels.wall_loss import (
        wall_loss_step_gpu as module_step,
    )

    assert wall_loss_step_gpu is module_step
    assert "wall_loss_step_gpu" in kernels.__all__
    assert "NeutralWallLossConfig" not in kernels.__all__
    assert not hasattr(kernels, "NeutralWallLossConfig")
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.geometry = "rectangular"
    assert isinstance(config, NeutralWallLossConfig)


def test_charged_configuration_defaults_and_fields_are_frozen() -> None:
    """P1 appends immutable charged fields without changing legacy defaults."""
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    config = _config()
    assert config.mode == "neutral"
    assert config.wall_potential == 0.0
    assert config.wall_electric_field == 0.0
    for field, value in (
        ("mode", "charged"),
        ("wall_potential", 2.0),
        ("wall_electric_field", 1.0),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(config, field, value)
    assert _charged_config().mode == "charged"
    assert isinstance(config, NeutralWallLossConfig)


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"mode": "other"}, ValueError, "config.mode"),
        ({"wall_potential": True}, TypeError, "config.wall_potential"),
        ({"wall_potential": np.array(1.0)}, TypeError, "config.wall_potential"),
        ({"wall_potential": "1"}, TypeError, "config.wall_potential"),
        ({"wall_potential": 1j}, TypeError, "config.wall_potential"),
        ({"wall_potential": np.nan}, ValueError, "config.wall_potential"),
        ({"wall_potential": np.inf}, ValueError, "config.wall_potential"),
        ({"wall_potential": -np.inf}, ValueError, "config.wall_potential"),
        (
            {"wall_electric_field": True},
            TypeError,
            "config.wall_electric_field",
        ),
        (
            {"wall_electric_field": np.array(1.0)},
            TypeError,
            "config.wall_electric_field",
        ),
        ({"wall_electric_field": "1"}, TypeError, "config.wall_electric_field"),
        ({"wall_electric_field": 1j}, TypeError, "config.wall_electric_field"),
        (
            {"wall_electric_field": np.nan},
            ValueError,
            "config.wall_electric_field",
        ),
        (
            {"wall_electric_field": -np.inf},
            ValueError,
            "config.wall_electric_field",
        ),
    ],
)
def test_charged_scalar_configuration_rejects_before_particles(
    updates, error, match
) -> None:
    """Configuration scalar contracts fail before any particle-field access."""
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    arguments = {"mode": "charged", **updates}
    config = NeutralWallLossConfig(
        "spherical", 0.01, chamber_radius=1.0, **arguments
    )
    with pytest.raises(error, match=match):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)


@pytest.mark.parametrize(
    ("field_factory", "match"),
    [
        (lambda wp: 3.0, "Warp array"),
        (lambda wp: np.array([1.0, 2.0, 3.0]), "Warp array"),
        (lambda wp: wp.zeros((1, 3), dtype=wp.float64, device="cpu"), "rank 1"),
        (lambda wp: wp.zeros(2, dtype=wp.float64, device="cpu"), "shape"),
        (
            lambda wp: wp.zeros(3, dtype=wp.float32, device="cpu"),
            "dtype float64",
        ),
    ],
)
def test_charged_rectangular_field_schema_rejection_is_atomic(
    field_factory, match
) -> None:
    """Invalid P1 rectangular fields reject before particle scans or RNG reset."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    particles = _particles()
    field = field_factory(wp)
    field_before = field.numpy().copy() if hasattr(field, "numpy") else None
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    config = NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_electric_field=field,
    )
    with pytest.raises(ValueError, match=f"wall_electric_field.*{match}"):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=config,
            rng_states=states,
            initialize_rng=True,
        )
    _assert_snapshot_unchanged(particles, snapshot, states)
    if field_before is not None:
        npt.assert_array_equal(field.numpy(), field_before)


def test_charged_rectangular_nonfinite_field_rejection_is_atomic() -> None:
    """A nonfinite charged field fails before charge scans or caller writes."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    field = wp.array([1.0, np.nan, 3.0], dtype=wp.float64, device="cpu")
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    field_before = field.numpy().copy()
    config = _charged_config("rectangular")
    object.__setattr__(config, "wall_electric_field", field)

    with pytest.raises(
        ValueError, match="config.wall_electric_field must be finite"
    ):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=config,
            rng_states=states,
            initialize_rng=True,
        )

    _assert_snapshot_unchanged(particles, snapshot, states)
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


@pytest.mark.cuda
def test_charged_rectangular_wrong_device_field_rejection_is_atomic() -> None:
    """A CUDA charged field cannot accompany CPU particle storage."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    config = _charged_config("rectangular", device="cuda:0")
    field = config.wall_electric_field
    field_before = field.numpy().copy()
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)

    with pytest.raises(ValueError, match="wall_electric_field device"):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=config,
            rng_states=states,
            initialize_rng=True,
        )

    _assert_snapshot_unchanged(particles, snapshot, states)
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


def test_charged_rectangular_field_value_error_precedes_charge_error() -> None:
    """P1 scans the charged field before scanning signed particle charge."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    charges = particles.charge.numpy()
    charges[0, 0] = np.nan
    particles.charge = wp.array(charges, dtype=wp.float64, device="cpu")
    config = _charged_config("rectangular")
    bad_field = config.wall_electric_field.numpy()
    bad_field[0] = np.nan
    object.__setattr__(
        config,
        "wall_electric_field",
        wp.array(bad_field, dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(
        ValueError, match="config.wall_electric_field must be finite"
    ):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=config)


def test_neutral_mode_rejects_rectangular_warp_field_before_particles() -> None:
    """Neutral mode accepts only its legacy scalar electric-field value."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    config = NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        wall_electric_field=wp.zeros(3, dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(TypeError, match="config.wall_electric_field"):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_zero_charge_charged_mode_matches_neutral_execution(
    geometry: str,
) -> None:
    """Zero-charge charged slots retain the exact neutral path and RNG state."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    neutral_particles = _particles()
    charged_particles = _particles()
    charges = np.zeros((2, 2), dtype=np.float64)
    neutral_particles.charge = wp.array(charges, dtype=wp.float64, device="cpu")
    charged_particles.charge = wp.array(charges, dtype=wp.float64, device="cpu")
    neutral_rng = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    charged_rng = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    wall_loss_step_gpu(
        neutral_particles,
        298.15,
        101325.0,
        1.0,
        config=_config(geometry),
        rng_states=neutral_rng,
    )
    wall_loss_step_gpu(
        charged_particles,
        298.15,
        101325.0,
        1.0,
        config=_charged_config(geometry),
        rng_states=charged_rng,
    )
    for field in ("masses", "concentration", "charge", "density", "volume"):
        npt.assert_array_equal(
            getattr(charged_particles, field).numpy(),
            getattr(neutral_particles, field).numpy(),
        )
    npt.assert_array_equal(charged_rng.numpy(), neutral_rng.numpy())


def test_charged_rectangular_zero_time_preserves_field_and_rng() -> None:
    """Charged rectangular zero time fully preflights without caller writes."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    config = _charged_config("rectangular")
    field = config.wall_electric_field
    field_before = field.numpy().copy()
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    assert (
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            0.0,
            config=config,
            rng_states=states,
            initialize_rng=True,
        )
        is particles
    )
    _assert_snapshot_unchanged(particles, snapshot, states)
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


def test_charged_rectangular_execution_preserves_field_identity_and_values() -> (
    None
):
    """Charged P1 execution never replaces or mutates the field sidecar."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    config = _charged_config("rectangular")
    field = config.wall_electric_field
    field_before = field.numpy().copy()
    rng_states = wp.array([7, 11], dtype=wp.uint32, device="cpu")

    assert (
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=config,
            rng_states=rng_states,
        )
        is particles
    )
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_valid_execution_returns_identity_and_advances_rng_sidecar(
    geometry: str,
) -> None:
    """Both supported geometries only preserve or fully clear active slots."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    rng_states = wp.array([3, 4], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)
    returned = wall_loss_step_gpu(
        particles,
        298.15,
        wp.array([101325.0, 100000.0], dtype=wp.float64, device="cpu"),
        1.0,
        config=_config(geometry),
        rng_seed=np.uint32(42),
        rng_states=rng_states,
        initialize_rng=np.bool_(True),
    )

    assert returned is particles
    _assert_slots_preserved_or_cleared(particles, snapshot)
    assert rng_states is snapshot["rng_states"][0]
    assert not np.array_equal(rng_states.numpy(), snapshot["rng_states"][1])


def test_config_type_fails_before_particle_access() -> None:
    """Reject a noncanonical config before accessing particle storage."""
    from particula.gpu.kernels import wall_loss_step_gpu

    with pytest.raises(TypeError, match="NeutralWallLossConfig"):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=object())


def test_distribution_fails_before_particle_access() -> None:
    """Reject unsupported distribution selector before accessing particles."""
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    config = NeutralWallLossConfig(
        "spherical", 0.01, chamber_radius=1.0, distribution_type="discrete"
    )
    with pytest.raises(ValueError, match="distribution_type"):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "geometry",
            "cylindrical",
            "config.geometry must be 'spherical' or 'rectangular'.",
        ),
        (
            "distribution_type",
            "discrete",
            "config.distribution_type must be 'particle_resolved'.",
        ),
        (
            "mode",
            "other",
            "config.mode must be 'neutral' or 'charged'.",
        ),
    ],
)
def test_selector_rejections_use_stable_exact_messages(
    field: str,
    value: str,
    expected: str,
) -> None:
    """Reject unsupported selectors with the documented exact messages."""
    from particula.gpu.kernels import wall_loss_step_gpu

    config = _config()
    object.__setattr__(config, field, value)

    with pytest.raises(ValueError) as error:
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)

    assert str(error.value) == expected


class _EqualToString:
    """Provide equality-compatible non-string selector test input."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return other == self.value


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("geometry", _EqualToString("spherical"), "geometry"),
        (
            "distribution_type",
            _EqualToString("particle_resolved"),
            "distribution_type",
        ),
        ("mode", _EqualToString("neutral"), "mode"),
    ],
)
def test_selector_requires_exact_string_before_particle_access(
    field: str,
    value: object,
    match: str,
) -> None:
    """Reject equality-compatible non-string selectors during config preflight."""
    from particula.gpu.kernels import wall_loss_step_gpu

    config = _config()
    object.__setattr__(config, field, value)

    with pytest.raises(ValueError, match=match):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("wall_eddy_diffusivity", 10**10000, "wall_eddy_diffusivity"),
        ("chamber_radius", 10**10000, "chamber_radius"),
        ("wall_potential", 10**10000, "wall_potential"),
        ("wall_electric_field", 10**10000, "wall_electric_field"),
    ],
    ids=["eddy_diffusivity", "radius", "potential", "electric_field"],
)
def test_huge_integer_config_scalar_raises_value_error(
    field: str,
    value: int,
    match: str,
) -> None:
    """Normalize overflowing integer configuration values to ValueError."""
    from particula.gpu.kernels import wall_loss_step_gpu

    config = _config()
    object.__setattr__(config, field, value)

    with pytest.raises(ValueError, match=match):
        wall_loss_step_gpu(object(), 298.15, 101325.0, 1.0, config=config)


def test_huge_integer_time_step_raises_value_error() -> None:
    """Normalize an overflowing integer duration to the stable value error."""
    from particula.gpu.kernels import wall_loss_step_gpu

    with pytest.raises(ValueError) as error:
        wall_loss_step_gpu(
            _particles(),
            298.15,
            101325.0,
            10**10000,
            config=_config(),
        )

    assert str(error.value) == "time_step must be finite and nonnegative."


@pytest.mark.parametrize(
    "geometry, match",
    [
        ("spherical", "spherical"),
        ("rectangular", "rectangular"),
    ],
)
def test_geometry_payload_exclusivity_is_validated(geometry, match) -> None:
    """Reject geometry payloads that belong to the other chamber shape."""
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    invalid = (
        NeutralWallLossConfig("spherical", 0.01, chamber_dimensions=(1.0,) * 3)
        if geometry == "spherical"
        else NeutralWallLossConfig("rectangular", 0.01, chamber_radius=1.0)
    )
    with pytest.raises(ValueError, match=match):
        wall_loss_step_gpu(_particles(), 298.15, 101325.0, 1.0, config=invalid)


@pytest.mark.parametrize("time_step", [True, "1", np.nan, -1.0])
def test_invalid_time_step_preserves_particles(time_step) -> None:
    """Time-step rejection follows particle scans and remains atomic."""
    from particula.gpu.kernels import wall_loss_step_gpu

    valid_particles = _particles()
    particles = SimpleNamespace(
        masses=valid_particles.masses,
        concentration=valid_particles.concentration,
        charge=valid_particles.charge,
        density=valid_particles.density,
        volume=valid_particles.volume,
    )
    snapshot = _snapshot(particles)
    with pytest.raises((TypeError, ValueError), match="time_step"):
        wall_loss_step_gpu(
            particles, 298.15, 101325.0, time_step, config=_config()
        )
    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize("field", ["masses", "concentration", "charge"])
def test_nonnegative_particle_metadata_rejects_nan(field: str) -> None:
    """Particle metadata domains reject NaN without mutating any field."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    value = getattr(particles, field).numpy()
    value.flat[0] = np.nan
    setattr(particles, field, wp.array(value, dtype=wp.float64, device="cpu"))
    snapshot = _snapshot(particles)
    with pytest.raises(ValueError, match=f"particles.{field} must be finite"):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=_config())
    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize("field", ["masses", "concentration"])
def test_nonnegative_particle_storage_rejects_negative_values(
    field: str,
) -> None:
    """Mass and concentration storage reject negative values atomically."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    values = getattr(particles, field).numpy()
    values.flat[0] = -1.0
    setattr(particles, field, wp.array(values, dtype=wp.float64, device="cpu"))
    snapshot = _snapshot(particles)

    with pytest.raises(
        ValueError,
        match=rf"particles\.{field} must be finite and nonnegative",
    ):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=_config())

    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize("charges", [[-2.0, -1.0], [-2.0, 3.0]])
def test_signed_finite_charge_metadata_is_accepted_with_complete_slot_updates(
    charges: list[float],
) -> None:
    """Neutral P5 retains signed charge or clears it with a removed slot."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    particles.charge = wp.array(
        [charges, [1.0, -4.0]], dtype=wp.float64, device="cpu"
    )
    rng_states = wp.array([3, 4], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)

    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0,
        config=_config(),
        rng_states=rng_states,
    )

    assert returned is particles
    _assert_slots_preserved_or_cleared(particles, snapshot)
    assert rng_states is snapshot["rng_states"][0]
    assert not np.array_equal(rng_states.numpy(), snapshot["rng_states"][1])


def test_rng_metadata_rejection_preserves_sidecar() -> None:
    """P3 validates but never consumes a supplied RNG sidecar."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    rng_states = wp.array([3, 4], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)
    with pytest.raises(ValueError, match="rng_seed"):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=_config(),
            rng_seed=2**32,
            rng_states=rng_states,
        )
    _assert_snapshot_unchanged(particles, snapshot, rng_states)


@pytest.mark.parametrize(
    ("scenario", "error", "match"),
    [
        ("wrong_type", TypeError, "NeutralWallLossConfig"),
        ("invalid_geometry", ValueError, "spherical"),
        ("missing_dimensions", ValueError, "rectangular"),
    ],
)
def test_config_validation_rejects_noncanonical_geometry_payloads(
    scenario, error, match
) -> None:
    """Validate canonical config type and geometry before particle work."""
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    configs = {
        "wrong_type": object(),
        "invalid_geometry": NeutralWallLossConfig(
            "triangle", 0.01, chamber_radius=1.0
        ),
        "missing_dimensions": NeutralWallLossConfig("rectangular", 0.01),
    }
    with pytest.raises(error, match=match):
        wall_loss_step_gpu(
            object(), 298.15, 101325.0, 1.0, config=configs[scenario]
        )


@pytest.mark.parametrize(
    "scenario, match",
    [
        ("zero_eddy", "wall_eddy_diffusivity"),
        ("string_radius", "chamber_radius"),
        ("list_dimensions", "chamber_dimensions"),
        ("infinite_dimension", "chamber_dimensions[1]"),
    ],
)
def test_config_physical_payload_validation_fails_before_particles(
    scenario, match
) -> None:
    """Reject invalid scalar and dimension payloads without particle access."""
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    configs = {
        "zero_eddy": NeutralWallLossConfig(
            "spherical", 0.0, chamber_radius=1.0
        ),
        "string_radius": NeutralWallLossConfig(
            "spherical", 0.01, chamber_radius=cast(Any, "1")
        ),
        "list_dimensions": NeutralWallLossConfig(
            "rectangular",
            0.01,
            chamber_dimensions=cast(Any, [1.0, 2.0, 3.0]),
        ),
        "infinite_dimension": NeutralWallLossConfig(
            "rectangular", 0.01, chamber_dimensions=(1.0, np.inf, 3.0)
        ),
    }

    with pytest.raises((TypeError, ValueError), match=re.escape(match)):
        wall_loss_step_gpu(
            object(), 298.15, 101325.0, 1.0, config=configs[scenario]
        )


@pytest.mark.parametrize("field", ["density", "volume"])
@pytest.mark.parametrize("invalid", [-1.0, np.inf])
def test_positive_particle_metadata_rejects_invalid_values(
    field, invalid
) -> None:
    """Density and volume require finite strictly positive device values."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    values = getattr(particles, field).numpy()
    values[0] = invalid
    setattr(particles, field, wp.array(values, dtype=wp.float64, device="cpu"))
    snapshot = _snapshot(particles)
    with pytest.raises(ValueError, match=f"particles.{field} must be finite"):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=_config())
    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("masses", np.ones((2, 2), dtype=np.float64), "rank 3"),
        ("concentration", np.ones((2, 3), dtype=np.float64), "shape"),
        ("charge", np.ones((2, 2), dtype=np.float32), "use dtype float64"),
        ("density", np.ones((3,), dtype=np.float64), "shape"),
        ("volume", np.ones((2, 1), dtype=np.float64), "rank 1"),
    ],
)
def test_particle_schema_rejection_is_atomic(field, values, match) -> None:
    """Each particle field enforces its fixed rank, shape, and dtype schema."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    valid_particles = _particles()
    particles = SimpleNamespace(
        masses=valid_particles.masses,
        concentration=valid_particles.concentration,
        charge=valid_particles.charge,
        density=valid_particles.density,
        volume=valid_particles.volume,
    )
    dtype = wp.float32 if values.dtype == np.float32 else wp.float64
    setattr(particles, field, wp.array(values, dtype=dtype, device="cpu"))
    snapshot = _snapshot(particles)
    with pytest.raises(ValueError, match=f"particles.{field}.*{match}"):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=_config())
    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize(
    ("temperature", "pressure", "environment"),
    [
        (None, 101325.0, None),
        (298.15, None, None),
        (298.15, 101325.0, SimpleNamespace()),
    ],
)
def test_environment_preflight_errors_preserve_particles(
    temperature, pressure, environment
) -> None:
    """Delegated environment-source errors occur without a particle write."""
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    snapshot = _snapshot(particles)
    with pytest.raises(ValueError, match="wall_loss_step_gpu"):
        wall_loss_step_gpu(
            particles,
            temperature,
            pressure,
            1.0,
            config=_config(),
            environment=environment,
        )
    _assert_snapshot_unchanged(particles, snapshot)


def test_late_environment_preflight_error_preserves_rng_sidecar() -> None:
    """Environment rejection leaves a validated caller-owned sidecar intact."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    rng_states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)

    with pytest.raises(ValueError, match="wall_loss_step_gpu"):
        wall_loss_step_gpu(
            particles,
            None,
            101325.0,
            1.0,
            config=_config(),
            rng_states=rng_states,
        )

    _assert_snapshot_unchanged(particles, snapshot, rng_states)


@pytest.mark.parametrize("source", ["direct", "environment"])
def test_positive_time_float32_environment_arrays_are_private_normalized(
    source: str,
) -> None:
    """Execute accepted float32 environments through private float64 buffers."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    rng_states = wp.array([3, 4], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)
    temperature = wp.array([298.15, 300.0], dtype=wp.float32, device="cpu")
    pressure = wp.array([101325.0, 100000.0], dtype=wp.float32, device="cpu")
    environment = (
        SimpleNamespace(temperature=temperature, pressure=pressure)
        if source == "environment"
        else None
    )
    returned = wall_loss_step_gpu(
        particles,
        None if environment else temperature,
        None if environment else pressure,
        1.0,
        config=_config(),
        rng_states=rng_states,
        environment=environment,
    )

    assert returned is particles
    assert temperature.dtype == wp.float32
    assert pressure.dtype == wp.float32
    npt.assert_allclose(temperature.numpy(), [298.15, 300.0], rtol=1e-6)
    npt.assert_allclose(pressure.numpy(), [101325.0, 100000.0], rtol=1e-6)
    _assert_slots_preserved_or_cleared(particles, snapshot)
    assert rng_states is snapshot["rng_states"][0]
    assert not np.array_equal(rng_states.numpy(), snapshot["rng_states"][1])


def test_explicit_float64_environment_is_accepted() -> None:
    """Accept an explicit float64 environment input source."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    environment = SimpleNamespace(
        temperature=wp.array([298.15, 300.0], dtype=wp.float64, device="cpu"),
        pressure=wp.array([101325.0, 100000.0], dtype=wp.float64, device="cpu"),
    )
    assert (
        wall_loss_step_gpu(
            particles,
            None,
            None,
            0.0,
            config=_config(),
            environment=environment,
        )
        is particles
    )


@pytest.mark.parametrize("source", ["direct", "environment"])
def test_zero_time_is_byte_for_byte_write_free_after_preflight(
    source: str,
) -> None:
    """Zero time validates every source form without normalizing or writing."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    rng_states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, rng_states)
    if source == "direct":
        returned = wall_loss_step_gpu(
            particles,
            wp.array([298.15, 300.0], dtype=wp.float64, device="cpu"),
            wp.array([101325.0, 100000.0], dtype=wp.float64, device="cpu"),
            0.0,
            config=_config(),
            rng_states=rng_states,
            initialize_rng=True,
        )
    else:
        environment = SimpleNamespace(
            temperature=wp.array(
                [298.15, 300.0], dtype=wp.float64, device="cpu"
            ),
            pressure=wp.array(
                [101325.0, 100000.0], dtype=wp.float64, device="cpu"
            ),
        )
        returned = wall_loss_step_gpu(
            particles,
            None,
            None,
            np.array(0.0),
            config=_config(),
            rng_states=rng_states,
            initialize_rng=True,
            environment=environment,
        )
    assert returned is particles
    _assert_snapshot_unchanged(particles, snapshot, rng_states)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rng_seed": True}, TypeError, "rng_seed"),
        ({"rng_seed": -1}, ValueError, "rng_seed"),
        ({"initialize_rng": 1}, TypeError, "initialize_rng"),
        ({"rng_states": object()}, ValueError, "rng_states"),
    ],
)
def test_rng_form_rejections_preserve_particles(kwargs, error, match) -> None:
    """Deferred RNG inputs validate metadata only and never mutate particles."""
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    snapshot = _snapshot(particles)
    with pytest.raises(error, match=match):
        wall_loss_step_gpu(
            particles, 298.15, 101325.0, 1.0, config=_config(), **kwargs
        )
    _assert_snapshot_unchanged(particles, snapshot)


def test_zero_time_never_calls_wall_loss_coefficient_helpers(
    monkeypatch,
) -> None:
    """The post-preflight zero-time path does not calculate coefficients."""
    from particula.gpu.kernels import wall_loss as wall_loss_module
    from particula.gpu.kernels import wall_loss_step_gpu

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("P3 preflight must not calculate wall-loss coefficients")

    monkeypatch.setattr(
        wall_loss_module, "spherical_wall_loss_coefficient_wp", fail_if_called
    )
    monkeypatch.setattr(
        wall_loss_module, "rectangle_wall_loss_coefficient_wp", fail_if_called
    )
    assert (
        wall_loss_step_gpu(
            _particles(), 298.15, 101325.0, 0.0, config=_config()
        )
        is not None
    )


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_large_time_step_removes_only_eligible_slots(geometry: str) -> None:
    """Finite positive coefficients clear eligible slots at underflow time."""
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    particles.masses = _warp().array(
        np.array(
            [
                [[1.0, 2.0], [0.0, 0.0]],
                [[3.0, 4.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        dtype=_warp().float64,
        device="cpu",
    )
    snapshot = _snapshot(particles)

    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0e300,
        config=_config(geometry),
        rng_seed=42,
    )

    assert returned is particles
    for box, particle in ((0, 0), (1, 0)):
        npt.assert_array_equal(particles.masses.numpy()[box, particle], 0.0)
        assert particles.concentration.numpy()[box, particle] == 0.0
        assert particles.charge.numpy()[box, particle] == 0.0
    # The inactive and zero-mass slots are never sampled or modified.
    npt.assert_array_equal(particles.masses.numpy()[0, 1], [0.0, 0.0])
    assert particles.charge.numpy()[0, 1] == snapshot["charge"][1][0, 1]
    npt.assert_array_equal(particles.masses.numpy()[1, 1], [0.0, 0.0])
    assert particles.concentration.numpy()[1, 1] == 3.0
    assert particles.charge.numpy()[1, 1] == snapshot["charge"][1][1, 1]
    npt.assert_array_equal(particles.density.numpy(), snapshot["density"][1])
    npt.assert_array_equal(particles.volume.numpy(), snapshot["volume"][1])


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_positive_infinite_coefficient_removes_without_advancing_rng(
    geometry: str,
) -> None:
    """An overflowing valid coefficient selects removal without an RNG draw."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    tiny_dimension = np.nextafter(0.0, 1.0)
    config = (
        NeutralWallLossConfig("spherical", 0.01, chamber_radius=tiny_dimension)
        if geometry == "spherical"
        else NeutralWallLossConfig(
            "rectangular",
            0.01,
            chamber_dimensions=(tiny_dimension, 1.0, 1.0),
        )
    )
    particles = _particles()
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)

    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0,
        config=config,
        rng_states=states,
    )

    assert returned is particles
    for box, particle in np.argwhere(snapshot["concentration"][1] > 0.0):
        npt.assert_array_equal(particles.masses.numpy()[box, particle], 0.0)
        assert particles.concentration.numpy()[box, particle] == 0.0
        assert particles.charge.numpy()[box, particle] == 0.0
    npt.assert_array_equal(states.numpy(), snapshot["rng_states"][1])
    npt.assert_array_equal(particles.density.numpy(), snapshot["density"][1])
    npt.assert_array_equal(particles.volume.numpy(), snapshot["volume"][1])


def test_zero_survival_removes_an_exact_zero_draw() -> None:
    """An exact-zero draw selects removal when survival probability is zero."""
    wp = _warp()
    from particula.gpu.kernels.wall_loss import _should_remove_for_survival_draw

    @wp.kernel
    def select_removals(
        draws: Any,
        survival_probabilities: Any,
        removal_mask: Any,
    ) -> None:
        index = wp.tid()
        if _should_remove_for_survival_draw(
            draws[index], survival_probabilities[index]
        ):
            removal_mask[index] = wp.int32(1)

    draws = wp.array([0.0, 0.25], dtype=wp.float32, device="cpu")
    survival_probabilities = wp.array(
        [0.0, 0.5], dtype=wp.float64, device="cpu"
    )
    removal_mask = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(
        select_removals,
        dim=2,
        inputs=[draws, survival_probabilities, removal_mask],
        device="cpu",
    )

    npt.assert_array_equal(removal_mask.numpy(), [1, 0])


def test_mask_application_preserves_unmarked_and_clears_marked_slots() -> None:
    """Private mask application clears every mutable lane only when selected."""
    wp = _warp()
    from particula.gpu.kernels.wall_loss import _apply_wall_loss_mask

    particles = _particles()
    snapshot = _snapshot(particles)
    mask = wp.array([[0, 1], [1, 0]], dtype=wp.int32, device="cpu")
    wp.launch(
        _apply_wall_loss_mask,
        dim=mask.shape,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            mask,
            2,
        ],
        device="cpu",
    )

    for box, particle in ((0, 1), (1, 0)):
        npt.assert_array_equal(particles.masses.numpy()[box, particle], 0.0)
        assert particles.concentration.numpy()[box, particle] == 0.0
        assert particles.charge.numpy()[box, particle] == 0.0
    for box, particle in ((0, 0), (1, 1)):
        npt.assert_array_equal(
            particles.masses.numpy()[box, particle],
            snapshot["masses"][1][box, particle],
        )
        assert (
            particles.concentration.numpy()[box, particle]
            == snapshot["concentration"][1][box, particle]
        )
        assert (
            particles.charge.numpy()[box, particle]
            == snapshot["charge"][1][box, particle]
        )
    npt.assert_array_equal(particles.density.numpy(), snapshot["density"][1])
    npt.assert_array_equal(particles.volume.numpy(), snapshot["volume"][1])


def test_equal_inputs_and_explicit_reset_produce_equal_slot_updates() -> None:
    """Same-device explicit resets reproducibly initialize supplied sidecars."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    first = _particles()
    second = _particles()
    first_rng = wp.array([13, 17], dtype=wp.uint32, device="cpu")
    second_rng = wp.array([23, 29], dtype=wp.uint32, device="cpu")
    first_rng_before = first_rng.numpy().copy()
    second_rng_before = second_rng.numpy().copy()

    wall_loss_step_gpu(
        first,
        298.15,
        101325.0,
        1.0,
        config=_config(),
        rng_seed=97,
        rng_states=first_rng,
        initialize_rng=True,
    )
    wall_loss_step_gpu(
        second,
        298.15,
        101325.0,
        1.0,
        config=_config(),
        rng_seed=97,
        rng_states=second_rng,
        initialize_rng=True,
    )

    for field in ("masses", "concentration", "charge", "density", "volume"):
        npt.assert_array_equal(
            getattr(first, field).numpy(), getattr(second, field).numpy()
        )
    assert not np.array_equal(first_rng.numpy(), first_rng_before)
    assert not np.array_equal(second_rng.numpy(), second_rng_before)
    npt.assert_array_equal(first_rng.numpy(), second_rng.numpy())


def test_subnormal_mass_slot_survives_without_transport_calculation() -> None:
    """Keep an active slot unchanged when its derived volume underflows."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    particles.masses = wp.array(
        np.array(
            [
                [[np.nextafter(0.0, 1.0), 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        dtype=wp.float64,
        device="cpu",
    )
    particles.concentration = wp.array(
        [[1.0, 0.0], [0.0, 0.0]], dtype=wp.float64, device="cpu"
    )
    snapshot = _snapshot(particles)

    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0e300,
        config=_config(),
        rng_seed=2**32 - 1,
    )

    assert returned is particles
    _assert_snapshot_unchanged(particles, snapshot)


@pytest.mark.parametrize("mask_values", [np.zeros((2, 2)), np.ones((2, 2))])
def test_mask_application_handles_all_survivor_and_all_removal_masks(
    mask_values: np.ndarray,
) -> None:
    """All-zero and all-one private masks preserve or clear complete slots."""
    wp = _warp()
    from particula.gpu.kernels.wall_loss import _apply_wall_loss_mask

    particles = _particles()
    snapshot = _snapshot(particles)
    mask = wp.array(mask_values, dtype=wp.int32, device="cpu")
    wp.launch(
        _apply_wall_loss_mask,
        dim=mask.shape,
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            mask,
            2,
        ],
        device="cpu",
    )

    if np.all(mask_values == 0.0):
        _assert_snapshot_unchanged(particles, snapshot)
        return
    npt.assert_array_equal(particles.masses.numpy(), 0.0)
    npt.assert_array_equal(particles.concentration.numpy(), 0.0)
    npt.assert_array_equal(particles.charge.numpy(), 0.0)
    npt.assert_array_equal(particles.density.numpy(), snapshot["density"][1])
    npt.assert_array_equal(particles.volume.numpy(), snapshot["volume"][1])


def test_missing_particle_field_and_nonarray_field_have_stable_errors() -> None:
    """Required particle fields reject missing and host-backed schema values."""
    from particula.gpu.kernels import wall_loss_step_gpu

    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        wall_loss_step_gpu(
            SimpleNamespace(), 298.15, 101325.0, 1.0, config=_config()
        )

    valid_particles = _particles()
    particles = SimpleNamespace(
        masses=valid_particles.masses,
        concentration=valid_particles.concentration,
        charge=np.zeros((2, 2), dtype=np.float64),
        density=valid_particles.density,
        volume=valid_particles.volume,
    )
    with pytest.raises(
        ValueError, match="particles.charge must be a Warp array"
    ):
        wall_loss_step_gpu(particles, 298.15, 101325.0, 1.0, config=_config())


@pytest.mark.parametrize(
    ("rng_states", "match"),
    [
        (lambda wp: wp.zeros(2, dtype=wp.int32, device="cpu"), "dtype"),
        (lambda wp: wp.zeros((2, 1), dtype=wp.uint32, device="cpu"), "shape"),
        (lambda wp: wp.zeros(3, dtype=wp.uint32, device="cpu"), "shape"),
    ],
)
def test_rng_sidecar_schema_rejections_are_atomic(rng_states, match) -> None:
    """Supplied RNG state must be one uint32 entry per particle box."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    sidecar = rng_states(wp)
    snapshot = _snapshot(particles, sidecar)
    with pytest.raises(ValueError, match=f"rng_states.*{match}"):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=_config(),
            rng_states=sidecar,
        )
    _assert_snapshot_unchanged(particles, snapshot, sidecar)


@pytest.mark.cuda
def test_rng_sidecar_device_rejection_is_atomic_when_cuda_available() -> None:
    """Reject CUDA state for CPU particles without mutating either owner."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles(device="cpu")
    sidecar = wp.array([7, 11], dtype=wp.uint32, device="cuda:0")
    snapshot = _snapshot(particles, sidecar)
    with pytest.raises(ValueError, match="rng_states device"):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=_config(),
            rng_states=sidecar,
        )
    _assert_snapshot_unchanged(particles, snapshot, sidecar)


@pytest.mark.stochastic
def test_omitted_rng_state_is_private_seeded_convenience() -> None:
    """Fresh omitted state produces repeatable same-device slot outcomes."""
    from particula.gpu.kernels import wall_loss_step_gpu

    first = _particles()
    second = _particles()
    for particles in (first, second):
        wall_loss_step_gpu(
            particles, 298.15, 101325.0, 1.0, config=_config(), rng_seed=91
        )
    for field in ("masses", "concentration", "charge"):
        npt.assert_array_equal(
            getattr(first, field).numpy(), getattr(second, field).numpy()
        )


@pytest.mark.stochastic
def test_supplied_rng_state_advances_without_implicit_reseed() -> None:
    """A caller-owned state advances across eligible calls with one seed."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    states = wp.zeros(2, dtype=wp.uint32, device="cpu")
    tiny_time = np.nextafter(0.0, 1.0)
    wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        tiny_time,
        config=_config(),
        rng_seed=41,
        rng_states=states,
        initialize_rng=True,
    )
    first_advance = states.numpy().copy()
    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        tiny_time,
        config=_config(),
        rng_seed=41,
        rng_states=states,
    )
    assert returned is particles
    assert not np.array_equal(states.numpy(), first_advance)


@pytest.mark.stochastic
def test_rng_states_advance_per_box_for_eligible_slots() -> None:
    """Each box owns and advances its own RNG word without a shared race."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import _initialize_rng_states

    particles = _particles()
    states = wp.zeros(2, dtype=wp.uint32, device="cpu")
    initialized = wp.zeros(2, dtype=wp.uint32, device="cpu")
    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[5, initialized],
        device="cpu",
    )
    expected_initialized = initialized.numpy().copy()
    wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        np.nextafter(0.0, 1.0),
        config=_config(),
        rng_seed=5,
        rng_states=states,
        initialize_rng=True,
    )
    initialized_and_advanced = states.numpy()
    assert np.all(initialized_and_advanced != expected_initialized)


@pytest.mark.stochastic
def test_all_ineligible_positive_time_preserves_rng_state() -> None:
    """Positive calls consume no state when every fixed slot is ineligible."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    particles.concentration = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    returned = wall_loss_step_gpu(
        particles, 298.15, 101325.0, 1.0, config=_config(), rng_states=states
    )
    assert returned is particles
    _assert_snapshot_unchanged(particles, snapshot, states)


@pytest.mark.stochastic
def test_all_ineligible_positive_time_does_not_reset_rng_state() -> None:
    """An explicit reset request is ignored when no slot can consume a draw."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    particles.concentration = wp.zeros((2, 2), dtype=wp.float64, device="cpu")
    states = wp.array([7, 11], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)

    returned = wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0,
        config=_config(),
        rng_seed=41,
        rng_states=states,
        initialize_rng=True,
    )

    assert returned is particles
    _assert_snapshot_unchanged(particles, snapshot, states)


@pytest.mark.stochastic
def test_rng_state_consumes_only_eligible_slots() -> None:
    """Inactive, zero-mass, and unusable slots do not advance box state."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    candidate = SimpleNamespace(
        masses=wp.array(
            [[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [5e-324, 0.0]]],
            dtype=wp.float64,
            device="cpu",
        ),
        concentration=wp.array(
            [[1.0, 0.0, 1.0, 1.0]], dtype=wp.float64, device="cpu"
        ),
        charge=wp.array(
            [[0.0, 3.0, -2.0, 4.0]], dtype=wp.float64, device="cpu"
        ),
        density=wp.array([1000.0, 1200.0], dtype=wp.float64, device="cpu"),
        volume=wp.array([1.0], dtype=wp.float64, device="cpu"),
    )
    control = SimpleNamespace(
        masses=wp.array([[[1.0, 2.0]]], dtype=wp.float64, device="cpu"),
        concentration=wp.array([[1.0]], dtype=wp.float64, device="cpu"),
        charge=wp.array([[0.0]], dtype=wp.float64, device="cpu"),
        density=wp.array([1000.0, 1200.0], dtype=wp.float64, device="cpu"),
        volume=wp.array([1.0], dtype=wp.float64, device="cpu"),
    )
    candidate_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    control_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
    candidate_snapshot = _snapshot(candidate, candidate_states)
    tiny_time = np.nextafter(0.0, 1.0)
    for particles, states in (
        (candidate, candidate_states),
        (control, control_states),
    ):
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            tiny_time,
            config=_config(),
            rng_seed=31,
            rng_states=states,
            initialize_rng=True,
        )
    npt.assert_array_equal(candidate_states.numpy(), control_states.numpy())
    npt.assert_array_equal(
        candidate.masses.numpy()[0, 1:], candidate_snapshot["masses"][1][0, 1:]
    )
    npt.assert_array_equal(
        candidate.concentration.numpy()[0, 1:],
        candidate_snapshot["concentration"][1][0, 1:],
    )
    npt.assert_array_equal(
        candidate.charge.numpy()[0, 1:], candidate_snapshot["charge"][1][0, 1:]
    )


def _nanoparticle(charge: float = 1.0) -> SimpleNamespace:
    """Create one active 1 nm particle with explicit float64 storage."""
    wp = _warp()
    radius = 1.0e-9
    mass = 4.0 * np.pi * radius**3 * 1000.0 / 3.0
    return SimpleNamespace(
        masses=wp.array([[[mass]]], dtype=wp.float64, device="cpu"),
        concentration=wp.array([[1.0]], dtype=wp.float64, device="cpu"),
        charge=wp.array([[charge]], dtype=wp.float64, device="cpu"),
        density=wp.array([1000.0], dtype=wp.float64, device="cpu"),
        volume=wp.array([1.0], dtype=wp.float64, device="cpu"),
    )


def _charged_dispatch_config(
    geometry: str,
    *,
    potential: float = 0.0,
    field: float = 0.0,
) -> Any:
    """Create one charged public-step configuration for dispatch tests."""
    wp = _warp()
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    if geometry == "spherical":
        return NeutralWallLossConfig(
            "spherical",
            0.01,
            chamber_radius=1.0,
            mode="charged",
            wall_potential=potential,
            wall_electric_field=field,
        )
    return NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_potential=potential,
        wall_electric_field=wp.array(
            [field, 0.0, 0.0], dtype=wp.float64, device="cpu"
        ),
    )


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_charged_image_only_dispatch_removes_when_neutral_control_survives(
    geometry: str,
) -> None:
    """Nonzero charge wires image enhancement through both public geometries."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    neutral = _nanoparticle(charge=0.0)
    charged = _nanoparticle(charge=3.0)
    neutral_states = wp.array([17], dtype=wp.uint32, device="cpu")
    charged_states = wp.array([17], dtype=wp.uint32, device="cpu")
    neutral_before = _snapshot(neutral, neutral_states)
    charged_before = _snapshot(charged, charged_states)

    wall_loss_step_gpu(
        neutral,
        298.15,
        101325.0,
        10.0,
        config=_config(geometry),
        rng_states=neutral_states,
    )
    wall_loss_step_gpu(
        charged,
        298.15,
        101325.0,
        10.0,
        config=_charged_dispatch_config(geometry),
        rng_states=charged_states,
    )

    npt.assert_array_equal(neutral.masses.numpy(), neutral_before["masses"][1])
    npt.assert_array_equal(
        neutral.concentration.numpy(), neutral_before["concentration"][1]
    )
    npt.assert_array_equal(charged.masses.numpy(), 0.0)
    npt.assert_array_equal(charged.concentration.numpy(), 0.0)
    npt.assert_array_equal(charged.charge.numpy(), 0.0)
    assert not np.array_equal(
        neutral_states.numpy(), neutral_before["rng_states"][1]
    )
    assert not np.array_equal(
        charged_states.numpy(), charged_before["rng_states"][1]
    )


@pytest.mark.parametrize(
    ("charge", "potential", "removes"),
    [
        (1.0, 1.0e100, True),
        (1.0, -1.0e100, False),
        (-1.0, -1.0e100, True),
        (-1.0, 1.0e100, False),
    ],
)
def test_charged_spherical_potential_and_charge_sign_control_draws(
    charge: float,
    potential: float,
    removes: bool,
) -> None:
    """Spherical signed potential drift removes only aligned charge lanes."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _nanoparticle(charge=charge)
    states = wp.array([23], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        1.0,
        config=_charged_dispatch_config("spherical", potential=potential),
        rng_states=states,
    )

    if removes:
        npt.assert_array_equal(particles.masses.numpy(), 0.0)
        npt.assert_array_equal(particles.concentration.numpy(), 0.0)
        npt.assert_array_equal(particles.charge.numpy(), 0.0)
        assert not np.array_equal(states.numpy(), snapshot["rng_states"][1])
    else:
        _assert_snapshot_unchanged(particles, snapshot, states)


@pytest.mark.parametrize("lane", [0, 1, 2], ids=["x", "y", "z"])
def test_charged_rectangular_field_vector_lanes_drive_public_step(
    lane: int,
) -> None:
    """Each rectangular field lane produces aligned drift through dispatch."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    field_values = np.zeros(3, dtype=np.float64)
    field_values[lane] = 1.0e6
    field = wp.array(field_values, dtype=wp.float64, device="cpu")
    config = NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_electric_field=field,
    )
    particles = _nanoparticle(charge=1.0)
    states = wp.array([29], dtype=wp.uint32, device="cpu")
    field_before = field.numpy().copy()

    wall_loss_step_gpu(
        particles, 298.15, 101325.0, 1.0, config=config, rng_states=states
    )

    npt.assert_array_equal(particles.masses.numpy(), 0.0)
    npt.assert_array_equal(particles.concentration.numpy(), 0.0)
    npt.assert_array_equal(particles.charge.numpy(), 0.0)
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


def test_charged_rectangular_zero_vector_uses_image_only_draw_path() -> None:
    """A zero rectangular vector leaves image-only survival stochastic."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _nanoparticle(charge=1.0)
    states = wp.array([29], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    config = _charged_dispatch_config("rectangular")
    field = config.wall_electric_field
    field_before = field.numpy().copy()

    wall_loss_step_gpu(
        particles,
        298.15,
        101325.0,
        np.nextafter(0.0, 1.0),
        config=config,
        rng_states=states,
    )

    npt.assert_array_equal(particles.masses.numpy(), snapshot["masses"][1])
    npt.assert_array_equal(
        particles.concentration.numpy(), snapshot["concentration"][1]
    )
    npt.assert_array_equal(particles.charge.numpy(), snapshot["charge"][1])
    assert not np.array_equal(states.numpy(), snapshot["rng_states"][1])
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), field_before)


def test_charged_rectangular_signed_potential_clips_opposing_vector_drift() -> (
    None
):
    """A signed rectangular potential can oppose the vector field without a draw."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    field = wp.array([1.0e100, 0.0, 0.0], dtype=wp.float64, device="cpu")
    config = NeutralWallLossConfig(
        "rectangular",
        0.01,
        chamber_dimensions=(1.0, 2.0, 3.0),
        mode="charged",
        wall_potential=-2.0e100,
        wall_electric_field=field,
    )
    particles = _nanoparticle(charge=1.0)
    states = wp.array([31], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)

    wall_loss_step_gpu(
        particles, 298.15, 101325.0, 1.0, config=config, rng_states=states
    )

    _assert_snapshot_unchanged(particles, snapshot, states)
    assert config.wall_electric_field is field
    npt.assert_array_equal(field.numpy(), [1.0e100, 0.0, 0.0])


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_charged_saturated_coefficient_consumes_rng_before_removal(
    geometry: str,
) -> None:
    """Charged saturation uses a survival draw rather than neutral infinity."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    tiny_dimension = np.nextafter(0.0, 1.0)
    field = (
        0.0
        if geometry == "spherical"
        else wp.zeros(3, dtype=wp.float64, device="cpu")
    )
    config = (
        NeutralWallLossConfig(
            "spherical",
            0.01,
            chamber_radius=tiny_dimension,
            mode="charged",
            wall_electric_field=field,
        )
        if geometry == "spherical"
        else NeutralWallLossConfig(
            "rectangular",
            0.01,
            chamber_dimensions=(tiny_dimension, 1.0, 1.0),
            mode="charged",
            wall_electric_field=field,
        )
    )
    particles = _nanoparticle(charge=1.0)
    states = wp.array([31], dtype=wp.uint32, device="cpu")
    before = _snapshot(particles, states)

    wall_loss_step_gpu(
        particles, 298.15, 101325.0, 1.0, config=config, rng_states=states
    )

    npt.assert_array_equal(particles.masses.numpy(), 0.0)
    npt.assert_array_equal(particles.concentration.numpy(), 0.0)
    npt.assert_array_equal(particles.charge.numpy(), 0.0)
    assert not np.array_equal(states.numpy(), before["rng_states"][1])


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_charged_all_inactive_positive_time_preserves_requested_rng_reset(
    geometry: str,
) -> None:
    """Charged all-inactive calls leave supplied sidecars untouched."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _nanoparticle(charge=-2.0)
    particles.concentration = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
    states = wp.array([37], dtype=wp.uint32, device="cpu")
    snapshot = _snapshot(particles, states)
    config = _charged_dispatch_config(geometry)
    field = config.wall_electric_field
    field_before = field.numpy().copy() if hasattr(field, "numpy") else field

    assert (
        wall_loss_step_gpu(
            particles,
            298.15,
            101325.0,
            1.0,
            config=config,
            rng_seed=99,
            rng_states=states,
            initialize_rng=True,
        )
        is particles
    )

    _assert_snapshot_unchanged(particles, snapshot, states)
    if hasattr(field, "numpy"):
        assert config.wall_electric_field is field
        npt.assert_array_equal(field.numpy(), field_before)


@pytest.mark.benchmark
def test_wall_loss_p5_benchmark_smoke() -> None:
    """Record an opt-in P5 execution duration without a throughput target."""
    from particula.gpu.kernels import wall_loss_step_gpu

    started = time.perf_counter()
    wall_loss_step_gpu(_particles(), 298.15, 101325.0, 1.0, config=_config())
    assert time.perf_counter() >= started
