"""Tests for the write-free GPU neutral wall-loss P3 boundary."""

from __future__ import annotations

import re
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


@pytest.mark.parametrize("geometry", ["spherical", "rectangular"])
def test_valid_preflight_returns_identity_without_mutation(
    geometry: str,
) -> None:
    """Both supported geometries validate without writing particle state."""
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
    _assert_snapshot_unchanged(particles, snapshot, rng_states)


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


def test_explicit_environment_and_float32_direct_array_are_accepted() -> None:
    """Accept explicit and hybrid scalar/per-box environment input forms."""
    wp = _warp()
    from particula.gpu.kernels import wall_loss_step_gpu

    particles = _particles()
    temperature = wp.array([298.15, 300.0], dtype=wp.float32, device="cpu")
    assert (
        wall_loss_step_gpu(
            particles, temperature, 101325.0, np.array(0.0), config=_config()
        )
        is particles
    )

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


def test_preflight_never_calls_wall_loss_coefficient_helpers(
    monkeypatch,
) -> None:
    """P3 does not reach either deferred neutral coefficient helper."""
    from particula.gpu.dynamics import wall_loss_funcs
    from particula.gpu.kernels import wall_loss_step_gpu

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("P3 preflight must not calculate wall-loss coefficients")

    monkeypatch.setattr(
        wall_loss_funcs, "spherical_wall_loss_coefficient_wp", fail_if_called
    )
    monkeypatch.setattr(
        wall_loss_funcs, "rectangle_wall_loss_coefficient_wp", fail_if_called
    )
    assert (
        wall_loss_step_gpu(
            _particles(), 298.15, 101325.0, 1.0, config=_config()
        )
        is not None
    )


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
