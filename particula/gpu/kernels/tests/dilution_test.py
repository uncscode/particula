"""Tests for the validation-only P1 GPU dilution contract."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp


def _warp():
    """Import Warp at test runtime to preserve marker deselection."""
    return pytest.importorskip("warp")


def _containers(n_boxes: int = 2, device: str = "cpu"):
    """Build minimal fixed-schema particle and gas containers on one device."""
    wp = _warp()
    from particula.gpu import WarpGasData, WarpParticleData

    n_particles, n_species = 2, 1
    particles = WarpParticleData()
    particles.masses = wp.ones(
        (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
    )
    particles.concentration = wp.ones(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    particles.charge = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    particles.density = wp.ones(n_species, dtype=wp.float64, device=device)
    particles.volume = wp.ones(n_boxes, dtype=wp.float64, device=device)

    gas = WarpGasData()
    gas.molar_mass = wp.ones(n_species, dtype=wp.float64, device=device)
    gas.concentration = wp.ones(
        (n_boxes, n_species), dtype=wp.float64, device=device
    )
    gas.vapor_pressure = wp.zeros(
        (n_boxes, n_species), dtype=wp.float64, device=device
    )
    gas.partitioning = wp.ones(
        (n_boxes, n_species), dtype=wp.int32, device=device
    )
    return particles, gas


def _concentration_snapshots(particles, gas) -> tuple[np.ndarray, np.ndarray]:
    """Copy caller-owned concentration fields for no-write assertions."""
    return (
        particles.concentration.numpy().copy(),
        gas.concentration.numpy().copy(),
    )


def _assert_concentrations_unchanged(particles, gas, snapshots) -> None:
    """Assert particle and gas concentrations match their exact snapshots."""
    particle_snapshot, gas_snapshot = snapshots
    npt.assert_array_equal(particles.concentration.numpy(), particle_snapshot)
    npt.assert_array_equal(gas.concentration.numpy(), gas_snapshot)


def test_concrete_module_signature_docstring_and_no_package_export() -> None:
    """Freeze the P1 concrete-only entry-point signature and documentation."""
    from particula.gpu import kernels
    from particula.gpu.kernels import dilution

    signature = inspect.signature(dilution.dilution_step_gpu)
    assert list(signature.parameters) == [
        "particles",
        "gas",
        "coefficient",
        "time_step",
    ]
    assert signature.return_annotation == "tuple[Any, Any]"
    assert "dilution_step_gpu" not in kernels.__all__
    missing_name = "dilution_step_gpu"
    with pytest.raises(AttributeError, match=missing_name):
        getattr(kernels, missing_name)
    assert "[s" in dilution.__doc__
    assert "exp(-alpha * time_step)" in dilution.__doc__


@pytest.mark.parametrize(
    "coefficient", [0, 0.5, np.float64(1.5), np.array(2.5)]
)
def test_normalize_scalar_coefficient_allocates_private_broadcast(
    coefficient,
) -> None:
    """Scalar coefficients become private requested-device float64 buffers."""
    wp = _warp()
    from particula.gpu.kernels.dilution import _normalize_coefficient

    normalized = _normalize_coefficient(coefficient, 3, wp.get_device("cpu"))
    assert normalized.dtype == wp.float64
    assert normalized.shape == (3,)
    assert str(normalized.device) == str(wp.get_device("cpu"))
    npt.assert_array_equal(normalized.numpy(), np.full(3, coefficient))


def test_normalize_per_box_coefficient_preserves_identity() -> None:
    """A valid caller-owned per-box coefficient is returned by identity."""
    wp = _warp()
    from particula.gpu.kernels.dilution import _normalize_coefficient

    coefficient = wp.array([0.0, 2.0], dtype=wp.float64, device="cpu")
    assert (
        _normalize_coefficient(coefficient, 2, coefficient.device)
        is coefficient
    )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_normalize_per_box_coefficient_defers_value_validation(value) -> None:
    """P1 retains metadata-valid per-box values for P3 physical validation."""
    wp = _warp()
    from particula.gpu.kernels.dilution import _normalize_coefficient

    coefficient = wp.array([value, 0.0], dtype=wp.float64, device="cpu")
    assert (
        _normalize_coefficient(coefficient, 2, coefficient.device)
        is coefficient
    )


@pytest.mark.parametrize(
    ("coefficient", "time_step"),
    [
        (1.0, 2.0),
        (0.0, 2.0),
        (1.0, 0.0),
        (1.0, np.float64(2.0)),
        (np.float64(1.0), np.array(2.0)),
    ],
)
def test_public_scalar_calls_are_identity_no_writes(
    coefficient, time_step
) -> None:
    """Valid scalar inputs return identical objects without concentration writes."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, time_step
    )
    assert returned_particles is particles
    assert returned_gas is gas
    _assert_concentrations_unchanged(particles, gas, snapshots)


@pytest.mark.parametrize("values", [[0.0, 0.0], [0.0, 1.0]])
def test_public_per_box_calls_are_identity_no_writes(values) -> None:
    """All-zero and mixed valid per-box inputs remain P1 identity calls."""
    wp = _warp()
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    coefficient = wp.array(values, dtype=wp.float64, device="cpu")
    assert dilution_step_gpu(particles, gas, coefficient, 1.0) == (
        particles,
        gas,
    )
    _assert_concentrations_unchanged(particles, gas, snapshots)


def test_documented_p2_equation_is_not_executed_in_p1() -> None:
    """Record the E6-F1 finite-step oracle while asserting P1 is no-write."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    coefficient = 0.5  # s^-1
    time_step = 2.0  # s
    expected_p2_factor = np.exp(-coefficient * time_step)

    assert expected_p2_factor == pytest.approx(np.exp(-1.0))
    assert dilution_step_gpu(particles, gas, coefficient, time_step) == (
        particles,
        gas,
    )
    _assert_concentrations_unchanged(particles, gas, snapshots)


def test_zero_box_scalar_and_per_box_calls_are_identity_no_writes() -> None:
    """Empty box dimensions do not impose deferred P3 schema validation."""
    wp = _warp()
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers(n_boxes=0)
    snapshots = _concentration_snapshots(particles, gas)
    assert dilution_step_gpu(particles, gas, 1.0, 1.0) == (particles, gas)
    coefficient = wp.zeros(0, dtype=wp.float64, device="cpu")
    assert dilution_step_gpu(particles, gas, coefficient, 1.0) == (
        particles,
        gas,
    )
    _assert_concentrations_unchanged(particles, gas, snapshots)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), 1j, "one", None, [1.0], np.array([1.0])],
)
def test_unsupported_scalar_coefficient_raises_typeerror_before_time(
    value,
) -> None:
    """Unsupported scalar forms fail before time validation or state access."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    with pytest.raises(TypeError, match="coefficient.*real scalar"):
        dilution_step_gpu(None, None, value, cast(Any, "invalid"))


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_scalar_domains_raise_valueerror_before_time(value) -> None:
    """Negative and nonfinite scalar coefficients have stable diagnostics."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    with pytest.raises(ValueError, match="coefficient.*finite and nonnegative"):
        dilution_step_gpu(None, None, value, cast(Any, "invalid"))


def test_invalid_scalar_coefficient_leaves_container_concentrations_unchanged() -> (
    None
):
    """Coefficient-domain failures occur before any P1 caller-state write."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    with pytest.raises(ValueError, match="coefficient.*finite and nonnegative"):
        dilution_step_gpu(particles, gas, -1.0, 1.0)
    _assert_concentrations_unchanged(particles, gas, snapshots)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        1j,
        "one",
        None,
        [1.0],
        np.array([1.0]),
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_time_step_raises_before_container_access(value) -> None:
    """A valid coefficient plus invalid time step does not access containers."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    error = (
        TypeError
        if (
            isinstance(value, (str, type(None), list, np.ndarray))
            or isinstance(value, (bool, np.bool_))
            or value == 1j
        )
        else ValueError
    )
    with pytest.raises(error, match="time_step"):
        dilution_step_gpu(None, None, 1.0, value)


def test_valid_scalars_access_particle_masses_only_after_validation() -> None:
    """P1 retrieves particle metadata only after scalar and time validation."""
    from particula.gpu.kernels.dilution import dilution_step_gpu

    with pytest.raises(AttributeError, match="masses"):
        dilution_step_gpu(object(), object(), 1.0, 1.0)


@pytest.mark.parametrize(
    ("coefficient", "message"),
    [
        ("float32", "dtype float64"),
        ("rank", "rank 1"),
        ("shape", "shape must match"),
    ],
)
def test_invalid_warp_coefficient_metadata_raises_without_writes(
    coefficient, message
) -> None:
    """Warp metadata failures leave all caller-owned concentrations unchanged."""
    wp = _warp()
    from particula.gpu.kernels.dilution import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    if coefficient == "float32":
        value = wp.ones(2, dtype=wp.float32, device="cpu")
    elif coefficient == "rank":
        value = wp.ones((1, 2), dtype=wp.float64, device="cpu")
    else:
        value = wp.ones(1, dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, value, 1.0)
    _assert_concentrations_unchanged(particles, gas, snapshots)


def test_warp_coefficient_device_mismatch_raises_without_writes() -> None:
    """A second available Warp device must not be accepted implicitly."""
    wp = _warp()
    from particula.gpu.kernels.dilution import dilution_step_gpu

    try:
        alternate_device = wp.get_device("cuda")
    except RuntimeError:
        pytest.skip("CUDA is unavailable for cross-device metadata validation.")

    particles, gas = _containers()
    snapshots = _concentration_snapshots(particles, gas)
    coefficient = wp.ones(2, dtype=wp.float64, device=alternate_device)
    with pytest.raises(ValueError, match="device must match"):
        dilution_step_gpu(particles, gas, coefficient, 1.0)
    _assert_concentrations_unchanged(particles, gas, snapshots)
