"""Tests for fixed-shape P2 GPU dilution execution."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp


def _warp():
    """Import Warp at test runtime to preserve marker deselection."""
    return pytest.importorskip("warp")


def _containers(
    n_boxes: int = 2,
    n_particles: int = 2,
    n_species: int = 2,
    device: str = "cpu",
):
    """Build discriminating fixed-schema particle and gas containers."""
    wp = _warp()
    from particula.gpu import WarpGasData, WarpParticleData

    particles = WarpParticleData()
    particles.masses = wp.ones(
        (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
    )
    particle_concentration = np.arange(n_boxes * n_particles, dtype=np.float64)
    particles.concentration = wp.array(
        particle_concentration.reshape(n_boxes, n_particles),
        dtype=wp.float64,
        device=device,
    )
    particles.charge = wp.ones(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    particles.density = wp.ones(n_species, dtype=wp.float64, device=device)
    particles.volume = wp.ones(n_boxes, dtype=wp.float64, device=device)

    gas = WarpGasData()
    gas.molar_mass = wp.ones(n_species, dtype=wp.float64, device=device)
    gas_concentration = 2.0 + np.arange(n_boxes * n_species, dtype=np.float64)
    gas.concentration = wp.array(
        gas_concentration.reshape(n_boxes, n_species),
        dtype=wp.float64,
        device=device,
    )
    gas.vapor_pressure = wp.ones(
        (n_boxes, n_species), dtype=wp.float64, device=device
    )
    gas.partitioning = wp.ones(
        (n_boxes, n_species), dtype=wp.int32, device=device
    )
    return particles, gas


def _state_snapshots(particles, gas, coefficient=None) -> dict[str, np.ndarray]:
    """Copy all mutable P2 state, including protected caller-owned fields."""
    snapshots = {
        "particle_masses": particles.masses.numpy().copy(),
        "particle_concentration": particles.concentration.numpy().copy(),
        "particle_charge": particles.charge.numpy().copy(),
        "particle_density": particles.density.numpy().copy(),
        "particle_volume": particles.volume.numpy().copy(),
        "gas_molar_mass": gas.molar_mass.numpy().copy(),
        "gas_concentration": gas.concentration.numpy().copy(),
        "gas_vapor_pressure": gas.vapor_pressure.numpy().copy(),
        "gas_partitioning": gas.partitioning.numpy().copy(),
    }
    if coefficient is not None:
        snapshots["coefficient"] = coefficient.numpy().copy()
    return snapshots


def _assert_protected_state(
    particles, gas, snapshots, coefficient=None
) -> None:
    """Assert that fields other than the two concentration arrays are unchanged."""
    current = _state_snapshots(particles, gas, coefficient)
    for field in current:
        if "concentration" not in field:
            npt.assert_array_equal(current[field], snapshots[field])


def _assert_identities(
    particles, gas, returned_particles, returned_gas, field_objects
) -> None:
    """Assert container and every caller-owned field retains its identity."""
    assert returned_particles is particles
    assert returned_gas is gas
    for current, original in zip(
        (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
            gas.molar_mass,
            gas.concentration,
            gas.vapor_pressure,
            gas.partitioning,
        ),
        field_objects,
        strict=True,
    ):
        assert current is original


def test_package_export_is_the_sole_supported_entry_point() -> None:
    """Publish only the public dilution step through the kernel package."""
    from particula.gpu import kernels
    from particula.gpu.kernels import dilution as dilution_module
    from particula.gpu.kernels import dilution_step_gpu

    assert dilution_step_gpu is dilution_module.dilution_step_gpu
    assert "dilution_step_gpu" in kernels.__all__
    assert (
        inspect.signature(dilution_step_gpu).return_annotation
        == "tuple[Any, Any]"
    )
    for private_name in (
        "_dilution_factors",
        "_apply_particle_dilution",
        "_apply_gas_dilution",
        "_normalize_coefficient",
    ):
        assert private_name not in kernels.__all__
        with pytest.raises(AttributeError):
            getattr(kernels, private_name)


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


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_normalize_per_box_coefficient_defers_value_validation(value) -> None:
    """Metadata-valid per-box values retain P3 physical-value deferral."""
    wp = _warp()
    from particula.gpu.kernels.dilution import _normalize_coefficient

    coefficient = wp.array([value, 0.0], dtype=wp.float64, device="cpu")
    assert (
        _normalize_coefficient(coefficient, 2, coefficient.device)
        is coefficient
    )


@pytest.mark.parametrize("coefficient", [0.5, np.float64(1.5)])
def test_scalar_dilution_matches_independent_oracle(coefficient) -> None:
    """Scalar P2 dilution mutates only concentrations by the E6-F1 factor."""
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _state_snapshots(particles, gas)
    field_objects = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.vapor_pressure,
        gas.partitioning,
    )
    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, 2.0
    )

    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    _assert_protected_state(particles, gas, snapshots)
    factor = np.exp(-coefficient * 2.0)
    npt.assert_allclose(
        particles.concentration.numpy(),
        snapshots["particle_concentration"] * factor,
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        snapshots["gas_concentration"] * factor,
        rtol=1e-12,
        atol=0.0,
    )


def test_per_box_dilution_preserves_zero_slots_and_compounds() -> None:
    """Nonuniform factors apply per box and leave zero/inactive slots zero."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    particles.concentration = wp.array(
        [[0.0, 2.0], [3.0, 0.0]], dtype=wp.float64, device="cpu"
    )
    gas.concentration = wp.array(
        [[0.0, 4.0], [5.0, 0.0]], dtype=wp.float64, device="cpu"
    )
    coefficient = wp.array([0.25, 1.0], dtype=wp.float64, device="cpu")
    snapshots = _state_snapshots(particles, gas, coefficient)
    factors = np.exp(-coefficient.numpy()[:, None] * 2.0)

    dilution_step_gpu(particles, gas, coefficient, 2.0)
    npt.assert_allclose(
        particles.concentration.numpy(),
        snapshots["particle_concentration"] * factors,
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        snapshots["gas_concentration"] * factors,
        rtol=1e-12,
        atol=0.0,
    )
    dilution_step_gpu(particles, gas, coefficient, 2.0)
    npt.assert_allclose(
        particles.concentration.numpy(),
        snapshots["particle_concentration"] * factors**2,
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        snapshots["gas_concentration"] * factors**2,
        rtol=1e-12,
        atol=0.0,
    )
    _assert_protected_state(particles, gas, snapshots, coefficient)


def test_zero_scalar_and_zero_time_short_circuit_bad_concentration_metadata() -> (
    None
):
    """Scalar zero and zero time return before concentration metadata access."""
    from particula.gpu.kernels import dilution_step_gpu

    particles = SimpleNamespace(concentration="not a Warp array")
    gas = SimpleNamespace(concentration="not a Warp array")
    assert dilution_step_gpu(particles, gas, 0.0, 1.0) == (particles, gas)
    assert dilution_step_gpu(particles, gas, 1.0, 0.0) == (particles, gas)


def test_per_box_zero_time_short_circuits_before_concentration_access() -> None:
    """A valid per-box coefficient returns before container metadata access."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    coefficient = wp.array([np.inf, 0.5], dtype=wp.float64, device="cpu")
    snapshots = _state_snapshots(particles, gas, coefficient)
    field_objects = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.vapor_pressure,
        gas.partitioning,
    )

    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, 0.0
    )

    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    _assert_protected_state(particles, gas, snapshots, coefficient)
    npt.assert_array_equal(
        particles.concentration.numpy(), snapshots["particle_concentration"]
    )
    npt.assert_array_equal(
        gas.concentration.numpy(), snapshots["gas_concentration"]
    )

    malformed_particles = SimpleNamespace(concentration="not a Warp array")
    malformed_gas = SimpleNamespace(concentration="not a Warp array")
    assert dilution_step_gpu(
        malformed_particles, malformed_gas, coefficient, 0.0
    ) == (malformed_particles, malformed_gas)


def test_per_box_zero_coefficient_box_is_write_free() -> None:
    """A zero-coefficient box retains exact sentinel concentration values."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    coefficient = wp.array([0.0, 0.5], dtype=wp.float64, device="cpu")
    snapshots = _state_snapshots(particles, gas, coefficient)
    dilution_step_gpu(particles, gas, coefficient, 3.0)
    npt.assert_array_equal(
        particles.concentration.numpy()[0],
        snapshots["particle_concentration"][0],
    )
    npt.assert_array_equal(
        gas.concentration.numpy()[0], snapshots["gas_concentration"][0]
    )
    factor = np.exp(-1.5)
    npt.assert_allclose(
        particles.concentration.numpy()[1],
        snapshots["particle_concentration"][1] * factor,
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy()[1],
        snapshots["gas_concentration"][1] * factor,
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "n_boxes,n_particles,n_species", [(0, 0, 0), (2, 0, 2), (2, 2, 0)]
)
def test_zero_extent_concentrations_are_supported(
    n_boxes, n_particles, n_species
) -> None:
    """Zero extents omit application launches while nonempty fields dilute."""
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers(n_boxes, n_particles, n_species)
    snapshots = _state_snapshots(particles, gas)
    field_objects = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.vapor_pressure,
        gas.partitioning,
    )
    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, 0.5, 1.0
    )
    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    factor = np.exp(-0.5)
    npt.assert_allclose(
        particles.concentration.numpy(),
        snapshots["particle_concentration"] * factor,
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        snapshots["gas_concentration"] * factor,
        rtol=1e-12,
        atol=0.0,
    )
    _assert_protected_state(particles, gas, snapshots)


@pytest.mark.parametrize(
    "field,value,message",
    [
        (
            "particle",
            "not a Warp array",
            "particles.concentration must be a Warp array",
        ),
        ("gas", "not a Warp array", "gas.concentration must be a Warp array"),
    ],
)
def test_non_warp_concentrations_reject_before_writes(
    field, value, message
) -> None:
    """Non-no-op calls reject malformed concentration storage before launches."""
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _state_snapshots(particles, gas)
    if field == "particle":
        particles = SimpleNamespace(
            masses=particles.masses,
            concentration=value,
        )
    else:
        gas = SimpleNamespace(concentration=value)
    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)
    if field == "particle":
        npt.assert_array_equal(
            gas.concentration.numpy(), snapshots["gas_concentration"]
        )
    else:
        npt.assert_array_equal(
            particles.concentration.numpy(), snapshots["particle_concentration"]
        )


@pytest.mark.parametrize(
    ("field", "shape", "message"),
    [
        ("particle", (4,), "particles.concentration must have rank 2"),
        (
            "particle",
            (1, 2),
            "particles.concentration box dimension must match",
        ),
        ("gas", (4,), "gas.concentration must have rank 2"),
        ("gas", (1, 2), "gas.concentration box dimension must match"),
    ],
)
def test_concentration_shape_errors_reject_before_writes(
    field, shape, message
) -> None:
    """Non-no-op calls reject incompatible launch shapes before writes."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _state_snapshots(particles, gas)
    if field == "particle":
        particles.concentration = wp.ones(shape, dtype=wp.float64, device="cpu")
    else:
        gas.concentration = wp.ones(shape, dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    if field == "particle":
        npt.assert_array_equal(
            gas.concentration.numpy(), snapshots["gas_concentration"]
        )
    else:
        npt.assert_array_equal(
            particles.concentration.numpy(), snapshots["particle_concentration"]
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("particle", "dtype float64"),
        ("gas", "dtype float64"),
    ],
)
def test_float32_concentrations_reject_before_any_write(field, message) -> None:
    """Float32 concentration metadata fails before either application launch."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    particle_values = particles.concentration.numpy().copy()
    gas_values = gas.concentration.numpy().copy()
    if field == "particle":
        particles = SimpleNamespace(
            masses=particles.masses,
            concentration=wp.array(
                particle_values, dtype=wp.float32, device="cpu"
            ),
        )
    else:
        gas = SimpleNamespace(
            concentration=wp.array(gas_values, dtype=wp.float32, device="cpu")
        )

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    npt.assert_array_equal(particles.concentration.numpy(), particle_values)
    npt.assert_array_equal(gas.concentration.numpy(), gas_values)


@pytest.mark.cuda
@pytest.mark.parametrize("mismatch", ["coefficient", "particle", "gas"])
def test_device_mismatch_rejects_before_any_write(mismatch) -> None:
    """Cross-device coefficient and concentration metadata rejects prelaunch."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    particle_values = particles.concentration.numpy().copy()
    gas_values = gas.concentration.numpy().copy()
    coefficient = 1.0
    if mismatch == "coefficient":
        coefficient = wp.ones(2, dtype=wp.float64, device="cuda:0")
    elif mismatch == "particle":
        particles = SimpleNamespace(
            masses=particles.masses,
            concentration=wp.array(
                particle_values, dtype=wp.float64, device="cuda:0"
            ),
        )
    else:
        gas = SimpleNamespace(
            concentration=wp.array(
                gas_values, dtype=wp.float64, device="cuda:0"
            )
        )

    with pytest.raises(ValueError, match="device must match particle device"):
        dilution_step_gpu(particles, gas, coefficient, 1.0)

    npt.assert_array_equal(particles.concentration.numpy(), particle_values)
    npt.assert_array_equal(gas.concentration.numpy(), gas_values)


@pytest.mark.parametrize(
    ("coefficient_kind", "message"),
    [
        ("float32", "coefficient must use dtype float64"),
        ("shape", "coefficient shape must match"),
    ],
)
def test_invalid_coefficient_metadata_rejects_before_writes(
    coefficient_kind, message
) -> None:
    """Invalid coefficient metadata leaves both concentration arrays unchanged."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    snapshots = _state_snapshots(particles, gas)
    if coefficient_kind == "float32":
        coefficient = wp.ones(2, dtype=wp.float32, device="cpu")
    else:
        coefficient = wp.ones(1, dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, coefficient, 1.0)

    npt.assert_array_equal(
        particles.concentration.numpy(), snapshots["particle_concentration"]
    )
    npt.assert_array_equal(
        gas.concentration.numpy(), snapshots["gas_concentration"]
    )


def test_per_box_coefficient_rank_error_precedes_time_and_state_access() -> (
    None
):
    """Malformed per-box coefficient rank fails before later input access."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    coefficient = wp.ones((2, 1), dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError, match="coefficient must have rank 1"):
        dilution_step_gpu(object(), object(), coefficient, cast(Any, "invalid"))


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_scalar_domains_raise_before_time_or_state_access(
    value,
) -> None:
    """Invalid scalar coefficients retain P1 validation ordering."""
    from particula.gpu.kernels import dilution_step_gpu

    with pytest.raises(ValueError, match="coefficient.*finite and nonnegative"):
        dilution_step_gpu(None, None, value, cast(Any, "invalid"))


@pytest.mark.parametrize(
    "value", [True, 1j, "one", None, [1.0], np.array([1.0])]
)
def test_unsupported_scalar_coefficient_raises_before_time_or_state_access(
    value,
) -> None:
    """Unsupported scalar coefficient forms retain P1 validation ordering."""
    from particula.gpu.kernels import dilution_step_gpu

    with pytest.raises(TypeError, match="coefficient.*real scalar"):
        dilution_step_gpu(None, None, value, cast(Any, "invalid"))


@pytest.mark.parametrize("value", [True, "one", None, [1.0], np.array([1.0])])
def test_invalid_time_step_raises_before_container_access(value) -> None:
    """A valid coefficient plus invalid time does not access containers."""
    from particula.gpu.kernels import dilution_step_gpu

    with pytest.raises((TypeError, ValueError), match="time_step"):
        dilution_step_gpu(None, None, 1.0, value)


def test_invalid_coefficient_metadata_precedes_time_and_particle_access() -> (
    None
):
    """Malformed per-box metadata wins over later invalid inputs."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    coefficient = wp.ones(2, dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="coefficient must use dtype float64"):
        dilution_step_gpu(object(), object(), coefficient, cast(Any, "invalid"))
