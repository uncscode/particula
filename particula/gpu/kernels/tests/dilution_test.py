"""Tests for fixed-shape P2 GPU dilution execution."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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


def _snapshot_field(value: Any) -> dict[str, Any]:
    """Record field identity, metadata, and values without changing it."""
    snapshot = {
        "identity": value,
        "shape": getattr(value, "shape", None),
        "dtype": getattr(value, "dtype", None),
        "device": getattr(value, "device", None),
    }
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        snapshot["values"] = to_numpy().copy()
    return snapshot


def _full_state_snapshot(
    particles: Any, gas: Any, coefficient: Any
) -> dict[str, Any]:
    """Record every fixture container and caller-owned field for rejection."""
    snapshot = {
        "particles": particles,
        "gas": gas,
        "coefficient": _snapshot_field(coefficient),
        "fields": {},
    }
    for container_name, container in (("particles", particles), ("gas", gas)):
        for field in vars(container):
            snapshot["fields"][(container_name, field)] = _snapshot_field(
                getattr(container, field)
            )
    return snapshot


def _assert_full_state_unchanged(
    particles: Any, gas: Any, coefficient: Any, snapshot: dict[str, Any]
) -> None:
    """Assert rejected preflight preserved every fixture object and value."""
    assert particles is snapshot["particles"]
    assert gas is snapshot["gas"]
    for value, expected in ((coefficient, snapshot["coefficient"]),):
        assert value is expected["identity"]
        assert getattr(value, "shape", None) == expected["shape"]
        assert getattr(value, "dtype", None) == expected["dtype"]
        assert getattr(value, "device", None) == expected["device"]
        if "values" in expected:
            assert np.array_equal(
                value.numpy(), expected["values"], equal_nan=True
            )
    for (container_name, field), expected in snapshot["fields"].items():
        container = particles if container_name == "particles" else gas
        value = getattr(container, field)
        assert value is expected["identity"]
        assert getattr(value, "shape", None) == expected["shape"]
        assert getattr(value, "dtype", None) == expected["dtype"]
        assert getattr(value, "device", None) == expected["device"]
        if "values" in expected:
            assert np.array_equal(
                value.numpy(), expected["values"], equal_nan=True
            )


def _forbid_preflight_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if preflight allocates factors or launches dilution kernels."""
    from particula.gpu.kernels import dilution as dilution_module

    def unexpected_side_effect(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("preflight allocated storage or launched a kernel")

    monkeypatch.setattr(dilution_module.wp, "full", unexpected_side_effect)
    monkeypatch.setattr(dilution_module.wp, "empty", unexpected_side_effect)
    original_launch = dilution_module.wp.launch

    def reject_dilution_launch(kernel: Any, *args: Any, **kwargs: Any) -> None:
        assert kernel in (
            dilution_module._scan_nonnegative_finite_1d,
            dilution_module._scan_nonnegative_finite_2d,
        )
        original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(dilution_module.wp, "launch", reject_dilution_launch)


@dataclass(frozen=True)
class P4DilutionCase:
    """Immutable finite-step fixture for deterministic P4 parity coverage."""

    name: str
    particle_concentration: np.ndarray
    gas_concentration: np.ndarray
    coefficient: float | np.ndarray
    time_step: float


P4_CASES = (
    P4DilutionCase(
        name="one_box_scalar_zero_cells",
        particle_concentration=np.array([[0.0, 2.0]], dtype=np.float64),
        gas_concentration=np.array([[0.0, 4.0]], dtype=np.float64),
        coefficient=0.25,
        time_step=2.0,
    ),
    P4DilutionCase(
        name="multi_box_scalar_multi_species",
        particle_concentration=np.array(
            [[1.0, 0.0], [3.0, 5.0]], dtype=np.float64
        ),
        gas_concentration=np.array(
            [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]], dtype=np.float64
        ),
        coefficient=np.float64(0.5),
        time_step=1.5,
    ),
    P4DilutionCase(
        name="multi_box_per_box_nonuniform",
        particle_concentration=np.array(
            [[0.0, 1.0], [2.0, 0.0], [4.0, 8.0]], dtype=np.float64
        ),
        gas_concentration=np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]], dtype=np.float64
        ),
        coefficient=np.array([0.0, 0.25, 1.0], dtype=np.float64),
        time_step=2.0,
    ),
)

# These bounds measure deterministic float64 NumPy/Warp agreement, not bitwise
# replay across execution backends.
P4_PARITY_RTOL = 1e-12
P4_PARITY_ATOL = 0.0


def _p4_numpy_oracle(
    particle_concentration: np.ndarray,
    gas_concentration: np.ndarray,
    coefficient: float | np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent finite-step dilution expectations using NumPy."""
    n_boxes = particle_concentration.shape[0]
    coefficient_array = np.asarray(coefficient, dtype=np.float64)
    if coefficient_array.ndim == 0:
        coefficient_array = np.full(
            n_boxes, coefficient_array, dtype=np.float64
        )
    factors = np.exp(-coefficient_array[:, None] * time_step)
    return (
        particle_concentration.copy() * factors,
        gas_concentration.copy() * factors,
    )


def _assert_p4_case_matches_reference(
    case: P4DilutionCase, device: str
) -> None:
    """Assert one P4 finite-step case against its independent NumPy oracle."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    n_boxes, n_particles = case.particle_concentration.shape
    n_species = case.gas_concentration.shape[1]
    particles, gas = _containers(n_boxes, n_particles, n_species, device)
    particles.concentration = wp.array(
        case.particle_concentration.copy(), dtype=wp.float64, device=device
    )
    gas.concentration = wp.array(
        case.gas_concentration.copy(), dtype=wp.float64, device=device
    )
    coefficient: Any = case.coefficient
    if isinstance(case.coefficient, np.ndarray):
        coefficient = wp.array(
            case.coefficient.copy(), dtype=wp.float64, device=device
        )
    coefficient_object = coefficient
    snapshots = _state_snapshots(
        particles,
        gas,
        coefficient if isinstance(case.coefficient, np.ndarray) else None,
    )
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
    expected_particle, expected_gas = _p4_numpy_oracle(
        case.particle_concentration.copy(),
        case.gas_concentration.copy(),
        case.coefficient,
        case.time_step,
    )

    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, case.time_step
    )

    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    _assert_protected_state(
        particles,
        gas,
        snapshots,
        coefficient if isinstance(case.coefficient, np.ndarray) else None,
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        expected_particle,
        rtol=P4_PARITY_RTOL,
        atol=P4_PARITY_ATOL,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        expected_gas,
        rtol=P4_PARITY_RTOL,
        atol=P4_PARITY_ATOL,
    )
    if isinstance(case.coefficient, np.ndarray):
        assert coefficient is coefficient_object
        npt.assert_array_equal(coefficient.numpy(), case.coefficient)


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


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_per_box_coefficient_rejects_invalid_values(value) -> None:
    """Public preflight rejects invalid per-box coefficient values."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    coefficient = wp.array([value, 0.0], dtype=wp.float64, device="cpu")
    particles, gas = _containers()
    with pytest.raises(
        ValueError, match="coefficient must be finite and nonnegative"
    ):
        dilution_step_gpu(particles, gas, coefficient, 1.0)


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


# P4 deterministic CPU/Warp finite-step parity and invariant coverage.
@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("case", P4_CASES, ids=lambda case: case.name)
def test_p4_warp_cpu_cases_match_independent_reference(
    case: P4DilutionCase,
) -> None:
    """Warp CPU finite-step dilution matches each independent P4 reference."""
    _assert_p4_case_matches_reference(case, device="cpu")


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("case", P4_CASES, ids=lambda case: case.name)
def test_p4_cuda_cases_match_independent_reference(
    case: P4DilutionCase,
) -> None:
    """CUDA finite-step dilution matches each independent P4 reference."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    _assert_p4_case_matches_reference(case, device="cuda:0")


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_p4_warp_cpu_repeated_per_box_dilution_matches_reference() -> None:
    """Repeated nonuniform Warp CPU dilution compounds independently by box."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    case = P4_CASES[2]
    n_boxes, n_particles = case.particle_concentration.shape
    n_species = case.gas_concentration.shape[1]
    particles, gas = _containers(n_boxes, n_particles, n_species)
    particles.concentration = wp.array(
        case.particle_concentration.copy(), dtype=wp.float64, device="cpu"
    )
    gas.concentration = wp.array(
        case.gas_concentration.copy(), dtype=wp.float64, device="cpu"
    )
    coefficient = wp.array(
        case.coefficient.copy(), dtype=wp.float64, device="cpu"
    )
    coefficient_object = coefficient
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
    expected_particle, expected_gas = _p4_numpy_oracle(
        case.particle_concentration.copy(),
        case.gas_concentration.copy(),
        case.coefficient,
        case.time_step,
    )
    expected_particle, expected_gas = _p4_numpy_oracle(
        expected_particle, expected_gas, case.coefficient, case.time_step
    )

    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, case.time_step
    )
    returned_particles, returned_gas = dilution_step_gpu(
        returned_particles, returned_gas, coefficient, case.time_step
    )

    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    _assert_protected_state(particles, gas, snapshots, coefficient)
    assert coefficient is coefficient_object
    npt.assert_array_equal(coefficient.numpy(), case.coefficient)
    npt.assert_allclose(
        particles.concentration.numpy(),
        expected_particle,
        rtol=P4_PARITY_RTOL,
        atol=P4_PARITY_ATOL,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        expected_gas,
        rtol=P4_PARITY_RTOL,
        atol=P4_PARITY_ATOL,
    )


@pytest.mark.parametrize(("coefficient", "time_step"), [(0.0, 1.0), (1.0, 0.0)])
def test_p4_warp_cpu_exact_no_ops_preserve_all_state(
    coefficient: float, time_step: float
) -> None:
    """Valid scalar-zero and zero-time calls preserve concentrations exactly."""
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
        particles, gas, coefficient, time_step
    )

    _assert_identities(
        particles, gas, returned_particles, returned_gas, field_objects
    )
    _assert_protected_state(particles, gas, snapshots)
    npt.assert_array_equal(
        particles.concentration.numpy(), snapshots["particle_concentration"]
    )
    npt.assert_array_equal(
        gas.concentration.numpy(), snapshots["gas_concentration"]
    )


def test_zero_scalar_and_zero_time_validate_container_metadata() -> None:
    """Scalar zero and zero time still require complete preflight."""
    from particula.gpu.kernels import dilution_step_gpu

    particles = SimpleNamespace(concentration="not a Warp array")
    gas = SimpleNamespace(concentration="not a Warp array")
    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        dilution_step_gpu(particles, gas, 0.0, 1.0)
    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        dilution_step_gpu(particles, gas, 1.0, 0.0)


def test_per_box_zero_time_validates_coefficient_and_containers() -> None:
    """A zero time step occurs only after full read-only preflight."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    coefficient = wp.array([np.inf, 0.5], dtype=wp.float64, device="cpu")
    snapshots = _state_snapshots(particles, gas, coefficient)

    with pytest.raises(
        ValueError, match="coefficient must be finite and nonnegative"
    ):
        dilution_step_gpu(particles, gas, coefficient, 0.0)

    _assert_protected_state(particles, gas, snapshots, coefficient)
    npt.assert_array_equal(
        particles.concentration.numpy(), snapshots["particle_concentration"]
    )
    npt.assert_array_equal(
        gas.concentration.numpy(), snapshots["gas_concentration"]
    )

    malformed_particles = SimpleNamespace(concentration="not a Warp array")
    malformed_gas = SimpleNamespace(concentration="not a Warp array")
    with pytest.raises(
        ValueError, match=r"particles\.masses must be a Warp array"
    ):
        dilution_step_gpu(malformed_particles, malformed_gas, coefficient, 0.0)


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


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_time_step_domain_raises_before_container_access(value) -> None:
    """Non-finite and negative time steps reject before container access."""
    from particula.gpu.kernels import dilution_step_gpu

    with pytest.raises(
        ValueError, match="time_step must be finite and nonnegative"
    ):
        dilution_step_gpu(None, None, 1.0, value)


@pytest.mark.parametrize(
    ("mass_kind", "message"),
    [
        ("missing", "particles.masses must be a Warp array"),
        ("dtype", "particles.masses must use dtype float64"),
        ("rank", "particles.masses must have rank 3"),
    ],
)
def test_mass_schema_rejections_are_atomic_and_side_effect_free(
    mass_kind, message, monkeypatch
) -> None:
    """Missing, wrong-dtype, and wrong-rank masses reject before effects."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    if mass_kind == "missing":
        particles = SimpleNamespace(concentration=particles.concentration)
    elif mass_kind == "dtype":
        particles = SimpleNamespace(
            masses=wp.ones((2, 2, 2), dtype=wp.float32, device="cpu"),
            concentration=particles.concentration,
        )
    else:
        particles = SimpleNamespace(
            masses=wp.ones((2, 2), dtype=wp.float64, device="cpu"),
            concentration=particles.concentration,
        )
    snapshot = _full_state_snapshot(particles, gas, 1.0)
    _forbid_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    _assert_full_state_unchanged(particles, gas, 1.0, snapshot)


@pytest.mark.parametrize(
    ("container_name", "message"),
    [
        ("particles", "particles.concentration must be a Warp array"),
        ("gas", "gas.concentration must be a Warp array"),
    ],
)
def test_missing_concentration_fields_are_atomic_and_side_effect_free(
    container_name, message, monkeypatch
) -> None:
    """Missing concentration fields reject without allocation, launch, or writes."""
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    if container_name == "particles":
        particles = SimpleNamespace(masses=particles.masses)
    else:
        gas = SimpleNamespace()
    snapshot = _full_state_snapshot(particles, gas, 1.0)
    _forbid_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    _assert_full_state_unchanged(particles, gas, 1.0, snapshot)


def test_invalid_coefficient_metadata_precedes_time_and_particle_access() -> (
    None
):
    """Malformed per-box metadata wins over later invalid inputs."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    coefficient = wp.ones(2, dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="coefficient must use dtype float64"):
        dilution_step_gpu(object(), object(), coefficient, cast(Any, "invalid"))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("coefficient_form", "coefficient must have rank 1"),
        ("time", "time_step must be a real scalar"),
        ("masses", "particles.masses must be a Warp array"),
        ("coefficient_values", "coefficient must be finite and nonnegative"),
        (
            "particle_values",
            "particles.concentration must be finite and nonnegative",
        ),
        ("gas_values", "gas.concentration must be finite and nonnegative"),
    ],
)
def test_preflight_rejections_are_atomic_and_side_effect_free(
    failure, message, monkeypatch
) -> None:
    """Each preflight group rejects before allocation, launch, or mutation."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    coefficient: Any = 1.0
    time_step: Any = 1.0
    if failure == "coefficient_form":
        coefficient = wp.ones((2, 1), dtype=wp.float64, device="cpu")
    elif failure == "time":
        time_step = cast(Any, "invalid")
    elif failure == "masses":
        particles = SimpleNamespace(
            masses="invalid",
            concentration=particles.concentration,
            charge=particles.charge,
            density=particles.density,
            volume=particles.volume,
        )
    elif failure == "coefficient_values":
        coefficient = wp.array([-1.0, 0.5], dtype=wp.float64, device="cpu")
    elif failure == "particle_values":
        particles.concentration = wp.array(
            [[-1.0, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
    else:
        gas.concentration = wp.array(
            [[np.inf, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )

    snapshot = _full_state_snapshot(particles, gas, coefficient)
    _forbid_preflight_side_effects(monkeypatch)
    with pytest.raises((TypeError, ValueError), match=message):
        dilution_step_gpu(particles, gas, coefficient, time_step)
    _assert_full_state_unchanged(particles, gas, coefficient, snapshot)


@pytest.mark.parametrize(("coefficient", "time_step"), [(0.0, 1.0), (1.0, 0.0)])
def test_valid_no_ops_complete_preflight_without_side_effects(
    coefficient, time_step, monkeypatch
) -> None:
    """Valid scalar-zero and zero-time calls are fully validated write-free no-ops."""
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    snapshot = _full_state_snapshot(particles, gas, coefficient)
    _forbid_preflight_side_effects(monkeypatch)

    returned_particles, returned_gas = dilution_step_gpu(
        particles, gas, coefficient, time_step
    )

    assert returned_particles is particles
    assert returned_gas is gas
    _assert_full_state_unchanged(particles, gas, coefficient, snapshot)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["particle", "gas"])
def test_invalid_concentration_values_preserve_full_caller_state(
    field, value
) -> None:
    """All nonnegative-domain failures preserve complete caller-owned state."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    if field == "particle":
        particles.concentration = wp.array(
            [[value, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
    else:
        gas.concentration = wp.array(
            [[value, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
    snapshot = _full_state_snapshot(particles, gas, 1.0)
    name = "particles" if field == "particle" else "gas"
    message = f"{name}.concentration must be finite and nonnegative"

    with pytest.raises(ValueError, match=message):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    _assert_full_state_unchanged(particles, gas, 1.0, snapshot)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    ("field", "coefficient", "time_step"),
    [
        ("coefficient", "per_box", 1.0),
        ("coefficient", "per_box", 0.0),
        ("particle", 1.0, 1.0),
        ("particle", 0.0, 1.0),
        ("particle", 1.0, 0.0),
        ("gas", 1.0, 1.0),
        ("gas", 0.0, 1.0),
        ("gas", 1.0, 0.0),
    ],
)
def test_invalid_values_reject_before_no_op_or_dilution_effects(
    field, coefficient, time_step, value, monkeypatch
) -> None:
    """Invalid arrays reject before no-op return, allocation, or dilution."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    if field == "coefficient":
        coefficient = wp.array([value, 0.5], dtype=wp.float64, device="cpu")
        message = "coefficient"
    elif field == "particle":
        particles.concentration = wp.array(
            [[value, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
        message = "particles.concentration"
    else:
        gas.concentration = wp.array(
            [[value, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
        )
        message = "gas.concentration"
    snapshot = _full_state_snapshot(particles, gas, coefficient)
    _forbid_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match=f"{message} must be finite"):
        dilution_step_gpu(particles, gas, coefficient, time_step)

    _assert_full_state_unchanged(particles, gas, coefficient, snapshot)


@pytest.mark.cuda
@pytest.mark.parametrize("field", ["coefficient", "particle", "gas"])
def test_cuda_invalid_values_use_device_scan(field) -> None:
    """CUDA invalid arrays reject through the device scan without source copies."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    from particula.gpu.kernels import dilution as dilution_module
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers(device="cuda:0")
    coefficient: Any = 1.0
    if field == "coefficient":
        coefficient = wp.array([np.nan, 0.5], dtype=wp.float64, device="cuda:0")
        message = "coefficient"
    elif field == "particle":
        particles.concentration = wp.array(
            [[np.nan, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cuda:0"
        )
        message = "particles.concentration"
    else:
        gas.concentration = wp.array(
            [[np.nan, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cuda:0"
        )
        message = "gas.concentration"
    snapshot = _full_state_snapshot(particles, gas, coefficient)

    assert "values.numpy" not in inspect.getsource(
        dilution_module._validate_nonnegative_finite_values
    )
    with pytest.raises(ValueError, match=f"{message} must be finite"):
        dilution_step_gpu(particles, gas, coefficient, 1.0)
    _assert_full_state_unchanged(particles, gas, coefficient, snapshot)


@pytest.mark.parametrize("field", ["particle", "gas"])
def test_concentration_second_dimension_mismatch_is_rejected(field) -> None:
    """Concentration dimensions must exactly match mass particle/species axes."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    if field == "particle":
        particles.concentration = wp.ones(
            (2, 1), dtype=wp.float64, device="cpu"
        )
    else:
        gas.concentration = wp.ones((2, 1), dtype=wp.float64, device="cpu")
    snapshot = _full_state_snapshot(particles, gas, 1.0)

    name = "particles" if field == "particle" else "gas"
    with pytest.raises(
        ValueError, match=f"{name}.concentration shape must match"
    ):
        dilution_step_gpu(particles, gas, 1.0, 1.0)

    _assert_full_state_unchanged(particles, gas, 1.0, snapshot)


def test_preflight_validation_order_at_remaining_boundaries() -> None:
    """Mass, coefficient, and particle failures precede later validation groups."""
    wp = _warp()
    from particula.gpu.kernels import dilution_step_gpu

    particles, gas = _containers()
    particles = SimpleNamespace(
        masses="invalid",
        concentration=particles.concentration,
        charge=particles.charge,
        density=particles.density,
        volume=particles.volume,
    )
    invalid_coefficient = wp.array([-1.0, 0.5], dtype=wp.float64, device="cpu")
    with pytest.raises(
        ValueError, match="particles.masses must be a Warp array"
    ):
        dilution_step_gpu(particles, gas, invalid_coefficient, 1.0)

    particles, gas = _containers()
    particles = SimpleNamespace(
        masses=particles.masses,
        concentration="invalid",
        charge=particles.charge,
        density=particles.density,
        volume=particles.volume,
    )
    invalid_coefficient = wp.array([-1.0, 0.5], dtype=wp.float64, device="cpu")
    with pytest.raises(
        ValueError, match="coefficient must be finite and nonnegative"
    ):
        dilution_step_gpu(particles, gas, invalid_coefficient, 1.0)

    particles, gas = _containers()
    particles.concentration = wp.array(
        [[-1.0, 1.0], [2.0, 3.0]], dtype=wp.float64, device="cpu"
    )
    gas = SimpleNamespace(
        concentration="invalid",
        molar_mass=gas.molar_mass,
        vapor_pressure=gas.vapor_pressure,
        partitioning=gas.partitioning,
    )
    with pytest.raises(
        ValueError,
        match="particles.concentration must be finite and nonnegative",
    ):
        dilution_step_gpu(particles, gas, 1.0, 1.0)
