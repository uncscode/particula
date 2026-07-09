"""End-to-end tests for GPU coagulation kernels."""

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import inspect

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")

import particula.gpu.kernels.coagulation as coagulation_module  # noqa: E402
from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
    get_brownian_kernel_via_system_state,
)
from particula.gas.environment_data import EnvironmentData  # noqa: E402
from particula.gas.gas_data import GasData  # noqa: E402
from particula.gpu.conversion import (  # noqa: E402
    from_warp_particle_data,
    to_warp_environment_data,
    to_warp_particle_data,
)
from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
)
from particula.gpu.kernels.coagulation import (  # noqa: E402
    _ensure_volume_array,
    _initialize_rng_states,
    _validate_collision_counts,
    _validate_collision_pairs,
    _validate_device_arrays,
    _validate_device_match,
    _validate_max_collisions,
    _validate_particle_arrays,
    _validate_rng_states,
    _validate_time_step,
    apply_coagulation_kernel,
    brownian_coagulation_kernel,
    coagulation_step_gpu,
)
from particula.gpu.properties.gas_properties import (  # noqa: E402
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (  # noqa: E402
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
)
from particula.gpu.tests.cuda_availability import (  # noqa: E402
    cuda_available,
    warp_devices,
)

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
from particula.particles.particle_data import ParticleData  # noqa: E402
from particula.util import constants  # noqa: E402


@pytest.fixture(params=warp_devices(wp))
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
    concentration_scale: float = 1.0,
) -> Any:
    """Create deterministic particle data for coagulation tests."""
    base_masses = np.linspace(1.0e-18, 2.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.2 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.full(
        (n_boxes, n_particles),
        concentration_scale,
        dtype=np.float64,
    )
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create gas data for Brownian kernel reference calculations."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.full((n_boxes, n_species), 1.0e-6, dtype=np.float64)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_environment_data(
    n_boxes: int,
    n_species: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> EnvironmentData:
    """Create deterministic environment data for contract tests."""
    return EnvironmentData(
        temperature=np.full((n_boxes,), temperature, dtype=np.float64),
        pressure=np.full((n_boxes,), pressure, dtype=np.float64),
        saturation_ratio=np.ones((n_boxes, n_species), dtype=np.float64),
    )


def _assert_particles_unchanged(
    gpu_particles: Any,
    initial_particles: ParticleData,
) -> None:
    """Assert GPU particle arrays still match a CPU snapshot."""
    result_particles = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(result_particles.masses, initial_particles.masses)
    npt.assert_allclose(
        result_particles.concentration,
        initial_particles.concentration,
    )


def _accumulate_collision_counts(
    *,
    particles: ParticleData,
    device: str,
    seeds: range,
    time_step: float,
    max_collisions: int,
    temperature: float | Any | None,
    pressure: float | Any | None,
    environment: Any | None = None,
    volume: float | Any | None = None,
) -> np.ndarray:
    """Accumulate per-box collision counts across a small fixed seed set."""
    total_counts = np.zeros(particles.masses.shape[0], dtype=np.int64)

    for seed in seeds:
        gpu_particles = to_warp_particle_data(particles, device=device)
        _, _, collision_counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            volume=volume,
            max_collisions=max_collisions,
            rng_seed=seed,
            environment=environment,
        )
        wp.synchronize()
        total_counts += np.asarray(collision_counts.numpy(), dtype=np.int64)

    return total_counts


def test_coagulation_step_gpu_signature_keeps_environment_keyword_only() -> (
    None
):
    """The RNG reset and environment inputs stay keyword-only."""
    parameters = inspect.signature(coagulation_step_gpu).parameters

    assert parameters["initialize_rng"].default is False
    assert parameters["initialize_rng"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["environment"].kind is inspect.Parameter.KEYWORD_ONLY


def test_coagulation_step_gpu_scalar_positional_call_remains_valid(
    device: str,
) -> None:
    """Legacy positional scalar callers remain source-compatible."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        298.15,
        101325.0,
        0.1,
        max_collisions=4,
        rng_seed=3,
    )
    wp.synchronize()

    assert collision_pairs.shape == (1, 4, 2)
    assert collision_counts.shape == (1,)


def test_coagulation_step_gpu_omitted_rng_states_keeps_legacy_behavior(
    device: str,
) -> None:
    """Omitting ``rng_states`` still allocates usable seeded state."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        rng_seed=19,
        max_collisions=4,
    )
    wp.synchronize()

    assert collision_pairs.shape == (1, 4, 2)
    assert collision_counts.shape == (1,)
    assert np.all(np.asarray(collision_counts.numpy()) >= 0)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, 101325.0),
        (298.15, None),
        (None, 101325.0),
    ],
)
def test_coagulation_step_gpu_rejects_mixed_environment_inputs(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Mixed scalar and environment inputs raise a stable contract error."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)

    with pytest.raises(
        ValueError,
        match="direct temperature/pressure inputs with environment",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_accepts_explicit_environment(
    device: str,
) -> None:
    """Pure ``environment=...`` execution succeeds when inputs are valid."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, scalar_pairs, scalar_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )
    scalar_pairs_np = np.asarray(scalar_pairs.numpy()).copy()
    scalar_counts_np = np.asarray(scalar_counts.numpy()).copy()

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, env_pairs, env_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=None,
        pressure=None,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
        environment=environment,
    )

    npt.assert_array_equal(env_pairs.numpy(), scalar_pairs_np)
    npt.assert_array_equal(env_counts.numpy(), scalar_counts_np)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_coagulation_step_gpu_rejects_missing_scalar_inputs_without_environment(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Scalar-mode calls require both temperature and pressure."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            298.15,
            101325.0,
            "direct temperature/pressure inputs with environment",
        ),
        (
            None,
            None,
            r"\(n_boxes,\)",
        ),
    ],
)
def test_coagulation_step_gpu_contract_errors_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
    message: str,
) -> None:
    """Contract errors fire before volume setup or any Warp launch."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment_data = _make_environment_data(n_boxes=1, n_species=1)
    if temperature is None and pressure is None:
        environment_data.temperature = np.array([298.15, 299.15])
    environment = to_warp_environment_data(environment_data, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (0.0, 101325.0, "temperature must be finite and > 0"),
        (298.15, 0.0, "pressure must be finite and > 0"),
        (float("nan"), 101325.0, "temperature must be finite and > 0"),
    ],
)
def test_coagulation_step_gpu_invalid_scalar_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float,
    pressure: float,
    message: str,
) -> None:
    """Invalid scalar domains fail before any setup or Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == []


def test_coagulation_step_gpu_invalid_environment_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment arrays fail before downstream setup or kernels."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1), device=device
    )
    environment.pressure = wp.array([0.0], dtype=wp.float64, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    original_launch = coagulation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


def test_coagulation_step_gpu_invalid_environment_preserves_buffers_and_particles(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment data leaves caller-owned buffers unchanged."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1), device=device
    )
    environment.pressure = wp.array(
        [101325.0, 0.0],
        dtype=wp.float64,
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(16, dtype=np.int32).reshape(2, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([1, 2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([17, 23], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
            environment=environment,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_accepts_direct_environment_arrays(
    device: str,
) -> None:
    """Direct ``(n_boxes,)`` Warp-array inputs execute successfully."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15, 301.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 100800.0], dtype=wp.float64, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert collision_pairs.shape == (2, 4, 2)
    assert collision_counts.shape == (2,)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, np.array([101325.0, 100800.0], dtype=np.float64)),
        (np.array([298.15, 301.15], dtype=np.float64), 101325.0),
    ],
)
def test_coagulation_step_gpu_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
) -> None:
    """Hybrid direct inputs execute successfully."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    if isinstance(temperature, np.ndarray):
        temperature = wp.array(temperature, dtype=wp.float64, device=device)
    if isinstance(pressure, np.ndarray):
        pressure = wp.array(pressure, dtype=wp.float64, device=device)

    _, collision_pairs, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert collision_pairs.shape == (2, 4, 2)
    assert collision_counts.shape == (2,)


def test_coagulation_step_gpu_preserves_direct_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Direct Warp arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    original_launch = coagulation_module.wp.launch
    launch_dtypes: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            launch_dtypes.append((inputs[4].dtype, inputs[5].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=wp.array([298.15, 301.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 100800.0], dtype=wp.float64, device=device
        ),
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


def test_coagulation_step_gpu_preserves_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explicit environment arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)

    class _EnvironmentLike:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15, 301.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )

    environment = _EnvironmentLike()
    gpu_particles = to_warp_particle_data(particles, device=device)
    original_launch = coagulation_module.wp.launch
    launch_dtypes: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            launch_dtypes.append((inputs[4].dtype, inputs[5].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=None,
        pressure=None,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
        environment=environment,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


def test_coagulation_step_gpu_environment_shape_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays must match ``(n_boxes,)`` before launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=device
    )
    environment.temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_environment_device_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays on the wrong device fail before launch work."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=wrong_device
    )

    with pytest.raises(ValueError, match="environment.temperature device"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_coagulation_step_gpu_direct_temperature_shape_mismatch_raises(
    device: str,
) -> None:
    """Direct temperature arrays must match ``(n_boxes,)`` before launch."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15], dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=101325.0,
            time_step=0.1,
        )


def test_coagulation_step_gpu_direct_pressure_device_mismatch_raises(
    device: str,
) -> None:
    """Direct pressure arrays on the wrong device fail before launch."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    pressure = wp.array([101325.0], dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="pressure device"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=pressure,
            time_step=0.1,
        )


def test_coagulation_step_gpu_reuses_direct_environment_arrays(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Coagulation forwards validated direct arrays without rebuilding them."""
    particles = _make_particle_data(n_boxes=2, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    temperature = wp.array([298.15, 301.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 100800.0], dtype=wp.float64, device=device)
    original_launch = coagulation_module.wp.launch
    forwarded_inputs: list[tuple[Any, Any]] = []

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if getattr(kernel, "key", "") == "brownian_coagulation_kernel":
            forwarded_inputs.append((inputs[4], inputs[5]))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(coagulation_module.wp, "launch", _tracking_launch)

    coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        max_collisions=4,
        rng_seed=3,
    )

    assert forwarded_inputs == [(temperature, pressure)]


def test_coagulation_step_gpu_uniform_direct_arrays_match_scalar_results(
    device: str,
) -> None:
    """Uniform direct arrays stay within the established tolerance band."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    seeds = range(11, 19)
    time_step = 0.5
    max_collisions = 16

    scalar_counts = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=seeds,
        time_step=time_step,
        max_collisions=max_collisions,
        temperature=298.15,
        pressure=101325.0,
    )
    uniform_counts = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=seeds,
        time_step=time_step,
        max_collisions=max_collisions,
        temperature=wp.array([298.15, 298.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 101325.0], dtype=wp.float64, device=device
        ),
    )

    diff = np.abs(uniform_counts - scalar_counts)
    tolerance = np.maximum(3.0 * np.sqrt(np.maximum(scalar_counts, 1.0)), 1.0)

    assert np.all(diff <= tolerance), (
        "Uniform direct arrays should preserve scalar coagulation behavior "
        "within the established stochastic tolerance band."
    )


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_coagulation_step_gpu_missing_scalar_inputs_short_circuit_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Missing scalar inputs fail before setup, launch, or input mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.array(
        np.array([17], dtype=np.uint32), dtype=wp.uint32, device=device
    )
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    result_particles = from_warp_particle_data(gpu_particles, sync=True)
    npt.assert_allclose(result_particles.masses, initial_particles.masses)
    npt.assert_allclose(
        result_particles.concentration,
        initial_particles.concentration,
    )


@wp.kernel
# type: ignore[misc]
def _brownian_kernel_matrix_kernel(
    radii: Any,
    masses: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    kernel_out: Any,
) -> None:
    """Compute Brownian kernel matrix using shared GPU building blocks."""  # type: ignore
    i_idx, j_idx = wp.tid()  # type: ignore[misc]
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_i = knudsen_number_wp(mean_free_path, radii[i_idx])
    knudsen_j = knudsen_number_wp(mean_free_path, radii[j_idx])
    slip_i = cunningham_slip_correction_wp(knudsen_i)
    slip_j = cunningham_slip_correction_wp(knudsen_j)
    mobility_i = aerodynamic_mobility_wp(
        radii[i_idx],
        slip_i,
        dynamic_viscosity,
    )
    mobility_j = aerodynamic_mobility_wp(
        radii[j_idx],
        slip_j,
        dynamic_viscosity,
    )
    diffusivity_i = brownian_diffusivity_wp(
        temperature,
        mobility_i,
        boltzmann_constant,
    )
    diffusivity_j = brownian_diffusivity_wp(
        temperature,
        mobility_j,
        boltzmann_constant,
    )
    speed_i = mean_thermal_speed_wp(
        masses[i_idx],
        temperature,
        boltzmann_constant,
    )
    speed_j = mean_thermal_speed_wp(
        masses[j_idx],
        temperature,
        boltzmann_constant,
    )
    mean_free_path_i = particle_mean_free_path_wp(diffusivity_i, speed_i)
    mean_free_path_j = particle_mean_free_path_wp(diffusivity_j, speed_j)
    g_term_i = g_collection_term_wp(mean_free_path_i, radii[i_idx])
    g_term_j = g_collection_term_wp(mean_free_path_j, radii[j_idx])
    kernel_out[i_idx, j_idx] = brownian_kernel_pair_wp(
        radii[i_idx],
        radii[j_idx],
        diffusivity_i,
        diffusivity_j,
        g_term_i,
        g_term_j,
        speed_i,
        speed_j,
        wp.float64(1.0),
    )


def test_brownian_kernel_matrix_parity_gpu_cpu(device: str) -> None:
    """GPU Brownian kernel matrix matches CPU reference."""
    temperature = 298.15
    pressure = 101325.0
    radii = np.array([1.0e-8, 5.0e-8, 1.0e-7], dtype=np.float64)
    masses = np.array([1.0e-21, 4.0e-21, 8.0e-21], dtype=np.float64)

    expected = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )

    radii_wp = wp.array(radii, dtype=wp.float64, device=device)
    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    kernel_wp = wp.zeros(
        (len(radii), len(radii)), dtype=wp.float64, device=device
    )

    wp.launch(
        _brownian_kernel_matrix_kernel,
        dim=(len(radii), len(radii)),
        inputs=[
            radii_wp,
            masses_wp,
            wp.float64(temperature),
            wp.float64(pressure),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
        ],
        outputs=[kernel_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(kernel_wp.numpy(), expected, rtol=1.0e-7)


def test_coagulation_statistical_collision_rate(device: str) -> None:
    """Collision counts follow expected Brownian rate statistics."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.5
    n_steps = 60
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    total_collisions = 0
    for step_idx in range(n_steps):
        _, _, collision_counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            rng_seed=42 + step_idx,
            max_collisions=16,
        )
        wp.synchronize()
        collision_array = np.asarray(collision_counts.numpy())
        total_collisions += int(collision_array.sum())

    density_value = float(np.asarray(particles.density).ravel().item(0))
    masses_array: np.ndarray = np.asarray(particles.masses, dtype=np.float64)
    masses_slice = np.ravel(masses_array[0])
    radii = np.cbrt(3.0 * masses_slice / (4.0 * np.pi * density_value))
    masses = masses_slice
    kernel_matrix = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )
    kernel_matrix_array = np.asarray(kernel_matrix, dtype=np.float64)
    kernel_values = np.asarray(kernel_matrix_array)[
        np.triu_indices(len(radii), k=1)
    ]
    kernel_values = np.atleast_1d(kernel_values)
    volume = float(np.asarray(particles.volume).ravel().item(0))
    expected_mean = np.sum(kernel_values) * time_step * n_steps / volume
    expected_sigma = np.sqrt(expected_mean)
    assert total_collisions == pytest.approx(
        expected_mean, abs=3.0 * expected_sigma
    )


def test_coagulation_multi_box_independence(device: str) -> None:
    """Collision counts remain isolated per box."""
    time_step = 1.0
    particles = _make_particle_data(n_boxes=3, n_particles=5, n_species=1)
    particles.concentration[1, :] = 0.0
    particles.concentration[2, 0] = 0.0
    temperature = wp.array(
        [300.0, 303.0, 297.0], dtype=wp.float64, device=device
    )
    pressure = wp.array(
        [101325.0, 100500.0, 102000.0],
        dtype=wp.float64,
        device=device,
    )

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=12,
        max_collisions=16,
    )
    wp.synchronize()
    result = np.asarray(collision_counts.numpy())

    assert result.reshape(-1)[1] == 0
    assert result.reshape(-1)[0] >= result.reshape(-1)[2]


def test_coagulation_step_gpu_nonuniform_environment_changes_collision_trend(
    device: str,
) -> None:
    """Nonuniform environment inputs shift box-local collisions directionally."""
    particles = _make_particle_data(n_boxes=2, n_particles=12, n_species=1)
    particles.masses[1, :, :] = particles.masses[0, :, :]
    temperature = np.array([250.0, 350.0], dtype=np.float64)
    pressure = np.array([150000.0, 50000.0], dtype=np.float64)
    environment = to_warp_environment_data(
        EnvironmentData(
            temperature=temperature,
            pressure=pressure,
            saturation_ratio=np.ones((2, 1), dtype=np.float64),
        ),
        device=device,
    )
    density_value = float(np.asarray(particles.density).item())
    mass_values = np.asarray(particles.masses[:, :, 0], dtype=np.float64)
    expected_rates = np.array(
        [
            np.sum(
                np.asarray(
                    get_brownian_kernel_via_system_state(
                        particle_radius=np.cbrt(
                            3.0
                            * box_mass_values
                            / (4.0 * np.pi * density_value)
                        ),
                        particle_mass=box_mass_values,
                        temperature=temp_value,
                        pressure=pressure_value,
                    ),
                    dtype=np.float64,
                )[np.triu_indices(len(box_mass_values), k=1)]
            )
            for box_mass_values, temp_value, pressure_value in zip(
                mass_values,
                temperature,
                pressure,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    collision_totals = _accumulate_collision_counts(
        particles=particles,
        device=device,
        seeds=range(31, 51),
        time_step=0.5,
        max_collisions=64,
        temperature=None,
        pressure=None,
        environment=environment,
        volume=1.0e-14,
    )

    higher_rate_idx = int(np.argmax(expected_rates))
    lower_rate_idx = int(np.argmin(expected_rates))

    assert expected_rates[higher_rate_idx] > expected_rates[lower_rate_idx]
    assert collision_totals.shape == (2,)
    assert (
        collision_totals[higher_rate_idx] > collision_totals[lower_rate_idx]
    ), (
        "Nonuniform environment coverage is directional/statistical: "
        "the box with the larger CPU Brownian-rate reference should accumulate "
        "more collisions across the fixed seed set, without requiring exact "
        "per-seed counts."
    )


def test_coagulation_step_gpu_nonuniform_arrays_keep_single_active_box_idle(
    device: str,
) -> None:
    """A box with fewer than two active particles still records no collisions."""
    particles = _make_particle_data(n_boxes=2, n_particles=4, n_species=1)
    particles.concentration[1, 1:] = 0.0
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=wp.array([298.15, 308.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 98000.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.5,
        rng_seed=29,
        max_collisions=8,
    )
    wp.synchronize()

    result = np.asarray(collision_counts.numpy())

    assert result.shape == (2,)
    assert result[1] == 0


def test_coagulation_mass_conservation(device: str) -> None:
    """Coagulation conserves total mass in each box."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)

    initial_mass = np.sum(particles.masses)
    coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=7,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    final_mass = np.sum(result.masses)
    npt.assert_allclose(final_mass, initial_mass, rtol=1.0e-12)


def test_coagulation_step_gpu_reuses_preallocated_buffers(
    device: str,
) -> None:
    """Preallocated coagulation buffers are reused without reallocation."""
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    volume = wp.array(np.array([1.0e-6]), dtype=wp.float64, device=device)
    collision_pairs = wp.zeros((1, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    _, returned_pairs, returned_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        volume=volume,
        max_collisions=4,
        rng_seed=11,
        collision_pairs=collision_pairs,
        n_collisions=n_collisions,
        rng_states=rng_states,
    )
    wp.synchronize()

    assert returned_pairs is collision_pairs
    assert returned_counts is n_collisions
    assert np.all(np.asarray(n_collisions.numpy()) >= 0)


def test_coagulation_step_gpu_persisted_rng_states_advance_across_repeated_valid_calls(
    device: str,
) -> None:
    """Repeated valid calls advance persisted caller-owned RNG state.

    Reusing the same ``rng_seed`` with the same caller-owned ``rng_states``
    buffer must not restore the original seed-derived state between valid
    calls.
    """
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    rng_seed = 37
    repeated_rng_states = rng_states

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(rng_seed), rng_states],
        device=device,
    )
    wp.synchronize()
    initial_state = np.asarray(rng_states.numpy()).copy()

    assert repeated_rng_states is rng_states

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=rng_seed,
        max_collisions=8,
        rng_states=repeated_rng_states,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=rng_seed,
        max_collisions=8,
        rng_states=repeated_rng_states,
    )
    wp.synchronize()
    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initial_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)
    assert not np.array_equal(state_after_second_call, initial_state)


def test_coagulation_step_gpu_multibox_persisted_rng_states_advance(
    device: str,
) -> None:
    """Repeated valid calls advance persisted RNG state for every box."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    rng_states = wp.zeros((2,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(37), rng_states],
        device=device,
    )
    wp.synchronize()
    initial_state = np.asarray(rng_states.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initial_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)
    assert not np.array_equal(state_after_second_call, initial_state)


def test_coagulation_step_gpu_initialize_rng_false_reuses_caller_owned_state(
    device: str,
) -> None:
    """Default ``initialize_rng=False`` reuses caller-owned RNG state."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(37), rng_states],
        device=device,
    )
    wp.synchronize()
    initialized_state = np.asarray(rng_states.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
        initialize_rng=False,
    )
    wp.synchronize()
    state_after_first_call = np.asarray(rng_states.numpy()).copy()

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
        rng_states=rng_states,
        initialize_rng=False,
    )
    wp.synchronize()

    state_after_second_call = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(state_after_first_call, initialized_state)
    assert not np.array_equal(state_after_second_call, initialized_state)
    assert not np.array_equal(state_after_second_call, state_after_first_call)


def test_coagulation_step_gpu_initialize_rng_true_resets_caller_owned_state(
    device: str,
) -> None:
    """Explicit ``True`` still forces reset from the provided seed."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    rng_states_a = wp.zeros((1,), dtype=wp.uint32, device=device)
    rng_states_b = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(5), rng_states_a],
        device=device,
    )
    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(9), rng_states_b],
        device=device,
    )
    wp.synchronize()
    pre_reset_state = np.asarray(rng_states_a.numpy()).copy()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_a,
        initialize_rng=True,
    )

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_b,
        initialize_rng=True,
    )
    wp.synchronize()

    post_reset_state_a = np.asarray(rng_states_a.numpy()).copy()
    post_reset_state_b = np.asarray(rng_states_b.numpy()).copy()

    assert not np.array_equal(post_reset_state_a, pre_reset_state)
    npt.assert_array_equal(post_reset_state_a, post_reset_state_b)


def test_coagulation_step_gpu_multibox_initialize_rng_true_resets_state(
    device: str,
) -> None:
    """Explicit multibox reset reproduces the same seed-derived state."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)
    rng_states_a = wp.zeros((2,), dtype=wp.uint32, device=device)
    rng_states_b = wp.zeros((2,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(5), rng_states_a],
        device=device,
    )
    wp.launch(
        _initialize_rng_states,
        dim=2,
        inputs=[wp.uint32(9), rng_states_b],
        device=device,
    )
    wp.synchronize()

    first_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_a,
        initialize_rng=True,
    )

    second_particles = to_warp_particle_data(particles, device=device)
    coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states_b,
        initialize_rng=True,
    )
    wp.synchronize()

    post_reset_state_a = np.asarray(rng_states_a.numpy()).copy()
    post_reset_state_b = np.asarray(rng_states_b.numpy()).copy()
    npt.assert_array_equal(post_reset_state_a, post_reset_state_b)


def test_coagulation_step_gpu_omitted_rng_states_repeat_seed_replays_results(
    device: str,
) -> None:
    """Omitted RNG state allocation is reseeded independently per call."""
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)

    first_particles = to_warp_particle_data(particles, device=device)
    _, first_pairs, first_counts = coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    wp.synchronize()

    second_particles = to_warp_particle_data(particles, device=device)
    _, second_pairs, second_counts = coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(first_pairs.numpy()),
        second_pairs.numpy(),
    )
    npt.assert_array_equal(
        np.asarray(first_counts.numpy()),
        second_counts.numpy(),
    )


def test_coagulation_step_gpu_multibox_omitted_rng_states_replay_results(
    device: str,
) -> None:
    """Omitted multibox RNG allocation still replays repeated seeded calls."""
    particles = _make_particle_data(n_boxes=2, n_particles=6, n_species=1)

    first_particles = to_warp_particle_data(particles, device=device)
    _, first_pairs, first_counts = coagulation_step_gpu(
        first_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    first_result = from_warp_particle_data(first_particles, sync=True)
    wp.synchronize()

    second_particles = to_warp_particle_data(particles, device=device)
    _, second_pairs, second_counts = coagulation_step_gpu(
        second_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=37,
        max_collisions=8,
    )
    second_result = from_warp_particle_data(second_particles, sync=True)
    wp.synchronize()

    npt.assert_array_equal(
        np.asarray(first_pairs.numpy()),
        second_pairs.numpy(),
    )
    npt.assert_array_equal(
        np.asarray(first_counts.numpy()),
        second_counts.numpy(),
    )
    npt.assert_allclose(first_result.masses, second_result.masses)
    npt.assert_allclose(
        first_result.concentration,
        second_result.concentration,
    )


def test_coagulation_marks_inactive_particles(device: str) -> None:
    """Merged particles are marked inactive and mass is transferred."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        volume=1.0e-18,
        rng_seed=3,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    assert np.asarray(collision_counts.numpy()).sum() > 0
    assert np.any(result.concentration == 0.0)
    assert np.max(result.masses) >= np.max(particles.masses)


def test_brownian_coagulation_kernel_inactive_particles(
    device: str,
) -> None:
    """Brownian kernel returns no collisions when particles are inactive."""
    n_boxes = 1
    n_particles = 2
    n_species = 1
    masses = wp.array(
        np.full((n_boxes, n_particles, n_species), 1.0e-18, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    concentration = wp.zeros(
        (n_boxes, n_particles),
        dtype=wp.float64,
        device=device,
    )
    density = wp.array(
        np.array([1000.0], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    volume = wp.array(
        np.array([1.0e-6], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    temperature = wp.array(
        np.array([298.15], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    pressure = wp.array(
        np.array([101325.0], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    active_flags = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    collision_pairs = wp.zeros((n_boxes, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)

    wp.launch(
        brownian_coagulation_kernel,
        dim=(n_boxes,),
        inputs=[
            masses,
            concentration,
            density,
            volume,
            temperature,
            pressure,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(1.0),
            radii,
            diffusivities,
            g_terms,
            speeds,
            active_flags,
            collision_pairs,
            n_collisions,
            rng_states,
        ],
        device=device,
    )
    wp.synchronize()

    assert np.asarray(n_collisions.numpy()).item() == 0
    npt.assert_allclose(np.asarray(radii.numpy()), 0.0)
    npt.assert_allclose(np.asarray(diffusivities.numpy()), 0.0)
    npt.assert_allclose(np.asarray(g_terms.numpy()), 0.0)
    npt.assert_allclose(np.asarray(speeds.numpy()), 0.0)


def test_apply_coagulation_kernel_merges_particles(device: str) -> None:
    """Apply kernel merges masses and zeroes merged particle concentration."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses[0, 0, 0], 3.0e-18)
    npt.assert_allclose(result_masses[0, 1, 0], 0.0)
    npt.assert_allclose(result_concentration[0, 1], 0.0)

    n_collisions_zero = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_zero,
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(np.asarray(masses_wp.numpy()), result_masses)
    npt.assert_allclose(
        np.asarray(concentration_wp.numpy()), result_concentration
    )


def test_apply_coagulation_kernel_skips_self_pair(device: str) -> None:
    """Apply kernel ignores self-collisions without mutating arrays."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 0]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses, masses)
    npt.assert_allclose(result_concentration, concentration)


def test_apply_coagulation_kernel_skips_empty_pair(device: str) -> None:
    """Apply kernel ignores entries when collision index is out of range."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([0], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses, masses)
    npt.assert_allclose(result_concentration, concentration)


def test_kernels_init_exports() -> None:
    """Kernel module exports coagulation utilities."""
    from particula.gpu import kernels

    assert kernels.brownian_coagulation_kernel is not None
    assert kernels.apply_coagulation_kernel is not None
    assert kernels.coagulation_step_gpu is not None


def test_coagulation_validation_rejects_bad_shapes(device: str) -> None:
    """Validation helpers reject mismatched shapes."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)

    gpu_particles.masses = wp.zeros(
        (1, 2, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle masses shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle concentration shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.volume = wp.zeros(
        (2,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle volume shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.density = wp.zeros(
        (3,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle density length"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    collision_pairs = wp.zeros(
        (1, 3, 2),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="collision_pairs shape"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (2,),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="n_collisions shape"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (2,),
        dtype=wp.uint32,
        device=device,
    )
    with pytest.raises(ValueError, match="rng_states shape"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_step_gpu_rejects_rng_state_shape_before_mutation(
    device: str,
) -> None:
    """Wrong-shape caller RNG state fails before any mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.array([7, 11], dtype=wp.uint32, device=device)
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match="rng_states shape"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            rng_seed=23,
            rng_states=rng_states,
        )

    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)


@pytest.mark.parametrize(
    ("buffer_name", "message"),
    [
        ("collision_pairs", "collision_pairs buffer must use dtype int32"),
        ("n_collisions", "n_collisions buffer must use dtype int32"),
        ("rng_states", "rng_states buffer must use dtype uint32"),
    ],
)
def test_coagulation_step_gpu_rejects_wrong_buffer_dtypes_before_mutation(
    device: str,
    buffer_name: str,
    message: str,
) -> None:
    """Wrong preallocated buffer dtypes fail before any mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)

    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([3], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([17], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )

    if buffer_name == "collision_pairs":
        collision_pairs = wp.array(
            np.arange(8, dtype=np.float64).reshape(1, 4, 2),
            dtype=wp.float64,
            device=device,
        )
    elif buffer_name == "n_collisions":
        n_collisions = wp.array(
            np.array([3.0], dtype=np.float64),
            dtype=wp.float64,
            device=device,
        )
    else:
        rng_states = wp.array(
            np.array([17], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()

    with pytest.raises(ValueError, match=message):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


@pytest.mark.parametrize(
    "time_step",
    [-0.1, float("nan"), float("inf")],
)
def test_coagulation_step_gpu_invalid_time_step_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    time_step: float,
) -> None:
    """Invalid time steps fail before volume setup, RNG init, or mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="time_step must be finite and nonnegative",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=time_step,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_invalid_volume_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid volumes fail before RNG init, launch, or particle mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            volume=-1.0,
            max_collisions=4,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


@pytest.mark.parametrize(
    "max_collisions",
    [0, -1, 1.5, True, np.iinfo(np.int32).max + 1],
)
def test_coagulation_step_gpu_invalid_max_collisions_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    max_collisions: object,
) -> None:
    """Invalid collision limits fail before allocation, launch, or mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    initial_particles = from_warp_particle_data(gpu_particles, sync=True)
    collision_pairs = wp.array(
        np.arange(8, dtype=np.int32).reshape(1, 4, 2),
        dtype=wp.int32,
        device=device,
    )
    n_collisions = wp.array(
        np.array([2], dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    rng_states = wp.array(
        np.array([19], dtype=np.uint32),
        dtype=wp.uint32,
        device=device,
    )
    initial_collision_pairs = np.asarray(collision_pairs.numpy()).copy()
    initial_n_collisions = np.asarray(n_collisions.numpy()).copy()
    initial_rng_states = np.asarray(rng_states.numpy()).copy()
    calls: list[str] = []

    def _unexpected_ensure_volume_array(*args: Any, **kwargs: Any) -> Any:
        calls.append("ensure_volume_array")
        raise AssertionError("_ensure_volume_array should not be called")

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(
        coagulation_module,
        "_ensure_volume_array",
        _unexpected_ensure_volume_array,
    )
    monkeypatch.setattr(coagulation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="max_collisions must be a positive integer",
    ):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            max_collisions=max_collisions,
            collision_pairs=collision_pairs,
            n_collisions=n_collisions,
            rng_states=rng_states,
        )

    assert calls == []
    npt.assert_array_equal(
        np.asarray(collision_pairs.numpy()),
        initial_collision_pairs,
    )
    npt.assert_array_equal(
        np.asarray(n_collisions.numpy()),
        initial_n_collisions,
    )
    npt.assert_array_equal(np.asarray(rng_states.numpy()), initial_rng_states)
    _assert_particles_unchanged(gpu_particles, initial_particles)


def test_coagulation_step_gpu_invalid_followup_preserves_advanced_rng_states(
    device: str,
) -> None:
    """Invalid follow-up calls preserve already-advanced caller RNG state.

    After one valid call advances a caller-owned ``rng_states`` buffer, a
    later invalid call must fail before mutating that persisted buffer.
    """
    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    first_gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)

    wp.launch(
        _initialize_rng_states,
        dim=1,
        inputs=[wp.uint32(41), rng_states],
        device=device,
    )
    wp.synchronize()
    initialized_state = np.asarray(rng_states.numpy()).copy()

    coagulation_step_gpu(
        first_gpu_particles,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        volume=1.0e-18,
        rng_seed=41,
        max_collisions=8,
        rng_states=rng_states,
    )
    wp.synchronize()
    advanced_state = np.asarray(rng_states.numpy()).copy()

    assert not np.array_equal(advanced_state, initialized_state)

    second_gpu_particles = to_warp_particle_data(particles, device=device)
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        coagulation_step_gpu(
            second_gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            volume=-1.0,
            rng_seed=41,
            max_collisions=8,
            rng_states=rng_states,
        )

    npt.assert_array_equal(np.asarray(rng_states.numpy()), advanced_state)


def test_coagulation_validation_rejects_device_mismatch(device: str) -> None:
    """Validation helpers reject device mismatches."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    collision_pairs = wp.zeros(
        (1, 2, 2),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="collision_pairs buffer"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (1,),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="n_collisions"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (1,),
        dtype=wp.uint32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="rng_states"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_step_gpu_rejects_rng_state_device_mismatch(
    device: str,
) -> None:
    """Wrong-device caller RNG state fails at the public entrypoint."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=3, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=wrong_device)

    with pytest.raises(ValueError, match="rng_states"):
        coagulation_step_gpu(
            gpu_particles,
            temperature=298.15,
            pressure=101325.0,
            time_step=0.1,
            rng_seed=23,
            rng_states=rng_states,
        )


def test_coagulation_validate_device_arrays(device: str) -> None:
    """Device validation passes when devices match and fails otherwise."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _validate_device_arrays(gpu_particles, gpu_particles.masses.device)

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.volume = wp.zeros((1,), dtype=wp.float64, device=wrong_device)
    with pytest.raises(ValueError, match="particle volume device mismatch"):
        _validate_device_arrays(gpu_particles, gpu_particles.masses.device)


def test_coagulation_ensure_volume_array(device: str) -> None:
    """Volume helper returns a device array and validates shapes."""
    volume_array = _ensure_volume_array(1.0e-6, n_boxes=2, device=device)
    assert volume_array.shape == (2,)
    npt.assert_allclose(np.asarray(volume_array.numpy()), [1.0e-6, 1.0e-6])

    bad_volume = wp.zeros((3,), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="volume shape does not match"):
        _ensure_volume_array(bad_volume, n_boxes=2, device=device)

    volume_array = _ensure_volume_array(1.5e-6, n_boxes=2, device=device)
    _validate_device_match("volume", volume_array, volume_array.device)


@pytest.mark.parametrize(
    "volume",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_coagulation_ensure_volume_array_rejects_invalid_scalars(
    device: str,
    volume: float,
) -> None:
    """Scalar volume inputs must be positive finite values."""
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        _ensure_volume_array(volume, n_boxes=1, device=device)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0], dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([np.inf], dtype=np.float64),
    ],
)
def test_coagulation_ensure_volume_array_rejects_invalid_arrays(
    device: str,
    values: np.ndarray,
) -> None:
    """Warp-array volume inputs must be positive finite values."""
    volume = wp.array(values, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="volume must be finite and > 0"):
        _ensure_volume_array(volume, n_boxes=1, device=volume.device)


def test_coagulation_ensure_volume_array_rejects_non_warp_tensor_like() -> None:
    """Tensor-like non-Warp volume inputs fail with a stable type error."""

    class _FakeTensorLike:
        def __init__(self) -> None:
            self.shape = (1,)

    with pytest.raises(
        ValueError,
        match=r"volume must be a Warp array with shape \(n_boxes,\)",
    ):
        _ensure_volume_array(_FakeTensorLike(), n_boxes=1, device="cpu")


def test_coagulation_ensure_volume_array_rejects_integer_scalar(
    device: str,
) -> None:
    """Integer scalar volumes are rejected at the GPU boundary."""
    with pytest.raises(ValueError, match="floating scalar"):
        _ensure_volume_array(1, n_boxes=1, device=device)


def test_coagulation_ensure_volume_array_rejects_integer_dtype_array(
    device: str,
) -> None:
    """Only supported Warp float dtypes are accepted for volume arrays."""
    volume = wp.array([1], dtype=wp.int32, device=device)

    with pytest.raises(ValueError, match="supported Warp float dtype"):
        _ensure_volume_array(volume, n_boxes=1, device=volume.device)


def test_coagulation_ensure_volume_array_skips_cuda_host_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA volume validation avoids implicit ``.numpy()`` synchronization."""
    if not cuda_available(wp):
        pytest.skip("CUDA not available for readback guard test")

    volume = wp.array([1.0e-6], dtype=wp.float64, device="cuda")

    def _forbidden_numpy(self: Any) -> np.ndarray:
        raise AssertionError("unexpected host readback")

    monkeypatch.setattr(volume, "numpy", _forbidden_numpy, raising=False)

    returned_volume = _ensure_volume_array(
        volume,
        n_boxes=1,
        device=volume.device,
    )

    assert returned_volume is volume


def test_coagulation_validation_helpers_accept_valid_inputs(
    device: str,
) -> None:
    """Validation helpers accept matching buffers and volume arrays."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    collision_pairs = wp.zeros((1, 2, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((1,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((1,), dtype=wp.uint32, device=device)
    volume = wp.array(np.array([1.0e-6]), dtype=wp.float64, device=device)

    _validate_particle_arrays(gpu_particles, 1, 2, 1)
    _validate_collision_pairs(
        collision_pairs,
        (1, 2, 2),
        gpu_particles.masses.device,
    )
    _validate_collision_counts(
        n_collisions,
        (1,),
        gpu_particles.masses.device,
    )
    _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)

    returned_volume = _ensure_volume_array(
        volume,
        n_boxes=1,
        device=volume.device,
    )
    assert returned_volume is volume


def test_validate_time_step_accepts_zero_and_float_like_values() -> None:
    """Time-step helper returns normalized finite nonnegative values."""
    assert _validate_time_step(0.0) == 0.0
    assert _validate_time_step(np.float64(0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "time_step",
    [True, -1.0, float("nan"), float("inf"), object()],
)
def test_validate_time_step_rejects_invalid_values(time_step: object) -> None:
    """Time-step helper rejects invalid domains and non-real inputs."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _validate_time_step(time_step)


def test_validate_max_collisions_accepts_positive_int_like_values() -> None:
    """Collision-limit helper returns normalized positive integer values."""
    assert _validate_max_collisions(1) == 1
    assert _validate_max_collisions(np.int64(7)) == 7


@pytest.mark.parametrize(
    "max_collisions",
    [True, 0, -1, 1.5, np.iinfo(np.int32).max + 1, object()],
)
def test_validate_max_collisions_rejects_invalid_values(
    max_collisions: object,
) -> None:
    """Collision-limit helper rejects unsupported values before allocation."""
    with pytest.raises(ValueError, match="positive integer"):
        _validate_max_collisions(max_collisions)


def test_initialize_rng_states_changes_output(device: str) -> None:
    """RNG state initialization writes nonzero data."""
    rng_states = wp.zeros((4,), dtype=wp.uint32, device=device)
    wp.launch(
        _initialize_rng_states,
        dim=4,
        inputs=[wp.uint32(123), rng_states],
        device=device,
    )
    wp.synchronize()

    rng_values = np.asarray(rng_states.numpy())
    assert np.any(rng_values != 0)
