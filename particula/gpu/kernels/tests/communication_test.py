"""Tests for the concrete direct-Warp volume-evolution primitive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


def _warp():
    """Import Warp at runtime so collection remains optional."""
    return pytest.importorskip("warp")


def _containers(
    volumes: np.ndarray,
    particle_concentration: np.ndarray,
    gas_concentration: np.ndarray,
    device: str = "cpu",
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build complete, nonaliasing fixed-shape Warp test containers."""
    wp = _warp()
    boxes, particles_count = particle_concentration.shape
    species = 2
    gas_species = gas_concentration.shape[1]
    particles = SimpleNamespace(
        masses=wp.array(
            np.full((boxes, particles_count, species), 2.0, dtype=np.float64),
            dtype=wp.float64,
            device=device,
        ),
        concentration=wp.array(
            particle_concentration, dtype=wp.float64, device=device
        ),
        charge=wp.array(
            np.arange(boxes * particles_count, dtype=np.float64).reshape(
                boxes, particles_count
            ),
            dtype=wp.float64,
            device=device,
        ),
        density=wp.full(species, 1000.0, dtype=wp.float64, device=device),
        volume=wp.array(volumes, dtype=wp.float64, device=device),
    )
    gas = SimpleNamespace(
        molar_mass=wp.full(gas_species, 0.1, dtype=wp.float64, device=device),
        concentration=wp.array(
            gas_concentration, dtype=wp.float64, device=device
        ),
        vapor_pressure=wp.full(
            (boxes, gas_species), 10.0, dtype=wp.float64, device=device
        ),
        partitioning=wp.ones(
            (boxes, gas_species), dtype=wp.int32, device=device
        ),
    )
    return particles, gas


@pytest.mark.parametrize(
    ("old_volumes", "final_volumes"),
    [
        (np.array([1.0]), np.array([2.0])),
        (np.array([2.0, 4.0]), np.array([1.0, 8.0])),
    ],
)
def test_volume_evolution_matches_independent_oracle(
    old_volumes: np.ndarray, final_volumes: np.ndarray
) -> None:
    """Scale both concentration ledgers by the NumPy volume ratio oracle."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particle = np.array([[0.0, 3.0], [5.0, 7.0]], dtype=np.float64)[
        : len(old_volumes)
    ]
    gas_values = np.array(
        [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float64
    )[: len(old_volumes)]
    particles, gas = _containers(old_volumes, particle, gas_values)
    final = wp.array(final_volumes, dtype=wp.float64, device="cpu")
    protected = (
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
    initial_masses = particles.masses.numpy().copy()
    initial_charge = particles.charge.numpy().copy()
    initial_particle = particle.copy()
    initial_gas = gas_values.copy()

    returned_particles, returned_gas = volume_evolution_step_gpu(
        particles, gas, final
    )

    factor = old_volumes / final_volumes
    assert returned_particles is particles
    assert returned_gas is gas
    assert particles.concentration is not None and gas.concentration is not None
    npt.assert_allclose(
        particles.concentration.numpy(),
        initial_particle * factor[:, None],
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        initial_gas * factor[:, None],
        rtol=1e-12,
        atol=0.0,
    )
    npt.assert_allclose(
        particles.volume.numpy(), final_volumes, rtol=0.0, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy() * particles.volume.numpy()[:, None],
        initial_particle * old_volumes[:, None],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        gas.concentration.numpy() * particles.volume.numpy()[:, None],
        initial_gas * old_volumes[:, None],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        (
            particles.concentration.numpy()[..., None]
            * initial_masses
            * particles.volume.numpy()[:, None, None]
        ),
        initial_particle[..., None]
        * initial_masses
        * old_volumes[:, None, None],
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        particles.concentration.numpy()
        * initial_charge
        * particles.volume.numpy()[:, None],
        initial_particle * initial_charge * old_volumes[:, None],
        rtol=1e-12,
        atol=1e-30,
    )
    for before, after in zip(
        protected,
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
        strict=True,
    ):
        assert after is before
    npt.assert_array_equal(final.numpy(), final_volumes)


def test_zero_boxes_are_a_write_free_no_op() -> None:
    """Accept canonical empty box schemas without launching an apply writer."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.empty(0, dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
    )
    final = wp.empty(0, dtype=wp.float64, device="cpu")
    fields = tuple(vars(particles).values()) + tuple(vars(gas).values())

    returned_particles, returned_gas = volume_evolution_step_gpu(
        particles, gas, final
    )

    assert returned_particles is particles
    assert returned_gas is gas
    assert tuple(vars(particles).values()) + tuple(vars(gas).values()) == fields
    assert final.shape == (0,)


def test_unchanged_volumes_are_write_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete preflight without launching an apply writer for equal volumes."""
    wp = _warp()
    from particula.gpu.kernels import communication

    particles, gas = _containers(
        np.array([2.0]), np.array([[3.0, 0.0]]), np.array([[4.0]])
    )
    final = wp.array([2.0], dtype=wp.float64, device="cpu")
    original_launch = communication.wp.launch

    def guarded_launch(kernel, *args, **kwargs):
        assert kernel not in (
            communication._apply_volume_evolution,
            communication._apply_scaled_concentration,
        )
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(communication.wp, "launch", guarded_launch)
    before = (
        particles.volume.numpy().copy(),
        particles.concentration.numpy().copy(),
        gas.concentration.numpy().copy(),
    )
    communication.volume_evolution_step_gpu(particles, gas, final)
    npt.assert_array_equal(particles.volume.numpy(), before[0])
    npt.assert_array_equal(particles.concentration.numpy(), before[1])
    npt.assert_array_equal(gas.concentration.numpy(), before[2])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_invalid_final_volumes_reject_without_mutation(bad: float) -> None:
    """Reject invalid final-volume domains before applying a writer."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    final = wp.array([bad], dtype=wp.float64, device="cpu")
    before = (
        particles.volume.numpy().copy(),
        particles.concentration.numpy().copy(),
        gas.concentration.numpy().copy(),
    )
    with pytest.raises(ValueError, match="finite positive"):
        volume_evolution_step_gpu(particles, gas, final)
    npt.assert_array_equal(particles.volume.numpy(), before[0])
    npt.assert_array_equal(particles.concentration.numpy(), before[1])
    npt.assert_array_equal(gas.concentration.numpy(), before[2])


def test_final_volume_alias_rejects_before_mutation() -> None:
    """Reject final-volume storage that aliases a mutable primary field."""
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    with pytest.raises(ValueError, match="alias"):
        volume_evolution_step_gpu(particles, gas, particles.volume)


@pytest.mark.parametrize(
    ("old_volume", "final_volume", "concentration", "message"),
    [
        (0.0, 1.0, 2.0, "finite positive"),
        (1.0, 1.0, -2.0, "finite and nonnegative"),
        (1.0, 1.0, np.nan, "finite and nonnegative"),
        (
            np.nextafter(0.0, 1.0),
            np.finfo(np.float64).max,
            2.0,
            "factor",
        ),
        (1.0, np.finfo(np.float64).max, np.nextafter(0.0, 1.0), "underflow"),
        (2.0, 1.0, np.finfo(np.float64).max, "underflow"),
    ],
)
def test_invalid_preflight_preserves_all_mutable_state(
    old_volume: float,
    final_volume: float,
    concentration: float,
    message: str,
) -> None:
    """Reject invalid domains and unsafe products before mutating containers."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([old_volume]),
        np.array([[concentration, 3.0]]),
        np.array([[4.0]]),
    )
    final = wp.array([final_volume], dtype=wp.float64, device="cpu")
    before = (
        particles.volume.numpy().copy(),
        particles.concentration.numpy().copy(),
        gas.concentration.numpy().copy(),
    )

    with pytest.raises(ValueError, match=message):
        volume_evolution_step_gpu(particles, gas, final)

    npt.assert_array_equal(particles.volume.numpy(), before[0])
    npt.assert_array_equal(particles.concentration.numpy(), before[1])
    npt.assert_array_equal(gas.concentration.numpy(), before[2])


def test_schema_and_missing_field_fail_before_mutation() -> None:
    """Reject malformed final input and incomplete containers deterministically."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    with pytest.raises(ValueError, match="rank"):
        volume_evolution_step_gpu(
            particles,
            gas,
            wp.array([[1.0]], dtype=wp.float64, device="cpu"),
        )
    with pytest.raises(ValueError, match="dtype"):
        volume_evolution_step_gpu(
            particles,
            gas,
            wp.array([1.0], dtype=wp.float32, device="cpu"),
        )
    with pytest.raises(ValueError, match="shape"):
        volume_evolution_step_gpu(
            particles,
            gas,
            wp.array([1.0, 2.0], dtype=wp.float64, device="cpu"),
        )
    with pytest.raises(ValueError, match="particles.masses"):
        volume_evolution_step_gpu(
            SimpleNamespace(),
            gas,
            wp.array([1.0], dtype=wp.float64, device="cpu"),
        )


def test_zero_width_particle_and_gas_concentrations_update_volume() -> None:
    """Update volume safely when both mutable concentration dimensions are zero."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([2.0]),
        np.empty((1, 0), dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
    )
    final = wp.array([4.0], dtype=wp.float64, device="cpu")

    returned_particles, returned_gas = volume_evolution_step_gpu(
        particles, gas, final
    )

    assert returned_particles is particles
    assert returned_gas is gas
    assert particles.concentration.shape == (1, 0)
    assert gas.concentration.shape == (1, 0)
    npt.assert_array_equal(particles.volume.numpy(), np.array([4.0]))


def test_primary_alias_rejects_before_mutation() -> None:
    """Reject primary fields that share storage even when their shapes agree."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0, 5.0]])
    )
    gas.concentration = particles.concentration
    final = wp.array([2.0], dtype=wp.float64, device="cpu")
    before = particles.concentration.numpy().copy()

    with pytest.raises(ValueError, match="alias"):
        volume_evolution_step_gpu(particles, gas, final)

    npt.assert_array_equal(particles.concentration.numpy(), before)


@pytest.mark.cuda
def test_cuda_volume_evolution_when_available() -> None:
    """Exercise the same direct mutation contract on optional CUDA hardware."""
    wp = _warp()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([2.0]),
        np.array([[3.0, 0.0]]),
        np.array([[4.0, 8.0]]),
        device="cuda:0",
    )
    final = wp.array([4.0], dtype=wp.float64, device="cuda:0")

    volume_evolution_step_gpu(particles, gas, final)

    npt.assert_allclose(
        particles.concentration.numpy(), np.array([[1.5, 0.0]]), rtol=1e-12
    )
    npt.assert_allclose(
        gas.concentration.numpy(), np.array([[2.0, 4.0]]), rtol=1e-12
    )


def test_private_schema_helpers_reject_non_warp_and_invalid_storage() -> None:
    """Validate private schema helpers' malformed-array and backing branches."""
    wp = _warp()
    from particula.gpu.kernels import communication

    with pytest.raises(ValueError, match="Warp array"):
        communication._validate_array(object(), "value", wp.float64, 1)

    malformed = SimpleNamespace(
        dtype=wp.float64,
        shape=(1,),
        strides=(4,),
        ptr=1,
        capacity=8,
    )
    with pytest.raises(ValueError, match="contiguous"):
        communication._array_range(malformed, "value")

    malformed.strides = (8,)
    malformed.ptr = 0
    with pytest.raises(ValueError, match="valid pointer"):
        communication._array_range(malformed, "value")

    malformed.ptr = 1
    malformed.capacity = 0
    with pytest.raises(ValueError, match="capacity"):
        communication._array_range(malformed, "value")
