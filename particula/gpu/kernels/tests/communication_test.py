"""Tests for the concrete direct-Warp volume-evolution primitive."""

from __future__ import annotations

import inspect
import sys
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
    """Keep equal-volume calls write-free while preserving state."""
    wp = _warp()
    from particula.gpu.kernels import communication

    particles, gas = _containers(
        np.array([2.0]), np.array([[3.0, 0.0]]), np.array([[4.0]])
    )
    final = wp.array([2.0], dtype=wp.float64, device="cpu")
    launches = []
    original_launch = communication.wp.launch

    def record_launch(kernel, *args, **kwargs):
        launches.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(communication.wp, "launch", record_launch)
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
def test_invalid_final_volumes_gate_writers_without_mutation(
    bad: float,
) -> None:
    """Invalid device domains suppress writers without a host status readback."""
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
    ("old_volume", "final_volume", "concentration"),
    [
        (0.0, 1.0, 2.0),
        (1.0, 1.0, -2.0),
        (1.0, 1.0, np.nan),
        (
            np.nextafter(0.0, 1.0),
            np.finfo(np.float64).max,
            2.0,
        ),
        (1.0, np.finfo(np.float64).max, np.nextafter(0.0, 1.0)),
        (2.0, 1.0, np.finfo(np.float64).max),
    ],
)
def test_invalid_preflight_preserves_all_mutable_state(
    old_volume: float,
    final_volume: float,
    concentration: float,
) -> None:
    """Invalid device values leave mutable containers unchanged."""
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

    volume_evolution_step_gpu(particles, gas, final)

    npt.assert_array_equal(particles.volume.numpy(), before[0])
    npt.assert_array_equal(particles.concentration.numpy(), before[1])
    npt.assert_array_equal(gas.concentration.numpy(), before[2])


def test_factor_first_scaling_accepts_representable_result() -> None:
    """Avoid an overflowing concentration-times-volume intermediate."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    largest = np.finfo(np.float64).max
    particles, gas = _containers(
        np.array([largest]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    final = wp.array([largest], dtype=wp.float64, device="cpu")

    volume_evolution_step_gpu(particles, gas, final)

    npt.assert_array_equal(particles.concentration.numpy(), [[2.0, 3.0]])
    npt.assert_array_equal(gas.concentration.numpy(), [[4.0]])


def test_direct_path_has_no_host_status_readback_or_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep validation and writer gating resident on the active device."""
    wp = _warp()
    from particula.gpu.kernels import communication

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    final = wp.array([2.0], dtype=wp.float64, device="cpu")

    def reject_synchronization(*_args, **_kwargs):
        raise AssertionError("direct path must not synchronize")

    monkeypatch.setattr(communication.wp, "synchronize", reject_synchronization)
    communication.volume_evolution_step_gpu(particles, gas, final)

    source = inspect.getsource(communication.volume_evolution_step_gpu)
    helpers = inspect.getsource(communication._scan_1d) + inspect.getsource(
        communication._scan_2d
    )
    assert ".numpy(" not in source + helpers
    assert "synchronize(" not in source + helpers


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

    malformed.shape = (sys.maxsize, 2)
    malformed.strides = (16, 8)
    malformed.capacity = sys.maxsize
    with pytest.raises(ValueError, match="safe address range"):
        communication._array_range(malformed, "value")


def test_private_schema_helpers_cover_shape_device_and_empty_scan_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject schema mismatches and skip launches for empty one-dimensional data."""
    wp = _warp()
    from particula.gpu.kernels import communication

    values = wp.array([1.0], dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        communication._validate_array(
            values, "value", wp.float64, 1, shape=(2,)
        )
    with pytest.raises(ValueError, match="device"):
        communication._validate_array(
            values, "value", wp.float64, 1, device="different-device"
        )

    launches = []
    monkeypatch.setattr(
        communication.wp,
        "launch",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    communication._scan_1d(
        wp.empty(0, dtype=wp.float64, device="cpu"),
        communication._scan_positive_finite,
        wp.zeros(1, dtype=wp.int32, device="cpu"),
    )
    assert launches == []


def test_gas_box_count_mismatch_rejects_before_mutation() -> None:
    """Reject gas concentration storage with a different number of boxes."""
    wp = _warp()
    from particula.gpu.kernels.communication import volume_evolution_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.array([[2.0, 3.0]]), np.array([[4.0]])
    )
    gas.concentration = wp.array([[4.0], [5.0]], dtype=wp.float64, device="cpu")
    final = wp.array([2.0], dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="shape"):
        volume_evolution_step_gpu(particles, gas, final)


def _gas_configuration(
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    enabled: np.ndarray | None = None,
):
    """Build the exact P1 declaration used by direct gas-transport tests."""
    wp = _warp()
    from particula.execution.communication import (
        CommunicationConfiguration,
        CommunicationMap,
        CommunicationMapForm,
        CommunicationTransportMode,
        PrescribedVolumeUpdate,
    )

    return CommunicationConfiguration(
        communication_map=CommunicationMap(
            form=CommunicationMapForm.ARBITRARY_PAIRS,
            transport_mode=CommunicationTransportMode.GAS,
            edge_capacity=len(source),
            source_boxes=wp.array(source, dtype=wp.int32, device="cpu"),
            destination_boxes=wp.array(
                destination, dtype=wp.int32, device="cpu"
            ),
            enabled=wp.array(
                np.ones(len(source), dtype=np.int32)
                if enabled is None
                else enabled,
                dtype=wp.int32,
                device="cpu",
            ),
            rates=wp.array(rates, dtype=wp.float64, device="cpu"),
        ),
        prescribed_volume=PrescribedVolumeUpdate(None),
        resource_shapes=(),
    )


def test_gas_communication_uses_immutable_synchronous_amount_ledger() -> None:
    """Chain transport reads every proposal from pre-step extensive amounts."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([2.0, 4.0, 1.0]),
        np.ones((3, 1), dtype=np.float64),
        np.array([[10.0, 2.0], [3.0, 4.0], [1.0, 8.0]]),
    )
    configuration = _gas_configuration(
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([0.25, 0.5]),
    )
    amounts = wp.empty((3, 2), dtype=wp.float64, device="cpu")
    deltas = wp.empty((3, 2), dtype=wp.float64, device="cpu")
    outbound = wp.empty((3, 2), dtype=wp.float64, device="cpu")
    before = gas.concentration.numpy().copy()
    volume = particles.volume.numpy().copy()
    expected_amounts = before * volume[:, None]
    expected_deltas = np.array(
        [
            -0.25 * expected_amounts[0],
            0.25 * expected_amounts[0] - 0.5 * expected_amounts[1],
            0.5 * expected_amounts[1],
        ]
    )

    returned = gas_communication_step_gpu(
        particles, gas, configuration, 1.0, amounts, deltas, outbound
    )

    assert returned == (particles, gas)
    npt.assert_allclose(amounts.numpy(), expected_amounts, rtol=1e-12)
    npt.assert_allclose(deltas.numpy(), expected_deltas, rtol=1e-12)
    npt.assert_allclose(
        outbound.numpy(),
        np.array(
            [0.25 * expected_amounts[0], 0.5 * expected_amounts[1], [0.0, 0.0]]
        ),
        rtol=1e-12,
    )
    npt.assert_allclose(
        gas.concentration.numpy(),
        (expected_amounts + expected_deltas) / volume[:, None],
        rtol=1e-12,
    )
    npt.assert_allclose(
        (gas.concentration.numpy() * volume[:, None]).sum(axis=0),
        expected_amounts.sum(axis=0),
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.parametrize(
    "order",
    [np.array([0, 1, 2]), np.array([2, 0, 1])],
)
def test_gas_communication_fan_in_is_edge_order_independent(
    order: np.ndarray,
) -> None:
    """Aggregate reciprocal and fan-in transfers from the initial ledger."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    source = np.array([0, 1, 2], dtype=np.int32)[order]
    destination = np.array([2, 2, 0], dtype=np.int32)[order]
    rates = np.array([0.1, 0.2, 0.25], dtype=np.float64)[order]
    particles, gas = _containers(
        np.array([2.0, 1.0, 4.0]),
        np.ones((3, 1), dtype=np.float64),
        np.array([[3.0, 2.0], [5.0, 1.0], [2.0, 4.0]]),
    )
    work = tuple(
        wp.empty((3, 2), dtype=wp.float64, device="cpu") for _ in range(3)
    )
    initial_amounts = (
        gas.concentration.numpy() * particles.volume.numpy()[:, None]
    )
    expected_deltas = np.zeros_like(initial_amounts)
    expected_outbound = np.zeros_like(initial_amounts)
    for left, right, rate in zip(source, destination, rates, strict=True):
        transfer = initial_amounts[left] * rate
        expected_deltas[left] -= transfer
        expected_deltas[right] += transfer
        expected_outbound[left] += transfer

    gas_communication_step_gpu(
        particles,
        gas,
        _gas_configuration(source, destination, rates),
        1.0,
        *work,
    )

    npt.assert_allclose(work[0].numpy(), initial_amounts, rtol=1e-12)
    npt.assert_allclose(work[1].numpy(), expected_deltas, rtol=1e-12)
    npt.assert_allclose(work[2].numpy(), expected_outbound, rtol=1e-12)
    npt.assert_allclose(
        gas.concentration.numpy(),
        (initial_amounts + expected_deltas) / particles.volume.numpy()[:, None],
        rtol=1e-12,
    )


def test_gas_communication_accounts_for_open_boundary_ledgers() -> None:
    """Record positive source and sink amounts without changing particle state."""
    wp = _warp()
    from particula.gpu.kernels.communication import (
        GasCommunicationBuffers,
        gas_communication_step_gpu,
    )

    particles, gas = _containers(
        np.array([2.0, 1.0]), np.ones((2, 1)), np.array([[4.0], [3.0]])
    )
    configuration = _gas_configuration(
        np.array([-1, 0], dtype=np.int32),
        np.array([1, -1], dtype=np.int32),
        np.array([0.5, 0.25]),
    )
    buffers = GasCommunicationBuffers(
        *(wp.empty((2, 1), dtype=wp.float64, device="cpu") for _ in range(5))
    )
    particle_before = particles.concentration.numpy().copy()

    gas_communication_step_gpu(particles, gas, configuration, 1.0, buffers)

    npt.assert_allclose(buffers.source_amounts.numpy(), [[0.0], [1.5]])
    npt.assert_allclose(buffers.sink_amounts.numpy(), [[2.0], [0.0]])
    npt.assert_allclose(gas.concentration.numpy(), [[3.0], [4.5]])
    npt.assert_array_equal(particles.concentration.numpy(), particle_before)


def test_gas_communication_zero_time_preserves_all_work_storage() -> None:
    """Keep zero-time gas communication write-free, including ledgers."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([2.0, 1.0]), np.ones((2, 1)), np.array([[4.0], [3.0]])
    )
    configuration = _gas_configuration(
        np.array([-1, 0], dtype=np.int32),
        np.array([1, -1], dtype=np.int32),
        np.array([0.5, 0.25]),
    )
    work = tuple(
        wp.array(np.full((2, 1), value), dtype=wp.float64, device="cpu")
        for value in range(1, 6)
    )
    before = tuple(values.numpy().copy() for values in work)
    gas_before = gas.concentration.numpy().copy()

    returned = gas_communication_step_gpu(
        particles, gas, configuration, 0.0, *work[:3], *work[3:]
    )

    assert returned == (particles, gas)
    npt.assert_array_equal(gas.concentration.numpy(), gas_before)
    for values, expected in zip(work, before, strict=True):
        npt.assert_array_equal(values.numpy(), expected)


def test_gas_communication_all_disabled_map_preserves_work_storage() -> None:
    """Treat a padded all-disabled map as an exact write-free no-op."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0, 2.0]), np.ones((2, 1)), np.array([[4.0], [3.0]])
    )
    configuration = _gas_configuration(
        np.array([0, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0.5, 0.25]),
        enabled=np.array([0, 0], dtype=np.int32),
    )
    work = tuple(
        wp.array(np.full((2, 1), index), dtype=wp.float64, device="cpu")
        for index in range(1, 4)
    )
    before_work = tuple(values.numpy().copy() for values in work)
    before_gas = gas.concentration.numpy().copy()

    returned = gas_communication_step_gpu(
        particles, gas, configuration, 1.0, *work
    )

    assert returned[0] is particles
    assert returned[1] is gas
    npt.assert_array_equal(gas.concentration.numpy(), before_gas)
    for values, expected in zip(work, before_work, strict=True):
        npt.assert_array_equal(values.numpy(), expected)


def test_gas_communication_invalid_open_map_gates_primary_commit() -> None:
    """Reject an open sink without its required accounting ledger on device."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[4.0]])
    )
    configuration = _gas_configuration(
        np.array([0], dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([0.5]),
    )
    work = tuple(
        wp.empty((1, 1), dtype=wp.float64, device="cpu") for _ in range(3)
    )
    gas_before = gas.concentration.numpy().copy()

    gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)

    npt.assert_array_equal(gas.concentration.numpy(), gas_before)


def test_gas_communication_aggregate_overdraw_gates_primary_commit() -> None:
    """Reject combined source debits larger than the staged source amount."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([2.0, 1.0]), np.ones((2, 1)), np.array([[4.0], [1.0]])
    )
    configuration = _gas_configuration(
        np.array([0, 0], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
        np.array([0.75, 0.75]),
    )
    work = tuple(
        wp.empty((2, 1), dtype=wp.float64, device="cpu") for _ in range(3)
    )
    gas_before = gas.concentration.numpy().copy()

    gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)

    npt.assert_array_equal(gas.concentration.numpy(), gas_before)


def test_gas_communication_invalid_prescribed_volume_gates_commit() -> None:
    """Validate read-only prescribed volumes before the primary writer."""
    wp = _warp()
    from particula.execution.communication import PrescribedVolumeUpdate
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[4.0]])
    )
    configuration = _gas_configuration(
        np.array([0], dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([0.5]),
    )
    configuration = type(configuration)(
        configuration.communication_map,
        PrescribedVolumeUpdate(
            wp.array([np.nan], dtype=wp.float64, device="cpu")
        ),
        configuration.resource_shapes,
    )
    work = tuple(
        wp.empty((1, 1), dtype=wp.float64, device="cpu") for _ in range(4)
    )
    gas_before = gas.concentration.numpy().copy()

    gas_communication_step_gpu(
        particles, gas, configuration, 1.0, *work[:3], sink_amounts=work[3]
    )

    npt.assert_array_equal(gas.concentration.numpy(), gas_before)


def test_buffer_carrier_rejects_individual_open_ledgers() -> None:
    """Reject ambiguous carrier and individual accounting-ledger arguments."""
    wp = _warp()
    from particula.gpu.kernels.communication import (
        GasCommunicationBuffers,
        gas_communication_step_gpu,
    )

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[4.0]])
    )
    configuration = _gas_configuration(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    buffers = GasCommunicationBuffers(
        *(wp.empty((1, 1), dtype=wp.float64, device="cpu") for _ in range(3))
    )
    source_amounts = wp.empty((1, 1), dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="carrier cannot be combined"):
        gas_communication_step_gpu(
            particles,
            gas,
            configuration,
            1.0,
            buffers,
            source_amounts=source_amounts,
        )


@pytest.mark.parametrize("time_step", [None, True, "1", -1.0, np.nan])
def test_gas_communication_rejects_invalid_time_before_container_access(
    time_step: object,
) -> None:
    """Validate the time scalar before reading an incomplete container."""
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    with pytest.raises((TypeError, ValueError), match="time_step"):
        gas_communication_step_gpu(
            SimpleNamespace(), SimpleNamespace(), object(), time_step, object()
        )


@pytest.mark.parametrize("no_op", ["zero_boxes", "zero_species", "empty_map"])
def test_gas_communication_schema_valid_no_ops_preserve_work_storage(
    no_op: str,
) -> None:
    """Preserve primary and work storage for each schema-valid no-op form."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    if no_op == "zero_boxes":
        volumes = np.empty(0, dtype=np.float64)
        particle = np.empty((0, 0), dtype=np.float64)
        gas_values = np.empty((0, 1), dtype=np.float64)
        source = destination = np.empty(0, dtype=np.int32)
        rates = np.empty(0, dtype=np.float64)
    elif no_op == "zero_species":
        volumes = np.array([1.0])
        particle = np.ones((1, 1), dtype=np.float64)
        gas_values = np.empty((1, 0), dtype=np.float64)
        source = destination = np.empty(0, dtype=np.int32)
        rates = np.empty(0, dtype=np.float64)
    else:
        volumes = np.array([1.0])
        particle = np.ones((1, 1), dtype=np.float64)
        gas_values = np.array([[2.0]])
        source = destination = np.empty(0, dtype=np.int32)
        rates = np.empty(0, dtype=np.float64)
    particles, gas = _containers(volumes, particle, gas_values)
    configuration = _gas_configuration(source, destination, rates)
    work = tuple(
        wp.array(
            np.full(gas_values.shape, index, dtype=np.float64),
            dtype=wp.float64,
            device="cpu",
        )
        for index in range(1, 4)
    )
    before_work = tuple(values.numpy().copy() for values in work)
    before_gas = gas.concentration.numpy().copy()

    returned = gas_communication_step_gpu(
        particles, gas, configuration, 1.0, *work
    )

    assert returned == (particles, gas)
    npt.assert_array_equal(gas.concentration.numpy(), before_gas)
    for values, expected in zip(work, before_work, strict=True):
        npt.assert_array_equal(values.numpy(), expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("configuration", object()), "exact CommunicationConfiguration"),
        (("communication_map", object()), "exact CommunicationMap"),
        (("prescribed_volume", object()), "exact PrescribedVolumeUpdate"),
        (("form", object()), "map form"),
        (("transport_mode", object()), "transport_mode"),
        (("transport_mode", "particles"), "transport_mode"),
        (("edge_capacity", True), "edge_capacity"),
        (("resource_shapes", []), "resource_shapes"),
    ],
)
def test_gas_communication_rejects_mutated_declaration_metadata(
    mutation: tuple[str, object], message: str
) -> None:
    """Fail closed for exact P1 declaration metadata before primary mutation."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[2.0]])
    )
    configuration = _gas_configuration(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    target, value = mutation
    if target == "configuration":
        configuration = value
    elif target in {
        "communication_map",
        "prescribed_volume",
        "resource_shapes",
    }:
        object.__setattr__(configuration, target, value)
    else:
        object.__setattr__(configuration.communication_map, target, value)
    work = tuple(
        wp.empty((1, 1), dtype=wp.float64, device="cpu") for _ in range(3)
    )
    before = gas.concentration.numpy().copy()

    with pytest.raises((TypeError, ValueError), match=message):
        gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)

    npt.assert_array_equal(gas.concentration.numpy(), before)


def test_gas_communication_validates_resource_shapes_and_required_work() -> (
    None
):
    """Reject duplicate resource roles and omitted required work ledgers."""
    wp = _warp()
    from particula.execution.communication import (
        CommunicationResourceShape,
        CommunicationShapeKind,
    )
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[2.0]])
    )
    configuration = _gas_configuration(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    shape = CommunicationResourceShape(
        "ledger", wp.float64, CommunicationShapeKind.BS
    )
    object.__setattr__(configuration, "resource_shapes", (shape, shape))
    amounts = wp.empty((1, 1), dtype=wp.float64, device="cpu")
    deltas = wp.empty((1, 1), dtype=wp.float64, device="cpu")
    outbound = wp.empty((1, 1), dtype=wp.float64, device="cpu")

    with pytest.raises(ValueError, match="roles must be unique"):
        gas_communication_step_gpu(
            particles, gas, configuration, 1.0, amounts, deltas, outbound
        )
    object.__setattr__(configuration, "resource_shapes", ())
    with pytest.raises(ValueError, match="amount_deltas"):
        gas_communication_step_gpu(particles, gas, configuration, 1.0, amounts)


def test_gas_communication_rejects_gas_box_mismatch_before_work_writes() -> (
    None
):
    """Reject incompatible gas box counts before modifying caller work arrays."""
    wp = _warp()
    from particula.gpu.kernels.communication import gas_communication_step_gpu

    particles, gas = _containers(
        np.array([1.0]), np.ones((1, 1)), np.array([[2.0]])
    )
    gas.concentration = wp.array([[2.0], [3.0]], dtype=wp.float64, device="cpu")
    configuration = _gas_configuration(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    work = tuple(
        wp.array([[float(index)]], dtype=wp.float64, device="cpu")
        for index in range(1, 4)
    )
    before = tuple(values.numpy().copy() for values in work)

    with pytest.raises(ValueError, match="gas.concentration shape"):
        gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)

    for values, expected in zip(work, before, strict=True):
        npt.assert_array_equal(values.numpy(), expected)
