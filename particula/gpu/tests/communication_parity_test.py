"""Independent multi-box parity evidence for direct communication kernels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.tests.cuda_availability import warp_devices

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]

RTOL = 1e-12
INVENTORY_ATOL = 1e-30


def _warp():
    """Import Warp only when a marked test is executed."""
    return pytest.importorskip("warp")


def _device_parameters() -> list[object]:
    """Return the CPU baseline and an optional CUDA test row."""
    wp = _warp()
    return [
        pytest.param(device, marks=pytest.mark.cuda)
        if device == "cuda"
        else device
        for device in warp_devices(wp)
    ]


def _containers(device: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build sparse three-box particle and gas storage."""
    wp = _warp()
    particles = SimpleNamespace(
        masses=wp.array(
            [
                [[2.0, 3.0], [0.0, 0.0], [5.0, 7.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [11.0, 13.0], [0.0, 0.0]],
                [[17.0, 19.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=wp.float64,
            device=device,
        ),
        concentration=wp.array(
            [[4.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
            dtype=wp.float64,
            device=device,
        ),
        charge=wp.array(
            [
                [-2.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [-4.0, 0.0, 0.0, 0.0],
            ],
            dtype=wp.float64,
            device=device,
        ),
        density=wp.array([1000.0, 1200.0], dtype=wp.float64, device=device),
        volume=wp.array([2.0, 1.0, 4.0], dtype=wp.float64, device=device),
    )
    gas = SimpleNamespace(
        molar_mass=wp.array([0.1, 0.2], dtype=wp.float64, device=device),
        concentration=wp.array(
            [[10.0, 2.0], [3.0, 4.0], [1.0, 8.0]],
            dtype=wp.float64,
            device=device,
        ),
        vapor_pressure=wp.zeros((3, 2), dtype=wp.float64, device=device),
        partitioning=wp.ones((3, 2), dtype=wp.int32, device=device),
    )
    return particles, gas


def _configuration(
    device: str,
    mode: object,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    *,
    form: object | None = None,
    enabled: np.ndarray | None = None,
):
    """Build a concrete arbitrary-pair communication configuration."""
    wp = _warp()
    from particula.execution.communication import (
        CommunicationConfiguration,
        CommunicationMap,
        CommunicationMapForm,
        PrescribedVolumeUpdate,
    )

    return CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ARBITRARY_PAIRS if form is None else form,
            mode,
            len(source),
            wp.array(source, dtype=wp.int32, device=device),
            wp.array(destination, dtype=wp.int32, device=device),
            wp.array(
                np.ones(len(source), dtype=np.int32)
                if enabled is None
                else enabled,
                dtype=wp.int32,
                device=device,
            ),
            wp.array(rates, dtype=wp.float64, device=device),
        ),
        PrescribedVolumeUpdate(None),
        (),
    )


def _gas_oracle(
    concentration: np.ndarray,
    volumes: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    time_step: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate immutable-ledger gas transport independently in NumPy."""
    amounts = concentration * volumes[:, None]
    deltas = np.zeros_like(amounts)
    outbound = np.zeros_like(amounts)
    source_ledger = np.zeros_like(amounts)
    sink_ledger = np.zeros_like(amounts)
    for left, right, rate in zip(source, destination, rates, strict=True):
        if left == -1:
            transfer = amounts[right] * rate * time_step
            deltas[right] += transfer
            source_ledger[right] += transfer
        elif right == -1:
            transfer = amounts[left] * rate * time_step
            deltas[left] -= transfer
            outbound[left] += transfer
            sink_ledger[left] += transfer
        else:
            transfer = amounts[left] * rate * time_step
            deltas[left] -= transfer
            deltas[right] += transfer
            outbound[left] += transfer
    return (
        (amounts + deltas) / volumes[:, None],
        amounts,
        deltas,
        outbound,
        source_ledger,
        sink_ledger,
    )


def _particle_oracle(  # noqa: C901
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    volume: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    time_step: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate immutable particle requests and canonical slot assignment.

    This deliberately mirrors the documented transport contract rather than
    importing any production planning helper: existing destination populations
    win, then equal populations share a reserved free slot, then the next
    ascending free slot is selected.
    """
    final_masses = masses.copy()
    final_concentration = concentration.copy()
    final_charge = charge.copy()
    debits = np.zeros_like(concentration)
    credits = np.zeros_like(concentration)
    assignments = np.full((len(source), concentration.shape[1]), -1, np.int32)
    requests = np.zeros_like(assignments, dtype=np.float64)
    initial_keys = {
        (box, slot): (*masses[box, slot], charge[box, slot])
        for box in range(concentration.shape[0])
        for slot in range(concentration.shape[1])
        if concentration[box, slot] > 0.0
    }
    reserved: dict[tuple[int, tuple[float, ...]], int] = {}
    occupied = {
        box: {
            key: slot
            for (key_box, slot), key in initial_keys.items()
            if key_box == box
        }
        for box in range(concentration.shape[0])
    }
    used = {box: set() for box in range(concentration.shape[0])}
    for source_box in range(concentration.shape[0]):
        for destination_box in range(concentration.shape[0]):
            for edge in range(len(source)):
                if (
                    source[edge] != source_box
                    or destination[edge] != destination_box
                ):
                    continue
                for source_slot in range(concentration.shape[1]):
                    request = (
                        concentration[source_box, source_slot]
                        * rates[edge]
                        * time_step
                    )
                    if request <= 0.0:
                        continue
                    key = initial_keys[source_box, source_slot]
                    assigned = occupied[destination_box].get(key)
                    if assigned is None:
                        reservation = (destination_box, key)
                        assigned = reserved.get(reservation)
                        if assigned is None:
                            assigned = next(
                                slot
                                for slot in range(concentration.shape[1])
                                if concentration[destination_box, slot] == 0.0
                                and slot not in used[destination_box]
                            )
                            reserved[reservation] = assigned
                            used[destination_box].add(assigned)
                    assignments[edge, source_slot] = assigned
                    requests[edge, source_slot] = request
                    debits[source_box, source_slot] += request
                    credits[destination_box, assigned] += (
                        request * volume[source_box] / volume[destination_box]
                    )
    final_concentration += credits - debits
    for box in range(concentration.shape[0]):
        for slot in range(concentration.shape[1]):
            if concentration[box, slot] == 0.0 and credits[box, slot] > 0.0:
                for edge, source_box in enumerate(source):
                    matches = np.flatnonzero(assignments[edge] == slot)
                    if destination[edge] == box and matches.size:
                        source_slot = int(matches[0])
                        final_masses[box, slot] = masses[
                            source_box, source_slot
                        ]
                        final_charge[box, slot] = charge[
                            source_box, source_slot
                        ]
                        break
            if final_concentration[box, slot] == 0.0:
                final_masses[box, slot] = 0.0
                final_charge[box, slot] = 0.0
    return (
        final_masses,
        final_concentration,
        final_charge,
        debits,
        credits,
        assignments,
        requests,
    )


@pytest.mark.parametrize("device", _device_parameters())
@pytest.mark.parametrize(
    ("source", "destination", "rates", "final_volumes"),
    [
        (
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.25, 0.5], dtype=np.float64),
            np.array([4.0, 2.0, 4.0], dtype=np.float64),
        ),
        (
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 0, 0], dtype=np.int32),
            np.array([0.1, 0.2, 0.125], dtype=np.float64),
            np.array([2.0, 1.0, 8.0], dtype=np.float64),
        ),
    ],
    ids=["chain-expansion", "reciprocal-compression"],
)
def test_gas_and_volume_match_independent_multibox_oracle(
    device: str,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    final_volumes: np.ndarray,
) -> None:
    """Match transport then volume evolution and separately conserve amounts."""
    wp = _warp()
    from particula.execution.communication import CommunicationTransportMode
    from particula.gpu.kernels.communication import (
        gas_communication_step_gpu,
        volume_evolution_step_gpu,
    )

    particles, gas = _containers(device)
    initial_gas = gas.concentration.numpy().copy()
    old_volumes = particles.volume.numpy().copy()
    expected, amounts, deltas, outbound, _, _ = _gas_oracle(
        initial_gas, old_volumes, source, destination, rates, 1.0
    )
    work = tuple(
        wp.empty((3, 2), dtype=wp.float64, device=device) for _ in range(3)
    )
    configuration = _configuration(
        device, CommunicationTransportMode.GAS, source, destination, rates
    )

    gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)
    volume_evolution_step_gpu(
        particles, gas, wp.array(final_volumes, dtype=wp.float64, device=device)
    )
    wp.synchronize()

    initial_particle_concentration = np.array(
        [
            [4.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    npt.assert_allclose(work[0].numpy(), amounts, rtol=RTOL, atol=0.0)
    npt.assert_allclose(work[1].numpy(), deltas, rtol=RTOL, atol=0.0)
    npt.assert_allclose(work[2].numpy(), outbound, rtol=RTOL, atol=0.0)
    npt.assert_allclose(
        gas.concentration.numpy(),
        expected * old_volumes[:, None] / final_volumes[:, None],
        rtol=RTOL,
        atol=0.0,
    )
    npt.assert_allclose(
        (gas.concentration.numpy() * particles.volume.numpy()[:, None]).sum(
            axis=0
        ),
        (initial_gas * old_volumes[:, None]).sum(axis=0),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        initial_particle_concentration
        * old_volumes[:, None]
        / final_volumes[:, None],
        rtol=RTOL,
        atol=0.0,
    )
    npt.assert_allclose(
        (
            particles.concentration.numpy() * particles.volume.numpy()[:, None]
        ).sum(),
        (initial_particle_concentration * old_volumes[:, None]).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )


@pytest.mark.parametrize("device", _device_parameters())
def test_open_boundary_accounting_matches_independent_oracle(
    device: str,
) -> None:
    """Keep open-source and open-sink accounting separate from closed parity."""
    wp = _warp()
    from particula.execution.communication import CommunicationTransportMode
    from particula.gpu.kernels.communication import (
        GasCommunicationBuffers,
        gas_communication_step_gpu,
    )

    particles, gas = _containers(device)
    source = np.array([-1, 0], dtype=np.int32)
    destination = np.array([1, -1], dtype=np.int32)
    rates = np.array([0.5, 0.25], dtype=np.float64)
    initial = gas.concentration.numpy().copy()
    volumes = particles.volume.numpy().copy()
    expected, amounts, deltas, outbound, source_ledger, sink_ledger = (
        _gas_oracle(initial, volumes, source, destination, rates, 1.0)
    )
    buffers = GasCommunicationBuffers(
        *(wp.empty((3, 2), dtype=wp.float64, device=device) for _ in range(5))
    )
    gas_communication_step_gpu(
        particles,
        gas,
        _configuration(
            device, CommunicationTransportMode.GAS, source, destination, rates
        ),
        1.0,
        buffers,
    )
    wp.synchronize()

    npt.assert_allclose(
        gas.concentration.numpy(), expected, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(buffers.amounts.numpy(), amounts, rtol=RTOL, atol=0.0)
    npt.assert_allclose(
        buffers.amount_deltas.numpy(), deltas, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        buffers.outbound_amounts.numpy(), outbound, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        buffers.source_amounts.numpy(), source_ledger, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        buffers.sink_amounts.numpy(), sink_ledger, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        (gas.concentration.numpy() * volumes[:, None]).sum(axis=0)
        - (initial * volumes[:, None]).sum(axis=0),
        source_ledger.sum(axis=0) - sink_ledger.sum(axis=0),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )


@pytest.mark.parametrize("device", _device_parameters())
@pytest.mark.parametrize(
    "order",
    [np.array([0, 1], dtype=np.int32), np.array([1, 0], dtype=np.int32)],
    ids=["registered", "permuted"],
)
def test_repeated_one_dimensional_calls_match_sequential_oracles(
    device: str, order: np.ndarray
) -> None:
    """Keep padded maps, sparse slots, and an isolated box independent."""
    wp = _warp()
    from particula.execution.communication import (
        CommunicationMapForm,
        CommunicationTransportMode,
    )
    from particula.gpu.kernels.communication import (
        gas_communication_step_gpu,
        particle_communication_step_gpu,
        volume_evolution_step_gpu,
    )

    map_source = np.array([0, 1], dtype=np.int32)[order]
    map_destination = np.array([1, 2], dtype=np.int32)[order]
    map_rates = np.array([0.125, 0.25], dtype=np.float64)[order]
    enabled = np.array([1, 0], dtype=np.int32)[order]
    # The disabled padded edge must not change the isolated third box.
    source = map_source[enabled == 1]
    destination = map_destination[enabled == 1]
    rates = map_rates[enabled == 1]
    particles, gas = _containers(device)
    old_volumes = particles.volume.numpy().copy()
    expected_gas = gas.concentration.numpy().copy()
    final_volumes = np.array([4.0, 2.0, 4.0], dtype=np.float64)
    configuration = _configuration(
        device,
        CommunicationTransportMode.GAS,
        map_source,
        map_destination,
        map_rates,
        form=CommunicationMapForm.ONE_DIMENSIONAL,
        enabled=enabled,
    )
    work = tuple(
        wp.empty((3, 2), dtype=wp.float64, device=device) for _ in range(3)
    )
    for step in range(2):
        expected_gas, _, _, _, _, _ = _gas_oracle(
            expected_gas, old_volumes, source, destination, rates, 1.0
        )
        gas_communication_step_gpu(particles, gas, configuration, 1.0, *work)
        if step == 0:
            expected_gas *= old_volumes[:, None] / final_volumes[:, None]
            volume_evolution_step_gpu(
                particles,
                gas,
                wp.array(final_volumes, dtype=wp.float64, device=device),
            )
            old_volumes = final_volumes.copy()
    wp.synchronize()
    npt.assert_allclose(
        gas.concentration.numpy(), expected_gas, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        gas.concentration.numpy()[2], [1.0, 8.0], rtol=0.0, atol=0.0
    )

    particles, _ = _containers(device)
    initial = (
        particles.masses.numpy().copy(),
        particles.concentration.numpy().copy(),
        particles.charge.numpy().copy(),
    )
    particle_configuration = _configuration(
        device,
        CommunicationTransportMode.PARTICLES,
        map_source,
        map_destination,
        map_rates,
        form=CommunicationMapForm.ONE_DIMENSIONAL,
        enabled=enabled,
    )
    from particula.gpu.kernels.communication import ParticleCommunicationBuffers

    buffers = ParticleCommunicationBuffers(
        wp.empty((3, 4), dtype=wp.float64, device=device),
        wp.empty((3, 4), dtype=wp.float64, device=device),
        wp.empty((2, 4), dtype=wp.int32, device=device),
        wp.empty((2, 4), dtype=wp.float64, device=device),
    )
    expected = initial
    for _ in range(2):
        expected = _particle_oracle(
            *expected,
            particles.volume.numpy(),
            source,
            destination,
            rates,
            1.0,
        )[:3]
        particle_communication_step_gpu(
            particles, particle_configuration, 1.0, buffers
        )
    wp.synchronize()
    npt.assert_allclose(
        particles.masses.numpy(), expected[0], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy(), expected[1], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.charge.numpy(), expected[2], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy()[2], initial[1][2], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("device", _device_parameters())
def test_particle_transport_conserves_sparse_multibox_inventories(
    device: str,
) -> None:
    """Verify a sparse closed particle transfer keeps all weighted inventories."""
    wp = _warp()
    from particula.execution.communication import CommunicationTransportMode
    from particula.gpu.kernels.communication import (
        ParticleCommunicationBuffers,
        particle_communication_step_gpu,
    )

    particles, _ = _containers(device)
    initial_mass = particles.masses.numpy().copy()
    initial_concentration = particles.concentration.numpy().copy()
    initial_charge = particles.charge.numpy().copy()
    volumes = particles.volume.numpy().copy()
    source = np.array([0, 1], dtype=np.int32)
    destination = np.array([1, 2], dtype=np.int32)
    rates = np.array([0.25, 0.5], dtype=np.float64)
    expected = _particle_oracle(
        initial_mass,
        initial_concentration,
        initial_charge,
        volumes,
        source,
        destination,
        rates,
        1.0,
    )
    buffers = ParticleCommunicationBuffers(
        wp.empty((3, 4), dtype=wp.float64, device=device),
        wp.empty((3, 4), dtype=wp.float64, device=device),
        wp.empty((2, 4), dtype=wp.int32, device=device),
        wp.empty((2, 4), dtype=wp.float64, device=device),
    )
    particle_communication_step_gpu(
        particles,
        _configuration(
            device,
            CommunicationTransportMode.PARTICLES,
            source,
            destination,
            rates,
        ),
        1.0,
        buffers,
    )
    wp.synchronize()

    final_mass = particles.masses.numpy()
    final_concentration = particles.concentration.numpy()
    final_charge = particles.charge.numpy()
    npt.assert_allclose(final_mass, expected[0], rtol=RTOL, atol=0.0)
    npt.assert_allclose(final_concentration, expected[1], rtol=RTOL, atol=0.0)
    npt.assert_allclose(final_charge, expected[2], rtol=RTOL, atol=0.0)
    npt.assert_allclose(
        buffers.source_debits.numpy(), expected[3], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        buffers.destination_credits.numpy(), expected[4], rtol=RTOL, atol=0.0
    )
    npt.assert_array_equal(buffers.assignments.numpy(), expected[5])
    npt.assert_allclose(
        buffers.request_concentrations.numpy(), expected[6], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        (final_concentration * volumes[:, None]).sum(),
        (initial_concentration * volumes[:, None]).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        (
            final_mass * final_concentration[..., None] * volumes[:, None, None]
        ).sum(axis=(0, 1)),
        (
            initial_mass
            * initial_concentration[..., None]
            * volumes[:, None, None]
        ).sum(axis=(0, 1)),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        (final_charge * final_concentration * volumes[:, None]).sum(),
        (initial_charge * initial_concentration * volumes[:, None]).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    assert np.all(buffers.assignments.numpy() >= -1)
