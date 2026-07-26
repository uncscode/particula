"""Focused P1 validation plus sidecar-only P2 and P3 staging tests."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.warp
wp = pytest.importorskip("warp")

import particula.gpu.kernels.nucleation as nucleation_module  # noqa: E402
from particula.gpu.kernels.exhaustion import ResamplingBuffers  # noqa: E402
from particula.gpu.kernels.nucleation import (  # noqa: E402
    _P2_GATE_GAS_LIMITED_OFFSET,
    _P2_GATE_LOW_SATURATION,
    _P2_GATE_ZERO_COEFFICIENT,
    _P2_GATE_ZERO_INVENTORY,
    _P2_GATE_ZERO_PRECURSOR,
    _P2_GATE_ZERO_SURVIVAL,
    _P2_GATE_ZERO_TIME,
    NucleationConfig,
    NucleationDiagnosticBuffers,
    NucleationExhaustionBuffers,
    NucleationExhaustionControls,
    NucleationFinalizedDemandBuffers,
    NucleationScratchBuffers,
    _orchestrate_nucleation_exhaustion,
    _plan_nucleation_demand,
    _preflight_nucleation,
    _real,
    _stage_nucleation_slots,
)
from particula.util.constants import AVOGADRO_NUMBER  # noqa: E402

_DEVICE_CASES = [pytest.param("cpu", id="cpu")]
if wp.is_cuda_available():
    _DEVICE_CASES.append(
        pytest.param("cuda:0", marks=pytest.mark.cuda, id="cuda")
    )


def _state(
    boxes: int = 1,
    particles: int = 2,
    species: int = 1,
    device: str = "cpu",
):
    """Build valid caller-owned fixed-capacity state on the requested device."""
    particle_data = SimpleNamespace(
        masses=wp.zeros(
            (boxes, particles, species), dtype=wp.float64, device=device
        ),
        concentration=wp.ones(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        charge=wp.zeros((boxes, particles), dtype=wp.float64, device=device),
        density=wp.ones(species, dtype=wp.float64, device=device)
        * wp.float64(1000.0),
        volume=wp.ones(boxes, dtype=wp.float64, device=device),
    )
    gas_data = SimpleNamespace(
        molar_mass=wp.ones(species, dtype=wp.float64, device=device)
        * wp.float64(0.1),
        concentration=wp.ones(
            (boxes, species), dtype=wp.float64, device=device
        ),
        partitioning=wp.ones((boxes, species), dtype=wp.int32, device=device),
    )
    return particle_data, gas_data


def _config(**changes):
    """Build a valid activation configuration."""
    values = dict(
        rate_law="activation",
        coefficient=1.0,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1,),
        formation_diameter=1e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    values.update(changes)
    return NucleationConfig(**values)


def _sidecars(boxes: int, particles: int, species: int, device: str = "cpu"):
    """Build all future caller-owned sidecars with deliberately stale values."""
    return (
        NucleationScratchBuffers(
            *[wp.ones(boxes, dtype=wp.float64, device=device) for _ in range(3)]
        ),
        NucleationFinalizedDemandBuffers(
            wp.ones(boxes, dtype=wp.int32, device=device),
            wp.ones(boxes, dtype=wp.float64, device=device),
            wp.ones((boxes, species), dtype=wp.float64, device=device),
        ),
        NucleationDiagnosticBuffers(
            wp.ones(boxes, dtype=wp.int32, device=device),
            wp.ones((boxes, particles), dtype=wp.int32, device=device),
            wp.ones((boxes, particles), dtype=wp.int32, device=device),
            wp.ones(boxes, dtype=wp.int32, device=device),
            wp.ones(boxes, dtype=wp.int32, device=device),
        ),
    )


def _snapshot_arrays(*owners):
    """Capture caller-owned Warp array identity, schema, and exact contents."""
    snapshot = []
    for owner in owners:
        for value in owner.__dict__.values():
            if hasattr(value, "numpy"):
                snapshot.append(
                    (
                        value,
                        tuple(value.shape),
                        value.dtype,
                        str(value.device),
                        value.numpy().copy(),
                    )
                )
    return snapshot


def _assert_snapshot_unchanged(snapshot):
    """Assert a complete caller-owned array snapshot remains unchanged."""
    for value, shape, dtype, device, contents in snapshot:
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        assert str(value.device) == device
        np.testing.assert_array_equal(value.numpy(), contents, strict=True)


def _oracle_plan(concentration, molar_mass, config, time_step):
    """Compute a NumPy reference for the private P2 demand planner."""
    precursor = (
        concentration[:, config.precursor_index]
        * AVOGADRO_NUMBER
        / molar_mass[config.precursor_index]
    )
    rate = config.survival_factor * config.coefficient * precursor
    if config.rate_law == "kinetic":
        rate = rate * precursor
    potential = rate * time_step
    event_mass = np.zeros_like(molar_mass)
    mask = np.asarray(config.molecule_counts, dtype=np.int32) > 0
    event_mass[mask] = (
        np.asarray(config.molecule_counts, dtype=np.float64)[mask]
        * molar_mass[mask]
        / AVOGADRO_NUMBER
    )
    admitted = np.minimum(
        potential,
        np.min(concentration[:, mask] / event_mass[mask], axis=1),
    )
    removal = admitted[:, None] * event_mass
    return precursor, rate, potential, admitted, removal


def _stage_slots(counts, particle_slots, device="cpu"):
    """Stage explicit integral P3 counts against a fixed particle-slot layout."""
    boxes, particles = particle_slots.shape
    particle_data, gas_data = _state(boxes, particles, 1, device)
    particle_data.concentration = wp.array(
        np.where(particle_slots, 1.0, 0.0), dtype=wp.float64, device=device
    )
    particle_data.masses = wp.array(
        np.where(particle_slots[:, :, None], 1.0, 0.0),
        dtype=wp.float64,
        device=device,
    )
    scratch, finalized, diagnostics = _sidecars(boxes, particles, 1, device)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array(
            np.asarray(counts, dtype=np.float64),
            dtype=wp.float64,
            device=device,
        ),
        finalized.precursor_mass_change,
    )
    preflight = _preflight_nucleation(
        particle_data,
        gas_data,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    _stage_nucleation_slots(preflight, finalized, diagnostics)
    return particle_data, gas_data, scratch, finalized, diagnostics


def _exhaustion_buffers(boxes, particles, species, device="cpu"):
    """Build distinct P4 and nested resampling sidecars."""
    resampling_buffers = ResamplingBuffers(
        retained_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        released_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        retained_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        released_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        sorted_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
        replacement_masses=wp.zeros(
            (boxes, particles, species), dtype=wp.float64, device=device
        ),
        replacement_concentration=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        replacement_charge=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        source_radii=wp.zeros(
            (boxes, particles), dtype=wp.float64, device=device
        ),
        radius_cubed_relative_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        mean_radius_relative_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        surface_relative_error=wp.zeros(boxes, dtype=wp.float64, device=device),
        diversity_absolute_error=wp.zeros(
            boxes, dtype=wp.float64, device=device
        ),
        planning_status=wp.zeros(boxes, dtype=wp.int32, device=device),
    )
    return NucleationExhaustionBuffers(
        resampling_buffers=resampling_buffers,
        demand_workspace=wp.zeros(boxes, dtype=wp.float64, device=device),
        final_demand=wp.zeros(boxes, dtype=wp.float64, device=device),
        requested_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_scale=wp.ones(boxes, dtype=wp.float64, device=device),
        minimum_volume=wp.ones(boxes, dtype=wp.float64, device=device),
        resolved_scale=wp.zeros(boxes, dtype=wp.float64, device=device),
        resampling_releasable_counts=wp.zeros(
            boxes, dtype=wp.int32, device=device
        ),
        required_release_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        scaling_required=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_counts=wp.zeros(boxes, dtype=wp.int32, device=device),
        final_selected_slot_indices=wp.zeros(
            (boxes, particles), dtype=wp.int32, device=device
        ),
    )


@pytest.mark.parametrize(
    ("counts", "slots", "expected_selected"),
    [
        ([2.0], np.array([[False, False]]), [[0, 1]]),
        ([2.0], np.array([[True, False, True, False]]), [[1, 3, -1, -1]]),
        ([4.0], np.array([[False, False]]), [[0, 1]]),
    ],
)
@pytest.mark.parametrize("device", _DEVICE_CASES)
def test_stage_nucleation_slots_writes_expected_layout(
    counts, slots, expected_selected, device
):
    """P3 retains full counts and selects the ascending free-slot prefix."""
    particles, gas, scratch, finalized, diagnostics = _stage_slots(
        counts, slots, device
    )
    particle_snapshot = _snapshot_arrays(particles)
    gas_snapshot = _snapshot_arrays(gas)
    scratch_snapshot = _snapshot_arrays(scratch)
    np.testing.assert_array_equal(
        finalized.accepted_counts.numpy(), np.asarray(counts, dtype=np.int32)
    )
    expected_free = np.where(~slots[0])[0].astype(np.int32)
    expected_free = np.pad(
        expected_free,
        (0, slots.shape[1] - len(expected_free)),
        constant_values=-1,
    )
    np.testing.assert_array_equal(
        diagnostics.free_slot_indices.numpy(), [expected_free]
    )
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), expected_selected
    )
    assert diagnostics.active_slot_counts.numpy()[0] == np.count_nonzero(slots)
    assert diagnostics.free_slot_counts.numpy()[0] == np.count_nonzero(~slots)
    np.testing.assert_array_equal(
        finalized.accepted_demand.numpy(), np.asarray(counts, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), np.ones((1, 1))
    )
    np.testing.assert_array_equal(diagnostics.gate_codes.numpy(), [1])
    _assert_snapshot_unchanged(particle_snapshot)
    _assert_snapshot_unchanged(gas_snapshot)
    _assert_snapshot_unchanged(scratch_snapshot)


@pytest.mark.parametrize("value", [1, np.bool_(True)])
def test_p4_controls_require_exact_python_booleans(value):
    """Policy controls reject integer and NumPy Boolean substitutes."""
    with pytest.raises(TypeError, match="exact Python bool"):
        NucleationExhaustionControls(value, False)


def test_p4_finalizes_free_capacity_without_mutating_p2_or_p3_handoffs():
    """P4 copies immutable demand and writes the current free-slot prefix."""
    particles, gas, scratch, finalized, diagnostics = _stage_slots(
        [1.0], np.array([[False, False]])
    )
    buffers = _exhaustion_buffers(1, 2, 1)
    object.__setattr__(
        buffers,
        "required_release_counts",
        wp.full(1, wp.int32(9), dtype=wp.int32, device="cpu"),
    )
    object.__setattr__(
        buffers,
        "scaling_required",
        wp.full(1, wp.int32(9), dtype=wp.int32, device="cpu"),
    )
    p2_p3_before = _snapshot_arrays(finalized, diagnostics)
    source_before = _snapshot_arrays(particles, gas, scratch)

    _orchestrate_nucleation_exhaustion(
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        ),
        finalized,
        diagnostics,
        NucleationExhaustionControls(False, False),
        buffers,
    )

    np.testing.assert_array_equal(buffers.demand_workspace.numpy(), [1.0])
    np.testing.assert_array_equal(buffers.final_demand.numpy(), [1.0])
    np.testing.assert_array_equal(buffers.final_counts.numpy(), [1])
    np.testing.assert_array_equal(buffers.required_release_counts.numpy(), [0])
    np.testing.assert_array_equal(buffers.scaling_required.numpy(), [0])
    np.testing.assert_array_equal(
        buffers.final_selected_slot_indices.numpy(), [[0, -1]]
    )
    np.testing.assert_array_equal(buffers.resolved_scale.numpy(), [1.0])
    _assert_snapshot_unchanged(p2_p3_before)
    _assert_snapshot_unchanged(source_before)


def test_p4_rejects_unresolved_capacity_before_writing_any_sidecar():
    """An exhausted row without a viable policy preserves all P4 state."""
    particles, gas, scratch, finalized, diagnostics = _stage_slots(
        [1.0], np.array([[True, True]])
    )
    buffers = _exhaustion_buffers(1, 2, 1)
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    before = _snapshot_arrays(
        particles,
        gas,
        scratch,
        finalized,
        diagnostics,
        buffers,
        buffers.resampling_buffers,
    )

    with pytest.raises(
        ValueError, match="P4 handoff, policy input, or capacity"
    ):
        _orchestrate_nucleation_exhaustion(
            preflight,
            finalized,
            diagnostics,
            NucleationExhaustionControls(False, False),
            buffers,
        )

    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda finalized, buffers: object.__setattr__(
            buffers,
            "requested_scale",
            wp.full(1, 0.5, dtype=wp.float64, device="cpu"),
        ),
        lambda finalized, buffers: object.__setattr__(
            finalized,
            "accepted_counts",
            wp.full(1, 2, dtype=wp.int32, device="cpu"),
        ),
    ],
    ids=["invalid_scale_bounds", "stale_p3_count"],
)
def test_p4_expected_handoff_rejections_preserve_complete_snapshots(
    corrupt, monkeypatch
):
    """Expected P4 rejections precede primitive calls and every sidecar write."""
    particles, gas, scratch, finalized, diagnostics = _stage_slots(
        [1.0], np.array([[False, False]])
    )
    buffers = _exhaustion_buffers(1, 2, 1)
    corrupt(finalized, buffers)
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    before = _snapshot_arrays(
        particles,
        gas,
        scratch,
        finalized,
        diagnostics,
        buffers,
        buffers.resampling_buffers,
    )

    def _primitive_must_not_run(*_args, **_kwargs):
        pytest.fail("P4 called an E6-F6 primitive after expected rejection.")

    monkeypatch.setattr(
        nucleation_module, "resampling_step_gpu", _primitive_must_not_run
    )
    monkeypatch.setattr(
        nucleation_module,
        "representative_volume_scaling_step_gpu",
        _primitive_must_not_run,
    )

    with pytest.raises(
        ValueError, match="P4 handoff, policy input, or capacity"
    ):
        _orchestrate_nucleation_exhaustion(
            preflight,
            finalized,
            diagnostics,
            NucleationExhaustionControls(False, False),
            buffers,
        )

    _assert_snapshot_unchanged(before)


def test_p4_scaling_records_full_deficit_and_finalized_count():
    """Scaling fallback retains the full deficit and derives scaled counts."""
    particles, gas, scratch, finalized, diagnostics = _stage_slots(
        [4.0], np.array([[True, True, False, False]])
    )
    buffers = _exhaustion_buffers(1, 4, 1)
    object.__setattr__(
        buffers,
        "requested_scale",
        wp.full(1, 0.5, dtype=wp.float64, device="cpu"),
    )
    object.__setattr__(
        buffers,
        "minimum_scale",
        wp.full(1, 0.5, dtype=wp.float64, device="cpu"),
    )
    object.__setattr__(
        buffers,
        "minimum_volume",
        wp.full(1, 0.5, dtype=wp.float64, device="cpu"),
    )
    p2_p3_before = _snapshot_arrays(finalized, diagnostics)

    _orchestrate_nucleation_exhaustion(
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        ),
        finalized,
        diagnostics,
        NucleationExhaustionControls(False, True),
        buffers,
    )

    np.testing.assert_array_equal(buffers.required_release_counts.numpy(), [2])
    np.testing.assert_array_equal(buffers.scaling_required.numpy(), [1])
    np.testing.assert_array_equal(buffers.final_demand.numpy(), [2.0])
    np.testing.assert_array_equal(buffers.final_counts.numpy(), [1])
    np.testing.assert_array_equal(
        buffers.final_selected_slot_indices.numpy(), [[2, -1, -1, -1]]
    )
    _assert_snapshot_unchanged(p2_p3_before)


def test_stage_nucleation_slots_rejects_replacement_records_without_writes():
    """P3 accepts only the exact P2/P3 sidecar records validated by P1."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array([1.0], dtype=wp.float64, device="cpu"),
        finalized.precursor_mass_change,
    )
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    replacement_finalized = NucleationFinalizedDemandBuffers(
        wp.ones(1, dtype=wp.int32, device="cpu"),
        wp.ones(1, dtype=wp.float64, device="cpu"),
        wp.ones((1, 1), dtype=wp.float64, device="cpu"),
    )
    replacement_diagnostics = NucleationDiagnosticBuffers(
        *[
            wp.ones(tuple(value.shape), dtype=wp.int32, device="cpu")
            for value in diagnostics.__dict__.values()
        ]
    )
    before = _snapshot_arrays(
        particles,
        gas,
        scratch,
        finalized,
        diagnostics,
        replacement_finalized,
        replacement_diagnostics,
    )

    with pytest.raises(ValueError, match="preflight-validated record"):
        _stage_nucleation_slots(preflight, replacement_finalized, diagnostics)
    with pytest.raises(ValueError, match="preflight-validated record"):
        _stage_nucleation_slots(preflight, finalized, replacement_diagnostics)

    _assert_snapshot_unchanged(before)


def test_stage_nucleation_slots_rejects_replaced_record_buffer_without_writes():
    """P3 rejects a record whose validated storage was substituted after P1."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array([1.0], dtype=wp.float64, device="cpu"),
        finalized.precursor_mass_change,
    )
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    replacement = wp.ones(1, dtype=wp.int32, device="cpu")
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)
    object.__setattr__(finalized, "accepted_counts", replacement)

    with pytest.raises(ValueError, match="preflight-validated storage"):
        _stage_nucleation_slots(preflight, finalized, diagnostics)

    _assert_snapshot_unchanged(before)
    np.testing.assert_array_equal(replacement.numpy(), [1])


def test_stage_nucleation_slots_preserves_nonzero_particle_gas_and_p2_state():
    """Successful P3 modifies only its count and slot-diagnostic outputs."""
    particles, gas = _state()
    particles.masses = wp.ones((1, 2, 1), dtype=wp.float64, device="cpu")
    particles.charge = wp.array([[-2.0, 3.0]], dtype=wp.float64, device="cpu")
    gas.concentration = wp.array([[7.0]], dtype=wp.float64, device="cpu")
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array([1.0], dtype=wp.float64, device="cpu"),
        finalized.precursor_mass_change,
    )
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    preserved = _snapshot_arrays(particles, gas, scratch)
    p2_preserved = _snapshot_arrays(
        SimpleNamespace(
            accepted_demand=finalized.accepted_demand,
            precursor_mass_change=finalized.precursor_mass_change,
            gate_codes=diagnostics.gate_codes,
        )
    )

    _stage_nucleation_slots(preflight, finalized, diagnostics)

    _assert_snapshot_unchanged(preserved)
    _assert_snapshot_unchanged(p2_preserved)
    np.testing.assert_array_equal(finalized.accepted_counts.numpy(), [1])
    np.testing.assert_array_equal(diagnostics.active_slot_counts.numpy(), [2])
    np.testing.assert_array_equal(diagnostics.free_slot_counts.numpy(), [0])
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), [[-1, -1]]
    )


@pytest.mark.parametrize(
    "field",
    ["free_slot_indices", "active_slot_counts", "free_slot_counts"],
)
def test_p3_diagnostic_sidecars_require_int32_nonoverlapping_storage(field):
    """P1 rejects malformed or aliased P3 diagnostic sidecars unchanged."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    values = dict(diagnostics.__dict__)
    values[field] = wp.ones(
        tuple(getattr(diagnostics, field).shape),
        dtype=wp.float64,
        device="cpu",
    )
    malformed = NucleationDiagnosticBuffers(**values)
    before = _snapshot_arrays(particles, gas, scratch, finalized, malformed)

    with pytest.raises(ValueError, match="required dtype"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=malformed,
        )
    _assert_snapshot_unchanged(before)

    aliased = NucleationDiagnosticBuffers(
        diagnostics.gate_codes,
        diagnostics.selected_slot_indices,
        diagnostics.free_slot_indices,
        finalized.accepted_counts,
        diagnostics.free_slot_counts,
    )
    before = _snapshot_arrays(particles, gas, scratch, finalized, aliased)
    with pytest.raises(ValueError, match="must not overlap"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=aliased,
        )
    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    "accepted_demand",
    [
        np.nan,
        -1.0,
        1.5,
        float(np.iinfo(np.int32).max) + 1.0,
    ],
    ids=["nonfinite", "negative", "nonintegral", "above_int32_maximum"],
)
def test_stage_nucleation_slots_rejects_invalid_event_conversion_without_writes(
    accepted_demand,
):
    """Invalid P3 conversions preserve every caller-owned P3 output."""
    particle_data, gas_data = _state(1, 2, 1)
    particle_data.concentration = wp.zeros(
        (1, 2), dtype=wp.float64, device="cpu"
    )
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array([accepted_demand], dtype=wp.float64, device="cpu"),
        finalized.precursor_mass_change,
    )
    before = _snapshot_arrays(finalized, diagnostics)
    preflight = _preflight_nucleation(
        particle_data,
        gas_data,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    with pytest.raises(ValueError, match="integral int32"):
        _stage_nucleation_slots(preflight, finalized, diagnostics)
    _assert_snapshot_unchanged(before)


def test_stage_nucleation_slots_handles_zero_capacity():
    """P3 retains a count and produces empty diagnostics at zero capacity."""
    _, _, _, finalized, diagnostics = _stage_slots(
        [3.0], np.zeros((1, 0), dtype=bool)
    )
    np.testing.assert_array_equal(finalized.accepted_counts.numpy(), [3])
    assert diagnostics.selected_slot_indices.shape == (1, 0)
    np.testing.assert_array_equal(diagnostics.active_slot_counts.numpy(), [0])
    np.testing.assert_array_equal(diagnostics.free_slot_counts.numpy(), [0])


def test_stage_nucleation_slots_accepts_int32_maximum_event_count():
    """The inclusive int32 event-count boundary is accepted independently."""
    _, _, _, finalized, _ = _stage_slots(
        [float(np.iinfo(np.int32).max)], np.array([[False]])
    )
    np.testing.assert_array_equal(
        finalized.accepted_counts.numpy(), [np.iinfo(np.int32).max]
    )


def test_stage_nucleation_slots_e6_f5_failure_preserves_p3_sidecars():
    """E6-F5 slot-state rejection occurs before P3 count or selection writes."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    finalized = NucleationFinalizedDemandBuffers(
        finalized.accepted_counts,
        wp.array([1.0], dtype=wp.float64, device="cpu"),
        finalized.precursor_mass_change,
    )
    before = _snapshot_arrays(finalized, diagnostics)
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    with pytest.raises(ValueError, match="Invalid particle slot state"):
        _stage_nucleation_slots(preflight, finalized, diagnostics)
    _assert_snapshot_unchanged(before)


def test_stage_nucleation_slots_zero_boxes_is_write_free():
    """P3 returns without sidecar writes after valid empty-shape preflight."""
    particles, gas = _state(boxes=0)
    scratch, finalized, diagnostics = _sidecars(0, 2, 1)
    before = _snapshot_arrays(finalized, diagnostics)
    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    _stage_nucleation_slots(preflight, finalized, diagnostics)
    _assert_snapshot_unchanged(before)


def test_p2_to_p3_zero_demand_seam_preserves_p2_outputs():
    """P3 stages zero P2 demand without changing P2-owned sidecars or state."""
    particles, gas = _state()
    particles.concentration = wp.zeros((1, 2), dtype=wp.float64, device="cpu")
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    source_before = _snapshot_arrays(particles, gas)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(),
        0.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    p2_before = _snapshot_arrays(scratch)
    accepted_demand_before = finalized.accepted_demand.numpy().copy()
    removal_before = finalized.precursor_mass_change.numpy().copy()
    gate_codes_before = diagnostics.gate_codes.numpy().copy()
    np.testing.assert_array_equal(finalized.accepted_counts.numpy(), [1])
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), [[1, 1]]
    )

    preflight = _preflight_nucleation(
        particles,
        gas,
        _config(),
        0.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    _stage_nucleation_slots(preflight, finalized, diagnostics)

    np.testing.assert_array_equal(finalized.accepted_counts.numpy(), [0])
    np.testing.assert_array_equal(
        diagnostics.free_slot_indices.numpy(), [[0, 1]]
    )
    np.testing.assert_array_equal(diagnostics.active_slot_counts.numpy(), [0])
    np.testing.assert_array_equal(diagnostics.free_slot_counts.numpy(), [2])
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), [[-1, -1]]
    )
    np.testing.assert_array_equal(
        finalized.accepted_demand.numpy(), accepted_demand_before
    )
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), removal_before
    )
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), gate_codes_before
    )
    _assert_snapshot_unchanged(source_before)
    _assert_snapshot_unchanged(p2_before)


def test_config_is_frozen_but_retains_arrays_by_identity():
    """Frozen records reject rebinding without copying caller array sidecars."""
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.coefficient = 2.0
    array = wp.zeros(1, dtype=wp.float64, device="cpu")
    buffers = NucleationScratchBuffers(array, array, array)
    assert buffers.precursor_number_concentration is array


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rate_law": "other"}, "rate_law"),
        ({"coefficient": True}, "coefficient"),
        ({"formation_diameter": 0.0}, "formation_diameter"),
        ({"molecule_counts": (0,)}, "molecule_counts"),
    ],
)
def test_config_rejects_invalid_scalar_rules(changes, message):
    """Configuration validation rejects invalid scalar-only contracts."""
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**changes)


@pytest.mark.parametrize(
    ("value", "positive", "exception"),
    [
        (True, False, TypeError),
        ("1.0", False, TypeError),
        (np.nan, False, ValueError),
        (-1.0, False, ValueError),
        (0.0, True, ValueError),
    ],
)
def test_real_rejects_nonphysical_scalar_values(value, positive, exception):
    """Scalar normalization rejects booleans, non-reals, and bad domains."""
    with pytest.raises(exception):
        _real(value, "value", positive=positive)


def test_config_accepts_kinetic_saturation_bounds():
    """A complete kinetic configuration retains valid paired bounds."""
    config = _config(
        rate_law="kinetic",
        saturation_lower=0.1,
        saturation_upper=1.2,
    )
    assert config.rate_law == "kinetic"
    assert config.saturation_upper == 1.2


@pytest.mark.parametrize(
    "changes",
    [
        {"precursor_index": True},
        {"molecule_counts": [1]},
        {"molecule_counts": (1.0,)},
        {"temperature_lower": 400.0, "temperature_upper": 200.0},
        {"saturation_lower": 0.1},
        {"saturation_lower": 1.0, "saturation_upper": 0.1},
    ],
)
def test_config_rejects_inconsistent_metadata_and_intervals(changes):
    """Configuration rejects malformed species metadata and paired intervals."""
    with pytest.raises((TypeError, ValueError)):
        _config(**changes)


def test_preflight_is_read_only_for_valid_inputs():
    """Valid P1 preflight retains state and returns normalized metadata."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    environment = SimpleNamespace(
        temperature=wp.full(
            1, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
    )
    before = _snapshot_arrays(
        particles, gas, environment, scratch, finalized, diagnostics
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        environment=environment,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.n_boxes == 1
    assert result.n_particles == 2
    _assert_snapshot_unchanged(before)


def test_preflight_accepts_finite_signed_particle_charge():
    """Finite signed elementary-charge state remains valid P1 input."""
    particles, gas = _state()
    particles.charge = wp.array(
        np.array([[-1.0, 1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )

    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
    )

    assert result.has_eligible_boxes


def test_preflight_accepts_environment_temperature_and_saturation():
    """Environment-owned arrays support configured saturation gating metadata."""
    particles, gas = _state()
    environment = SimpleNamespace(
        temperature=wp.ones(1, dtype=wp.float64, device="cpu")
        * wp.float64(300.0),
        saturation_ratio=wp.ones((1, 1), dtype=wp.float64, device="cpu"),
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        environment=environment,
    )
    assert result.temperature is environment.temperature
    assert result.saturation is environment.saturation_ratio
    assert result.has_eligible_boxes


def test_preflight_empty_capacity_returns_no_eligible_boxes():
    """Empty fixed-capacity storage is valid but has no eligible work."""
    particles, gas = _state(particles=0)
    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
    )
    assert result.n_particles == 0
    assert not result.has_eligible_boxes
    assert not result.has_gated_boxes


@pytest.mark.parametrize(
    ("config", "saturation", "message"),
    [
        (
            _config(precursor_number_concentration_upper=1.0),
            None,
            "precursor_number_concentration",
        ),
        (
            _config(saturation_lower=0.5, saturation_upper=1.5),
            1.6,
            "saturation",
        ),
    ],
)
def test_p2_empty_capacity_validates_before_writing_sidecars(
    config, saturation, message
):
    """Invalid nonempty-box inputs reject before P2 writes any sidecar."""
    particles, gas = _state(particles=0)
    scratch, finalized, diagnostics = _sidecars(1, 0, 1)
    saturation_data = None
    if saturation is not None:
        saturation_data = wp.full(
            (1, 1), wp.float64(saturation), dtype=wp.float64, device="cpu"
        )
    owners = (particles, gas, scratch, finalized, diagnostics)
    if saturation_data is not None:
        owners += (SimpleNamespace(saturation=saturation_data),)
    before = _snapshot_arrays(*owners)

    with pytest.raises(ValueError, match=message):
        _plan_nucleation_demand(
            particles,
            gas,
            config,
            1.0,
            temperature=300.0,
            saturation=saturation_data,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )

    _assert_snapshot_unchanged(before)


def test_p2_derived_invalid_demand_preserves_all_caller_state():
    """A derived overflow rejects before P2 commits its owned sidecars."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)

    with pytest.raises(ValueError, match="Derived nucleation demand"):
        _plan_nucleation_demand(
            particles,
            gas,
            _config(coefficient=np.finfo(np.float64).max),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )

    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 100.0, "temperature"),
        ("temperature", 500.0, "temperature"),
    ],
)
def test_preflight_rejects_scalar_temperature_outside_bounds(
    field, value, message
):
    """Scalar environmental values must remain inside configured bounds."""
    particles, gas = _state()
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("temperature", "saturation", "message"),
    [
        (100.0, 1.0, "temperature"),
        (500.0, 1.0, "temperature"),
        (300.0, 1.6, "saturation"),
    ],
)
def test_preflight_rejects_direct_array_values_outside_bounds(
    temperature, saturation, message
):
    """Device-resident temperature and saturation arrays enforce bounds."""
    particles, gas = _state()
    temperature_data = wp.ones(1, dtype=wp.float64, device="cpu") * wp.float64(
        temperature
    )
    saturation_data = wp.ones(
        (1, 1), dtype=wp.float64, device="cpu"
    ) * wp.float64(saturation)
    before = temperature_data.numpy().copy()
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(saturation_lower=0.5, saturation_upper=1.5),
            1.0,
            temperature=temperature_data,
            saturation=saturation_data,
        )
    np.testing.assert_array_equal(temperature_data.numpy(), before)


def test_preflight_low_saturation_is_a_read_only_gate():
    """Saturation below its lower endpoint gates rather than rejects."""
    particles, gas = _state()
    saturation_data = wp.full(
        (1, 1), wp.float64(0.4), dtype=wp.float64, device="cpu"
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            1, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation_data,
    )
    assert result.gate_reason == "low_saturation"


@pytest.mark.parametrize(
    ("time_step", "changes", "reason"),
    [
        (0.0, {}, "zero_time"),
        (1.0, {"coefficient": 0.0}, "zero_coefficient"),
        (1.0, {"survival_factor": 0.0}, "zero_survival"),
    ],
)
def test_zero_gates_leave_stale_sidecars_untouched(time_step, changes, reason):
    """All-box scalar gates return zero metadata without clearing sidecars."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    temperature = wp.full(1, wp.float64(300.0), dtype=wp.float64, device="cpu")
    before = _snapshot_arrays(
        particles, gas, temperature, scratch, finalized, diagnostics
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(**changes),
        time_step,
        temperature=temperature,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.gate_reason == reason
    assert result.potential_rate == 0.0
    _assert_snapshot_unchanged(before)


def test_rejection_does_not_write_state():
    """Schema/science rejection occurs before caller state is modified."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    temperature = wp.full(1, wp.float64(100.0), dtype=wp.float64, device="cpu")
    before = _snapshot_arrays(
        particles, gas, temperature, scratch, finalized, diagnostics
    )
    with pytest.raises(ValueError, match="temperature"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=temperature,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    ("configure", "message"),
    [
        (
            lambda particles, gas: setattr(
                gas,
                "partitioning",
                wp.zeros((1, 1), dtype=wp.int32, device="cpu"),
            ),
            "precursor partitioning",
        ),
        (
            lambda particles, gas: setattr(
                gas,
                "partitioning",
                wp.ones((1, 1), dtype=wp.int32, device="cpu") * 2,
            ),
            "binary",
        ),
    ],
)
def test_preflight_rejects_invalid_precursor_partitioning(configure, message):
    """Precursor partitioning must be enabled and use binary metadata."""
    particles, gas = _state()
    configure(particles, gas)
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
        )


def test_preflight_rejects_conflicting_or_unconfigured_environment_inputs():
    """Direct and environment forms cannot be mixed or silently ignored."""
    particles, gas = _state()
    environment = SimpleNamespace(
        temperature=wp.ones(1, dtype=wp.float64, device="cpu")
        * wp.float64(300.0),
    )
    with pytest.raises(ValueError, match="require no environment"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            environment=environment,
        )
    with pytest.raises(ValueError, match="not configured"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            saturation=wp.ones((1, 1), dtype=wp.float64, device="cpu"),
        )


def test_preflight_rejects_sidecar_overlap_without_mutating_state():
    """Sidecars cannot alias mutable particle fields during a read-only call."""
    particles, gas = _state()
    before = particles.concentration.numpy().copy()
    scratch = NucleationScratchBuffers(
        particles.volume,
        wp.zeros(1, dtype=wp.float64, device="cpu"),
        wp.zeros(1, dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
        )
    np.testing.assert_array_equal(particles.concentration.numpy(), before)


def test_real_rejects_oversized_scalars_as_value_errors():
    """Scalar conversion overflow follows the finite-domain error contract."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _real(10**1000, "value")


def test_config_and_time_step_reject_oversized_scalars_as_value_errors():
    """Configuration and direct scalar overflow do not leak OverflowError."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _config(coefficient=10**1000)
    particles, gas = _state()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _preflight_nucleation(
            particles, gas, _config(), 10**1000, temperature=300.0
        )


def test_preflight_rejects_lookalike_and_mutated_config_without_writes():
    """Runtime config validation rejects untrusted indices before kernel use."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)
    lookalike = SimpleNamespace(**_config().__dict__)
    with pytest.raises(ValueError, match="NucleationConfig"):
        _preflight_nucleation(
            particles,
            gas,
            lookalike,
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    config = _config()
    object.__setattr__(config, "precursor_index", -1)
    with pytest.raises(ValueError, match="config scalar values"):
        _preflight_nucleation(
            particles,
            gas,
            config,
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize("temperature", [100.0, 500.0])
@pytest.mark.parametrize(
    "changes", [{"coefficient": 0.0}, {"survival_factor": 0.0}]
)
def test_scalar_zero_config_gates_before_temperature_interval(
    temperature, changes
):
    """Valid zero rate inputs short-circuit configured scalar temperature bounds."""
    particles, gas = _state()
    result = _preflight_nucleation(
        particles, gas, _config(**changes), 1.0, temperature=temperature
    )
    assert result.gate_reason in ("zero_coefficient", "zero_survival")


@pytest.mark.parametrize("temperature", [100.0, 500.0])
@pytest.mark.parametrize(
    "changes", [{"coefficient": 0.0}, {"survival_factor": 0.0}]
)
def test_device_zero_config_gates_before_temperature_interval(
    temperature, changes
):
    """Valid zero rate inputs short-circuit device temperature range checks."""
    particles, gas = _state()
    temperature_data = wp.full(
        1, wp.float64(temperature), dtype=wp.float64, device="cpu"
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(**changes),
        1.0,
        temperature=temperature_data,
    )
    assert result.gate_reason in ("zero_coefficient", "zero_survival")


def test_disjoint_precursor_and_saturation_gates_are_all_box_union():
    """Disjoint per-box gates do not incorrectly leave an eligible box."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.array(
        np.array([[1.0], [0.4]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            2, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation,
    )
    assert not result.has_eligible_boxes
    assert result.has_gated_boxes


def test_mixed_gate_union_retains_eligible_box():
    """A gate union smaller than box count retains eligible work metadata."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.ones((2, 1), dtype=wp.float64, device="cpu")
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            2, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation,
    )
    assert result.has_eligible_boxes
    assert result.has_gated_boxes


def test_p2_plans_activation_demand_and_preserves_p3_sidecars():
    """P2 commits only demand diagnostics using shared inventory admission."""
    particles, gas = _state(boxes=2, particles=0, species=2)
    gas.molar_mass = wp.array(
        np.array([0.1, 0.2], dtype=np.float64), dtype=wp.float64, device="cpu"
    )
    gas.concentration = wp.array(
        np.array([[1.0, 0.1], [2.0, 2.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 0, 2)
    source_before = _snapshot_arrays(particles, gas)
    selected_before = diagnostics.selected_slot_indices.numpy().copy()
    counts_before = finalized.accepted_counts.numpy().copy()
    config = _config(molecule_counts=(1, 1), coefficient=1.0)

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    precursor = gas.concentration.numpy()[:, 0] * 6.02214076e23 / 0.1
    potential = config.coefficient * precursor
    event_mass = np.array([0.1, 0.2]) / 6.02214076e23
    expected = np.minimum(
        potential,
        np.min(gas.concentration.numpy() / event_mass, axis=1),
    )
    np.testing.assert_allclose(
        scratch.precursor_number_concentration.numpy(), precursor, rtol=1e-12
    )
    np.testing.assert_allclose(
        scratch.potential_rate.numpy(), potential, rtol=1e-12
    )
    np.testing.assert_allclose(
        scratch.potential_demand.numpy(), potential, rtol=1e-12
    )
    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        finalized.precursor_mass_change.numpy(),
        expected[:, None] * event_mass,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(),
        [_P2_GATE_GAS_LIMITED_OFFSET + 1] * 2,
    )
    np.testing.assert_array_equal(
        finalized.accepted_counts.numpy(), counts_before
    )
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), selected_before
    )
    _assert_snapshot_unchanged(source_before)


@pytest.mark.parametrize("device", _DEVICE_CASES)
@pytest.mark.parametrize(
    ("boxes", "species", "rate_law"),
    [(1, 1, "activation"), (2, 3, "kinetic")],
)
def test_p2_matches_numpy_oracle_across_devices_and_scales(
    device, boxes, species, rate_law
):
    """P2 rate, demand, and removal match a NumPy float64 oracle."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    particles, gas = _state(
        boxes=boxes, particles=0, species=species, device=device
    )
    gas.molar_mass = wp.array(
        np.linspace(0.1, 0.1 + 0.1 * (species - 1), species, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    gas.concentration = wp.array(
        np.arange(1, boxes * species + 1, dtype=np.float64).reshape(
            boxes, species
        )
        * 1.0e-12,
        dtype=wp.float64,
        device=device,
    )
    config = _config(
        rate_law=rate_law,
        coefficient=1.0e-30,
        survival_factor=0.75,
        molecule_counts=tuple(range(1, species + 1)),
        precursor_number_concentration_upper=1.0e40,
    )
    scratch, finalized, diagnostics = _sidecars(
        boxes, 0, species, device=device
    )
    before = _snapshot_arrays(particles, gas)

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        2.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    concentration = gas.concentration.numpy()
    molar_mass = gas.molar_mass.numpy()
    precursor, rate, potential, admitted, removal = _oracle_plan(
        concentration, molar_mass, config, 2.0
    )
    np.testing.assert_allclose(
        scratch.precursor_number_concentration.numpy(), precursor, rtol=1e-12
    )
    np.testing.assert_allclose(scratch.potential_rate.numpy(), rate, rtol=1e-12)
    np.testing.assert_allclose(
        scratch.potential_demand.numpy(), potential, rtol=1e-12
    )
    np.testing.assert_allclose(finalized.accepted_demand.numpy(), admitted)
    np.testing.assert_allclose(
        finalized.precursor_mass_change.numpy(), removal, rtol=1e-12
    )
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), np.zeros(boxes, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        finalized.accepted_counts.numpy(), np.ones(boxes, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(),
        np.full((boxes, 0), -1, dtype=np.int32),
    )
    _assert_snapshot_unchanged(before)


def test_p2_zero_time_writes_zero_outputs_and_gate_code():
    """P2 scalar gates commit their defined zero diagnostics."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(),
        0.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(scratch.potential_rate.numpy(), [0.0])
    np.testing.assert_array_equal(scratch.potential_demand.numpy(), [0.0])
    np.testing.assert_array_equal(finalized.accepted_demand.numpy(), [0.0])
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), [[0.0]]
    )
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [_P2_GATE_ZERO_TIME]
    )


@pytest.mark.parametrize(
    ("changes", "concentration", "saturation_value", "expected_code"),
    [
        ({"coefficient": 0.0}, 1.0, 1.0, _P2_GATE_ZERO_COEFFICIENT),
        ({"survival_factor": 0.0}, 1.0, 1.0, _P2_GATE_ZERO_SURVIVAL),
        ({}, 0.0, 1.0, _P2_GATE_ZERO_PRECURSOR),
        ({}, 1.0, 0.4, _P2_GATE_LOW_SATURATION),
        ({}, 1.0, 1.0, _P2_GATE_ZERO_INVENTORY),
    ],
)
def test_p2_writes_all_non_time_gate_codes(
    changes, concentration, saturation_value, expected_code
):
    """P2 commits zero demand for each valid gate with defined precedence."""
    particles, gas = _state()
    gas.concentration = wp.full(
        (1, 1), wp.float64(concentration), dtype=wp.float64, device="cpu"
    )
    if expected_code == _P2_GATE_ZERO_INVENTORY:
        gas.concentration = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
        changes = {"molecule_counts": (0, 1)}
        particles, gas = _state(species=2)
        gas.concentration = wp.array(
            np.array([[1.0, 0.0]], dtype=np.float64),
            dtype=wp.float64,
            device="cpu",
        )
    boxes, particle_count, species = 1, 2, gas.molar_mass.shape[0]
    scratch, finalized, diagnostics = _sidecars(boxes, particle_count, species)
    config = _config(
        **changes,
        saturation_lower=0.5,
        saturation_upper=1.5,
    )
    saturation = wp.full(
        (1, species),
        wp.float64(saturation_value),
        dtype=wp.float64,
        device="cpu",
    )

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [expected_code]
    )
    if expected_code == _P2_GATE_ZERO_INVENTORY:
        assert scratch.potential_rate.numpy()[0] > 0.0
    else:
        np.testing.assert_array_equal(scratch.potential_rate.numpy(), [0.0])
    np.testing.assert_array_equal(finalized.accepted_demand.numpy(), [0.0])
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), np.zeros((1, species))
    )


def test_p2_kinetic_tied_inventory_uses_lowest_limiting_species():
    """Kinetic P2 demand uses its squared rate law and stable limiter tie."""
    particles, gas = _state(species=2)
    gas.molar_mass = wp.array(
        np.array([0.1, 0.2], dtype=np.float64), dtype=wp.float64, device="cpu"
    )
    event_mass = np.array([0.1, 0.2], dtype=np.float64) / 6.02214076e23
    gas.concentration = wp.array(
        np.array([event_mass * 5.0], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(1, 2, 2)
    config = _config(
        rate_law="kinetic",
        coefficient=1.0,
        molecule_counts=(1, 1),
        precursor_number_concentration_upper=1e40,
    )

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    precursor = gas.concentration.numpy()[0, 0] * 6.02214076e23 / 0.1
    np.testing.assert_allclose(scratch.potential_rate.numpy(), [precursor**2])
    np.testing.assert_allclose(finalized.accepted_demand.numpy(), [5.0])
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [_P2_GATE_GAS_LIMITED_OFFSET]
    )


def test_p2_gate_precedence_is_scalar_then_precursor_then_saturation():
    """P2 emits the documented deterministic gate precedence per box."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.array(
        np.array([[0.4], [0.4]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 2, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        0.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(),
        [_P2_GATE_ZERO_TIME, _P2_GATE_ZERO_TIME],
    )

    _plan_nucleation_demand(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(),
        [_P2_GATE_ZERO_PRECURSOR, _P2_GATE_LOW_SATURATION],
    )


def test_p2_shared_admission_preserves_nonparticipant_zero_removal():
    """P2 limits each box independently and never removes nonparticipants."""
    particles, gas = _state(boxes=2, particles=0, species=3)
    molar_mass = np.array([0.1, 0.3, 0.2], dtype=np.float64)
    event_mass = np.array([0.1, 0.0, 0.4], dtype=np.float64) / AVOGADRO_NUMBER
    gas.molar_mass = wp.array(molar_mass, dtype=wp.float64, device="cpu")
    gas.concentration = wp.array(
        np.array(
            [
                [1.0, 9.0, event_mass[2] * 2.0],
                [0.5, 8.0, event_mass[2] * 7.0],
            ],
            dtype=np.float64,
        ),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 0, 3)
    before = _snapshot_arrays(particles, gas)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(
            coefficient=1.0e-10,
            molecule_counts=(1, 0, 2),
            precursor_number_concentration_upper=1.0e30,
        ),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    potential = (
        1.0e-10 * gas.concentration.numpy()[:, 0] * AVOGADRO_NUMBER / 0.1
    )
    expected_demand = np.minimum(potential, [2.0, 7.0])
    expected_removal = expected_demand[:, None] * event_mass
    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), expected_demand, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        finalized.precursor_mass_change.numpy(),
        expected_removal,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy()[:, 1], [0.0, 0.0]
    )
    _assert_snapshot_unchanged(before)


def test_p2_empty_boxes_are_an_exact_write_free_noop():
    """Valid empty-box planning preserves every supplied sidecar byte-for-byte."""
    particles, gas = _state(boxes=0)
    scratch, finalized, diagnostics = _sidecars(0, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scratch", None, "scratch must be NucleationScratchBuffers"),
        ("scratch", object(), "scratch must be NucleationScratchBuffers"),
        (
            "finalized_demand",
            None,
            "finalized_demand must be NucleationFinalizedDemandBuffers",
        ),
        (
            "diagnostics",
            object(),
            "diagnostics must be NucleationDiagnosticBuffers",
        ),
    ],
)
def test_p2_rejects_missing_or_wrong_sidecar_records(field, value, message):
    """P2 refuses missing or wrong records before any caller-owned writes."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)
    sidecars = {
        "scratch": scratch,
        "finalized_demand": finalized,
        "diagnostics": diagnostics,
    }
    sidecars[field] = value

    with pytest.raises(ValueError, match=message):
        _plan_nucleation_demand(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            **sidecars,
        )

    _assert_snapshot_unchanged(before)


def test_p2_inventory_rounding_uses_bounded_nextafter_correction():
    """P2 corrects an unsafe division-product ULP without a coarse scale loss."""
    event_mass = np.float64(0.1 / AVOGADRO_NUMBER)
    inventory = None
    expected_demand = None
    for events in range(1, 4097):
        for direction in (np.inf, -np.inf):
            candidate_inventory = np.nextafter(
                event_mass * np.float64(events), direction
            )
            candidate_demand = candidate_inventory / event_mass
            if candidate_demand * event_mass > candidate_inventory:
                inventory = candidate_inventory
                expected_demand = candidate_demand
                break
        if inventory is not None:
            break
    assert inventory is not None
    assert expected_demand is not None

    for _ in range(4):
        if expected_demand * event_mass <= inventory:
            break
        expected_demand = np.nextafter(expected_demand, -np.inf)
    assert expected_demand * event_mass <= inventory

    particles, gas = _state(particles=0)
    gas.concentration = wp.array(
        np.array([[inventory]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(1, 0, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(coefficient=1.0e10),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), [expected_demand], rtol=0.0, atol=0.0
    )
    assert finalized.precursor_mass_change.numpy()[0, 0] <= inventory
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [_P2_GATE_GAS_LIMITED_OFFSET]
    )


def test_p2_inventory_rounding_raises_exact_failure_after_four_corrections(
    monkeypatch,
):
    """P2 surfaces the exact inventory-safety failure when correction stalls."""
    particles, gas = _state(particles=0)
    gas.molar_mass = wp.array([0.1], dtype=wp.float64, device="cpu")
    gas.concentration = wp.array(
        np.array([[1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(1, 0, 1)

    original_launch = nucleation_module.wp.launch

    def _launch(kernel, dim=None, inputs=None, device=None):
        if kernel is nucleation_module._plan_demand_work:
            monkeypatch.setattr(
                inputs[-1],
                "numpy",
                lambda: np.array([0, 1], dtype=np.int32),
                raising=False,
            )
            return None
        return original_launch(kernel, dim=dim, inputs=inputs, device=device)

    monkeypatch.setattr(nucleation_module.wp, "launch", _launch)

    with pytest.raises(
        ValueError, match="Nucleation demand cannot be made inventory-safe."
    ):
        _plan_nucleation_demand(
            particles,
            gas,
            _config(coefficient=1.0),
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
