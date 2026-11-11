"""Native direct/prepared GPU wall-loss weighted sinks and ownership."""

from dataclasses import replace
from math import fsum
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from particula.particles import ParticleData

pytestmark = pytest.mark.warp
FAMILIES = [
    "spherical",
    "rectangular",
    "charged_spherical",
    "charged_rectangular",
]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Require installed Warp CPU and record optional CUDA availability."""
    wp = pytest.importorskip("warp")
    from particula.gpu.tests.cuda_availability import cuda_available

    if request.param == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA unavailable")
    return request.param


def _state(device, family, *, slots=8, boxes=3):
    import warp as wp

    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    radius = np.resize(np.array([20e-9, 100e-9, 2e-6]), slots)
    material_volume = 4 * np.pi * radius**3 / 3
    masses = np.stack(
        [0.4 * 1000 * material_volume, 0.6 * 1500 * material_volume], axis=1
    )
    counts = np.resize(
        np.array([0.25, 1.0, 4.0, 0.0, 2.5, 0.5, 0.0, 8.0]), slots
    )
    charge = np.resize(np.array([2.0, -1.0, 0, 13, -2, 3, -9, 1]), slots)
    if slots > 6:
        masses[6] = 0
    cpu = ParticleData(
        masses=np.tile(masses, (boxes, 1, 1)),
        concentration=np.tile(counts, (boxes, 1)),
        charge=np.tile(charge, (boxes, 1)),
        density=np.array([1000.0, 1500.0]),
        volume=np.resize(np.array([0.25, 1.0, 4.0]), boxes),
        distribution_type="particle_resolved",
    )
    particles = to_warp_particle_data(cpu, device=device)
    geometry = family.removeprefix("charged_")
    charged = family.startswith("charged_")
    field = (
        wp.array([3.0, -6.0, 1.5], dtype=wp.float64, device=device)
        if charged and geometry == "rectangular"
        else (3.0 if charged else 0.0)
    )
    config = NeutralWallLossConfig(
        geometry,
        1e-3,
        chamber_radius=0.75 if geometry == "spherical" else None,
        chamber_dimensions=(1.2, 0.8, 0.6)
        if geometry == "rectangular"
        else None,
        mode="charged" if charged else "neutral",
        wall_potential=2.0 if charged else 0.0,
        wall_electric_field=field,
    )
    rng = wp.full(boxes, 123, dtype=wp.uint32, device=device)
    return SimpleNamespace(
        p=particles,
        cpu=cpu,
        rng=rng,
        config=config,
        kwargs=dict(
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            config=config,
            rng_seed=42,
            rng_states=rng,
            initialize_rng=True,
        ),
    )


def _snapshot(state):
    result = {
        name: getattr(state.p, name).numpy().copy()
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    result["rng"] = state.rng.numpy().copy()
    if hasattr(state.config.wall_electric_field, "numpy"):
        result["field"] = state.config.wall_electric_field.numpy().copy()
    return result


def _call(state, mode):
    from particula.gpu.kernels.wall_loss import (
        _enqueue_prepared_wall_loss_call,
        _prepare_wall_loss_step_gpu,
        wall_loss_step_gpu,
    )

    if mode == "direct":
        return wall_loss_step_gpu(state.p, **state.kwargs)
    prepared = _prepare_wall_loss_step_gpu(state.p, **state.kwargs)
    return _enqueue_prepared_wall_loss_call(prepared)


def _assert_sink(before, after):
    """Inventory accounting uses raw counts and independently integrated sums."""
    lost = (before["concentration"] > 0) & (after["concentration"] == 0)
    npt.assert_array_equal(
        after["concentration"], np.where(lost, 0, before["concentration"])
    )
    npt.assert_array_equal(
        after["masses"], np.where(lost[:, :, None], 0, before["masses"])
    )
    npt.assert_array_equal(after["charge"], np.where(lost, 0, before["charge"]))
    old_weights = before["concentration"]
    new_weights = after["concentration"]

    # Sum the independent oracle accurately: NumPy's strided species axis
    # otherwise accumulates O(n) roundoff over thousands of repeated masses.
    def inventory(masses, weights):
        return np.array(
            [
                [
                    fsum(masses[box, :, species] * weights[box])
                    for species in range(masses.shape[2])
                ]
                for box in range(masses.shape[0])
            ]
        )

    initial = inventory(before["masses"], old_weights)
    remaining = inventory(after["masses"], new_weights)
    removed = inventory(before["masses"], old_weights * lost)
    npt.assert_allclose(remaining + removed, initial, rtol=3e-15, atol=0)
    volumes = before["volume"][:, None]
    npt.assert_allclose(
        remaining / volumes + removed / volumes,
        initial / volumes,
        rtol=3e-15,
        atol=0,
    )
    initial_charge = np.sum(before["charge"] * old_weights, axis=1)
    remaining_charge = np.sum(after["charge"] * new_weights, axis=1)
    removed_charge = np.sum(before["charge"] * old_weights * lost, axis=1)
    npt.assert_allclose(
        remaining_charge + removed_charge, initial_charge, rtol=3e-15, atol=0
    )
    for name in ("density", "volume", "field"):
        if name in before:
            npt.assert_array_equal(after[name], before[name])
    return lost


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["direct", "prepared"])
def test_complete_removal_accounts_for_unequal_count_species_and_charge(
    device, family, mode
):
    """Known complete removal closes all species sinks at V=.25/1/4."""
    state = _state(device, family)
    state.kwargs["time_step"] = 1e30
    before = _snapshot(state)
    identities = {
        name: getattr(state.p, name)
        for name in before
        if hasattr(state.p, name)
    }
    assert _call(state, mode) is state.p
    lost = _assert_sink(before, _snapshot(state))
    npt.assert_array_equal(lost, before["concentration"] > 0)
    assert all(
        getattr(state.p, name) is identity
        for name, identity in identities.items()
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["direct", "prepared"])
@pytest.mark.parametrize("survival", [0.2, 0.5, 0.8])
@pytest.mark.stochastic
def test_weighted_survival_matches_independent_rate_and_binomial_variance(
    device, family, mode, survival
):
    """Independent CPU coefficient authority supplies a weighted survival law."""
    import warp as wp

    from particula.gpu.kernels.tests.wall_loss_parity_test import (
        _charged_cpu_oracle,
        _cpu_oracle,
    )

    state = _state(device, family, slots=6)
    oracle = (
        _charged_cpu_oracle if family.startswith("charged") else _cpu_oracle
    )
    sample = state.cpu.masses[:, 1:2].copy()
    # Homogeneous material and signed charge; unequal weight strata are not
    # treated as independent physical particles within a representative.
    state.p.masses = wp.array(sample, dtype=wp.float64, device=device)
    state.p.concentration = wp.ones((3, 1), dtype=wp.float64, device=device)
    state.p.charge = wp.full((3, 1), 2.0, dtype=wp.float64, device=device)
    _, coefficients = oracle(
        state.p, np.full(3, 298.15), np.full(3, 101325.0), state.config
    )
    assert np.all(coefficients > 0)
    state.kwargs["time_step"] = -np.log(survival) / coefficients[0, 0]
    slots = 4095
    weights = np.tile([0.25, 1.0, 4.0], slots // 3)
    state.p.masses = wp.array(
        np.repeat(sample, slots, axis=1), dtype=wp.float64, device=device
    )
    state.p.concentration = wp.array(
        np.tile(weights, (3, 1)), dtype=wp.float64, device=device
    )
    state.p.charge = wp.full((3, slots), 2.0, dtype=wp.float64, device=device)
    before = _snapshot(state)
    _call(state, mode)
    after = _snapshot(state)
    _assert_sink(before, after)
    for box in range(3):
        for selected in (
            np.ones(slots, dtype=bool),
            weights == 0.25,
            weights == 1,
            weights == 4,
        ):
            selected_weights = weights[selected]
            expected = survival * selected_weights.sum()
            sigma = np.sqrt(
                survival
                * (1 - survival)
                * np.dot(selected_weights, selected_weights)
            )
            observed = after["concentration"][box, selected].sum()
            # Existing coefficient parity has up to 1.002e-3 relative error
            # for spherical charged transport. Bound that approximation as
            # well as sampling variance; neither is a conservation tolerance.
            assert (
                abs(observed - expected)
                <= 6 * sigma + 0.002 * selected_weights.sum()
            )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["direct", "prepared"])
def test_persistent_stream_and_removal_match_legacy_on_same_backend(
    device, family, mode
):
    """Native admission changes neither the retained kernel nor its RNG draws."""
    import warp as wp

    from particula.gpu.kernels.wall_loss import (
        _enqueue_prepared_wall_loss_call,
        _initialize_rng_states,
        _prepare_wall_loss_step_gpu,
        wall_loss_step_gpu,
    )

    native, legacy = _state(device, family), _state(device, family)
    legacy.p.distribution_kind = 0
    prepared = None
    for state in (native, legacy):
        wp.launch(
            _initialize_rng_states,
            dim=3,
            inputs=[wp.int32(42), state.rng],
            device=device,
        )
        state.kwargs.update(initialize_rng=False, time_step=2000.0)
    if mode == "prepared":
        prepared = _prepare_wall_loss_step_gpu(native.p, **native.kwargs)
    for _ in range(3):
        before = _snapshot(native)
        if prepared is None:
            wall_loss_step_gpu(native.p, **native.kwargs)
        else:
            _enqueue_prepared_wall_loss_call(prepared)
        wall_loss_step_gpu(legacy.p, **legacy.kwargs)
        after = _snapshot(native)
        _assert_sink(before, after)
        expected = _snapshot(legacy)
        for name in after:
            npt.assert_array_equal(after[name], expected[name])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["direct", "prepared"])
@pytest.mark.parametrize("case", ["zero_time", "inactive", "empty"])
def test_noop_preserves_primary_state_and_rng(device, family, mode, case):
    """Validated zero-work calls preserve values even with reset requested."""
    import warp as wp

    state = _state(device, family, slots=0 if case == "empty" else 8)
    if case == "zero_time":
        state.kwargs["time_step"] = 0.0
    if case == "inactive":
        wp.copy(state.p.concentration, wp.zeros_like(state.p.concentration))
    before = _snapshot(state)
    assert _call(state, mode) is state.p
    after = _snapshot(state)
    for name in before:
        npt.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize("mode", ["direct", "prepared"])
@pytest.mark.parametrize(
    "case",
    [
        "mass",
        "count",
        "density",
        "charge",
        "volume",
        "zero_mass",
        "underflow_volume",
        "density_overflow",
        "sum_overflow",
    ],
)
def test_invalid_current_population_rejects_without_caller_writes(
    device, mode, case
):
    """Native physical admission precedes caller state and RNG writers."""
    import warp as wp

    state = _state(device, "spherical")
    name = {
        "mass": "masses",
        "count": "concentration",
        "zero_mass": "masses",
        "underflow_volume": "masses",
        "density_overflow": "volume",
        "sum_overflow": "masses",
    }.get(case, case)
    values = getattr(state.p, name).numpy().copy()
    if case == "zero_mass":
        values[0, 0] = 0
    elif case == "underflow_volume":
        values[0, 0] = np.nextafter(0.0, 1.0)
    elif case == "density_overflow":
        values[0] = np.nextafter(0.0, 1.0)
    elif case == "sum_overflow":
        values[0, 0] = np.finfo(float).max
    else:
        values.flat[0] = np.nan
    wp.copy(
        getattr(state.p, name),
        wp.array(values, dtype=wp.float64, device=device),
    )
    before = _snapshot(state)
    with pytest.raises(ValueError):
        _call(state, mode)
    after = _snapshot(state)
    for field in before:
        npt.assert_array_equal(after[field], before[field])


@pytest.mark.parametrize(
    "alias", ["charge", "volume", "temperature", "rng", "field"]
)
@pytest.mark.parametrize("duration", [0.0, 1.0])
def test_native_writers_cannot_overlap_primary_or_protected_inputs(
    device, alias, duration
):
    """Reject overlapping destinations even when the time step is zero."""
    import warp as wp

    state = _state(
        device, "charged_rectangular" if alias == "field" else "spherical"
    )
    protected = wp.array(
        ptr=state.p.concentration.ptr,
        dtype=wp.float64,
        shape=(3,),
        device=device,
    )
    if alias == "charge":
        state.p.charge = state.p.concentration
    elif alias == "volume":
        state.p.volume = protected
    elif alias == "temperature":
        state.kwargs["temperature"] = protected
    elif alias == "rng":
        state.rng = wp.array(
            ptr=state.p.concentration.ptr,
            dtype=wp.uint32,
            shape=(3,),
            device=device,
        )
        state.kwargs["rng_states"] = state.rng
    else:
        state.config = replace(state.config, wall_electric_field=protected)
        state.kwargs["config"] = state.config
    state.kwargs["time_step"] = duration
    before = _snapshot(state)
    with pytest.raises(ValueError, match="overlap"):
        _call(state, "direct")
    after = _snapshot(state)
    for name in before:
        npt.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_direct_arrays_and_explicit_environment_preserve_input_ownership(
    device, family, dtype_name
):
    """Per-box inputs and explicit environment normalize to the same physics."""
    import warp as wp

    from particula.gpu import WarpEnvironmentData

    direct, environment_state = _state(device, family), _state(device, family)
    dtype = getattr(wp, dtype_name)
    temperature = wp.array([280.0, 300.0, 320.0], dtype=dtype, device=device)
    pressure = wp.array(
        [90000.0, 100000.0, 110000.0], dtype=dtype, device=device
    )
    environment = WarpEnvironmentData()
    # The explicit environment carrier is fp64; direct arrays also accept
    # fp32 and normalize locally. These integer-valued inputs are exact in both.
    environment.temperature = wp.array(
        [280.0, 300.0, 320.0], dtype=wp.float64, device=device
    )
    environment.pressure = wp.array(
        [90000.0, 100000.0, 110000.0], dtype=wp.float64, device=device
    )
    environment.saturation_ratio = wp.full(
        (3, 2), 0.5, dtype=wp.float64, device=device
    )
    inputs = (
        temperature,
        pressure,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )
    snapshots = [array.numpy().copy() for array in inputs]
    direct.kwargs.update(
        temperature=temperature, pressure=pressure, time_step=2000.0
    )
    environment_state.kwargs.update(
        temperature=None,
        pressure=None,
        environment=environment,
        time_step=2000.0,
    )
    _call(direct, "direct")
    _call(environment_state, "prepared")
    for name, values in _snapshot(direct).items():
        npt.assert_array_equal(_snapshot(environment_state)[name], values)
    for array, before in zip(
        inputs,
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(array.numpy(), before)
    assert (
        environment.temperature is inputs[2]
        and environment.pressure is inputs[3]
    )


def test_environment_inspection_storage_cannot_alias_native_writer(device):
    """Unread environment fields remain protected from primary array writers."""
    import warp as wp

    from particula.gpu import WarpEnvironmentData

    state = _state(device, "spherical")
    environment = WarpEnvironmentData()
    environment.temperature = wp.full(
        3, 298.15, dtype=wp.float64, device=device
    )
    environment.pressure = wp.full(3, 101325.0, dtype=wp.float64, device=device)
    environment.saturation_ratio = state.p.charge
    state.kwargs.update(
        temperature=None, pressure=None, environment=environment
    )
    before = _snapshot(state)
    with pytest.raises(ValueError, match="overlap"):
        _call(state, "prepared")
    for name, values in _snapshot(state).items():
        npt.assert_array_equal(values, before[name])


def test_noncontiguous_native_storage_rejects_before_writer(device):
    """The native ownership check rejects unsupported stride layouts."""
    import warp as wp

    state = _state(device, "spherical")
    state.p.charge = wp.ones(
        (8, 3), dtype=wp.float64, device=device
    ).transpose()
    before = _snapshot(state)
    with pytest.raises(ValueError, match="contiguous"):
        _call(state, "direct")
    for name, values in _snapshot(state).items():
        npt.assert_array_equal(values, before[name])


@pytest.mark.parametrize(
    "field", ["masses", "concentration", "charge", "density", "volume"]
)
def test_native_primary_schema_rejects_float32_storage(device, field):
    """The population tag does not expand the direct fp64 storage contract."""
    import warp as wp

    state = _state(device, "spherical")
    # Warp's typed struct rejects invalid dtype assignment itself. Exercise
    # the direct boundary's WarpParticleData-like admission with a plain view.
    from types import SimpleNamespace

    state.p = SimpleNamespace(
        **{
            name: getattr(state.p, name)
            for name in (
                "masses",
                "concentration",
                "charge",
                "density",
                "volume",
                "distribution_kind",
            )
        }
    )
    setattr(
        state.p,
        field,
        wp.array(
            getattr(state.p, field).numpy(), dtype=wp.float32, device=device
        ),
    )
    before = _snapshot(state)
    with pytest.raises(ValueError, match="float64"):
        _call(state, "direct")
    for name, values in _snapshot(state).items():
        npt.assert_array_equal(values, before[name])


@pytest.mark.parametrize("family", FAMILIES)
def test_prepared_references_are_frozen_and_enqueue_has_no_setup(
    device, family, monkeypatch
):
    """Only the frozen launch may execute after preparation."""
    import warp as wp

    from particula.gpu.kernels import _wall_loss_native, wall_loss

    state = _state(device, family)
    prepared = wall_loss._prepare_wall_loss_step_gpu(state.p, **state.kwargs)
    arrays = {
        name: getattr(state.p, name)
        for name in ("masses", "concentration", "charge")
    }
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("prepared enqueue performed host setup or readback")

    def launch(kernel, *, dim, inputs, device):
        assert kernel is prepared.launch_kernel
        assert dim == prepared.launch_dim
        assert inputs is prepared.launch_inputs
        assert device is prepared.device
        for offset, name in enumerate(arrays):
            assert inputs[offset] is arrays[name]
        calls.append(kernel)

    for name in arrays:
        setattr(state.p, name, None)
    state.p.distribution_kind = 2
    for name in (
        "_validate_particle_values",
        "_validate_particle_schema",
        "_ensure_environment_arrays",
        "_normalize_execution_environment_array",
        "_validate_rng",
    ):
        monkeypatch.setattr(wall_loss, name, forbidden)
    for name in ("validate_physics", "validate_ownership"):
        monkeypatch.setattr(_wall_loss_native, name, forbidden)
    for name in (
        "array",
        "zeros",
        "empty",
        "synchronize",
        "synchronize_device",
    ):
        monkeypatch.setattr(wp, name, forbidden)
    monkeypatch.setattr(wp, "launch", launch)
    assert wall_loss._enqueue_prepared_wall_loss_call(prepared) is state.p
    assert calls == [prepared.launch_kernel]


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_reset_flag_remains_frozen_in_prepared_call(device, family):
    """An explicitly reset prepared call repeats that request on each launch."""
    import warp as wp

    from particula.gpu.kernels.wall_loss import (
        _enqueue_prepared_wall_loss_call,
        _prepare_wall_loss_step_gpu,
    )

    state = _state(device, family)
    state.kwargs["time_step"] = 2000.0
    prepared = _prepare_wall_loss_step_gpu(state.p, **state.kwargs)
    original = _snapshot(state)
    _enqueue_prepared_wall_loss_call(prepared)
    first = _snapshot(state)
    for name in ("masses", "concentration", "charge"):
        wp.copy(
            getattr(state.p, name),
            wp.array(original[name], dtype=wp.float64, device=device),
        )
    # Retain advanced RNG words: the frozen explicit reset must override them.
    _enqueue_prepared_wall_loss_call(prepared)
    second = _snapshot(state)
    for name in first:
        npt.assert_array_equal(second[name], first[name])
