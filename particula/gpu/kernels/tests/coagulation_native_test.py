"""Native GPU count normalization, conservative merges and prepared ownership."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from particula.gpu.kernels.tests._coagulation_validation_support import (
    EXECUTABLE_ROWS,
    Fixture,
    pair_rate,
)
from particula.particles import ParticleData

pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Require installed Warp CPU and report optional CUDA availability."""
    wp = pytest.importorskip("warp")
    from particula.gpu.tests.cuda_availability import cuda_available

    if request.param == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA unavailable")
    return request.param


def _state(device, row, weight=1.0, slots=5):
    """Use multiple materials, three physical volumes and an inactive sentinel."""
    import warp as wp

    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import CoagulationMechanismConfig

    base = (
        np.array(
            [
                [1.0, 0.0, 2.0],
                [7.0, 4.0, 3.0],
                [4.0, 3.0, 6.0],
                [17.0, 5.0, 10.0],
                [64.0, 30.0, 20.0],
            ]
        )
        * 1e-18
    )
    masses = np.tile(base[:slots], (3, 1, 1))
    counts = np.full((3, slots), weight)
    if slots > 1:
        counts[:, 1] = 0
    charge = np.tile(np.array([1.0, 11.0, -1.0, 2.0, 3.0])[:slots], (3, 1))
    cpu = ParticleData(
        masses=masses,
        concentration=counts,
        charge=charge,
        density=np.array([1000.0, 1500.0, 1800.0]),
        volume=np.array([0.25, 1.0, 4.0]),
        distribution_type="particle_resolved",
    )
    particles = to_warp_particle_data(cpu, device=device)
    pairs = wp.full((3, 2, 2), -31, dtype=wp.int32, device=device)
    collisions = wp.full(3, -17, dtype=wp.int32, device=device)
    rng = wp.full(3, 123, dtype=wp.uint32, device=device)
    kwargs = dict(
        temperature=298.15,
        pressure=101325.0,
        time_step=1e30,
        collision_pairs=pairs,
        n_collisions=collisions,
        rng_states=rng,
        max_collisions=2,
        rng_seed=42,
        initialize_rng=True,
        mechanism_config=CoagulationMechanismConfig(row.mechanisms),
    )
    if row.mask & 8:
        kwargs.update(turbulent_dissipation=2e-4, fluid_density=1.2)
    return SimpleNamespace(
        p=particles,
        cpu=cpu,
        pairs=pairs,
        counts=collisions,
        rng=rng,
        kwargs=kwargs,
    )


def _snapshot(state):
    """Read state only at the explicit test observation boundary."""
    return {
        **{
            name: getattr(state.p, name).numpy().copy()
            for name in (
                "masses",
                "concentration",
                "charge",
                "density",
                "volume",
            )
        },
        "pairs": state.pairs.numpy().copy(),
        "counts": state.counts.numpy().copy(),
        "rng": state.rng.numpy().copy(),
    }


def _inventory(snapshot):
    """Compute per-box species/charge inventories without container helpers."""
    weights = snapshot["concentration"]
    return (
        np.sum(snapshot["masses"] * weights[:, :, None], axis=1),
        np.sum(snapshot["charge"] * weights, axis=1),
    )


@pytest.mark.parametrize("row", EXECUTABLE_ROWS, ids=lambda row: str(row.mask))
@pytest.mark.parametrize("weight", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("mode", ["direct", "prepared"])
def test_native_merges_conserve_all_species_and_charge(  # noqa: C901
    device, row, weight, mode
):
    """Every supported mask conserves raw inventories across three volumes."""
    from particula.gpu.kernels.coagulation import (
        _enqueue_prepared_coagulation_call,
        _prepare_coagulation_step_gpu,
        coagulation_step_gpu,
    )

    state = _state(device, row, weight)
    initial = _snapshot(state)
    inventory = _inventory(initial)
    identities = {
        name: getattr(state.p, name)
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    prepared = None
    if mode == "prepared":
        prepared = _prepare_coagulation_step_gpu(state.p, **state.kwargs)
        assert prepared.native_counts
    for replay, expected_collisions in enumerate((2, 1, 0)):
        before = _snapshot(state)
        if mode == "direct":
            state.kwargs["initialize_rng"] = replay == 0
            result = coagulation_step_gpu(state.p, **state.kwargs)
        else:
            result = _enqueue_prepared_coagulation_call(prepared)
        assert all(
            a is b
            for a, b in zip(
                result, (state.p, state.pairs, state.counts), strict=True
            )
        )
        after = _snapshot(state)
        npt.assert_array_equal(after["counts"], expected_collisions)
        for actual, expected in zip(_inventory(after), inventory, strict=True):
            npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-30)
        for name, original in identities.items():
            assert getattr(state.p, name) is original
        for name in ("density", "volume"):
            npt.assert_array_equal(after[name], initial[name])
        for name in ("masses", "concentration", "charge"):
            npt.assert_array_equal(after[name][:, 1], initial[name][:, 1])
        for box, count in enumerate(after["counts"]):
            pairs = after["pairs"][box, :count]
            assert np.unique(pairs).size == 2 * count
            for recipient, donor in pairs:
                npt.assert_allclose(
                    after["masses"][box, recipient],
                    before["masses"][box, recipient]
                    + before["masses"][box, donor],
                    rtol=1e-12,
                    atol=1e-30,
                )
                npt.assert_array_equal(after["masses"][box, donor], 0)
                assert after["concentration"][box, donor] == 0
                assert after["concentration"][box, recipient] == weight
        if expected_collisions == 0:
            npt.assert_array_equal(after["rng"], before["rng"])


@pytest.mark.parametrize("row", EXECUTABLE_ROWS, ids=lambda row: str(row.mask))
@pytest.mark.parametrize("mode", ["direct", "prepared"])
def test_native_scheduling_matches_independent_pair_hazard(device, row, mode):
    """One-pair Bernoulli observations establish N/V for every mechanism."""
    import warp as wp

    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        _enqueue_prepared_coagulation_call,
        _prepare_coagulation_step_gpu,
        coagulation_step_gpu,
    )

    fixture = Fixture(
        name="native_pair",
        radii=np.array([[1e-7, 5e-7]]),
        density=np.full((1, 2), 1000.0),
        charges=np.array([[1.0, -1.0]]),
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        dissipation=np.array([2e-4]),
        fluid_density=np.array([1.2]),
        active=((0, 1),),
    )
    rate = pair_rate(fixture, 0, 0, 1, row.mask)
    assert rate > 0
    repetitions = 1024
    volume_groups = np.repeat([0.25, 1.0, 4.0], 3)
    weight_groups = np.tile([0.25, 1.0, 4.0], 3)
    volumes = np.repeat(volume_groups, repetitions)
    weights = np.repeat(weight_groups, repetitions)
    boxes = volumes.size
    material = fixture.masses[0, :, None] * np.array([0.4, 0.6])
    cpu = ParticleData(
        masses=np.tile(material, (boxes, 1, 1)),
        concentration=np.repeat(weights[:, None], 2, axis=1),
        charge=np.tile(fixture.charges, (boxes, 1)),
        density=np.full(2, 1000.0),
        volume=volumes,
        distribution_type="particle_resolved",
    )
    particles = to_warp_particle_data(cpu, device=device)
    kwargs = dict(mechanism_config=CoagulationMechanismConfig(row.mechanisms))
    if row.mask & 8:
        kwargs.update(turbulent_dissipation=2e-4, fluid_density=1.2)
    if mode == "direct":
        _, _, counts = coagulation_step_gpu(
            particles,
            298.15,
            101325.0,
            0.05 / rate,
            max_collisions=1,
            rng_seed=873,
            **kwargs,
        )
    else:
        # Current counts are dynamic device state, not host-cached weights.
        particles.concentration.assign(np.ones_like(cpu.concentration))
        prepared = _prepare_coagulation_step_gpu(
            particles,
            298.15,
            101325.0,
            0.05 / rate,
            max_collisions=1,
            rng_seed=873,
            **kwargs,
        )
        particles.concentration.assign(cpu.concentration)
        _, _, counts = _enqueue_prepared_coagulation_call(prepared)
    observed = counts.numpy().reshape(9, repetitions)
    assert np.all((observed == 0) | (observed == 1))
    expected_probability = 0.05 * weight_groups / volume_groups
    means = observed.mean(axis=1)
    standard_error = np.sqrt(
        expected_probability * (1 - expected_probability) / repetitions
    )
    assert np.all(
        np.abs(means - expected_probability)
        < 6 * standard_error + 1 / repetitions
    )
    # Species conservation is separately checked even for zero-event boxes.
    final = (
        particles.masses.numpy() * particles.concentration.numpy()[:, :, None]
    )
    original = cpu.masses * cpu.concentration[:, :, None]
    npt.assert_allclose(
        final.sum(axis=1), original.sum(axis=1), rtol=1e-12, atol=1e-30
    )
    assert wp.get_device(device) == particles.masses.device


@pytest.mark.parametrize("row", EXECUTABLE_ROWS, ids=lambda row: str(row.mask))
def test_unequal_counts_reject_without_mutation(device, row):
    """Unsupported weights cannot reach RNG initialization or any writer."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, row)
    counts = state.p.concentration.numpy()
    counts[0, 0] = 2
    state.p.concentration.assign(counts)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="equal|concentrations"):
        coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    for name in before:
        npt.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize(
    "field,value",
    [
        ("masses", -1.0),
        ("masses", np.nan),
        ("masses", np.inf),
        ("concentration", -1.0),
        ("concentration", np.nan),
        ("charge", np.inf),
        ("density", 0.0),
        ("density", np.nan),
        ("volume", 0.0),
        ("volume", np.inf),
    ],
)
def test_bad_native_state_rejects_even_zero_time(device, field, value):
    """Directly mutated native physical fields fail before caller writes."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    values = getattr(state.p, field).numpy()
    values.flat[0] = value
    getattr(state.p, field).assign(values)
    state.kwargs["time_step"] = 0
    before = _snapshot(state)
    with pytest.raises(ValueError):
        coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    for name in before:
        npt.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize(
    "target", ["charge", "pairs", "counts", "rng", "temperature"]
)
def test_writers_cannot_alias_primary_or_protected_arrays(device, target):
    """Aliases reject even when each view separately has valid metadata."""
    import warp as wp

    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    if target == "charge":
        state.p.charge = state.p.concentration
    elif target == "temperature":
        # Positive charge values can simultaneously look like temperatures.
        state.p.charge = wp.full(
            (3, 5), 298.15, dtype=wp.float64, device=device
        )
        state.kwargs["temperature"] = wp.array(
            ptr=state.p.charge.ptr, shape=(3,), dtype=wp.float64, device=device
        )
    elif target == "pairs":
        state.pairs = wp.array(
            ptr=state.p.masses.ptr,
            shape=(3, 2, 2),
            dtype=wp.int32,
            device=device,
        )
        state.kwargs["collision_pairs"] = state.pairs
    else:
        array = wp.array(
            ptr=state.p.masses.ptr,
            shape=(3,),
            dtype=wp.uint32 if target == "rng" else wp.int32,
            device=device,
        )
        if target == "rng":
            state.rng = array
            state.kwargs["rng_states"] = array
        else:
            state.counts = array
            state.kwargs["n_collisions"] = array
    before = _snapshot(state)
    with pytest.raises(ValueError, match="overlap"):
        coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    for name in before:
        npt.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize("form", ["scalar", "array"])
def test_explicit_volume_must_match_population_authority(device, form):
    """A separate direct volume cannot silently reinterpret native counts."""
    import warp as wp

    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    state.kwargs["volume"] = (
        1.0
        if form == "scalar"
        else wp.array(
            np.array([0.25, 1.0, 2.0]), dtype=wp.float64, device=device
        )
    )
    before = _snapshot(state)
    with pytest.raises(ValueError, match="match positive particle volume"):
        coagulation_step_gpu(state.p, **state.kwargs)
    for name, value in _snapshot(state).items():
        npt.assert_array_equal(value, before[name])


@pytest.mark.parametrize("slots", [0, 1])
def test_empty_and_one_active_preserve_native_rng_and_particles(device, slots):
    """No-work calls validate then clear counts without advancing the stream."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0], slots=slots)
    state.kwargs["initialize_rng"] = False
    before = _snapshot(state)
    coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    for name in (
        "masses",
        "concentration",
        "charge",
        "density",
        "volume",
        "rng",
    ):
        npt.assert_array_equal(after[name], before[name])
    npt.assert_array_equal(after["counts"], 0)


@pytest.mark.parametrize("row", EXECUTABLE_ROWS, ids=lambda row: str(row.mask))
def test_prepared_enqueue_has_no_host_rebinding_or_resource_setup(
    device, row, monkeypatch
):
    """Prepared replay uses pinned arrays without readback, allocation or reset."""
    import warp as wp

    from particula.gpu.kernels import coagulation as module

    state = _state(device, row)
    prepared = module._prepare_coagulation_step_gpu(state.p, **state.kwargs)
    # Warm compilation before asserting the enqueue boundary.
    module._enqueue_prepared_coagulation_call(prepared)
    wp.synchronize_device(state.p.masses.device)
    pinned_mass = state.p.masses
    state.p.masses = wp.clone(pinned_mass)
    replacement = state.p.masses.numpy().copy()
    state.p.distribution_kind = 1

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "prepared enqueue performed host setup or readback"
        )

    with monkeypatch.context() as patch:
        for name in (
            "zeros",
            "empty",
            "full",
            "clone",
            "synchronize",
            "synchronize_device",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        patch.setattr(module, "initialize_coagulation_rng_states", forbidden)
        patch.setattr(module, "particle_distribution_code", forbidden)
        module._enqueue_prepared_coagulation_call(prepared)
    npt.assert_array_equal(state.p.masses.numpy(), replacement)
    assert np.count_nonzero(
        np.sum(pinned_mass.numpy(), axis=2) > 0
    ) < np.count_nonzero(np.sum(replacement, axis=2) > 0)


@pytest.mark.parametrize("row", EXECUTABLE_ROWS, ids=lambda row: str(row.mask))
def test_zero_time_preserves_primary_state_and_rng(device, row):
    """Zero time performs complete native admission but no stochastic draws."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, row, 0.25)
    state.kwargs.update(time_step=0.0, initialize_rng=False)
    before = _snapshot(state)
    coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    for name in (
        "masses",
        "concentration",
        "charge",
        "density",
        "volume",
        "rng",
    ):
        npt.assert_array_equal(after[name], before[name])
    npt.assert_array_equal(after["counts"], 0)


@pytest.mark.parametrize(
    "field", ["masses", "concentration", "charge", "density", "volume"]
)
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_native_primary_schema_rejects_before_any_payload_launch(
    device, field, fault, monkeypatch
):
    """Malformed native layouts reject with a clear error before device work."""
    import warp as wp

    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    fields = {
        name: getattr(state.p, name)
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    original = fields[field]
    fields[field] = wp.zeros(
        original.shape if fault == "dtype" else (2,),
        dtype=wp.float32 if fault == "dtype" else wp.float64,
        device=device,
    )
    malformed = SimpleNamespace(**fields, distribution_kind=3)

    def forbidden(*args, **kwargs):
        pytest.fail("schema rejection reached a device launch")

    monkeypatch.setattr(wp, "launch", forbidden)
    with pytest.raises(ValueError, match="native coagulation"):
        coagulation_step_gpu(malformed, **state.kwargs)


@pytest.mark.parametrize("form", ["array", "scalar"])
def test_matching_explicit_volume_is_supported(device, form):
    """An explicit equal volume retains the existing direct call option."""
    import warp as wp

    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    if form == "scalar":
        state.p.volume.assign(np.full(3, 0.25))
        state.kwargs["volume"] = 0.25
    else:
        state.kwargs["volume"] = wp.clone(state.p.volume)
    before = _snapshot(state)
    coagulation_step_gpu(state.p, **state.kwargs)
    after = _snapshot(state)
    npt.assert_array_equal(after["counts"], 2)
    for expected, actual in zip(
        _inventory(before), _inventory(after), strict=True
    ):
        npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-30)


def test_occupied_zero_material_rejects_and_inactive_zero_material_is_valid(
    device,
):
    """Positive native counts require physical material before scheduling."""
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[0])
    masses = state.p.masses.numpy()
    masses[:, 0] = 0
    state.p.masses.assign(masses)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="physical state"):
        coagulation_step_gpu(state.p, **state.kwargs)
    for name, values in _snapshot(state).items():
        npt.assert_array_equal(values, before[name])
    counts = state.p.concentration.numpy()
    counts[:, 0] = 0
    state.p.concentration.assign(counts)
    coagulation_step_gpu(state.p, **state.kwargs)
    npt.assert_array_equal(state.counts.numpy(), 1)


@pytest.mark.parametrize(
    "mechanisms",
    [
        ("brownian", "charged_hard_sphere", "sedimentation_sp2016"),
        ("brownian", "charged_hard_sphere", "turbulent_shear_st1956"),
        ("brownian", "sedimentation_sp2016", "turbulent_shear_st1956"),
        (
            "charged_hard_sphere",
            "sedimentation_sp2016",
            "turbulent_shear_st1956",
        ),
        ("turbulent_dns",),
    ],
)
def test_native_metadata_does_not_expand_mechanism_support(device, mechanisms):
    """Native provenance does not admit deferred triples or a GPU DNS law."""
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        coagulation_step_gpu,
    )

    state = _state(device, EXECUTABLE_ROWS[-1])
    state.kwargs["mechanism_config"] = CoagulationMechanismConfig(mechanisms)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="deferred|Unknown"):
        coagulation_step_gpu(state.p, **state.kwargs)
    for name, values in _snapshot(state).items():
        npt.assert_array_equal(values, before[name])


@pytest.mark.parametrize("form", ["arrays", "environment"])
def test_native_environment_sources_preserve_physical_state(device, form):
    """Direct arrays and explicit environment use the same admitted native path."""
    import warp as wp

    from particula.gas import EnvironmentData
    from particula.gpu import from_warp_particle_data, to_warp_environment_data
    from particula.gpu.kernels.coagulation import coagulation_step_gpu

    state = _state(device, EXECUTABLE_ROWS[-1], 0.25)
    if form == "arrays":
        state.kwargs.update(
            temperature=wp.full(3, 298.15, dtype=wp.float64, device=device),
            pressure=wp.full(3, 101325.0, dtype=wp.float64, device=device),
        )
    else:
        environment = to_warp_environment_data(
            EnvironmentData(
                temperature=np.full(3, 298.15),
                pressure=np.full(3, 101325.0),
                saturation_ratio=np.full((3, 3), 0.5),
            ),
            device=device,
        )
        state.kwargs.update(
            temperature=None, pressure=None, environment=environment
        )
    before = _snapshot(state)
    coagulation_step_gpu(state.p, **state.kwargs)
    restored = from_warp_particle_data(state.p)
    assert restored.distribution_type == "particle_resolved"
    npt.assert_array_equal(restored.volume, [0.25, 1.0, 4.0])
    after = _snapshot(state)
    npt.assert_array_equal(restored.concentration, after["concentration"])
    for expected, actual in zip(
        _inventory(before), _inventory(after), strict=True
    ):
        npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-30)
    if form == "environment":
        npt.assert_array_equal(environment.temperature.numpy(), 298.15)
        npt.assert_array_equal(environment.pressure.numpy(), 101325.0)
        npt.assert_array_equal(environment.saturation_ratio.numpy(), 0.5)


@pytest.mark.parametrize("mask", [1, 2, 4, 8])
def test_native_admission_preserves_mechanism_specific_active_limits(
    device, mask
):
    """The shared physical scan does not impose charged caps on other laws."""
    from particula.gpu import to_warp_particle_data
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        coagulation_step_gpu,
    )

    count = 257
    cpu = ParticleData(
        masses=np.full((1, count, 2), 1e-18),
        concentration=np.ones((1, count)),
        charge=np.zeros((1, count)),
        density=np.full(2, 1000.0),
        volume=np.ones(1),
        distribution_type="particle_resolved",
    )
    particles = to_warp_particle_data(cpu, device=device)
    row = next(row for row in EXECUTABLE_ROWS if row.mask == mask)
    kwargs = dict(mechanism_config=CoagulationMechanismConfig(row.mechanisms))
    if mask == 8:
        kwargs.update(turbulent_dissipation=2e-4, fluid_density=1.2)
    if mask in (2, 4):
        with pytest.raises(ValueError, match="active particle count exceeds"):
            coagulation_step_gpu(particles, 298.15, 101325.0, 0.0, **kwargs)
    else:
        _, _, collisions = coagulation_step_gpu(
            particles, 298.15, 101325.0, 0.0, **kwargs
        )
        npt.assert_array_equal(collisions.numpy(), 0)
    npt.assert_array_equal(particles.masses.numpy(), cpu.masses)
    npt.assert_array_equal(particles.concentration.numpy(), cpu.concentration)
