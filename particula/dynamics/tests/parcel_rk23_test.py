"""Independent physical, controller and derivative checks for adaptive parcels."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel import (
    CP,
    LATENT,
    RD,
    ParcelInitialState,
    ParcelRK23,
    ParcelROS3,
    run_parcel,
)


@pytest.fixture(params=["warp:cpu", "cuda:0"])
def device(request):
    """Exercise the same program on Warp CPU and native CUDA."""
    import warp as wp

    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("requires an NVIDIA CUDA device")
    return request.param


@pytest.fixture
def parcel():
    """Start close enough to cloud base for a short mixed-activation run."""
    return ParcelInitialState(
        dry_radius=np.array([45.0, 70.0, 110.0]) * 1e-9,
        number=np.array([2.5e7, 5e7, 2.5e7]),
        kappa=0.5,
        temperature=288.15,
        pressure=90000.0,
        vapor=0.0118,
    )


@pytest.mark.parametrize("method", [ParcelRK23, ParcelROS3])
def test_adaptive_ascent_and_reverse(parcel, device, method):
    """Refinement and independently adapted perturbations at fixed time."""
    options = dict(timestep=0.5, steps=200)

    def advance(speed, tolerance, gradient=False):
        return run_parcel(
            parcel,
            updraft=speed,
            device=device,
            **options,
            gradient=gradient,
            adaptive=method(rtol=tolerance),
        )

    ref = run_parcel(parcel, updraft=1.0, **options)
    loose = advance(1.0, 1e-3)
    tight = advance(1.0, 1e-5, True)
    assert ref.statistics is None
    np.testing.assert_array_equal(tight.time, ref.time)
    stats = tight.statistics
    assert stats.accepted_steps > options["steps"] and stats.rejected_steps > 0
    assert 0 < stats.minimum_step < stats.maximum_step <= options["timestep"]
    for field in ("wet_radius", "supersaturation"):
        exact = getattr(ref, field)
        assert np.max(abs(getattr(tight, field) - exact)) < np.max(
            abs(getattr(loose, field) - exact)
        )
    assert np.max(abs(tight.wet_radius - ref.wet_radius)) < 1e-9
    assert abs(tight.supersaturation.max() - ref.supersaturation.max()) < 2e-6
    for field, atol in (
        ("height", 1e-10),
        ("temperature", 1e-5),
        ("pressure", 5e-4),
        ("volume", 1e-7),
        ("liquid", 2e-9),
        ("vapor", 2e-9),
    ):
        np.testing.assert_allclose(
            getattr(tight, field), getattr(ref, field), atol=atol, rtol=0
        )
    # Reconstruct liquid independently from the public wet-radius history.
    inventory = (
        (4 * np.pi / 3 * 1000)
        * (tight.wet_radius**3 - parcel.dry_radius**3)
        @ parcel.number
        / parcel.dry_air_mass
    )
    np.testing.assert_allclose(inventory, tight.liquid, atol=1e-18, rtol=1e-13)
    np.testing.assert_allclose(
        tight.vapor + inventory,
        ref.vapor[0] + ref.liquid[0],
        atol=5e-17,
        rtol=0,
    )
    alpha = tight.volume / parcel.dry_air_mass
    work = np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(tight.pressure))
    energy = CP * (tight.temperature[1:] - parcel.temperature) - LATENT * (
        inventory[1:] - inventory[0]
    )
    np.testing.assert_allclose(energy, work, atol=1e-5, rtol=0)
    # Perturbed calls select their own mesh and can cross controller branches.
    assert tight.updraft_gradient > 0
    for delta in (1e-3, 1e-4):
        plus = advance(1 + delta, 1e-5)
        minus = advance(1 - delta, 1e-5)
        fd = (plus.liquid[-1] - minus.liquid[-1]) / (2 * delta)
        np.testing.assert_allclose(tight.updraft_gradient, fd, rtol=2e-5)


@pytest.mark.parametrize("method", [ParcelRK23, ParcelROS3])
def test_passive_dry_parcel_on_decimal_observation_grid(parcel, device, method):
    """No derivative or artificial short tail on the dry, passive trajectory."""
    result = run_parcel(
        replace(parcel, vapor=0.0),
        timestep=0.1,
        steps=33,
        updraft=1.0,
        adaptive=method(),
        device=device,
        gradient=True,
        phase_exchange=False,
    )
    assert result.statistics.accepted_steps == 33
    assert result.statistics.rejected_steps == 0
    assert result.statistics.minimum_step == pytest.approx(0.1)
    np.testing.assert_array_equal(result.liquid, np.zeros(34))
    np.testing.assert_array_equal(result.vapor, np.zeros(34))
    theta = result.temperature * (parcel.pressure / result.pressure) ** (
        RD / CP
    )
    np.testing.assert_allclose(theta, parcel.temperature, rtol=1e-12)
    assert result.updraft_gradient == 0.0


@pytest.mark.parametrize("method", [ParcelRK23, ParcelROS3])
def test_stationary_moist_parcel_sensitivity(parcel, device, method):
    """An equilibrated forward state still has a fast tangent relaxation."""
    cadence = dict(timestep=0.5, steps=4)
    controls = dict(device=device, adaptive=method())
    rest = run_parcel(parcel, updraft=0.0, gradient=True, **cadence, **controls)
    np.testing.assert_allclose(
        rest.liquid, rest.liquid[0], rtol=2e-13, atol=0.0
    )
    if method is ParcelRK23:
        assert rest.statistics.maximum_step < cadence["timestep"]
    else:
        assert rest.statistics.accepted_steps == cadence["steps"]
    delta = 0.001
    for backend in (controls, {"device": "cpu"}):
        plus = run_parcel(parcel, updraft=delta, **cadence, **backend)
        minus = run_parcel(parcel, updraft=-delta, **cadence, **backend)
        fd = (plus.liquid[-1] - minus.liquid[-1]) / (2 * delta)
        assert fd > 0.0
        np.testing.assert_allclose(
            rest.updraft_gradient, fd, rtol=2e-5, atol=0.0
        )


@pytest.mark.parametrize("method", [ParcelRK23, ParcelROS3])
def test_trial_failures_are_bounded(parcel, device, method):
    """An unusable step raises at the floor or cap without changing inputs."""
    initial = parcel.number.copy()
    for controls, message in [
        (method(max_trials=1), "max_trials"),
        (method(min_step=1.0, rtol=1e-12), "tolerance"),
    ]:
        with pytest.raises(ValueError, match=message):
            run_parcel(
                parcel,
                timestep=1.0,
                steps=2,
                updraft=1.0,
                device=device,
                adaptive=controls,
            )
    np.testing.assert_array_equal(parcel.number, initial)
    descending = run_parcel(
        parcel,
        timestep=0.25,
        steps=12,
        updraft=-1.0,
        device=device,
        adaptive=method(),
    )
    assert descending.liquid[-1] < descending.liquid[0]
    assert descending.vapor[-1] > descending.vapor[0]


@pytest.mark.parametrize(
    "invalid",
    [
        {"rtol": 0.0},
        {"rtol": np.nan},
        {"min_step": np.inf},
        {"min_step": 0.0},
        {"max_trials": True},
        {"max_trials": 1.25},
        {"max_trials": 0},
    ],
)
def test_reject_invalid_controls(invalid):
    """Invalid tolerance, timestep or trial budget fails at admission."""
    with pytest.raises(ValueError):
        ParcelRK23(**invalid)


def test_reference_and_fixed_controls(parcel):
    """Adaptive selection is explicit and cannot be mistaken for Radau."""
    options = dict(updraft=1.0, timestep=1.0, steps=1)
    with pytest.raises(ValueError, match="requires a Warp device"):
        run_parcel(parcel, **options, adaptive=ParcelRK23())
    with pytest.raises(TypeError, match="ParcelRK23"):
        run_parcel(parcel, **options, adaptive=True)


def test_batched_trials_keep_the_serial_mesh(parcel, device, monkeypatch):
    """Discard speculative states and adjoints after each controller change."""
    from particula.gpu.kernels import parcel_rk23

    case = replace(parcel, vapor=0.01205)
    options = dict(
        updraft=1.0,
        timestep=1.0,
        steps=10,
        device=device,
        gradient=True,
        adaptive=ParcelRK23(),
    )
    batched = run_parcel(case, **options)
    monkeypatch.setattr(parcel_rk23, "_BATCH_CAPACITY", 1)
    serial = run_parcel(case, **options)
    work = batched.statistics
    assert work.evaluated_trials > work.accepted_steps + work.rejected_steps
    assert (
        serial.statistics.evaluated_trials
        == work.accepted_steps + work.rejected_steps
    )
    for name in (
        "accepted_steps",
        "rejected_steps",
        "minimum_step",
        "maximum_step",
    ):
        assert getattr(work, name) == getattr(serial.statistics, name)
    for name in (
        "height",
        "pressure",
        "temperature",
        "volume",
        "vapor",
        "liquid",
        "wet_radius",
        "supersaturation",
    ):
        np.testing.assert_array_equal(
            getattr(batched, name), getattr(serial, name)
        )
    np.testing.assert_allclose(
        batched.updraft_gradient, serial.updraft_gradient, rtol=1e-13, atol=0.0
    )


def test_ros3_stage_solve_and_forcing_derivative(parcel, device):
    """Compare a coupled ROS3 solve to a dense numerical rate Jacobian."""
    from particula.dynamics.parcel import _initial_radii
    from particula.execution._ros3_coefficients import GAMMA
    from particula.gpu.kernels.parcel_ros3 import (
        _back_substitute,
        _ROS3Stepper,
        _water_increment,
    )

    # Unequal loadings and a zero-count tracer exercise both directions of
    # coupling. Move off equilibrium to include derivatives of resistance.
    parcel = replace(parcel, number=np.array([2e9, 0.0, 5e8]))
    backend = "cpu" if device == "warp:cpu" else device
    stepper = _ROS3Stepper(
        parcel, _initial_radii(parcel), 1.3, backend, False, True
    )
    y = stepper.initial_state.numpy() * np.array([0.998, 1.4, 0.6, 2.0])
    t, h = 13.0, 0.09
    stages = stepper.scratch()
    state = stepper.array(y)
    stepper.advance(state, t, h, stepper.empty(4), stages)

    def rate(values, time):
        result = stepper.empty(4)
        stepper.rate(stepper.array(values), time, result, stepper.empty(1))
        return result.numpy()

    columns = []
    for axis, value in enumerate(y):
        shift = np.zeros_like(y)
        shift[axis] = 2e-5 * value
        columns.append(
            (rate(y + shift, t) - rate(y - shift, t)) / (2 * shift[axis])
        )
    jacobian = np.column_stack(columns)
    rhs = np.array([0.1, 2.0, 3.0, -0.8])
    expected = np.linalg.solve(np.eye(4) / (float(GAMMA) * h) - jacobian, rhs)
    factors = [stepper.array(rhs), stages.inverse, stages.dp, stages.dl]
    sums = stepper.empty(2)
    result = stepper.empty(4)
    stepper.launch(_water_increment, 1, [*factors, stepper.weights], sums)
    stepper.launch(_back_substitute, 4, [*factors, sums], result)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-12)
    fd_time = (rate(y, t + 0.01) - rate(y, t - 0.01)) / 0.02
    np.testing.assert_allclose(
        stages.ft.numpy(), fd_time, rtol=2e-7, atol=1e-14
    )
