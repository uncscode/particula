"""Independent physical, controller and derivative checks for parcel RK23."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel import (
    CP,
    LATENT,
    RD,
    ParcelInitialState,
    ParcelRK23,
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


def test_adaptive_ascent_and_reverse(parcel, device):
    """Refinement and independently adapted perturbations at fixed time."""
    options = dict(timestep=0.5, steps=200)

    def advance(speed, tolerance, gradient=False):
        return run_parcel(
            parcel,
            updraft=speed,
            device=device,
            **options,
            gradient=gradient,
            adaptive=ParcelRK23(rtol=tolerance),
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


def test_passive_dry_parcel_on_decimal_observation_grid(parcel, device):
    """No derivative or artificial short tail on the dry, passive trajectory."""
    result = run_parcel(
        replace(parcel, vapor=0.0),
        timestep=0.1,
        steps=33,
        updraft=1.0,
        adaptive=ParcelRK23(),
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


def test_stationary_moist_parcel_sensitivity(parcel, device):
    """An equilibrated forward state still has a fast tangent relaxation."""
    cadence = dict(timestep=0.5, steps=4)
    controls = dict(device=device, adaptive=ParcelRK23())
    rest = run_parcel(parcel, updraft=0.0, gradient=True, **cadence, **controls)
    np.testing.assert_allclose(
        rest.liquid, rest.liquid[0], rtol=2e-13, atol=0.0
    )
    assert rest.statistics.maximum_step < cadence["timestep"]
    delta = 0.001
    for backend in (controls, {"device": "cpu"}):
        plus = run_parcel(parcel, updraft=delta, **cadence, **backend)
        minus = run_parcel(parcel, updraft=-delta, **cadence, **backend)
        fd = (plus.liquid[-1] - minus.liquid[-1]) / (2 * delta)
        assert fd > 0.0
        np.testing.assert_allclose(
            rest.updraft_gradient, fd, rtol=2e-5, atol=0.0
        )


def test_trial_failures_are_bounded(parcel, device):
    """An unusable step raises at the floor or cap without changing inputs."""
    initial = parcel.number.copy()
    for controls, message in [
        (ParcelRK23(max_trials=1), "max_trials"),
        (ParcelRK23(min_step=1.0), "tolerance"),
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
        adaptive=ParcelRK23(),
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
