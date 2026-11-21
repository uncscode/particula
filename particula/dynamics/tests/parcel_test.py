"""Analytic controls, independent moist integration and trajectory adjoints."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel import (
    CP,
    LATENT,
    RD,
    RHO_W,
    RV,
    SIGMA,
    ParcelInitialState,
    run_parcel,
)
from scipy.optimize import minimize_scalar


@pytest.fixture
def initial():
    """One kilogram of dry air carries one three-class soluble mode."""
    return ParcelInitialState(
        dry_radius=np.array([45, 70, 110]) * 1e-9,
        number=np.array([2.5e7, 5e7, 2.5e7]),
        kappa=0.5,
        temperature=288.15,
        pressure=90000,
        vapor=0.0118,
    )


@pytest.fixture(params=["warp:cpu", "cuda:0"])
def warp_device(request):
    """Exercise Warp CPU and actual CUDA when available."""
    import warp as wp

    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("requires an NVIDIA CUDA device")
    return request.param


def test_dry_adiabat_and_dilution(initial, warp_device):
    """Check the dry adiabat and extensive inventories in expanding air."""
    for device in ("cpu", warp_device):
        for vapor in (0.0, initial.vapor):
            case = replace(
                initial, vapor=vapor, dry_air_mass=0.35, height=500.0
            )
            result = run_parcel(
                case,
                device=device,
                updraft=1.25,
                timestep=0.1,
                steps=100,
                phase_exchange=False,
            )
            np.testing.assert_allclose(result.height, 500 + 1.25 * result.time)
            np.testing.assert_allclose(result.vapor, vapor, rtol=3e-14)
            np.testing.assert_allclose(
                result.liquid, result.liquid[0], rtol=3e-13
            )
            np.testing.assert_allclose(
                result.wet_radius / result.wet_radius[0], 1.0
            )
            assert result.volume[-1] > result.volume[0]
            first, last = (
                result.aerosol_at(i, dry_density=1770) for i in (0, -1)
            )
            for aerosol, vol in [
                (first, result.volume[0]),
                (last, result.volume[-1]),
            ]:
                np.testing.assert_array_equal(aerosol.particles.volume, [1.0])
                np.testing.assert_allclose(
                    aerosol.particles.concentration[0] * vol, case.number
                )
                np.testing.assert_allclose(
                    aerosol.gas.concentration[0] * vol,
                    case.dry_air_mass * vapor,
                )
            np.testing.assert_allclose(
                last.particles.concentration / first.particles.concentration,
                result.volume[0] / result.volume[-1],
            )
            if vapor == 0:
                theta = result.temperature * (
                    case.pressure / result.pressure
                ) ** (RD / CP)
                np.testing.assert_allclose(theta, case.temperature, rtol=4e-12)


def test_stationary_and_evaporating_parcel(initial, warp_device):
    """Equilibrated air is stationary at rest and loses liquid on descent."""
    for device in ("cpu", warp_device):
        stationary = run_parcel(
            initial, device=device, updraft=0.0, timestep=0.1, steps=100
        )
        for name in (
            "temperature",
            "pressure",
            "vapor",
            "volume",
            "liquid",
            "wet_radius",
            "supersaturation",
        ):
            history = getattr(stationary, name)
            np.testing.assert_allclose(history / history[0], 1.0, rtol=5e-10)
        falling = run_parcel(
            initial, device=device, updraft=-1.0, timestep=0.1, steps=100
        )
        assert falling.liquid[-1] < falling.liquid[0]
        assert falling.vapor[-1] > falling.vapor[0]
        assert np.all(falling.wet_radius[-1] < falling.wet_radius[0])
        np.testing.assert_allclose(
            falling.vapor + falling.liquid,
            falling.vapor[0] + falling.liquid[0],
            atol=2e-14,
            rtol=0,
        )


def _activated_fraction(case, trajectory, sample):
    """Locate the maximum of the full κ-Köhler curve for each dry class."""
    kelvin = 2 * SIGMA / (RV * RHO_W * trajectory.temperature[sample])
    critical = []
    for dry in case.dry_radius:

        def negative_saturation(log_ratio, dry=dry):
            ratio = np.exp(log_ratio)
            return (
                -(ratio**3 - 1)
                / (ratio**3 - 1 + case.kappa)
                * np.exp(kelvin / (ratio * dry))
            )

        maximum = minimize_scalar(
            negative_saturation, bounds=(0.0, np.log(1000.0)), method="bounded"
        )
        assert maximum.success
        critical.append(dry * np.exp(maximum.x))
    return np.average(
        trajectory.wet_radius[sample] > critical, weights=case.number
    )


def test_moist_cloud_base_and_gradient(initial, warp_device):
    """At 0.05 s resolve the peak to 1% and each wet-radius history to 25 nm."""

    def solve(device, dt, speed=1.0, gradient=False):
        return run_parcel(
            initial,
            updraft=speed,
            timestep=dt,
            steps=round(100 / dt),
            device=device,
            gradient=gradient,
        )

    reference = solve("cpu", 0.05)
    fine = solve(warp_device, 0.05, gradient=True)
    coarse = solve(warp_device, 0.1, gradient=True)
    assert 0 < fine.supersaturation.argmax() < len(fine.time) - 1
    assert fine.supersaturation[0] < 0 < fine.supersaturation.max()
    assert fine.supersaturation[-1] < fine.supersaturation.max()
    assert np.all(fine.wet_radius[-1] > fine.wet_radius[0])
    # Absolute differences at every common observation, in the native units.
    limits = dict(
        height=1e-8,
        pressure=1e-3,
        temperature=2e-4,
        volume=1e-6,
        vapor=8e-8,
        liquid=8e-8,
        supersaturation=2e-5,
        wet_radius=25e-9,
    )
    for name, limit in limits.items():
        np.testing.assert_allclose(
            getattr(fine, name),
            getattr(reference, name),
            rtol=0,
            atol=limit,
            err_msg=name,
        )
    for name in ("supersaturation", "wet_radius", "liquid"):
        error = np.max(
            abs(getattr(fine, name)[::2] - getattr(reference, name)[::2])
        )
        coarse_error = np.max(
            abs(getattr(coarse, name) - getattr(reference, name)[::2])
        )
        assert error < 0.65 * coarse_error
    np.testing.assert_allclose(
        fine.supersaturation.max(), reference.supersaturation.max(), rtol=0.01
    )
    assert (
        abs(
            fine.height[fine.supersaturation.argmax()]
            - reference.height[reference.supersaturation.argmax()]
        )
        < 0.5
    )
    for t, expected in [(0, 0.0), (50, 0.75), (100, 1.0)]:
        for trajectory in (reference, fine):
            assert _activated_fraction(initial, trajectory, t * 20) == expected
    # Test the first law independently by integrating alpha dp (J/kg dry air).
    alpha = fine.volume / initial.dry_air_mass
    work = np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(fine.pressure))
    sensible = CP * (fine.temperature[1:] - fine.temperature[0])
    latent = LATENT * (fine.liquid[1:] - fine.liquid[0])
    np.testing.assert_allclose(sensible - latent, work, atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        fine.vapor + fine.liquid,
        reference.vapor[0] + reference.liquid[0],
        rtol=0,
        atol=4e-17,
    )
    for i in (0, 1000, 2000):
        snap = fine.aerosol_at(i, dry_density=1770.0)
        count = snap.particles.concentration[0] * fine.volume[i]
        inventories = count @ snap.particles.masses[0]
        expected_dry = initial.number @ (
            (4 * np.pi / 3) * initial.dry_radius**3 * 1770.0
        )
        np.testing.assert_allclose(count, initial.number, rtol=1e-14)
        np.testing.assert_allclose(
            inventories,
            [expected_dry, fine.liquid[i] * initial.dry_air_mass],
            rtol=5e-14,
        )
    # Check the derivative of the discrete program, then its continuum limit.
    assert fine.updraft_gradient > 0
    for delta in (0.01, 0.001, 0.0001):
        fd = (
            solve(warp_device, 0.05, 1 + delta).liquid[-1]
            - solve(warp_device, 0.05, 1 - delta).liquid[-1]
        ) / (2 * delta)
        np.testing.assert_allclose(fine.updraft_gradient, fd, rtol=1e-5)
    delta = 0.001
    cpu_fd = (
        solve("cpu", 1.0, 1 + delta).liquid[-1]
        - solve("cpu", 1.0, 1 - delta).liquid[-1]
    ) / (2 * delta)
    assert abs(fine.updraft_gradient - cpu_fd) < 0.65 * abs(
        coarse.updraft_gradient - cpu_fd
    )
    np.testing.assert_allclose(fine.updraft_gradient, cpu_fd, rtol=1e-4)


@pytest.mark.parametrize(
    "changes",
    [
        {"dry_radius": np.array([0.0, 2.0])},
        {"number": np.array([-1.0, 1.0, 1.0])},
        {"vapor": -1e-4},
        {"vapor": 0.05},
        {"kappa": 0.0},
        {"dry_air_mass": 0.0},
        {"temperature": np.nan},
        {"pressure": 0.0},
        {"height": np.inf},
    ],
)
def test_reject_invalid_state(initial, changes):
    """Reject negative inventories and unsupported initial conditions."""
    with pytest.raises(ValueError):
        run_parcel(
            replace(initial, **changes), updraft=1.0, timestep=1.0, steps=1
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 1.5},
        {"timestep": 0.0},
        {"updraft": np.nan},
        {"device": "other"},
        {"device": "cpu", "gradient": True},
    ],
)
def test_reject_invalid_run(initial, changes):
    """Reject invalid integration and device requests before launch."""
    options = dict(steps=1, timestep=1.0, updraft=1.0)
    with pytest.raises(ValueError):
        run_parcel(initial, **(options | changes))
