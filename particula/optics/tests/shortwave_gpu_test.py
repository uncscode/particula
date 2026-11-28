"""Batched and live Warp shortwave transport, numerical and lifetime checks."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
import warp as wp
from numpy.testing import assert_allclose
from particula.optics import (
    OpticalConfig,
    SolarInput,
    SpectralColumn,
    dusty_ground,
    earth_reference,
    lambertian_surface,
    ocean_surface,
    population_optics,
    ross_li_surface,
    sensor_band_radiance,
    solve_shortwave,
)
from particula.optics.gpu.shortwave import prepare_shortwave
from particula.optics.gpu.shortwave_resident import ResidentShortwave

WAVE = np.array([450e-9, 550e-9, 850e-9])


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Select Warp CPU or physical CUDA when available."""
    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def reference():
    """Build a reproducible two-layer clear solar scene."""
    return earth_reference(
        35, -106, datetime(2024, 6, 21, 19, tzinfo=UTC), [120000, 2000, 0], WAVE
    )


def test_batched_directional_flux(device):
    """Nonidentical boundaries, thick layers, tiling, partial final tile."""
    ref = reference()
    tau = np.array([[0.007, 0.004, 0.001], [3, 1, 0.9]])
    aero = SpectralColumn(
        WAVE,
        [120000, 2000, 0],
        tau,
        np.full_like(tau, 0.96),
        np.broadcast_to(0.79 ** np.arange(33), (*tau.shape, 33)),
    )
    columns = (
        ref.column,
        ref.column.plus(aero),
        ref.column.plus(aero.scaled(6.1)),
    )
    surfaces = (
        dusty_ground(WAVE),
        ocean_surface(WAVE, wind_speed=4.0, leaving_radiance=1e6),
        ross_li_surface(WAVE, 0.22, 0.06, 0.02),
    )
    suns = (
        ref.sun,
        replace(ref.sun, mu0=0.3),
        replace(ref.sun, diffuse_horizontal=1.1e8),
    )
    response = np.array([0.1, 0.9, 0.2])
    kwargs = dict(streams=4, view_zenith_deg=51.0, relative_azimuth_deg=157.0)
    work = prepare_shortwave(
        columns,
        suns,
        surfaces,
        device=device,
        case_tile=4,
        stream=wp.Stream(device) if device.startswith("cuda") else None,
        reference_cases=[0, 0, 1],
        sensor_response=response,
        **kwargs,
    )
    with pytest.raises(RuntimeError, match="Enqueue"):
        work.download()
    work.enqueue()
    results = work.download()
    totals = work.broadband.numpy()
    assert_allclose(work.broadband_delta.numpy(), totals - totals[[0, 0, 1]])
    for i, r in enumerate(results):
        oracle = solve_shortwave(columns[i], suns[i], surfaces[i], **kwargs)
        for component in (
            "up",
            "direct_down",
            "diffuse_down",
            "absorption",
            "radiance",
        ):
            assert_allclose(
                getattr(r, component),
                getattr(oracle, component),
                rtol=2e-9,
                atol=0.002,
            )
        assert_allclose(
            work.broadband.numpy()[i],
            np.stack(
                [
                    oracle.integrate(getattr(oracle, k))
                    for k in ("direct_down", "diffuse_down", "up")
                ],
                axis=-1,
            ),
            rtol=2e-9,
            atol=2e-9,
        )
        assert_allclose(
            work.band_radiance.numpy()[i],
            sensor_band_radiance(oracle, response),
            rtol=2e-9,
        )


def test_device_inputs_and_analytic_limit(device):
    """Unscattered attenuation, conservative bright reflection and repeat use."""
    sun = SolarInput(WAVE, 0.5, 1e9, 0, quadrature_weights_m=[1e-8, 2e-8, 4e-8])
    shape = (1, len(WAVE))
    tau = np.array([[0.1, 0.5, 3.0]])
    col = SpectralColumn(
        WAVE, [500, 0], tau, np.zeros(shape), np.ones((*shape, 1))
    )
    work = prepare_shortwave(
        col, sun, lambertian_surface(WAVE, 0), device=device
    )
    work.enqueue()
    (a,) = work.download()
    assert_allclose(a.direct_down[1], 5e8 * np.exp(-tau[0] / 0.5))
    assert_allclose(a.up, 0, atol=1e-8)
    assert_allclose(a.diffuse_down, 0, atol=1e-8)
    new_tau = tau * 1.6
    work.layers.depth.assign(new_tau[None, ...])
    work.enqueue()
    (b,) = work.download()
    assert_allclose(b.direct_down[1], 5e8 * np.exp(-new_tau[0] / 0.5))
    assert_allclose(a.direct_down[1], 5e8 * np.exp(-tau[0] / 0.5))
    assert_allclose(
        work.broadband.numpy()[0, 1, 0],
        np.dot(b.direct_down[1], sun.quadrature_weights_m),
    )
    work.layers.depth.assign(-tau[None, ...])
    work.enqueue()
    with pytest.raises(RuntimeError, match="transport failed"):
        work.download()
    work.layers.depth.assign(tau[None, ...])
    work.enqueue()
    work.check()


def test_live_runnable_binding(device):
    """Follow dilution, mixing reference gas and mapped live box on device."""
    from particula.execution.tests.prepared_restart_capture_test import scenario

    initial, process, dt = scenario("dilution")
    ref = reference()
    config = OpticalConfig(WAVE, particle_index=1.5 + 0.01j)
    surfaces = (ocean_surface(WAVE), dusty_ground(WAVE))
    with process.prepare(initial, time_step=dt, device=device) as sim:
        observer = ResidentShortwave(
            sim,
            config,
            (ref.column, ref.column),
            ref.sun,
            surfaces,
            box_indices=[[-1, 0], [-1, -1]],
            coefficient_capacity=64,
            particle_tile=32,
            streams=4,
        )
        for step in (0, 2):
            if step:
                sim.run(step)
            observer.enqueue()
            observer.check()
            actual = observer.download()
            pop = population_optics(initial.particles, config)
            aerosol = SpectralColumn.from_population(
                WAVE, ref.column.interface_height_m, pop, [0, 0]
            )
            depth = aerosol.optical_depth.copy()
            depth[0] = 0
            aerosol = replace(
                aerosol, optical_depth=depth * np.exp(-0.2 * dt * step)
            )
            for i, col in enumerate((ref.column.plus(aerosol), ref.column)):
                expected = solve_shortwave(col, ref.sun, surfaces[i], streams=4)
                assert_allclose(
                    actual[i].up, expected.up, rtol=2e-8, atol=0.001
                )
                assert_allclose(
                    actual[i].direct_down,
                    expected.direct_down,
                    rtol=2e-8,
                    atol=0.001,
                )
        assert sim.completed_steps == 2
    with pytest.raises(RuntimeError):
        observer.enqueue()


def test_pair_coordinates():
    """Reject ambiguous reference assignments before allocating a workspace."""
    ref = reference()
    with pytest.raises(ValueError, match="2–16"):
        prepare_shortwave(
            ref.column, ref.sun, dusty_ground(WAVE), streams=32, device="cpu"
        )
    for indices in ([1], [-1], [0.1], []):
        with pytest.raises(ValueError, match="reference case"):
            prepare_shortwave(
                ref.column,
                ref.sun,
                dusty_ground(WAVE),
                reference_cases=indices,
                device="cpu",
            )
    other = replace(ref.column, interface_height_m=[120000.0, 1500.0, 0.0])
    with pytest.raises(ValueError, match="heights"):
        prepare_shortwave(
            (ref.column, other),
            ref.sun,
            dusty_ground(WAVE),
            reference_cases=[0, 0],
            device="cpu",
        )


def test_disort_pair(device):
    """Compare spectral boundary flux and paired broadband effect to C DISORT."""
    from pathlib import Path

    from particula.integration_tests.spectral_radiative_transfer_test import (
        native_scene,
    )

    ref, aerosol, _ = native_scene()
    wave = ref.column.wavelength_m
    columns = (ref.column, ref.column.plus(aerosol))
    ocean, land = ocean_surface(wave), dusty_ground(wave)
    work = prepare_shortwave(
        columns * 2,
        ref.sun,
        (ocean, ocean, land, land),
        streams=16,
        reference_cases=[0, 0, 2, 2],
        device=device,
    )
    work.enqueue()
    results = work.download()
    data = np.genfromtxt(
        Path(__file__).resolve().parents[2]
        / "integration_tests/data/shortwave_disort.csv",
        delimiter=",",
        names=True,
        encoding="utf8",
        dtype=None,
    )
    for pair, label in enumerate(("ocean", "soil")):
        observations = []
        for offset, state in enumerate(("base", "case")):
            rows = data[(data["surface"] == label) & (data["state"] == state)]
            r = results[2 * pair + offset]
            np.testing.assert_array_equal(
                rows["wavelength_nm"], np.rint(wave * 1e9)
            )
            assert_allclose(
                r.up[0] * 1e-9, rows["up_top_W_m2_nm"], rtol=0, atol=1e-5
            )
            assert_allclose(
                (r.direct_down[-1] + r.diffuse_down[-1]) * 1e-9,
                rows["down_ground_W_m2_nm"],
                rtol=0,
                atol=1e-5,
            )
            observations.append(rows["up_top_W_m2_nm"])
        assert_allclose(
            work.broadband_delta.numpy()[2 * pair + 1, 0, 2],
            np.trapezoid(observations[1] - observations[0], wave * 1e9),
            atol=0.005,
            rtol=0,
        )
