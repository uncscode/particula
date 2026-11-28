"""Analytic limits, public boundary conventions and independent solar angles."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from numpy.testing import assert_allclose
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import (
    sensor_band_radiance,
    solve_shortwave,
)
from particula.optics.solar import SolarInput, earth_reference, solar_position
from particula.optics.surface import (
    dusty_ground,
    lambertian_surface,
    ocean_surface,
    ross_li_surface,
)

WAVE = np.array([400.0, 500.0, 700.0]) * 1e-9
SUN = SolarInput(WAVE, 0.5, 2e9, 0.0)
BLACK = lambertian_surface(WAVE, 0.0)


def layer(depth, albedo, *, moment_count=1, g=0.0, heights=(1000, 0)):
    """Create homogeneous test layers at each wavelength."""
    shape = (len(heights) - 1, len(WAVE))
    return SpectralColumn(
        WAVE,
        heights,
        np.broadcast_to(depth, shape),
        np.broadcast_to(albedo, shape),
        np.broadcast_to(g ** np.arange(moment_count), (*shape, moment_count)),
    )


@pytest.mark.parametrize(
    ("lat", "lon", "instant", "zenith", "azimuth"),
    [
        (67.3, 26.6, "2022-07-15T10:00", 45.915728, 173.643801),
        (-45, 179.9, "2024-12-21T18:00", 73.466944, 106.851411),
        (80, 0, "2024-12-21T12:00", 103.441008, 180.402982),
        (-80, -179.9, "2024-12-21T20:00", 61.803186, 63.783868),
        (35, -106, "2024-06-21T11:50", 90.936773, 60.196061),
    ],
)
def test_solar_geometry(lat, lon, instant, zenith, azimuth):
    """Pvlib v0.13.1 NREL SPA geometric angles, delta T 69 s, height 0 m."""
    utc = datetime.fromisoformat(instant).replace(tzinfo=timezone.utc)
    pos = solar_position(lat, lon, utc)
    assert_allclose(
        [pos.zenith_deg, pos.azimuth_deg], [zenith, azimuth], atol=0.01, rtol=0
    )
    assert pos == solar_position(
        lat, lon, utc.astimezone(timezone(timedelta(hours=-7)))
    )


def test_solar_reference_energy_and_polar_night():
    """Integrate TSIS bins over the actual REPTRAN bounds, test both grids."""
    t = datetime(2024, 12, 21, 12, tzinfo=timezone.utc)
    for grid in ("full", "reduced"):
        ref = earth_reference(-80, 0, t, [120000, 2000, 0], grid=grid)
        energy = np.dot(ref.sun.direct_normal, ref.sun.quadrature_weights_m)
        # Integral of the native TSIS 1 nm bins minus the partial UV end-bin.
        assert_allclose(
            energy * ref.position.distance_au**2,
            1343.0814111182467,
            atol=0.02,
            rtol=0,
        )
        assert np.all(ref.column.optical_depth >= 0)
    night = earth_reference(80, 0, t, [120000, 0], WAVE)
    assert np.all(night.sun.direct_normal == 0)
    assert night.sun.mu0 == 0


def test_absorbing_slab_and_energy_units():
    """Beer–Lambert absorption, exact direct projection, spectral integral."""
    tau = np.array([0.2, 1.0, 2.0])
    out = solve_shortwave(layer(tau, 0), SUN, BLACK)
    assert_allclose(out.direct_down[0], 1e9)
    assert_allclose(out.direct_down[1], 1e9 * np.exp(-tau / 0.5))
    assert_allclose(out.diffuse_down, 0, atol=1e-6)
    assert_allclose(out.up, 0, atol=1e-6)
    assert_allclose(out.absorption[0], 1e9 * (1 - np.exp(-tau / 0.5)))
    assert_allclose(out.integrate(out.direct_down[0]), 300.0)
    assert_allclose(np.trapezoid(out.direct_down[0] * 1e-9, WAVE * 1e9), 300.0)


@pytest.mark.parametrize("depth", [1e-4, 0.4, 30.0])
def test_conservative_scattering_and_slab_subdivision(depth):
    """Conservative scattering redistributes energy even with bright ground."""
    ground = lambertian_surface(WAVE, 0.85)
    params = dict(moment_count=33, g=0.78)
    full = solve_shortwave(layer(depth, 1.0, **params), SUN, ground)
    split = solve_shortwave(
        layer(depth / 3, 1.0, heights=(3, 2, 1, 0), **params), SUN, ground
    )
    assert_allclose(full.absorption, 0, atol=0.02)
    assert_allclose(
        full.net_down, split.net_down[[0, -1]], atol=0.02, rtol=1e-10
    )
    assert_allclose(full.direct_down[-1], 1e9 * np.exp(-depth / 0.5))
    assert np.all(full.diffuse_down[-1] > 0)


def test_lambertian_radiance_sensor_and_response_order():
    """Band means preserve a constant and follow prescribed spectral order."""
    for rho in (np.full(3, 0.4), np.array([0.1, 0.8, 0.3])):
        out = solve_shortwave(
            layer(0, 0),
            SUN,
            lambertian_surface(WAVE, rho),
            view_zenith_deg=57,
            relative_azimuth_deg=86,
        )
        assert_allclose(out.radiance, rho * 1e9 / np.pi)
        assert_allclose(np.pi * out.radiance, out.up[0])
        assert_allclose(
            sensor_band_radiance(out, [0, 1, 0]), rho[1] * 1e9 / np.pi
        )
        assert_allclose(
            sensor_band_radiance(out, [1, 0, 0]), rho[0] * 1e9 / np.pi
        )
    with pytest.raises(ValueError, match="positive integral"):
        sensor_band_radiance(out, np.zeros(3))


def test_surface_models():
    """Check glint geometry, water-leaving source and Ross–Li normalization."""
    water = ocean_surface(WAVE, wind_speed=6.0, leaving_radiance=3.0)
    assert np.all(water.brdf(0.5, 0.5, np.pi) > 100 * water.brdf(0.5, 0.5, 0))
    dark = SolarInput(WAVE, 0.0, 0.0, 0.0)
    out = solve_shortwave(layer(0, 0), dark, water, view_zenith_deg=60)
    assert_allclose(out.radiance, 3.0)
    assert_allclose(out.up[0], 3 * np.pi)
    land = ross_li_surface(WAVE, 0.2, 0, 0)
    assert_allclose(land.brdf(0.3, 0.6, 1.2), 0.2 / np.pi)
    assert np.all(
        dusty_ground(WAVE).brdf(0.5, 1, 0) > ocean_surface(WAVE).brdf(0.5, 1, 0)
    )


def test_reject_misaligned_axes_and_invalid_bounds():
    """Require physical geometry, coordinate alignment, and positive weights."""
    with pytest.raises(ValueError, match="descending"):
        layer(0, 0, heights=(0, 10))
    with pytest.raises(ValueError, match="wavelengths"):
        solve_shortwave(layer(0, 0), SUN, lambertian_surface(WAVE * 2, 0.2))
    with pytest.raises(ValueError, match="quadrature_weights"):
        SolarInput(WAVE, 1, 0, 0, [-1, 1, 1])
    with pytest.raises(ValueError, match="timezone"):
        solar_position(0, 0, datetime(2025, 3, 20))
    with pytest.raises(ValueError, match="120000"):
        earth_reference(0, 0, datetime.now(timezone.utc), [123000, 0])
    with pytest.raises(ValueError, match="band|coverage"):
        earth_reference(0, 0, datetime.now(timezone.utc), [120000, 0], [200e-9])


def test_partial_bands_and_profile_fields():
    """Integral over interpolated endpoints and a native representative band."""
    result = solve_shortwave(layer(0.0, 0.0), SUN, BLACK)
    assert_allclose(
        result.integrate(result.direct_down, band_m=(430e-9, 575e-9)), 145.0
    )
    t = datetime(2023, 6, 21, 12, tzinfo=timezone.utc)
    ref = earth_reference(20.0, 30.0, t, [120000.0, 0.0, -420.0])
    out = solve_shortwave(
        ref.column, ref.sun, dusty_ground(ref.sun.wavelength_m)
    )
    lo, hi = ref.band_bounds_m[0]
    keep = np.all(ref.band_bounds_m == [lo, hi], axis=1)
    assert_allclose(
        out.integrate(out.up, band_m=(lo, hi)),
        out.up[:, keep] @ ref.sun.quadrature_weights_m[keep],
    )
    with pytest.raises(ValueError, match="whole quadrature bands"):
        out.integrate(out.up, band_m=(lo + 0.1 * (hi - lo), hi))
    override = earth_reference(
        20.0,
        30.0,
        t,
        [120000.0, 0.0],
        WAVE,
        pressure_pa=[0.0, 101325.0],
        gas_optical_depth=np.zeros((1, 3)),
        profile_name="test sounding",
    )
    assert_allclose(override.column.single_scattering_albedo, 1)
    assert "test sounding pressure" in override.sources
    with pytest.raises(ValueError, match="pressure"):
        earth_reference(
            20.0, 30.0, t, [120000.0, 0.0], WAVE, pressure_pa=[1.0, 0.0]
        )


def test_ocean_angular_quadrature_and_reflection():
    """Independent dense hemispheric integration of the rough-sea BRDF."""
    water = ocean_surface(WAVE, wind_speed=6.0)
    mu, weights = np.polynomial.legendre.leggauss(80)
    mu = (mu + 1) / 2
    phi = (np.arange(360) + 0.5) * np.pi / 360
    ground = water.brdf(0.5, mu[:, None], phi).mean(-1) @ (np.pi * mu * weights)
    flux = solve_shortwave(layer(0, 0), SUN, water, streams=24)
    assert_allclose(
        flux.up[0] / (SUN.mu0 * SUN.direct_normal), ground, rtol=1e-9
    )
    assert np.all((ground > 0) & (ground < 1))
    glint = solve_shortwave(
        layer(0, 0),
        SUN,
        water,
        streams=24,
        view_zenith_deg=60,
        relative_azimuth_deg=180.0,
    )
    assert_allclose(
        glint.radiance,
        water.brdf(0.5, 0.5, np.pi) * SUN.direct_normal * SUN.mu0,
        rtol=1e-8,
    )


def test_sounding_absorber_number_column():
    """Two absorbers in a linear density profile have an analytic depth."""
    from particula.optics.solar import profile_gas_optical_depth
    from particula.util.constants import BOLTZMANN_CONSTANT

    cross = np.array([[[2e-24, 1e-23], [3e-24, 0.0]]])
    actual = profile_gas_optical_depth(
        [100, 0], [9e4, 1e5], [280, 280], [[0.1, 0.2], [0.1, 0.2]], cross
    )
    expected = (
        100 * 9.5e4 / (280 * BOLTZMANN_CONSTANT) * np.array([[2.2e-24, 3e-25]])
    )
    assert_allclose(actual, expected)
