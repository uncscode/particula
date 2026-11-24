"""Analytical dispersion and kinematic flow controls."""

import numpy as np
import pytest
from particula.coupling.dispersion import gaussian_plume, puff_concentration
from particula.coupling.prescribed import DuneFlow, Sounding
from particula.dynamics.parcel import ParcelInitialState


@pytest.fixture
def column():
    """Give a uniform 5 m/s westerly with nonuniform thermodynamics."""
    return Sounding(
        [0, 100], [290, 285], [100000, 99000], [0.008, 0.006], [5, 5], [0, 0]
    )


@pytest.mark.parametrize("stability", list("ABCDEF"))
def test_plume_flux(stability):
    """Integrating a cross-section including the image recovers kg/s."""
    y = np.linspace(-500, 500, 601)
    z = np.linspace(0, 500, 301)
    c = gaussian_plume(
        250, y[:, None], z, rate=0.003, wind=5, height=9, stability=stability
    )
    flux = 5 * np.trapezoid(np.trapezoid(c, z), y)
    assert flux == pytest.approx(0.003, rel=1e-10)
    assert (
        gaussian_plume(
            -1, 0, 9, rate=0.003, wind=5, height=9, stability=stability
        )
        == 0
    )


def test_class_d_width():
    """Check the neutral crosswind width at 1 km against Briggs Table A-1."""
    c = gaussian_plume(
        1000, [0, 80 / np.sqrt(1.1)], 0, rate=1, wind=9, height=0, stability="D"
    )
    assert c[1] / c[0] == pytest.approx(np.exp(-0.5))


def test_puff_moments():
    """Recover pulse mass, displacement, and variance independent of the kernel."""
    x = np.linspace(-50, 110, 201)
    y = np.linspace(-60, 60, 121)
    z = np.linspace(0, 70, 71)
    c = puff_concentration(
        x[:, None, None],
        y[None, :, None],
        z,
        mass=0.6,
        age=10,
        wind=3,
        height=2,
        diffusivity=[2, 1, 1],
    )
    marginal = np.trapezoid(np.trapezoid(c, z), y)
    assert np.trapezoid(marginal, x) == pytest.approx(0.6, rel=1e-10)
    assert np.trapezoid(marginal * x, x) / 0.6 == pytest.approx(30)
    assert np.trapezoid(marginal * (x - 30) ** 2, x) / 0.6 == pytest.approx(40)


def test_flat_tracks_and_impact(column):
    """Recover exact positions through impact and pin later samples at the bed."""
    flow = DuneFlow(column, 0, 20, 100)
    t = np.array([0, 1, 2, 4, 10])
    air = flow.track((0, 0, 5), t)
    np.testing.assert_allclose(air.xyz[:, 0], 5 * t, atol=1e-10)
    np.testing.assert_allclose(air.xyz[:, 2], 5)
    dust = flow.track((0, 0, 5), t, settling=2)
    assert dust.deposition_time == pytest.approx(2.5)
    np.testing.assert_allclose(
        dust.xyz[:, 0], 5 * np.minimum(t, 2.5), atol=1e-10
    )
    np.testing.assert_allclose(dust.xyz[:, 2], np.maximum(5 - 2 * t, 0))
    assert flow.track((0, 0, 0), t, settling=2).deposition_time == 0


def test_dune_streamline_and_travel_time(column):
    """The integral of J dx / U gives the exact travel time between two x."""
    from scipy.integrate import quad

    flow = DuneFlow(column, 10, 20, 100)
    t_end = quad(lambda x: (1 - 0.1 * np.exp(-(x**2) / 800)) / 5, -40, 40)[0]
    z_start = 10 + 0.9 * 10 * np.exp(-2)
    t = np.array([0, t_end / 2, t_end])
    template = ParcelInitialState(
        np.array([5e-8]), np.array([50]), 0.4, 1e5, 300, 0
    )
    track, trace = flow.parcel_trace(
        "air", [-40, 0, z_start], t, template, provenance="test"
    )
    np.testing.assert_allclose(track.xyz[:, 0], [-40, 0, 40], atol=2e-6)
    np.testing.assert_allclose(trace.height, [z_start, 19, z_start], atol=2e-6)
    np.testing.assert_allclose(trace.temperature, 290 - 0.05 * trace.height)
    np.testing.assert_allclose(trace.initial.number, [50])
    # Per kg of dry air, the mean molecular constants are about 287 and 461.5.
    np.testing.assert_allclose(
        trace.volume,
        (287.05 + 461.5 * trace.vapor) * trace.temperature / trace.pressure,
        rtol=1e-4,
    )


def test_pickup_budget_and_no_pickup(column):
    """Crest speedup crosses the threshold and release is capped by bed mass."""
    flow = DuneFlow(column, 10, 10, 100)
    args = dict(
        reference_height=1,
        drag_coefficient=0.01,
        threshold=0.52,
        available_mass=0.001,
        pulse_mass=0.5,
    )
    assert flow.pickup(-100, **args) == 0
    assert flow.pickup(0, **args) == 0.001


def test_invalid_profiles_and_boundaries(column):
    """Reject extrapolation, reversed time, buried positions and invalid sources."""
    with pytest.raises(ValueError):
        Sounding([1, 0], [290, 290], [1e5, 1e5], [0, 0], [1, 1], [0, 0])
    with pytest.raises(ValueError):
        DuneFlow(column, 5, 10, 110)
    flow = DuneFlow(column, 5, 10, 100)
    for xyz, t in [([0, 0, 2], [0, 1]), ([0, 0, 8], [1, 0])]:
        with pytest.raises(ValueError):
            flow.track(xyz, t)
    with pytest.raises(ValueError):
        gaussian_plume(20, 0, 0, rate=1, wind=0, height=0, stability="D")
    with pytest.raises(ValueError):
        puff_concentration(
            20, 0, 0, mass=1, age=1, wind=0, height=0, diffusivity=[0, 1, 1]
        )


def test_top_hat_boundary_and_mass():
    """A ground-centered sphere folds its mass uniformly into a hemisphere."""
    radius = np.sqrt(30)  # K=1, t=3, R²=5 times the variance
    mass = 0.65
    density = puff_concentration(
        [3, 3, 3],
        [0, 0.99 * radius, 1.01 * radius],
        0,
        mass=mass,
        age=3,
        wind=1,
        height=0,
        diffusivity=[1, 1, 1],
        shape="top_hat",
    )
    hemisphere = 2 * np.pi * radius**3 / 3
    np.testing.assert_allclose(density * hemisphere, [mass, mass, 0])


def test_read_sounding(tmp_path):
    """Named columns accept arbitrary order and reject an unspecified humidity basis."""
    p = tmp_path / "column.csv"
    header = "east_m_s,height_m,temperature_K,pressure_Pa,vapor_kg_per_kg_dry_air,north_m_s"
    p.write_text(header + "\n4,0,290,100000,0.008,0\n7,100,285,99000,0.006,1\n")
    profile = Sounding.from_csv(p)
    np.testing.assert_allclose(
        profile.thermodynamics(50), [287.5, 99500, 0.007]
    )
    p.write_text(p.read_text().replace("vapor_kg_per_kg_dry_air", "humidity"))
    with pytest.raises(ValueError):
        Sounding.from_csv(p)
