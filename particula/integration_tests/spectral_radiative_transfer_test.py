"""Public native-state shortwave budget and independent DISORT comparison."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from particula.optics import OpticalConfig, population_optics
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import earth_reference
from particula.optics.surface import dusty_ground, ocean_surface
from particula.particles import ParticleData

WAVE = np.array([450.0, 550.0, 850.0]) * 1e-9
HEIGHT = [120000.0, 1200.0, 0.0]
TIME = datetime(2024, 6, 21, 19, tzinfo=timezone.utc)


def native_scene():
    """Two boxes represent clear overlying air and a populated mixing layer."""
    radii = np.array([80e-9, 175e-9])
    mass = (4 * np.pi / 3 * 1700 * radii**3)[:, None]
    particles = ParticleData(
        masses=np.stack((np.zeros_like(mass), mass)),
        density=np.array([1700.0]),
        population=np.array([[0.0, 0.0], [8e8, 1.2e8]]),
        charge=np.zeros((2, 2)),
        volume=np.ones(2),
        distribution_type="discrete",
    )
    config = OpticalConfig(WAVE, particle_index=1.53 + 0.025j)
    native = population_optics(particles, config)
    ref = earth_reference(35.0, -106.0, TIME, HEIGHT, WAVE)
    aerosol = SpectralColumn.from_population(WAVE, HEIGHT, native, hg_order=64)
    return ref, aerosol, native


@pytest.mark.parametrize(
    ("surface", "label"), [(ocean_surface, "ocean"), (dusty_ground, "soil")]
)
def test_native_flux_budget_and_pair(surface, label):
    """Compare both boundary fluxes and their differences on an offline case."""
    ref, aerosol, pop = native_scene()
    ground = surface(WAVE)
    simulated = []
    oracle = np.genfromtxt(
        Path(__file__).with_name("data") / "shortwave_disort.csv",
        delimiter=",",
        names=True,
        encoding="utf8",
        dtype=None,
    )
    for state, column in (
        ("base", ref.column),
        ("case", ref.column.plus(aerosol)),
    ):
        result = solve_shortwave(column, ref.sun, ground, streams=16)
        observed = oracle[
            (oracle["surface"] == label) & (oracle["state"] == state)
        ]
        assert_allclose(
            result.up[0] * 1e-9, observed["up_top_W_m2_nm"], rtol=0, atol=1e-5
        )
        assert_allclose(
            (result.direct_down + result.diffuse_down)[-1] * 1e-9,
            observed["down_ground_W_m2_nm"],
            rtol=0,
            atol=1e-5,
        )
        assert_allclose(
            result.absorption.sum(axis=0) + result.net_down[-1] + result.up[0],
            ref.sun.mu0 * ref.sun.direct_normal,
            rtol=1e-12,
        )
        simulated.append(result)
    delta = simulated[1].difference(simulated[0])
    expected = oracle[oracle["surface"] == label]["up_top_W_m2_nm"].reshape(
        2, -1
    )
    assert_allclose(
        delta.integrate(delta.up[0]),
        np.trapezoid(expected[1] - expected[0], WAVE * 1e9),
        rtol=0,
        atol=0.005,
    )
    assert_allclose(
        delta.up[0] * 1e-9, expected[1] - expected[0], rtol=0, atol=1e-5
    )
    # Keep one parcel's optical coefficients fixed while splitting its path.
    split = SpectralColumn.from_population(
        WAVE, [1200.0, 600.0, 0.0], pop, box_indices=[1, 1]
    )
    one = SpectralColumn.from_population(
        WAVE, [1200.0, 0.0], pop, box_indices=[1]
    )
    a = solve_shortwave(one, ref.sun, ground)
    b = solve_shortwave(split, ref.sun, ground)
    assert_allclose(a.net_down, b.net_down[[0, -1]], rtol=1e-11)
