"""UTC hourly data, spectral energy assignment and source identity."""

import json
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from particula.data.power import read_power_hour
from particula.optics.solar import SolarInput
from particula_docs import documentation_path

_PACKET = documentation_path() / "data/shortwave/sodankyla-20220715"
_START = datetime(2022, 7, 15, 10, tzinfo=UTC)


def test_archived_interval():
    """Match the provider labels, units, coordinates and checksum."""
    manifest = json.loads((_PACKET / "manifest.json").read_text())
    local_time = _START.astimezone(timezone(timedelta(hours=3)))
    sample = read_power_hour(
        _PACKET / manifest["file"], local_time, sha256=manifest["sha256"]
    )
    assert sample.api_version == manifest["api_version"]
    assert sample.start_utc == _START
    assert (sample.latitude_deg, sample.longitude_deg) == (67.367, 26.629)
    assert (
        sample.global_horizontal,
        sample.direct_normal,
        sample.diffuse_horizontal,
    ) == (411.12, 139.49, 269.37)
    shape = SolarInput([450e-9, 650e-9], 0.7, [1, 2], [2, 1])
    band = sample.slab_boundary(
        shape, direct_fraction=0.4, diffuse_fraction=0.6
    )
    np.testing.assert_allclose(
        np.trapezoid(band.direct_normal, band.wavelength_m), 139.49 * 0.4
    )
    np.testing.assert_allclose(
        np.trapezoid(band.diffuse_horizontal, band.wavelength_m), 269.37 * 0.6
    )
    assert sample.sha256 in band.source
    with pytest.raises(ValueError, match="SHA-256"):
        read_power_hour(_PACKET / manifest["file"], _START, sha256="mismatch")


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("header", "time_standard"), "LST", "UTC"),
        (("geometry", "coordinates"), [0, 91, 10], "coordinates"),
        (("parameters", "ALLSKY_SFC_SW_DNI", "units"), "kW/m²", "units"),
        (
            ("properties", "parameter", "ALLSKY_SFC_SW_DIFF", "2022071510"),
            -999,
            "Missing",
        ),
    ],
)
def test_reject_unsupported_power(tmp_path, path, value, reason):
    """A missing channel cannot silently become an idealized background."""
    payload = json.loads((_PACKET / "power-hourly.json").read_text())
    node = payload
    for name in path[:-1]:
        node = node[name]
    node[path[-1]] = value
    file = tmp_path / "input.json"
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=reason):
        read_power_hour(file, _START)


def test_replay_as_added_slab_boundary():
    """A transparent slab transmits the prescribed direct and diffuse energy."""
    from particula.optics.column import SpectralColumn
    from particula.optics.radiative_transfer import solve_shortwave
    from particula.optics.surface import lambertian_surface

    hour = read_power_hour(_PACKET / "power-hourly.json", _START)
    wave = np.array([440e-9, 675e-9])
    bounds = np.array([[400, 500], [600, 800]]) * 1e-9
    shape = SolarInput(
        wave, 0.65, [3, 2], [1, 2], [1e-7, 2e-7], band_bounds_m=bounds
    )
    sun = hour.slab_boundary(shape, direct_fraction=0.4, diffuse_fraction=0.6)
    np.testing.assert_array_equal(sun.band_bounds_m, bounds)
    vacuum = np.zeros((1, 2))
    slab = SpectralColumn(wave, [100.0, 0.0], vacuum, vacuum, vacuum[..., None])
    result = solve_shortwave(slab, sun, lambertian_surface(wave, 0))
    np.testing.assert_allclose(
        result.integrate(result.direct_down[-1]),
        0.65 * hour.direct_normal * 0.4,
    )
    np.testing.assert_allclose(
        result.integrate(result.diffuse_down[-1]),
        hour.diffuse_horizontal * 0.6,
    )
