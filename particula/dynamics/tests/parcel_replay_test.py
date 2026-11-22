"""Independent budget, ownership and restart checks for offline replay."""

import json
from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.parcel_replay import (
    OrganicForcing,
    ParcelTrace,
    ReplayCheckpoint,
    replay,
)
from particula.dynamics.tests.parcel_fixtures import (  # noqa: F401
    parcel_device_fixture,
    parcel_trace_fixture,
)


def test_trace_roundtrip_and_units(trace, tmp_path):
    """Round-trip SI forcing and reject ambiguous units or time order."""
    table = tmp_path / "forcing.csv"
    trace.save(table)
    copy = ParcelTrace.load(table)
    assert copy.identity() == trace.identity()
    np.testing.assert_array_equal(trace.reference_radius, copy.reference_radius)
    with pytest.raises(ValueError, match="increasing"):
        replace(trace, time=np.array([0, 0.25, 0.25, 0.75, 1]))
    with pytest.raises(ValueError, match="physical"):
        replace(trace, pressure=-trace.pressure)
    meta = table.with_suffix(".json")
    data = json.loads(meta.read_text())
    data["humidity"] = "percent"
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="units"):
        ParcelTrace.load(table)
    changed = replace(
        trace, reference_radius=np.zeros_like(trace.reference_radius)
    )
    np.testing.assert_array_equal(
        replay(trace, max_step=0.1).wet_radius,
        replay(changed, max_step=0.1).wet_radius,
    )


def test_source_pulse_restart_and_species(trace, device, tmp_path):
    """Close organic and water ledgers across a pulse and reordered gases."""
    cfg = OrganicForcing(
        "test-gas", 1.0e-8, 0.22, (1e-10,) * 4, (0.0, 5e-9, 0.0, 0.0)
    )
    t = replace(trace, organic=cfg)
    opts = dict(max_step=0.125, device=device, species=("test-gas", "water"))
    result = replay(t, **opts)
    np.testing.assert_allclose(
        result.organic_gas + result.organic_particle,
        cfg.initial_gas + 1e-10 * t.time + 5e-9 * (t.time >= 0.5),
        rtol=3e-15,
        atol=0,
    )
    sign_water = t.vapor * t.initial.dry_air_mass + result.liquid
    np.testing.assert_allclose(
        sign_water - sign_water[0], result.water_restored, atol=4e-18
    )
    other = replay(t, **{**opts, "species": ("water", "test-gas")})
    np.testing.assert_array_equal(other.gas_masses[:, ::-1], result.gas_masses)
    for cut in (0, 1, 2, 4):
        before = replay(t, stop_index=cut, **opts)
        before.checkpoint.save(tmp_path / "checkpoint.json")
        checkpoint = ReplayCheckpoint.load(tmp_path / "checkpoint.json")
        after = replay(t, checkpoint=checkpoint, **opts)
        for name in (
            "gas_masses",
            "wet_radius",
            "organic_particle",
            "water_restored",
            "water_exchange",
        ):
            np.testing.assert_array_equal(
                getattr(after, name), getattr(result, name)[cut:]
            )
        with pytest.raises(ValueError, match="checkpoint"):
            replay(
                replace(t, provenance="different run"),
                checkpoint=checkpoint,
                **opts,
            )
    with pytest.raises(ValueError, match="nonphysical"):
        replay(replace(t, organic=replace(cfg, rates=(-1.0,) * 4)), **opts)


def test_humidity_sensitivity(trace, device):
    """Check the terminal water adjoint against centered perturbations."""
    options = dict(max_step=0.025, device=device)
    ad = replay(trace, gradient=True, **options).humidity_gradient
    assert ad > 0
    for h in (1e-5, 3e-6):
        plus = replay(trace, humidity_amplitude=h, **options).liquid[-1]
        minus = replay(trace, humidity_amplitude=-h, **options).liquid[-1]
        assert ad == pytest.approx((plus - minus) / (2 * h), rel=2e-6)
