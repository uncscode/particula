"""Check both measured timing schemas against width-specific references."""

import json

from stage3_width_confirmation import table


def test_capacity_and_midpoint_keep_separate_advance_times(tmp_path):
    """Older capacity rows and newer split clocks retain distinct timings."""
    obs = dict(n=16, gas=[1e-8], particle=[2e-8, 0.0], ledger=0.0)
    case = dict(
        n=16,
        species=1,
        density=1e12,
        charge="neutral",
        family=101,
        horizon=1.0,
        history=[obs, obs],
        status="completed",
        kernel="off",
        total_s=3.0,
        cuda_peak_allocated=2048,
    )
    for folder in ("width-moving", "scale", "width-confirm"):
        (tmp_path / folder).mkdir()
    for bins in (16, 32):
        reference = dict(case, bins=bins, charge_extent=3, truncation_ok=True)
        (tmp_path / "width-moving" / f"{bins}.json").write_text(
            json.dumps(reference)
        )
    (tmp_path / "scale" / "scaling-a.json").write_text(
        json.dumps(dict(case, advance_and_observe_s=2.0))
    )
    (tmp_path / "width-confirm" / "a.json").write_text(
        json.dumps(dict(case, advance_s=1.0))
    )
    rows = table(tmp_path)
    assert {
        r["vapor_integrator"]: r["median_device_advance_s"] for r in rows
    } == {"original": 2.0, "midpoint": 1.0}
    assert all(
        r["mass_upper_pct"] == 0 and r["number_upper_pct"] == 0 for r in rows
    )
