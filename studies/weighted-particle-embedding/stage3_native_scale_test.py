"""Regression checks for the native scaling study's campaign and inventory."""

from types import SimpleNamespace

import numpy as np
import pytest
import stage3_native_large as campaign
import stage3_native_scale as scale


@pytest.fixture
def forecast(monkeypatch):
    """Supply two observations per forecast without allocating a GPU campaign."""
    seeds = []
    snapshots = np.zeros((1, 2, 76))
    # 32 volatile masses, dry core, then 32 gas concentrations.
    snapshots[..., 0] = 2
    snapshots[..., 33] = 4
    snapshots[..., 34] = 10
    snapshots[:, 1, 1] = 3
    snapshots[:, 1, 34] = 7

    def run(inputs, streams):
        seeds.extend(streams)
        return snapshots

    monkeypatch.setattr(
        scale, "NativeSplit", lambda *a, **k: SimpleNamespace(run=run)
    )
    monkeypatch.setattr(
        scale,
        "box",
        lambda *args: SimpleNamespace(volume=1.0, charge=np.zeros(2)),
    )
    monkeypatch.setattr(scale.wp, "synchronize_device", lambda device: None)
    monkeypatch.setattr(scale.wp, "init", lambda: None)
    monkeypatch.setattr(scale.wp, "get_mempool_used_mem_high", lambda device: 0)
    return seeds, snapshots


def test_sparse_campaign_resume(forecast, tmp_path):
    """Only the two missing seeds run, and completed output stays byte-exact."""
    seeds, _ = forecast
    completed = tmp_path / "native-1000000-546002.json"
    completed.write_bytes(b"previous result\n")
    campaign.run(tmp_path, float("inf"), million=True)
    assert seeds == [546001, 546003]
    assert completed.read_bytes() == b"previous result\n"
    for seed in seeds:
        assert (tmp_path / f"native-1000000-{seed}.json").is_file()
    campaign.run(tmp_path, float("inf"), million=True)
    assert seeds == [546001, 546003]


@pytest.mark.parametrize("dry_mass", [3.0, np.inf, np.nan])
def test_native_inventory_rejects_dry_loss_or_nonfinite_observations(
    forecast, dry_mass
):
    """A balanced vapor exchange cannot mask a corrupted dry core."""
    _, snapshots = forecast
    snapshots[:, 1, 33] = dry_mass
    with pytest.raises(ArithmeticError, match="inventory|Nonfinite"):
        scale.measure(2, 32, 2, 307, 1, 1, 1, 1, 1, 1)
