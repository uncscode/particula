"""Guard the mass-class denominators used by the literature comparison."""

import json
from pathlib import Path

import numpy as np

DATA = (
    Path(__file__).resolve().parents[1]
    / "website/results/shear-drop-spectrum/data.json"
)


def test_exact_eightfold_abundance_and_fitted_rate():
    """Identify the exact class, independently deriving equation 3's clock."""
    data = json.loads(DATA.read_text())
    result = next(
        row
        for row in data["summary"]
        if row["case"] == "large_fine_pooled" and row["K"] == 0.09
    )
    np.testing.assert_allclose(
        result["n8"], result["n8_count"] / (result["slots"] * result["boxes"])
    )
    np.testing.assert_allclose(data["mean"][7], result["n8"])
    z = 1 / result["total"] - 1
    np.testing.assert_allclose(result["matched_constant"], z**7 / (1 + z) ** 9)
    theory = next(
        row
        for row in data["reference"]
        if row["cutoff"] == 128 and row["K"] == 0.09 and row["mass_class"] == 8
    )
    assert result["n8_low"] < theory["shear"] < result["n8_high"]
    assert result["matched_constant"] < result["n8_low"]


def test_cumulative_thresholds_use_the_physical_box_volume():
    """Recover drops per litre from integer counts and the simulated volume."""
    data = json.loads(DATA.read_text())
    case = next(row for row in data["cases"] if row["name"] == "physical")
    volume_litres = case["boxes"] * case["slots"] / 600_000
    assert [(r["minutes"], r["mass_at_least"]) for r in data["physical"]] == [
        (16, 4),
        (90, 8),
    ]
    for row in data["physical"]:
        np.testing.assert_allclose(
            row["cuda_per_L"], row["clusters"] / volume_litres
        )
