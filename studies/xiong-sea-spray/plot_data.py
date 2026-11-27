"""Extract just the samples and model values used by the five public figures."""

import argparse
from pathlib import Path

import pandas as pd

PUBLIC = Path(__file__).resolve().parents[2] / "docs/data/xiong-sea-spray"
MODELS = {
    "direct": ("primary", "direct"),
    "parallel_neutral": ("primary", "parallel_neutral"),
    "retention_neutral": ("retention", "retention_neutral"),
    "direct_blank": ("blank", "direct"),
}


def time_comparison(fits, filename, fields, prediction):
    """Align four fits on the observation time and original calibration role."""
    result = None
    for series, (stage, network) in MODELS.items():
        rows = pd.read_csv(fits / stage / filename)
        rows = rows.loc[rows.network.eq(network)]
        if filename == "growth.csv":
            rows = rows.loc[rows.diameter_nm.eq(200) & rows.rh_pct.eq(80)]
        observed = rows.set_index("time_h")[fields].sort_index()
        if result is None:
            result = observed.copy()
        else:
            pd.testing.assert_frame_equal(observed, result[fields])
        result[series] = rows.set_index("time_h")[prediction]
    return result.reset_index()


def extract(run, output):
    """Work from a completed private replay or the original research packet."""
    fits = run / "aiomfac"
    growth = pd.read_csv(fits / "primary/growth.csv")
    sizes = growth.query(
        "network == 'parallel_neutral' and source == 'S1' and time_h == 3 and rh_pct in [60,80]"
    )
    assert len(sizes) == 14 and not sizes.fitted.any()
    budget = pd.read_csv(fits / "results/budget.csv")
    budget = budget.query(
        "spectrum == 'mean' and time_h == 3 and network in ['direct','neutral','parallel_neutral','retention_neutral']"
    )
    phase = fits / "coating/parameter-fit"
    observations = pd.read_csv(phase / "phase-data.csv")
    isothermal = (
        pd.read_csv(phase / "predictions.csv")
        .query("salt == 'AS'")
        .rename(columns={"OVF_pct": "ovf_pct", "model_RH": "predicted_rh_pct"})
    )
    microscope = (
        pd.read_csv(phase / "chamber-path.csv")
        .assign(salt="NaCl")
        .rename(
            columns={"OVF_pct": "ovf_pct", "model_RH_610Pa": "predicted_rh_pct"}
        )
    )
    keys = ["salt", "ovf_pct"]
    prediction = pd.concat([isothermal, microscope])[
        keys + ["predicted_rh_pct"]
    ]
    onset = observations.merge(prediction, on=keys, validate="one_to_one")
    onset["crystallization_rh_pct"] = onset.erh_pct.where(
        onset.srh_status.eq("not_observed")
    )
    tables = {
        "growth-size": sizes[
            ["diameter_nm", "rh_pct", "gf", "spread", "predicted_gf"]
        ].sort_values(["rh_pct", "diameter_nm"]),
        "growth-time": time_comparison(
            fits, "growth.csv", ["gf", "spread", "fitted"], "predicted_gf"
        ),
        "ion-time": time_comparison(
            fits,
            "ion-ratios.csv",
            [
                "late_to_early_peak_area_ratio",
                "raster_edge_min",
                "raster_edge_max",
                "fitted",
            ],
            "predicted_ratio",
        ),
        "size-budget": budget[
            ["network", "sulfate200_ug_m3_decade", "sulfate300_ug_m3_decade"]
        ],
        "phase-onset": onset[
            keys
            + [
                "srh_status",
                "srh_pct",
                "srh_sd_pct",
                "predicted_rh_pct",
                "crystallization_rh_pct",
            ]
        ],
    }
    output.mkdir(exist_ok=True, parents=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Private numerical replay directory"
    )
    parser.add_argument("--output", type=Path, default=PUBLIC)
    args = parser.parse_args()
    extract(args.run, args.output)
