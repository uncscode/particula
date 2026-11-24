"""Extract the reported cases from a rebuilt Mt. Soledad study directory."""

import argparse
from pathlib import Path

import pandas as pd

OUTPUT = Path(__file__).resolve().parent


def select(root):
    """Keep the primary cut, instrument settings, recipes and held dates."""

    def table(source, name, query=None):
        data = pd.read_csv(root / source)
        if query:
            data = data.query(query)
        if data.empty:
            raise ValueError(f"No matching records in {source}")
        data.to_csv(OUTPUT / name, index=False, float_format="%.10g")
        return data

    table(
        "final/summary.csv",
        "integrated-counts.csv",
        "scenario=='may' and split=='held' and band=='all' and metric in ['obs','rh','bypass_obs','denuded_obs','common']",
    )
    pairs = pd.read_csv(root / "final/120-250nm-pairs.csv")
    pairs.groupby("day")[
        ["bypass_obs", "denuded_obs", "obs", "common", "rh"]
    ].mean().reset_index().to_csv(
        OUTPUT / "large-seed-date-means.csv", index=False, float_format="%.10g"
    )
    table(
        "extension/diagnostics.csv",
        "processed-counter.csv",
        "period=='all' and S==0.8 and metric=='pulse_normalized'",
    )
    table(
        "mechanisms/droplets/scores.csv",
        "optical-controls.csv",
        "source=='selected' and split=='test' and model in ['uniform_amplitude','partially_delayed','uniform_joint','scalar_loss'] and metric in ['model_ratio','model_mean_shift','obs_mean_shift','observed_ratio']",
    )
    table(
        "mechanisms/droplets/parameters.csv",
        "optical-training.csv",
        "source=='selected' and floor==0.5",
    )
    field = "field-mixtures/"
    uniform = "clock=='raw_UTC' and cut_nm==1000 and held and recipe=='low' and closure=='sodium'"
    table(
        field + "comparison/paired-contrasts.csv",
        "growth-contrasts.csv",
        uniform
        + " and size_nm==60 and metric in ['ae_pred_mean','ae_pred_mean_RH_AS','ae_pred_mean_RH_NaCl']",
    )
    table(
        field + "comparison/growth-scores.csv",
        "growth-scores.csv",
        uniform + " and size_nm==60 and population=='uniform'",
    )
    table(
        field + "comparison/activation-scores.csv",
        "activation.csv",
        uniform + " and S==0.8 and mapping=='previous_mapping'",
    )
    day = pd.read_csv(root / field / "comparison/growth-by-date.csv.gz").query(
        "clock=='raw_UTC' and cut_nm==180 and recipe=='low' and closure=='sodium' and population=='uniform'"
    )
    day[["date", "size_nm", "model", "gf_mean", "pred_mean"]].to_csv(
        OUTPUT / "small-filter-growth.csv", index=False, float_format="%.10g"
    )
    table(field + "observations/standards.csv", "salt-standards.csv")
    table(
        field + "observations/partial-scores.csv",
        "near-dry-class.csv",
        "held and cut_nm==1000 and recipe=='low' and model=='partial'",
    )
    table(
        field + "chem/film-bounds.csv",
        "c16-bounds.csv",
        "mask==2 and filter_id in ['EPP337','EPP349','EPP538']",
    )
    # Average the six-hour June 10 recipes on the same held outlet benchmark.
    exposure = pd.read_csv(root / field / "films/final/held-summary.csv").query(
        "filter_id=='EPP538' and recipe=='high_C16_max2'"
    )
    exposure.groupby(
        ["filter_id", "recipe", "exposure", "model"], as_index=False
    )[
        [
            "count_ratio",
            "wet_um",
            "bypass_wet_um",
            "binned_wet_um",
            "mean_exposure_s",
        ]
    ].mean().to_csv(OUTPUT / "june-film.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    select(parser.parse_args().study_root)
