"""Extract only the published plot quantities from regenerated study outputs."""

import argparse
import gzip
from pathlib import Path

import pandas as pd


def matched(seasonal, frequency, keys):
    """Join cloud and flux changes from the same modeled comparison."""
    cloud = pd.read_csv(
        seasonal / "analysis" / f"resolved-{frequency}-cloud.csv"
    )
    suffix = "csv.gz" if frequency == "diurnal" else "csv"
    rad = pd.read_csv(seasonal / "radiation" / f"resolved-{frequency}.{suffix}")
    cloud_metrics = ["delta_activated_cm3", "delta_median_um"]
    if frequency == "seasonal":
        cloud_metrics = [
            "dark_activated_cm3",
            "activated_cm3",
            "dark_median_um",
            "median_um",
        ]
    flux = ["delta_up_site_wm2"]
    if frequency == "seasonal":
        flux.append("dark_up_site_wm2")
    cloud = cloud.loc[cloud.model.eq("ranked"), keys + cloud_metrics]
    rad = rad.loc[rad.model.eq("ranked"), keys + flux]
    if set(map(tuple, cloud[keys].values)) != set(map(tuple, rad[keys].values)):
        raise ValueError(f"Unpaired {frequency} cloud and flux groups")
    return cloud.merge(rad, on=keys, validate="one_to_one").sort_values(keys)


def prepare(run_root, output):
    """Write CSVs, preserving the precision of the source result tables."""
    output.mkdir(parents=True, exist_ok=True)
    regional = run_root / "seasonal"
    for frequency, keys in [
        ("monthly", ["month", "profile", "w"]),
        ("diurnal", ["season", "solar_hour", "profile", "w"]),
    ]:
        matched(regional, frequency, keys).to_csv(
            output / f"{frequency}.csv", index=False
        )
    seasons = matched(regional, "seasonal", ["season", "profile", "w"])
    case = seasons.query("season == 'JJA' and profile == 'clean' and w == 0.25")
    if len(case) != 1:
        raise ValueError("Missing the clean, slow summer comparison")
    case.to_csv(output / "summer-pair.csv", index=False)
    cohorts = pd.read_csv(regional / "sampling/cohort-months.csv")
    july = cohorts.query("model == 'ranked' and month == 7")
    july[
        [
            "profile",
            "w",
            "lost_seed_cm3",
            "gained_field_cm3",
            "shared_growth_um",
            "exit_selection_um",
            "entry_selection_um",
        ]
    ].to_csv(output / "july-cohorts.csv", index=False)
    spectra = pd.read_csv(
        regional / "analysis/seasonal-fine-bin-differences.csv.gz"
    )
    data = spectra.to_csv(index=False).encode()
    (output / "size-spectra.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    film = pd.read_csv(run_root / "film/replicates.csv")
    selected = film.query("precursor == 'limonene' and constant_rh")
    if set(selected.trial) != {"S4", "S5", "S7"}:
        raise ValueError("Expected the three constant-humidity limonene trials")
    selected[
        ["trial", "mass_fraction", "kappa", "kappa_bar", "water_ratio"]
    ].to_csv(output / "film-water.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.run_root, args.output_dir)
