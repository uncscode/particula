"""Pack only the plotted SABRE model results from a completed replay."""

import argparse
from pathlib import Path

import pandas as pd
from figure_archive import write_archive
from summary_cold_acid import compile_cold_acid

HERE = Path(__file__).resolve().parent


def collect(source):
    """Extract graph vertices, annotations and flight medians."""

    def csv(name):
        return pd.read_csv(source / (name + ".csv"))

    curves = csv("figure-data")[
        [
            "n2o_lo_ppbv",
            "diameter_nm",
            "fit_number",
            "small_number",
            "fit_surface",
            "retained_surface",
            "small_extinction",
            "full_extinction",
        ]
    ]
    annotations = csv("comparisons").query("case == 'reference'")[
        [
            "n2o_lo_ppbv",
            "area_fraction",
            "ext_fraction",
        ]
    ]
    flights = csv("flight-windows")
    predictions = ["full_ext", "without_below150_ext", "without_mode_ext"]
    residuals = flights[predictions].sub(flights.observed_ext, axis=0)
    medians = residuals.groupby(flights.flight).median()
    medians["count"] = flights.groupby("flight").size()
    sens = csv("flight-sensitivity").query(
        "n2o_lo_ppbv == 0 and imaginary_index == 0"
    )[["real_index", "diameter_scale", "median_full_bias_Mm1"]]
    field = csv("chemistry/tracer-bin-predictions").query(
        "case == 'aiomfac' and basis == 'wet'"
    )[["model", "n2o_lo", "fitted"]]
    profile = csv("chemistry/field-channel-ratios")[
        [
            "small_to_other",
            "median_mode_rate_share",
            "median_rate_s",
        ]
    ]
    toc = csv("tracer-contrast").query(
        "case == 'aiomfac' and basis == 'wet' and subset == 'all_groups'"
    )[
        [
            "min_small_weight",
            "max_small_weight",
            "field_max_over_min",
            "two_population_max_over_min_bound",
        ]
    ]
    hydration = csv("chemistry/hydration").query(
        "case in ['aiomfac', 'ideal', 'inlet_10pct_rh']"
    )
    area_change = (
        (100 * (hydration.wet_total_area / hydration.dry_total_area - 1))
        .groupby(hydration.case)
        .median()
        .rename("change_pct")
        .reset_index()
    )
    contrasts = csv("tracer-contrast").query(
        "case == 'aiomfac' and basis == 'wet'"
    )
    quarters = csv("integrated-contrasts").query("subset != 'all'")
    # The Fried et al. abstract reports these two approximate compositions
    # at 230 K (doi:10.1029/93JD01907). No plot digitization is required.
    acid_uptake = pd.DataFrame(
        {"acid_mass_pct": [60, 70], "uptake_probability": [0.077, 0.146]}
    )
    return dict(
        curves=curves,
        annotations=annotations,
        flights=medians.reset_index(),
        sensitivity=sens,
        field=field,
        profile=profile,
        toc=toc,
        contrast=contrasts[
            [
                "groups",
                "field_max_over_min",
                "two_population_max_over_min_bound",
            ]
        ],
        quarters=quarters[
            [
                "median_small_wet_fraction",
                "median_shared_effective_gamma",
                "median_separate_effective_gamma",
            ]
        ],
        acid_uptake=acid_uptake,
        area_change=area_change,
    )


def main():
    """Pack graph columns deterministically at eight significant digits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE)
    args = parser.parse_args()
    packed = {}
    for name, table in collect(args.source).items():
        if table.empty or table.isna().any().any():
            raise ValueError(f"Missing plotted values in {name}")
        packed[name] = {
            key: [float(f"{value:.8g}") for value in column]
            if column.dtype.kind == "f"
            else column.tolist()
            for key, column in table.items()
        }
    packed["cold_acid"] = compile_cold_acid(args.source)
    write_archive(args.source, packed)


if __name__ == "__main__":
    main()
