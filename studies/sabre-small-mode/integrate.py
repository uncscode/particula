"""Join optical and chemical predictions on the original SABRE windows."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ["flight", "start_s"]


def keyed(path, **filters):
    """Index each selected source by its original unique flight window."""
    frame = pd.read_csv(HERE / path)
    for name, value in filters.items():
        frame = frame.loc[frame[name] == value]
    return frame.set_index(KEY, verify_integrity=True).sort_index()


def tracer_contrasts(flights):
    """Bound effective uptake contrasts for any fixed pair of probabilities."""
    groups = pd.read_csv(HERE / "chemistry/tracer-bin-predictions.csv")
    groups = groups.loc[groups.model == "two-class"]
    hydration = pd.read_csv(HERE / "chemistry/hydration.csv").merge(
        flights.n2o_ppbv.reset_index(), on=KEY, validate="many_to_one"
    )
    hydration = hydration.loc[
        hydration.flight.isin((20230308, 20230311, 20230315))
    ].copy()
    hydration["n2o_lo"] = (hydration.n2o_ppbv // 5 * 5).astype(int)
    results = []
    for (case, basis), full in groups.groupby(["case", "basis"]):
        for label, group in [
            ("all_groups", full),
            ("omit_315_to_320", full.loc[full.n2o_lo != 315]),
        ]:
            weights = group[["a", "b"]]
            if not (weights > 0).all().all():
                raise ValueError(
                    "The ratio bound requires positive area weights"
                )
            # Each group-to-group ratio is a convex average of the two
            # weight ratios, with weights set by the denominator group's loss.
            bounds = weights.max() / weights.min()
            windows = hydration.loc[
                (hydration.case == case) & hydration.n2o_lo.isin(group.n2o_lo)
            ]
            sample_weights = windows[["wet_small_area", "wet_other_area"]].div(
                windows[f"{basis}_total_area"], axis=0
            )
            sample_weights.columns = ["a", "b"]
            sample_means = sample_weights.groupby(windows.n2o_lo).mean()
            np.testing.assert_allclose(
                sample_means, group.set_index("n2o_lo").sort_index()[["a", "b"]]
            )
            if len(windows) != group.n_windows.sum():
                raise ValueError("Mismatch with the chemical fit population")
            any_weight_bound = (
                sample_weights.max() / sample_weights.min()
            ).max()
            results.append(
                {
                    "case": case,
                    "basis": basis,
                    "subset": label,
                    "groups": len(group),
                    "windows": group.n_windows.sum(),
                    "min_small_weight": weights.a.min(),
                    "max_small_weight": weights.a.max(),
                    "min_field_gamma": group.gamma_marker.min(),
                    "max_field_gamma": group.gamma_marker.max(),
                    "field_max_over_min": group.gamma_marker.max()
                    / group.gamma_marker.min(),
                    "small_only_max_over_min": bounds.a,
                    "other_only_max_over_min": bounds.b,
                    "two_population_max_over_min_bound": bounds.max(),
                    "arbitrary_sample_weight_upper_bound": any_weight_bound,
                }
            )
    pd.DataFrame(results).to_csv(HERE / "tracer-contrast.csv", index=False)


def main():
    """Compare the same two particle definitions across optical and gas loss."""
    flights = keyed("flight-windows.csv")
    wet = keyed("chemistry/hydration.csv", case="aiomfac")
    shared = keyed("chemistry/rates-shared.csv")
    separate = keyed("chemistry/rates-two-class.csv")
    for frame in (wet, shared, separate):
        pd.testing.assert_index_equal(flights.index, frame.index)
    mode_area = wet.wet_small_area / wet.wet_total_area
    n_flights = flights.index.get_level_values("flight").nunique()
    records = []
    for population, dry_area, residual_ext, wet_area, loss_share in [
        (
            "fitted_small_mode",
            flights.removed_mode_area,
            flights.without_mode_ext,
            mode_area,
            "mode_loss_fraction",
        ),
        (
            "measured_diameter_below_150nm",
            flights.removed_below150_area,
            flights.without_below150_ext,
            wet.wet_below150_area / wet.wet_total_area,
            "cutoff_loss_fraction",
        ),
    ]:
        shares = pd.DataFrame(
            {
                "dry_area": dry_area / flights.full_area,
                "extinction_532nm": 1 - residual_ext / flights.full_ext,
                "wet_area": wet_area,
                "shared_gamma_loss": shared[loss_share],
                "separate_gamma_loss": separate[loss_share],
            }
        )
        if not ((shares >= 0) & (shares <= 1)).all().all():
            raise ValueError("Shares must be finite fractions")
        np.testing.assert_allclose(wet_area, shared[loss_share], atol=1e-8)
        records.append(
            {
                "population": population,
                "windows": len(shares),
                "flights": n_flights,
                **(100 * shares.median()).add_suffix("_pct").to_dict(),
            }
        )
    pd.DataFrame(records).to_csv(
        HERE / "integrated-comparison.csv", index=False
    )

    # The bins order windows by the same property that drives the model split.
    lower, upper = mode_area.quantile([0.25, 0.75])
    groups = {
        "all": np.ones(len(flights), dtype=bool),
        "lower_quartile": mode_area <= lower,
        "upper_quartile": mode_area >= upper,
    }
    ratio = separate.k_total_s / shared.k_total_s
    fits = (
        pd.read_csv(HERE / "chemistry/field-fits.csv")
        .query("case == 'aiomfac' and basis == 'wet'")
        .set_index("model")
    )
    gamma = fits.loc["two-class"]
    effective = (
        mode_area * gamma.gamma_small + (1 - mode_area) * gamma.gamma_other
    )
    contrasts = []
    for label, mask in groups.items():
        selected = mode_area[mask]
        contrasts.append(
            {
                "subset": label,
                "windows": len(selected),
                "flights": selected.index.get_level_values("flight").nunique(),
                "min_small_wet_fraction": selected.min(),
                "median_small_wet_fraction": selected.median(),
                "max_small_wet_fraction": selected.max(),
                "median_shared_effective_gamma": fits.loc[
                    "shared", "gamma_small"
                ],
                "median_separate_effective_gamma": effective[mask].median(),
                "median_shared_k_s1": shared.k_total_s[mask].median(),
                "median_separate_k_s1": separate.k_total_s[mask].median(),
                "median_paired_k_ratio_separate_over_shared": ratio[
                    mask
                ].median(),
                "lower_quartile_cut": lower,
                "upper_quartile_cut": upper,
            }
        )
    pd.DataFrame(contrasts).to_csv(
        HERE / "integrated-contrasts.csv", index=False
    )
    tracer_contrasts(flights)


if __name__ == "__main__":
    main()
