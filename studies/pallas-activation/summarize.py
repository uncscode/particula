"""Recover reported percentage-point errors, daily profile decisions and bulk counts."""

import json

import numpy as np
import pandas as pd
from observations import P, screened_scans


def main():
    archived, _ = screened_scans()
    usable = archived[archived.valid & (archived.kappa > 0)]
    # Effective Kelvin diameter coefficient in nm for the archived conversion.
    recovered = 4 * (2.09915 / usable.D) ** 3 / (
        27 * np.log1p(usable.SS / 100) ** 2
    )
    np.testing.assert_allclose(recovered.round(2), usable.kappa, atol=1e-10)
    fits = (
        pd.read_csv(P / "fits.csv")
        .query("model == 'two organics'")
        .set_index("date")
    )
    empirical = (
        pd.read_csv(P / "comparator.csv").set_index("date").loc[fits.index]
    )
    profile = pd.read_csv(P / "profiles.csv").set_index("date").loc[fits.index]
    curves = pd.read_csv(P / "curves.csv")
    daily = (
        curves.assign(square_error=(curves.observed - curves.fitted) ** 2)
        .groupby(["date", "SS"])
        .square_error.mean()
        .groupby("date")
        .mean()
        ** 0.5
    )
    np.testing.assert_allclose(daily.loc[fits.index], fits.rmse, atol=2e-10)
    assert len(fits) == 27 and len(curves) == 1823
    difference = 100 * (fits.rmse - empirical.rmse)
    generator = np.random.default_rng(20261124)
    draws = generator.integers(0, len(fits), (20000, len(fits)))
    bulk = pd.read_csv(P / "bulk.csv").query("model == 'two organics'")
    baseline_bulk = pd.read_csv(P / "comparator-bulk.csv")
    assert bulk.n.sum() == 4110
    answer = {
        "archived_kappa_values_reproduced": len(usable),
        "mean_daily_rmse_pp": {
            "two_organics": 100 * fits.rmse.mean(),
            "empirical_kappa": 100 * empirical.rmse.mean(),
        },
        "paired_difference_pp": difference.mean(),
        "paired_date_percentile_95_pp": np.quantile(
            difference.to_numpy()[draws].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "mean_salt_volume_pct": {
            "50_nm": 100 * fits.salt50.mean(),
            "120_nm": 100 * fits.salt120.mean(),
        },
        "dates_positive_enrichment": int((fits.salt120 > fits.salt50).sum()),
        "dates_requiring_enrichment_within_1pp": int(
            (profile.delta_lo > 1e-6).sum()
        ),
        "dates_requiring_both_organics_within_1pp": int(
            (
                (profile.organic_sorbitol_lo > 1e-6)
                & (profile.organic_sorbitol_hi < 1 - 1e-6)
            ).sum()
        ),
        "mean_omitted_setting_rmse_pp": {
            "two_organics": 100
            * pd.read_csv(P / "omitted.csv").omitted_rmse.mean(),
            "empirical_kappa": 100
            * pd.read_csv(P / "comparator-omitted.csv").omitted_rmse.mean(),
        },
        "bulk_mean_ccn_cm3": {
            "measured": bulk.observed.mean(),
            "two_organics": bulk.fitted.mean(),
            "empirical_kappa": baseline_bulk.fitted.mean(),
        },
        "bulk_mean_absolute_error_cm3": {
            "two_organics": bulk.mae.mean(),
            "empirical_kappa": baseline_bulk.mae.mean(),
        },
    }
    (P / "headline.json").write_text(json.dumps(answer, indent=2) + "\n")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
