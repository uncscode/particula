"""Recalculate the publication comparisons from the saved Dunne packet."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent


def formation_and_dimers():
    """Compare fixed pathway shares and paired cluster predictions."""
    profile = pd.read_csv(HERE / "profile.csv")
    print("Fixed binary share and median training error factor")
    print(profile.groupby("share").train_factor.median().to_string())

    dimers = pd.read_csv(HERE / "cluster_dimers.csv")
    cloud = dimers.loc[
        dimers.case.str.startswith("CLOUD5_")
        & dimers.chemical.isin(["optimized", "posterior_median"])
    ].copy()
    cloud["log_error_squared"] = (
        np.log(cloud.dimer_cm3 / cloud.comparison_cm3) ** 2
    )
    grouped = cloud.groupby(["hamaker_J", "id", "chemical"])
    if not (grouped.case.nunique() == 3).all():
        raise ValueError("Expected three dimer observations per comparison.")
    errors = np.exp(np.sqrt(grouped.log_error_squared.mean())).unstack()
    if errors.isna().any().any():
        raise ValueError("Missing dimer prediction in paired comparison.")
    for kernel, pairs in errors.groupby(level="hamaker_J"):
        wins = (pairs.optimized < pairs.posterior_median).sum()
        print(
            f"Dimer errors at {kernel:g} J, optimized and marginal medians: "
            f"{pairs.optimized.median():.2f}, "
            f"{pairs.posterior_median.median():.2f}. "
            f"Optimized lower in {wins}/{len(pairs)} pairs."
        )


def cold_scan_and_volume():
    """Evaluate 223 K extrapolation and finite-volume crossings."""
    neutral = pd.read_csv(PACKET / "neutral_33_43rh.csv")
    cold = neutral.loc[neutral.temperature_K == 223]
    held = cold.source_csv_line.isin([332, 366])
    rate = cold.loc[held].sort_values("H2SO4_cm-3")["J_1.7_cm-3_s-1"]
    training = cold.loc[~held, ["H2SO4_cm-3", "NH3_pptv"]]
    print("223 K training gases after withholding the two scan points")
    print(training.to_string(index=False))
    scan = pd.read_csv(HERE / "scan-bounds.csv")
    (upper,) = scan.loc[
        (scan.split == "223_low")
        & (scan["shape"] == "linear")
        & (scan.topology == "full")
        & (scan.scenario == "new_common"),
        "p95",
    ]
    ratio = rate.iloc[1] / rate.iloc[0]
    print(
        f"Observed ratio {ratio:.2f}. "
        f"Ratio divided by upper scenario bound {ratio / upper:.2f}."
    )

    runs = pd.read_csv(HERE / "transport-population.csv")
    window = runs.loc[
        (runs.window_end_h == 18)
        & (runs.dt_s == 30)
        & (runs.line == 336)
        & (runs.source == "full")
    ]
    volumes = window.pivot(
        index="id", columns="law", values="sample_volume_cm3"
    )
    np.testing.assert_array_equal(volumes.attraction, volumes.vapor)
    print(
        f"Paired simulation volumes {volumes.vapor.min():.3f}–"
        f"{volumes.vapor.max():.2f} cm³."
    )
    for law, draws in window.groupby("law"):
        print(f"{law}: {(draws.J10 == 0).sum()}/{len(draws)} zero crossings.")


def warm_peak():
    """Compare warm particle-count errors and acid inputs during the peak."""
    gases = pd.read_csv(PACKET / "simba/wall-sulfur/gas_inputs.csv")
    training = (gases.hour < 3.8) & gases.finite_acid.eq(1)
    scale = (
        gases.loc[training, "acid_observed_cm3"].mean()
        / gases.loc[training, "OH_SO2_acid_cm3"].mean()
    )
    peak_gas = gases.loc[
        gases.hour.between(3.8, 6.0001) & gases.finite_acid.eq(1)
    ]
    relative = (
        scale
        * peak_gas.OH_SO2_acid_cm3.mean()
        / peak_gas.acid_observed_cm3.mean()
    )
    print(f"Peak-window mean acid change {100 * (relative - 1):+.2f}%.")

    with np.load(HERE / "warm-trajectories.npz") as run:
        hours = run["hours"]
        peak = (hours >= 3.8) & (hours <= 6.0001)
        band = run["band"].tolist().index("≥9.7 nm")
        measured = run["observed"][peak, band]
        predicted = run["prediction"][:, :, peak, band]
        families = run["family"].tolist()
    errors = np.sqrt(np.mean((predicted - measured) ** 2, axis=2))
    baseline = errors[:, families.index("measured")]
    for family in ("gas_matched", "gas_absolute"):
        idx = families.index(family)
        reduction = baseline - errors[:, idx]
        peaks = predicted[:, idx].max(axis=1)
        print(
            f"{family}: {(reduction > 0).sum()}/{len(reduction)} improved, "
            f"median reduction {np.median(reduction):.0f} cm⁻³. "
            f"Median peak {np.median(peaks):.0f} versus measured "
            f"{measured.max():.0f} cm⁻³."
        )


if __name__ == "__main__":
    formation_and_dimers()
    cold_scan_and_volume()
    warm_peak()
