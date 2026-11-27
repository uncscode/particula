# %% Paired 20 ppm reaction-clock scenarios, anchored at one measured size
"""The 100 nm wet-concentration slope ratio is a scenario clock multiplier."""

import numpy as np
import pandas as pd
from inverse import inputs, PACKET
from joint import JointModel


def main():
    src = pd.read_csv(PACKET / "population/author-retrieval.csv").pivot(
        index="time_h", columns="diameter_nm", values="sodium_fraction"
    )
    thresholds = pd.read_csv(PACKET / "population/crossovers.csv")
    mean = thresholds.query("shape=='mean'").iloc[0].threshold
    ds, t, rh, y, sd, c, w = inputs()
    m = JointModel(y, 2 * sd + 0.005, c, w)
    spans, _ = m.extremes()
    d200, d300 = [list(ds).index(x) for x in [200, 300]]
    clock = 0.2254 / 0.6523
    rows = []
    # Compare at low-exposure age giving 200 nm an observed-equivalent 2 h.
    # Vary only the 300 nm clock and choose tabulated times to avoid kinetic extrapolation.
    j = list(t).index(2.0)
    for age300 in [0.5, 1, 1.5, 2, 2.5, 3]:
        k = list(t).index(age300)
        env = m.condition_pair(d200 * m.T + j, d300 * m.T + k, spans[d200, j], 64)
        r = src.loc[age300, 300] / src.loc[2, 200]
        rows.append(
            (
                2 / clock,
                clock,
                age300 / 2,
                age300,
                2,
                r,
                r / mean,
                env[0],
                env[-1],
                env[0] / thresholds.graph_hi.max(),
                env[-1] / thresholds.graph_lo.min(),
            )
        )
    pd.DataFrame(
        rows,
        columns=[
            "scenario_hours_at_20_ppm",
            "assumed_clock_200",
            "clock_300_relative_to_200",
            "equivalent_200ppm_hours_for_300",
            "equivalent_200ppm_hours_for_200",
            "author_fraction_ratio",
            "central_mean_mass_ratio",
            "fraction_ratio_lo",
            "fraction_ratio_hi",
            "airmass_ratio_all_spectra_lo",
            "airmass_ratio_all_spectra_hi",
        ],
    ).to_csv(PACKET / "challenges/exposure-clock.csv", index=False)
    # Mean-spectrum break-even, linear interpolation of the measured age curve.
    for label, target in zip(thresholds["shape"], thresholds.threshold):
        cross = (
            np.interp(target * src.loc[2, 200], np.r_[0, src[300]], np.r_[0, src.index])
            / 2
        )
        print(label, "relative 300 nm clock at break-even", cross)


if __name__ == "__main__":
    main()
