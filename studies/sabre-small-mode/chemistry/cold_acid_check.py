"""Compare flat pure-acid predictions with Tabazadeh et al. (1997)."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def acid_weight_fraction(aw, temp):
    """Return H2SO4 wt% for liquid RH in the supported flight range."""
    if not (0.01 <= aw < 0.85 and 185.0 <= temp <= 260.0):
        raise ValueError("Outside Tabazadeh et al. Table 2 domain used here.")
    low = [
        [12.372089320, -0.16125516114, -30.490657554, -2.1133114241],
        [13.455394705, -0.19213122550, -34.285174607, -1.7620073078],
    ]
    middle = [
        [11.820654354, -0.20786404244, -4.8073063730, -5.1727540348],
        [12.891938068, -0.23233847708, -6.4261237757, -4.9005471319],
    ]
    params = low if aw < 0.05 else middle
    y1, y2 = [a * aw**b + c * aw + d for a, b, c, d in params]
    molality = y1 + (temp - 190.0) * (y2 - y1) / 70.0
    return 9800 * molality / (98 * molality + 1000)


def main():
    """Save one binary-acid parameterization comparison per eligible window."""
    frame = pd.read_csv(HERE / "hydration.csv")
    d = frame.query("case == 'no_kelvin' and 0.01 <= rh_liquid < 0.85").copy()
    d["tabazadeh_h2so4_wt_pct"] = [
        acid_weight_fraction(w, t)
        for w, t in zip(d.rh_liquid, d.temperature_k, strict=True)
    ]
    d["aiomfac_h2so4_wt_pct"] = d.other_h2so4_wt_pct
    d["aiomfac_minus_tabazadeh_pct_points"] = (
        d.aiomfac_h2so4_wt_pct - d.tabazadeh_h2so4_wt_pct
    )
    d[
        [
            "flight",
            "start_s",
            "temperature_k",
            "rh_liquid",
            "tabazadeh_h2so4_wt_pct",
            "aiomfac_h2so4_wt_pct",
            "aiomfac_minus_tabazadeh_pct_points",
        ]
    ].to_csv(HERE / "cold-acid.csv", index=False)
    print(
        len(d), np.quantile(d.aiomfac_minus_tabazadeh_pct_points, [0, 0.5, 1])
    )


if __name__ == "__main__":
    main()
