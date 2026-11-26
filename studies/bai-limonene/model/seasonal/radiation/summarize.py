"""Monthly, season-by-clock and surface-sensitive paired cloudy-column results."""

import numpy as np
import pandas as pd
from optical import HERE

SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
PAIRED_KEYS = [
    "up_black_wm2",
    "up_site_wm2",
    "up_surface02_wm2",
    "up_surface08_wm2",
    "up_direct_black_wm2",
    "up_diffuse_black_wm2",
    "up_swapped_spectra_black_wm2",
    "up_circumsolar_black_wm2",
    "up_circumsolar_site_wm2",
]
KEYS = ["profile", "w", "model"]


def summarize(data, groups, include_days):
    """Calculate time-weighted fluxes, energy-weighted albedos and moment changes."""
    rows = []
    for key, g in data.groupby(groups, sort=True):
        item = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        incident = g.ghi_wm2.sum()
        item.update(
            hours=len(g),
            days=g[["month", "day"]].drop_duplicates().shape[0],
            mean_incoming_wm2=g.ghi_wm2.mean(),
            mean_diffuse_wm2=g.dhi_wm2.mean(),
            sunlit_hours=int(g.ghi_wm2.gt(0).sum()),
            tau550_dark=(g.tau_550nm - g.delta_tau_550nm).mean(),
            delta_tau550=g.delta_tau_550nm.mean(),
            scatter550_dark=(g.scatter_550nm - g.delta_scatter_550nm).mean(),
            delta_scatter550=g.delta_scatter_550nm.mean(),
            dark_g550=(g.g_550nm - g.delta_g_550nm).mean(),
            delta_g550=g.delta_g_550nm.mean(),
        )
        for field in PAIRED_KEYS:
            d = g["delta_" + field]
            item["dark_" + field] = (g[field] - d).mean()
            item["delta_" + field] = d.mean()
        for target in ("black", "site"):
            src = "up_" + target + "_wm2"
            item["dark_albedo_" + target] = (
                (g[src] - g["delta_" + src]).sum() / incident
                if incident > 0
                else np.nan
            )
            item["delta_albedo_" + target + "_pp"] = (
                100 * g["delta_" + src].sum() / incident
                if incident > 0
                else np.nan
            )
        item["delta_black_absorbed_wm2"] = (
            -g.delta_up_black_wm2 - g.delta_down_black_wm2
        ).mean()
        if include_days:
            daily = g.groupby(["month", "day"])[
                ["delta_up_black_wm2", "delta_up_site_wm2"]
            ].mean()
            item["day_min_delta_up_black_wm2"] = daily.delta_up_black_wm2.min()
            item["day_max_delta_up_black_wm2"] = daily.delta_up_black_wm2.max()
            item["day_min_delta_up_site_wm2"] = daily.delta_up_site_wm2.min()
            item["day_max_delta_up_site_wm2"] = daily.delta_up_site_wm2.max()
            item["days_lower_site_flux"] = int(
                (daily.delta_up_site_wm2 < -1e-5).sum()
            )
            item["hour_min_delta_up_site_wm2"] = g.delta_up_site_wm2.min()
            item["hour_max_delta_up_site_wm2"] = g.delta_up_site_wm2.max()
        for name, sign in (
            ("wet_path_gm2", 1.0),
            ("column_reff_um", -1.0),
            ("tau_550nm", 1.0),
        ):
            item["mean_dlog_" + name] = (
                sign * np.log(g[name] / (g[name] - g["delta_" + name])).mean()
            )
        item["mean_dlog_Qext"] = (
            item["mean_dlog_tau_550nm"]
            - item["mean_dlog_wet_path_gm2"]
            - item["mean_dlog_column_reff_um"]
        )
        rows.append(item)
    return pd.DataFrame(rows)


def write(df, name):
    """Retain the study table with sufficient significant digits for plotting."""
    df.to_csv(HERE / name, index=False, float_format="%.8g")
