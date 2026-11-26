"""Keep compact high-resolution, interpolation, surface and chemistry comparisons."""

import numpy as np
import pandas as pd
from optical import TMP
from summarize import KEYS, SEASON, summarize, write


def load(tag, prefixes=()):
    """Read retained computational groups and assign UTC-day and solar-hour bins."""
    paths = list((TMP / tag).glob("*.csv.gz"))
    if prefixes:
        paths = [p for p in paths if p.name.startswith(prefixes)]
    if not paths:
        return pd.DataFrame()
    d = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    d["day"] = d.hour_in_week // 24
    d["solar_hour"] = np.floor(d.local_solar_hour % 24).astype(int)
    d["season"] = d.month.map(SEASON)
    return d


def compact(d, keys, by_day):
    """Keep the flux, cloud-state and variability terms used for grid comparisons."""
    relevant = [
        "hours",
        "days",
        "sunlit_hours",
        "mean_incoming_wm2",
        "mean_diffuse_wm2",
        "dark_up_black_wm2",
        "delta_up_black_wm2",
        "dark_up_site_wm2",
        "delta_up_site_wm2",
        "delta_up_surface02_wm2",
        "delta_up_surface08_wm2",
        "delta_albedo_black_pp",
        "delta_albedo_site_pp",
        "delta_tau550",
        "tau550_dark",
        "mean_dlog_tau_550nm",
        "mean_dlog_wet_path_gm2",
        "mean_dlog_column_reff_um",
        "mean_dlog_Qext",
        "dark_g550",
        "delta_g550",
        "delta_scatter550",
        "day_min_delta_up_site_wm2",
        "day_max_delta_up_site_wm2",
        "days_lower_site_flux",
        "delta_up_circumsolar_site_wm2",
        "delta_up_swapped_spectra_black_wm2",
    ]
    # Early sensitivity files may predate the optional sky perturbation. Their
    # baseline and surface fluxes already used the final measured forcing.
    for field in ("up_circumsolar_black_wm2", "up_circumsolar_site_wm2"):
        if field not in d:
            d[field] = np.nan
            d["delta_" + field] = np.nan
    full = summarize(d, keys, by_day)
    return full[keys + [c for c in relevant if c in full]]


def run():
    """Compare matched numerical, surface and chemistry treatments by week."""
    d = load("resolved")
    r = d.query('model=="ranked"')
    assert (
        len(r) == 48 * 168
        and not r.duplicated(KEYS + ["month", "hour_in_week"]).any()
    )
    for suffix, keys in [
        ("monthly", ["month"]),
        ("seasonal", ["season"]),
        ("diurnal", ["season", "solar_hour"]),
    ]:
        ext = ".csv.gz" if suffix == "diurnal" else ".csv"
        write(
            summarize(r, KEYS + keys, suffix != "diurnal"),
            "resolved-" + suffix + ext,
        )
    alternatives = d.query('model!="ranked"')
    if len(alternatives):
        write(
            compact(alternatives, KEYS + ["month"], True),
            "resolved-july-alternatives.csv",
        )


if __name__ == "__main__":
    run()
