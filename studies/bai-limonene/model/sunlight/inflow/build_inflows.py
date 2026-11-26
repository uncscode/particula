"""Rebuild mixed-parcel dry background counts.

Run with the main Particula .venv, passing a source
directory containing the two original PaCE v4 ZIP files.
"""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
MD5 = {
    "dmps_pm25_PaCE2022.zip": "8024e751c7ad42a828815d9fbe020e3e",
    "CCN_tot_size_resolved_PaCE2022.zip": "a7d17c687d50526f5889316526676ed9",
    "CCN_tot_total_conc_PaCE2022.zip": "cd38e48e79e82903b876fd279407af9c",
}
LEVELS = [0.1, 0.2, 0.3, 0.6, 1.0]
PROFILES = {
    "clean": 0.10,
    "q25": 0.25,
    "q40": 0.40,
    "q60": 0.60,
    "q75": 0.75,
    "rich": 0.90,
}
P_REF, T_REF = 90000.0, 285.0


def read_group(directory, file, match=""):
    """Read and verify an archive using the author-published MD5."""
    path = directory / file
    if (
        hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        != MD5[file]
    ):
        raise ValueError(f"Source checksum mismatch for {path}")
    with ZipFile(path) as z:
        return pd.concat(
            [
                pd.read_csv(z.open(name))
                for name in z.namelist()
                if name.endswith(".csv") and match in name
            ],
            ignore_index=True,
        ).replace(-9999.9, np.nan)


def checked_thresholds(source):
    """Screen fitted diameters against measured activation fractions."""
    file = "CCN_tot_size_resolved_PaCE2022.zip"
    fits = read_group(source, file, "_k.").rename(
        columns={"Dc [nm]": "D50_nm", "std_error": "D50_fit_se_nm"}
    )
    size = read_group(source, file, "_s.")
    fits["date"] = pd.to_datetime(fits.date, format="%d.%m.%Y").dt.strftime(
        "%Y-%m-%d"
    )
    size["date"] = pd.to_datetime(size.Date, format="%d.%m.%Y").dt.strftime(
        "%Y-%m-%d"
    )
    usable = []
    for row in fits.itertuples():
        if not (row.D50_nm > 0 and row.D50_fit_se_nm > 0 and row.kappa > 0):
            continue
        obs = size.loc[size.date == row.date]
        frac = obs[f"Act({row.SS:.1f})"]
        lower = obs.loc[
            (obs.Dp < row.D50_nm) & frac.between(0, 0.5, inclusive="left"), "Dp"
        ].max()
        upper = obs.loc[
            (obs.Dp > row.D50_nm) & frac.between(0.5, 1, inclusive="right"),
            "Dp",
        ].min()
        if pd.notna(lower) and pd.notna(upper):
            usable.append(
                (
                    row.date,
                    row.SS,
                    row.D50_nm,
                    row.D50_fit_se_nm,
                    row.kappa,
                    lower,
                    upper,
                )
            )
    return pd.DataFrame(
        usable,
        columns=[
            "date",
            "SS_pct",
            "D50_nm",
            "D50_fit_se_nm",
            "kappa",
            "below_half_nm",
            "above_half_nm",
        ],
    )


def select(source, out):
    """Rank dates and integrate each selected dry number spectrum."""
    dmps = read_group(source, "dmps_pm25_PaCE2022.zip")
    channels = list(dmps.columns[1:31])
    center = np.array(channels, dtype=float) * 1e9
    x = np.log10(center)
    edges = np.r_[
        x[0] - (x[1] - x[0]) / 2,
        (x[1:] + x[:-1]) / 2,
        x[-1] + (x[-1] - x[-2]) / 2,
    ]
    widths = np.diff(edges)
    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], np.log10(400.0)) - np.maximum(edges[:-1], 2.0),
    )
    dmps["end"] = pd.to_datetime(dmps.datetime)
    dmps["date"] = dmps.end.dt.strftime("%Y-%m-%d")
    dmps["valid"] = (
        (dmps.total > 0)
        & np.isfinite(dmps[channels]).all(axis=1)
        & (dmps[channels] >= 0).all(axis=1)
        & dmps["in_cloud"].eq(False)
    )
    clear = dmps.loc[dmps.valid].copy()
    if not (
        np.allclose(clear.pressure, 101325.0)
        and np.allclose(clear.temperature, 273.15)
    ):
        raise ValueError("Unexpected DMPS reporting pressure or temperature")
    factor = P_REF / clear.pressure * clear.temperature / T_REF
    clear["accum_cm3"] = (clear[channels].to_numpy() @ overlap) * factor
    clear["total_cm3"] = (clear[channels].to_numpy() @ widths) * factor
    clear["integral_ratio"] = clear.total_cm3 / (clear.total * factor)
    if np.max(np.abs(clear.integral_ratio - 1)) >= 1e-5:
        raise ValueError(
            "Integrated distribution disagrees with reported total"
        )
    thresholds = checked_thresholds(source)
    days = clear.groupby("date").agg(
        n_clear=("total", "size"),
        accum_cm3=("accum_cm3", "mean"),
        accum_p10_cm3=("accum_cm3", lambda s: s.quantile(0.10)),
        accum_p90_cm3=("accum_cm3", lambda s: s.quantile(0.90)),
        total_cm3=("total_cm3", "mean"),
        max_relative_integral_error=(
            "integral_ratio",
            lambda s: np.max(np.abs(s - 1)),
        ),
    )
    days["n_thresholds"] = (
        thresholds.groupby("date")
        .size()
        .reindex(days.index)
        .fillna(0)
        .astype(int)
    )
    eligible = (
        days.loc[(days.n_clear >= 24) & (days.n_thresholds >= 3)]
        .sort_values(["accum_cm3", "date"])
        .copy()
    )
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    eligible["profile"] = ""
    frames = []
    for profile, q in PROFILES.items():
        pos = int(np.floor((len(eligible) - 1) * q + 0.5))
        date = eligible.index[pos]
        if eligible.loc[date, "profile"]:
            raise ValueError("A sampling date selected twice")
        eligible.loc[date, "profile"] = profile
        obs = clear.loc[clear.date == date]
        counts = np.mean(
            obs[channels].to_numpy()
            * (P_REF / 101325 * 273.15 / T_REF)
            * widths,
            axis=0,
        )
        t = thresholds.loc[thresholds.date == date].sort_values("D50_nm")
        kappa = np.exp(
            np.interp(np.log(center), np.log(t.D50_nm), np.log(t.kappa))
        )
        frames.append(
            pd.DataFrame(
                dict(
                    profile=profile,
                    date=date,
                    diameter_nm=center,
                    lower_nm=10 ** edges[:-1],
                    upper_nm=10 ** edges[1:],
                    count_cm3=counts,
                    kappa=kappa,
                )
            )
        )
    eligible.to_csv(out / "eligible-days.csv")
    selected = eligible.reset_index().query('profile!=""')[["date", "profile"]]
    thresholds.merge(selected, on="date").to_csv(
        out / "thresholds.csv", index=False
    )
    pd.concat(frames).to_csv(out / "profiles.csv", index=False)
    return (
        dmps,
        channels,
        edges,
        thresholds.merge(selected, on="date"),
        selected,
    )


def kappa_models(size_nm, fits):
    """Evaluate three effective dry-size hygroscopicity curves."""
    # The slope is the equilibrium threshold transfer from an earlier Pallas
    # training period (19 independent dates for fitting, 10 held-out dates).
    # Freeze extrapolation at each day's measured diameter bounds.
    diam = fits.D50_nm.to_numpy()
    kappa = fits.kappa.to_numpy()
    order = np.argsort(diam)
    diam, kappa = diam[order], kappa[order]
    anchor = fits.loc[fits.SS_pct == 0.3].iloc[0]
    bounded = np.clip(size_nm, diam[0], diam[-1])
    return dict(
        interpolated=np.exp(
            np.interp(np.log(size_nm), np.log(diam), np.log(kappa))
        ),
        smooth=anchor.kappa * (bounded / anchor.D50_nm) ** 1.2097580061242332,
        constant=np.full_like(size_nm, anchor.kappa),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    select(parser.parse_args().source, OUT)
