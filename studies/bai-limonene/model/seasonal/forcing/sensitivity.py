"""Construct two different UV-index-constrained dose alternatives and diagnose sampling."""

from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from build import OUT, BASE

SEASONS = ("DJF", "MAM", "JJA", "SON")


def get_ozone(raw):
    if raw is not None:
        sha = hashlib.md5(raw.read_bytes()).hexdigest()
        if sha != "c2b00d277776d5388a66cdc463cd6b2e":
            raise ValueError(f"Check the laboratory file revision (MD5 {sha})")
        # Column 1 wavelength in vacuum, column 9 223 K cross section.
        original = np.loadtxt(raw, skiprows=45, usecols=(0, 8))
        original = original[
            (original[:, 0] >= 279.7) & (original[:, 0] <= 400.3)
        ]
        # Average to the original action spectrum's 0.25 nm resolution.
        bins = np.arange(280, 400.01, 0.25)
        cross = np.array(
            [
                original[
                    (original[:, 0] >= w - 0.125)
                    & (original[:, 0] < w + 0.125),
                    1,
                ].mean()
                for w in bins
            ]
        )
        pd.DataFrame(
            {"wavelength_nm": bins, "ozone_cm2_molecule_223K": cross}
        ).to_csv(OUT / "ozone_223k.csv", index=False)
    return pd.read_csv(OUT / "ozone_223k.csv")


def angular_adjustment(d, band):
    frac = (d.direct_horizontal_wm2 / d.ghi_wm2.replace(0, np.nan)).fillna(0)
    mu = d.mu0
    m = 1 / (mu + 0.50572 * (96.07995 - np.rad2deg(np.arccos(mu))) ** (-1.6364))

    def tau(lam):
        return (
            0.008569
            * lam ** (-4)
            * (1 + 0.0113 * lam ** (-2) + 0.00013 * lam ** (-4))
        )

    return 2 + frac * np.exp(
        -(tau(0.355 if band == "uva" else 0.310) - tau(0.55)) * m
    ) * (1 / mu.replace(0, np.nan) - 2).fillna(0)


def spectral_for_cross(ozone):
    points = pd.read_csv(BASE / "field-uv/digitized.csv")
    groups = points[points.detector == "DM-SR"].groupby("panel")
    w = ozone.wavelength_nm.to_numpy()

    def interp(name):
        g = groups.get_group(name).sort_values("wavelength_nm")
        return np.interp(w, g.wavelength_nm, g.flux)

    photons = np.where(w <= 310, interp("a"), interp("b")).clip(0)
    photons[w < 293] = 0
    energy = photons * 1e4 * 6.62607015e-34 * 299792458 / (w * 1e-9)
    action = pd.read_csv(BASE / "solar/spectrum.csv")
    standard = pd.read_csv(BASE / "solar/rates.csv").set_index("species")
    ery = np.where(
        w <= 298,
        1,
        np.where(
            w <= 328, 10 ** (0.094 * (298 - w)), 10 ** (0.015 * (140 - w))
        ),
    )
    xs = ozone.ozone_cm2_molecule_223K.to_numpy().clip(0)

    def coeff(column_du):
        transmission = np.exp(-column_du * 2.687e16 * xs)
        result = {}
        for band, select in [("uva", w >= 315), ("uvb", w <= 315)]:
            e = energy[select] * transmission[select]
            total = np.trapezoid(e, w[select])
            c = [40 * np.trapezoid(e * ery[select], w[select]) / total]
            act = select & (w <= 370)
            for sp in ["mvos", "lvos"]:
                cross = np.interp(
                    w[act], action.wavelength_nm, action[f"action_{sp}_cm2"]
                )
                c.append(
                    np.trapezoid(
                        photons[act] * transmission[act] * cross, w[act]
                    )
                    / total
                    / standard.loc[sp, "standard_j_per_s"]
                )
            result[band] = np.array(c)
        return result

    return coeff


def main(raw):
    ozone = get_ozone(raw)
    coefficients = spectral_for_cross(ozone)
    d = pd.concat(
        [pd.read_csv(OUT / (name + ".csv")) for name in ["history", "forcing"]],
        ignore_index=True,
    )
    d = d.sort_values(["month", "hour_in_week"]).reset_index(drop=True)
    d["season"] = d.month.map(lambda m: SEASONS[m % 12 // 3])
    day = pd.to_datetime(d.datetime_utc, utc=True).dt.day
    d["fit_day"] = day % 2 == 1
    for band in ["uva", "uvb"]:
        d["angular_" + band] = angular_adjustment(d, band)
    checks = []
    uvbands = ["uva", "uvb"]
    for season in SEASONS:
        mask = d.season == season
        train = mask & d.fit_day
        test = mask & ~d.fit_day
        station = d.loc[train, "fmi_uv_index"].sum()
        model = d.loc[train, "two_band_uv_index"].sum()
        gray = station / model
        train_band = {
            band: d.loc[train, "nasa_" + band + "_wm2"].sum()
            for band in uvbands
        }

        def residual(dx):
            coeff = coefficients(dx)
            return sum(train_band[b] * coeff[b][0] for b in uvbands) - station

        cap = 3000
        dx = (
            brentq(residual, -25, cap)
            if residual(-25) * residual(cap) < 0
            else min((-25, cap), key=lambda x: abs(residual(x)))
        )
        coeff = coefficients(dx)
        d.loc[mask, "diagnostic_uvi_shape"] = sum(
            d.loc[mask, "nasa_" + b + "_wm2"] * coeff[b][0] for b in uvbands
        )
        d.loc[mask, "diagnostic_uvi_gray"] = (
            d.loc[mask, "two_band_uv_index"] * gray
        )
        for i, sp in enumerate(["fast", "slow"]):
            d.loc[mask, "j_" + sp + "_scale_uvi_gray"] = (
                d.loc[mask, "j_" + sp + "_scale"] * gray
            )
            d.loc[mask, "j_" + sp + "_scale_uvi_shape"] = sum(
                d.loc[mask, "nasa_" + b + "_wm2"]
                * d.loc[mask, "angular_" + b]
                * coeff[b][i + 1]
                for b in uvbands
            )
        episode = d[mask & (d.hour_in_week >= 0)]
        di = {
            "season": season,
            "train_odd_hours": int(train.sum()),
            "test_even_hours": int(test.sum()),
            "gray_factor": gray,
            "equivalent_extra_slant_DU": dx,
            "slant_bound_hit": dx in (-25, cap),
            "holdout_FMI_uvi_mean": d.loc[test, "fmi_uv_index"].mean(),
            "holdout_central_uvi_mean": d.loc[test, "two_band_uv_index"].mean(),
            "holdout_gray_uvi_mean": d.loc[test, "diagnostic_uvi_gray"].mean(),
            "holdout_shape_uvi_mean": d.loc[
                test, "diagnostic_uvi_shape"
            ].mean(),
            "week_mean_central_jfast_scale": episode.j_fast_scale.mean(),
            "week_mean_gray_jfast_scale": episode.j_fast_scale_uvi_gray.mean(),
            "week_mean_shape_jfast_scale": episode.j_fast_scale_uvi_shape.mean(),
        }
        checks.append(di)
    out = [
        "month",
        "hour_in_week",
        "datetime_utc",
        "season",
        "fit_day",
        "fmi_uv_index",
        "two_band_uv_index",
        "diagnostic_uvi_gray",
        "diagnostic_uvi_shape",
        "j_fast_scale_uvi_gray",
        "j_slow_scale_uvi_gray",
        "j_fast_scale_uvi_shape",
        "j_slow_scale_uvi_shape",
    ]
    pd.DataFrame(checks).to_csv(OUT / "uvi_sensitivity_checks.csv", index=False)
    for fname, mask in [
        ("forcing", d.hour_in_week >= 0),
        ("history", d.hour_in_week < 0),
    ]:
        d.loc[mask, out].to_csv(
            OUT / (fname + "_uvi_sensitivity.csv"),
            index=False,
            float_format="%.9g",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-ozone", type=Path)
    main(p.parse_args().raw_ozone)
