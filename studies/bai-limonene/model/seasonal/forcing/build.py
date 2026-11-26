"""Reconstruct hourly Bai photolysis rates from FMI radiation and CERES UV bands."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent / "sunlight"
LAT, LON = 67.36663, 26.62901
POWER_KEYS = {
    "ALLSKY_SFC_UVA": "nasa_uva_wm2",
    "ALLSKY_SFC_UVB": "nasa_uvb_wm2",
    "ALLSKY_SFC_UV_INDEX": "nasa_uv_index",
    "ALLSKY_SFC_SW_DWN": "nasa_ghi_wm2",
}
FMI_KEYS = {
    "GLOB_1MIN": "ghi",
    "DIR_1MIN": "dni",
    "DIFF_1MIN": "dhi",
    "REFL_1MIN": "refl",
    "UVB_U": "uvi",
}


def geometry(index):
    """NOAA fractional-year approximation. Hours/minutes are UTC."""
    utc_hour = index.hour + index.minute / 60 + index.second / 3600
    gamma = 2 * np.pi / 365 * (index.dayofyear - 1 + (utc_hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )
    solar_hour = (utc_hour + LON / 15 + eqtime / 60) % 24
    lat = np.deg2rad(LAT)
    mu = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(
        np.deg2rad(15 * (solar_hour - 12))
    )
    return np.asarray(solar_hour), np.asarray(mu)


def hourly_fmi(cache):
    blocks = []
    manifest = []
    for path in sorted(cache.glob("2022-*.xml.gz")):
        root = ET.fromstring(gzip.open(path, "rb").read())
        fields = [
            FMI_KEYS[el.attrib["name"]]
            for el in root.findall(".//{*}DataRecord/{*}field")
        ]
        coords = np.fromstring(
            root.find(".//{*}positions").text, sep=" "
        ).reshape(-1, 3)
        values = np.fromstring(
            root.find(".//{*}doubleOrNilReasonTupleList").text, sep=" "
        ).reshape(-1, len(fields))
        assert np.allclose(coords[:, :2], [LAT, LON])
        time = pd.to_datetime(coords[:, 2], unit="s", utc=True)
        assert len(time.unique()) == len(time)
        day = pd.DataFrame(values, columns=fields, index=time)
        manifest.append(
            {
                "date": path.name[:10],
                "rows": len(day),
                "numeric_sha256": hashlib.sha256(
                    np.column_stack([coords, values]).astype("<f8").tobytes()
                ).hexdigest(),
            }
        )
        minute_mu = np.maximum(geometry(time)[1], 0)
        day["dni_pos"] = day.dni.clip(lower=0)
        day["dirh"] = day.dni_pos * minute_mu
        day["mu_day"] = minute_mu
        hourly = day.resample("h").mean()
        count = day.resample("h").count().add_prefix("n_")
        blocks.append(hourly.join(count))
    pd.DataFrame(manifest).to_csv(OUT / "fmi_manifest.csv", index=False)
    frame = pd.concat(blocks).sort_index()
    assert len(frame) == 2880 and frame.index.is_unique
    return frame


def spectral_coefficients():
    original = pd.read_csv(BASE / "field-uv/digitized.csv")
    action = pd.read_csv(BASE / "solar/spectrum.csv")
    ref = pd.read_csv(BASE / "solar/rates.csv").set_index("species")
    wave = np.arange(280, 400.01, 0.25)
    views = original.loc[original.detector == "DM-SR"].groupby("panel")

    def panel(key):
        d = views.get_group(key).sort_values("wavelength_nm")
        return np.interp(wave, d.wavelength_nm, d.flux)

    flux = np.where(wave <= 310, panel("a"), panel("b")).clip(0)
    flux[wave < 293] = 0.0
    energy = flux * 1e4 * 6.62607015e-34 * 299792458 / (wave * 1e-9)
    ery = np.where(
        wave <= 298,
        1,
        np.where(
            wave <= 328,
            10 ** (0.094 * (298 - wave)),
            10 ** (0.015 * (140 - wave)),
        ),
    )
    rows = []
    for band, mask in [("uvb", wave <= 315), ("uva", wave >= 315)]:
        wave_b = wave[mask]
        joules = np.trapezoid(energy[mask], wave_b)
        row = {
            "band": band,
            "julich_band_spectral_actinic_wm2": joules,
            "uvi_per_wm2": 40
            * np.trapezoid((energy * ery)[mask], wave_b)
            / joules,
        }
        for label, sp in [("fast", "mvos"), ("slow", "lvos")]:
            # Limit the action quadrature to measured/support grid <=370 nm.
            eligible = mask & (wave <= 370)
            sigma = np.interp(
                wave[eligible], action.wavelength_nm, action[f"action_{sp}_cm2"]
            )
            row["j_" + label + "_scale_per_wm2"] = (
                np.trapezoid((flux[eligible] * sigma), wave[eligible])
                / joules
                / ref.loc[sp, "standard_j_per_s"]
            )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("band")
    table.to_csv(OUT / "band_coefficients.csv")
    return table


def main(cache, power):
    f = hourly_fmi(cache)
    source = json.loads(power.read_text())
    assert source["header"]["time_standard"] == "UTC"
    n = pd.DataFrame(source["properties"]["parameter"]).rename(
        columns=POWER_KEYS
    )
    n.index = pd.to_datetime(n.index, format="%Y%m%d%H", utc=True)
    n = n.mask(n == source["header"]["fill_value"])
    assert (
        n[["nasa_uva_wm2", "nasa_uvb_wm2", "nasa_ghi_wm2"]].notna().all().all()
        and len(n) == 8760
    )
    n.to_csv(
        OUT / "power_2022_hourly.csv.gz",
        index_label="datetime_utc",
        compression={"method": "gzip", "mtime": 0},
    )
    raw = f.join(n, how="left")
    raw.to_csv(
        OUT / "hourly_inputs.csv.gz",
        index_label="datetime_utc",
        float_format="%.17g",
        compression={"method": "gzip", "mtime": 0},
    )
    render(raw)


def render(f):
    j = spectral_coefficients()
    col = pd.DataFrame(index=f.index)
    col["month"] = f.index.month
    col["hour_in_week"] = (f.index.day - 12) * 24 + f.index.hour
    col["datetime_utc"] = f.index.strftime("%Y-%m-%dT%H:00:00Z")
    solar, mus = geometry(f.index + pd.Timedelta(minutes=30))
    col["local_solar_hour"] = solar
    # Direct beam term is a minute projection at the sun's changing elevation.
    # At instrument gaps, residual GHI–DHI determines the horizontal direct flux.
    enough_dni = f.n_dni >= 45
    enough_dhi = f.n_dhi >= 45
    sun = mus > np.sin(np.deg2rad(-6))
    g = f.ghi.clip(lower=0).where(sun, 0)
    rawdir = f.dirh.clip(lower=0)
    rawdiff = f.dhi.clip(lower=0)
    dfrac = (rawdir / (rawdir + rawdiff).replace(0, np.nan)).fillna(0)
    dfrac = dfrac.where(enough_dni, 1 - rawdiff / g.replace(0, np.nan))
    dfrac = (
        dfrac.where(enough_dhi, rawdir / g.replace(0, np.nan))
        .fillna(0)
        .clip(0, 1)
    )
    mu = (f.dirh / f.dni_pos.replace(0, np.nan)).clip(0, 1)
    # Tiny noisy/negative DNI values make their mean an unstable denominator.
    mu = mu.where((f.dni > 3) & enough_dni, f.mu_day).clip(0, 1)
    # In dim light dark offsets in the diffuse pyranometer can exceed its
    # signal. Retain measured absolute projected beam instead of amplifying
    # it by normalizing the noisy horizontal components to GHI.
    offset_limited = (g < 10) | (rawdiff + rawdir < 5)
    dfrac = dfrac.where(
        ~offset_limited, (rawdir / g.replace(0, np.nan)).clip(0, 1)
    ).fillna(0)
    dfrac = dfrac.where(mu > 0, 0)
    col["mu0"] = mu
    col["ghi_wm2"] = g
    col["direct_horizontal_wm2"] = g * dfrac
    col["dhi_wm2"] = g * (1 - dfrac)
    col["dni_wm2"] = (col.direct_horizontal_wm2 / mu.replace(0, np.nan)).fillna(
        0
    )
    col["solar_cosine_midpoint"] = mus
    col["fmi_uv_index"] = f.uvi.clip(lower=0)
    col["fmi_refl_wm2"] = f.refl.clip(lower=0).where(sun, 0)
    col["nasa_uva_wm2"] = f.nasa_uva_wm2
    col["nasa_uvb_wm2"] = f.nasa_uvb_wm2
    # Rayleigh differential transmission moves a fraction of broadband beam
    # into the diffuse UV component. CERES already supplies the plane total.
    # The geometry brackets span purely diffuse UV through broadband fraction.
    angle = np.rad2deg(np.arccos(mu))
    airmass = 1 / (mu + 0.50572 * (96.07995 - angle) ** (-1.6364))

    def rayleigh(wavelength_um):
        return (
            0.008569
            * wavelength_um ** (-4)
            * (1 + 0.0113 * wavelength_um**-2 + 0.00013 * wavelength_um**-4)
        )

    for band, wavelength in [("uva", 0.355), ("uvb", 0.310)]:
        beam = dfrac * np.exp(
            -(rayleigh(wavelength) - rayleigh(0.550)) * airmass
        )
        multipliers = {
            "iso": np.full(len(f), 2.0),
            "bb": 2 + dfrac * (1 / mu.replace(0, np.nan) - 2).fillna(0),
            "central": 2 + beam * (1 / mu.replace(0, np.nan) - 2).fillna(0),
        }
        for name, mult in multipliers.items():
            actinic = f["nasa_" + band + "_wm2"] * mult
            for label in ["fast", "slow"]:
                key = f"j_{label}_scale" + (
                    "" if name == "central" else "_" + name
                )
                col[key] = (
                    col.get(key, 0.0)
                    + actinic * j.loc[band, "j_" + label + "_scale_per_wm2"]
                )
    col["two_band_uv_index"] = sum(
        f["nasa_" + band + "_wm2"] * j.loc[band, "uvi_per_wm2"]
        for band in ["uva", "uvb"]
    )
    assert np.isfinite(col.select_dtypes("number")).all().all()
    assert (
        np.max(np.abs(col.direct_horizontal_wm2 + col.dhi_wm2 - col.ghi_wm2))
        < 1e-10
    )
    assert (
        np.max(np.abs(col.dni_wm2 * col.mu0 - col.direct_horizontal_wm2))
        < 1e-10
    )
    for name, keep, expected in [
        ("forcing", col.hour_in_week >= 0, 168),
        ("history", col.hour_in_week < 0, 72),
    ]:
        sub = col[keep]
        assert (sub.groupby("month").size() == expected).all()
        sub.to_csv(OUT / (name + ".csv"), index=False, float_format="%.9g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmi-cache", type=Path)
    parser.add_argument("--power-json", type=Path)
    args = parser.parse_args()
    if args.fmi_cache and args.power_json:
        main(args.fmi_cache, args.power_json)
    else:
        data = pd.read_csv(
            OUT / "hourly_inputs.csv.gz",
            index_col="datetime_utc",
            parse_dates=True,
            float_precision="round_trip",
        )
        render(data)
