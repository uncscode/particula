# %% Pallas September 2022 cloud-probe comparison
"""Rebuild native September spectra and daily comparisons from Zenodo v3."""

from pathlib import Path
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import particula as par

ROOT = Path(os.environ.get("PALLAS_RUN_DIR", ".")).resolve()
FILES = {
    "F": ("FMI.FSSP.b1.20220913.csv", "562cc1731e83646e660e147a80443b45"),
    "I": ("OULU.ICEMET.b1.20220914.csv", "ecc008f5c469023f5a7d11c283832121"),
    "C": ("FMI.CAS.b1.20220913.csv", "fc3143139a6e9facb95487f6aeca86f4"),
}
COUNT = "number concentration (cm-3)"
LWC = "liquid water content (g m-3)"
ED = "effective diameter (µm)"
WIND = "Wind dir (deg)"
FSSP = np.r_[np.arange(3, 46, 1.5), 47.0]
CAS = np.array(
    [
        0.61,
        0.68,
        0.75,
        0.82,
        0.89,
        0.96,
        1.03,
        1.1,
        1.17,
        1.25,
        1.5,
        2,
        2.5,
        3,
        3.5,
        4,
        5,
        6.5,
        7.2,
        7.9,
        10.2,
        12.5,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ]
)


def read_source(probe):
    name, checksum = FILES[probe]
    path = ROOT / "input" / name
    assert hashlib.md5(path.read_bytes()).hexdigest() == checksum
    df = pd.read_csv(path, encoding="latin1").replace(-9999.9, np.nan)
    timecols = df.iloc[:, :5].copy()
    timecols.columns = ["year", "month", "day", "hour", "minute"]
    df.index = pd.to_datetime(timecols)
    return df.loc[~df.index.duplicated(keep=False)]


def inputs():
    f, i = read_source("F"), read_source("I")
    common = f.index.intersection(i.index).sort_values()
    assert len(common) == 7089
    f, i = f.loc[common], i.loc[common]
    ok = (
        (f[COUNT] > 0)
        & (i[COUNT] > 0)
        & (f["Temp 570m (C)"] > 0)
        & (f["FD12P visibility (m)"].between(1, 999))
        & (i.flag == 0)
    )
    assert ok.sum() == 4283
    return f.loc[ok], i.loc[ok]


def edges(probe, convention):
    if probe == "I":
        return np.arange(5.0, 201.0)
    coords = FSSP if probe == "F" else CAS
    if convention == "upper":
        return np.r_[2.0 if probe == "F" else 0.5, coords]
    return np.r_[
        coords[0] - (coords[1] - coords[0]) / 2,
        (coords[1:] + coords[:-1]) / 2,
        coords[-1] + (coords[-1] - coords[-2]) / 2,
    ]


def channels(df):
    matrix = df.filter(regex="^dN/dlogDp").to_numpy()
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    return matrix


def weights(probe, geometry):
    return (
        np.diff(np.log10(geometry))
        if probe == "I"
        else np.full(len(geometry) - 1, 0.05)
    )


def window(geometry, low, high, linear=False):
    lower = np.clip(low, geometry[:-1], geometry[1:])
    upper = np.clip(high, geometry[:-1], geometry[1:])
    if linear:
        return (upper - lower) / np.diff(geometry)
    return np.log(upper / lower) / np.diff(np.log(geometry))


def native(frame, probe, convention):
    e = edges(probe, convention)
    n = channels(frame) * weights(probe, e)
    representative = (e[1:] + e[:-1]) / 2
    # Build each instrument as a measured population of water drops per minute.
    # The public initializer accepts one distribution. Reuse the validated bin
    # masses and SI concentration conversion for subsequent independent spectra.
    particle = par.particles.create_particle_data_from_radii(
        radii=representative / 2,
        radius_units="um",
        concentration=n[0],
        concentration_units="1/cm^3",
        density=1000.0,
        density_units="kg/m^3",
        distribution_type="discrete",
    )
    np.testing.assert_allclose(
        particle.slot_concentration_density[0], n[0] * 1e6
    )
    mass = particle.total_mass[0]
    np.testing.assert_allclose(
        mass, np.pi / 6 * 1e3 * (representative * 1e-6) ** 3
    )
    return e, n, mass


def pilot():
    f, i = inputs()
    c = read_source("C")
    target = f.index[f.index.day == 14]
    rows = []
    for probe, df in [
        ("F", f.loc[target]),
        ("I", i.loc[target]),
        ("C", c.loc[c.index.intersection(target)].query("flag==0")),
    ]:
        if len(df) == 0:
            continue
        for convention in ["center"] if probe == "I" else ["upper", "center"]:
            e, n, mass = native(df, probe, convention)
            for rule, w in [
                ("constant_0.05", np.full(len(e) - 1, 0.05)),
                ("geometric_log10", np.diff(np.log10(e))),
            ]:
                pop = channels(df) * w
                total = pop.sum(axis=1)
                d = (e[:-1] + e[1:]) / 2
                volume = pop @ mass * 1e9
                ed = (pop @ d**3) / (pop @ d**2)
                rows.append(
                    dict(
                        probe=probe,
                        geometry=convention,
                        rule=rule,
                        minutes=len(df),
                        median_archived=df[COUNT].median(),
                        median_rebuilt=np.median(total),
                        median_ratio=np.median(total / df[COUNT]),
                        max_count_abs=np.max(np.abs(total - df[COUNT])),
                        median_lwc_ratio=np.median(volume / df[LWC]),
                        median_ed_ratio=np.median(ed / df[ED]),
                    )
                )
    pd.DataFrame(rows).to_csv(ROOT / "pilot.csv", index=False)


def pack(table, day, convention, f, i, extra):
    row = dict(
        day=day,
        geometry=convention,
        minutes=len(f),
        f_native=np.mean(f),
        i_native=np.mean(i),
        native=np.mean(f - i),
        native_median=np.median(f - i),
        native_mae=np.mean(abs(f - i)),
    )
    a, b = extra["f_common"], extra["i_common"]
    for k, v in extra.items():
        row[k] = np.mean(v)
    row.update(
        gap=np.mean(a - b),
        gap_median=np.median(a - b),
        gap_mae=np.mean(abs(a - b)),
        improvement=np.mean(abs(f - i) - abs(a - b)),
    )
    table.append(row)


def analyze():
    f, i = inputs()
    ei, ni, _ = native(i, "I", "center")
    native_i = i[COUNT].to_numpy()
    native_f = f[COUNT].to_numpy()
    rows = []
    for convention in ["upper", "center"]:
        ef, nf, _ = native(f, "F", convention)
        extra = {}
        for label, e, n in [("f", ef, nf), ("i", ei, ni)]:
            for name, lo, hi in [
                ("small", 0, 6),
                ("common", 6, 45),
                ("large", 45, np.inf),
                ("core", 13.07, 45),
                ("lower_common", 6, 13.07),
            ]:
                extra[label + "_" + name] = n @ window(e, lo, hi)
            np.testing.assert_allclose(
                extra[label + "_small"]
                + extra[label + "_common"]
                + extra[label + "_large"],
                n.sum(axis=1),
            )
            extra[label + "_closure"] = n.sum(axis=1) - (
                native_f if label == "f" else native_i
            )
        detail = pd.DataFrame(extra, index=f.index)
        detail["f_native"] = native_f
        detail["i_native"] = native_i
        detail.to_csv(
            ROOT / f"minutes-{convention}.csv.gz",
            index_label="utc",
            compression={"method": "gzip", "mtime": 0},
            float_format="%.9g",
        )
        for day in np.unique(f.index.day):
            mask = f.index.day == day
            pack(
                rows,
                day,
                convention,
                native_f[mask],
                native_i[mask],
                {k: v[mask] for k, v in extra.items()},
            )
    pd.DataFrame(rows).to_csv(ROOT / "dates.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "analyze"])
    args = parser.parse_args()
    {"pilot": pilot, "analyze": analyze}[args.command]()
