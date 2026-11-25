"""Recover native daily scan fractions and time-match dry spectra to bulk-counter minutes."""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

P = Path(os.environ.get("PALLAS_RUN", "pallas-run")).resolve()
SET = np.array([0.1, 0.2, 0.3, 0.6, 1.0])
FILES = [
    "CCN_tot_size_resolved_PaCE2022.zip",
    "CCN_tot_total_conc_PaCE2022.zip",
    "dmps_pm25_PaCE2022.zip",
]
HASHES = [
    "a7d17c687d50526f5889316526676ed9",
    "cd38e48e79e82903b876fd279407af9c",
    "8024e751c7ad42a828815d9fbe020e3e",
]


def screened_scans():
    assert (
        hashlib.md5((P / FILES[0]).read_bytes()).hexdigest()
        == "a7d17c687d50526f5889316526676ed9"
    )
    with ZipFile(P / FILES[0]) as z:

        def read(kind):
            return pd.concat(
                [pd.read_csv(z.open(f)) for f in z.namelist() if kind in f],
                ignore_index=True,
            ).replace(-9999.9, np.nan)

        fits, scans = read("_k."), read("_s.")
    fits = fits.rename(columns={"Dc [nm]": "D", "std_error": "SE"})
    fits["date"] = pd.to_datetime(fits.date, format="%d.%m.%Y")
    scans["date"] = pd.to_datetime(scans.Date, format="%d.%m.%Y")
    scans = scans.melt(
        id_vars=["date", "Dp", "CN"],
        value_vars=[f"Act({s:.1f})" for s in SET],
        value_name="fraction",
    )
    scans["SS"] = scans.variable.str.extract(r"\(([.\d]+)\)").astype(float)
    scans = scans[scans.fraction.between(0, 1)].copy()
    rows = []
    for r in fits.itertuples():
        a = scans[(scans.date == r.date) & (scans.SS == r.SS)].sort_values("Dp")
        low = a.loc[(a.Dp < r.D) & (a.fraction < 0.5), "Dp"].max()
        high = a.loc[(a.Dp > r.D) & (a.fraction > 0.5), "Dp"].min()
        rows.append((low, high))
    fits[["low", "high"]] = rows
    fits["valid"] = (fits.D > 0) & (fits.SE > 0)
    fits["bracket"] = fits.valid & fits.low.notna() & fits.high.notna()
    return fits, scans


def cohort():
    fits, scans = screened_scans()
    dates = fits.groupby("date").bracket.agg(["size", "sum"])
    dates = dates[
        (dates["size"] == len(SET)) & (dates["sum"] == len(SET))
    ].index
    assert len(dates) == 29
    scans = scans[scans.date.isin(dates)].copy()
    scans["date"] = scans.date.dt.strftime("%Y-%m-%d")
    scans["j"] = np.searchsorted(SET, scans.SS)
    dates = dates.strftime("%Y-%m-%d").tolist()
    return scans, dates[:19], dates[19:]


def recipes():
    chem = (
        pd.read_csv(Path(__file__).with_name("matorova-chemistry.csv"))
        .dropna(subset=["OC_ugC_m3"])
        .set_index("date")
    )
    moles = np.minimum(chem.sodium_ug_m3 / 22.990, chem.chloride_ug_m3 / 35.45)
    chem["NaCl_volume"] = moles.fillna(0) * 58.44 / 2.165
    chem["AS_volume"] = (
        np.minimum(
            chem.sulfate_ug_m3 / 96.06, chem.ammonium_ug_m3 / (2 * 18.039)
        )
        * 132.14
        / 1.77
    )
    chem["AS_all_sulfate_volume"] = chem.sulfate_ug_m3 / 96.06 * 132.14 / 1.77
    return chem


def volume_spectra(dates, cut=300):
    output = {}
    with ZipFile(P / "dmps_pm25_PaCE2022.zip") as archive:
        for day in dates:
            name = "FMI.DMPS.c1." + day.replace("-", "") + ".csv"
            if name not in archive.namelist():
                continue
            s = pd.read_csv(archive.open(name)).iloc[:, 1:31]
            d = s.columns.to_numpy(float) * 1e9
            valid = np.isfinite(s).all(axis=1) & (s >= 0).all(axis=1)
            weights = s[valid].mean().to_numpy() * np.gradient(np.log(d)) * d**3
            keep = d <= cut
            output[day] = (
                np.log(d[keep] / 80),
                weights[keep] / weights[keep].sum(),
            )
    return output


def load_bulk(test):
    with ZipFile(P / FILES[1]) as archive:
        total = pd.concat(
            [pd.read_csv(archive.open(f)) for f in archive.namelist()],
            ignore_index=True,
        ).replace(-9999.9, np.nan)
    total["time"] = pd.to_datetime(
        total["datetime UTC"], format="%d-%m-%y %H:%M"
    )
    total["mid"] = total.time - pd.Timedelta(seconds=30)
    total["date"] = total.time.dt.strftime("%Y-%m-%d")
    total["measured"] = np.nan
    for ss in SET[[0, 1, 3, 4]]:
        at = total.SS == ss
        total.loc[at, "measured"] = total.loc[at, f"SS{ss:.1f}"]
    total = total[
        (total.measured >= 0) & (total.CN > 0) & total.date.isin(test)
    ]
    with ZipFile(P / FILES[2]) as archive:
        frames = [
            pd.read_csv(
                archive.open("FMI.DMPS.c1." + d.replace("-", "") + ".csv")
            )
            for d in test
        ]
    spec = pd.concat(frames, ignore_index=True)
    ch = spec.columns[1:31]
    c = np.log(ch.to_numpy(float) * 1e9)
    edges = np.r_[
        c[0] - (c[1] - c[0]) / 2,
        (c[:-1] + c[1:]) / 2,
        c[-1] + (c[-1] - c[-2]) / 2,
    ]
    spec["end"] = pd.to_datetime(spec.datetime)
    merged = pd.merge_asof(
        total.sort_values("mid"),
        spec.sort_values("end"),
        left_on="mid",
        right_on="end",
        direction="forward",
        tolerance=pd.Timedelta(seconds=390),
    )
    valid = (
        merged[ch].notna().all(axis=1)
        & (merged[ch] >= 0).all(axis=1)
        & (merged[ch].sum(axis=1) > 0)
        & (merged.total > 0)
    )
    merged = merged[valid]
    assert len(merged) == 4110 and merged.time.is_unique
    groups = {}
    for (day, ss), g in merged.groupby(["date", "SS"]):
        n = g[ch].to_numpy() * np.diff(edges)
        groups[day, np.searchsorted(SET, ss)] = (
            n / n.sum(axis=1, keepdims=True) * g.CN.to_numpy()[:, None],
            g.measured.to_numpy(),
        )
    return edges, groups


def retrieve(fetch):
    P.mkdir(parents=True, exist_ok=True)
    for name, digest in zip(FILES, HASHES, strict=True):
        path = P / name
        if fetch and not path.exists():
            url = (
                "https://zenodo.org/records/23309964/files/"
                + name
                + "?download=1"
            )
            tmp = path.with_suffix(".partial")
            with urlopen(url, timeout=90) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            if hashlib.md5(tmp.read_bytes()).hexdigest() != digest:
                raise ValueError("Checksum mismatch: " + name)
            tmp.replace(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Checksum mismatch: " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    retrieve(parser.parse_args().download)
    scans, train, test = cohort()
    print(
        f"{len(train)} early dates, {len(test)} late dates, {sum(len(obs) for _, obs in load_bulk(test)[1].values())} matched minutes"
    )
