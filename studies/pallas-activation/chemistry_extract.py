"""Extract the paired Matorova chemistry from the two original EBAS NetCDF files."""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from observations import cohort

NAMES = [
    {
        "organic_carbon_amean": "OC_ugC_m3",
        "elemental_carbon_amean": "EC_ugC_m3",
    },
    {
        "sulphate_total_ug_per_m3": "sulfate_ug_m3",
        "ammonium_ug_per_m3": "ammonium_ug_m3",
        "sodium": "sodium_ug_m3",
        "nitrate_ug_per_m3": "nitrate_ug_m3",
        "chloride": "chloride_ug_m3",
    },
]
HASH = [
    "ef431dd8d6aa284631639bd685d421e4cf73c1b73a33b04c2ae5e8010dc1ef07",
    "077f2ddbc065a2b2eec173f9d3d8cd94f6703bc73539218f036cc13fd2e70382",
]


def values(f, key):
    s = subprocess.check_output(
        [
            "h5dump",
            "-y",
            "-w",
            "0",
            "-m",
            "%.12g",
            "-A",
            "0",
            "-d",
            key,
            str(f),
        ],
        text=True,
    )
    return np.fromstring(s.split("DATA {")[-1].split("}")[0].strip(), sep=",")


def extract(files):
    _, train, test = cohort()
    days = pd.DataFrame(
        {
            "date": train + test,
            "phase": ["train"] * len(train) + ["test"] * len(test),
        }
    ).set_index("date")
    for f, names, h in zip(files, NAMES, HASH):
        if hashlib.sha256(f.read_bytes()).hexdigest() != h:
            raise ValueError(f"Unexpected revision of {f}")
        dates = pd.to_datetime(
            values(f, "time"), origin="1900-01-01", unit="D"
        ).strftime("%Y-%m-%d")
        for var, out in names.items():
            y = values(f, var)
            qc = values(f, var + "_qc").reshape(-1, len(y))
            ok = (
                np.isfinite(y)
                & (qc == 0).all(axis=0)
                & np.isin(dates, days.index)
            )
            v = pd.Series(y[ok], index=dates[ok])
            assert v.index.is_unique
            days[out] = v
    return days


def composition(carbon, ions):
    # extract checks the pinned EBAS carbon and ion files' hashes.
    data = extract([carbon, ions]).query("phase == 'test'").copy()
    dates = pd.to_datetime(values(ions, "time"), origin="1900-01-01", unit="D")
    dates = dates.strftime("%Y-%m-%d")
    for name in ["calcium", "magnesium", "potassium"]:
        mass = values(ions, name)
        flags = values(ions, name + "_qc").reshape(-1, len(mass))
        valid = (
            np.isin(dates, data.index)
            & np.isfinite(mass)
            & (flags == 0).all(axis=0)
        )
        column = pd.Series(mass[valid], index=dates[valid])
        assert column.index.is_unique
        data[name + "_ug_m3"] = column
    ion_columns = data.filter(regex="_ug_m3$")
    assert ion_columns.shape[1] == 8
    assert (
        data[["sulfate_ug_m3", "ammonium_ug_m3", "OC_ugC_m3"]]
        .notna()
        .all()
        .all()
    )
    data["valid_ion_count"] = ion_columns.count(axis=1)
    for label, mass in [
        ("sulfate_ammonium", data.sulfate_ug_m3 + data.ammonium_ug_m3),
        ("available_ions", ion_columns.sum(axis=1, min_count=1)),
    ]:
        ion_volume = mass / 1.77
        for case, conversion, density in [
            ("central", 2.1, 1.2),
            ("low_organic_volume", 1.6, 1.5),
            ("high_organic_volume", 2.3, 1.2),
        ]:
            organic_volume = conversion * data.OC_ugC_m3 / density
            data[f"{label}_{case}"] = ion_volume / (ion_volume + organic_volume)
    data["sodium_chloride_ion_mass_fraction"] = (
        data.sodium_ug_m3 + data.chloride_ug_m3
    ) / ion_columns.sum(axis=1, min_count=8)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("carbon", type=Path)
    p.add_argument("ions", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--eight-ions",
        action="store_true",
        help="Write the ten-date bulk chemistry sensitivity table.",
    )
    a = p.parse_args()
    if a.eight_ions:
        composition(a.carbon, a.ions).to_csv(a.out)
    else:
        extract([a.carbon, a.ions]).to_csv(a.out)
