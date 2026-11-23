"""Prepare hourly inputs from Xu et al., Zenodo 22008294 (CC BY 4.0)."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(raw, output):
    """Verify the archive and retain the common campaign hours."""
    checks = {
        "htdma": "702bc563f2b3fde1990d569303dd6f9a",
        "smps": "63c4226af08879dd5ae473329989ab90",
        "ccn": "c720f0a9509508bb27d26065a40bf7cb",
    }
    for name, expected in checks.items():
        with (raw / name).open("rb") as fh:
            digest = hashlib.file_digest(fh, "md5").hexdigest()
        if digest != expected:
            raise ValueError(f"Unexpected checksum for {name}")
    growth = pd.read_csv(raw / "htdma", parse_dates=["date"])
    spectra = (
        pd.read_csv(raw / "smps", parse_dates=["date"])
        .set_index("date")
        .dropna()
    )
    counter = pd.read_csv(raw / "ccn", parse_dates=["date"])
    counter["sc"] = counter.cloud_sc.str.extract(r"([\d.]+)")[0].astype(float)
    counter = counter.pivot(
        index="date", columns="sc", values="measured_CCN_number"
    ).dropna()
    growth = growth[
        growth.date.between("2023-01-24", "2024-02-13", inclusive="left")
    ].copy()
    groups = growth.groupby(["date", "size"], sort=True)
    if not ((groups.size() == 351).all()):
        raise ValueError("Unexpected archive checksum, shape or coordinate")
    valid = groups.value.sum().unstack().dropna()
    hours = (
        valid.index[(valid > 0).all(axis=1)]
        .intersection(spectra.index)
        .intersection(counter.index)
        .sort_values()
    )
    part = growth[growth.date.isin(hours)].copy()
    part["row"] = part.groupby(["date", "size"]).cumcount()
    matrix = part.pivot(index=["date", "size"], columns="row", values="value")
    archived = part.pivot(
        index=["date", "size"], columns="row", values="GF"
    ).values
    if not ((archived == archived[0]).all() and len(hours) == 373):
        raise ValueError("Unexpected archive checksum, shape or coordinate")
    raw_pdf = matrix.values.reshape(len(hours), 5, -1)
    gf = archived[0]
    if not (np.all(np.diff(gf) >= 0) and np.min(raw_pdf) >= 0):
        raise ValueError("Unexpected archive checksum, shape or coordinate")
    if not (
        list(matrix.index.levels[1]) == ["D035", "D050", "D075", "D110", "D165"]
    ):
        raise ValueError("Unexpected archive checksum, shape or coordinate")
    if not (np.array_equal(matrix.index.get_level_values(0)[::5], hours)):
        raise ValueError("Unexpected archive checksum, shape or coordinate")
    output.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        output / "inputs.npz",
        dates=hours.to_numpy(dtype="datetime64[s]"),
        sizes=np.array([35.0, 50.0, 75.0, 110.0, 165.0]),
        raw_pdf=raw_pdf,
        gf=gf,
        diam=spectra.columns.astype(float),
        density=spectra.loc[hours].values,
        sc=counter.columns.values,
        measured=counter.loc[hours].values,
    )
    print(
        len(hours),
        len(pd.DatetimeIndex(hours).normalize().unique()),
        "hours, local days",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.output)
