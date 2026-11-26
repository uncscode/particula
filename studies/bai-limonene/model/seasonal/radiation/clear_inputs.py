"""Retain the independent CERES hourly clear-sky illumination boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?"
    "parameters=CLRSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DIFF,CLRSKY_SFC_SW_DNI"
    "&community=RE&longitude=26.62901&latitude=67.36663&start=20220101&end=20221231"
    "&format=JSON&time-standard=UTC"
)


def run(path):
    """Generate a closure-preserving, independent clear-sky boundary."""
    raw = json.loads(path.read_text())
    d = pd.DataFrame(raw["properties"]["parameter"])
    if raw["header"]["time_standard"] != "UTC" or len(d) != 8760:
        raise ValueError(raw["header"])
    if set(p["units"] for p in raw["parameters"].values()) != {"Wh/m^2"}:
        raise ValueError(raw["parameters"])
    d.index = pd.to_datetime(d.index, format="%Y%m%d%H", utc=True)
    d = d.rename(
        columns={
            "CLRSKY_SFC_SW_DWN": "clear_ghi",
            "CLRSKY_SFC_SW_DIFF": "clear_dhi",
            "CLRSKY_SFC_SW_DNI": "clear_dni",
        }
    ).rename_axis("timestamp")
    f = pd.read_csv(HERE.parent / "forcing" / "forcing.csv")
    f["timestamp"] = pd.to_datetime(f.datetime_utc, utc=True)
    s = f.merge(d, on="timestamp", validate="one_to_one")
    s["clear_dirh"] = s.clear_ghi - s.clear_dhi
    s["clear_mu"] = (s.clear_dirh / s.clear_dni.replace(0, np.nan)).fillna(0)
    if (
        len(s) != 2016
        or s[["clear_ghi", "clear_dirh", "clear_dhi", "clear_dni"]].min().min()
        < 0
        or not s.clear_mu.between(0, 1).all()
    ):
        raise ValueError("Nonphysical independent boundary")
    if (s.clear_dirh - s.clear_mu * s.clear_dni).abs().max() > 0.011:
        raise ValueError("Direct-beam closure")
    s[
        [
            "month",
            "hour_in_week",
            "datetime_utc",
            "clear_ghi",
            "clear_dhi",
            "clear_dni",
            "clear_dirh",
            "clear_mu",
        ]
    ].to_csv(HERE / "clear-forcing.csv.gz", index=False, float_format="%.10g")
    check = dict(
        source=SOURCE,
        documentation="https://power.larc.nasa.gov/docs/methodology/data/derived/syn1deg/",
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        header=raw["header"],
        units=raw["parameters"],
        sample_hours=len(s),
        local_allsky_wm2=s.ghi_wm2.mean(),
        ceres_clearsky_wm2=s.clear_ghi.mean(),
        clear_energy_fraction_at_zero_measured_ghi=s.loc[
            s.ghi_wm2 == 0, "clear_ghi"
        ].sum()
        / s.clear_ghi.sum(),
    )
    (HERE / "clear-forcing-checks.json").write_text(
        json.dumps(check, indent=2) + "\n"
    )
    print(check)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    run(p.parse_args().input)
