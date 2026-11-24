"""Rebuild filter sampling weights from the authors' minute flow and valve records."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from response import FIELD, STUDY
from recipe import SRC

CHANNELS = {1: ("B", 1), 2: ("F", 5), 3: ("D", 3)}
SPECIES = ["organics", "sulfate", "nitrate", "ammonium", "chloride"]


def extract(path):
    filters = pd.read_csv(SRC / "ftir.csv")
    filters["start"] = pd.to_datetime(
        filters.RealStart, format="mixed", utc=True, errors="coerce"
    )
    filters["stop"] = pd.to_datetime(
        filters.RealStop, format="mixed", utc=True, errors="coerce"
    )
    filters = filters.query("MaintenanceFlag==0 and x3OFG==1 and FilterMask in [1,2,3]")
    filters = filters.loc[
        (filters.start < "2023-07-01") & (filters.stop >= "2023-05-01")
    ]
    used = [
        "Time",
        *[f"Alicat_{ch}_Flow" for ch, _ in CHANNELS.values()],
        *[f"Solenoid_{valve}_Status" for _, valve in CHANNELS.values()],
    ]
    f = pd.read_csv(path, usecols=used)
    f.index = pd.to_datetime(f.pop("Time"), format="%d-%b-%Y %H:%M:%S", utc=True)
    f = f.sort_index().loc["2023-04-29":"2023-07-02"]
    if f.index.has_duplicates:
        raise ValueError("Duplicate minute records")
    raw = pd.read_csv(STUDY / "extension/chemistry/ams.csv.gz").dropna(
        subset=["organics_fraction"]
    )
    raw.index = pd.to_datetime(raw.time_unix, utc=True, unit="s")
    rows = []
    subwindows = []
    summary = []
    for r in filters.itertuples():
        ch, v = CHANNELS[r.FilterMask]
        interval = f[(f.index >= r.start) & (f.index < r.stop)]
        flow = interval[f"Alicat_{ch}_Flow"].clip(lower=0)
        valve = interval[f"Solenoid_{v}_Status"]
        if ((valve < 0) | (valve > 1)).any():
            raise ValueError("Invalid valve state")
        rate = flow * valve
        # Each reported timestamp contributes its overlapping five-minute mean sampling rate.
        buckets = rate.resample("5min").mean()
        a = raw[(raw.index >= r.start) & (raw.index < r.stop)].copy()
        a["flow_lpm"] = buckets.reindex(a.index.floor("5min")).to_numpy()
        a = a.dropna(subset=["flow_lpm"])

        def stats(a):
            result = {}
            for sp in SPECIES:
                y = a[sp + "_ug_m3"]
                result[sp + "_open"] = (
                    np.average(y, weights=a.flow_lpm)
                    if a.flow_lpm.sum() > 0
                    else np.nan
                )
                result[sp + "_unweighted"] = y.mean()
            return result

        summary.append(
            dict(
                filter_id=r.FilterID,
                mask=r.FilterMask,
                n_ams=len(a),
                n_open=(a.flow_lpm > 0).sum(),
                valid_flow_minutes=rate.count(),
                valve_open_fraction=valve.mean(),
                sampled_m3=rate.sum() / 1000,
                reported_m3=r.volume,
                **stats(a),
            )
        )
        for b, part in a.groupby(a.time_unix // 21600):
            subwindows.append(
                dict(
                    filter_id=r.FilterID,
                    mask=r.FilterMask,
                    utc=b * 21600,
                    **stats(part),
                )
            )
        rows.extend(
            (r.FilterID, r.FilterMask, int(t), v)
            for t, v in zip(a.time_unix, a.flow_lpm, strict=True)
        )
    out = FIELD / "comparison"
    pd.DataFrame(rows, columns=["filter_id", "mask", "time_unix", "sample_lpm"]).to_csv(
        out / "ams-filter-weights.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.7g",
    )
    pd.DataFrame(summary).to_csv(out / "filter-flow.csv", index=False)
    pd.DataFrame(subwindows).to_csv(out / "window-flow.csv", index=False)
    print(
        pd.DataFrame(summary)
        .query("filter_id in ['EPP337','EPP349','EPP538']")
        .to_string(index=False)
    )
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("minute_csv", type=Path)
    extract(parser.parse_args().minute_csv)
