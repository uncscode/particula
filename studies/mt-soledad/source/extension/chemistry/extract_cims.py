"""Convert the QC'd non-cloud CIMS channel at Mt Soledad to unambiguous UTC."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def main(source, output):
    expected = "615ecc549846186fc5eba6634b30d3e9a1c7801e4c5c05be19648339b1d189eb"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Source ZIP differs from the study version")
    member = "EPCAPE_Soledad_CIMS_processed_Non_Cloud.csv"
    with ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError("Invalid source ZIP")
        raw = pd.read_csv(z.open(member))
    t = pd.to_datetime(raw["Date_Time_UTC"], format="%d/%m/%Y %H:%M", utc=True)
    if not t.is_monotonic_increasing or not t.is_unique:
        raise ValueError("Unexpected duplicate/out-of-order timestamps")
    fields, meta = {}, {}
    for name in raw:
        m = re.fullmatch(r"msd__5min__(\w+)_([A-Za-z0-9]+)_NOCLOUDS_ppb", name)
        if not m:
            continue
        species, formula = m.groups()
        key = species.lower() + "_ppbv"
        vals = pd.to_numeric(raw[name], errors="raise")
        fields[key] = vals
        meta[key] = dict(
            source=name,
            formula=formula,
            finite=int(vals.count()),
            negative=int(vals.lt(0).sum()),
            min=float(vals.min()),
            median=float(vals.median()),
            max=float(vals.max()),
        )
    if len(fields) != 27:
        raise ValueError("Unexpected number of species")
    times = pd.DataFrame(
        {
            "time_utc": t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_unix": t.astype("int64")
            // (10**9 if str(t.dtype).startswith("datetime64[ns") else 10**6),
        }
    )
    assert (
        pd.to_datetime(times.time_unix, unit="s", utc=True).equals(t)
        or (pd.to_datetime(times.time_unix, unit="s", utc=True) == t).all()
    )
    frame = pd.concat([times, pd.DataFrame(fields)], axis=1)
    frame.to_csv(
        output / "cims.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    metadata = dict(
        doi="10.6075/J0RV0P3H",
        member=member,
        zip_sha256=expected,
        start=times.time_utc.iloc[0],
        end=times.time_utc.iloc[-1],
        rows=len(frame),
        rows_any=int(frame[list(fields)].notna().any(axis=1).sum()),
        species=meta,
    )
    (output / "cims-source.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    main(args.source, args.output)
