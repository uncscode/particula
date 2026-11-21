"""Retrieve native counter records for the paired Manchester campaign runs."""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_run2 import get_utility

HERE = Path(__file__).resolve().parent
DOIS = {
    8: "4RKZ-H076",
    10: "Y275-K605",
    12: "JDMS-ZR65",
    13: "T6H3-JQ42",
    14: "KP9S-F341",
    15: "E0EP-QR41",
    16: "4QB0-PE88",
    18: "4KH1-4152",
    19: "E3WH-DX73",
    21: "4909-6448",
    24: "1D6D-TQ82",
}
CHANNELS = {"WCPC": "total_NumberConc", "DMPS": "Total_number"}


def counter_data(raw, instrument, clock):
    """Read counter measurements on the corresponding AMS lights-on clock."""
    header, rows = raw.decode("latin1").split("&&&&&&&&&&")
    fields = dict(
        line.split("=", 1) for line in header.splitlines() if "=" in line
    )
    names = [
        fields[f"COLUMN {i + 1}"].strip()
        for i in range(int(fields["NUMBER OF COLUMNS"]))
    ]
    if CHANNELS[instrument] not in names:
        return None
    source = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) != len(names):
            raise ValueError("Invalid counter EDF row")
        t, n = (
            cells[names.index("TIME")],
            cells[names.index(CHANNELS[instrument])],
        )
        if not t.strip() or not n.strip():
            continue
        t, n = float(t), float(n)
        if t > 1e8 and "1904" in fields["UNITS(0)"]:
            lights = datetime.strptime(
                clock, "seconds since %Y-%m-%d %H:%M:%S GMT"
            )
            t -= (lights - datetime(1904, 1, 1)).total_seconds()
        if n > 0 and -320 <= t <= 18320 and np.isfinite([t, n]).all():
            source.append([t, n])
    return np.array(source)


def samples(pairs, r, instrument):
    """Keep one physical reading per observed scan near analysis knots."""
    used = set()
    for sec in np.arange(0, 18001, 360) if len(pairs) else ():
        j = np.abs(pairs[:, 0] - sec).argmin()
        if j not in used and abs(pairs[j, 0] - sec) <= 310:
            used.add(j)
            yield (
                dict(
                    run=r,
                    instrument=instrument,
                    time_s=pairs[j, 0],
                    count_cm3=pairs[j, 1],
                )
            )


def build(cache):
    """Select the nearest valid count to each six-minute analysis knot."""
    helper = get_utility()
    clocks = pd.read_csv(HERE / "campaign_channels.csv")
    clocks = (
        clocks[clocks.variable == "soa_ug_m3"].set_index("run").edf_time_units
    )
    manifest = []
    observations = []
    for r, doi in DOIS.items():
        path = cache / str(r) / doi
        path.mkdir(parents=True, exist_ok=True)
        record = helper._record(path, doi)
        for item in record["data"]:
            instrument = item["name"].split("_")[-1].split(".")[0]
            if instrument not in CHANNELS:
                continue
            raw = helper._source_file(item, path)
            pairs = counter_data(raw, instrument, clocks.loc[r])
            if pairs is None:
                continue
            manifest.append(
                dict(
                    run=r,
                    instrument=instrument,
                    doi="10.25326/" + doi,
                    file=item["name"],
                    url=item["url"],
                    sha256=hashlib.sha256(raw).hexdigest(),
                    column=CHANNELS[instrument],
                    matched_rows=len(pairs),
                )
            )
            observations.extend(samples(pairs, r, instrument))
    for name, rows in (
        ("wall_counter_sources", manifest),
        ("wall_counts", observations),
    ):
        out = HERE / (name + ".csv")
        data = pd.DataFrame(rows)
        if out.exists() and name == "wall_counter_sources":
            old = pd.read_csv(out)
            if set(old.sha256) != set(data.sha256):
                raise ValueError("Source contents changed")
        data.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    build(parser.parse_args().cache)
