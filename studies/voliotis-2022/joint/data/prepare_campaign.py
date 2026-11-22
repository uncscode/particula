"""Extract time-resolved organics and speciated VOCs from the campaign EDFs."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

from prepare_run2 import get_utility

HERE = Path(__file__).resolve().parent
RECORDS = {
    7: "RNGC-1628",
    8: "Y7RC-MK49",
    10: "G0X4-TB38",
    11: "4XSN-AB47",
    12: "3WY2-D921",
    13: "T6H3-JQ42",
    14: "VTCR-KD24",
    15: "GV1J-CE17",
    16: "JBMD-1D11",
    18: "7NBS-C877",
    19: "E3WH-DX73",
    20: "3GCH-NT90",
    21: "SY6A-H369",
    22: "5NXY-WN10",
    23: "28KM-B382",
    24: "P859-2T83",
}
GENERIC = {
    8: "o_cresol_fraction",
    12: "isoprene_fraction",
    13: "isoprene_fraction",
}
COLUMN_MAP = {
    "SOA_mass_conc": "soa_ug_m3",
    "pinene": "alpha_pinene_fraction",
    "isoprene": "isoprene_fraction",
    "cresol": "o_cresol_fraction",
    "half_cresol": "o_cresol_fraction",
}


def extract(content, run):
    """Preserve empty tab fields and skip measurements without timestamps."""
    header, values = content.decode("latin1").split("&&&&&&&&&&")
    fields = dict(
        line.split("=", 1) for line in header.splitlines() if "=" in line
    )
    size = int(fields["NUMBER OF COLUMNS"])
    names = [fields[f"COLUMN {i + 1}"].strip() for i in range(size)]
    records, audit = [], []
    for index, name in enumerate(names[1:], start=1):
        variable = GENERIC[run] if name.lower() == "voc" else COLUMN_MAP[name]
        missing_time = missing_value = 0
        obs = []
        for line in values.splitlines():
            if not line.strip():
                continue
            cells = line.split("\t")
            if len(cells) != size:
                raise ValueError(f"Expected {size} tab-separated cells: {line}")
            if not cells[index].strip():
                missing_value += 1
                continue
            if not cells[0].strip():
                missing_time += 1
                continue
            t, value = float(cells[0]), float(cells[index])
            if value == float(fields[f"MISSING_VALUE({index})"]):
                missing_value += 1
                continue
            if not (math.isfinite(t) and math.isfinite(value)):
                raise ValueError("Nonfinite EDF value")
            obs.append((run, variable, t, value))
        times = [o[2] for o in obs]
        if times != sorted(set(times)):
            raise ValueError("Repeated or unsorted sample time")
        records.extend(obs)
        audit.append(
            (
                name,
                variable,
                fields[f"UNITS({index})"].strip(),
                fields["UNITS(0)"].strip(),
                fields["DESCRIPTION"].strip(),
                len(obs),
                missing_value,
                missing_time,
                times[0],
                times[-1],
            )
        )
    return records, audit


def save(filename, header, rows):
    """Write a deterministic comma-delimited file."""
    with (HERE / filename).open("w", newline="") as out:
        w = csv.writer(out, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def prepare(raw_dir):
    """Download instrument files and verify previously recorded hashes."""
    edf = get_utility()
    observations, manifest, channels = [], [], []
    prior_path = HERE / "campaign_source_manifest.csv"
    prior = {}
    if prior_path.exists():
        with prior_path.open() as f:
            prior = {r["source_url"]: r["sha256"] for r in csv.DictReader(f)}
    for run, doi in RECORDS.items():
        dest = raw_dir / str(run) / doi
        dest.mkdir(parents=True, exist_ok=True)
        for item in edf._record(dest, doi)["data"]:
            if not item["name"].endswith(("_AMS.edf", "_VOC.edf")):
                continue
            content = edf._source_file(item, dest)
            digest = hashlib.sha256(content).hexdigest()
            if prior and prior.get(item["url"]) != digest:
                raise ValueError(f"Source file changed: {item['name']}")
            manifest.append(
                (run, "10.25326/" + doi, item["name"], item["url"], digest)
            )
            data, audit = extract(content, run)
            observations.extend(data)
            channels.extend((run, item["name"], *a) for a in audit)
    if prior and set(prior) != {row[3] for row in manifest}:
        raise ValueError("Archive inventory changed")
    save(
        "campaign_observations.csv",
        ("run", "variable", "time_s", "value"),
        sorted(observations),
    )
    save(
        "campaign_source_manifest.csv",
        ("run", "doi", "file", "source_url", "sha256"),
        manifest,
    )
    save(
        "campaign_channels.csv",
        (
            "run",
            "file",
            "edf_column",
            "variable",
            "edf_units",
            "edf_time_units",
            "edf_description",
            "retained_n",
            "missing_value_n",
            "missing_time_n",
            "first_s",
            "last_s",
        ),
        channels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    prepare(parser.parse_args().raw_dir)
