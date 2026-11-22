"""Fetch public EUROCHAMP EDFs and derive the chamber comparison data.

Run with Python 3.12, using --raw-dir to reuse a previously downloaded archive.
The source manifest pins file contents to the records used in this study.
"""

import argparse
import csv
import hashlib
import json
from itertools import groupby
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
# Voliotis et al. (2022), supplement Table S1. The first DOI covers context.
RECORDS = {
    1: ("FFTE-ED96", "0QTA-SM15"),
    4: ("9WDC-BG18", "5AJ0-6150"),
    5: ("AH45-9F95", "VM0N-Q983"),
    6: ("59RG-Y252", "3ATZ-0780"),
}


def edf_rows(content):
    """Parse the numeric EDF block using its explicit column names."""
    header, raw = content.decode("latin1").split("&&&&&&&&&&")
    names = [
        line.split("=", 1)[1].strip()
        for line in header.splitlines()
        if line.startswith("COLUMN ")
    ]
    rows = [
        dict(zip(names, map(float, line.split()), strict=True))
        for line in raw.splitlines()
        if line.strip()
    ]
    return rows


def write_csv(name, header, rows):
    """Write a compact, platform-independent table."""
    with (HERE / name).open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def load_public_json(url):
    """Load JSON from the specified public HTTPS archive URL."""
    if not url.startswith("https://"):
        raise ValueError("Expected an HTTPS archive URL")
    with urlopen(url, timeout=40) as response:  # noqa: S310
        return json.load(response)


def _record(directory, suffix):
    record_file = directory / "metadata.json"
    if not record_file.exists():
        registration = load_public_json(
            "https://api.datacite.org/dois/10.25326/" + suffix
        )
        uuid = registration["data"]["attributes"]["url"].split("/")[-1]
        record = load_public_json(
            "https://eurochamp.ipsl.upmc.fr/datacenter-rest/metadata/" + uuid
        )
        record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    record = json.loads(record_file.read_text())
    if record["doi"].lower() != ("10.25326/" + suffix).lower():
        raise ValueError(f"Source DOI mismatch for {suffix}")
    return record


def _source_file(item, directory):
    name = item["name"]
    if Path(name).name != name:
        raise ValueError(f"Invalid archive filename: {name}")
    if not item["url"].startswith(
        "https://eurochamp.ipsl.upmc.fr/datacenter-rest/data/"
    ):
        raise ValueError("Unexpected source archive")
    path = directory / name
    if not path.exists():
        with urlopen(  # noqa: S310
            quote(item["url"], safe=":/"), timeout=40
        ) as response:
            path.write_bytes(response.read())
    content = path.read_bytes()
    if len(content) != item["size"]:
        raise ValueError(f"Incomplete source file {name}")
    return content


def _context_rows(run, instrument, rows):
    if run == 4 and instrument == "GAS":
        return  # Clock conflicts with the elapsed-time header used elsewhere.
    for row in rows:
        t = row["TIME"]
        if t < -90 or t > 21690:
            continue
        for column, value in row.items():
            if (
                column
                in (
                    "NO",
                    "NO2",
                    "O3",
                    "NOx",
                    "T_a",
                    "RH_a",
                    "Total_number",
                    "total_NumberConc",
                    "mode_onl",
                )
                and value != -9999.9
            ):
                if column in ("T_a", "RH_a") and value == 0:
                    continue  # sensor dropout
                yield run, instrument, column, t, value


def build(raw_dir):
    """Retrieve the source files and build observations with checked hashes."""
    observations, context, sources = [], [], []
    for run, dois in RECORDS.items():
        for suffix in dois:
            directory = raw_dir / str(run) / suffix
            directory.mkdir(parents=True, exist_ok=True)
            for item in _record(directory, suffix)["data"]:
                name = item["name"]
                instrument = name.split("_")[-1].split(".")[0]
                if instrument not in (
                    "AMS",
                    "VOC",
                    "GAS",
                    "RHT",
                    "WCPC",
                    "DMPS",
                ):
                    continue
                content = _source_file(item, directory)
                checksum = hashlib.sha256(content).hexdigest()
                sources.append(
                    (run, "10.25326/" + suffix, name, item["url"], checksum)
                )
                rows = edf_rows(content)
                kinds = {"AMS": "soa_ug_m3", "VOC": "apinene_fraction"}
                if instrument in kinds:
                    for row in rows:
                        t, value = row.values()
                        if value == -9999.9:
                            raise ValueError(f"Missing observation in {name}")
                        observations.append((run, kinds[instrument], t, value))
                else:
                    context.extend(_context_rows(run, instrument, rows))
    _save_outputs(observations, context, sources)


def _save_outputs(observations, context, sources):
    # Context is retained at 6 min (nearest sample, ±5 min for DMPS). For
    # numerical use, times are the actual source times. Preserve raw EDFs via
    # manifest for analyses requiring finer sampling or additional metadata.
    compact = []

    for key, group in groupby(sorted(context), key=lambda r: r[:3]):
        series = list(group)
        used = set()
        for target in range(0, 21601, 360):
            record = min(series, key=lambda r: abs(r[3] - target))
            if (
                abs(record[3] - target) <= (300 if key[1] == "DMPS" else 90)
                and record[3] not in used
            ):
                compact.append(record)
                used.add(record[3])
    manifest = HERE / "source_manifest.csv"
    if manifest.exists():
        with manifest.open() as file:
            prior = {
                (row["run"], row["file"]): row["sha256"]
                for row in csv.DictReader(file)
            }
        current = {
            (str(r), name): checksum for r, _, name, _, checksum in sources
        }
        if prior != current:
            raise ValueError("Archive file checksum mismatch")
    write_csv(
        "observations.csv", ("run", "kind", "time_s", "value"), observations
    )
    write_csv(
        "context.csv",
        ("run", "instrument", "channel", "time_s", "value"),
        compact,
    )

    write_csv(
        "source_manifest.csv",
        ("run", "doi", "file", "source_url", "sha256"),
        sources,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Local archive cache or empty download directory",
    )
    build(parser.parse_args().raw_dir)
