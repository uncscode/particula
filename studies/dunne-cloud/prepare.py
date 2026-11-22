"""Verify the archived inputs and generate the primary neutral screening table."""

import csv
import hashlib
import math
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def prepare():
    with (ROOT / "sources.csv").open(newline="") as manifest:
        for item in csv.DictReader(manifest):
            data = (ROOT / item["file"]).read_bytes()
            if hashlib.md5(data).hexdigest() != item["publisher_md5"]:
                raise ValueError(f"Archive checksum mismatch for {item['file']}")

    source = RAW / "cloud_data_table.csv"
    with ZipFile(RAW / "aaf2649-Dunne-SM-data-S1.csv.zip") as archive:
        if archive.read("aaf2649-Dunne-SM-data-S1.csv") != source.read_bytes():
            raise ValueError("The Zenodo and Caltech copies of Data S1 differ")

    selected = []
    humidity_count = 0
    neutral_count = 0
    with source.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header != [
        "Temperature (K)", "Sulphuric Acid (10^6 cm-3)", "Ammonia (ppt)",
        "Ionization rate", "Relative humidity", "Nucleation rate",
    ]:
        raise ValueError("Unexpected Data S1 columns")
    for line, row in enumerate(rows, start=2):
        values = list(map(float, row))
        if len(values) != 6 or not all(map(math.isfinite, values)):
            raise ValueError(f"Invalid measurement on CSV line {line}")
        temperature, acid, ammonia, ion_production, humidity, rate = values
        in_window = 33 <= humidity <= 43
        humidity_count += in_window
        neutral_count += ion_production == 0
        if in_window and ion_production == 0:
            selected.append([line, temperature, round(acid * 1e6), ammonia,
                             ion_production, humidity, rate])

    with (ROOT / "neutral_33_43rh.csv").open("w", newline="") as stream:
        output = csv.writer(stream)
        output.writerow(["source_csv_line", "temperature_K", "H2SO4_cm-3",
                         "NH3_pptv", "ion_pair_production_cm-3_s-1", "RH_percent",
                         "J_1.7_cm-3_s-1"])
        output.writerows(selected)
    print(f"{len(rows)} rows, {humidity_count} in RH window, {neutral_count} neutral "
          f"overall, {len(selected)} neutral in window")


if __name__ == "__main__":
    prepare()
