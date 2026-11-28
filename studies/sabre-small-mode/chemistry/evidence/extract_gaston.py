"""Recover absolute and normalized N2O5 uptake values from Gaston Table 1."""

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKSUM = "d7303b4575b59271e117bb62e06d947cfd2809e3ccf51207e9c426b28c0150ab"
FIELDS = [
    "rh_pct",
    "solution_type",
    "solution_wt_pct",
    "organic",
    "organic_mass_fraction",
    "oc_atomic",
    "gamma",
    "gamma_ci95_half",
    "gamma_normalized",
    "normalized_ci95_half",
]


def extract(pdf):
    """Extract original page 5 by matching its two numerical ± columns."""
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("Unexpected PDF checksum.")
    command = [shutil.which("pdftotext"), "-layout", "-f", "5", "-l", "5"]
    page = subprocess.check_output([*command, str(pdf), "-"], text=True)  # noqa: S603
    rows = []
    for line in page.splitlines():
        if not re.match(r"^\s*(30|50|70)\s+[123]\s+", line):
            continue
        rh, solution, weight, material, organic, oc, gamma, norm = re.split(
            r"\s{2,}", line.strip()
        )
        # The 70 % azelaic entry contains the publisher typo "azelaic aid".
        material = material.removeprefix("Ammonium bisulfate + ").replace(
            "azelaic aid", "azelaic acid"
        )
        if organic == "N/A" and material == "Ammonium bisulfate":
            organic = "0"
        gamma_value, gamma_width = gamma.split(" ± ")
        norm_value, norm_width = norm.split(" ± ")
        rows.append(
            [
                rh,
                solution,
                weight,
                material,
                organic,
                oc,
                gamma_value,
                gamma_width,
                norm_value,
                norm_width,
            ]
        )
    if len(rows) != 52:
        raise ValueError(f"Unexpected table size: {len(rows)} rows")
    return rows


def main():
    """Write the compact measured-value table."""
    args = argparse.ArgumentParser()
    args.add_argument("pdf", type=Path)
    args.add_argument("--output", type=Path, default=HERE / "gaston-table1.csv")
    opts = args.parse_args()
    with opts.output.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(extract(opts.pdf))


if __name__ == "__main__":
    main()
