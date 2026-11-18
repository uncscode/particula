"""Extract BAT CCN source data from the publisher ZIP and MATLAB archive."""

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from defusedxml import ElementTree
from scipy.io import loadmat

COLUMNS = (
    "chemical",
    "validation_flag",
    "formula",
    "bat_functionality",
    "molar_mass_g_mol",
    "h_c",
    "o_c",
    "kappa_bat",
    "kappa_aiomfac",
    "kappa_measured",
    "error_minus",
    "error_plus",
    "fractional_error",
    "short_citation",
    "primary_citation",
    "secondary_citation",
)
ARCHIVE_FILES = {
    "mid": "547511f763dd427a5e0c13d9b7ebfe4c174bfca55974723926c68561f288f64a",
    "lower": "46d27a7a7039c08dc48da25b681bded6332f955fdd2aec95a0a3dca5459e5909",
    "zero": "c603f3a39efb2890ab1bc87114c79030ef953982afcd9bda572e89a289310347",
}
SUPPLEMENT_SHA256 = (
    "a9179b4cb803302eb408ced9767c99d1709f48db4a7fe94aa8f635d97b9217be"
)


def write_csv(path, header, rows):
    """Write a UTF-8 table with explicit column labels."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def extract_measurements(supplement, output):
    """Read Figure 12 data rows and citations from Table S6."""
    if hashlib.sha256(supplement.read_bytes()).hexdigest() != SUPPLEMENT_SHA256:
        raise ValueError("Unexpected publisher ZIP checksum")
    with ZipFile(supplement) as archive:
        workbook = archive.read("SI_for_ACP/data_used_in_paper.xlsx")
    with ZipFile(BytesIO(workbook)) as book:
        strings = [
            "".join(item.itertext())
            for item in ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        ]
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet12.xml"))
    rows = []
    for row in sheet.findall(".//{*}row"):
        number = int(row.attrib["r"])
        if 2 <= number <= 31:
            fields = {}
            for cell in row.findall("{*}c"):
                value = cell.findtext("{*}v")
                if value is not None:
                    fields[cell.attrib["r"]] = (
                        strings[int(value)] if cell.get("t") == "s" else value
                    )
            rows.append(
                [number]
                + [
                    fields.get(f"{letter}{number}", "").strip()
                    for letter in "ABCDEFGHIJKLMNOP"
                ]
            )
    if len(rows) != 30:
        raise ValueError("Expected 30 measurement records")
    write_csv(
        output / "figure12-measurements.csv", ("workbook_row", *COLUMNS), rows
    )


def extract_activation(folder, output):
    """Convert the Figure 9 critical values to nm and percent."""
    rows = []
    for case, checksum in ARCHIVE_FILES.items():
        file = folder / f"DATA for figure kappaCCN {case} O2C.mat"
        if hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Unexpected archived data for {case}")
        data = loadmat(file, simplify_cells=True)
        branches = [("primary", data)]
        beta = data["alternate"]["beta"]
        if beta["phase"]:
            branches.append(("alternate_beta", beta))
        for branch, values in branches:
            ratio = float(values["SatCritical"])
            rows.append(
                (
                    case,
                    data["O2C_value"],
                    branch,
                    data["Molar_mass"],
                    data["Dp_sim_nm"],
                    data["Temp"],
                    ratio,
                    (ratio - 1) * 100,
                    values["Dp_critical"] * 1e9,
                    values["kappa_SatCritical"],
                )
            )
    write_csv(
        output / "figure9-activation.csv",
        (
            "case",
            "o_c",
            "branch",
            "organic_molar_mass_g_mol",
            "dry_diameter_nm",
            "archive_temperature_k",
            "critical_saturation_ratio",
            "critical_supersaturation_percent",
            "critical_wet_diameter_nm",
            "critical_kappa",
        ),
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supplement_zip", type=Path)
    parser.add_argument("mat_folder", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    extract_measurements(args.supplement_zip, destination)
    extract_activation(args.mat_folder, destination)
