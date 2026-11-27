"""Extract Chun Xiong et al. (2023) sulfate phase transitions from Zenodo 8079001."""

import csv
import math
from xml.etree import ElementTree
from zipfile import ZipFile

XML = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_sheet(book, sheet, labels):
    root = ElementTree.fromstring(book.read(f"xl/worksheets/sheet{sheet}.xml"))
    result = {}
    for cell in root.findall(".//s:sheetData/s:row/s:c", XML):
        val = cell.findtext("s:v", namespaces=XML)
        if val is not None:
            result[cell.get("r")] = (
                labels[int(val)] if cell.get("t") == "s" else val
            )
    return result


def water_vapor_pressure(t):
    """Liquid water saturation pressure in Pa, T in K, Murphy & Koop (2005) eq. 10."""
    return math.exp(
        54.842763
        - 6763.22 / t
        - 4.210 * math.log(t)
        + 0.000367 * t
        + math.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * math.log(t) + 0.014025 * t)
    )


def inferred_temp_c(srh):
    target = 610.0 / (srh / 100.0)
    lo, hi = 273.15, 298.15
    for _ in range(50):
        mid = (lo + hi) / 2
        if water_vapor_pressure(mid) < target:
            lo = mid
        else:
            hi = mid
    return f"{(lo + hi) / 2 - 273.15:.2f}"


def mean_sd(sheet, row, cols):
    raw = [sheet[f"{col}{row}"] for col in cols]
    if raw == ["Not observed", "Not observed"]:
        return "", ""
    return [f"{float(value):.1f}" for value in raw]


def phase_data(workbook, output):
    with ZipFile(workbook) as book:
        strings = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        labels = ["".join(node.itertext()) for node in strings]
        sheets = {i: read_sheet(book, i, labels) for i in range(1, 5)}
    header = [
        "salt",
        "ovf_pct",
        "srh_status",
        "srh_pct",
        "srh_sd_pct",
        "drh_pct",
        "drh_sd_pct",
        "erh_pct",
        "erh_sd_pct",
        "inferred_srh_temp_c",
        "comparison_role",
        "srh_workbook_label",
        "srh_workbook_cells",
    ]
    with (output / "phase-data.csv").open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for salt, srh_row, drh_sheet, erh_row in [
            ("AS", 4, 2, 11),
            ("NaCl", 14, 1, 27),
        ]:
            # The two Figure 6 workbook C8 labels are swapped relative to article
            # Figure 6 (sulfate in orange) and the numerical series in §3.2.1.
            label = sheets[3][f"A{srh_row}"]
            assert label == "C8H17SO3Na"
            for ovf, cols in [(50, "BC"), (70, "DE"), (90, "FG")]:
                srh, srh_sd = mean_sd(sheets[3], srh_row, cols)
                drh = mean_sd(sheets[drh_sheet], 11, cols)
                erh = mean_sd(sheets[4], erh_row, cols)
                role = "calibration" if salt == "AS" else "cross_salt_check"
                if not srh:
                    role = "qualitative_cross_salt_check"
                writer.writerow(
                    [
                        salt,
                        ovf,
                        "observed" if srh else "not_observed",
                        srh,
                        srh_sd,
                        *drh,
                        *erh,
                        inferred_temp_c(float(srh)) if srh else "",
                        role,
                        label,
                        f"Figure 6!{cols[0]}{srh_row}:{cols[1]}{srh_row}",
                    ]
                )
