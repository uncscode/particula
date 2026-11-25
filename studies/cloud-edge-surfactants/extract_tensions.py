"""Extract Shahabadi et al. Figure 5 from the original publisher ZIP."""

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
CHECKSUM = "858763efe5f2383fa41b02bd4c789c1189480960a13fda7a75e9c95551fdd965"
# Figure 5 source worksheet, inclusive row ranges, checked against the plot.
BLOCKS = (
    ("acid", "pendant", 2, 7),
    ("acid", "trap", 8, 20),
    ("octyl", "pendant", 21, 24),
    ("octyl", "trap", 25, 39),
    ("dodecyl", "pendant", 40, 43),
    ("dodecyl", "trap", 44, 56),
)


def extract(path):
    """Verify the original archive and regenerate the Figure 5 CSV."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CHECKSUM:
        raise ValueError(
            "Source archive checksum differs from the inspected release"
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        payload = archive.read("Surface-tension-and-RI.xlsx")
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        # The archive passed the hard-coded publisher checksum above.
        root = ElementTree.fromstring(  # noqa: S314
            book.read("xl/worksheets/sheet4.xml")
        )
    values = {c.get("r"): c.findtext("{*}v") for c in root.findall(".//{*}c")}
    with (HERE / "figure5.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "recipe",
                "method",
                "worksheet_row",
                "rh_percent",
                "sigma_mN_m",
                "reported_error_mN_m",
            ]
        )
        for recipe, method, start, end in BLOCKS:
            for row in range(start, end + 1):
                fields = [values.get(f"{col}{row}") for col in "BCD"]
                writer.writerow(
                    [recipe, method, row]
                    + [format(float(v), ".12g") if v else "" for v in fields]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publisher_zip", type=Path)
    extract(parser.parse_args().publisher_zip)
