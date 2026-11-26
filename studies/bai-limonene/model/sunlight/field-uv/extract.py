"""Digitize the absolute measured ground spectra in Bohn & Lohse Fig. 17."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
from xml.etree import ElementTree

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PDF_HASH = "02e89b0a179aa267a7fca1b690e5cc6499911ed3c62d9b8643e29681a6425d8a"
SS = "{http://www.w3.org/2000/svg}"


def svg_paths(svg):
    for elem in ElementTree.parse(svg).getroot().iter(SS + "path"):
        attr = elem.attrib
        if not attr.get("stroke") or "matrix" not in attr.get("transform", ""):
            continue
        matrix = np.array(
            [float(v) for v in re.findall(r"[-+]?[\d.]+", attr["transform"])]
        )
        xy = np.array(
            [float(v) for v in re.findall(r"[-+]?[\d.]+", attr["d"])]
        ).reshape(-1, 2)
        yield attr["stroke"], xy @ matrix[:4].reshape(2, 2) + matrix[4:]


def measured(svg, panel, color):
    # Derive axes from published vector lines. These point coordinates identify
    # each set of axes unambiguously on physical page 19 in the cited PDF.
    panel_wavelength, max_flux, y_lo, y_hi = {
        "a": (320.0, 5e13, 69.69, 183.40),
        "b": (420.0, 5e14, 209.85, 323.56),
    }[panel]
    paths = list(svg_paths(svg))
    axes = [
        q
        for c, q in paths
        if c == "rgb(0%, 0%, 0%)"
        and len(q) == 2
        and np.ptp(q[:, 0]) > 150
        and np.ptp(q[:, 1]) < 0.001
        and y_lo - 0.05 <= q[0, 1] <= y_hi + 0.05
    ]
    if len(axes) != 2:
        raise ValueError(f"Ambiguous axes in published panel {panel}")
    ybottom, ytop = sorted([a[0, 1] for a in axes], reverse=True)
    xleft, xright = sorted(axes[0][:, 0])
    rows = []
    for c, q in paths:
        if (
            c != color
            or len(q) != 2
            or not ytop - 0.5 < np.mean(q[:, 1]) < ybottom + 0.5
        ):
            continue
        nm = 280.0 + (panel_wavelength - 280.0) * (q[:, 0] - xleft) / (
            xright - xleft
        )
        # Coloured line segments linking consecutive wavelength measurements,
        # eliminating same-colour vertical uncertainty bars and legend marks.
        if (
            0.70 < nm[1] - nm[0] < 1.1
            and nm.min() > 279.99
            and nm.max() <= panel_wavelength + 0.02
        ):
            rows.extend(
                zip(
                    nm,
                    (ybottom - q[:, 1]) / (ybottom - ytop) * max_flux,
                    strict=True,
                )
            )
    d = pd.DataFrame(rows, columns=["wavelength_nm", "flux"]).drop_duplicates()
    # DM-SR scans at 1 nm. Retain the actual 0.8-nm CCD wavelengths.
    if color == "rgb(100%, 0%, 0%)":
        d["wavelength_nm"] = np.round(d["wavelength_nm"])
    # The cropped panel sometimes creates a clipping segment at a published
    # symbol (on panel b below 300 nm). Panel a handles these smaller signals.
    return (
        d.groupby("wavelength_nm", as_index=False)
        .median()
        .sort_values("wavelength_nm")
    )


def main(pdf):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_HASH:
        raise ValueError("Unexpected primary PDF revision")
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "p19.svg"
        subprocess.run(
            ["pdftocairo", "-f", "19", "-l", "19", "-svg", str(pdf), str(svg)],
            check=True,
        )
        rows = []
        for detector, color in [
            ("DM-SR", "rgb(100%, 0%, 0%)"),
            ("CCD-SR", "rgb(0%, 0%, 100%)"),
        ]:
            for panel in ["a", "b"]:
                d = measured(svg, panel, color)
                d["detector"], d["panel"] = detector, panel
                rows.append(d)
    points = pd.concat(rows)
    points.to_csv(OUT / "digitized.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    main(parser.parse_args().pdf)
