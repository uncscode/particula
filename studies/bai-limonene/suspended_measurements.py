"""Retrieve Wilson Table S1 and digitize Liu Figure 1b."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from replay import fetch
from scipy.optimize import minimize_scalar

PDFS = {
    "wilson-si.pdf": {
        "url": "https://ndownloader.figshare.com/files/3854887",
        "sha256": (
            "1d342660d4f9308edd9042ff00c6cf40250ab32b9ec48034924d870e294a7c17"
        ),
    },
    "liu.pdf": {
        "url": "https://d-nb.info/1170978797/34",
        "sha256": (
            "5924948c05655d40fda84380b6d9d4bcee00f2ce51a4b76431b097ccc01aa421"
        ),
    },
}


def wilson_volatility(pdf):
    """Read the fresh wet–wet limonene fractions on page S5."""
    page = subprocess.check_output(  # noqa: S603
        ["pdftotext", "-layout", "-f", "5", "-l", "5", pdf, "-"],  # noqa: S607
        text=True,
    )
    limonene = page.split("limonene\n")[-1]
    row = re.search(r"^wet-wet 90% RH\s+([\d. ]+)$", limonene, re.M)
    if row is None:
        raise ValueError("Cannot locate the wet–wet limonene row in Table S1")
    fractions = np.array(row[1].split(), dtype=float)
    result = pd.DataFrame(
        {"Cstar_ug_m3": [0.001, 0.01, 0.1, 1, 10], "mass_fraction": fractions}
    )
    np.testing.assert_allclose(fractions.sum(), 1.0)
    return result


def film_activation(diameter_nm):
    """Find peak κ-Köhler supersaturation (%) for fresh film κ at 298.15 K."""
    kelvin = 4 * 0.072 / (461.5228 * 1000 * 298.15 * diameter_nm * 1e-9)

    def negative_supersaturation(log_growth):
        water = np.expm1(3 * log_growth)
        return -100 * np.expm1(
            np.log(water / (water + 0.0465)) + kelvin / np.exp(log_growth)
        )

    return -minimize_scalar(
        negative_supersaturation, bounds=(1e-8, 7), method="bounded"
    ).fun


def liu_activation(pdf):
    """Read green marker centres using the logarithmic ticks of Figure 1b."""
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "p3.svg"
        subprocess.run(  # noqa: S603
            ["pdftocairo", "-f", "3", "-l", "3", "-svg", pdf, svg],  # noqa: S607
            check=True,
        )
        paths = ElementTree.parse(svg).getroot()  # noqa: S314 (local Poppler output)
    rows = []
    for marker in paths.iter():
        if not marker.tag.endswith("path") or not marker.get(
            "fill", ""
        ).startswith("rgb(58.03833%"):
            continue
        sx, _, _, sy, tx, ty = map(
            float, re.findall(r"-?[\d.]+", marker.attrib["transform"])
        )
        vertices = np.array(
            re.findall(r"-?[\d.]+(?:e-?\d+)?", marker.attrib["d"]), dtype=float
        ).reshape(-1, 2)
        x, y = (vertices.min(axis=0) + vertices.max(axis=0)) / 2 * [sx, sy] + [
            tx,
            ty,
        ]
        if x > 307 and 105 < y < 210:
            # Axis anchors are the 50/100 nm and 1.0/0.1% ticks.
            diameter = 50 * 2 ** ((x - 337.157924) / (377.355035 - 337.157924))
            ss = 10 ** (-(y - 107.221373) / (199.575049 - 107.221373))
            rows.append((diameter, ss, film_activation(diameter)))
    if len(rows) != 8:
        raise ValueError(
            f"Expected eight limonene markers in Figure 1b, found {len(rows)}"
        )
    return pd.DataFrame(
        sorted(rows), columns=["diameter_nm", "measured_ss_pct", "film_ss_pct"]
    )


def main():
    """Save the two independent reference tables in the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Download cache and derived CSV directory"
    )
    dest = parser.parse_args().output
    dest.mkdir(parents=True, exist_ok=True)
    fetch(dest, PDFS)
    wilson = wilson_volatility(dest / "wilson-si.pdf")
    liu = liu_activation(dest / "liu.pdf")
    wilson.to_csv(dest / "wilson-wet-wet.csv", index=False)
    liu.to_csv(dest / "liu-activation.csv", index=False)
    print(
        "Mass fraction at or below 0.1 µg/m³:",
        wilson.loc[wilson.Cstar_ug_m3 <= 0.1, "mass_fraction"].sum(),
    )
    print(liu.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
