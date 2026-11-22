"""Render the Manchester yield and two-hour size TOC from the archived fits.

The print PNG uses Inkscape at 300 dpi with the same SVG as the website.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.render_literature_tocs import IMAGES, render_svg

HERE = Path(__file__).resolve().parent


def spectrum_marks():
    """Map the two-hour size spectra to the logarithmic diameter axis."""
    spectra = pd.read_csv(HERE / "joint/resolved/results/psd.csv.gz")
    snapshot = spectra.query(
        "run == 24 and time_s == 7200 and n == 2048 "
        "and dt == 15 and seed == 203"
    )
    bins = snapshot.pivot(
        index=["low_nm", "high_nm", "dndlog10d_cm3"],
        columns="case",
        values="simulated",
    ).reset_index()
    diameter = np.sqrt(bins.low_nm * bins.high_nm)
    x = 40 + 231 * np.log(diameter / 25) / np.log(500 / 25)

    def y(values):
        return 245 - 48 * values / 20000

    marks = {}
    for case in ["refit", "frozen"]:
        marks[case] = " ".join(
            f"{a:.2f},{b:.2f}" for a, b in zip(x, y(bins[case]), strict=True)
        )
    marks["measurements"] = "\n".join(
        f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.8" fill="#243E43" '
        'stroke="#FBF2E3" stroke-width=".7"/>'
        for a, b in zip(x, y(bins.dndlog10d_cm3), strict=True)
    )
    return marks


def render():
    """Write the gallery vector, study copy and matching print-size PNG."""
    yields = pd.read_csv(HERE / "joint/companions/yield_points.csv")
    joint = yields.query("fit == 'joint' and run == 24")
    endpoint = joint.loc[joint.consumed.idxmax()]
    output = IMAGES / "manchester-soa/graphical-abstract.svg"
    render_svg(
        HERE / "toc.svg.in",
        output,
        consumed=f"{endpoint.consumed:.0f}",
        observed=f"{endpoint.observed * 100:.1f}",
        additive=f"{endpoint.paper_single_curve_yield * 100:.1f}",
        modeled=f"{endpoint.model * 100:.1f}",
        **spectrum_marks(),
    )
    print_svg = HERE / "results/figures/manchester-soa-toc.svg"
    print_svg.write_bytes(output.read_bytes())
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError("Install Inkscape to export the print PNG")
    subprocess.run(  # noqa: S603 - fixed Inkscape export arguments
        [
            inkscape,
            "--export-dpi=300",
            f"--export-filename={print_svg.with_suffix('.png')}",
            str(print_svg),
        ],
        check=True,
    )


if __name__ == "__main__":
    render()
