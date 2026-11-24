"""Fill the public graphical abstract from the three held optical statistics."""

import csv
from pathlib import Path

from scripts.render_literature_tocs import render_svg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIELDS = ("observed_bypass_wet_um", "observed_wet_um", "observed_ratio")


def render():
    """Write the square SVG, retaining the measured wet diameter ratio."""
    with (HERE / "toc-reference.csv").open() as stream:
        (row,) = csv.DictReader(stream)
    bypass, denuded, ratio = (float(row[k]) for k in FIELDS)
    template = (
        ROOT
        / "docs/studies/literature/particle-water-uptake/mt-soledad/toc.svg.in"
    )
    output = ROOT / "docs/images/mt-soledad/graphical-abstract.svg"
    render_svg(
        template,
        output,
        bypass=f"{bypass:.2f}",
        denuded=f"{denuded:.2f}",
        ratio=f"{ratio:.2f}",
        bypass_r="22",
        denuded_r=f"{22 * denuded / bypass:.4f}",
        bar=f"{104 * ratio:.4f}",
    )


if __name__ == "__main__":
    render()
