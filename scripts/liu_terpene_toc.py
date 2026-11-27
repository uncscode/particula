"""Regenerate the ocimene graphical abstract from the published data."""

import runpy
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/studies/literature/evaporation-and-uptake/liu-terpene"


def main():
    """Render the measurement and forecast ranges shared with Figure 2."""
    example = runpy.run_path(str(DOC / "compare_residues.py"))
    renderer = runpy.run_path(str(ROOT / "studies/liu-terpene/replot.py"))
    with TemporaryDirectory() as scratch:
        bounds = example["measurement_bounds"](ROOT, Path(scratch))
        renderer["toc"](
            ROOT / "docs/images/liu-terpene",
            ROOT / "docs/data/liu-terpene/figure2.csv",
            bounds,
        )


if __name__ == "__main__":
    main()
