"""Render the published figures offline from archived graph values."""

import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import replot
import replot_flights
import summary_chemistry
import summary_optics
import toc
from chemistry import plot_field_fit
from figure_archive import ARCHIVE
from summary_cold_acid import plot_cold_acid

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1] / "docs/images/sabre-small-mode"


def main():
    """Render optical and chemical comparisons and the graphical abstract."""
    with ZipFile(ARCHIVE) as archive:
        packed = json.loads(archive.read("figure-values.json"))
        cold_acid = packed.pop("cold_acid")
        tables = {key: pd.DataFrame(value) for key, value in packed.items()}
        with archive.open("jensen_spectra.csv") as stream:
            spectra = pd.read_csv(stream)
        with archive.open("decker-tracer-marks.csv") as stream:
            marks = pd.read_csv(stream)
    IMAGES.mkdir(exist_ok=True, parents=True)
    replot.render(
        tables["curves"],
        spectra,
        tables["annotations"],
        IMAGES / "mode-removal.webp",
    )
    replot_flights.render(
        tables["flights"],
        tables["sensitivity"],
        IMAGES / "flight-comparison.webp",
    )
    plot_field_fit.render(
        tables["field"],
        tables["profile"],
        marks,
        IMAGES / "field-uptake.webp",
    )
    summaries = {
        "number-spectrum": summary_optics.plot_spectrum(
            tables["curves"], spectra
        ),
        "aircraft-optics": summary_optics.plot_flight_optics(tables["flights"]),
        "cold-acid": plot_cold_acid(cold_acid),
        "field-predictions": summary_chemistry.plot_predictions(
            tables["quarters"]
        ),
    }
    for name, figure in summaries.items():
        figure.savefig(
            IMAGES / f"{name}.webp",
            dpi=155,
            facecolor="white",
            pil_kwargs={"lossless": True, "method": 6, "quality": 100},
        )
        plt.close(figure)
    toc.main()


if __name__ == "__main__":
    main()
