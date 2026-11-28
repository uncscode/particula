"""Check the published Pallas diameter spectra against independent band totals."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_pallas_rendered_spectra_recover_probe_band_totals(monkeypatch):
    """Integrating each plotted curve from 6 to 13 µm must recover its probe."""
    data = ROOT / "docs/data/pallas-cloud-probes"
    totals = (
        pd.read_csv(data / "attribution.csv")
        .query(
            "source == 'CDA' and stage == 'refined' and case == 'nominal' "
            "and calibration == 'frozen' and low_um == 6 and high_um == 13"
        )
        .set_index("day")
    )
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    try:
        runpy.run_path(
            str(
                ROOT
                / "docs/studies/literature/pallas-cloud-probes/pallas_cloud_probes.py"
            )
        )
        for axes, day in zip(figures[0].axes[:2], ["15", "16"], strict=True):
            expected = totals.loc[day, ["observed_f", "observed_i"]]
            for line, total in zip(axes.patches, expected, strict=True):
                heights, edges, _ = line.get_data()
                band = (edges[:-1] >= 6) & (edges[1:] <= 13)
                widths = np.diff(np.log10(edges))
                assert band.sum() == 5
                np.testing.assert_allclose(
                    heights[band] @ widths[band], total, rtol=1e-12
                )
    finally:
        for figure in figures:
            plt.close(figure)
