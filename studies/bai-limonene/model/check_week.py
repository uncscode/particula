"""Reproduce paired July cloud and radiation means for clean 0.25 m/s ascent."""

import os
from pathlib import Path
import numpy as np
import pandas as pd

results = Path(__file__).resolve().parent / "seasonal/results/resolved-week"
dark = pd.read_csv(results / "m00_clean_dark.csv").iloc[0]
light = pd.read_csv(results / "m07_clean_ranked.csv").sort_values(
    "hour_in_week"
)
fields = ["activated_cm3", "median_um"]
changes = [
    np.interp(np.arange(168), light.hour_in_week, light[k]).mean() - dark[k]
    for k in fields
]
flux = pd.read_csv(
    Path(os.environ["SEASONAL_TMP"])
    / "radiation/july-week/resolved-week_07_clean_0.25_ranked.csv.gz"
)
assert len(flux) == 168 and flux.hour_in_week.nunique() == 168
changes.append(flux.delta_up_site_wm2.mean())
np.testing.assert_allclose(
    changes, [-6.3422663, 1.2473335, -1.1800089], rtol=0, atol=1e-5
)
print(
    "July paired number (cm^-3), median (µm), and upward sunlight (W/m²)",
    changes,
)
