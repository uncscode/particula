"""Regression test for the Bai weighted seasonal spectrum table."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = (
    Path(__file__).resolve().parents[1] / "studies/bai-limonene/model/seasonal"
)


def test_seasonal_table_uses_time_weighted_monthly_spectra(
    tmp_path, monkeypatch
):
    """Unequal observation intervals must survive the sun-minus-dark comparison."""
    columns = tmp_path / "columns"
    columns.mkdir()
    (tmp_path / "analysis").mkdir()
    # Samples at hours 0 and 55 represent 28 and 140 hours of the week.
    for month in range(13):
        mode = "ranked" if month else "dark"
        for profile, scale in [("clean", 1), ("rich", 2)]:
            for speed in [0.25, 2.0]:
                dark = scale * speed
                np.savez(
                    columns
                    / f"resolved_m{month:02d}_{profile}_{speed}_{mode}_0.npz",
                    wet_radius_m=np.full((2, 1, 1), 1.8e-6),
                    number_m3=np.array([dark + 6 * month, dark])[:, None, None]
                    * 1e6,
                    hour_in_week=[55, 0],
                    month=[month, month],
                    profile=[profile, profile],
                    w=[speed, speed],
                )

    monkeypatch.setenv("SEASONAL_TMP", str(tmp_path))
    monkeypatch.syspath_prepend(str(STUDY))
    main = runpy.run_path(str(STUDY / "size_spectra.py"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    main()

    table = pd.read_csv(
        tmp_path / "analysis/seasonal-fine-bin-differences.csv.gz"
    )
    signals = ["sun_number_cm3", "dark_number_cm3", "delta_number_cm3"]
    populated = table[table.lower_um == 3.5]
    assert len(table) == 16 * 146
    assert len(populated) == 16
    assert (table.loc[table.lower_um != 3.5, signals] == 0).all().all()
    for row in populated.itertuples():
        month_mean = {"DJF": 5, "MAM": 4, "JJA": 7, "SON": 10}[row.season]
        dark = {"clean": 1, "rich": 2}[row.profile] * row.w
        delta = 5 * month_mean
        assert row.sun_number_cm3 == dark + delta
        assert row.dark_number_cm3 == dark
        assert row.delta_number_cm3 == delta
