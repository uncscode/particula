"""Independent unit and population checks for the Manchester particle study."""

import numpy as np
import pandas as pd
from pathlib import Path

from scoring import spectra
from simulate import Chamber


def main():
    """Check log-bin units and the no-event, identical-particle limit."""
    test = pd.DataFrame(
        {"time_s": [0, 0], "low_nm": [30, 100], "high_nm": [100, 320]}
    )
    density = 23000.0
    volume = 2e-7
    hist = np.full((1, 60), density * volume * 1e6 * np.log10(1.12))
    np.testing.assert_allclose(spectra(hist, volume, test), density)
    # Every digitized row includes its own time and band. Check the row gather
    # against linear interpolation of cumulative number in log diameter.
    target = pd.read_csv(Path(__file__).parent / "size_data/run24_psd.csv")
    times = target.time_s.to_numpy(dtype=int) // 300
    h = (1 + np.arange(times.max() + 1))[:, None] * np.arange(1., 61)
    edges = np.log10(2) + np.arange(61) * np.log10(1.12)
    bounds = np.log10(target[["low_nm", "high_nm"]].to_numpy())
    expected = [
        np.diff(np.interp(pair, edges, np.r_[0, np.cumsum(h[t])]))[0]
        / (pair[1] - pair[0])
        for t, pair in zip(times, bounds, strict=True)
    ]
    np.testing.assert_allclose(spectra(h, 1e-6, target), expected, rtol=1e-13)
    chamber = Chamber()
    x = chamber.baseline()
    x[-3] = 1.0
    x[-1] = 0.0
    series, _, events, particle, _ = chamber.run(
        [x],
        [6, 15],
        n=128,
        dt=15,
        minutes=30,
        rng=4,
        snapshot=True,
        wall_enabled=False,
        coag_enabled=False,
    )
    for j, inventories in enumerate(particle):
        occupied = inventories[:128]
        np.testing.assert_allclose(
            occupied,
            np.broadcast_to(occupied[0], occupied.shape),
            atol=1e-11,
        )
        np.testing.assert_equal(events[j, [0, 1, 2, 3, 5]], 0)
        np.testing.assert_equal(series[0, j, :, 2], 128)
    print("Log-bin units and identical-particle controls passed.")


if __name__ == "__main__":
    main()
