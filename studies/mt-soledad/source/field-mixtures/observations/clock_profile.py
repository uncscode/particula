"""Compare the authors' entire black mean-GF traces to raw scalar timestamps."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

base = Path(__file__).resolve().parent
im = np.asarray(Image.open(base / "author-summary.png").convert("RGB"))
obs = pd.read_csv(base / "scan-qc.csv.gz")
# Convert a tick-mark pixel coordinate to nominal UTC. Dates lack clock labels.
tick = np.array([1145.5, 1724.5, 2302.5, 2881.5, 3460.5])
slope, intercept = np.polyfit(tick, np.arange(5) * 10, 1)
cols = np.arange(691, 3735)
stamp = (
    pd.Timestamp("2023-05-01", tz="UTC").timestamp()
    + (slope * cols + intercept) * 86400
)
# End ticks at GF 2.5 and 0.5 bordering each source spacetime panel.
rows = []
for size, y25, y05 in [(90, 125, 1093), (60, 1219, 2283), (30, 2380, 3345)]:
    ys = np.arange(y25, y05)
    # The scalar-data line occupies the lower section, clear of the legend.
    g = 2.5 - 2 * (ys - y25) / (y05 - y25)
    pix = np.flatnonzero((g >= 0.75) & (g <= 1.75))
    crop = im[ys[pix]][:, cols]
    black = crop.max(axis=2) < 30
    colvalid = black.any(axis=0)
    estimated = np.array(
        [np.median(g[pix][col]) if col.any() else np.nan for col in black.T]
    )
    raw = obs[(obs.size_nm == size) & obs.author_mean.notna()].copy()
    for dh in np.arange(-12, 12.1, 0.25):
        t = raw.raw_time + dh * 3600
        j = np.searchsorted(t, stamp)
        j = np.clip(j, 1, len(t) - 1)
        # Short size switches need interpolation. Leave gaps >4 hours out.
        ok = (
            colvalid
            & ((t.to_numpy()[j] - t.to_numpy()[j - 1]) <= 4 * 3600)
            & (stamp >= t.min())
            & (stamp <= t.max())
        )
        field = np.interp(stamp[ok], t, raw.author_mean)
        vals = estimated[ok] - field
        rows.append(
            dict(
                size_nm=size,
                shift_raw_to_figure_h=dh,
                supported_pixels=ok.sum(),
                mae=np.abs(vals).mean(),
                rmse=np.sqrt(np.mean(vals**2)),
                correlation=np.corrcoef(estimated[ok], field)[0, 1],
            )
        )
df = pd.DataFrame(rows)
df.to_csv(base / "clock-profile.csv", index=False)
print(df.loc[df.groupby("size_nm").mae.idxmin()].to_string(index=False))
print(df.loc[df.groupby("size_nm").correlation.idxmax()].to_string(index=False))
