"""Score the seven candidate views under changes to flow, grid and optical registration."""

from itertools import product
import numpy as np
import pandas as pd
from experiment import HERE, VIEWS, simulate

KEYS = list(VIEWS)
cases = ["import_above", "import_below", "local_growth", "local_mergers"]
SETTINGS = [
    (576, "jets", 1, 0.025),
    (576, "jets", 2, 0.05),
    (1536, "jets", 2, 0.05),
    (1944, "jets", 2, 0.025),
    (1944, "jets", 2, 0.05),
    (1944, "reverse", 2, 0.05),
    (2304, "jets", 2, 0.05),
]


def half(t, y):
    # Baseline-subtracted first upward half-height crossing of each camera's own positive peak.
    k = np.argmax(y)
    if y[k] < 0.003:
        return np.nan
    j = np.flatnonzero(y[: k + 1] >= 0.5 * y[k])[0]
    return t[j] if j == 0 else np.interp(0.5 * y[k], y[j - 1 : j + 1], t[j - 1 : j + 1])


def analyze():
    score = []
    timing = []
    bounded = []
    for mesh, flow, width, dt in SETTINGS:
        for case in ["control", *cases]:
            file = HERE / "runs" / f"{mesh}-{flow}-{width}-{dt}-{case}.npz"
            if not file.exists():
                simulate(mesh, flow, width, dt, case, "cuda:0")
        f = HERE / "runs" / f"{mesh}-{flow}-{width}-{dt}-control.npz"
        b = np.load(f)
        t = b["t"]
        use = (t >= 4.25) & (t < 6.25)
        setting = f.stem.rsplit("-", 1)[0]
        ys = {
            ce: np.load(f.parent / (setting + "-" + ce + ".npz"))["tail"].reshape(len(t), 3, 7)
            - b["tail"].reshape(len(t), 3, 7)
            for ce in cases
        }
        assert all(np.max(abs(v[t < 4.25])) < 1e-7 for v in ys.values())
        for r, shift in enumerate(b["registration_m"]):
            for ce, y in ys.items():
                for j, name in enumerate(KEYS):
                    timing.append(
                        dict(
                            setting=setting,
                            registration_cm=100 * shift,
                            case=ce,
                            view=name,
                            excess_peak_cm3=y[:, r, j].max(),
                            peak_s=t[np.argmax(y[:, r, j])],
                            half_rise_s=half(t, y[:, r, j]),
                            lead_s=half(t, y[:, r, 0]) - half(t, y[:, r, j]),
                        )
                    )
            for source, local in product(cases[:2], cases[2:]):
                diffs = []
                for ce in (source, local):
                    v = ys[ce][use, r, :]
                    diffs.append((v - v[:, [0]]) / max(v[:, 0].max(), 1e-10))
                rank = np.mean((diffs[0] - diffs[1]) ** 2, axis=0) ** 0.5
                contrasts = []
                for ce in (source, local):
                    v = ys[ce][use, r, :]
                    contrasts.append((v - v[:, [0]]) / max(0.01, v[:, 0].max()))
                bounded_rank = np.mean((contrasts[0] - contrasts[1]) ** 2, axis=0) ** 0.5
                for j, view in enumerate(KEYS):
                    bounded.append((source, local, view, bounded_rank[j]))
                for j, view in enumerate(KEYS):
                    score.append((setting, shift * 100, source, local, view, rank[j]))
    sr = pd.DataFrame(
        score, columns=["setting", "registration_cm", "source", "local", "view", "normalized_rms"]
    )
    sr.to_csv(HERE / "design-scores.csv", index=False, float_format="%.7g")
    pd.DataFrame(timing).to_csv(HERE / "timing.csv", index=False, float_format="%.7g")
    tab = sr.groupby("view").normalized_rms.agg(["min", "median", "max"])
    tab.to_csv(HERE / "placement-summary.csv", float_format="%.7g")
    pd.DataFrame(bounded, columns=["source", "local", "view", "score"]).groupby("view").score.agg(
        ["min", "median", "max"]
    ).to_csv(HERE / "placement-floor.csv", float_format="%.7g")
    return sr, tab


if __name__ == "__main__":
    analyze()
