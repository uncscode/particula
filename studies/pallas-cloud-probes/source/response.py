# %% Fixed FSSP calibration scenarios on the same independent observed spectra
"""Run with pilot or full on a CUDA host. Each Warp thread integrates one spectrum."""

import sys
import time
import json
import warp as wp
import numpy as np
import pandas as pd
from study import ROOT, inputs, native, window


@wp.kernel
def integrate(
    n: wp.array2d(dtype=wp.float64),
    e: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
):
    t, k = wp.tid()
    total = wp.float64(0.0)
    for b in range(n.shape[1]):
        left = e[b] * scale[k]
        right = e[b + 1] * scale[k]
        low = wp.clamp(wp.float64(6.0), left, right)
        high = wp.clamp(wp.float64(45.0), left, right)
        total += n[t, b] * wp.log(high / low) / wp.log(right / left)
    out[t, k] = total


def sweep(n, e, scales):
    dev = "cuda:0"
    ns = wp.array(n, dtype=wp.float64, device=dev)
    es = wp.array(e, dtype=wp.float64, device=dev)
    ss = wp.array(scales, dtype=wp.float64, device=dev)
    outs = wp.empty((len(n), len(scales)), dtype=wp.float64, device=dev)
    wp.launch(integrate, dim=outs.shape, inputs=[ns, es, ss, outs], device=dev)
    result = outs.numpy()
    for k, s in enumerate(scales):
        np.testing.assert_allclose(
            result[:, k], n @ window(e * s, 6, 45), atol=2e-11, rtol=1e-12
        )
    return result


def run(mode):
    start = time.monotonic()
    f, i = inputs()
    if mode == "pilot":
        f = f[f.index.day == 14].iloc[::10]
        i = i.loc[f.index]
    ei, ni, _ = native(i, "I", "center")
    obs = ni @ window(ei, 6, 45)
    fine_rows = []
    fit_rows = []
    grids = []
    calls = 0
    for nc, ng in (
        [(3, 3)] if mode == "pilot" else [(21, 17), (81, 65), (161, 129)]
    ):
        scales = np.linspace(0.8, 1.2, nc)
        gains = np.linspace(0.84, 1.16, ng)
        for conv in ["upper", "center"]:
            ef, nf, _ = native(f, "F", conv)
            counts = sweep(nf, ef, scales)
            calls += counts.size
            pilot = f.index.day == 14
            # For flat minima prefer the solution nearest nominal calibration.
            scores = np.array(
                [
                    abs(counts[pilot] * g - obs[pilot, None]).mean(axis=0)
                    for g in gains
                ]
            )
            best = scores.min()
            gj, sj = min(
                np.argwhere(np.isclose(scores, best, rtol=0, atol=1e-10)),
                key=lambda pair: (gains[pair[0]] - 1) ** 2
                + (scales[pair[1]] - 1) ** 2,
            )
            for day in np.unique(f.index.day):
                mask = f.index.day == day
                x = counts[mask]
                y = obs[mask]
                fit_rows.append(
                    dict(
                        nscale=nc,
                        ngain=ng,
                        geometry=conv,
                        day=day,
                        scale=scales[sj],
                        gain=gains[gj],
                        nominal_mae=np.mean(abs(x[:, nc // 2] - y)),
                        adjusted_mae=np.mean(abs(x[:, sj] * gains[gj] - y)),
                        adjusted_gap=np.mean(x[:, sj] * gains[gj] - y),
                        min_size_gap=min(x.mean(axis=0)) - y.mean(),
                        max_size_gap=max(x.mean(axis=0)) - y.mean(),
                        min_joint_gap=min(x.mean(axis=0)) * 0.84 - y.mean(),
                        max_joint_gap=max(x.mean(axis=0)) * 1.16 - y.mean(),
                    )
                )
                if nc == 161:
                    for k, s in enumerate(scales):
                        fine_rows.append(
                            dict(
                                geometry=conv,
                                day=day,
                                scale=s,
                                f_common=x[:, k].mean(),
                                i_common=y.mean(),
                                nominal_gain_mae=np.mean(abs(x[:, k] - y)),
                            )
                        )
            grids.append(
                dict(
                    size_steps=nc,
                    gain_steps=ng,
                    geometry=conv,
                    calibration_minimum_mae=best,
                )
            )
    pd.DataFrame(fit_rows).to_csv(
        ROOT / ("response-" + mode + ".csv"), index=False
    )
    if fine_rows:
        pd.DataFrame(fine_rows).to_csv(ROOT / "sizing.csv", index=False)
    info = dict(
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        seconds=time.monotonic() - start,
        completed_cuda_spectrum_integrals=calls,
        checked_against_numpy=calls,
        grids=grids,
        seed=None,
    )
    (ROOT / ("response-" + mode + ".json")).write_text(
        json.dumps(info, indent=2) + "\n"
    )


if __name__ == "__main__":
    run(sys.argv[1])
