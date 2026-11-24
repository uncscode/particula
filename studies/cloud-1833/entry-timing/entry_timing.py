"""Cross the amplitude and timing of the inlet independently of downstream growth."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from physics import PACKET, common_input, d, e, frame, geometry, metrics, segments, stage, t, y

outdir = Path(os.environ["RESULTS"])
idx = segments["test"]
hours = t[idx]
width = np.maximum(0, e[1:] - np.maximum(e[:-1], np.log10(9.7)))
observation = y[idx] @ width
crossings = [hours[np.flatnonzero(observation >= threshold)[0]] for threshold in [1000, 5000]]
early = hours <= crossings[1]
drivers = np.load(PACKET / "inverse-model/final-gas.npz")
estimated = common_input()[idx]
predicted = drivers["test_288_1_1_0_80"][:, 1]
ratio = np.trapezoid(predicted[early], hours[early]) / np.trapezoid(estimated[early], hours[early])
entries = {
    "empirical": estimated,
    "predicted timing, empirical total": predicted / ratio,
    "empirical timing, predicted total": estimated * ratio,
    "predicted": predicted,
}
gases = {
    "measured": frame.measured_cal.to_numpy()[idx],
    "empirical-branch gas": drivers["test_288_1_0_0_80"][:, 0],
    "predicted-branch gas": drivers["test_288_1_1_0_80"][:, 0],
}

def replay(n, flux):
    geom = geometry(n, "aiomfac", 1.)
    velocities = np.array([geom["vel"][0], geom["vel"][1], geom["vel"][0]])
    count = len(gases)
    initial = geom["mi"] @ y[idx[0]]
    inputs = [
        hours, np.array(list(gases.values())), np.ones(len(idx)), initial,
        np.r_[velocities.ravel(), velocities.max(axis=1)], geom["wall"],
        np.tile([1.1, .5, 1., 1.], (count, 1)), geom["K"], *geom["pairs"],
        geom["over"], np.zeros(len(idx)), flux,
    ]
    arrays = [wp.array(np.ascontiguousarray(x), device="cuda:0") for x in inputs]
    result = wp.empty((count, len(idx), n), dtype=wp.float64, device="cuda:0")
    ledgers = wp.empty((count, n, 6), dtype=wp.float64, device="cuda:0")
    lib = ctypes.CDLL(os.environ["CLOUD_SOLVER"])
    lib.replay.argtypes = [ctypes.c_int]*3 + [ctypes.c_double]*2 + [ctypes.POINTER(ctypes.c_void_p)]
    args = (ctypes.c_void_p * 17)(*[v.ptr for v in arrays + [result, ledgers]])
    wp.synchronize()
    assert lib.replay(count, n, len(idx), geom["dx"], 6 if n == 288 else 12, args) == 0
    values = result.numpy()
    np.testing.assert_allclose(values[:, 0], np.tile(initial, (count, 1)), atol=1e-10)
    budget = ledgers.numpy().sum(axis=1)
    added = initial.sum() + budget[:, 0]
    error = np.max(abs(values[:, -1].sum(axis=1) - added + budget[:, 1:].sum(axis=1)) / added)
    assert np.all(np.isfinite(values)) and np.min(values) >= 0 and error < 1e-9
    return values @ geom["mo"].T / np.diff(e), error

results = []
series = {"archive_h": hours, "observed": observation}
max_error = 0.
for n in [192, 288]:
    for entry, flux in entries.items():
        spectra, error = replay(n, flux)
        max_error = max(max_error, error)
        for gas, spec in zip(gases, spectra):
            rows = metrics(spec, idx)
            aggregate = rows[0]
            mode = next(r["mode_nm"] for r in rows if r["stage"] == 12)
            first = aggregate["first1000"]
            second = aggregate["first5000"]
            results.append(dict(cells=n, entry=entry, acid=gas, first_error_min=(first-crossings[0])*60,
                                second_error_min=(second-crossings[1])*60, rise_minutes=(second-first)*60,
                                stage12_mode_nm=mode, spectrum_error_factor=aggregate["factor"]))
            if n == 288:
                series[f"{entry} | {gas}"] = spec @ width
pd.DataFrame(results).to_csv(outdir / "comparisons.csv", index=False)
pd.DataFrame(series).to_csv(outdir / "tail.csv", index=False)
(outdir / "checks.json").write_text(json.dumps(dict(device=wp.get_device("cuda:0").name, boxes=len(results),
    supply_ratio_before_observed_second_crossing=ratio, observed_crossings_archive_hours=crossings,
    max_relative_number_budget_error=max_error), indent=2))
