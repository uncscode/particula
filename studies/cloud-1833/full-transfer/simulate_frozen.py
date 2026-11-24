"""Execute the previously selected three common-entry models without refitting."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from study import simulate

here = Path(__file__).resolve().parent
packet = Path(os.environ.get("CLOUD_PACKET", here.parent)).resolve()


def run(output):
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(packet / "spectra.npz") as a:
        t, d, edges, obs, stage = [
            a[k] for k in ("time", "diameter", "edges", "observed", "stage")
        ]
    f = pd.read_csv(packet / "forcing.csv")
    np.testing.assert_array_equal(f.stage, stage)
    np.testing.assert_allclose(f.archive_h, t, atol=1e-12)
    gas = f.measured_cal.to_numpy()[None, :]
    missing = f.small_missing.to_numpy()
    settings = json.loads((packet / "campaign.json").read_text())
    growth, wall = settings["common_growth"], settings["common_wall"]
    cases = [(0, growth, wall, law, 1) for law in (0, 2, 1)]
    data = t, d, edges, obs, gas, stage, missing
    wp.init()
    device = wp.get_device("cuda:0")
    ref_case = [(0, growth, wall, 0, 0)]
    sample = np.arange(16)
    cpu = simulate(data, sample, ref_case, 64, 24, reference=True)
    gpu, pilot = simulate(data, sample, ref_case, 64, 24)
    np.testing.assert_allclose(cpu, gpu[0], rtol=1e-9, atol=1e-7)
    logs = []
    for cells, step in [(192, 12), (288, 6)]:
        histories = {}
        for label, window, fill in [
            ("training", (1, 7), "zero"),
            ("excimer_led", (9, 19), "zero"),
            ("excimer_led_hold", (9, 19), "last"),
            ("mixed", (21, 25), "zero"),
        ]:
            ix = np.flatnonzero((stage >= window[0]) & (stage <= window[1]))
            assert (
                np.diff(ix).min() == np.diff(ix).max() == 1
                and not missing[ix[0]]
            )
            y = obs.copy()
            if fill == "last":
                y[missing, 3] = obs[np.flatnonzero(missing)[0] - 1, 3]
            pred, info = simulate(
                (t, d, edges, y, gas, stage, missing),
                ix,
                cases,
                cells,
                step,
                common_entry=growth,
            )
            assert np.isfinite(pred).all() and pred.min() >= 0.0
            supplied = np.array(info["ledgers"])[:, 1]
            # Different CFL substeps integrate the common time-varying inlet slightly differently.
            info["relative_entry_spread"] = float(
                np.ptp(supplied) / max(1.0, supplied.mean())
            )
            logs.append(dict(segment=label, **info))
            histories[label] = pred
            if cells == 288 and label == "excimer_led":
                with np.load(packet / "mechanism.npz") as old:
                    nx = old["fixed_flux"].shape[1]
                    np.testing.assert_allclose(
                        pred[:, :nx],
                        old["fixed_flux"][:3],
                        rtol=1e-9,
                        atol=2e-7,
                    )
        np.savez_compressed(output / f"predictions-{cells}.npz", **histories)

    root = os.environ["PARTICULA_ROOT"]
    info = dict(
        device=device.name,
        warp=wp.__version__,
        revision=subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip(),
        cases=1 + sum(row["cases"] for row in logs),
        seconds=time.monotonic() - start,
        pilot=pilot,
        cpu_gpu_max_difference=float(np.max(abs(cpu - gpu[0]))),
        runs=logs,
    )
    (output / "execution.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
