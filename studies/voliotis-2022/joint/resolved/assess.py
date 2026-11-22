"""Higher-resolution checks of the refit and the three microphysical kernels."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from refit import RESULTS, write_json
from scoring import Targets
from simulate import BANK, HERE, OUT, Chamber


def main():  # noqa: C901 - selection and common replay controls
    """Select a higher-resolution calibration and test separate realizations."""
    saved = json.loads((RESULTS / "fit.json").read_text())
    c = Chamber()
    y = Targets(c, surface_window=True)
    base, previous, tuned = [
        np.array(saved[k]) for k in ("initial", "previous_joint", "selected")
    ]
    # Correct to the DMPS surface window and compare family of refits at more
    # slots.
    choices = {
        "frozen": base,
        "previous_joint": previous,
        "optimized": tuned,
        "halfway": 0.5 * (tuned + base),
    }
    # Direct AD through all 50 downstream rate constants, conditional on
    # individual-particle states sampled from the selected stochastic run.
    warm = c.run(tuned[None, :-2], c.ids, n=768, dt=30, rng=113, snapshot=True)
    h = warm[0][0]
    grid = np.arange(61) * 300
    pools = []
    residual = []
    normalization = []
    for i, (_, _, mt, observed, _) in enumerate(c.runs):
        for first in range(0, 55, 3):
            end = first + 6
            if grid[end] > mt[-1]:
                continue
            labels = h[i, first, 22:37].reshape(3, 5)
            pools.append(np.vstack((labels, c.gas_history[i, first, 26:31])).T)
            residual.append(h[i, end, 0] - np.interp(grid[end], mt, observed))
            normalization.append(max(1.0, observed.max()))
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "chemical_inputs.npz"
        np.savez(
            path,
            pools=pools,
            residual=residual,
            normalization=normalization,
            rates=tuned[36:86],
        )
        subprocess.run(  # noqa: S603 - local study interpreter and script
            [
                os.environ.get("MANCHESTER_TORCH_PYTHON", sys.executable),
                str(HERE / "aging_ad.py"),
                str(OUT / "resolution_checked_mechanism.json"),
                str(path),
                str(RESULTS / "aging_ad.json"),
            ],
            check=True,
        )
    aging = tuned.copy()
    aging[36:86] = json.loads((RESULTS / "aging_ad.json").read_text())["rates"]
    choices["aging_AD"] = aging
    choices["aging_AD_half"] = 0.5 * (aging + tuned)
    for j in (0.0, 0.02, 0.10, 0.30):
        x = tuned.copy()
        x[-3] = j
        choices[f"J_{j:g}"] = x
    for gsd in (1.45, 1.65, 1.85, 2.05):
        changed = tuned.copy()
        changed[-5] = gsd
        choices[f"GSD_{gsd:g}"] = changed
    scores = []

    def evaluate(label, pars, n, dt, seed, **kwargs):
        run = c.run(pars[:, :-2], c.ids, n=n, dt=dt, rng=seed, **kwargs)
        status, recs = y.evaluate(run, pars, c.ids, details=len(pars) == 1)
        status = status.assign(
            case=label, n=n, dt=dt, seed=seed, pmax=run[2][:, 4].max()
        )
        return status, recs, run

    calibrated = []
    # Batch pairs to bound temporary GPU memory.
    names = list(choices)
    for pair in range(0, len(names), 2):
        labels = names[pair : pair + 2]
        pars = np.array([choices[name] for name in labels])
        score, _, _ = evaluate("selection", pars, 768, 30, 113)
        calibrated.append(score.assign(candidate=labels))
    screen = pd.concat(calibrated, ignore_index=True)
    screen.to_csv(RESULTS / "selection.csv", index=False)
    winner = screen.loc[screen.loss.idxmin(), "candidate"]
    best = choices[winner]
    write_json(
        "selected.json",
        dict(
            candidate=winner,
            vector=best.tolist(),
            chemistry_keys=[r.key for r in BANK],
            ids=c.ids,
        ),
    )
    print(
        "selected",
        winner,
        screen[["candidate", "loss"]].to_dict("records"),
        flush=True,
    )
    detail = {}
    runs = []
    for label, p, n, dt, seed, options in [
        ("frozen", base, 1024, 30, 201, {}),
        ("refit", best, 1024, 30, 201, {}),
        ("frozen", base, 1024, 30, 202, {}),
        ("refit", best, 1024, 30, 202, {}),
        ("frozen", base, 2048, 15, 203, {}),
        ("refit", best, 2048, 15, 203, {"snapshot": True}),
        ("finer_step", best, 1024, 15, 201, {}),
        ("more_particles", best, 2048, 30, 201, {}),
        ("no_wall", best, 1024, 30, 201, {"wall_enabled": False}),
        ("no_coag", best, 1024, 30, 201, {"coag_enabled": False}),
        ("no_birth", best, 1024, 30, 201, {}),
        ("birth_0.10", best, 1024, 30, 201, {}),
    ]:
        v = p.copy()
        if label == "no_birth":
            v[-3] = 0.0
        if label == "birth_0.10":
            v[-3] = 0.10
        sc, report, result = evaluate(label, v[None], n, dt, seed, **options)
        scores.append(sc)
        print(label, n, dt, seed, sc.loss.iloc[0], flush=True)
        for key, frame in report:
            detail.setdefault(key, []).append(
                frame.assign(case=label, n=n, dt=dt, seed=seed)
            )
        if label in ("frozen", "refit"):
            states, dist, events, snapshot, air = result
            row = c.ids.index(24)
            centers = 2 * 1.12 ** (np.arange(60) + 0.5)
            inside = (centers >= 22) & (centers <= 540)
            freqs = dist[row, :, :60][:, inside]
            gm = np.exp((freqs @ np.log(centers[inside])) / freqs.sum(axis=1))
            for j, runid in enumerate(c.ids):
                for k, frame in enumerate(states[0, j]):
                    runs.append(
                        dict(
                            case=label,
                            seed=seed,
                            n=n,
                            dt=dt,
                            run=runid,
                            time_s=k * 300,
                            soa=frame[0],
                            count_cm3=frame[3] / (air[j] * 1e6),
                            run24_gmd_nm=gm[k] if runid == 24 else np.nan,
                        )
                    )
            if snapshot is not None:
                np.savez_compressed(
                    RESULTS / "particle_examples.npz",
                    run_ids=[6, 15, 24],
                    particle_ug=np.stack(
                        [
                            snapshot[c.ids.index(i)] * air[c.ids.index(i)]
                            for i in [6, 15, 24]
                        ]
                    ),
                    sample_volume_m3=air[[c.ids.index(i) for i in [6, 15, 24]]],
                )
    pd.concat(scores, ignore_index=True).to_csv(
        RESULTS / "checks.csv", index=False
    )
    for key, value in detail.items():
        pd.concat(value, ignore_index=True).to_csv(
            RESULTS / f"{key}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    pd.DataFrame(runs).to_csv(
        RESULTS / "histories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print("completed", flush=True)


if __name__ == "__main__":
    main()
