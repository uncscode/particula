"""Paired first-passage simulations from the size-resolved cold CLOUD source."""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
from design import draws

sys.path.insert(0, str(HERE.parent / "analysis"))
from particles import BASE, COLUMNS, F, follow, wp, par
from reference import ACID, AIR, NH3, T
from attraction import acid_pair

LINES = (332, 334, 336, 338, 344, 366)
LAWS = {"vapor": 0, "attraction": 5}
GRID = np.arange(111) * 0.1 + 1
P = 1.005 * 101325
RADIUS = (3 * 26.1 / (4 * np.pi)) ** (1 / 3)
NATIVE = par.dynamics.get_spherical_wall_loss_coefficient_via_system_state


@lru_cache(None)
def reference_eddy(temp):
    # Match the historical cold 0.001/D_nm sink at 3 nm, at this row's temperature.
    return brentq(
        lambda e: NATIVE(e, 1.5e-9, 1500.0, temp, P, RADIUS) - 0.001 / 3,
        1e-6,
        100,
    )


def conditions(line, draw, law):
    i = line - 2
    return dict(
        line=line,
        id=draw["id"],
        block=draw["block"],
        law=law,
        temp=T[i],
        acid=ACID[i] * 1e6 * draw["acid"],
        base=NH3[i] * AIR[i] * 1e-12 * draw["base0"],
        wall_scale=draw["wall"],
        dilution=9.6e-5,
        background_cm3=1000 * draw["u0"],
        water_per_acid=2 * draw["u1"],
        sticking=0.5 + 0.5 * draw["u2"],
        eddy_s=reference_eddy(T[i]) * draw["wall"] ** 2,
    )


@lru_cache(None)
def collector(temp):
    r = np.r_[GRID, 50.0] * 0.5e-9
    m = 4 * np.pi / 3 * r**3 * 1500.0
    return par.dynamics.get_brownian_kernel_via_system_state(r, m, temp, P)[
        :-1, -1
    ]


@lru_cache(None)
def acid_table(temp, water):
    size, kernel = acid_pair(temp, P, 6.4e-20, water)
    return np.interp(GRID, size, kernel)


def gpu_inputs(c):
    p = [
        c["temp"],
        c["acid"],
        c["base"],
        1.0,
        c["dilution"],
        c["background_cm3"],
        c["water_per_acid"],
        BASE["capacity"],
        c["sticking"],
        LAWS[c["law"]],
        1.0,
        0.3,
        0.0,
    ]
    acid = (
        np.zeros(len(GRID))
        if c["law"] == "vapor"
        else acid_table(c["temp"], c["water_per_acid"])
    )
    wall = NATIVE(c["eddy_s"], GRID * 0.5e-9, 1500.0, c["temp"], P, RADIUS)
    assert np.all(wall > 0)
    return p, np.r_[collector(c["temp"]), acid, wall]


def replay(cases, tracks, seed, device):
    inputs = [gpu_inputs(c) for c in cases]
    params, kernels = (
        wp.array(np.array(x), dtype=F, device=device) for x in zip(*inputs)
    )
    states = wp.zeros(
        (len(cases), tracks, len(COLUMNS)), dtype=F, device=device
    )
    wp.launch(
        follow,
        dim=(len(cases), tracks),
        inputs=[params, kernels, states, tracks, seed],
        device=device,
    )
    vals = states.numpy()
    assert np.all(np.isfinite(vals)) and vals[:, :, 12].min() >= 10
    error = abs(vals[:, :, 1] + vals[:, :, 4:7].sum(axis=2) - 1)
    mass = abs(vals[:, :, 7:9])
    assert error.max() < 1e-9 and mass.max() < 1e-7
    for c, rows, err, bal in zip(cases, vals, error, mass):
        assert np.all(
            (rows[:, 1] >= 0) & (rows[:, 1] <= rows[:, 0]) & (rows[:, 0] <= 1)
        )
        yield dict(
            **c,
            seed=seed,
            tracks=tracks,
            **dict(zip(COLUMNS, rows.mean(axis=0))),
            se32=rows[:, 0].std(ddof=1) / np.sqrt(tracks),
            se10=rows[:, 1].std(ddof=1) / np.sqrt(tracks),
            max_number_error=err.max(),
            max_species_error=bal.max(),
        )


def analytic_oracle(cases, device):
    """Integrate the known hazard along a prescribed 2 nm/h growth trajectory."""
    values = []
    for case in cases:
        params, kernel = gpu_inputs(case)
        params[9:11] = [2, 0.125]
        out = wp.zeros((1, 1, len(COLUMNS)), dtype=F, device=device)
        wp.launch(
            follow,
            dim=(1, 1),
            inputs=[
                wp.array(np.array([params]), dtype=F, device=device),
                wp.array(np.array([kernel]), dtype=F, device=device),
                out,
                1,
                7531,
            ],
            device=device,
        )
        result = out.numpy()[0, 0]
        for i, threshold in enumerate([3.2, 10.0]):
            d = np.linspace(1.4, threshold - 0.3, 10001)
            native_wall = NATIVE(
                case["eddy_s"], d * 0.5e-9, 1500.0, case["temp"], P, RADIUS
            )
            coag = (
                np.interp(d, GRID, collector(case["temp"]))
                * 1e6
                * case["background_cm3"]
            )
            integral = 1800.0 * np.trapezoid(
                native_wall + case["dilution"] + coag, d
            )
            ratio = float(result[i] / np.exp(-integral))
            assert abs(ratio - 1) < 0.01, ratio
            values.append(
                dict(
                    id=case["id"],
                    line=case["line"],
                    threshold_nm=threshold,
                    simulated=float(result[i]),
                    analytic=np.exp(-integral),
                    ratio=ratio,
                )
            )
    with (HERE / "transport-oracle.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=values[0])
        writer.writeheader()
        writer.writerows(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    cases = [
        conditions(line, d, law)
        for d in draws()
        for line in LINES
        for law in LAWS
    ]
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This stochastic replay uses the local CUDA device")
    dev = wp.get_device("cuda:0")
    if args.oracle:
        analytic_oracle(
            [
                c
                for c in cases
                if c["id"] in (0, 63)
                and c["line"] in (332, 336)
                and c["law"] == "vapor"
            ],
            dev,
        )
        return
    if args.pilot:
        print(list(replay(cases[:2], 64, 71827, dev)), flush=True)
        return
    for label, selected, tracks, seed in [
        ("tracks", cases, 384, 71827),
        ("replicate", [c for c in cases if c["id"] % 8 == 0], 768, 71941),
    ]:
        path = HERE / f"transport-{label}.csv"
        with path.with_suffix(".tmp").open("w", newline="") as f:
            for start in range(0, len(selected), 48):
                rows = list(
                    replay(selected[start : start + 48], tracks, seed, dev)
                )
                if start == 0:
                    writer = csv.DictWriter(f, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
                f.flush()
                if start % 192 == 0:
                    print(label, start, len(selected), flush=True)
        path.with_suffix(".tmp").replace(path)
    (HERE / "transport-device.json").write_text(
        json.dumps(
            dict(
                device=dev.name,
                particula=par.__version__,
                warp=wp.__version__,
                nominal_eddy_s={
                    str(t): reference_eddy(t) for t in T[np.array(LINES) - 2]
                },
                cases=len(cases),
                primary_paths=len(cases) * 384,
                independent_paths=192 * 768,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
