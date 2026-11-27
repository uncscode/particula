# %% Inputs, reference solution and uncertainty in the two-component inversion
"""Run from the study checkout with PYTHONPATH=. and the selected environment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
import warp as wp

PACKET = Path(__file__).resolve().parent
SALT_DENSITY = 2.165  # g cm^-3; spherical anhydrous dry NaCl
PRODUCT_DENSITY = 2.742  # g cm^-3; spherical anhydrous dry NaHSO4
M_SALT, M_PRODUCT, M_SULFATE = 58.443, 120.06, 96.06  # g mol^-1


def to_sodium(volume, r=PRODUCT_DENSITY / M_PRODUCT / (SALT_DENSITY / M_SALT)):
    """Fraction of the original sodium now present in the product."""
    return volume * r / (1 - volume + volume * r)


def inputs():
    data = pd.read_csv(PACKET / "xiong/table-s1.csv").sort_values(
        ["diameter_nm", "time_h", "rh_pct"]
    )
    e = pd.read_csv(PACKET / "xiong/endmembers.csv").sort_values("rh_pct")
    sizes = data.diameter_nm.unique()
    times = data.time_h.unique()
    gf = data.gf.to_numpy().reshape(len(sizes), len(times), len(e))
    # Half last digit in addition to the printed sample scatter.
    spread = data.spread.to_numpy().reshape(gf.shape)
    center = e[["nacl_gf", "bisulfate_gf"]].to_numpy().T
    width = e[["nacl_spread", "bisulfate_spread"]].to_numpy().T + 0.010
    return sizes, times, e.rh_pct.to_numpy(), gf, spread, center, width


def zsr(v, a, b):
    return np.cbrt((1 - v) * a**3 + v * b**3)


def central_checks(out):
    sizes, times, rh, y, sd, end, width = inputs()
    a, b = end
    checks = []
    for name, fit in [
        ("low", rh <= 70),
        ("high", rh >= 80),
        ("all", np.ones(len(rh), bool)),
    ]:
        delta = (b**3 - a**3)[fit]
        # Linear least squares in cubed growth, scaled by first-order response of GF^3.
        weight = 1 / (3 * y[:, :, fit] ** 2 * (sd[:, :, fit] + 0.005)) ** 2
        raw = np.sum(
            weight * delta * (y[:, :, fit] ** 3 - a[fit] ** 3), axis=-1
        ) / np.sum(weight * delta**2, axis=-1)
        pred = zsr(np.clip(raw, 0, 1)[:, :, None], a, b)
        for i, d in enumerate(sizes):
            for j, t in enumerate(times):
                for k, h in enumerate(rh):
                    checks.append(
                        (
                            name,
                            d,
                            t,
                            h,
                            fit[k],
                            y[i, j, k],
                            sd[i, j, k],
                            raw[i, j],
                            pred[i, j, k],
                            pred[i, j, k] - y[i, j, k],
                        )
                    )
    pd.DataFrame(
        checks,
        columns=[
            "fit",
            "diameter_nm",
            "time_h",
            "rh_pct",
            "used_in_fit",
            "gf",
            "printed_spread",
            "unclipped_volume_fraction",
            "predicted_gf",
            "residual",
        ],
    ).to_csv(out / "humidity-check.csv", index=False)
    return checks


# %% Exact fixed-endmember inverse, independently solved for every diameter and exposure
@wp.kernel
def invert(
    end: wp.array3d(dtype=wp.float64),
    y: wp.array2d(dtype=wp.float64),
    err: wp.array2d(dtype=wp.float64),
    bound: wp.array3d(dtype=wp.float64),
):
    e, i = wp.tid()
    lower = wp.float64(0.0)
    upper = wp.float64(1.0)
    for h in range(6):
        a = end[e, 0, h]
        b = end[e, 1, h]
        fresh = a * a * a
        change = b * b * b - fresh
        lower_growth = y[i, h] - err[i, h]
        u = y[i, h] + err[i, h]
        x = (lower_growth * lower_growth * lower_growth - fresh) / change
        z = (u * u * u - fresh) / change
        lower = wp.max(lower, wp.min(x, z))
        upper = wp.min(upper, wp.max(x, z))
    bound[e, i, 0] = lower
    bound[e, i, 1] = upper


def numpy_bounds(end, y, err):
    a, b = end[:, 0, None, None, :], end[:, 1, None, None, :]
    left = ((y - err) ** 3 - a**3) / (b**3 - a**3)
    right = ((y + err) ** 3 - a**3) / (b**3 - a**3)
    return np.stack(
        [
            np.maximum(0, np.minimum(left, right).max(axis=-1)),
            np.minimum(1, np.maximum(left, right).min(axis=-1)),
        ],
        axis=-1,
    )


def outer_volume(y, err, center, width):
    # Union over all four rectangle vertices at each humidity, then intersect RH constraints.
    # Each endpoint derivative has constant sign inside the physical 0..1 mixing interval.
    lo = []
    hi = []
    for a in (center[0] - width[0], center[0] + width[0]):
        for b in (center[1] - width[1], center[1] + width[1]):
            assert np.all(np.sign(b - a) == np.sign(center[1] - center[0]))
            u = ((y - err) ** 3 - a**3) / (b**3 - a**3)
            v = ((y + err) ** 3 - a**3) / (b**3 - a**3)
            lo.append(np.minimum(u, v))
            hi.append(np.maximum(u, v))
    return np.stack(
        [
            np.maximum(0, np.min(lo, axis=0).max(axis=-1)),
            np.minimum(1, np.max(hi, axis=0).min(axis=-1)),
        ],
        axis=-1,
    )


def monotone(bounds):
    lower = np.maximum.accumulate(bounds[..., 0], axis=-1)
    upper = np.minimum.accumulate(bounds[..., 1][..., ::-1], axis=-1)[..., ::-1]
    return np.stack((lower, upper), axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage", required=True)
    args = p.parse_args()
    start = time.monotonic()
    out = PACKET / args.stage
    out.mkdir(exist_ok=True)
    sizes, times, rh, y, sd, c, w = inputs()
    central_checks(out)
    wp.init()
    device = wp.get_device(args.device)
    endpoints_grid = qmc.Sobol(12, scramble=True, seed=971).random_base2(args.power)
    # Add common shifts, including a common high/low product displacement, to the space-filling scan.
    shift = np.linspace(-1, 1, 5)
    aligned = [
        np.r_[np.full(6, i), np.full(4, j), np.full(2, k)]
        for i in shift
        for j in shift
        for k in shift
    ]
    offsets = np.vstack([np.zeros(12), aligned, 2 * endpoints_grid - 1]).reshape(
        -1, 2, 6
    )
    endpoints = c + offsets * w
    y_gpu = wp.array(y.reshape(-1, 6), dtype=wp.float64, device=device)
    records = []
    outer_records = []
    witness_rows = []
    maxdiff = 0.0
    members = []
    for multiple in [1, 2]:
        err = multiple * sd + 0.005
        outer = monotone(outer_volume(y, err, c, w))
        for i, d in enumerate(sizes):
            for j, t in enumerate(times):
                outer_records.append((multiple, d, t, *to_sodium(outer[i, j])))
        err_gpu = wp.array(err.reshape(-1, 6), dtype=wp.float64, device=device)
        counts = [0, 0]
        best = np.inf
        best_info = None
        extrema = np.full((len(times), 2), [np.inf, -np.inf])
        cum_ext = np.full((len(sizes), len(times), 2), [np.inf, -np.inf])
        i2, i3 = [list(sizes).index(d) for d in [200, 300]]
        for offset in range(0, len(endpoints), 1024):
            batch = endpoints[offset : offset + 1024]
            result = wp.empty(
                (len(batch), y.shape[0] * y.shape[1], 2),
                dtype=wp.float64,
                device=device,
            )
            wp.launch(
                invert,
                dim=result.shape[:2],
                inputs=[
                    wp.array(batch, dtype=wp.float64, device=device),
                    y_gpu,
                    err_gpu,
                    result,
                ],
                device=device,
            )
            raw = result.numpy().reshape(len(batch), len(sizes), len(times), 2)
            maxdiff = max(
                maxdiff,
                float(np.max(np.abs(raw[:1] - numpy_bounds(batch[:1], y, err)))),
            )
            bound = monotone(raw)
            ok = (bound[..., 0] <= bound[..., 1]).all(axis=2)
            both = ok[:, i2] & ok[:, i3]
            all_sizes = ok.all(axis=1)
            counts[0] += int(both.sum())
            counts[1] += int(all_sizes.sum())
            if both.any():
                f = to_sodium(bound[both])
                for side in range(2):
                    ratio = f[:, i3, :, side] / np.maximum(f[:, i2, :, 1 - side], 1e-15)
                    extrema[:, side] = (np.minimum if side == 0 else np.maximum)(
                        extrema[:, side],
                        (np.min if side == 0 else np.max)(ratio, axis=0),
                    )
                # Direct mass rate ratios can be unbounded if 200 nm plateaus. Save joint D3 and D2 ranges separately in outer output.
            if all_sizes.any():
                inds = np.flatnonzero(all_sizes)
                feasible = bound[inds]
                f = to_sodium(feasible)
                cum_ext[:, :, 0] = np.minimum(cum_ext[:, :, 0], f[..., 0].min(axis=0))
                cum_ext[:, :, 1] = np.maximum(cum_ext[:, :, 1], f[..., 1].max(axis=0))
                # Select a common physical witness nearest the measured central endmembers.
                distance = np.sum(offsets[offset + inds] ** 2, axis=(1, 2))
                ix = int(np.argmin(distance))
                if distance[ix] < best:
                    best = distance[ix]
                    idx = inds[ix]
                    best_info = (batch[idx], bound[idx], offset + idx)
        if best_info is not None:
            end, bound, idx = best_info
            members.append(
                dict(multiple=multiple, index=int(idx), endpoints=end.tolist())
            )
            v = bound.mean(axis=-1)
            pred = zsr(v[:, :, None], end[0], end[1])
            assert np.all(abs(pred - y) <= err + 1e-10)
            for i, d in enumerate(sizes):
                for j, t in enumerate(times):
                    witness_rows.append(
                        (
                            multiple,
                            d,
                            t,
                            v[i, j],
                            to_sodium(v[i, j]),
                            *to_sodium(bound[i, j]),
                            *cum_ext[i, j],
                        )
                    )
        for j, t in enumerate(times):
            records.append(
                (
                    multiple,
                    t,
                    *counts,
                    len(endpoints),
                    *(extrema[j] if counts[0] else [np.nan] * 2),
                )
            )
    pd.DataFrame(
        records,
        columns=[
            "scatter_multiple",
            "time_h",
            "pair_witnesses",
            "all_size_witnesses",
            "endpoint_sets",
            "sample_ratio_min",
            "sample_ratio_max",
        ],
    ).to_csv(out / "feasible.csv", index=False)
    pd.DataFrame(
        outer_records,
        columns=["scatter_multiple", "diameter_nm", "time_h", "sodium_lo", "sodium_hi"],
    ).to_csv(out / "outer.csv", index=False)
    pd.DataFrame(
        witness_rows,
        columns=[
            "scatter_multiple",
            "diameter_nm",
            "time_h",
            "volume_fraction",
            "sodium_fraction",
            "witness_lo",
            "witness_hi",
            "sample_lo",
            "sample_hi",
        ],
    ).to_csv(out / "witness.csv", index=False)
    assert maxdiff < 1e-11
    metadata = dict(
        power=args.power,
        endpoint_sets=len(endpoints),
        inverse_cases=len(endpoints) * y.shape[0] * y.shape[1] * 2,
        device=str(device),
        device_name=device.name,
        precision="float64",
        cpu_max_abs_error=maxdiff,
        seconds=time.monotonic() - start,
        warp=wp.__version__,
        witness_endpoints=members,
    )
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "witness_endpoints"}))
    print(
        pd.read_csv(out / "feasible.csv")
        .groupby("scatter_multiple")
        .first()
        .to_string()
    )


if __name__ == "__main__":
    main()
