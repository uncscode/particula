# %% Four matched optical measurements; run from the pinned Particula environment
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import warp as wp
from instruments import (
    COLORS,
    MARKERS,
    WATER,
    BEADS,
    water_index,
    profiles,
    geometry,
    pulse_lookup,
    fssp,
    sobol,
)
from pixels import pixel_area

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from study import edges, window


def device_areas(profile, d, p, draws, ix, dr, threshold, quad):
    dev = "cuda:0"
    assert (2 * (np.ceil((d.max() / 2 + 6) / p.min()) + 1) + 1) < 64
    args = [
        wp.array(np.ascontiguousarray(data), dtype=typ, device=dev)
        for data, typ in [
            (profile, wp.float32),
            (d, wp.float32),
            (p, wp.float32),
            (draws[:, :2], wp.float32),
            (ix, wp.int32),
        ]
    ]
    sizes = wp.empty((len(d), len(draws)), dtype=wp.float32, device=dev)
    wp.launch(
        pixel_area,
        dim=sizes.shape,
        inputs=[*args, dr, threshold, quad, sizes],
        device=dev,
    )
    return sizes.numpy().astype(float)


def bead_fit(raw):
    x = np.array(
        [np.median(row[row > 0]) if np.any(row > 0) else np.nan for row in raw]
    )
    # A multiplicative linear factor below the published breakpoint, anchored
    # at 1 at the breakpoint. Only detectable small beads inform its slope.
    use = (x < 13.07) & (MARKERS < 13.07) & np.isfinite(x)
    h = x[use] * (13.07 - x[use])
    a = np.dot(h, MARKERS[use] - x[use]) / np.dot(h, h) if np.any(use) else 0.0
    return x, a


def corrected(raw, a):
    return raw * (1 + a * np.maximum(13.07 - raw, 0))


def response(raw, probe, d, triggered):
    e = edges(probe, "upper" if probe == "F" else "center")
    ids = np.searchsorted(e, raw, side="right") - 1
    valid = (ids >= 0) & (ids < len(e) - 1)
    row = {
        "true_um": d,
        "trigger": triggered.mean(axis=1),
        "detect": valid.mean(axis=1),
    }
    for high, key in [(13.0, "band13"), (13.07, "band1307")]:
        membership = window(e, 6.0, high)
        row[key] = (valid * membership[ids.clip(0, len(e) - 2)]).mean(axis=1)
    row["unbinned13"] = ((raw >= 6) & (raw < 13)).mean(axis=1)
    row["unbinned1307"] = ((raw >= 6) & (raw < 13.07)).mean(axis=1)
    row["below13"] = ((raw > 0) & valid & (raw < 13)).mean(axis=1)
    row["nativebelow13"] = (
        valid * window(e, e[0], 13.0)[ids.clip(0, len(e) - 2)]
    ).mean(axis=1)
    row["mean_reported_um"] = np.divide(
        np.where(valid, raw, 0).sum(axis=1),
        valid.sum(axis=1),
        out=np.full(len(d), np.nan),
        where=valid.any(axis=1),
    )
    return pd.DataFrame(row)


def run(stage, out):
    settings = {
        "pilot": (0.5, 4, 32, 24, 2, 0.1),
        "main": (0.125, 12, 256, 48, 2, 0.05),
        "refined": (0.0625, 20, 512, 80, 3, 0.04),
        "convergence": (0.03125, 28, 1024, 96, 4, 0.04),
    }
    step, nz, trials, order, quad, dr = settings[stage]
    grid = np.arange(1.0 + step / 2, 40.0, step)
    d = np.unique(np.r_[grid, MARKERS])
    seed = 118 if stage == "convergence" else 816
    draws = sobol(trials, seed)
    z, p, ix = geometry(nz, draws)
    r = np.arange(0, 52, dr)
    cache = {nm: pulse_lookup(nm, order) for nm in COLORS}
    tables = []
    cal = []
    start = time.monotonic()
    ntargets = 0
    scenarios = [
        ("nominal", "mie", 0.40, 0.12),
        ("low_threshold", "mie", 0.28, 0.06),
        ("high_threshold", "mie", 0.52, 0.20),
        ("shadow", "shadow", 0.40, 0.12),
        ("dispersion", "dispersion", 0.40, 0.12),
    ]
    if stage == "pilot":
        scenarios = [scenarios[0], scenarios[-2]]
    if stage == "convergence":
        scenarios = scenarios[:1]
    for kind in sorted({s[1] for s in scenarios}):
        raw = {}
        beads = {}
        optical_cache = (
            cache
            if kind != "dispersion"
            else {nm: pulse_lookup(nm, order, water_index(nm)) for nm in COLORS}
        )
        for nm in COLORS:
            field = profiles(
                d,
                nm,
                water_index(nm) if kind == "dispersion" else WATER,
                z,
                r,
                order,
                kind,
            )
            beadfield = profiles(MARKERS, nm, BEADS, z, r, order, kind)
            for label, k, th, spread in scenarios:
                if k != kind:
                    continue
                for count, pr, diam, dic in [
                    (False, field, d, raw),
                    (True, beadfield, MARKERS, beads),
                ]:
                    dic[nm, label] = device_areas(
                        pr, diam, p, draws, ix, dr, th, quad
                    )
                    ntargets += len(diam) * trials
        for label, k, th, spread in scenarios:
            if k != kind:
                continue
            for nm in COLORS:
                bead_sizes, a = bead_fit(beads[nm, label])
                for true, median, row in zip(
                    MARKERS, bead_sizes, beads[nm, label]
                ):
                    cal.append(
                        dict(
                            case=label,
                            nm=nm,
                            true_um=true,
                            raw_detect=(row > 0).mean(),
                            raw_median=median,
                            corrected_median=corrected(median, a),
                            factor_slope_per_um=a,
                        )
                    )
                for tuning in ["frozen", "retuned", "no_holo_correction"]:
                    fc = 633 if tuning != "retuned" else nm
                    hc = 660 if tuning != "retuned" else nm
                    a = (
                        0.0
                        if tuning == "no_holo_correction"
                        else bead_fit(beads[hc, label])[1]
                    )
                    for probe, measured in [
                        (
                            "F",
                            fssp(
                                d, nm, order, draws, fc, spread, optical_cache
                            ),
                        ),
                        ("I", corrected(raw[nm, label], a)),
                    ]:
                        triggered = (
                            measured > 0 if probe == "F" else raw[nm, label] > 0
                        )
                        result = response(measured, probe, d, triggered)
                        result = result[np.isin(result.true_um, grid)].copy()
                        result = result.assign(
                            case=label, calibration=tuning, probe=probe, nm=nm
                        )
                        tables.append(result)
    frame = pd.concat(tables, ignore_index=True)
    assert (
        (frame.detect >= frame.band1307 - 1e-12)
        & (frame.detect <= 1)
        & (frame.band13 >= 0)
    ).all()
    out.mkdir(exist_ok=True, parents=True)
    frame.to_csv(
        out / f"{stage}-curves.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(cal).to_csv(out / f"{stage}-beads.csv", index=False)
    (out / f"{stage}-run.json").write_text(
        json.dumps(
            dict(
                seconds=time.monotonic() - start,
                device=wp.get_device("cuda:0").name,
                depth_planes=nz,
                trials_per_diameter=trials,
                seed=seed,
                diameter_step_um=step,
                quadrature=quad,
                radial_step_um=dr,
                angular_nodes=order,
                cuda_target_images=ntargets,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["pilot", "main", "refined", "convergence"]
    )
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.stage, args.out)
