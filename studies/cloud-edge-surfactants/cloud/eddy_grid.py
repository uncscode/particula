"""Follow cloud-base releases through a gridded, incompressible eddy field."""

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import campaign as gpu
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
LENGTH = 240.0
DEPTH = 160.0
WMEAN = 0.2
XOFFSET = LENGTH / 64
PHASE = np.pi / 5
INLETS = {
    "quadrature": 16,
    "quadrature32": 32,
    "quadrature64": 64,
    "quadrature128": 128,
}
STAGES = ("main", "grid", "numerics", *INLETS)


@dataclass(frozen=True)
class Setting:
    """Relative amplitude, environmental pattern and adsorption scenario."""

    amplitude: float
    environment: str
    adsorption: str = "central"


def field(grid, amplitude):
    """Differentiate the periodic tensor-product cubic streamfunction."""
    x = np.linspace(0, LENGTH, grid + 1)
    z = np.linspace(0, DEPTH, grid + 1)
    sine = np.sin(2 * np.pi * x / LENGTH)
    sine[-1] = sine[0]
    fx = CubicSpline(x, sine, bc_type="periodic")
    fz = CubicSpline(z, np.cos(2 * np.pi * z / DEPTH), bc_type="periodic")
    scale = amplitude * WMEAN * LENGTH / (2 * np.pi)

    def psi(x, z):
        return WMEAN * x + scale * fx(x % LENGTH) * fz(z % DEPTH)

    def vel(x, z):
        return (
            -scale * fx(x % LENGTH) * fz(z % DEPTH, 1),
            WMEAN + scale * fx(x % LENGTH, 1) * fz(z % DEPTH),
        )

    return psi, vel


@lru_cache(maxsize=256)
def path(grid, amplitude, x0):
    """Integrate actual x(t), z(t) until the parcel first reaches +100 m."""
    psi, vel = field(grid, amplitude)

    def at_top(time_s, state):
        return state[1] - 100

    at_top.terminal = True
    at_top.direction = 1
    for rtol, atol, step in ((1e-9, 2e-9, 8), (2e-11, 5e-11, 3)):
        sol = solve_ivp(
            lambda t, s: vel(*s),
            (0, 3500),
            [x0, 0.0],
            events=at_top,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=step,
            dense_output=True,
        )
        if not sol.success or len(sol.t_events[0]) != 1:
            raise RuntimeError(
                "A streamline did not reach the first +100 m crossing"
            )
        end = float(sol.t_events[0][0])
        mid = brentq(lambda t, position=sol.sol: position(t)[1] - 50, 0, end)
        xmid = float(sol.sol(mid)[0])
        coords = sol.sol(np.linspace(0, end, 101))
        drift = np.max(np.abs(psi(*coords) - psi(x0, 0)))
        if drift <= 1e-6 and abs(coords[1, -1] - 100) <= 1e-8:
            return end, xmid, float(vel(x0, 0)[1]), drift, sol.sol
    raise ValueError("Streamfunction or first-crossing check failed")


def mixing(environment, x50):
    """Return dry-air replacement fraction and RH at the interface."""
    if environment == "closed":
        return 0.0, 0.90
    if environment == "uniform":
        return 0.10, 0.90
    phase = PHASE + (np.pi if environment == "opposed" else 0.0)
    cell = np.cos(2 * np.pi * x50 / LENGTH + phase)
    return 0.10 + 0.06 * cell, 0.90 - 0.06 * cell


def settings(stage):
    """Choose paired flow, import and adsorption controls."""
    if stage == "main":
        return (
            [Setting(a, "closed") for a in (0, 0.4, 0.8)]
            + [Setting(0, "uniform"), Setting(0, "patterned")]
            + [Setting(0.8, k) for k in ("uniform", "patterned", "opposed")]
            + [Setting(0.8, "patterned", a) for a in ("low", "high")]
        )
    if stage in INLETS:
        return [Setting(0.8, "patterned"), Setting(0.8, "opposed")]
    return [Setting(0.8, "patterned")]


def get_ids(stage):
    """Place fine inlet nodes midway between the eight original releases."""
    if stage in ("main", "grid", "numerics"):
        return np.arange(8) / 8
    if stage in INLETS:
        half = INLETS[stage] // 2
        return (np.arange(half) + 0.5) / half
    return np.array([0.0])


def make_jobs(stage):
    """Set up independent air histories and location-specific mixing cases."""
    grid, nodes, bins, dt = (32, 1025, 48, 0.1)
    if stage == "grid":
        grid = 64
    if stage == "numerics":
        grid, nodes, bins, dt = 64, 2049, 64, 0.05
    jobs = []
    for setting in settings(stage):
        phases = get_ids(stage)
        if setting.amplitude == 0 and setting.environment in (
            "closed",
            "uniform",
        ):
            phases = np.array([0.0])  # exact homogeneous inlet solution
        for phase in phases:
            x0 = float(phase * LENGTH + XOFFSET)
            end, xmid, wbase, drift, position = path(
                grid, setting.amplitude, x0
            )
            entr, rh = mixing(setting.environment, xmid)
            case = gpu.Case(
                regime="smoke",
                w=WMEAN,
                entr=entr,
                gas=1.0,
                scenario=setting.adsorption,
                rh_ext=rh,
            )
            jobs.append(
                dict(
                    stage=stage,
                    amp=setting.amplitude,
                    env=setting.environment,
                    ads=setting.adsorption,
                    phase=phase,
                    grid=grid,
                    bins=bins,
                    nodes=nodes,
                    dt_limit=dt,
                    time_s=end,
                    x0=x0,
                    x50=xmid,
                    wbase=wbase,
                    drift=drift,
                    position=position,
                    case=case,
                )
            )
    return jobs


def collect(j, case, result, parts, history, steps, outputs, modes, paths):
    """Retain the fine-bracket counts and synchronize air and spatial paths."""
    fields = {
        key: j[key]
        for key in (
            "stage",
            "amp",
            "env",
            "ads",
            "phase",
            "grid",
            "bins",
            "nodes",
            "dt",
            "time_s",
            "x0",
            "x50",
            "wbase",
            "drift",
        )
    }
    fields.update(entr=case.entr, rh_ext=case.rh_ext, steps=steps)
    width = sum(
        r["uncertainty_width_cm3"]
        for r in parts
        if r["mode"] in ("octyl", "dodecyl")
    )
    fields["contrast_bracket_width_cm3"] = width
    outputs.append({**fields, **result})
    modes.extend([{**fields, **r} for r in parts])
    coords = j["position"](np.linspace(0, j["time_s"], 101))
    for h, (x, z) in zip(history, coords.T[1:], strict=True):
        if abs(h[0] - z) > 1e-7:
            raise ValueError("CUDA height differs from spatial trajectory")
        paths.append(
            dict(
                stage=j["stage"],
                amp=j["amp"],
                env=j["env"],
                ads=j["ads"],
                phase=j["phase"],
                x_m=x,
                z_m=z,
                s_pct=100 * h[1],
                water_kgkg=h[3],
            )
        )


def step_count(jobs):
    """Give a release group the original common time-grid length."""
    return 100 * int(
        np.ceil(max(j["time_s"] / j["dt_limit"] for j in jobs) / 100)
    )


def discretize_wind(jobs):
    """Sample vertical velocities on each physical transit interval."""
    steps = step_count(jobs)
    for j in jobs:
        dt = j["time_s"] / steps
        coords = j["position"](np.linspace(0, j["time_s"], steps + 1))
        j["dt"] = dt
        j["velocity"] = np.ascontiguousarray(np.diff(coords[1]) / dt)
        if (
            np.min(j["velocity"]) <= 0
            or abs(dt * j["velocity"].sum() - 100) > 1e-8
        ):
            raise ValueError(
                "Incorrect vertical transport in CUDA forcing history"
            )
    return steps


def replay(jobs, outdir, timer):
    """Advance CUDA batches on trajectory-averaged vertical velocities."""
    func = gpu.build_cuda()
    steps = discretize_wind(jobs)
    brackets = [{} for _ in jobs]
    outputs = []
    paths = []
    modes = []
    nrounds = 3
    for iteration in range(nrounds):
        buckets = defaultdict(list)
        for k, j in enumerate(jobs):
            signature = tuple(
                (m.label, len(brackets[k].get(m.label, [])))
                for m in gpu.modes(j["case"])
            )
            buckets[signature].append(k)
        batch_size = 24 if jobs[0]["stage"] in INLETS else 6
        for keys in buckets.values():
            for start in range(0, len(keys), batch_size):
                kk = keys[start : start + batch_size]
                batch = []
                for k in kk:
                    j = jobs[k]
                    ms, groups, params, _ = gpu.prepare(
                        j["case"], j["bins"], j["nodes"], j["dt"], brackets[k]
                    )
                    params[0] = j["dt"]
                    batch.append((ms, groups, params, j["velocity"]))
                launched = time.monotonic()
                results, particles, hist = gpu.call_gpu(
                    func, batch, jobs[0]["nodes"], steps
                )
                timer[0] += time.monotonic() - launched
                (outdir / "eddy_gpu_seconds.txt").write_text(
                    str(timer[0]) + "\n"
                )
                for i, k in enumerate(kk):
                    j = jobs[k]
                    case = j["case"]
                    result, parts, brackets[k] = gpu.unpack(
                        case, batch[i], results[i], particles[i], j["bins"]
                    )
                    if iteration != nrounds - 1:
                        continue
                    collect(
                        j,
                        case,
                        result,
                        parts,
                        hist[i],
                        steps,
                        outputs,
                        modes,
                        paths,
                    )
        print(
            jobs[0]["stage"],
            "pass",
            iteration + 1,
            "GPU seconds",
            round(timer[0], 1),
            flush=True,
        )
    stage = jobs[0]["stage"]
    gpu.write(outdir / f"eddy_{stage}.csv", outputs)
    gpu.write(outdir / f"eddy_{stage}_pop.csv", modes)
    with gzip.open(
        outdir / f"eddy_{stage}_paths.csv.gz", "wt", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=paths[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(paths)


def tag_totals(group, tag_map, keys):
    """Add target retention and incoming smoke to each saved air trajectory."""
    for r in group:
        tagged = tag_map[tuple(r[k] for k in keys)]
        targets = [p for p in tagged if p["mode"] in ("octyl", "dodecyl")]
        r["total_size_bracket_width_cm3"] = sum(
            float(p["uncertainty_width_cm3"]) for p in tagged
        )
        r["available_target"] = sum(float(p["available_cm3"]) for p in targets)
        r["activated_target"] = sum(float(p["activated_cm3"]) for p in targets)
        r["incoming_bg"] = sum(
            float(p["activated_cm3"])
            for p in tagged
            if p["mode"].startswith("incoming")
        )


def matching_rows(stage, load):
    """Join each inlet refinement to its lower-density release positions."""
    rows = load(stage)
    tag_map = defaultdict(list)
    stages = (stage,)
    if stage in INLETS:
        stages = (
            stage,
            "main",
            *(s for s in INLETS if INLETS[s] < INLETS[stage]),
        )
    keys = ("stage", "amp", "env", "ads", "phase")
    for each in stages:
        for pop in load(each + "_pop"):
            tag_map[tuple(pop[k] for k in keys)].append(pop)
    if stage in INLETS:
        rows += [
            r
            for additional in stages[1:]
            for r in load(additional)
            if r["amp"] == "0.8"
            and r["env"] in ("patterned", "opposed")
            and r["ads"] == "central"
        ]
    return rows, tag_map, keys


def summarize(outdir):
    """Aggregate spatial phases by inlet area and by upward flux."""

    def load(stage):
        with (outdir / f"eddy_{stage}.csv").open() as stream:
            return list(csv.DictReader(stream))

    records = []
    for stage in STAGES:
        if not (outdir / f"eddy_{stage}.csv").exists():
            continue
        if stage in INLETS:
            required = [
                "main",
                *(s for s in INLETS if INLETS[s] < INLETS[stage]),
            ]
            if any(not (outdir / f"eddy_{s}.csv").exists() for s in required):
                print(
                    f"Saved {stage} endpoints. Run earlier inlet stages "
                    "for its nested flux mean."
                )
                continue
        rows, tag_map, keys = matching_rows(stage, load)
        for case in sorted({(r["amp"], r["env"], r["ads"]) for r in rows}):
            group = [r for r in rows if (r["amp"], r["env"], r["ads"]) == case]
            group.sort(key=lambda r: float(r["phase"]))
            tag_totals(group, tag_map, keys)

            def col(key, rows=group):
                return np.array([float(r[key]) for r in rows])

            w_top = []
            for r in group:
                grid, amp, x0 = int(r["grid"]), float(r["amp"]), float(r["x0"])
                end, _, _, _, position = path(grid, amp, x0)
                _, velocity = field(grid, amp)
                w_top.append(velocity(*position(end))[1])
            weights = {
                "inlet_area": np.ones(len(group)),
                "inlet_flux": col("wbase"),
                "outlet_area": col("wbase") / np.asarray(w_top),
            }
            for weighting, w in weights.items():
                w = w / w.sum()
                delta = col("dodecyl_minus_octyl_cm3")
                half = col(
                    "contrast_bracket_width_cm3"
                )  # conservative full widths
                records.append(
                    dict(
                        stage=stage,
                        amp=case[0],
                        env=case[1],
                        ads=case[2],
                        releases=len(group),
                        weighting=weighting,
                        mean_time_s=w @ col("time_s"),
                        min_time_s=min(col("time_s")),
                        max_time_s=max(col("time_s")),
                        mean_peak_s_pct=w @ col("peak_pct"),
                        mean_total_cm3=w @ col("activated_cm3"),
                        mean_total_size_bracket_width_cm3=w
                        @ col("total_size_bracket_width_cm3"),
                        mean_entrained_fraction=w @ col("entr"),
                        mean_environment_rh=w @ col("rh_ext"),
                        mean_available_target_cm3=w @ col("available_target"),
                        mean_activated_target_cm3=w @ col("activated_target"),
                        target_activated_fraction=(w @ col("activated_target"))
                        / (w @ col("available_target")),
                        mean_activated_incoming_bg_cm3=w @ col("incoming_bg"),
                        mean_difference_cm3=w @ delta,
                        min_difference_cm3=min(delta),
                        max_difference_cm3=max(delta),
                        max_contrast_bracket_width_cm3=max(half),
                        positive_resolved=int(np.count_nonzero(delta > half)),
                        negative_resolved=int(np.count_nonzero(delta < -half)),
                        unresolved=int(np.count_nonzero(abs(delta) <= half)),
                    )
                )
    gpu.write(outdir / "eddy_summary.csv", records)


def main():
    """Run matched release designs and report CUDA wall time."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=HERE / "results")
    cli.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=STAGES,
    )
    cli.add_argument("--used-gpu-seconds", type=float, default=0.0)
    args = cli.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    timer = [args.used_gpu_seconds]
    begin = time.monotonic()
    for stage in args.stages:
        replay(make_jobs(stage), args.out, timer)
        (args.out / "eddy_gpu_seconds.txt").write_text(str(timer[0]) + "\n")
    summarize(args.out)
    (args.out / "eddy_run.json").write_text(
        json.dumps(
            dict(
                device=gpu.subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
                cuda_seconds_charged=timer[0],
                most_recent_invocation_wall_seconds=time.monotonic() - begin,
                prior_cuda_seconds_rounded_up=args.used_gpu_seconds,
                kernel_sha256=hashlib.sha256(
                    (gpu.HERE / "cloud.cu").read_bytes()
                ).hexdigest(),
                organic_header_sha256=hashlib.sha256(
                    (gpu.HERE / "organic.cuh").read_bytes()
                ).hexdigest(),
                inlet_offset_m=XOFFSET,
                rounds_per_case=3,
                mean_ascent_ms=WMEAN,
                length_m=LENGTH,
                depth_m=DEPTH,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
