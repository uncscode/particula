"""Compare GPU forecasts to independent physical references."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t
from stage3_charge_modes import canonical_charge
from stage3_evidence import obs, write


class Moments:
    """Accumulate means and variances for whole simulated trajectories."""

    def __init__(self, x):
        """Allocate the time-by-field accumulators."""
        self.n = 0
        self.mean = np.zeros_like(x)
        self.squares = np.zeros_like(x)

    def add(self, x):
        """Include one independently seeded trajectory."""
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.squares += d * (x - self.mean)

    def se(self):
        """Return the standard error of the trajectory mean."""
        if self.n < 2:
            return np.full_like(self.mean, np.inf)
        return np.sqrt(np.maximum(0, self.squares) / (self.n * (self.n - 1)))


def summarize(pred, ref, refse, refn, floor, horizon, deterministic=False):
    """Compare mean forecasts including sampling and refinement errors."""
    df = min(10**9 if deterministic else pred.n - 1, refn - 1)
    crit = t.ppf(1 - 0.05 / (2 * pred.mean.size * 2), df) if df > 0 else np.inf
    noise = np.hypot(np.zeros_like(ref) if deterministic else pred.se(), refse)
    delta = abs(pred.mean - ref)
    gate = np.full_like(ref, 1e-11)
    gate[:, 0] = 1.0
    resolved = abs(ref) > gate
    denom = np.maximum(abs(ref), gate)
    upper = np.full_like(ref, np.inf)
    if np.isfinite(crit):
        lower = np.where(resolved, abs(ref) - crit * refse - floor, gate)
        np.divide(
            delta + crit * noise + floor, lower, out=upper, where=lower > 0
        )
    raw = delta / denom
    floorpct = 100 * floor / denom
    nr, mr = 100 * raw[:, 0].max(), 100 * raw[:, 1:].max()
    nu, mu = 100 * upper[:, 0].max(), 100 * upper[:, 1:].max()
    s = (ref.shape[1] - 4) // 2
    labels = (
        [f"gas_{i}" for i in range(s)]
        + ["particle_dry"]
        + [f"particle_{i}" for i in range(s)]
        + ["particle_total", "all_mass"]
    )
    phase_bounds = upper[:, 1:]
    tied = phase_bounds == phase_bounds.max()
    # With a single stochastic realization every bound is infinite. Report
    # the largest observed discrepancy among tied bounds, including that case.
    limiting = np.where(tied, raw[:, 1:], -np.inf)
    time_id, j = np.unravel_index(np.argmax(limiting), limiting.shape)
    return dict(
        limiting_mass_field=labels[j],
        limiting_mass_time_s=horizon * time_id / (len(ref) - 1),
        limiting_reference_mass_kg_m3=ref[time_id, j + 1],
        limiting_mass_absolute_error_kg_m3=delta[time_id, j + 1],
        gpu_seeds=pred.n,
        number_mean_pct=nr,
        mass_mean_pct=mr,
        number_upper_pct=nu,
        mass_upper_pct=mu,
        max_convergence_change_number_pct=floorpct[:, 0].max(),
        max_convergence_change_mass_pct=floorpct[:, 1:].max(),
        decision="passes tested bounds"
        if nu <= 1 and mu < 0.1
        else "exceeds target in mean"
        if nr > 1 or mr >= 0.1
        else "unresolved",
    )


def group(r):
    """Identify a physical box independently of numerical method."""
    return (
        r["n"],
        r["species"],
        r["density"],
        canonical_charge(r["charge"]),
        r["family"],
        r["horizon"],
    )


def _native_trials(trials, native, result):
    for trial in trials:
        if trial.get("status", "completed") != "completed":
            continue
        key = group(trial) + (
            "native-pilot",
            "native-warp",
            trial["steps"],
            trial["vapor_micro"],
            "native",
            0.0,
            0.0,
        )
        for ii, hist in enumerate(trial["history"]):
            uid = (key, trial["first_seed"] + ii)
            if uid in native:
                continue
            native.add(uid)
            v = obs(dict(history=hist))
            if key not in result:
                result[key] = Moments(v)
            result[key].add(v)


def collect(inputs):
    """Read finished GPU and native runs into matching case groups."""
    result = {}
    native = set()
    for p in inputs:
        for f in p.glob("*.json"):
            if f.name in ("device.json", "closed.json"):
                continue
            a = json.loads(f.read_text())
            if isinstance(a, list):
                _native_trials(a, native, result)
            elif a.get("status") == "completed":
                key = group(a) + (
                    a["panel"],
                    a["kernel"],
                    a["steps"],
                    a["vmax"],
                    a.get("vapor_method", "first-order"),
                    a.get("collision_budget", 0.0),
                    a.get("growth_budget", 0.0),
                )
                v = obs(a)
                if key not in result:
                    result[key] = Moments(v)
                result[key].add(v)
    return result


def read_sections(folder):
    """Collect independent volume-grid solutions at matching resolutions."""
    sec = defaultdict(dict)
    for f in folder.glob("*.json"):
        r = json.loads(f.read_text())
        if r["status"] == "completed" and r["kernel"] == "physical":
            sec[
                (
                    r["n"],
                    r["species"],
                    r["density"],
                    "neutral",
                    r["family"],
                    r["horizon"],
                )
            ][r["bins"]] = obs(r)
    return sec


def read_charged(folder, prefix="sectional", observe=obs):
    """Require a size and a charge-domain refinement at the same fine point."""
    grids = defaultdict(dict)
    tolerances = {}
    for path in sorted(folder.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("truncation_ok") and row["status"] == "completed":
            case = group(row)
            point = row["bins"], row["charge_extent"]
            key = (case, point)
            # The initial fixed-section panel hard-codes rtol=2e-8.
            tol = row["tol"] if "tol" in row else 2e-8
            if tol < tolerances.get(key, np.inf):
                grids[case][point] = observe(row)
                tolerances[key] = tol
    out = {}
    for case, grid in grids.items():
        for b, q in sorted(grid, reverse=True):
            widths = [i for i, j in grid if i < b and j == q]
            if case[3] == "neutral" and widths:
                small = max(widths)
                out[case] = (
                    f"{prefix}-{b}-{small}",
                    grid[b, q],
                    abs(grid[b, q] - grid[small, q]),
                )
                break
            extents = [j for i, j in grid if i == b and j < q]
            if widths and extents:
                small, narrow = max(widths), max(extents)
                fine = grid[b, q]
                change = abs(fine - grid[small, q]) + abs(
                    fine - grid[b, narrow]
                )
                out[case] = (
                    f"{prefix}-{b}-{small}-charge-{q}-{narrow}",
                    fine,
                    change,
                )
                break
    return out


def read_uptake(folder):
    """Convergence of the collision-free ODE for the actual particle masses."""
    grids = defaultdict(dict)
    for file in folder.glob("*.json"):
        r = json.loads(file.read_text())
        key = (r["n"], r["species"], r["density"], r["family"], r["horizon"])
        q = r["groups"]
        old = grids[key].get(q)
        if old is None or r["tol"] < old[0]:
            grids[key][q] = (r["tol"], obs(r))
    output = {}
    for key, rows in grids.items():
        points = sorted(rows)
        if len(points) < 2:
            continue
        q, small = points[-1], points[-2]
        fine = rows[q][1]
        output[key] = (
            f"exchange-only-{q}-{small}",
            fine,
            abs(fine - rows[small][1]),
        )
    return output


def read_direct(folder):
    """Use one tightest-tolerance result per seed and method."""
    files = defaultdict(dict)
    for f in folder.glob("*.json"):
        r = json.loads(f.read_text())
        if r["kernel"] != "physical":
            continue
        slot = files[group(r)].setdefault(r["seed"], {})
        method = r["method"]
        if method not in slot or r["tol"] < slot[method][0]:
            slot[method] = (r["tol"], f)
    direct, diffs = {}, {}
    for case, runs in files.items():
        for methods in runs.values():
            if "DOP853" not in methods:
                continue
            v = obs(json.loads(methods["DOP853"][1].read_text()))
            if case not in direct:
                direct[case] = Moments(v)
            direct[case].add(v)
            if "Radau" in methods:
                check = obs(json.loads(methods["Radau"][1].read_text()))
                diffs[case] = np.maximum(diffs.get(case, 0), abs(check - v))
    return direct, diffs


def read_strata(folders):
    """Pool means and variances across disjoint conditional samples."""
    groups = defaultdict(dict)
    for folder in folders:
        for path in folder.glob("*.npz"):
            with np.load(path) as values:
                n, s, d, f, h = values["case"].tolist()
                case = (
                    int(n),
                    int(s),
                    d,
                    canonical_charge(str(values["charge"])),
                    int(f),
                    h,
                )
                seed, count = int(values["start"]), int(values["count"])
                tol = float(values["tol"])
                key = (seed, count)
                versions = groups[case].setdefault(key, {})
                if tol in versions:
                    raise ValueError("Duplicate stratified sample window")
                versions[tol] = (
                    values["noevent"] + values["mean"],
                    values["m2"],
                )
    result = {}
    for case, blocks in groups.items():
        last = -1
        total = None
        refinements = []
        for (seed, count), versions in sorted(blocks.items()):
            if seed < last:
                raise ValueError("Overlapping conditional sample seeds")
            last = seed + count
            fine, m2 = versions[min(versions)]
            if len(versions) > 1:
                refinements.append(abs(fine - versions[max(versions)][0]))
            if total is None:
                total = Moments(fine)
            n = total.n
            delta = fine - total.mean
            total.mean += delta * count / (count + n)
            total.squares += m2 + delta * delta * n * count / (n + count)
            total.n += count
        check = (
            np.maximum.reduce(refinements)
            if refinements
            else np.zeros_like(total.mean)
        )
        result[case] = (total, check, len(refinements))
    return result


def run(a):
    """Compare stochastic estimates against each independent reference."""
    gpu = collect(a.gpu)
    sec = read_sections(a.section)
    charged = read_charged(a.charged) if a.charged else {}
    moving = read_charged(a.moving, "moving") if a.moving else {}
    uptake = read_uptake(a.uptake) if a.uptake else {}
    direct, diffs = read_direct(a.direct)
    strata = read_strata(a.strata or [])
    out = []
    for k, v in sorted(gpu.items()):
        case = k[:6]
        types = []
        if case in sec and len(sec[case]) >= 2:
            b = sorted(sec[case])
            hi, lo = sec[case][b[-1]], sec[case][b[-2]]
            types.append(
                (
                    f"sectional-{b[-1]}-{b[-2]}",
                    hi,
                    np.zeros_like(hi),
                    10**9,
                    abs(hi - lo),
                    0,
                )
            )
        if case in charged:
            label, hi, change = charged[case]
            types.append((label, hi, np.zeros_like(hi), 10**9, change, 0))
        if case in moving:
            label, hi, change = moving[case]
            types.append((label, hi, np.zeros_like(hi), 10**9, change, 0))
        exchange_case = case[:3] + case[4:]
        if k[7] == "off" and exchange_case in uptake:
            label, hi, change = uptake[exchange_case]
            types.append((label, hi, np.zeros_like(hi), 10**9, change, 0))
        if case in direct:
            d = direct[case]
            types.append(
                (
                    "finite-continuous",
                    d.mean,
                    d.se(),
                    d.n,
                    diffs.get(case, np.zeros_like(d.mean)),
                    int(case in diffs),
                )
            )
        if case in strata:
            moment, change, checked = strata[case]
            types.append(
                (
                    "finite-conditional",
                    moment.mean,
                    moment.se(),
                    moment.n,
                    change,
                    0,
                )
            )
        for name, ref, se, nn, floor, has_check in types:
            names = [
                "n",
                "species",
                "density",
                "charge",
                "family",
                "horizon_s",
                "panel",
                "kernel",
                "steps",
                "micro_or_vmax",
                "vapor_method",
                "collision_budget",
                "growth_budget",
            ]
            out.append(
                dict(zip(names, k, strict=True))
                | dict(
                    reference=name,
                    reference_seeds=0 if nn == 10**9 else nn,
                    independent_integrator_check=has_check,
                )
                | summarize(
                    v, ref, se, nn, floor, case[-1], deterministic=k[7] == "off"
                )
            )
    write(a.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("section", type=Path)
    p.add_argument("direct", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--charged", type=Path)
    p.add_argument("--moving", type=Path)
    p.add_argument("--strata", nargs="+", type=Path)
    p.add_argument("--uptake", type=Path)
    p.add_argument("--gpu", type=Path, nargs="+", required=True)
    run(p.parse_args())
