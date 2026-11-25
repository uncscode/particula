"""Closed warm-cloud parcel experiments for the two citric-acid recipes."""

import argparse
import csv
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from particula.dynamics import parcel
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent / "results"
PRESSURE = 90000.0
TEMP = 285.0
NUMBER = 300.0  # per cm3 at cloud base
MEDIAN = 100.0  # dry diameter, nm
GSD = 1.5
LIMITS = (40.0, 400.0)  # truncate and normalize the dry spectrum here
HEIGHT = 100.0
KAPPA_CONTROL = 0.16
REFERENCE = ("control", "native", "central")


def cdf(diameter):
    """Return the cumulative lognormal number at dry diameter in nm."""
    return ndtr(np.log(diameter / MEDIAN) / np.log(GSD))


def fraction(left, right):
    """Integrate normalized number between dry diameters in nm."""
    return (cdf(right) - cdf(left)) / (cdf(LIMITS[1]) - cdf(LIMITS[0]))


def equilibrium(dry, wet, temp, case):
    """Return the equilibrium ratio at the supplied radii (m) and T."""
    if callable(case):
        return case(dry, wet, temp)
    if case == REFERENCE:
        return parcel._equilibrium(wet, dry, temp, KAPPA_CONTROL)
    from equilibrium import equilibrium_ratio, state

    if case[1] == "fixed70":
        recipe, _, scenario = case
        activity = state(dry, wet, recipe, "finite", scenario)[0]
        tension_n_m = {"acid": 0.0644, "octyl": 0.0418, "dodecyl": 0.0365}[
            recipe
        ]
        return activity * np.exp(
            2 * tension_n_m / (parcel.RV * parcel.RHO_W * temp * wet)
        )

    return equilibrium_ratio(dry, wet, temp, *case)


class Curves:
    """Tabulate each equilibrium curve over wet radius and temperature."""

    def __init__(self, diameters, case, nodes):
        """Tabulate the ratio at two temperatures and log-spaced sizes."""
        self.dry = np.asarray(diameters) * 0.5e-9
        self.x = np.linspace(np.log(1.00001), np.log(1500.0), nodes)
        self.step = self.x[1] - self.x[0]
        self.classes = np.arange(len(self.dry))
        wet = self.dry[:, None] * np.exp(self.x)
        # All cloud runs stay within this 2 K interval.
        self.warm = equilibrium(self.dry[:, None], wet, TEMP, case)
        cold = equilibrium(self.dry[:, None], wet, TEMP - 2, case)
        self.dt = (self.warm - cold) * 0.5

    def interpolate(self, wet, temperature):
        """Interpolate the ratio at wet radii and temperature."""
        u = (np.log(wet / self.dry) - self.x[0]) / self.step
        j = np.clip(u.astype(int), 0, len(self.x) - 2)
        a = u - j
        ii = self.classes
        t = temperature - TEMP
        left = self.warm[ii, j] + t * self.dt[ii, j]
        right = self.warm[ii, j + 1] + t * self.dt[ii, j + 1]
        return left + a * (right - left)

    def critical(self, temp):
        """Find the wet radii (m) at the global maximum equilibrium ratios."""
        y = self.warm + (temp - TEMP) * self.dt
        # Fit the three samples around the largest equilibrium ratio.
        j = np.argmax(y, axis=1)
        if np.any((j == 0) | (j == len(self.x) - 1)):
            raise ValueError("Critical radius outside sampled curve")
        i = self.classes
        left, mid, right = y[i, j - 1], y[i, j], y[i, j + 1]
        dx = 0.5 * (left - right) / (left + right - 2 * mid)
        return self.dry * np.exp(self.x[j] + self.step * dx)

    def initial(self, rh):
        """Find the smallest equilibrium root at the starting humidity."""
        above = self.warm >= rh
        j = np.argmax(above, axis=1)
        if np.any(j == 0):
            raise ValueError("Missing stable equilibrium root at cloud base")
        i = self.classes
        a = (rh - self.warm[i, j - 1]) / (self.warm[i, j] - self.warm[i, j - 1])
        return self.dry * np.exp(self.x[j - 1] + a * self.step)


@lru_cache(maxsize=60)
def cached_curves(diameters, case, nodes):
    """Reuse tables for central runs with identical sizes and chemistry."""
    return Curves(diameters, case, nodes)


def forcing(mean, amplitude=0.0, tau=20.0, seed=0):
    """Construct positive correlated ascent at set mean speed and height."""
    end = HEIGHT / mean
    times = np.linspace(0, end, round(end) + 1)
    if amplitude == 0:
        return times, np.full_like(times, mean), times * mean
    rng = np.random.default_rng(seed)
    trace = np.empty_like(times)
    trace[0] = rng.normal()
    phi = np.exp(-(times[1] - times[0]) / tau)
    for i in range(1, len(times)):
        trace[i] = phi * trace[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    trace -= np.trapezoid(trace, times) / end
    trace /= np.max(np.abs(trace))
    velocity = mean * (1 + amplitude * trace)
    z = cumulative_trapezoid(velocity, times, initial=0.0)
    if np.any(velocity <= 0) or abs(z[-1] - HEIGHT) >= 1e-9:
        raise ValueError("Ascent failed its speed or displacement check")
    return times, velocity, z


def ambient(p, t, q):
    """Calculate ambient water saturation ratio from pressure, T and vapor."""
    return p * q / (parcel.RD / parcel.RV + q) / parcel._saturation_pressure(t)


def specific_volume(p, t, q):
    """Calculate air volume per kilogram of dry air."""
    return (parcel.RD + q * parcel.RV) * t / p


def growth(rh, eq, p, t):
    """Return the diffusion- and heat-limited rate of wet radius squared."""
    diffusion = 2.11e-5 * (t / 273.15) ** 1.94 * (101325.0 / p)
    resistance = parcel.RHO_W * (
        parcel.RV * t / (diffusion * parcel._saturation_pressure(t))
        + parcel.LATENT
        / (parcel.get_thermal_conductivity(t) * t)
        * (parcel.LATENT / (parcel.RV * t) - 1)
    )
    return 2 * (rh - eq) / resistance  # d(r²)/dt


def integrate(case, wind, bins, nodes, tol, rh0=1.0, sample_dt=0.25):
    """Advance the coupled pressure, T, water vapor and class wet radii."""
    times, speeds, _ = wind
    edges = np.geomspace(*LIMITS, bins + 1)
    dia = np.sqrt(edges[:-1] * edges[1:])
    curves = cached_curves(tuple(dia), case, nodes)
    wet0 = curves.initial(rh0)
    vapor_pressure = rh0 * parcel._saturation_pressure(TEMP)
    q0 = parcel.RD / parcel.RV * vapor_pressure / (PRESSURE - vapor_pressure)
    vol0 = specific_volume(PRESSURE, TEMP, q0)
    number = NUMBER * 1e6 * vol0 * fraction(edges[:-1], edges[1:])
    weights = parcel.WATER_FACTOR * number  # 1 kg dry air
    water0 = q0 + weights @ (wet0**3 - curves.dry**3)

    def rhs(time, state):
        p, t, q = state[:3]
        wet = wet0 * np.sqrt(state[3:])
        w = np.interp(time, times, speeds)
        rate = growth(ambient(p, t, q), curves.interpolate(wet, t), p, t)
        dliquid = weights @ (1.5 * wet * rate)
        v = specific_volume(p, t, q)
        dp = -(1 + water0) * parcel.STANDARD_GRAVITY * w / v
        dt = (v * dp + parcel.LATENT * dliquid) / parcel.CP
        return np.r_[dp, dt, -dliquid, rate / wet0**2]

    varying = np.ptp(speeds) > 0
    sol = solve_ivp(
        rhs,
        (0.0, times[-1]),
        np.r_[PRESSURE, TEMP, q0, np.ones(bins)],
        method="LSODA",
        rtol=tol,
        atol=np.r_[1e-6, 1e-9, 1e-14, np.full(bins, 1e-11)],
        max_step=(1.0 if varying else 5.0),
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    sample = np.linspace(0, times[-1], round(times[-1] / sample_dt) + 1)
    values = sol.sol(sample)
    p, t, q = values[:3]
    wet = wet0[:, None] * np.sqrt(values[3:])
    liquid = weights @ (wet**3 - curves.dry[:, None] ** 3)
    residual = np.max(np.abs(q + liquid - water0))
    if residual >= 2e-8 or np.any(~np.isfinite(wet)) or np.any(q <= 0):
        raise ValueError("Parcel lost water or left its valid moisture range")
    if (
        np.any(wet < curves.dry[:, None])
        or t.min() < TEMP - 2
        or t.max() > TEMP + 1e-8
        or np.max(wet / curves.dry[:, None]) >= np.exp(curves.x[-1])
    ):
        raise ValueError("Parcel left its equilibrium lookup domain")
    vol = specific_volume(p, t, q)
    scrit_radius = curves.critical(t[-1])
    last = np.flatnonzero(sample >= sample[-1] - 5)
    activated = np.all(
        np.column_stack([wet[:, j] > curves.critical(t[j]) for j in last]),
        axis=1,
    )
    if np.any(np.diff(activated.astype(int)) < 0):
        raise ValueError("Activation set spans disconnected dry sizes")
    info = dict(
        peak_s_percent=float((ambient(p, t, q).max() - 1) * 100),
        discrete_count_cm3=float(number[activated].sum() / vol[-1] * 1e-6),
        water_residual_kgkg=float(residual),
        temperature_final_k=float(t[-1]),
        liquid_final_gkg=float(liquid[-1] * 1e3),
        volume_ratio=vol0 / vol[-1],
    )
    return (
        sol,
        info,
        (dia, activated, wet[:, -1], scrit_radius, number / vol[-1] * 1e-6),
        (sample, p, t, q, liquid),
    )


def tracer_status(dia, case, nodes, sample, air, tol, persistence=5.0):
    """Classify zero-weight tracer sizes in one prescribed air history."""
    curves = Curves(dia, case, nodes)
    wet0 = curves.initial(1.0)
    p_hist, t_hist, q_hist = air
    # RH is interpolated directly to avoid subtractive error near equilibrium.
    rh_hist = ambient(p_hist, t_hist, q_hist)

    def rhs(time, scaled):
        p, t, rh = [
            np.interp(time, sample, a) for a in (p_hist, t_hist, rh_hist)
        ]
        wet = wet0 * np.sqrt(scaled)
        return growth(rh, curves.interpolate(wet, t), p, t) / wet0**2

    last = np.flatnonzero(sample >= sample[-1] - persistence)
    sol = solve_ivp(
        rhs,
        (0, sample[-1]),
        np.ones_like(dia),
        method="LSODA",
        t_eval=sample[last],
        rtol=tol,
        atol=1e-11,
        max_step=2.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    critical = np.column_stack([curves.critical(t_hist[j]) for j in last])
    return np.all(wet0[:, None] * np.sqrt(sol.y) > critical, axis=1)


def cutoff(case, nodes, diag, history, tol, persistence=5.0):
    """Bracket the critical dry diameter in nanometres by tracer replay."""
    dia, act = diag[:2]
    index = np.searchsorted(act, True)
    # Check bins at either end against the physical support edges as well.
    lo = dia[index - 2] if index >= 2 else LIMITS[0]
    hi = dia[index + 1] if index + 1 < len(dia) else LIMITS[1]
    sample = history[0]
    for _ in range(4):
        grid = np.geomspace(lo, hi, 9)
        flags = tracer_status(
            grid, case, nodes, sample, history[1:4], tol, persistence
        )
        if np.any(np.diff(flags.astype(int)) < 0):
            raise ValueError("Tracer activation is not monotone in dry size")
        j = np.searchsorted(flags, True)
        if j == len(grid):
            if hi != LIMITS[1]:
                raise ValueError(
                    "Tracer and sectional activation disagree at upper bracket"
                )
            return hi, hi
        if j == 0:
            if lo != LIMITS[0]:
                raise ValueError(
                    "Tracer and sectional activation disagree at lower bracket"
                )
            return lo, lo
        lo, hi = grid[j - 1 : j + 1]
    return lo, hi


def record(case, mean, amp, tau, seed, bins, nodes, tol):
    """Collect concentration, size-resolved activation and a 1-s trajectory."""
    wind = forcing(mean, amp, tau, seed)
    sample_dt = 0.0625 if bins >= 64 else 0.25
    _, info, diag, history = integrate(
        case, wind, bins, nodes, tol, sample_dt=sample_dt
    )
    lo, hi = cutoff(case, nodes, diag, history, tol)
    info.update(
        dry_cutoff_nm=(lo + hi) / 2,
        cutoff_width_nm=hi - lo,
        count_cm3=NUMBER
        * info["volume_ratio"]
        * fraction((lo + hi) / 2, LIMITS[1]),
    )
    if bins >= 64:
        a, b = cutoff(case, nodes, diag, history, tol, persistence=0.0)
        info["instant_count_cm3"] = (
            NUMBER * info["volume_ratio"] * fraction((a + b) / 2, LIMITS[1])
        )
    else:
        info["instant_count_cm3"] = None
    keys = dict(
        recipe=case[0],
        model=case[1],
        scenario=case[2],
        mean_w_ms=mean,
        max_relative_perturbation=amp,
        input_correlation_s=tau,
        seed=seed,
        bins=bins,
        curve_nodes=nodes,
        rtol=tol,
    )
    summary = {**keys, **info}
    bysize = [
        dict(
            **keys,
            dry_nm=d,
            final_wet_radius_um=r * 1e6,
            critical_wet_radius_um=rc * 1e6,
            activated=int(a),
            final_seed_cm3=n,
        )
        for d, a, r, rc, n in zip(*diag, strict=True)
    ]
    sample, p, t, q, liquid = history
    stride = round(1 / sample_dt)  # 1 second intervals in the archive
    path = [
        dict(
            **keys,
            time_s=time,
            w_ms=w,
            height_m=z,
            supersaturation_pct=s,
            liquid_gkg=liquid_gkg,
        )
        for time, w, z, s, liquid_gkg in zip(
            sample[::stride],
            wind[1],
            wind[2],
            ((ambient(p, t, q) - 1) * 100)[::stride],
            (liquid * 1e3)[::stride],
            strict=True,
        )
    ]
    return summary, bysize, path


def reference():
    """Compare with Particula's CPU parcel at constant properties."""
    wind = forcing(1.0)
    _, _, diag, hist = integrate(REFERENCE, wind, 10, 4097, 2e-8, rh0=0.995)
    time, p, t, q, liquid = hist
    dia, _, terminal, _, density = diag
    number = density * specific_volume(p[-1], t[-1], q[-1]) * 1e6
    initial = parcel.ParcelInitialState(
        dia * 0.5e-9, number, KAPPA_CONTROL, PRESSURE, TEMP, q[0]
    )
    native = parcel.run_parcel(
        initial, updraft=1.0, timestep=0.25, steps=len(time) - 1
    )
    check = dict(
        max_pressure_pa=float(max(abs(p - native.pressure))),
        max_temperature_k=float(max(abs(t - native.temperature))),
        max_supersaturation_percentage_points=float(
            100 * max(abs(ambient(p, t, q) - 1 - native.supersaturation))
        ),
        max_liquid_kgkg=float(max(abs(liquid - native.liquid))),
        max_final_radius_relative=float(
            max(abs(terminal / native.wet_radius[-1] - 1))
        ),
    )
    if (
        check["max_supersaturation_percentage_points"] >= 0.002
        or check["max_final_radius_relative"] >= 0.002
    ):
        raise ValueError("Constant-property parcel failed native comparison")
    return check


def write_csv(path, rows):
    """Write records with stable field names and ordering."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    """Pair outcomes and integrate number above the cutoff in each bin."""
    pair_rows = []
    for stage in ("steady", "fluctuations", "refine", "uncertainty"):
        with (OUT / f"parcel_{stage}_summary.csv").open() as stream:
            summary = list(csv.DictReader(stream))
        case_fields = list(summary[0])[:10]  # recipe through rtol
        by_case = {tuple(r[k] for k in case_fields): r for r in summary}
        for key, octyl in by_case.items():
            if key[0] != "octyl":
                continue
            dodecyl = by_case[("dodecyl", *key[1:])]
            pair_rows.append(
                {
                    "stage": stage,
                    **{k: octyl[k] for k in case_fields[1:]},
                    "octyl_cm3": octyl["count_cm3"],
                    "dodecyl_cm3": dodecyl["count_cm3"],
                    "dodecyl_minus_octyl_cm3": float(dodecyl["count_cm3"])
                    - float(octyl["count_cm3"]),
                    "octyl_peak_s_pct": octyl["peak_s_percent"],
                    "dodecyl_peak_s_pct": dodecyl["peak_s_percent"],
                }
            )
        size_path = OUT / f"parcel_{stage}_sizes.csv"
        with size_path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            summary_row = by_case[tuple(row[k] for k in case_fields)]
            d = float(row["dry_nm"])
            half_bin_ratio = (LIMITS[1] / LIMITS[0]) ** (0.5 / int(row["bins"]))
            lo, hi = d / half_bin_ratio, d * half_bin_ratio
            threshold = float(summary_row["dry_cutoff_nm"])
            part = fraction(np.clip(threshold, lo, hi), hi) / fraction(lo, hi)
            row.update(
                dry_lower_nm=lo, dry_upper_nm=hi, activated_fraction=part
            )
        write_csv(size_path, rows)
        path = OUT / f"parcel_{stage}_paths.csv"
        if path.exists():
            target = path.with_suffix(".csv.gz")
            target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            path.unlink()
    write_csv(OUT / "parcel_pairs.csv", pair_rows)


def steady_cases():
    """Select thermodynamic scenario and its matching reference models."""
    shared = ("central", "growth-low", "growth-high")
    for scenario in shared:
        yield "acid", "finite", scenario
    for recipe in ("octyl", "dodecyl"):
        for scenario in (
            "central",
            "low",
            "high",
            *shared[1:],
            "dilute-low",
            "dilute-high",
        ):
            yield recipe, "finite", scenario
        for model in ("bulk", "fixed70"):
            yield recipe, model, "central"


def configurations(stage):
    """Construct the paired forcing, recipe, and resolution matrix."""
    from itertools import product

    recipes = ("octyl", "dodecyl")
    standard = (32, 1025, 2e-6)
    if stage == "steady":
        for mean, case in product((0.2, 1.0), steady_cases()):
            yield case, mean, 0, 0, 0, *standard
    if stage == "fluctuations":
        for (amp, tau), seed, recipe in product(
            ((0.4, 5), (0.8, 5), (0.8, 30)), range(6), recipes
        ):
            yield (recipe, "finite", "central"), 0.2, amp, tau, seed, *standard
    if stage == "uncertainty":
        variants = (
            "response-low",
            "response-high",
            "error-low",
            "error-high",
            "lab-small",
            "lab-large",
        )
        for mean, scenario, recipe in product((0.2, 1.0), variants, recipes):
            yield (recipe, "finite", scenario), mean, 0, 0, 0, *standard
        for scenario, seed, recipe in product(
            ("low", "high"), range(6), recipes
        ):
            yield (recipe, "finite", scenario), 0.2, 0.8, 30, seed, *standard
    if stage == "refine":
        winds = ((0.2, 0, 0, 0), (1.0, 0, 0, 0), (0.2, 0.8, 30, 0))
        resolutions = ((48, 2049, 5e-7), (64, 4097, 1e-7))
        for wind, recipe, resolution in product(winds, recipes, resolutions):
            yield (recipe, "finite", "central"), *wind, *resolution


def main():
    """Run one stage of the manual parcel experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "reference",
            "steady",
            "fluctuations",
            "refine",
            "uncertainty",
            "summarize",
        ],
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == "summarize":
        summarize()
        return
    if args.stage == "reference":
        checks = reference()
        (OUT / "native_parcel_check.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        print(checks)
        return
    configs = configurations(args.stage)
    summaries, sizes, paths = [], [], []
    for c in configs:
        summary, bysize, path = record(*c)
        summaries.append(summary)
        sizes.extend(bysize)
        if c[0][2] == "central" and c[4] == 0 and c[5] in (32, 64):
            paths.extend(path)
        print(
            c,
            f"Nd={summary['count_cm3']:.3f}, "
            f"peak={summary['peak_s_percent']:.4f}%",
            flush=True,
        )
    for name, rows in (
        ("summary", summaries),
        ("sizes", sizes),
        ("paths", paths),
    ):
        if rows:
            write_csv(OUT / f"parcel_{args.stage}_{name}.csv", rows)


if __name__ == "__main__":
    main()
