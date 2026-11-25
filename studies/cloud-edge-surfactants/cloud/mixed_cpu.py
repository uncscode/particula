"""Coupled CPU reference for externally mixed aerosol populations."""

import argparse
import csv
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# The existing case study supplies its equilibrium closures and parcel physics.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parcel_study as prior  # noqa: E402
from particula.dynamics import parcel  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
RECIPES = ("octyl", "dodecyl")
LEVELS = {
    32: (32, 1025, 2e-6, 0.25),
    48: (48, 2049, 5e-7, 0.125),
    64: (64, 4097, 1e-7, 0.0625),
    128: (128, 4097, 5e-8, 0.0625),
}


def background_equilibrium(dry, wet, temperature):
    """Return the κ=0.3 constant-water-tension benchmark equilibrium ratio."""
    return parcel._equilibrium(wet, dry, temperature, 0.3)


def population_list(target, background_cm3, scenario):
    """Define dry loadings (initial cm−3) and independent particle recipes."""
    if target == "mixed":
        target_counts = (("octyl", 150.0), ("dodecyl", 150.0))
    else:
        target_counts = ((target, 300.0),)
    populations = [
        (recipe, (recipe, "finite", scenario), count)
        for recipe, count in target_counts
    ]
    if background_cm3:
        populations.append(("background", "background", background_cm3))
    return populations


def joint_curves(dia, populations, nodes):
    """Build one lookup table with population-specific equilibrium rows."""
    labels = []
    counts = []
    cases = []
    for label, case, loading in populations:
        labels.append(label)
        counts.append(loading)
        cases.append(background_equilibrium if case == "background" else case)

    def combined_equilibrium(dry, wet, temperature):
        blocks = zip(
            cases,
            np.split(dry, len(cases)),
            np.split(wet, len(cases)),
            strict=True,
        )
        return np.concatenate(
            [
                prior.equilibrium(d, r, temperature, case)
                for case, d, r in blocks
            ]
        )

    curves = prior.Curves(np.tile(dia, len(cases)), combined_equilibrium, nodes)
    return labels, np.array(counts), cases, curves, combined_equilibrium


def initial_state(curves, direct_eq, direct):
    """Find stable cloud-base roots for the selected equilibrium evaluator."""
    lookup = (
        (lambda wet, t: direct_eq(curves.dry, wet, t))
        if direct
        else curves.interpolate
    )
    wet0 = curves.initial(1.0)
    if direct:
        # Bracket the lower branch locally around the lookup root and bisect.
        lo, hi = wet0 * 0.999, wet0 * 1.001
        if np.any(lookup(lo, prior.TEMP) > 1) or np.any(
            lookup(hi, prior.TEMP) < 1
        ):
            raise ValueError("Failed to bracket direct cloud-base equilibrium")
        for _ in range(28):
            mid = (lo + hi) / 2
            lower = lookup(mid, prior.TEMP) <= 1
            lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
        wet0 = (lo + hi) / 2
    return lookup, wet0


def simulate(
    populations,
    wind,
    bins=48,
    nodes=2049,
    tol=5e-7,
    sample_dt=0.125,
    direct=False,
):
    """Integrate all size–recipe classes in one parcel and close vapor and heat.

    The input populations are (label, equilibrium case, initial number cm−3).
    All share the baseline 100 nm geometric mean and geometric width of 1.5.
    Returns dictionaries for air diagnostics, tagged populations and bins, plus
    a tuple of arrays (time, pressure, T, vapor mixing ratio, liquid by tag).
    """
    times, speeds, _ = wind
    edges = np.geomspace(*prior.LIMITS, bins + 1)
    dia = np.sqrt(edges[:-1] * edges[1:])
    labels, counts, cases, curves, direct_eq = joint_curves(
        dia, populations, nodes
    )
    lookup, wet0 = initial_state(curves, direct_eq, direct)
    psat = parcel._saturation_pressure(prior.TEMP)
    q0 = parcel.RD / parcel.RV * psat / (prior.PRESSURE - psat)
    vol0 = prior.specific_volume(prior.PRESSURE, prior.TEMP, q0)
    shares = prior.fraction(edges[:-1], edges[1:])
    number = (counts[:, None] * (1e6 * vol0) * shares[None, :]).ravel()
    weights = number * parcel.WATER_FACTOR
    initial_water = q0 + weights @ (wet0**3 - curves.dry**3)
    loading_g = (1 + initial_water) * parcel.STANDARD_GRAVITY
    liquid_coeff = 1.5 * weights * wet0**3

    def rhs(time, state):
        p, t, q = state[:3]
        root = np.sqrt(state[3:])
        wet = wet0 * root
        rate = (
            prior.growth(prior.ambient(p, t, q), lookup(wet, t), p, t) / wet0**2
        )
        dliquid = (liquid_coeff * root) @ rate
        speed = np.interp(time, times, speeds)
        dp = -loading_g * speed / prior.specific_volume(p, t, q)
        dt = (-loading_g * speed + parcel.LATENT * dliquid) / parcel.CP
        return np.r_[dp, dt, -dliquid, rate]

    def jac(time, state):
        # Each droplet growth law depends only on its size and common air.
        # One simultaneous perturbation gives every diagonal size derivative.
        rates = rhs(time, state)
        n = state.size
        matrix = np.zeros((n, n))
        h = 3e-8 * np.abs(state)
        for j in range(3):
            shifted = state.copy()
            shifted[j] += h[j]
            matrix[:, j] = (rhs(time, shifted) - rates) / h[j]
        shifted = state.copy()
        shifted[3:] += h[3:]
        diag = (rhs(time, shifted)[3:] - rates[3:]) / h[3:]
        idx = np.arange(3, n)
        matrix[idx, idx] = diag
        root = np.sqrt(state[3:])
        dl = liquid_coeff * (0.5 * rates[3:] / root + root * diag)
        matrix[1, 3:] = dl * parcel.LATENT / parcel.CP
        matrix[2, 3:] = -dl
        return matrix

    sample = np.linspace(0, times[-1], round(times[-1] / sample_dt) + 1)
    sol = solve_ivp(
        rhs,
        (0.0, times[-1]),
        np.r_[prior.PRESSURE, prior.TEMP, q0, np.ones(len(wet0))],
        t_eval=sample,
        jac=jac,
        method="Radau" if direct else "LSODA",
        rtol=tol,
        atol=np.r_[1e-6, 1e-9, 1e-14, np.full(len(wet0), 1e-11)],
        max_step=1.0 if np.ptp(speeds) else 5.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    p, t, q = sol.y[:3]
    wet = wet0[:, None] * np.sqrt(sol.y[3:])
    if (
        not np.isfinite(wet).all()
        or np.any(wet < curves.dry[:, None])
        or np.any(wet / curves.dry[:, None] >= np.exp(curves.x[-1]))
        or np.min(t) < prior.TEMP - 2
        or np.max(t) > prior.TEMP + 1e-8
        or np.min(q) <= 0
    ):
        raise ValueError("Parcel state left the physical or lookup range")
    ns = len(sample)
    wclass = (weights[:, None] * (wet**3 - curves.dry[:, None] ** 3)).reshape(
        len(cases), bins, ns
    )
    liquid = wclass.sum(axis=1)
    water_error = np.max(np.abs(q + liquid.sum(axis=0) - initial_water))
    if water_error >= 2e-8:
        raise ValueError(f"Water mixing ratio drifted by {water_error}")
    rh = prior.ambient(p, t, q)
    vol = prior.specific_volume(p, t, q)
    info = dict(
        peak_s_pct=100 * (rh.max() - 1),
        t_peak_s=sample[np.argmax(rh)],
        final_air_volume_m3=vol[-1],
        final_temperature_k=t[-1],
        final_pressure_pa=p[-1],
        final_vapor_kgkg=q[-1],
        final_liquid_gkg=1e3 * liquid[:, -1].sum(),
        max_water_error_kgkg=water_error,
    )
    # Tick-by-tick maximum of the analytic curve, at the local temperature.
    # Avoid a large array of wet-radius tables over all times.
    last = np.flatnonzero(sample >= sample[-1] - 5)
    crit = np.stack([curves.critical(t[j]) for j in last], axis=1)
    flags = (wet[:, last] > crit).all(axis=1).reshape(len(cases), bins)
    tags, sizes = [], []
    history = (sample, p, t, q, liquid.sum(axis=0))
    for i, (label, case, loading) in enumerate(
        zip(labels, cases, counts, strict=True)
    ):
        if np.any(np.diff(flags[i].astype(int)) < 0):
            raise ValueError(f"Disconnected activation region for {label}")
        low, high = prior.cutoff(case, nodes, (dia, flags[i]), history, tol)
        threshold = (low + high) / 2
        expansion = vol0 / vol[-1]
        scale = loading * expansion
        active = prior.fraction(threshold, prior.LIMITS[1])
        tag = dict(
            population=label,
            initial_cm3=loading,
            cutoff_dry_nm=threshold,
            bracket_low_nm=low,
            bracket_high_nm=high,
            activated_fraction=active,
            activated_cm3=scale * active,
            bracket_width_cm3=scale * prior.fraction(low, high),
            discrete_cm3=scale * np.sum(shares[flags[i]]),
            final_liquid_gkg=1e3 * liquid[i, -1],
        )
        tags.append(tag)
        for j in range(bins):
            gj = i * bins + j
            part = (
                prior.fraction(
                    np.clip(threshold, edges[j], edges[j + 1]), edges[j + 1]
                )
                / shares[j]
            )
            sizes.append(
                dict(
                    population=label,
                    dry_nm=dia[j],
                    initial_count_cm3=shares[j] * loading,
                    wet_um_cloudbase=wet0[gj] * 1e6,
                    final_wet_radius_um=wet[gj, -1] * 1e6,
                    final_critical_radius_um=crit[gj, -1] * 1e6,
                    active_center=int(flags[i, j]),
                    activated_fraction=part,
                )
            )
    return info, tags, sizes, (sample, p, t, q, liquid)


def write(path, rows):
    """Write CSV tables with optional gzip compression."""
    memory = io.StringIO(newline="")
    writer = csv.DictWriter(memory, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = memory.getvalue().encode("utf-8")
    path.write_bytes(
        gzip.compress(data, mtime=0) if path.suffix == ".gz" else data
    )


def runs():
    """Generate cases across loading, recipe, updraft and resolution."""
    for w in (0.2, 1.0):
        for background in (0, 1000):
            for target in ("mixed", "acid", "octyl", "dodecyl"):
                for bins in (32, 64, 128) if target == "mixed" else (32, 64):
                    yield w, background, target, "central", bins
        for scenario in ("low", "high"):
            for bins in (32, 64):
                yield w, 0, "mixed", scenario, bins


def crosscheck(out):
    """Check four mixed cases using direct equilibrium and Radau."""
    with (out / "cpu_populations.csv").open() as stream:
        population_data = list(csv.DictReader(stream))
    checks = []
    for w in (0.2, 1.0):
        for bg in (0, 1000):
            info, tags, _, _ = simulate(
                population_list("mixed", bg, "central"),
                prior.forcing(w),
                64,
                4097,
                1e-8,
                0.0625,
                direct=True,
            )
            for tag in tags:
                ref = next(
                    r
                    for r in population_data
                    if r["target"] == "mixed"
                    and r["scenario"] == "central"
                    and float(r["updraft_ms"]) == w
                    and int(r["background_initial_cm3"]) == bg
                    and int(r["bins_per_population"]) == 64
                    and r["population"] == tag["population"]
                )
                checks.append(
                    dict(
                        updraft_ms=w,
                        background_initial_cm3=bg,
                        population=tag["population"],
                        activated_cm3=tag["activated_cm3"],
                        direct_minus_lookup_cm3=tag["activated_cm3"]
                        - float(ref["activated_cm3"]),
                        direct_peak_s_pct=info["peak_s_pct"],
                        max_water_error_kgkg=info["max_water_error_kgkg"],
                    )
                )
    write(out / "cpu_direct_radau.csv", checks)


def read_rows(path):
    """Read one result table with optional gzip compression."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(out):
    """Check counts, water and energy against the recorded classes."""
    tables = [
        read_rows(out / name)
        for name in ("cpu_air.csv", "cpu_populations.csv", "cpu_classes.csv.gz")
    ]
    air, populations, sizes = tables
    ids = list(air[0])[:5]

    def key(row):
        return tuple(row[field] for field in ids)

    tagged = {(key(r), r["population"]): r for r in populations}
    init = parcel._saturation_pressure(prior.TEMP)
    q0 = parcel.RD / parcel.RV * init / (prior.PRESSURE - init)
    vol0 = prior.specific_volume(prior.PRESSURE, prior.TEMP, q0)
    phase_error, heat_error, number_error = [], [], []
    lookup = {key(r): r for r in air}
    for run in air:
        selected = [r for r in sizes if key(r) == key(run)]
        vol = float(run["final_air_volume_m3"])
        centers = np.array([float(r["dry_nm"]) * 0.5e-9 for r in selected])
        base = np.array([float(r["wet_um_cloudbase"]) * 1e-6 for r in selected])
        final = np.array(
            [float(r["final_wet_radius_um"]) * 1e-6 for r in selected]
        )
        n = np.array(
            [float(r["initial_count_cm3"]) * vol0 * 1e6 for r in selected]
        )
        liquid0 = parcel.WATER_FACTOR * n @ (base**3 - centers**3)
        liquid1 = parcel.WATER_FACTOR * n @ (final**3 - centers**3)
        q1 = float(run["final_vapor_kgkg"])
        phase_error.append(abs(liquid1 + q1 - liquid0 - q0))
        heat_error.append(
            abs(
                parcel.CP * (float(run["final_temperature_k"]) - prior.TEMP)
                + (1 + liquid0 + q0) * parcel.STANDARD_GRAVITY * prior.HEIGHT
                + parcel.LATENT * (q1 - q0)
            )
        )
        for label in sorted({r["population"] for r in selected}):
            part = [r for r in selected if r["population"] == label]
            activated = sum(
                float(r["initial_count_cm3"])
                * vol0
                / vol
                * float(r["activated_fraction"])
                for r in part
            )
            number_error.append(
                abs(
                    activated
                    - float(tagged[(key(run), label)]["activated_cm3"])
                )
            )
    if (
        max(phase_error) > 2e-8
        or max(heat_error) > 1e-5
        or max(number_error) > 1e-8
    ):
        raise ValueError("CPU replay failed water, energy or number accounting")
    direct = read_rows(out / "cpu_direct_radau.csv")
    delta_s = []
    for r in direct:
        k = (
            r["updraft_ms"],
            r["background_initial_cm3"],
            "mixed",
            "central",
            "64",
        )
        delta_s.append(
            abs(float(r["direct_peak_s_pct"]) - float(lookup[k]["peak_s_pct"]))
        )
    prior_dir = Path(prior.__file__).resolve().parent / "results"
    prior_rows = read_rows(prior_dir / "parcel_steady_summary.csv")
    prior_rows.extend(read_rows(prior_dir / "parcel_refine_summary.csv"))
    replacement_error = []
    for r in prior_rows:
        if (
            r["model"] != "finite"
            or r["scenario"] != "central"
            or float(r["max_relative_perturbation"])
            or r["bins"] not in {"32", "64"}
        ):
            continue
        k = (r["mean_w_ms"], "0", r["recipe"], "central", r["bins"])
        replacement_error.append(
            abs(
                float(tagged[(k, r["recipe"])]["activated_cm3"])
                - float(r["count_cm3"])
            )
        )
    if max(replacement_error) > 0.01 or max(delta_s) > 5e-5:
        raise ValueError(
            "CPU reference failed prior-run or direct-solver check"
        )
    summary = dict(
        coupled_cases=len(air),
        prior_replacement_comparisons=len(replacement_error),
        direct_radau_cases=4,
        max_water_drift_kgkg=max(float(a["max_water_error_kgkg"]) for a in air),
        terminal_water_error_kgkg=max(phase_error),
        terminal_energy_error_J_kg=max(heat_error),
        max_number_reconstruction_error_cm3=max(number_error),
        max_prior_replacement_difference_cm3=max(replacement_error),
        max_direct_peak_s_difference_percentage_points=max(delta_s),
        max_direct_count_difference_cm3=max(
            abs(float(r["direct_minus_lookup_cm3"])) for r in direct
        ),
    )
    (out / "cpu_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


def main():
    """Save air histories, class results and refined activation counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--crosscheck", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.audit:
        audit(args.out)
        return
    if args.crosscheck:
        crosscheck(args.out)
        return
    air, pop, grid, paths = [], [], [], []
    for w, bg, target, scenario, bins in runs():
        controls = LEVELS[bins]
        keys = dict(
            updraft_ms=w,
            background_initial_cm3=bg,
            target=target,
            scenario=scenario,
            bins_per_population=bins,
        )
        populations = population_list(target, bg, scenario)
        info, tags, sizes, history = simulate(
            populations, prior.forcing(w), *controls
        )
        air.append({**keys, **info})
        pop.extend({**keys, **r} for r in tags)
        grid.extend({**keys, **r} for r in sizes)
        if target == "mixed" and scenario == "central":
            time, p, t, q, liquid = history
            stride = round(1 / controls[-1])
            labels = [r[0] for r in populations]
            for j in range(0, len(time), stride):
                grouped = dict(zip(labels, liquid[:, j] * 1000, strict=True))
                paths.append(
                    dict(
                        **keys,
                        time_s=time[j],
                        height_m=time[j] * w,
                        pressure_pa=p[j],
                        temperature_k=t[j],
                        qv_kgkg=q[j],
                        supersaturation_pct=100
                        * (prior.ambient(p[j], t[j], q[j]) - 1),
                        octyl_liquid_gkg=grouped["octyl"],
                        dodecyl_liquid_gkg=grouped["dodecyl"],
                        background_liquid_gkg=grouped.get("background", 0.0),
                    )
                )
        print(
            keys,
            [(r["population"], round(r["activated_cm3"], 5)) for r in tags],
            flush=True,
        )
    for name, rows in (
        ("cpu_air.csv", air),
        ("cpu_populations.csv", pop),
        ("cpu_classes.csv.gz", grid),
        ("cpu_paths.csv.gz", paths),
    ):
        write(args.out / name, rows)
    print(
        json.dumps(
            {
                "cases": len(air),
                "max_water_error_kgkg": max(
                    row["max_water_error_kgkg"] for row in air
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
