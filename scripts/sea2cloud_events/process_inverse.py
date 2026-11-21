"""Fit pre-event Sea2Cloud counts and predict held-out size bands on CUDA.

Run from the checkout with python -m scripts.sea2cloud_events.process_inverse.
"""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import warp as wp
from particula import dynamics, particles
from scipy.constants import Avogadro, R
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq, lsq_linear, minimize
from scipy.special import expit
from scripts.sea2cloud_events.marine_history import read
from scripts.sea2cloud_events.study import DENSITY, EXPONENT, PRESSURE, TEMP

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "website/public/study-data/sea2cloud"
START_SIZES = np.array([1.0, 5.0, 1.5, 3.0, 6.0, 8.0])
DX = 0.025  # h at unit organic concentration
SCENARIOS = {
    "dry Brownian": (0.0, "brownian"),
    "humid Brownian": (0.80, "brownian"),
    "humid hard sphere": (0.80, "hard"),
    "humid Coulomb neutral": (0.80, "neutral"),
    "humid Coulomb mixed": (0.80, "mixed"),
    "RH 78.1%": (0.781, "brownian"),
    "RH 82.7%": (0.827, "brownian"),
}


def wet_sizes(dry, rh):
    """Equilibrium wet nm by kappa-Köhler, including water curvature."""
    if rh == 0:
        return dry.copy()
    kelvin_nm = 4 * 0.072 * 0.01801528 / (1000 * R * TEMP) * 1e9

    def residual(g, d):
        water = g**3 - 1
        return water / (water + 0.3) * np.exp(kelvin_nm / (g * d)) - rh

    return dry * np.array([brentq(residual, 1, 4, args=(d,)) for d in dry])


def physics(start, onset, scenario):
    """Return dry size, scavenging hazard /h and unit-vapor growth nm/h."""
    background = read("background")
    window = background[
        (background[:, 0] >= start) & (background[:, 0] < onset)
    ]
    spectrum = pd.DataFrame(window).groupby(1)[2].median()
    diameter = np.geomspace(1, 400, 600)
    dry = np.r_[diameter, spectrum.index]
    rh, law = SCENARIOS[scenario]
    wet = wet_sizes(dry, rh)
    radius = wet * 0.5e-9
    mass = np.pi / 6 * 1e-27 * (DENSITY * dry**3 + 1000 * (wet**3 - dry**3))
    n = len(diameter)
    if law == "brownian":
        kernel = dynamics.get_brownian_kernel_via_system_state(
            radius, mass, TEMP, PRESSURE
        )
    elif law == "hard":
        kernel = dynamics.get_hard_sphere_kernel_via_system_state(
            radius, mass, np.zeros_like(dry), TEMP, PRESSURE
        )
    else:
        coulomb = dynamics.get_coulomb_kernel_gatti2008_via_system_state
        kernel = coulomb(radius, mass, np.zeros_like(dry), TEMP, PRESSURE)
        if law == "mixed":
            # Charge fractions 10% (small) and 50% (collectors), balanced signs.
            kernel *= 0.95
            for sign in (-1, 1):
                z = np.r_[np.ones(n), np.full(len(spectrum), sign)]
                kernel += 0.025 * coulomb(radius, mass, z, TEMP, PRESSURE)
    hazard = kernel[:n, n:] @ spectrum.to_numpy() * 3.6e9
    diffusivity, molar_mass = 6e-6, 0.2
    thermal_speed = np.sqrt(8 * R * TEMP / (np.pi * molar_mass))
    kn = 3 * diffusivity / (thermal_speed * radius[:n])
    transport = dynamics.get_first_order_mass_transport_k(
        radius[:n],
        particles.get_vapor_transition_correction(kn, 1.0),
        diffusivity,
    )
    # 10^6 molecules/cm3, dry mass gain at the wet surface.
    growth = (
        2
        * transport
        * (1e12 * molar_mass / Avogadro)
        / (np.pi * DENSITY * (diameter * 1e-9) ** 2)
        * 3.6e12
    )
    if not (
        np.all(np.isfinite(hazard) & (hazard >= 0))
        and np.all(np.isfinite(growth) & (growth > 0))
    ):
        raise ValueError("Invalid collision or condensation rates")
    return diameter, hazard, growth


def exposure_table(diameter, hazard, growth):
    """Tabulate dry size and mean trajectory hazard against vapor × age."""
    integral = cumulative_trapezoid(1 / growth, diameter, initial=0)
    optical = cumulative_trapezoid(hazard / growth, diameter, initial=0)
    x = np.arange(0, 1025, DX)
    sizes, means = [], []
    for initial in START_SIZES:
        t0 = np.interp(initial, diameter, integral)
        if t0 + x[-1] >= integral[-1]:
            raise ValueError("Size grid does not span the exposure table")
        size = np.interp(t0 + x, integral, diameter)
        tau = np.interp(t0 + x, integral, optical)
        mean = np.full_like(x, np.interp(initial, diameter, hazard))
        mean[1:] = (tau[1:] - tau[0]) / x[1:]
        sizes.append(size)
        means.append(mean)
    return np.array(sizes), np.array(means)


@wp.kernel
def count(
    row: wp.array(dtype=int),
    kind: wp.array(dtype=int),
    age: wp.array(dtype=wp.float64),
    duration: wp.array(dtype=wp.float64),
    birth: wp.array(dtype=wp.float64),
    cut: wp.array(dtype=wp.float64),
    size: wp.array2d(dtype=wp.float64),
    hazard: wp.array2d(dtype=wp.float64),
    p: wp.array(dtype=wp.float64),
    dx: wp.float64,
    out: wp.array(dtype=wp.float64),
):
    """Accumulate each source cohort into its measurement size band."""
    j = wp.tid()
    k = kind[j]
    a = age[j]
    x = p[0] * a / dx
    i = wp.int32(wp.floor(x))
    f = x - wp.float64(i)
    d = size[k, i] + f * (size[k, i + 1] - size[k, i])
    h = hazard[k, i] + f * (hazard[k, i + 1] - hazard[k, i])
    number = wp.float64(250.0) * p[1]
    if k == 0:
        number = birth[j] + duration[j] * wp.float64(100.0) * p[2]
    elif k == 1:
        number = duration[j] * wp.float64(100.0) * p[3]
    # Smooth upper diameter cut with a prescribed 5% roll-off scale.
    c = cut[row[j]]
    response = wp.float64(1.0) / (
        wp.float64(1.0) + wp.exp((d - c) / (wp.float64(0.05) * c))
    )
    wp.atomic_add(out, row[j], number * response * wp.exp(-a * h))


@wp.kernel
def objective(
    pred: wp.array(dtype=wp.float64),
    target: wp.array(dtype=wp.float64),
    norm: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    """Accumulate normalized pre-event squared error."""
    i = wp.tid()
    residual = pred[i] - target[i]
    wp.atomic_add(loss, 0, norm * residual * residual)


def device_array(a, dtype):
    """Copy a typed input to the study GPU."""
    return wp.array(np.asarray(a), dtype=dtype, device="cuda:0")


class Model:
    """Pack midpoint cohorts once per evaluation schedule and instrument cut."""

    def __init__(self, start, times, cuts, tables, dt):
        """Assemble source weights at each requested time."""
        ages, kinds, duration, births, rows = [], [], [], [], []
        acid = read("hourly-acid")
        for i, t in enumerate(times):
            span = t - start
            steps = max(1, int(np.ceil(span / dt)))
            width = span / steps
            age = (np.arange(steps) + 0.5) * width
            forcing = np.interp(t - age, acid[:, 0], acid[:, 1], left=0)
            births.append(
                np.r_[
                    3600 * width * 3.5e-37 * forcing**EXPONENT,
                    np.zeros(steps + 4),
                ]
            )
            ages.append(np.r_[age, age, np.full(4, span)])
            kinds.append(
                np.r_[np.zeros(steps), np.ones(steps), np.arange(2, 6)]
            )
            duration.append(np.r_[np.full(2 * steps, width), np.zeros(4)])
            rows.append(np.full(2 * steps + 4, i))
        self.birth_ceiling = np.array([b.sum() for b in births])
        self.n = len(times)
        self.inputs = [
            device_array(np.concatenate(v), dtype)
            for v, dtype in [
                (rows, int),
                (kinds, int),
                (ages, wp.float64),
                (duration, wp.float64),
                (births, wp.float64),
            ]
        ]
        self.inputs.extend(device_array(a, wp.float64) for a in (cuts, *tables))
        self.length = sum(map(len, rows))
        self.max_vapor = (tables[0].shape[1] - 1) * DX / (max(times) - start)

    def evaluate(self, params, target=None):
        """Return counts, or the CUDA loss and its four derivatives."""
        if not 0 <= params[0] < self.max_vapor:
            raise ValueError("Vapor exposure outside size table")
        grad = target is not None
        p = wp.array(
            params, dtype=wp.float64, device="cuda:0", requires_grad=grad
        )
        pred = wp.zeros(
            self.n, dtype=wp.float64, device="cuda:0", requires_grad=grad
        )
        loss = wp.zeros(
            1, dtype=wp.float64, device="cuda:0", requires_grad=grad
        )
        with wp.Tape() as tape:
            wp.launch(
                count, self.length, [*self.inputs, p, DX, pred], device="cuda:0"
            )
            if grad:
                wp.launch(
                    objective,
                    self.n,
                    [pred, target, 1 / (self.n * 500**2), loss],
                    device="cuda:0",
                )
        if grad:
            tape.backward(loss)
            return loss.numpy()[0], p.grad.numpy()
        return pred.numpy()


def optimize(model, y, form):
    """Minimize the pre-event loss from three vapor starting points."""
    target = device_array(y, wp.float64)
    bounds = [(0, 4), (0, 6), (0, 12), (0, 12)]
    if form == "acid only":
        return np.zeros(4)
    if form != "1 nm supply":
        bounds[2] = (0, 0)
    if form != "5 nm supply":
        bounds[3] = (0, 0)
    if form == "no uptake":
        bounds[0] = (0, 0)
        bounds[2] = (0, 12)
    results = [
        minimize(
            model.evaluate,
            [u, 0.4, 0.1, 0.1],
            args=(target,),
            bounds=bounds,
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-12, "maxiter": 180},
        )
        for u in (0.06, 1.2, 3.5)
    ]
    valid = [r for r in results if r.success]
    if not valid:
        raise RuntimeError([r.message for r in results])
    return min(valid, key=lambda r: r.fun).x


def counter_data(onset, end):
    """Extract high-flow records bracketed by two candidate inlet plateaus."""
    with ZipFile(OUTPUT / "tangaroa-counts.zip") as archive:
        with archive.open("cpc-shift-pairs.csv") as stream:
            raw = pd.read_csv(stream)
    utc = pd.to_datetime(raw.utc, format="mixed", utc=True)
    raw["hour"] = (
        utc - pd.Timestamp("2020-03-19", tz="UTC")
    ).dt.total_seconds() / 3600
    raw = raw[
        raw.hour.between(onset, end, inclusive="left")
        & raw.same_tracer_triplet
        & raw.inlet_evidence.eq("VOC candidate")
    ].copy()
    a = raw.psm_cm3 - raw.backward_psm_cm3
    b = raw.psm_cm3 - raw.forward_psm_cm3
    raw["step_low_cm3"], raw["step_high_cm3"] = (
        np.minimum(a, b),
        np.maximum(a, b),
    )
    return raw[
        [
            "utc",
            "hour",
            "step_difference_cm3",
            "step_low_cm3",
            "step_high_cm3",
            "difference_cm3",
            "flag",
        ]
    ]


def interval_error(pred, bounds):
    """Return distance to the nearest edge, or zero inside the span."""
    return np.maximum.reduce(
        [bounds[:, 0] - pred, pred - bounds[:, 1], np.zeros_like(pred)]
    )


def spectrum_test(start, onset, end, tables):
    """Construct a held-out 12–25 nm number stock with a frozen background."""
    data = pd.DataFrame(
        read("background"), columns=["hour", "dry_nm", "bin_cm3"]
    )
    size = data[data.dry_nm.between(12, 25, inclusive="left")]
    pre = size[(size.hour >= start) & (size.hour < onset)]
    q = size[size.hour.between(onset, end + 6)].groupby("hour").bin_cm3.sum()
    q = q.rename("measured_cm3").reset_index()
    q["baseline_cm3"] = pre.groupby("dry_nm").bin_cm3.median().sum()
    model = Model(
        start, np.tile(q.hour, 2), np.repeat([12, 25], len(q)), tables, 0.05
    )
    return q, model


def vapor_profile(day, fit, y, model, offset):
    """Solve amplitudes at fixed vapor, including wider stress cases."""
    result = []
    for vapor in (0, 1, 2, 4, 8, 16, 32):
        p = np.array([vapor, 0.0, 0.0, 0.0])
        base = fit.evaluate(p)
        design = np.column_stack(
            [fit.evaluate(p + e) - base for e in np.eye(4)[1:3]]
        )
        amplitudes = lsq_linear(design, y - base, bounds=([0, 0], [6, 12])).x
        p[1:3] = amplitudes
        rmse = np.sqrt(np.mean((fit.evaluate(p) - y) ** 2))
        result.append(
            [
                day,
                vapor,
                1000 * p[1],
                100 * p[2],
                rmse,
                model.evaluate(p)[offset:].max(),
            ]
        )
    return result


def validate_gradients(fit, obs, train, day, checks):
    """Compare all four smooth local derivatives with central differences."""
    p0 = np.array([0.78913, 0.31, 0.12, 0.17])
    y = device_array(obs[train, 1:].mean(axis=1), wp.float64)
    _, ad = fit.evaluate(p0, y)
    delta = np.eye(4) * 1e-6
    fd = [
        (fit.evaluate(p0 + v, y)[0] - fit.evaluate(p0 - v, y)[0]) / 2e-6
        for v in delta
    ]
    err = np.max(abs(ad - fd) / np.maximum(1e-6, abs(ad)))
    np.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-10)
    checks.append([day, "gradient maximum scaled error", err])


def independent_count(start, hour, law, params, cut):
    """Integrate sizes and losses directly in time for a CPU reference."""
    vapor, initial, source1, source5 = params
    span = hour - start
    age = np.linspace(0, span, 2001)
    acid = read("hourly-acid")
    forcing = np.interp(hour - age, acid[:, 0], acid[:, 1], left=0)
    rate = 3600 * 3.5e-37 * forcing**EXPONENT
    diameter, hazard, growth = law

    def rhs(t, state):
        d, survival = state
        return [
            vapor * np.interp(d, diameter, growth),
            -survival * np.interp(d, diameter, hazard),
        ]

    total = 0.0
    for k, d in enumerate(START_SIZES):
        sol = solve_ivp(
            rhs, (0, span), (d, 1.0), t_eval=age, rtol=1e-8, atol=1e-10
        )
        size, surv = sol.y
        window = expit((cut - size) / (0.05 * cut)) * surv
        if k < 2:
            supply = rate + 100 * source1 if k == 0 else 100 * source5
            total += np.trapezoid(supply * window, age)
        else:
            total += 250 * initial * window[-1]
    return total


def inventory_bound(obs, onset, end):
    """Bound extra net input between the peak bin and an earlier sampled bin."""
    event = obs[(obs[:, 0] >= onset) & (obs[:, 0] < end)]
    peak = event[np.argmax(event[:, 2])]
    # Use the most restrictive earlier bin and allow the longest elapsed
    # time consistent with both half-hour windows.
    earlier = obs[obs[:, 0] < peak[0]]
    acid = read("hourly-acid")
    candidates = []
    for hour, _, high in earlier:
        max_hours = peak[0] - hour + 0.5
        t = np.linspace(hour - 0.25, peak[0] + 0.25, 1001)
        # Upper Riemann sum for the linearly interpolated hourly acid.
        t = np.unique(
            np.r_[t, acid[(acid[:, 0] > t[0]) & (acid[:, 0] < t[-1]), 0]]
        )
        forcing = np.interp(t, acid[:, 0], acid[:, 1], left=0)
        local = np.sum(
            np.diff(t)
            * 3600
            * 3.5e-37
            * np.maximum(forcing[1:], forcing[:-1]) ** EXPONENT
        )
        gain = max(0, peak[2] - high - 60 - local)
        candidates.append(
            [
                hour,
                peak[0],
                high,
                peak[2],
                local,
                gain,
                max_hours,
                gain / max_hours,
            ]
        )
    return max(candidates, key=lambda r: r[-1])


def process_rates(day, scenario, law):
    """Sample the per-size collision hazard and organic growth rate."""
    return [
        [day, scenario, d, *[np.interp(d, law[0], v) for v in law[1:]]]
        for d in (1, 3, 5, 10, 25)
    ]


def counter_predictions(day, form, raw, pred, offset):
    """Attach predicted nominal bands to the independent counter records."""
    sub = raw.copy()
    n = len(raw)
    sub["day"], sub["model"] = day, form
    sub["pred_1_3_cm3"] = pred[offset : offset + n]
    sub["pred_1_10_cm3"] = pred[offset + n : offset + 2 * n]
    return sub


def study_cases(scenario):
    """Select paired physics controls and competing source models."""
    if scenario == "dry Brownian":
        forms = ["1 nm supply"]
    elif scenario == "humid Brownian":
        forms = [
            "acid only",
            "initial only",
            "no uptake",
            "1 nm supply",
            "5 nm supply",
        ]
    else:
        forms = []
    cases = [(form, "midpoint") for form in forms]
    if scenario == "humid Brownian":
        cases += [("1 nm supply", s) for s in ("low", "high")]
    return cases


def run():
    """Run the independent March 19 and March 22 inversions and controls."""
    spectra, profiles, gains = [], [], []
    scores, traces, counters, checks, physics_rows = [], [], [], [], []
    observations = read("observations")
    for day, start, onset, end in [(19, -3, 9, 16), (22, 54, 78, 79)]:
        obs = observations[
            (observations[:, 0] >= start) & (observations[:, 0] <= end + 6)
        ]
        train = obs[:, 0] < onset
        gains.append([day, *inventory_bound(obs, onset, end)])
        event = (~train) & (obs[:, 0] < end)
        raw = counter_data(onset, end)
        curves = np.arange(onset, end + 0.01, 0.05)
        times = np.r_[obs[:, 0], raw.hour, raw.hour, curves]
        cuts = np.r_[
            np.full(len(obs), 10),
            np.full(len(raw), 3),
            np.full(len(raw) + len(curves), 10),
        ]
        n, m = len(obs), len(raw)
        for scenario in SCENARIOS:
            law = physics(start, onset, scenario)
            tables = exposure_table(*law)
            physics_rows.extend(process_rates(day, scenario, law))
            fit = Model(start, obs[train, 0], cuts[: train.sum()], tables, 0.05)
            model = Model(start, times, cuts, tables, 0.05)
            if scenario == "humid Brownian":
                validate_gradients(fit, obs, train, day, checks)
                profiles.extend(
                    vapor_profile(
                        day, fit, obs[train, 1:].mean(axis=1), model, n + 2 * m
                    )
                )
                size_schedule, spectrum_model = spectrum_test(
                    start, onset, end, tables
                )
            cases = study_cases(scenario)
            for form, target in [*cases, ("fixed inputs", "control")]:
                ys = obs[train, 1:]
                y = dict(
                    midpoint=ys.mean(axis=1), low=ys[:, 0], high=ys[:, 1]
                ).get(target)
                p = (
                    np.array([1.0, 0.35, 0.50, 0.0])
                    if y is None
                    else optimize(fit, y, form)
                )
                pred = model.evaluate(p)
                pre = obs[train, 1:].mean(axis=1)
                peak = pred[n + 2 * m :].max()
                errs = interval_error(pred[:n], obs[:, 1:])
                scores.append(
                    [
                        day,
                        scenario,
                        form,
                        target,
                        *p,
                        np.sqrt(np.mean((pred[:n][train] - pre) ** 2)),
                        errs[event].mean(),
                        errs[obs[:, 0] >= end].mean(),
                        peak,
                        model.birth_ceiling[n + 2 * m :].max(),
                    ]
                )
                if scenario == "humid Brownian" and target == "midpoint":
                    q = size_schedule.copy()
                    q["day"], q["model"] = day, form
                    lo, hi = np.split(spectrum_model.evaluate(p), 2)
                    q["pred_cm3"] = q.baseline_cm3 + hi - lo
                    spectra.append(q)
                    traces.extend(
                        [day, form, *row, val]
                        for row, val in zip(obs, pred[:n], strict=True)
                    )
                    if form in ("1 nm supply", "5 nm supply"):
                        counters.append(
                            counter_predictions(day, form, raw, pred, n)
                        )
                    if form == "1 nm supply":
                        direct = independent_count(start, times[n], law, p, 3)
                        direct_err = abs(pred[n] - direct)
                        np.testing.assert_allclose(
                            pred[n], direct, rtol=0.001, atol=0.01
                        )
                        checks.append(
                            [
                                day,
                                "CPU time-domain 1–3 nm absolute cm-3",
                                direct_err,
                            ]
                        )
                        fine = Model(
                            start, times, cuts, tables, 0.025
                        ).evaluate(p)
                        e = max(abs(pred - fine))
                        np.testing.assert_allclose(pred, fine, rtol=0, atol=2)
                        checks.append(
                            [day, "half timestep max absolute cm-3", e]
                        )
                if form == "acid only":
                    np.testing.assert_array_less(
                        pred, model.birth_ceiling + 1e-9
                    )
        print(f"Finished March {day}", flush=True)
    frames = {
        "process-spectrum": pd.concat(spectra),
        "process-gain": pd.DataFrame(
            gains,
            columns=(
                "day earlier_hour peak_hour earlier_high peak_high "
                "acid_upper_cm3 min_extra_cm3 max_hours min_extra_cm3_h"
            ).split(),
        ),
        "process-profile": pd.DataFrame(
            profiles,
            columns=(
                "day vapor_million initial_cm3 source1_cm3_h "
                "train_mid_rmse event_peak_cm3"
            ).split(),
        ),
        "process-scores": pd.DataFrame(
            scores,
            columns=(
                "day scenario model target vapor_million initial_thousand "
                "source1_hundreds_per_h source5_hundreds_per_h train_mid_rmse "
                "event_interval_mae post_interval_mae event_peak_cm3 "
                "acid_birth_ceiling_cm3"
            ).split(),
        ),
        "process-observations": pd.DataFrame(
            traces, columns="day model hour obs_low obs_high pred_cm3".split()
        ),
        "process-counters": pd.concat(counters),
        "process-checks": pd.DataFrame(
            checks, columns=["day", "check", "value"]
        ),
        "process-physics": pd.DataFrame(
            physics_rows,
            columns=(
                "day scenario dry_nm hazard_per_h growth_nm_per_h_at_unit_vapor"
            ).split(),
        ),
    }
    for name, frame in frames.items():
        frame.to_csv(OUTPUT / (name + ".csv"), index=False, float_format="%.8g")
    plot(frames)


def plot(frames):
    """Draw the fitted series and independent particle counter comparison."""
    import matplotlib.pyplot as plt

    colors = {
        "acid only": "#555555",
        "1 nm supply": "#0072b2",
        "5 nm supply": "#d55e00",
        "no uptake": "#009e73",
    }
    fig, axs = plt.subplots(3, 2, figsize=(10, 9), layout="constrained")
    for i, (day, onset, end) in enumerate([(19, 9, 16), (22, 78, 79)]):
        ax = axs[0, i]
        df = frames["process-observations"].query("day == @day")
        measured = df.drop_duplicates("hour")
        ax.vlines(
            measured.hour - onset,
            measured.obs_low,
            measured.obs_high,
            color=".35",
            lw=2,
            label="Figure 5a span",
        )
        ax.scatter(
            measured.hour - onset,
            (measured.obs_low + measured.obs_high) / 2,
            color=".35",
            s=7,
        )
        for name in ["acid only", "1 nm supply", "5 nm supply"]:
            s = df[df.model == name]
            ax.plot(
                s.hour - onset,
                s.pred_cm3,
                ".-",
                ms=3,
                color=colors[name],
                label=name,
            )
        ax.axvspan(0, end - onset, color=".85", alpha=0.5)
        ax.set(
            title=f"March {day} · published 1–10 nm",
            xlabel="Hours from event start",
            ylabel="Number (cm⁻³)",
        )
        if i == 1:
            ax.legend(fontsize=7)
        ax = axs[1, i]
        s = frames["process-counters"].query(
            "day == @day and model == '1 nm supply'"
        )
        x = (s.hour - onset) * 60
        ax.vlines(
            x,
            s.step_low_cm3,
            s.step_high_cm3,
            color=".55",
            label="Adjacent low-flow readings",
        )
        ax.scatter(
            x,
            s.step_difference_cm3,
            s=15,
            color="black",
            label="PSM step difference",
        )
        for name in ["1 nm supply", "5 nm supply"]:
            q = frames["process-counters"].query(
                "day == @day and model == @name"
            )
            ax.scatter(
                (q.hour - onset) * 60,
                q.pred_1_3_cm3,
                s=14,
                color=colors[name],
                marker="x",
                label=name,
            )
        ax.axhline(0, color=".6", lw=0.6)
        ax.set(
            title="Candidate inlet · nominal 1–3 nm",
            xlabel="Minutes from event start",
            ylabel="Number (cm⁻³)",
        )
        if i == 0:
            ax.legend(fontsize=7)
        ax = axs[2, i]
        z = frames["process-spectrum"].query("day == @day")
        observed = z.drop_duplicates("hour")
        ax.scatter(
            observed.hour - onset,
            observed.measured_cm3,
            s=18,
            c="black",
            label="Figure 3b stock",
        )
        for name in ("no uptake", "1 nm supply", "5 nm supply"):
            q = z[z.model == name]
            ax.plot(
                q.hour - onset,
                q.pred_cm3,
                ".-",
                ms=3,
                color=colors[name],
                label=name,
            )
        ax.set(
            yscale="log",
            title="Held-out 12–25 nm stock",
            xlabel="Hours from event start",
            ylabel="Number (cm⁻³)",
        )
        if i == 1:
            ax.legend(fontsize=7)
    fig.savefig(
        ROOT / "docs/images/sea2cloud-events/process-comparison.png", dpi=160
    )
    plt.close(fig)


if __name__ == "__main__":
    run()
