"""Finite-time CCN growth at prescribed water supersaturation."""

# The assertions qualify numerical results before publication.
# ruff: noqa: S101
import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import particula as par
import scipy
from scipy.constants import R
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtri, roots_legendre

OUT = Path.cwd() / ".artifacts" / "ccn-retrieval"
T, RHO, MW, SIGMA = 298.15, 1000.0, 0.01801528, 0.072
A = 2 * par.particles.get_kelvin_radius(SIGMA, RHO, MW, T)
# Representative 298 K / one-atmosphere transport properties in SI units.
PRESSURE, ES, DV, KA, CP, MA, LV = (
    101325.0,
    3169.0,
    2.5e-5,
    0.026,
    1005.0,
    0.02897,
    2.442e6,
)
AIR_DENSITY = PRESSURE * MA / (R * T)
EXPOSURES = np.unique(
    np.r_[
        0.0, np.geomspace(0.03, 20.0, 61), 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0
    ]
)
PLATEAUS = (0.2, 0.4, 0.6, 0.63, 0.65, 0.67, 0.7, 1.0)
POPULATIONS = {
    "Lab": (1.05, ((0.03, 1.0),)),
    "External": (1.55, ((0.03, 0.7), (0.61, 0.3))),
    "Internal": (1.55, ((0.204, 1.0),)),
}


def revision():
    """Return the checked-out Git revision recorded with the results."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record the study revision.")
    return subprocess.check_output(  # noqa: S603 -- fixed Git arguments
        [git, "rev-parse", "HEAD"], text=True
    ).strip()


def seq(wet, dry, k):
    """Return equilibrium saturation ratio with the Particula Kelvin term."""
    water = (wet / dry) ** 3 - 1.0
    return water / (water + k) * par.particles.get_kelvin_term(wet / 2, A / 2)


def bisect_vector(residual, lo, hi):
    """Increasing residual with a bracket for every diameter."""
    assert np.all(residual(lo) <= 0) and np.all(residual(hi) >= 0)
    for _ in range(55):
        mid = (lo + hi) / 2
        lower = residual(mid) < 0
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    return (lo + hi) / 2


def critical(dry, k):
    """Find critical wet diameter and saturation ratio at each dry size."""
    b = A / dry

    def derivative_root(x):
        return b * (x**3 - 1) * (x**3 - 1 + k) - 3 * k * x**4

    ratio = bisect_vector(
        derivative_root, np.ones_like(dry), np.full_like(dry, 256.0)
    )
    wet = dry * ratio
    return wet, seq(wet, dry, k)


def shortcut(dry, sc_ratio):
    """Infer κ from dry diameter and critical saturation ratio with Eq. 10."""
    return 4 * (A / dry) ** 3 / (27 * np.log(sc_ratio) ** 2)


def resistance(wet, alpha):
    """Four times the total denominator of the diameter growth law, s/m²."""
    d_eff = DV / (
        1 + 2 * DV * np.sqrt(2 * np.pi * MW / (R * T)) / (alpha * wet)
    )
    k_eff = KA / (
        1
        + 2 * KA * np.sqrt(2 * np.pi * MA / (R * T)) / (wet * AIR_DENSITY * CP)
    )
    return RHO * R * T / (ES * d_eff * MW) + LV * RHO / (k_eff * T) * (
        LV * MW / (R * T) - 1
    )


@lru_cache(maxsize=None)
def nodes(order):
    """Return transformed quadrature nodes and weights on the unit interval."""
    x, w = roots_legendre(order)
    # Cluster nodes near both endpoints to resolve peaks near marginal
    # activation.
    theta = (x + 1) * np.pi / 4
    return np.sin(theta) ** 2, w * np.pi / 4 * np.sin(2 * theta)


def transit(start, end, dry, k, forcing, alpha, order):
    """Integrate the time to grow between two wet diameters in seconds."""
    z, weight = nodes(order)
    width = (end - start)[:, None]
    wet = start[:, None] + width * z
    drive = forcing - seq(wet, dry[:, None], k[:, None])
    assert np.all(drive > 0)
    return np.sum(
        weight * width * wet * resistance(wet, alpha) / (4 * drive), axis=1
    )


def arrival(target, initial, peak, dry, k, forcing, alpha, order):
    """Integrate dt/dD up to the target or return infinity at a barrier."""
    target = np.broadcast_to(target, dry.shape)
    barrier = np.clip(peak, initial, np.maximum(initial, target))
    reachable = (target > initial) & (forcing > seq(barrier, dry, k))
    times = np.full(dry.shape, np.inf)
    times[target <= initial] = 0.0
    take = np.flatnonzero(reachable)
    args = dry[take], k[take], forcing, alpha, order
    times[take] = transit(initial[take], barrier[take], *args) + transit(
        barrier[take], target[take], *args
    )
    return times


def thresholds(dry, true_k, method, forcing, alpha, order):
    """Calculate times to the critical diameter and optical counting gate."""
    k = np.full_like(dry, true_k)
    peak, sc = critical(dry, k)
    if method == "shortcut":
        k = shortcut(dry, sc)
        peak, sc = critical(dry, k)
    initial = bisect_vector(lambda wet: seq(wet, dry, k) - 0.95, dry, peak)
    assert np.all(initial > dry)
    args = initial, peak, dry, k, forcing, alpha, order
    return arrival(peak, *args), arrival(1e-6, *args), forcing > sc


def simulate(n, order, alpha, plateaus):
    """Calculate time-resolved counts for lab and mixed populations."""
    rows = []
    q = ndtri((np.arange(n) + 0.5) / n)
    for name, (gsd, components) in POPULATIONS.items():
        dry = 100.0e-9 * np.exp(np.log(gsd) * q)
        assert np.isclose(sum(f for _, f in components), 1.0)
        for ss in plateaus:
            for method in ("input", "shortcut"):
                crossed = np.zeros(len(EXPOSURES))
                optical = np.zeros_like(crossed)
                eligible = 0.0
                for k, fraction in components:
                    activation, detection, possible = thresholds(
                        dry, k, method, 1 + ss / 100, alpha, order
                    )
                    assert np.array_equal(np.isfinite(activation), possible)
                    crossed += fraction * np.mean(
                        activation[:, None] <= EXPOSURES, axis=0
                    )
                    optical += fraction * np.mean(
                        detection[:, None] <= EXPOSURES, axis=0
                    )
                    eligible += fraction * possible.mean()
                assert np.all(np.diff(optical) >= 0) and np.all(
                    np.diff(crossed) >= 0
                )
                assert np.all(crossed <= eligible + 1e-12) and np.all(
                    optical <= 1 + 1e-12
                )
                for t, a, o in zip(EXPOSURES, crossed, optical, strict=True):
                    rows.append(
                        dict(
                            population=name,
                            method=method,
                            alpha=alpha,
                            ss_pct=ss,
                            seconds=t,
                            critical_fraction=a,
                            optical_fraction=o,
                            equilibrium_fraction=eligible,
                        )
                    )
    return rows


def fraction_error(a, b):
    """Return the largest absolute difference between paired count fractions."""
    return max(
        abs(x[key] - y[key]) * 100
        for x, y in zip(a, b, strict=True)
        for key in (
            "critical_fraction",
            "optical_fraction",
            "equilibrium_fraction",
        )
    )


def lab_midpoints():
    # The lab has one composition. For monotone detection vs dry size, the
    # median dry diameter gives the continuous distribution's 50% count point.
    """Find the optical midpoint and its apparent κ at each exposure time."""
    dry = np.array([100.0e-9])
    _, equilibrium = critical(dry, 0.03)
    output = []
    for seconds in (1.0, 2.0, 5.0, 10.0, 20.0):

        def residual(ss, seconds=seconds):
            _, detection, _ = thresholds(
                dry, 0.03, "input", 1 + ss / 100, 1.0, 96
            )
            return detection[0] - seconds

        ss = brentq(residual, 100 * (equilibrium[0] - 1) + 1e-9, 5.0)
        sample = dry * np.exp(
            np.log(1.05) * ndtri((np.arange(512) + 0.5) / 512)
        )
        _, detect, _ = thresholds(sample, 0.03, "input", 1 + ss / 100, 1.0, 96)
        assert np.array_equal(detect <= seconds, sample > dry[0])
        full_k = brentq(
            lambda k, ss=ss: critical(dry, k)[1][0] - (1 + ss / 100),
            0.00001,
            0.5,
        )
        output.append(
            dict(
                seconds=seconds,
                optical_ss50_pct=ss,
                full_equilibrium_inversion=full_k,
                shortcut_equilibrium_inversion=shortcut(dry[0], 1 + ss / 100),
            )
        )
    return output


def time_check():
    # Independent stiff, direct-in-time integration of the same growth law.
    """Compare the growth-time integral with direct stiff ODE integration."""
    errors = []
    for size, k, ss in (
        (100.0, 0.03, 0.65),
        (100.0, 0.204, 0.4),
        (150.0, 0.61, 0.4),
    ):
        d = np.array([size * 1e-9])
        p, _ = critical(d, k)
        init = bisect_vector(lambda wet, d=d, k=k: seq(wet, d, k) - 0.95, d, p)

        def rate(t, wet_um, d=d, k=k, ss=ss):
            wet = wet_um * 1e-6
            return (
                1e6
                * 4
                * ((1 + ss / 100) - seq(wet, d, k))
                / (wet * resistance(wet, 1.0))
            )

        def event(t, wet_um):
            return wet_um[0] - 1.0

        event.terminal = True
        for max_step, tol in ((0.05, 1e-7), (0.025, 1e-9)):
            sol = solve_ivp(
                rate,
                (0, 100),
                init * 1e6,
                method="Radau",
                rtol=tol,
                atol=tol * 1e-3,
                max_step=max_step,
                events=event,
            )
            assert sol.success and len(sol.t_events[0]) == 1
            ref = thresholds(d, k, "input", 1 + ss / 100, 1.0, 96)[1][0]
            errors.append(abs(sol.t_events[0][0] / ref - 1) * 100)
    return max(errors)


def save_csv(name, rows):
    """Save a sequence of records using the first record’s column order."""
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot():
    """Plot the time dependence of the modeled optical counts."""
    with (OUT / "dynamic-counts.csv").open() as f:
        data = list(csv.DictReader(f))
    colors = {"Lab": "#007179", "External": "#934c1d", "Internal": "#44559e"}
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.3), sharey=True, layout="constrained"
    )
    for ax, names, ss, title in zip(
        axes,
        [["Lab"], ["External", "Internal"]],
        [0.65, 0.4],
        ["Narrow 100 nm laboratory aerosol", "Ambient mixing-state comparison"],
        strict=True,
    ):
        for name in names:
            for method, style in [("input", "-"), ("shortcut", "--")]:
                rows = [
                    r
                    for r in data
                    if r["population"] == name
                    and float(r["alpha"]) == 1.0
                    and r["method"] == method
                    and float(r["ss_pct"]) == ss
                    and float(r["seconds"]) >= 0.05
                ]
                coefficient = "input κ" if method == "input" else "retrieved κ"
                ax.plot(
                    [float(r["seconds"]) for r in rows],
                    [100 * float(r["optical_fraction"]) for r in rows],
                    color=colors[name],
                    linestyle=style,
                    linewidth=2,
                    label=f"{name} · {coefficient}",
                )
        ax.set(
            xscale="log",
            xlabel="Exposure time (s)",
            title=f"{title}\n{ss:.2f}% supersaturation",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Number reaching 1 µm wet diameter (%)")
    axes[0].set_ylim(-2, 102)
    fig.savefig(OUT / "dynamic-counts.png", dpi=165)
    plt.close(fig)


def main():
    """Run the laboratory calculation and record the numerical checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if args.replot:
        plot()
        return
    start = time.perf_counter()
    assert Path(par.__file__).resolve().is_relative_to(Path.cwd() / "particula")
    # Smoke before the complete sweep.
    simulate(32, 24, 1.0, (0.65,))
    production = simulate(1024, 64, 1.0, PLATEAUS)
    size_error = fraction_error(production, simulate(512, 64, 1.0, PLATEAUS))
    integration_error = fraction_error(
        production, simulate(1024, 32, 1.0, PLATEAUS)
    )
    slower = simulate(1024, 64, 0.1, (0.4, 0.65))
    size_error = max(
        size_error, fraction_error(slower, simulate(512, 64, 0.1, (0.4, 0.65)))
    )
    integration_error = max(
        integration_error,
        fraction_error(slower, simulate(1024, 32, 0.1, (0.4, 0.65))),
    )
    time_error = time_check()
    assert max(size_error, integration_error) < 1.0 and time_error < 0.1
    midpoints = lab_midpoints()
    save_csv("dynamic-counts.csv", production + slower)
    save_csv("dynamic-lab-inversion.csv", midpoints)
    plot()
    checks = dict(
        size_refinement_pp=size_error,
        quadrature_refinement_pp=integration_error,
        ode_crosscheck_relative_pct=time_error,
        wall_seconds=time.perf_counter() - start,
        revision=revision(),
        python=platform.python_version(),
        particula=par.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        backend="CPU",
    )
    (OUT / "dynamic-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(checks)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
