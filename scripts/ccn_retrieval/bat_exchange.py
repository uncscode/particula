"""Coupled liquid water and semivolatile organic exchange."""

# The assertions qualify numerical results before publication.
# ruff: noqa: S101
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
import warp as wp
from ambient_gpu import R0, R1
from dynamic import (
    MW,
    OUT,
    RHO,
    SIGMA,
    A,
    R,
    T,
    resistance,
    revision,
    save_csv,
    shortcut,
)
from matplotlib import style as mplstyle
from particula import plotting
from particula.activity import bat_activity_coefficients
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csr_matrix, diags
from scipy.special import ndtr, ndtri

RHO_O, M_O, D_O, CSTAR = 1243.493333, 0.200, 6e-6, 10.0
K_IDEAL = RHO_O * MW / (RHO * M_O)
A_ORG = 4 * SIGMA * M_O / (R * T * RHO_O)
ORG_LENGTH = 2 * D_O * np.sqrt(2 * np.pi * M_O / (R * T))
DIAM = np.unique(
    np.r_[
        np.geomspace(6, 550, 1101),
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        95.0,
        100.0,
        125.0,
        150.0,
        200.0,
    ]
)
MEDIANS = np.unique(
    np.r_[
        np.geomspace(10, 200, 41), 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0
    ]
)
SS = np.unique(
    np.round(
        np.r_[
            np.geomspace(0.05, 1.4, 43),
            0.2,
            0.3,
            0.32,
            0.34,
            0.35,
            0.36,
            0.38,
            0.4,
            0.65,
            1.0,
        ],
        9,
    )
)
TIMES = np.array([1.0, 2.0, 5.0, 10.0])
CASES = [(model, c) for model in ("ideal", "BAT") for c in (0.0, 10.0, 30.0)]
F = wp.float64
ONE = wp.constant(F(1.0))
THREE = wp.constant(F(3.0))
CO = wp.constant(F(12 * D_O * 1e-9 / RHO_O))
ELL = wp.constant(F(ORG_LENGTH))
KW = wp.constant(F(A))
KO = wp.constant(F(A_ORG))
MIX = wp.constant(F(K_IDEAL))


def activities(x, model):
    """Return water and organic activities for the selected liquid model."""
    if model == "ideal":
        return 1 - x, x
    return bat_activity_coefficients(MW / M_O, x, 0.5, RHO_O)[:2]


def tables(model, n=8192):
    """Tabulate binary activities on the organic mole-fraction grid."""
    grid = np.linspace(0, 1, n + 1)
    return np.column_stack(activities(grid, model))


def initial_water(d, model):
    # Return liquid water volume divided by initial organic volume.
    """Solve for the initial water volume at 95% relative humidity."""
    lo = np.full(d.shape, -14.0)
    hi = np.full(d.shape, 9.0)

    def residual(logw):
        w = np.exp(logw)
        aw, _ = activities(K_IDEAL / (K_IDEAL + w), model)
        return aw * np.exp(A / (d * (1 + w) ** (1 / 3))) - 0.95

    assert np.all(residual(lo) < 0) and np.all(residual(hi) > 0)
    for _ in range(52):
        mid = (lo + hi) / 2
        low = residual(mid) < 0
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return np.exp((lo + hi) / 2)


@wp.func
def rhs(y: wp.vec2d, d: F, s: F, c: F, tab: wp.array2d(dtype=F)):
    # State comprises ln(Vwater/Vorg,0) and ln(Vorg/Vorg,0).
    """Evaluate coupled water and organic log-volume rates."""
    w = wp.exp(y[0])
    o = wp.exp(y[1])
    wet = d * wp.pow(w + o, ONE / THREE)
    x = MIX * o / (MIX * o + w)
    n = tab.shape[0] - 1
    p = x * F(n)
    i = wp.clamp(int(p), 0, n - 1)
    f = p - F(i)
    aw = tab[i, 0] + f * (tab[i + 1, 0] - tab[i, 0])
    ao = tab[i, 1] + f * (tab[i + 1, 1] - tab[i, 1])
    d3 = d * d * d
    rw = (
        F(12.0)
        * wet
        * wet
        * (s - aw * wp.exp(KW / wet))
        / ((wet * R0 + R1) * d3 * w)
    )
    ro = F(0.0)
    if c > F(0.0):
        ro = (
            CO
            * wet
            * (c - F(10.0) * ao * wp.exp(KO / wet))
            / ((ONE + ELL / wet) * d3 * o)
        )
    return wp.vec2d(rw, ro)


@wp.func
def step(y: wp.vec2d, h: F, d: F, s: F, c: F, tab: wp.array2d(dtype=F)):
    # L-stable, linearly implicit Euler increment; two half steps give the
    # local error estimate and a Richardson-corrected accepted state.
    """Advance the coupled state with one linearly implicit step."""
    eps = F(2e-6)
    f = rhs(y, d, s, c, tab)
    a = (rhs(y + wp.vec2d(eps, F(0.0)), d, s, c, tab) - f) / eps
    b = (rhs(y + wp.vec2d(F(0.0), eps), d, s, c, tab) - f) / eps
    aa = ONE - h * a[0]
    bb = -h * b[0]
    cc = -h * a[1]
    dd = ONE - h * b[1]
    det = aa * dd - bb * cc
    return h * wp.vec2d(dd * f[0] - bb * f[1], aa * f[1] - cc * f[0]) / det


@wp.kernel
def trajectories(
    dry: wp.array(dtype=F),
    w0: wp.array(dtype=F),
    forcing: wp.array(dtype=F),
    tab: wp.array2d(dtype=F),
    times: wp.array(dtype=F),
    gas: F,
    tol: F,
    diameter: wp.array3d(dtype=F),
    organic: wp.array3d(dtype=F),
    status: wp.array2d(dtype=int),
):
    """Integrate prescribed-vapor trajectories to each observation time."""
    j, i = wp.tid()
    d = dry[i]
    s = forcing[j]
    y = wp.vec2d(wp.log(w0[i]), F(0.0))
    clock = F(0.0)
    h = F(1e-6)
    tries = int(0)
    evaporated = int(0)
    for m in range(times.shape[0]):
        while clock < times[m] and tries < 40000 and evaporated == 0:
            if d * wp.pow(wp.exp(y[0]) + wp.exp(y[1]), ONE / THREE) < F(2e-9):
                evaporated = 1
                break
            tries += 1
            h = wp.min(h, times[m] - clock)
            a = step(y, h, d, s, gas, tab)
            b = step(y, h * F(0.5), d, s, gas, tab)
            if wp.max(wp.abs(b[0]), wp.abs(b[1])) > F(0.4):
                h = h * F(0.25)
                continue
            b = b + step(y + b, h * F(0.5), d, s, gas, tab)
            err = wp.max(wp.abs(a[0] - b[0]), wp.abs(a[1] - b[1])) / tol
            if not wp.isfinite(err):
                h = h * F(0.25)
                continue
            if err <= ONE:
                y = y + F(2.0) * b - a
                clock = clock + h
            h = wp.min(
                F(0.15),
                h
                * wp.clamp(
                    F(0.88) / wp.sqrt(wp.max(F(1e-12), err)), F(0.2), F(3.0)
                ),
            )
        diameter[j, i, m] = (
            d * wp.pow(wp.exp(y[0]) + wp.exp(y[1]), ONE / THREE) * F(1e6)
        )
        organic[j, i, m] = wp.exp(y[1])
        if evaporated == 1:
            diameter[j, i, m] = F(0.0)
            organic[j, i, m] = F(0.0)
    status[j, i] = tries


def cuda(d_nm, ss, model, gas, tol, table_n=8192):
    """Run the binary exchange calculation for the selected dry sizes."""
    dev = wp.get_device("cuda:0")
    assert dev.is_cuda

    def upload(a):
        return wp.array(np.ascontiguousarray(a), dtype=F, device=dev)

    shape = (len(ss), len(d_nm), len(TIMES))
    wet, org = [wp.empty(shape, dtype=F, device=dev) for _ in range(2)]
    status = wp.empty(shape[:2], dtype=int, device=dev)
    d = d_nm * 1e-9
    inputs = [
        upload(d),
        upload(initial_water(d, model)),
        upload(1 + ss / 100),
        upload(tables(model, table_n)),
        upload(TIMES),
        F(gas),
        F(tol),
    ]
    wp.launch(
        trajectories, shape[:2], inputs=[*inputs, wet, org, status], device=dev
    )
    n = status.numpy()
    assert n.max() < 40000, (model, gas, n.max(), np.argwhere(n == n.max())[:4])
    wet, org = wet.numpy(), org.numpy()
    assert (
        np.all(np.isfinite(wet))
        and np.all(wet >= 0)
        and np.all(np.isfinite(org))
        and np.all(org >= 0)
    )
    if gas == 0:
        assert np.array_equal(org, np.ones_like(org))
    return wet, org


def cpu(d_nm, model, gas, ss, number=0.0, tol=1e-7):
    """Integrate with direct activities and optional finite organic supply."""
    d = d_nm * 1e-9
    n = len(d)
    initial = initial_water(d, model)
    vol = d**3 * np.pi / 6
    weights = (
        vol * RHO_O * number * 1e6 * 1e9 / n
    )  # initial organic µg/m³ in each cohort

    def rates(t, y):
        w, o = np.exp(y[:n]), np.exp(y[n : 2 * n])
        wet = d * np.cbrt(w + o)
        aw, ao = activities(K_IDEAL * o / (K_IDEAL * o + w), model)
        rw = (
            12
            * wet
            * (1 + ss / 100 - aw * np.exp(A / wet))
            / (resistance(wet, 1.0) * d**3 * w)
        )
        ro = np.zeros_like(o)
        if gas:
            ro = (
                12
                * D_O
                * wet
                * (y[-1] - CSTAR * ao * np.exp(A_ORG / wet))
                * 1e-9
                / ((1 + ORG_LENGTH / wet) * RHO_O * d**3 * o)
            )
        return np.r_[rw, ro, -weights @ (o * ro)]

    def jacobian(t, y):
        # Cohorts couple only through the gas concentration. Perturb all local
        # water or organic log volumes in one call, then form the budget row.
        base = rates(t, y)
        eps = 1e-7
        columns = []
        for indices in (slice(0, n), slice(n, 2 * n), -1):
            yp = y.copy()
            yp[indices] += eps
            columns.append((rates(t, yp)[: 2 * n] - base[: 2 * n]) / eps)
        u, v, g = columns
        masses = weights * np.exp(y[n : 2 * n])
        return bmat(
            [
                [diags(u[:n]), diags(v[:n]), csr_matrix(g[:n, None])],
                [diags(u[n:]), diags(v[n:]), csr_matrix(g[n:, None])],
                [
                    csr_matrix(-masses * u[n:]),
                    csr_matrix(-masses * (v[n:] + base[n : 2 * n])),
                    csr_matrix([[-masses @ g[n:]]]),
                ],
            ],
            format="csc",
        )

    sol = solve_ivp(
        rates,
        (0, TIMES[-1]),
        np.r_[np.log(initial), np.zeros(n), gas],
        method="BDF",
        t_eval=TIMES,
        rtol=tol,
        atol=tol * 1e-2,
        jac=jacobian,
        max_step=0.12,
    )
    assert sol.success, sol.message
    w, o = np.exp(sol.y[:n]), np.exp(sol.y[n : 2 * n])
    wet = d[:, None] * np.cbrt(w + o) * 1e6
    closure = np.max(
        np.abs(sol.y[-1] + weights @ o - (gas + weights.sum()))
    ) / (gas + weights.sum() + 1e-30)
    assert sol.y[-1].min() >= 0 and closure < 1e-5
    return wet, o, sol.y[-1], closure


def concentration_weights(edges, med, width, moment):
    """Integrate a lognormal number or volume moment over size bins."""
    logw = np.log(width)
    return np.diff(ndtr((np.log(edges / med) - moment * logw**2) / logw))


def reduce_counts(d, wet, organic, median, width):
    # Integrate counts across each threshold crossing by interpolating in
    # log dry
    # size. Integrate organic mass with exact lognormal dry volume weights.
    """Weight optical threshold crossings and organic gains across sizes."""
    edges = np.r_[d[0], np.sqrt(d[:-1] * d[1:]), d[-1]]
    weights = concentration_weights(edges, median, width, 3)
    lo, hi = ndtr(np.log(d[[0, -1]] / median) / np.log(width))
    keep = wet >= 1
    if keep.any() and not keep.all():
        switches = np.where(keep[:-1] != keep[1:])[0]
        i = switches
        cuts = np.exp(
            np.log(d[i])
            + np.log(d[i + 1] / d[i]) * (1 - wet[i]) / (wet[i + 1] - wet[i])
        )
        bounds = np.r_[lo, ndtr(np.log(cuts / median) / np.log(width)), hi]
        fraction = np.diff(bounds)[
            (np.arange(len(bounds) - 1) % 2) == (0 if keep[0] else 1)
        ].sum() / (hi - lo)
    else:
        fraction = float(keep.all())
    return float(100 * fraction), float(
        100 * (weights @ organic / weights.sum() - 1)
    )


@mplstyle.context("particula.default")
def plot():
    """Plot the scan curves and matched count differences."""
    with (OUT / "bat-lab-scan.csv").open() as file:
        rows = list(csv.DictReader(file))
    fig, axes = plotting.subplots(1, 2)
    colors = {0.0: "#444d56", 10.0: "#087e83", 30.0: "#aa5729"}
    for model, gas in CASES:
        r = [
            x
            for x in rows
            if x["model"] == model
            and float(x["gas_ug_m3"]) == gas
            and float(x["median_nm"]) == 100
            and float(x["gsd"]) == 1.1
            and float(x["seconds"]) == 5
        ]
        for ax, key in zip(
            axes, ("optical_pct", "organic_mass_change_pct"), strict=True
        ):
            ax.plot(
                [float(x["ss_pct"]) for x in r],
                [float(x[key]) for x in r],
                color=colors[gas],
                ls="-" if model == "BAT" else "--",
                label=f"{model} · "
                + (f"{gas:g} µg/m³" if gas else "fixed organic"),
            )
            ax.set(xlabel="Water supersaturation (%)", xlim=(0.05, 0.8))
    axes[0].set_ylabel("Particles reaching 1 µm (%)")
    axes[1].set_ylabel("Change in particle organic mass (%)")
    plotting.legend(axes[0])
    axes[1].grid(alpha=0.2)
    axes[0].grid(alpha=0.2)
    fig.suptitle(
        "Water + a miscible organic  |  100 nm dry median, GSD 1.10, 5 s"
    )
    plotting.fit(fig)
    fig.savefig(OUT / "bat-cocondensation.png", dpi=300)
    plt.close(fig)

    # Compare one physical assumption at a time on the common size/forcing grid.
    def surface(model, gas):
        values = [
            r
            for r in rows
            if r["model"] == model
            and float(r["gas_ug_m3"]) == gas
            and float(r["gsd"]) == 1.1
            and float(r["seconds"]) == 5.0
        ]
        return (
            np.asarray([float(r["optical_pct"]) for r in values])
            .reshape(len(MEDIANS), len(SS))
            .T
        )

    fixed = surface("BAT", 0.0)
    contrasts = (fixed - surface("ideal", 0.0), surface("BAT", 10.0) - fixed)
    fig, axes = plotting.subplots(1, 2, sharey=True)
    for ax, delta, title in zip(
        axes,
        contrasts,
        (
            "BAT minus ideal, organic mass fixed",
            "Organic uptake minus fixed mass, both BAT",
        ),
        strict=True,
    ):
        extent = max(1, np.ceil(abs(delta).max()))
        ax.grid(False)
        g = ax.pcolormesh(
            MEDIANS,
            SS,
            delta,
            cmap=plotting.colormap("difference"),
            vmin=-extent,
            vmax=extent,
            shading="nearest",
        )
        ax.plot(100, 0.35, "+", color="black", ms=10, mew=1.4)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Median dry diameter (nm)",
            title=title,
            xlim=(10, 200),
            ylim=(0.05, 1.4),
        )
        ax.set_xticks(
            [10, 20, 50, 100, 200], labels=["10", "20", "50", "100", "200"]
        )
        ax.set_yticks(
            [0.05, 0.1, 0.2, 0.35, 0.7, 1.4],
            labels=["0.05", "0.1", "0.2", "0.35", "0.7", "1.4"],
        )
        fig.colorbar(g, ax=ax, label="Count difference (percentage points)")
    axes[0].set_ylabel("Water supersaturation (%)")
    fig.suptitle(
        "Sensitivity of the 1 µm optical count  ·  GSD 1.10, 5 "
        "s, supplied vapor 10 µg/m³"
    )
    plotting.fit(fig)
    fig.savefig(OUT / "bat-design-contrast.png", dpi=300)
    plt.close(fig)


def run():  # noqa: C901 -- scan and verification sequence
    """Generate the binary scans and finite-reservoir comparisons."""
    assert Path(par.__file__).resolve().is_relative_to(Path.cwd() / "particula")
    start = time.perf_counter()
    rows = []
    mids = []
    errors = []
    refinements = []
    size_errors = []
    rng = np.random.default_rng(77)
    x = rng.uniform(size=20001)
    aw, ao = activities(x, "BAT")
    tab = tables("BAT")
    interp = max(
        np.max(abs(np.interp(x, np.linspace(0, 1, len(tab)), tab[:, i]) - a))
        for i, a in enumerate((aw, ao))
    )
    assert interp < 1e-5
    for model, gas in CASES:
        wet, org = cuda(DIAM, SS, model, gas, 3e-5)
        np.savez_compressed(
            OUT / f"bat-states-{model}-{int(gas)}.npz",
            dry_nm=DIAM,
            ss_pct=SS,
            seconds=TIMES,
            wet_um=wet.astype(np.float32),
            organic_ratio=org.astype(np.float32),
        )
        for ss in (0.2, 0.35, 0.65):
            test = np.array([10.0, 50.0, 100.0, 200.0])
            ix = np.searchsorted(DIAM, test)
            j = np.searchsorted(SS, ss)
            reference = cpu(test, model, gas, ss)
            for actual, expected in zip(
                (wet[j, ix], org[j, ix]), reference[:2], strict=True
            ):
                errors.append(100 * np.max(abs(actual / expected - 1)))
        j = np.searchsorted(SS, 0.35)
        w2, o2 = cuda(DIAM, np.array([0.35]), model, gas, 7.5e-6, 16384)
        for t in range(len(TIMES)):
            base = reduce_counts(DIAM, wet[j, :, t], org[j, :, t], 100.0, 1.1)
            ref = reduce_counts(DIAM, w2[0, :, t], o2[0, :, t], 100.0, 1.1)
            refinements.append(abs(base[0] - ref[0]))
        # Horizontal boundary in dry diameter changes rapidly near marginal
        # activation. Include a finer dry mesh for the prespecified comparison.
        fine_d = np.unique(np.r_[DIAM, np.sqrt(DIAM[:-1] * DIAM[1:])])
        wf, of = cuda(fine_d, np.array([0.35]), model, gas, 3e-5)
        for t in range(len(TIMES)):
            base = reduce_counts(DIAM, wet[j, :, t], org[j, :, t], 100.0, 1.1)
            fine = reduce_counts(fine_d, wf[0, :, t], of[0, :, t], 100.0, 1.1)
            size_errors.append(abs(base[0] - fine[0]))
        print(
            model,
            gas,
            "CPU, time and size checks",
            max(errors),
            max(refinements),
            max(size_errors),
            flush=True,
        )
        for med in MEDIANS:
            for width in (1.1, 1.2):
                for it, t in enumerate(TIMES):
                    curve = []
                    for j, ss in enumerate(SS):
                        count, mass = reduce_counts(
                            DIAM, wet[j, :, it], org[j, :, it], med, width
                        )
                        curve.append(count)
                        rows.append(
                            dict(
                                model=model,
                                gas_ug_m3=gas,
                                median_nm=med,
                                gsd=width,
                                ss_pct=ss,
                                seconds=t,
                                optical_pct=count,
                                organic_mass_change_pct=mass,
                            )
                        )
                    if np.any(np.diff(curve) < -0.02):
                        raise ValueError("Nonmonotone scan")
                    if curve[0] < 50 < curve[-1]:
                        ix = np.searchsorted(curve, 50)
                        s50 = np.interp(
                            50, curve[ix - 1 : ix + 1], SS[ix - 1 : ix + 1]
                        )
                        mids.append(
                            dict(
                                model=model,
                                gas_ug_m3=gas,
                                median_nm=med,
                                gsd=width,
                                seconds=t,
                                ss50_pct=s50,
                                ss_lower=SS[ix - 1],
                                ss_upper=SS[ix],
                                apparent_kappa=shortcut(
                                    med * 1e-9, 1 + s50 / 100
                                ),
                            )
                        )
    assert max(errors) < 0.5 and max(refinements) < 1 and max(size_errors) < 1
    save_csv("bat-lab-scan.csv", rows)
    save_csv("bat-optical-midpoints.csv", mids)
    # Ambient organic-rich mode. The common organic gas concentration is now
    # an additional ODE state shared by all cohorts in a single population.
    ambient = []
    closure = []
    finite_refine = []
    for model in ("ideal", "BAT"):
        for gas, number in [
            (0.0, 1000.0),
            (10.0, 0.0),
            (10.0, 1000.0),
            (10.0, 10000.0),
        ]:
            for n in (96, 384):
                d = 95 * 1.5 ** ndtri((np.arange(n) + 0.5) / n)
                w, o, g, res = cpu(d, model, gas, 0.35, number)
                closure.append(res)
                results = []
                for it, t in enumerate(TIMES):
                    count = 100 * np.mean(w[:, it] >= 1.0)
                    mass = 100 * (np.average(o[:, it], weights=d**3) - 1)
                    results.append(count)
                    if n == 384:
                        ambient.append(
                            dict(
                                model=model,
                                initial_gas_ug_m3=gas,
                                particles_cm3=number,
                                reservoir="prescribed"
                                if number == 0
                                else "finite",
                                ss_pct=0.35,
                                seconds=t,
                                optical_pct=count,
                                organic_mass_change_pct=mass,
                                remaining_gas_ug_m3=g[it],
                            )
                        )
                if n == 96:
                    coarse = results
                else:
                    finite_refine.append(
                        np.max(np.abs(np.array(coarse) - results))
                    )
    save_csv("bat-finite-ambient.csv", ambient)
    info = dict(
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        source=par.__version__,
        revision=revision(),
        sampled_cpu_gpu_max_relative_state_pct=max(errors),
        primary_time_refinement_pp=max(refinements),
        primary_size_refinement_pp=max(size_errors),
        bat_table_activity_abs_error=interp,
        finite_organic_budget_relative_error=max(closure),
        ambient_size_refinement_pp=max(finite_refine),
        ss_count=len(SS),
        lab_medians=len(MEDIANS),
        particle_diameters=len(DIAM),
        elapsed_s=time.perf_counter() - start,
    )
    print(info, flush=True)
    (OUT / "bat-checks.json").write_text(json.dumps(info, indent=2) + "\n")
    plot()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    if parser.parse_args().replot:
        plot()
    else:
        run()
