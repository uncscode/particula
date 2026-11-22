"""Forward characteristics from observed 2.9 nm physical boundary in CLOUD 1833."""

import csv
import itertools
import numpy as np
import particula as par
from attraction import acid_pair
from population import transport
from reference import ROOT, OUT, dump_csv
from zipfile import ZipFile
import sys

sys.path.insert(0, str(ROOT / "companion"))
from extract_run1833 import spectrum

TEMP = 297.8
PRESSURE = 98700.0
RHO = 1500.0
DMASS = (98.079 + 2 * 17.03052 + 18.01528) * 1.66053906660e-27


def load():
    obs = np.genfromtxt(
        ROOT / "companion/donahue1833_observations.csv",
        names=True,
        delimiter=",",
    )
    edges = np.genfromtxt(
        ROOT / "companion/donahue1833_initial_spectrum.csv",
        names=True,
        delimiter=",",
    )
    return obs, edges


def replay(model, gas_scale, dt, wall_scale, coag, low=2.9):
    obs, init = load()
    time = obs["elapsed_from_first_spectrum_h"] * 3600
    acid = obs["H2SO4_UFRA_cm3"].copy()
    valid = np.isfinite(acid)
    acid = np.interp(time, time[valid], acid[valid]) * gas_scale
    grid = np.geomspace(2.9, 500, 350)
    diffusion = 8.0e-6 * (TEMP / 293.15) ** 1.75
    if model in ["gas_transfer", "finite_hard", "vdw", "vdw_wet"]:
        if model == "gas_transfer":
            k = transport(grid, TEMP, 0.098079, diffusion)
        else:
            # The transfer experiment varies the growth operator only up to
            # 10 nm mobility. Use the identical gas-transfer law above it.
            g, p = acid_pair(
                TEMP,
                PRESSURE,
                6.4e-20 if model.startswith("vdw") else 0.0,
                1.0,
                RHO,
                10.0,
                141,
            )
            k = transport(grid, TEMP, 0.098079, diffusion)
            active = grid <= 9.7
            k[active] = np.interp(grid[active], g, p)
        # nm/s per molecule cm^-3, NH3 in considerable excess under this warm run
        gr = k * 1e6 * DMASS / (np.pi / 2 * RHO * (grid * 1e-9) ** 2) * 1e9
    else:
        gr = (2.68 * (grid + 0.3) ** -1.27 + 0.81) / 3.6e10
    above = grid > 9.7
    gr[above] = (
        transport(grid[above], TEMP, 0.098079, diffusion)
        * 1e6
        * DMASS
        / (np.pi / 2 * RHO * (grid[above] * 1e-9) ** 2)
        * 1e9
    )
    if model == "vdw_wet":
        # Acid inventory in an equilibrated droplet at 58% RH. Differentiate
        # it with respect to diameter so pre-existing salt also takes up water.
        xwater = 0.58 * np.exp(
            -4 * 0.079 * 0.01801528 / (1000 * 8.314462618 * TEMP * grid * 1e-9)
        )
        waters = 3 * xwater / (1 - xwater)
        unit = (98.079 + 2 * 17.03052 + 18.01528 * waters) * 1.66053906660e-27
        acid_per_drop = 1350 * np.pi / 6 * (grid * 1e-9) ** 3 / unit
        gr = k * 1e6 / np.gradient(acid_per_drop, grid, edge_order=2)
    kd = 0.0001754037
    dp = par.particles.get_diffusion_coefficient_via_system_state(
        grid * 0.5e-9, TEMP, PRESSURE
    )
    kw = 0.002 * np.sqrt(dp / diffusion) * wall_scale
    high = 10 ** init["upper_log10_bin_edge_nm"][-1]
    diam = []
    n = []
    for r in init:
        le = max(r["lower_log10_bin_edge_nm"], np.log10(low))
        ue = r["upper_log10_bin_edge_nm"]
        if le >= ue:
            continue
        e = np.linspace(le, ue, 17)
        diam.extend(10 ** ((e[1:] + e[:-1]) / 2))
        n.extend(np.diff(e) * r["dN_dlog10dp_cm3"])
    diam = np.array(diam)
    n = np.array(n)
    with ZipFile(ROOT / "companion/donahue2026-figure-data.zip") as z:
        _, dbin, ed, measured = spectrum(z)
    # Conditional scavenging test using observed collector concentrations.
    radius = np.r_[grid, dbin] * 0.5e-9
    K = par.dynamics.get_brownian_kernel_via_system_state(
        radius, 4 * np.pi / 3 * radius**3 * RHO, TEMP, PRESSURE
    )
    n_at_edge = np.array(
        [
            np.interp(np.log(low), np.log(dbin), v)
            for v in measured[:, : len(time)].T
        ]
    )
    g0 = np.interp(low, grid, gr)
    cs = K[: len(grid), len(grid) :] @ (
        measured[:, : len(time)] * np.diff(ed)[:, None] * 1e6
    )
    series = []
    emitted = 0.0
    removed = 0.0
    initial = n.sum()
    for idx, t in enumerate(time):
        sel = (diam >= 9.7) & (diam <= high)
        series.append(
            dict(
                model=model,
                gas_scale=gas_scale,
                dt_s=dt,
                wall_scale=wall_scale,
                coag=coag,
                hour=t / 3600,
                pred_N10_cm3=n[sel].sum(),
                obs_N10_cm3=obs["derived_N_ge9p7phys_cm3"][idx],
                boundary_nm=low,
                pred_N_above_boundary=n[diam <= high].sum(),
            )
        )
        if idx == len(time) - 1:
            break
        ns = int(np.ceil((time[idx + 1] - t) / dt))
        h = (time[idx + 1] - t) / ns
        for j in range(ns):
            w = (j + 0.5) / ns
            ac = (1 - w) * acid[idx] + w * acid[idx + 1]
            nlog = (1 - w) * n_at_edge[idx] + w * n_at_edge[idx + 1]
            rate = g0 * ac * nlog / (np.log(10) * low)

            def move(d):
                mid = d + 0.5 * h * np.interp(d, grid, gr) * ac
                return h * np.interp(mid, grid, gr) * ac

            dd = move(diam)
            hazard = kw + kd
            if coag:
                hazard = hazard + cs[:, idx] * (1 - w) + cs[:, idx + 1] * w
            newn = n * np.exp(-h * np.interp(diam + dd * 0.5, grid, hazard))
            removed += (n - newn).sum()
            diam = np.r_[diam + dd, low + 0.5 * h * g0 * ac]
            birth = rate * h
            newborn = birth * np.exp(-0.5 * h * np.interp(low, grid, hazard))
            emitted += birth
            removed += birth - newborn
            n = np.r_[newn, newborn]
    assert abs(initial + emitted - removed - n.sum()) < 1e-7 * max(1.0, emitted)
    return series


if __name__ == "__main__":
    series = []
    summary = []
    for model in ["gas_transfer", "finite_hard", "vdw", "vdw_wet", "warm_law"]:
        for g, dt, w, c in [
            (1.0, 30, 1.0, 0),
            (1.15, 30, 1.0, 0),
            (1.05, 30, 1.0, 0),
            (1.25, 30, 1.0, 0),
            (1.15, 15, 1.0, 0),
            (1.15, 30, 0.5, 0),
            (1.15, 30, 2.0, 0),
            (1.15, 30, 1.0, 1),
        ]:
            for low in (
                [2.9, 5.0, 7.1]
                if (g, dt, w, c) == (1.15, 30, 1.0, 0)
                else [2.9]
            ):
                s = replay(model, g, dt, w, c, low)
                series.extend(s)
                p = np.array([r["pred_N10_cm3"] for r in s])
                o = np.array([r["obs_N10_cm3"] for r in s])
                h = np.array([r["hour"] for r in s])
                summary.append(
                    dict(
                        model=model,
                        boundary_nm=low,
                        gas_scale=g,
                        dt_s=dt,
                        wall_scale=w,
                        coag=c,
                        predicted_peak_cm3=p.max(),
                        observed_peak_cm3=o.max(),
                        observed_peak_h=h[o.argmax()],
                        predicted_peak_h=h[p.argmax()],
                        median_abs_error_cm3=np.median(abs(p - o)),
                        rmse_cm3=np.mean((p - o) ** 2) ** 0.5,
                        area_ratio=np.trapezoid(p, h) / np.trapezoid(o, h),
                        end_ratio=p[-1] / o[-1],
                    )
                )
    dump_csv(OUT / "warm_1833.csv", series)
    dump_csv(OUT / "warm_scores.csv", summary)
