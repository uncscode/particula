"""Equilibrium BAT reference and independent fits to published κ equations."""

# The assertions qualify numerical results before publication.
# ruff: noqa: S101
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula
from bat_exchange import K_IDEAL, TIMES, activities, cpu
from dynamic import OUT, A, revision, save_csv, shortcut
from scipy.optimize import brentq, minimize_scalar

SIZES = np.arange(10.0, 201.0)
TRAIN = (50.0, 75.0, 100.0, 150.0)


def saturation(log_ratio, d, model):
    """Evaluate the fixed-organic equilibrium saturation ratio."""
    ratio = np.exp(log_ratio)
    x = K_IDEAL / (K_IDEAL + np.expm1(3 * log_ratio))
    aw, _ = activities(x, model)
    return aw * np.exp(A / (d * ratio))


def direct_peak(d, model):
    """Maximize the direct binary equilibrium curve over wet diameter."""
    grid = np.linspace(0, np.log(256), 513)
    observed = saturation(grid, d, model)
    i = np.argmax(observed)
    assert 0 < i < len(grid) - 1
    fit = minimize_scalar(
        lambda log_size: -saturation(log_size, d, model),
        bounds=(grid[i - 1], grid[i + 1]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert fit.success
    return 100 * (-fit.fun - 1), np.exp(fit.x)


def threshold(d, k):
    """Return critical water supersaturation in percent for constant κ."""
    b = A / d

    def stationary(x):
        return b * (x**3 - 1) * (x**3 - 1 + k) - 3 * k * x**4

    x = brentq(stationary, 1.0, 512.0)
    return 100 * np.expm1(b / x - np.log1p(k / (x**3 - 1)))


def invert(d, sc):
    """Fit constant κ to a full-equation critical supersaturation."""
    if not threshold(d, 3.0) < sc < threshold(d, 1e-7):
        return np.nan
    return brentq(lambda k: threshold(d, k) - sc, 1e-7, 3.0, xtol=1e-14)


def percent(actual, reference):
    """Return the signed difference relative to the reference in percent."""
    return 100 * (actual / reference - 1)


def read_rows(path):
    """Read a published table into a list of records."""
    with path.open() as f:
        return list(csv.DictReader(f))


def plot(eq):
    """Compare the local κ fits and held-out threshold errors."""
    d = np.array([r["dry_nm"] for r in eq])
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.65), layout="constrained")
    for key, style, label in (
        ("full_k", "-", "Full κ-Köhler fit"),
        ("approx_k", "--", "Eq. 10 shortcut"),
    ):
        axes[0].plot(d, [r[key] for r in eq], style, label=label, lw=2)
    ax = axes[1]
    for key, style, label in (
        ("full_100_sc_error_pct", "-", "Full fit at 100 nm"),
        ("approx_100_sc_error_pct", "--", "Eq. 10 fit → full equation"),
        (
            "eq10_forward_100_sc_error_pct",
            ":",
            "Eq. 10 fit → Eq. 10 prediction",
        ),
    ):
        ax.plot(d, [r[key] for r in eq], style, label=label, lw=2)
    ax.axhline(0, color=".5", lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(
            [10, 20, 30, 50, 75, 100, 150, 200],
            labels=["10", "20", "30", "50", "75", "100", "150", "200"],
        )
        ax.set_xlabel("Dry diameter (nm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Effective κ retrieved from BAT threshold")
    axes[1].set_ylabel("Critical supersaturation error relative to BAT (%)")
    axes[0].set_title("Fit separately at each diameter")
    axes[1].set_title("Use one fitted κ at every diameter")
    fig.suptitle(
        "Equilibrium activation of the miscible BAT organic  ·  "
        "fixed organic mass"
    )
    fig.savefig(OUT / "bat-kappa-fit.png", dpi=180)
    plt.close(fig)


def main():  # noqa: C901 -- scan and verification sequence
    """Run local fits, held-out predictions and optical midpoint checks."""
    assert (
        Path(particula.__file__)
        .resolve()
        .is_relative_to(Path.cwd() / "particula")
    )
    eq = []
    check = []
    ideal = []
    roundtrip = []
    fine = np.linspace(0, np.log(256), 32769)
    for nm in SIZES:
        d = nm * 1e-9
        sc, peak = direct_peak(d, "BAT")
        k = invert(d, sc)
        q = shortcut(d, 1 + sc / 100)
        roundtrip.append(abs(percent(threshold(d, k), sc)))
        if nm in (10, 20, 30, 50, 75, 100, 150, 200):
            check.append(
                abs(percent(100 * (np.max(saturation(fine, d, "BAT")) - 1), sc))
            )
            ideal.append(
                abs(percent(invert(d, direct_peak(d, "ideal")[0]), K_IDEAL))
            )
        eq.append(
            dict(
                dry_nm=nm,
                critical_wet_nm=nm * peak,
                bat_sc_pct=sc,
                full_k=k,
                approx_k=q,
                approx_k_bias_pct=percent(q, k),
                approx_minus_full_k=q - k,
            )
        )
    by_size = {r["dry_nm"]: r for r in eq}
    training = [by_size[d] for d in TRAIN]

    def objective(k):
        return sum(
            np.log(threshold(r["dry_nm"] * 1e-9, k) / r["bat_sc_pct"]) ** 2
            for r in training
        )

    multi = minimize_scalar(
        objective,
        bounds=(0.01, 0.5),
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert multi.success
    fits = dict(
        full_100=by_size[100]["full_k"],
        approx_100=by_size[100]["approx_k"],
        full_multi=multi.x,
        approx_multi=float(
            np.exp(np.mean(np.log([r["approx_k"] for r in training])))
        ),
    )
    summary = {}
    for name, k in list(fits.items()) + [
        ("eq10_forward_100", fits["approx_100"]),
        ("eq10_forward_multi", fits["approx_multi"]),
    ]:
        for r in eq:
            d = r["dry_nm"] * 1e-9
            if name.startswith("eq10_forward"):
                pred = 100 * np.expm1(np.sqrt(4 * (A / d) ** 3 / (27 * k)))
            else:
                pred = threshold(d, k)
            r[name + "_sc_error_pct"] = percent(pred, r["bat_sc_pct"])
            r[name + "_sc_difference_pp"] = pred - r["bat_sc_pct"]
        for lo in (10, 50):
            hold = [
                r
                for r in eq
                if r["dry_nm"] >= lo
                and r["dry_nm"] not in ((100.0,) if "100" in name else TRAIN)
            ]
            errors = np.array([r[name + "_sc_error_pct"] for r in hold])
            summary[f"{name}_{lo}_200"] = dict(
                n=len(hold),
                min_relative_pct=errors.min(),
                max_relative_pct=errors.max(),
                rmse_relative_pct=np.sqrt(np.mean(errors**2)),
                max_abs_pp=max(
                    abs(r[name + "_sc_difference_pp"]) for r in hold
                ),
            )
    save_csv("bat-kappa-equilibrium.csv", eq)
    optical = []
    for r in read_rows(OUT / "bat-optical-midpoints.csv"):
        if r["model"] != "BAT":
            continue
        nm = float(r["median_nm"])
        d = nm * 1e-9
        s = float(r["ss50_pct"])
        reference = (
            by_size[nm]["full_k"]
            if nm in by_size
            else invert(d, direct_peak(d, "BAT")[0])
        )
        k = invert(d, s)
        q = shortcut(d, 1 + s / 100)
        optical.append(
            dict(
                dry_nm=nm,
                gsd=r["gsd"],
                seconds=r["seconds"],
                organic_gas_ug_m3=r["gas_ug_m3"],
                optical_ss50_pct=s,
                scan_lower_ss_pct=r["ss_lower"],
                scan_upper_ss_pct=r["ss_upper"],
                eq6_equilibrium_k=reference,
                eq6_optical_k=k,
                eq10_optical_k=q,
                eq6_vs_equilibrium_pct=percent(k, reference),
                eq10_vs_equilibrium_pct=percent(q, reference),
                eq10_vs_eq6_same_optical_pair_pct=percent(q, k),
            )
        )
    save_csv("bat-kappa-optical-errors.csv", optical)
    # Refine the primary optical midpoint inside the retained GPU scan bracket.
    # For a continuous monotone, single-composition lognormal selection, half
    # the particles meet the optical gate when its median particle meets it.
    primary = []
    for row in optical:
        if (
            float(row["dry_nm"]) != 100
            or float(row["gsd"]) != 1.1
            or float(row["seconds"]) != 5
        ):
            continue
        gas = float(row["organic_gas_ug_m3"])

        def median_residual(ss, gas=gas):
            wet, *_ = cpu(np.array([100.0]), "BAT", gas, ss, tol=1e-8)
            return wet[0, list(TIMES).index(5.0)] - 1.0

        lo, hi = map(
            float, [row["scan_lower_ss_pct"], row["scan_upper_ss_pct"]]
        )
        assert median_residual(lo) < 0 < median_residual(hi)
        ss = brentq(median_residual, lo, hi, xtol=2e-10)
        # Verify the threshold ordering on either side of the median.
        neighbors = cpu(np.array([95.0, 105.0]), "BAT", gas, ss, tol=1e-8)[0][
            :, list(TIMES).index(5.0)
        ]
        assert neighbors[0] < 1 < neighbors[1]
        k = invert(100e-9, ss)
        q = shortcut(100e-9, 1 + ss / 100)
        primary.append(
            dict(
                gas_ug_m3=gas,
                refined_ss50_pct=ss,
                scan_delta_ss_pp=ss - float(row["optical_ss50_pct"]),
                eq6_k=k,
                eq10_k=q,
                eq6_vs_fixed_equilibrium_pct=percent(k, fits["full_100"]),
                eq10_vs_fixed_equilibrium_pct=percent(q, fits["full_100"]),
                eq10_vs_eq6_pct=percent(q, k),
            )
        )
    save_csv("bat-kappa-midpoint-check.csv", primary)
    growth = []
    for rh in (0.9, 0.95, 0.99):
        d = 100e-9
        peak = by_size[100]["critical_wet_nm"] / 100
        bat = np.exp(
            brentq(
                lambda log_size, d=d, rh=rh: saturation(log_size, d, "BAT")
                - rh,
                0,
                np.log(peak),
            )
        )
        k = fits["full_100"]

        def eq6(x, k=k, d=d, rh=rh):
            return (x**3 - 1) / (x**3 - 1 + k) * np.exp(A / (d * x)) - rh

        gf = brentq(eq6, 1.0, peak)
        growth.append(
            dict(
                rh=rh,
                bat_growth_factor=bat,
                kappa_growth_factor=gf,
                wet_diameter_bias_pct=percent(gf, bat),
                water_volume_bias_pct=percent(gf**3 - 1, bat**3 - 1),
            )
        )
    save_csv("bat-kappa-growth.csv", growth)
    assert max(check) < 0.001 and max(ideal) < 1e-5 and max(roundtrip) < 1e-7
    results = dict(
        revision=revision(),
        fits=fits,
        held_out=summary,
        max_grid_check_pct=max(check),
        max_ideal_k_error_pct=max(ideal),
        max_roundtrip_pct=max(roundtrip),
    )
    (OUT / "bat-kappa-fit-summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    plot(eq)
    for nm in (10, 30, 50, 75, 100, 150, 200):
        print(
            nm,
            [
                round(by_size[nm][key], 6)
                for key in (
                    "bat_sc_pct",
                    "full_k",
                    "approx_k",
                    "approx_k_bias_pct",
                    "full_100_sc_error_pct",
                )
            ],
        )
    print(json.dumps(results, indent=2))
    print(growth)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
