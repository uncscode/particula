"""Pair the optical retrievals and replay the frozen leave-one-wind-out predictor."""

import json
from itertools import permutations
from scipy.optimize import linprog
from study import read_workbook
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optical_test import (
    ROOT,
    DIAM,
    WINDS,
    NOMINAL,
    WATER,
    LAMBDA,
    frac,
    med,
    read,
    save,
    spectrum,
    transit,
    scattering,
    ANGLES,
)
from global_fit import data, predict, NAMES, T
from global_summary import encode
from particula.optics import (
    get_mie_coefficients,
    get_scattering_amplitudes,
    get_mie_efficiencies,
)
from scipy.special import roots_legendre


def run():
    rs = read("optical-cases.csv")
    bins = np.load(ROOT / "optical-bins.npz")["spectra"]
    vals = {
        (int(r["wind"]), r["case"], float(r["hours"])): (r, q)
        for r, q in zip(rs, bins, strict=True)
    }
    cases = sorted({r["case"] for r in rs})
    base = data(5)
    names = NAMES["conditional"]
    fits = {
        r["fold"]: r
        for r in json.loads((ROOT / "global-core.json").read_text())["results"]
        if r["case"] == "conditional"
    }
    allrows = []
    for case in cases:
        z = base.copy()
        z["shape"] = base["shape"].clone()
        for wi, w in enumerate(WINDS):
            z["shape"][wi, 0] = T(vals[w, case, 0.5][1])
        for wi, w in enumerate(WINDS):
            er, e = vals[w, case, 0.5]
            lr, l = vals[w, case, 3.0]
            fit = fits[f"wind{w}"]
            model = predict(
                encode(fit["params"], names), names, "conditional", z, {}
            )
            y = model[wi, -1].numpy()
            y /= y.sum()
            ee = float(base["mass"][wi, 0, -1])
            ll = float(base["mass"][wi, -1, -1])

            def middle(q):
                return frac(q, 20) - frac(q, 10)

            allrows.append(
                dict(
                    wind=w,
                    family=er["family"],
                    case=case,
                    group=er["group"],
                    d50_early=med(e),
                    d50_late=med(l),
                    delta_d50=med(l) - med(e),
                    delta_middle_pp=100 * (middle(l) - middle(e)),
                    implied_middle_gain_mg=ll * middle(l) / frac(l, 63)
                    - ee * middle(e) / frac(e, 63),
                    under20_over63_late_pct=100 * frac(l, 20) / frac(l, 63),
                    late_predicted_d50=med(y),
                    late_d50_residual=med(l) - med(y),
                    max_endpoint_rms_pct=max(
                        float(er["rms_pct"]), float(lr["rms_pct"])
                    ),
                    max_endpoint_ring_pct=max(
                        float(er["max_pct"]), float(lr["max_pct"])
                    ),
                )
            )
    # Direct forward comparisons use the archived early input (no optical recalibration).
    mismatch = []
    full = scattering(NOMINAL.real, NOMINAL.imag)
    for wi, w in enumerate(WINDS):
        observed = spectrum(w, "measured_pct")
        for modelname, model_q in [
            ("early", spectrum(w, "early_pct")),
            (
                "shared_physics",
                np.array(fits[f"wind{w}"]["prediction"])[wi, -1],
            ),
        ]:
            model_q = model_q / model_q.sum()
            for cover, keep in [
                ("full", np.ones(len(full), bool)),
                ("limited", (ANGLES[:-1] >= 1) & (ANGLES[1:] <= 65)),
            ]:
                b = (full @ observed)[keep]
                a = (full @ model_q)[keep]
                floor = np.maximum(b, 0.01 * b.max())
                a = a / floor
                b = b / floor
                M = np.vstack(
                    [np.c_[a, -np.ones_like(a)], np.c_[-a, -np.ones_like(a)]]
                )
                res = linprog(
                    [0, 1],
                    A_ub=M,
                    b_ub=np.r_[b, -b],
                    bounds=(0, None),
                    method="highs",
                )
                if not res.success:
                    raise ValueError(res.message)
                mismatch.append(
                    dict(
                        wind=w,
                        case=modelname,
                        coverage=cover,
                        min_peak_mismatch_pct=100 * res.fun,
                        relative_gain=res.x[0],
                    )
                )
    save("optical-model-signals.csv", mismatch)
    replicate = read_workbook()[1]
    paired = []
    for w in WINDS:
        for h in (0.5, 3.0):
            small = np.array(replicate[w, h, 20])
            large = replicate[w, h, 63]
            if len(small) != len(large):
                continue
            feasible = [
                100 * np.mean(small / perm)
                for perm in permutations(large)
                if np.all(small <= perm)
            ]
            if feasible:
                paired.append(
                    dict(
                        wind=w,
                        hours=h,
                        num_id_assignments=len(feasible),
                        mean_ratio_low_pct=min(feasible),
                        mean_ratio_high_pct=max(feasible),
                        ratio_of_means_pct=100 * small.mean() / np.mean(large),
                    )
                )
    save("optical-tray-pairing.csv", paired)
    save("optical-transfer.csv", allrows)
    noise = read("optical-noise.csv")
    summary = []
    for w in WINDS:
        nn = [r for r in noise if int(r["wind"]) == w]
        early = sorted(
            [r for r in nn if float(r["hours"]) == 0.5],
            key=lambda x: int(x["draw"]),
        )
        late = sorted(
            [r for r in nn if float(r["hours"]) == 3.0],
            key=lambda x: int(x["draw"]),
        )
        shift = [
            float(l["d50_um"]) - float(e["d50_um"])
            for l, e in zip(late, early, strict=True)
        ]
        summary.append(
            dict(
                wind=w,
                draws=len(shift),
                **{
                    k: float(v)
                    for k, v in zip(
                        ("p05", "p50", "p95"),
                        np.percentile(shift, [5, 50, 95]),
                        strict=True,
                    )
                },
            )
        )
    save("optical-noise-summary.csv", summary)
    # Check the unpolarized angular energy integral independently against Qsca.
    ds = np.array([0.25, 2.0, 10.0, 32.0, 65.7444])
    ka = 2 * np.pi * WATER / LAMBDA
    mu, wt = roots_legendre(650)
    coef = get_mie_coefficients(NOMINAL / WATER, ka * ds / 2)
    s1, s2 = get_scattering_amplitudes(coef, np.arccos(mu))
    cs = np.pi * ((abs(s1) ** 2 + abs(s2) ** 2) @ wt) / ka**2
    eff = get_mie_efficiencies(ds * 0.5e-6, LAMBDA * 1e-6, NOMINAL, WATER)
    css = np.pi * (ds / 2) ** 2 * eff.scattering
    err = max(abs(cs / css - 1))
    if err > 1e-7:
        raise ValueError(err)
    transit_check = max(
        np.max(abs(transit(ar, off, order=16) - transit(ar, off, order=32)))
        for ar in [1.5, 2.0]
        for off in [0.0, 0.35]
    )
    (ROOT / "optical-checks.json").write_text(
        json.dumps(
            dict(
                cases=len(rs),
                retrieval_pairs=len(allrows),
                noise_retrievals=len(noise),
                max_energy_relative_error=err,
                transit_max_matrix_entry_refinement=transit_check,
            ),
            indent=2,
        )
    )
    plot(allrows)


def plot(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.6), layout="constrained")
    groups = [
        ("Archived spectrum", None),
        ("Transit · random orientations", "random"),
        ("Transit · directional scans", "aligned"),
        ("Transit · size-dependent detection", "acceptance"),
        ("Diffraction · 1% signal tolerance", 1.0),
        ("Diffraction · 5% signal tolerance", 5.0),
    ]
    limits = read("optical-feasible.csv")
    for ax, w, key, label, original in [
        (
            a,
            2,
            "delta_d50",
            "Change in volume median (µm)",
            med(spectrum(2, "measured_pct")) - med(spectrum(2, "early_pct")),
        ),
        (
            b,
            6,
            "under20_over63_late_pct",
            "Volume below 20 µm / volume below 63 µm (%)",
            100
            * frac(spectrum(6, "measured_pct"), 20)
            / frac(spectrum(6, "measured_pct"), 63),
        ),
    ]:
        for i, (name, g) in enumerate(groups):
            if isinstance(g, float):
                cells = {
                    float(r["hours"]): r
                    for r in limits
                    if int(r["wind"]) == w
                    and r["case"] == "n1.52_k0.002"
                    and r["coverage"] == "full"
                    and float(r["tolerance_pct"]) == g
                }
                l = cells[3.0]
                if w == 2:
                    e = cells[0.5]
                    v = [
                        float(l["median_lo"]) - float(e["median_hi"]),
                        float(l["median_hi"]) - float(e["median_lo"]),
                    ]
                else:
                    v = [
                        float(l["under20_over63_lo"]),
                        float(l["under20_over63_hi"]),
                    ]
                ax.plot(v, [i, i], lw=4, color="#815d91", solid_capstyle="butt")
            else:
                v = (
                    [r[key] for r in rows if r["wind"] == w and r["group"] == g]
                    if g
                    else [original]
                )
                color = "#226b86" if g else "#253039"
                ax.plot(
                    [min(v), max(v)],
                    [i, i],
                    lw=3,
                    color=color,
                    solid_capstyle="round",
                )
                ax.scatter(
                    np.median(v), i, marker="o", color=color, zorder=3, s=27
                )
        ax.set_yticks(
            range(len(groups)), [g[0] for g in groups] if ax is a else []
        )
        ax.set_ylim(len(groups) - 0.5, -0.6)
        ax.set_xlabel(label)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.axvline(0 if w == 2 else 19.82, ls="--", lw=1.2, c="#aa532c")
    a.set_title("2 m/s · 30 min to 3 h", loc="left")
    b.set_title("6 m/s · three-hour tray", loc="left")
    fig.suptitle(
        "How much could the optical response change the tray result?",
        fontweight="bold",
    )
    fig.savefig(ROOT / "optical-sensitivity.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    run()
