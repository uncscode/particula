"""Compare CUDA pulse ensembles and steady retrievals with Figure 6."""

import csv
import json
import platform

import matplotlib
import numpy as np
import particula
import scipy
import warp
from study import revision

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from instrument import HERE, TARGETS, D, probabilities, save, selected
from scipy.stats import t


def read(name):
    """Read a table from the resolved study output."""
    with (HERE / name).open() as f:
        return list(csv.DictReader(f))


def pulse_summary():
    """Aggregate the four seeds for each pulse condition."""
    cases = [
        "reference",
        "refined",
        "elevated",
        "single_size",
        "large_parent",
        "dispersion",
        "dispersion_refined",
        "dispersion_ideal",
    ]
    results, meta = [], []
    for case in cases:
        data = []
        for seed in [103, 227, 431, 653]:
            item = json.loads((HERE / f"{case}-{seed}.json").read_text())
            np.testing.assert_equal(
                item["reps"] == 2048 and item["mass_charge_number_checks"], True
            )
            meta.append(item)
            data.append(read(f"{case}-{seed}.csv"))
        for d in TARGETS:

            def batch(cohort, d=d, data=data):
                return [
                    next(
                        v
                        for v in rows
                        if float(v["target_nm"]) == d
                        and float(v["time_s"]) == 300
                        and float(v["initial_collectors_cm3"]) > 0
                        and v["cohort"] == cohort
                    )
                    for rows in data
                ]

            rows = batch("small")
            beta = np.array(
                [
                    float(v["weighted_capture_sum"])
                    / float(v["weighted_exposure_sum"])
                    for v in rows
                ]
            )
            unweighted = [
                float(v["capture_sum"]) / float(v["exposure_sum"]) for v in rows
            ]
            h = (
                t.ppf(0.975, len(beta) - 1)
                * beta.std(ddof=1)
                / np.sqrt(len(beta))
            )
            rec = dict(
                case=case,
                target_nm=d,
                beta_weighted_cm3_s=beta.mean(),
                beta_mc95_halfwidth=h,
                beta_true_cm3_s=np.mean(unweighted),
            )
            for cohort in ("small", "large"):
                subset = batch(cohort)
                n = np.array([int(v["sample_particles"]) for v in subset])
                for key in [
                    "mean_charge_e",
                    "neg_fraction",
                    "neutral_fraction",
                    "pos_fraction",
                    "multiple_fraction",
                    "mean_d_nm",
                ]:
                    rec[f"{cohort}_{key}"] = np.average(
                        [float(v[key]) for v in subset], weights=n
                    )
                for key in [
                    "alive_per_box",
                    "wall_per_box",
                    "diluted_per_box",
                    "coagulated_tracers_per_box",
                ]:
                    rec[f"{cohort}_{key}"] = np.mean(
                        [float(v[key]) for v in subset]
                    )
                den = sum(float(v["detected_per_box"]) for v in subset)
                rec[f"{cohort}_sensor_multiple"] = (
                    sum(
                        float(v["detector_z2"]) + float(v["detector_z3"])
                        for v in subset
                    )
                    / den
                )
            results.append(rec)
    save("gpu-summary.csv", results)
    return results, meta


def main():
    """Compare the pulse ensembles and the steady retrieval with experiment."""
    results, meta = pulse_summary()
    steady = read("steady.csv")
    prior = read("../coefficients.csv")

    def steady_row(case, model, d):
        return next(
            r
            for r in steady
            if r["case"] == case
            and r["model"] == model
            and float(r["target_nm"]) == d
        )

    def app(case, model, d):
        return float(steady_row(case, model, d)["apparent_cm3_s"])

    comparison = []
    for d, old in zip(TARGETS, prior, strict=True):
        np.testing.assert_equal(d == float(old["d_nm"]), True)
        measured = float(old["measured"])
        current = app("reference", "dispersion", d)
        factor = current / app("single_size", "dispersion", d)
        anchored = app("reference", "published_anchor", d)
        comparison.append(
            dict(
                d_nm=d,
                measured_cm3_s=measured,
                published_vdw_cm3_s=old["authors_enhanced"],
                native_steady_cm3_s=app("reference", "native", d),
                native_plus_vdw_steady_cm3_s=current,
                transfer_factor=factor,
                anchored_published_cm3_s=anchored,
                anchored_vs_measured_percent=100 * (anchored / measured - 1),
            )
        )
    save("comparison.csv", comparison)

    def sampled(case, key):
        return np.array([r[key] for r in results if r["case"] == case])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax.plot(
        TARGETS,
        [float(r["measured"]) * 1e7 for r in prior],
        "ko",
        label="Cai measurements",
    )
    ax.plot(
        TARGETS,
        [r["native_plus_vdw_steady_cm3_s"] * 1e7 for r in comparison],
        color="#00796b",
        label="Steady SMPS retrieval, dispersion",
    )
    ax.errorbar(
        TARGETS,
        sampled("dispersion_refined", "beta_weighted_cm3_s") * 1e7,
        yerr=sampled("dispersion_refined", "beta_mc95_halfwidth") * 1e7,
        fmt="x",
        color="#cc5900",
        capsize=3,
        label="Refined GPU pulse (95% MC)",
    )
    ax.plot(
        TARGETS,
        [r["anchored_published_cm3_s"] * 1e7 for r in comparison],
        "--",
        color="#514a94",
        label="Published attraction, with transfer",
    )
    ax.set(
        xlabel="Nominal tracer diameter (nm)",
        ylabel="Capture coefficient (10⁻⁷ cm³/s)",
    )
    ax.legend(frameon=False, fontsize=8)
    p = probabilities(D, "upper")
    dist = selected(100, p, large=True)
    for j, col in enumerate(["#38658d", "#d17727", "#429b83"]):
        bx.plot(
            D,
            dist[:, j] / np.log(D[1] / D[0]),
            color=col,
            label=f"Selected at +{j + 1} charge",
        )
    bx.set(
        xlim=(75, 240),
        xlabel="True collector diameter (nm)",
        ylabel="Transmitted number per ln(diameter)",
    )
    bx.legend(frameon=False, fontsize=8)
    fig.savefig(HERE / "chamber-comparison.png", dpi=180)
    plt.close(fig)
    numerical = max(
        abs(app("numerical", m, d) / app("reference", m, d) - 1)
        for d in TARGETS
        for m in ["native", "dispersion", "published_anchor"]
    )
    manifest = dict(
        revision=revision(),
        python=platform.python_version(),
        particula=particula.__version__,
        warp=warp.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        device=meta[0]["device"],
        production_runs=len(meta),
        box_histories=sum(x["boxes"] for x in meta),
        slots_per_box=9,
        production_seconds=sum(x["seconds"] for x in meta),
        accepted_native_collisions=sum(x["accepted_collisions"] for x in meta),
        accepted_added_dispersion_collisions=sum(
            x["dispersion_events"] for x in meta
        ),
        max_relative_steady_quadrature_change=numerical,
    )
    (HERE / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
