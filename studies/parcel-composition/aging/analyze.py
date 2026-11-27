"""Select representative counts and score independent histories."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from physics import EDGES, SS
from run import SEEDS

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parents[2] / "docs/images/parcel-composition-aging.png"
LIMITS = np.array([2.0, 5.0, 5.0])
LABELS = [
    "mean",
    "mean + SD",
    "signed linear",
    "quadratic",
    "six knots",
    "full profile",
]
METRICS = [
    "activation_rmse_pp",
    "activation_max_pp",
    "absorption_error_pct",
    "chi_rmse",
    "rows",
]


def first_pass(worst, limits=LIMITS):
    """Return the smallest tested cap satisfying all selected tolerances."""
    valid = np.flatnonzero((worst[:, : len(limits)] <= limits).all(axis=1))
    return int(valid[0] + 1) if valid.size else None


def continuous_scores(scores, checked, split):
    """Replace the sampled maximum by the supremum on the checked interval."""
    maxima = checked[checked.split == split].sort_values("cap")
    np.testing.assert_array_equal(maxima.cap, np.arange(1, len(maxima) + 1))
    result = scores[: len(maxima), :3].copy()
    result[:, 1] = maxima.continuous_max_pp
    return result


def summaries(chi, count, template):
    """Encode a size profile using one, two, three, six or twelve numbers."""
    w = count / count.sum()
    active = count > 0
    x = np.linspace(-1, 1, 12)
    mu = w[active] @ chi[active]
    sd = np.sqrt(w[active] @ (chi[active] - mu) ** 2)
    centered = template - w @ template
    sigma = np.sqrt(w @ centered**2)
    profiles = [np.full(12, mu), mu + sd * centered / sigma]
    knots = np.linspace(-1, 1, 6)
    basis = np.stack([np.interp(x, knots, axis) for axis in np.eye(6)], axis=1)
    for design in [np.vander(x, 2), np.vander(x, 3), basis]:
        weight = np.sqrt(w[active])
        fit = np.linalg.lstsq(
            design[active] * weight[:, None], chi[active] * weight, rcond=None
        )[0]
        profiles.append(design @ fit)
    profiles.append(np.nan_to_num(chi))
    return np.clip(profiles, 0, 1)


def template_from_training(records):
    """Estimate one shared, oriented profile shape on calibration histories."""
    profile_sum = np.zeros(12)
    obs = np.zeros(12)
    for data in records:
        for chi, count in zip(data["chi"], data["counts"], strict=True):
            active = count > 0
            mu = np.average(chi[active], weights=count[active])
            sd = np.sqrt(
                np.average((chi[active] - mu) ** 2, weights=count[active])
            )
            profile_sum[active] += (chi[active] - mu) / sd
            obs += active
    return profile_sum / obs


def analyze(folder):
    """Compute snapshot metrics and fixed-budget curves for both partitions."""
    data = [np.load(folder / f"{s}.npz") for s in SEEDS]
    template = template_from_training(data[:8])
    worst_train = np.concatenate([d["scores"] for d in data[:8]]).max(axis=0)
    checked = pd.read_csv(HERE / "continuum_carrier.csv")
    chosen = first_pass(continuous_scores(worst_train, checked, "calibration"))
    if chosen is None:
        raise RuntimeError(
            "No calibration cap passes the prespecified tolerances"
        )
    budgets, profiles, snapshots, curves = [], [], [], []
    for idx, (seed, d) in enumerate(zip(SEEDS, data, strict=True)):
        split = "calibration" if idx < 8 else "held-out"
        for j, scores in enumerate(d["scores"]):
            meta = dict(seed=seed, split=split, hour=j // 2, parcel="AB"[j % 2])
            for cap, values in enumerate(scores, start=1):
                budgets.append(
                    dict(
                        **meta,
                        cap=cap,
                        **dict(zip(METRICS, values, strict=True)),
                    )
                )
            count, chi = d["counts"][j], d["chi"][j]
            active = count > 0
            w = count[active] / count.sum()
            fits = summaries(chi, count, template)
            bulk = d["bulk_entropy"][j, active]
            basis = d["bases"][j, active]
            for label, fit in zip(LABELS, fits, strict=True):
                internal = np.log1p(fit[active] * np.expm1(bulk)) / bulk
                approx = (
                    internal[:, None] * basis[:, 0]
                    + (1 - internal[:, None]) * basis[:, 1]
                ).sum(0)
                delta = 100 * (approx[:-1] - d["ccn"][j])
                snapshots.append(
                    dict(
                        **meta,
                        method=label,
                        chi_rmse=np.sqrt(w @ (fit[active] - chi[active]) ** 2),
                        activation_rmse_pp=np.sqrt(np.mean(delta**2)),
                        activation_max_pp=abs(delta).max(),
                        absorption_error_pct=100
                        * abs(approx[-1] * 1e-3 / d["babs"][j] - 1),
                    )
                )
            for b in np.flatnonzero(active):
                profiles.append(
                    dict(
                        **meta,
                        bin=b,
                        count=count[b],
                        chi=chi[b],
                        representative=d["rep_chi"][j, chosen - 1, b],
                        **{
                            label: fit[b]
                            for label, fit in zip(
                                LABELS[:-1], fits[:-1], strict=True
                            )
                        },
                    )
                )
            for ss, ref, rec in zip(
                SS, d["ccn"][j], d["rep_ccn"][j, chosen - 1], strict=True
            ):
                curves.append(
                    dict(
                        **meta,
                        ss_pct=ss,
                        reference=ref,
                        representative=rec,
                        reference_abs_Mm_inv=d["babs"][j],
                    )
                )
    budget = pd.DataFrame(budgets)
    summary = budget.groupby(["split", "cap"])[METRICS].max().reset_index()
    summary = summary.merge(checked, on=["split", "cap"], validate="one_to_one")
    summary.to_csv(HERE / "budget.csv", index=False, float_format="%.8g")
    profile, snapshot, curve = map(pd.DataFrame, [profiles, snapshots, curves])
    for frame, name in [
        (profile, "profiles"),
        (snapshot, "shape_scores"),
        (curve, "curves"),
        (budget[budget.cap == chosen], "selected_scores"),
    ]:
        frame.to_csv(HERE / f"{name}.csv", index=False, float_format="%.8g")
    return chosen, summary, profile, snapshot, curve


def plot(chosen, summary, profile, snapshot, curve):
    """Show the budget, hardest held-out curve and χ shape failure."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained")
    score = summary[summary.split == "held-out"]
    for col, limit, label in zip(
        [METRICS[0], "continuous_max_pp", METRICS[2]],
        LIMITS,
        ["Activation RMSE", "Anywhere on curve", "Absorption"],
        strict=True,
    ):
        axes[0, 0].plot(score.cap, score[col] / limit, label=label)
    axes[0, 0].axhline(1, color="0.45", ls=":")
    axes[0, 0].axvline(chosen, color="0.5", ls="--")
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Representative cap per size bin",
        ylabel="Worst error / tolerance",
        title="Recovery of both responses",
    )
    axes[0, 0].legend(fontsize=8, frameon=False)
    hold = curve[curve.split == "held-out"]
    where = (hold.reference - hold.representative).abs().idxmax()

    def select(frame, row):
        return frame[
            (frame.seed == row.seed)
            & (frame.parcel == row.parcel)
            & (frame.hour == row.hour)
        ]

    c = select(hold, hold.loc[where])
    axes[0, 1].plot(c.ss_pct, c.reference * 100, label="Full population", lw=2)
    axes[0, 1].plot(
        c.ss_pct, c.representative * 100, "x", label=f"Cap {chosen}", ms=4
    )
    axes[0, 1].set(
        xscale="log",
        xlabel="Supersaturation (%)",
        ylabel="Activated number (%)",
        title="Largest held-out channel error",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)
    mask = (snapshot.method == "mean + SD") & (snapshot.split == "held-out")
    row = snapshot.loc[snapshot[mask].chi_rmse.idxmax()]
    p = select(profile, row)
    centers = np.sqrt(EDGES[1:] * EDGES[:-1])[p.bin]
    for label, style in [
        ("chi", "o-"),
        ("mean + SD", "--"),
        ("six knots", "-."),
        ("representative", ":"),
    ]:
        axes[1, 0].plot(
            centers,
            p[label],
            style,
            label={"chi": "Resolved χ", "representative": f"Cap {chosen}"}.get(
                label, label
            ),
            ms=3,
        )
    axes[1, 0].set(
        xscale="log",
        xlabel="Dry diameter (nm)",
        ylabel="Bin χ",
        title="Where a shared shape misses",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    held = snapshot[snapshot.split == "held-out"]
    for i, label in enumerate(LABELS):
        d = held[held.method == label]
        axes[1, 1].plot(
            [d.chi_rmse.min(), d.chi_rmse.max()], [i, i], color="0.55"
        )
        axes[1, 1].plot(d.chi_rmse.median(), i, "o", color="#006e78")
    axes[1, 1].set(
        yticks=np.arange(len(LABELS)),
        yticklabels=LABELS,
        xlabel="Count-weighted χ profile RMSE",
        title="Held-out range and median",
    )
    axes[1, 1].invert_yaxis()
    fig.savefig(IMAGE, dpi=200)
    plt.close(fig)


def plot_history(profile, curve):
    """Follow the held-out refinement history at common observation settings."""
    pair = curve[curve.seed == 1199]
    parcel = profile[(profile.seed == 1199) & (profile.parcel == "B")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    centers = np.sqrt(EDGES[1:] * EDGES[:-1])
    for hour, color in [(0, "#993e24"), (3, "#97720c"), (6, "#036a79")]:
        sample = parcel[parcel.hour == hour]
        axes[0].plot(
            centers[sample.bin],
            sample.chi,
            "o-",
            ms=3,
            color=color,
            label=f"{hour} h",
        )
    axes[0].set(
        xscale="log",
        xlabel="Dry diameter (nm)",
        ylabel="Bin χ",
        title="Parcel B changes shape",
        ylim=(0.25, 1),
    )
    ss = SS[np.argmin(abs(SS - 0.2))]
    for name, color in [("A", "#6c4680"), ("B", "#036a79")]:
        sample = pair[
            (pair.parcel == name)
            & np.isclose(pair.ss_pct, ss, rtol=1e-7, atol=0)
        ]
        axes[1].plot(
            sample.hour,
            sample.reference * 100,
            "o-",
            ms=3,
            color=color,
            label=f"Parcel {name}",
        )
        axes[2].plot(
            sample.hour,
            sample.reference_abs_Mm_inv,
            "o-",
            ms=3,
            color=color,
            label=f"Parcel {name}",
        )
    axes[1].set(
        xlabel="Time (h)",
        ylabel="Activated number (%)",
        title=f"Activation at {ss:.3f}%",
    )
    axes[2].set(
        xlabel="Time (h)",
        ylabel="Absorption (Mm⁻¹)",
        title="Dry absorption at 550 nm",
    )
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(
        IMAGE.with_name("parcel-composition-aging-history.png"), dpi=200
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    chosen, summary, profile, snapshot, curve = analyze(
        parser.parse_args().folder
    )
    plot(chosen, summary, profile, snapshot, curve)
    plot_history(profile, curve)
    print("Calibration-selected cap:", chosen)
    print(summary[summary.cap == chosen].to_string(index=False))
    print(
        snapshot[snapshot.split == "held-out"]
        .groupby("method")[METRICS[:4]]
        .agg(["median", "max"])
        .round(4)
    )
