"""Test rare count disagreement after early physical corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from stage2_bootstrap import grouped_observables
from stage2_rare_count import bound
from stage2_score_split import arrays
from stage3_evidence import write

FIRST = 465001
LOOKS = 8


def integers(values):
    """Recover particles per finite box and validate integer multiplicity."""
    n = 8 * values[:, :, 0] / values[:, 0:1, 0]
    np.testing.assert_allclose(n, n.round(), rtol=0, atol=1e-12)
    return n.round().astype(int)


def load_block(raw, species, first, policy):
    """Require an entire registered seed block and two physical tolerances."""
    tag = f"{species}-{first}"
    root = raw / "number-power-ref" / tag
    paths = list(
        (raw / "number-power" / tag).glob(f"*-ode-{policy}-joint*.npz")
    )
    if len(paths) != 1 or len(list(root.glob("*.npz"))) != 512:
        return None
    with np.load(paths[0]) as record:
        keys = record["keys"]
        np.testing.assert_array_equal(keys[:, 3], np.arange(first, first + 256))
        if not np.all(keys[:, :3] == [85001, 1, 8]):
            raise ValueError("Unexpected targeted confirmation case")
        meta = json.loads(str(record["meta"]))
        fine, coarse, num = arrays(dict(keys=keys), root, meta)
        if num != 2:
            raise ValueError("Both tolerances are required for qualification")
        preds = record["obs"]
        work = json.loads(str(record["work"]))
        return (
            tuple(
                grouped_observables(x, species) for x in (preds, fine, coarse)
            ),
            record["t"],
            list(record["names"][: 2 * species + 3])
            + ["particle_total", "system_total"],
            float(record["seconds"].sum()),
            sum(entry.get("physical_box_intervals", 0) for entry in work),
            sum(
                entry["forecast_observation_steps"]
                * len(keys[i * meta["batch"] : (i + 1) * meta["batch"]])
                for i, entry in enumerate(work)
            ),
        )


def mass_error(pred, fine, coarse, times, names):
    """Bound paired means across observed times and phases."""
    p, f, c = [a[:, :, 1:] for a in (pred, fine, coarse)]
    n, nt, nf = f.shape
    critical = t.ppf(1 - 0.05 / (4 * LOOKS * nt * nf), n - 1)
    mean = f.mean(0)
    denom = np.maximum(
        abs(mean) - critical * f.std(0, ddof=1) / np.sqrt(n), 1e-11
    )
    e = p - f
    upper = (
        100
        * (
            abs(e.mean(0))
            + critical * e.std(0, ddof=1) / np.sqrt(n)
            + abs(c - f).mean(0)
        )
        / denom
    )
    i, j = np.unravel_index(upper.argmax(), upper.shape)
    return dict(
        largest_mass_upper_pct=upper[i, j],
        limiting_mass_field=names[j + 1],
        limiting_mass_time_s=times[i],
        largest_mass_mean_pct=np.max(
            100 * abs(e.mean(0)) / np.maximum(abs(mean), 1e-11)
        ),
    )


def score(raw, results):
    """Assess only the registered contiguous looks of 1,024 and 2,048 paths."""
    rows = []
    for s in [8, 32]:
        for policy in ["none", "early"]:
            blocks = []
            elapsed = []
            intervals = 0
            available = 0
            for first in range(FIRST, FIRST + 2048, 256):
                data = load_block(raw, s, first, policy)
                if data is None:
                    break
                arr, times, names, seconds, corrected, total = data
                elapsed.append(seconds)
                intervals += corrected
                available += total
                blocks.append(arr)
                if len(blocks) not in [4, 8]:
                    continue
                p, f, c = (np.concatenate(v) for v in zip(*blocks, strict=True))
                rows.append(
                    dict(
                        species=s,
                        policy=policy,
                        family=85001,
                        mixture=1,
                        density_class=8,
                        first_seed=FIRST,
                        last_seed=first + 255,
                        measured_cpu_ensemble_seconds=sum(elapsed),
                        median_256_box_seconds=float(np.median(elapsed)),
                        physical_box_intervals=intervals,
                        available_box_intervals=available,
                        **bound(
                            integers(p),
                            integers(f),
                            integers(c),
                            8,
                            looks=LOOKS,
                        ),
                        **mass_error(p, f, c, times, names),
                    )
                )
    write(results / "reversible-count-confirmation.csv", rows)
    if rows:
        chart(rows, results / "early-physical-correction.png")


def chart(rows, target):
    """Plot cost and simultaneous bounds at completed matched sample sizes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(11.7, 4.8), layout="constrained")
    settings = sorted({(r["species"], r["seeds"]) for r in rows})
    x = np.arange(len(settings))
    for policy, shift, label, color in (
        ("none", -0.16, "Learned forecast throughout", "#1f6589"),
        ("early", 0.16, "Physical ODE after each pulse", "#bd6537"),
    ):
        values = {
            (r["species"], r["seeds"]): r for r in rows if r["policy"] == policy
        }
        for ax, key in zip(
            axs,
            (
                "median_256_box_seconds",
                "simultaneous_number_upper_pct",
                "largest_mass_upper_pct",
            ),
            strict=True,
        ):
            y = [values[k][key] if k in values else np.nan for k in settings]
            ax.scatter(x + shift, y, s=60, label=label, color=color)
    for ax, ylabel in zip(
        axs,
        (
            "CPU time per 256 boxes (s)",
            "Number error bound (%)",
            "Phase mass error bound (%)",
        ),
        strict=True,
    ):
        ax.set_xticks(x, [f"{s} vapors\n{n:,} seeds" for s, n in settings])
        ax.set_xlim(-0.5, len(settings) - 0.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    for ax, limit in zip(axs[1:], (1, 0.1), strict=True):
        ax.axhline(limit, ls="--", lw=1, color="#444")
        ax.set_yscale("log")
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.suptitle("Selective physical steps in a reversible eight-particle box")
    fig.savefig(target, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("raw", type=Path)
    a.add_argument("results", type=Path)
    opts = a.parse_args()
    score(opts.raw, opts.results)
