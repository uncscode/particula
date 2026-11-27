"""Qualify the default checkpoint's count distribution within each air mass."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from burst_counts import DT, FRESH, SEEDS, SPEC, START, N, advance
from campaign import audit, inventory
from model import step
from redraw_checkpoint import draw_box, fit_box, load, serialize
from redraw_design import FORCING
from scipy.stats import beta, poisson
from stage_two_design import REGIME, Case, initialize

HISTORIES = SEEDS + FRESH
LOOKS = (192, 576)
TOLERANCE = 0.05
# Two fixed looks and one one-sided coverage statement per history at each.
ALPHA = 0.05 / (len(HISTORIES) * len(LOOKS))


class FormationQuantiles:
    """Use one fresh uniform per active scalar Poisson draw."""

    def __init__(self, seed):
        """Initialize a separate formation stream for this realization."""
        self.rng = np.random.default_rng(seed)

    def poisson(self, mean):
        """Sample an integer Poisson count by its inverse CDF."""
        u = max(self.rng.random(), np.nextafter(0.0, 1.0))
        return int(poisson.ppf(u, mean))


def disagreement_bound(k, n):
    """Return the exact one-sided binomial upper confidence bound."""
    return 1.0 if k == n else beta.isf(ALPHA, k + 1, n - k)


def simulate(job):
    """Couple two model continuations while preserving both count laws."""
    seed, levels, first, stop = job
    case = Case(seed, *levels, n=N, dt=DT)
    original, saved, forcing = initialize(case)
    initial = inventory(original)
    source = np.zeros(8)
    for t in range(0, START, DT):
        source += step(original, saved, forcing, REGIME, t, DT)
    audit(original, initial, N, source)
    fields = [fit_box(box, *SPEC) for box in original]
    packed, header = load(serialize(fields, case, forcing, START, saved, SPEC))
    np.testing.assert_equal(header["seconds"], START)
    np.testing.assert_array_equal(header["recipe"], SPEC)
    for field, value in forcing.items():
        np.testing.assert_array_equal(header["forcing"][field], value)
    rows = []
    for draw in range(first, stop):
        key = [seed, N, START, 7561, draw]
        sample = np.random.default_rng([*key, 0])
        new = [draw_box(b, sample) for b in packed]
        event = np.random.default_rng([*key, 1])
        populations = [copy.deepcopy(original), new]
        counts = []
        for population in populations:
            count, _, _ = advance(
                population,
                copy.deepcopy(event),
                forcing,
                5,
                FormationQuantiles([*key, 2]),
            )
            counts.append(count)
        rows.append((seed, draw, *counts[0], *counts[1]))
    return rows


def score(data):
    """Bound every CDF threshold using the probability of unequal totals."""
    out = []
    for seed, g in data.groupby("seed"):
        np.testing.assert_array_equal(np.sort(g.draw), np.arange(len(g)))
        full = g.full_a + g.full_b
        redraw = g.redraw_a + g.redraw_b
        k = int((full != redraw).sum())
        upper = disagreement_bound(k, len(g))
        support = np.union1d(full, redraw)
        cdf = np.abs(
            np.searchsorted(np.sort(full), support, side="right")
            - np.searchsorted(np.sort(redraw), support, side="right")
        )
        out.append(
            dict(
                seed=seed,
                trials=len(g),
                different=k,
                first_look_different=int(
                    ((full != redraw) & (g.draw < LOOKS[0])).sum()
                ),
                full_q05=np.quantile(full, 0.05, method="inverted_cdf"),
                full_q95=np.quantile(full, 0.95, method="inverted_cdf"),
                redraw_q05=np.quantile(redraw, 0.05, method="inverted_cdf"),
                redraw_q95=np.quantile(redraw, 0.95, method="inverted_cdf"),
                full_mean=full.mean(),
                redraw_mean=redraw.mean(),
                observed_cdf_gap=cdf.max() / len(g),
                upper_bound=upper,
                status="qualified" if upper <= TOLERANCE else "unresolved",
            )
        )
    return pd.DataFrame(out)


def figure(data, path):
    """Display each active history on its own count axis."""
    import matplotlib.pyplot as plt

    groups = []
    inactive = []
    for seed, g in data.groupby("seed"):
        full = (g.full_a + g.full_b).to_numpy()
        redraw = (g.redraw_a + g.redraw_b).to_numpy()
        if max(full.max(), redraw.max()) == 0:
            inactive.append(str(seed))
        else:
            groups.append((seed, full, redraw))

    columns = 4
    panels = len(groups) + bool(inactive)
    fig, axes = plt.subplots(
        (panels + columns - 1) // columns,
        columns,
        figsize=(11, 4.8),
        squeeze=False,
        layout="constrained",
        sharey=True,
    )
    for ax, (seed, full, redraw) in zip(axes.flat, groups, strict=False):
        y = np.arange(1, len(full) + 1) / len(full)
        low = min(full.min(), redraw.min()) - 1
        for cnt, color, style, label in [
            (full, "#364358", "-", "Full resolution"),
            (redraw, "#d16a22", "--", "16-row redraw"),
        ]:
            ax.step(
                np.r_[low, np.sort(cnt)],
                np.r_[0, y],
                where="post",
                color=color,
                ls=style,
                lw=1.7,
                label=label,
            )
        ax.set_title(f"History {seed}", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    if inactive:
        ax = axes.flat[len(groups)]
        ax.axis("off")
        ax.text(0, 0.88, "Zero births in every pair", transform=ax.transAxes)
        for j in range(0, len(inactive), 3):
            ax.text(
                0,
                0.66 - j * 0.055,
                ", ".join(inactive[j : j + 3]),
                transform=ax.transAxes,
                fontsize=9,
            )
        if groups:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="lower left",
                frameon=False,
                borderaxespad=0,
                fontsize=9,
            )
    for ax in list(axes.flat)[panels:]:
        ax.axis("off")
    fig.supxlabel("Births across both parcels in 75 s")
    fig.supylabel("Cumulative probability")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    """Run the prespecified two-look design and save every realization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figure", type=Path)
    parser.add_argument(
        "--from-pairs",
        action="store_true",
        help="Rebuild the report from completed saved trials",
    )
    args = parser.parse_args()
    if args.from_pairs:
        data = pd.read_csv(args.output / "burst-conditional-pairs.csv.gz")
        validate_completed(data)
        publish(data, args.output, args.figure)
        return
    levels = dict(zip(HISTORIES, FORCING * 4, strict=True))
    selected = list(levels)
    all_rows = []
    first = 0
    for stop in (2,) if args.smoke else LOOKS:
        jobs = [
            (s, levels[s], begin, min(begin + 32, stop))
            for begin in range(first, stop, 32)
            for s in (selected[:1] if args.smoke else selected)
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            for batch in pool.map(simulate, jobs):
                all_rows.extend(batch)
        data = pd.DataFrame(
            all_rows,
            columns=[
                "seed",
                "draw",
                "full_a",
                "full_b",
                "redraw_a",
                "redraw_b",
            ],
        )
        report = score(data)
        selected = report.loc[report.status == "unresolved", "seed"].tolist()
        print(
            f"After {stop} trials per continuing history, "
            f"{len(selected)} unresolved",
            flush=True,
        )
        if not selected:
            break
        first = stop
    args.output.mkdir(parents=True, exist_ok=True)
    data.sort_values(["seed", "draw"]).to_csv(
        args.output / "burst-conditional-pairs.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    if not args.smoke:
        validate_completed(data)
    publish(data, args.output, args.figure)


def validate_completed(data):
    """Check the design and the stopping decision using trial-level records."""
    np.testing.assert_array_equal(
        np.sort(data.seed.unique()), sorted(HISTORIES)
    )
    counts = data[["full_a", "full_b", "redraw_a", "redraw_b"]]
    if (
        not all(np.issubdtype(t, np.integer) for t in counts.dtypes)
        or (counts < 0).any().any()
    ):
        raise ValueError("Invalid formation counts")
    for _, g in data.groupby("seed"):
        initial = g[g.draw < LOOKS[0]]
        np.testing.assert_equal(len(initial), LOOKS[0])
        k = (
            (initial.full_a + initial.full_b)
            != (initial.redraw_a + initial.redraw_b)
        ).sum()
        stopped = disagreement_bound(k, LOOKS[0]) <= TOLERANCE
        np.testing.assert_equal(len(g), LOOKS[0] if stopped else LOOKS[1])


def _cdf_difference(indices, total):
    """Return the integer numerator of the two-sample CDF distance."""
    small = np.cumsum(np.bincount(indices, minlength=len(total)))
    return np.abs(total[-1] * small - len(indices) * total).max()


def production_check(data):
    """Compare first-look full counts to independent production draws."""
    path = Path(__file__).with_name("burst-count-draws.csv.gz")
    ref = (
        pd.read_csv(path)
        .query("dt == 5")
        .groupby(["seed", "draw"])
        .independent.sum()
    )
    rng = np.random.default_rng(975613)
    out = []
    for seed, g in data[data.draw < LOOKS[0]].groupby("seed"):
        full = (g.full_a + g.full_b).to_numpy()
        independent = ref.loc[seed].to_numpy()
        ns, nl = len(independent), len(full)
        np.testing.assert_equal(ns, 32)
        counts = np.r_[independent, full]
        _, ranks, frequency = np.unique(
            counts, return_inverse=True, return_counts=True
        )
        total = np.cumsum(frequency)

        observed = _cdf_difference(ranks[:ns], total)
        if observed == 0:
            p_value = 1.0
        else:
            exceed = sum(
                _cdf_difference(rng.choice(ranks, ns, replace=False), total)
                >= observed
                for _ in range(1999)
            )
            p_value = (exceed + 1) / 2000
        out.append(
            dict(
                seed=seed,
                control_cdf_gap=observed / (ns * nl),
                control_permutation_p=p_value,
            )
        )
    return pd.DataFrame(out)


def publish(data, output, image):
    """Save the per-history table and optionally its empirical distributions."""
    report = score(data).merge(
        production_check(data), on="seed", validate="one_to_one"
    )
    report.to_csv(
        output / "burst-conditional-qualification.csv",
        index=False,
        float_format="%.10g",
    )
    if image is not None:
        figure(data, image)


if __name__ == "__main__":
    main()
