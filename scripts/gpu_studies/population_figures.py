"""Readable dynamics figures and data-derived descriptions for box ensembles."""

import matplotlib.pyplot as plt
import numpy as np


def _facets(labels, title, width=4):
    """Use one condition per panel so legends never encode nineteen curves."""
    height = (len(labels) + width - 1) // width
    fig, axes = plt.subplots(
        height,
        width,
        figsize=(13, 2.5 * height),
        constrained_layout=True,
        squeeze=False,
    )
    for i, ax in enumerate(axes.flat):
        if i >= len(labels):
            ax.set_visible(False)
        else:
            ax.set_title(labels[i], fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.15)
    fig.suptitle(title)
    return fig, axes.flat


def population_figures(report, labels, *, time_scale=1.0, time_unit="s"):
    """Plot overview, size spectra, charge evolution and cumulative mergers.

    Histogram ordinates are dN/(N_initial dlog10 D), never a per-bin PDF.
    Charge changes and losses retain absolute counts. All panels preserve
    supplied physical order; conditions are not ranked by noisy outcomes.
    """
    history = report["history"]
    time = np.array([r["time"] for r in history])
    shown_time = time * time_scale
    number = np.array([r["number"] for r in history])
    total = number.sum(axis=1)
    initial = total[0]
    mergers = initial - total
    quantiles = (
        np.array([r["pooled_diameter_quantiles"] for r in history]) * 1e9
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(shown_time, mergers, marker=".")
    ax.set(
        xlabel=f"Time [{time_unit}]",
        ylabel="Cumulative accepted mergers",
        title="Aerosol number loss = aerosol mergers",
    )
    ax.secondary_yaxis(
        "right",
        functions=(
            lambda n: 100 * n / initial,
            lambda percent: percent * initial / 100,
        ),
    ).set_ylabel("Initial number lost [%]")
    axes[0, 1].stairs(np.diff(mergers) / np.diff(time), shown_time)
    axes[0, 1].set(
        xlabel=f"Time [{time_unit}]",
        ylabel="Accepted mergers / s",
        title="Interval-average rate · all boxes pooled",
    )
    axes[1, 0].fill_between(
        shown_time, quantiles[:, 0], quantiles[:, 2], alpha=0.2, label="D10–D90"
    )
    axes[1, 0].plot(shown_time, quantiles[:, 1], label="D50")
    axes[1, 0].set(
        xlabel=f"Time [{time_unit}]",
        ylabel="Diameter [nm]",
        title="Number quantiles · all conditions pooled",
    )
    axes[1, 0].legend()
    categories = [
        lambda q: q == 0,
        lambda q: abs(q) == 1,
        lambda q: abs(q) >= 2,
    ]
    for select, label in zip(
        categories, ["Neutral", "|q| = 1", "|q| ≥ 2"], strict=True
    ):
        values = [
            sum(sum(v) for q, v in r["charge_counts"].items() if select(int(q)))
            for r in history
        ]
        axes[1, 1].plot(shown_time, 100 * np.array(values) / total, label=label)
    axes[1, 1].set(
        xlabel=f"Time [{time_unit}]",
        ylabel="Current particles [%]",
        title="Charge-state fractions · all boxes pooled",
    )
    axes[1, 1].legend()
    fig.suptitle(f"Dynamics overview · {initial:,} initial aerosols")

    fig, axes = _facets(labels, "Size distributions over time")
    sample = sorted(set([0, 1, len(history) // 2, len(history) - 1]))
    edges = np.array(report["diameter_edges"])
    centers = np.sqrt(edges[:-1] * edges[1:]) * 1e9
    widths = np.diff(np.log10(edges))
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        for j in sample:
            density = np.array(history[j]["diameter_histogram"][i]) / (
                number[0, i] * widths
            )
            ax.plot(centers, density, label=f"{shown_time[j]:g} {time_unit}")
        ax.set(
            xscale="log",
            xlim=(10, 3000),
            xlabel="Diameter [nm]",
            ylabel="dN / (N₀ dlog₁₀ D)",
        )
    handles, legend = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncols=4)

    fig, axes = _facets(
        labels, "Charge population changes · aerosol population"
    )
    selectors = [
        lambda q: q <= -2,
        lambda q: q == -1,
        lambda q: q == 0,
        lambda q: q == 1,
        lambda q: q >= 2,
    ]
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        for select, label in zip(
            selectors,
            ["q ≤ −2", "q = −1", "q = 0", "q = +1", "q ≥ +2"],
            strict=True,
        ):
            values = np.array(
                [
                    sum(
                        v[i]
                        for q, v in row["charge_counts"].items()
                        if select(int(q))
                    )
                    for row in history
                ]
            )
            ax.plot(shown_time, values - values[0], label=label)
        ax.set(xlabel=f"Time [{time_unit}]", ylabel="Δ particle count")
    handles, legend = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncols=5)

    fig, axes = _facets(
        labels, "Cumulative collisions and particle-number loss"
    )
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        ax.step(shown_time, number[0, i] - number[:, i], where="post")
        ax.set(xlabel=f"Time [{time_unit}]", ylabel="Accepted mergers = N₀ − N")
    return number


def describe_population(report, labels):
    """Print measured changes without claiming statistical significance."""
    first, last = report["history"][0], report["history"][-1]
    n0, n1 = np.array(first["number"]), np.array(last["number"])
    mergers = n0 - n1
    d0, d1 = (
        first["pooled_diameter_quantiles"][1] * 1e9,
        last["pooled_diameter_quantiles"][1] * 1e9,
    )
    print(
        f"Over {last['time']:.6g} s: {mergers.sum():,} accepted mergers; "
        f"{n1.sum():,} particles remain "
        f"({100 * mergers.sum() / n0.sum():.5f}% number loss)."
    )
    print(
        f"Pooled number-median diameter changes from {d0:.3f} to {d1:.3f} nm."
    )
    print(
        f"Largest recorded merger count: {labels[int(np.argmax(mergers))]} "
        f"({mergers.max():,}); smallest: {labels[int(np.argmin(mergers))]} "
        f"({mergers.min():,})."
    )
    print("Condition | initial count | final count | mergers | final D50 [nm]")
    for i, label in enumerate(labels):
        print(
            f"{label} | {n0[i]} | {n1[i]} | {mergers[i]} | "
            f"{last['diameter_quantiles'][i][1] * 1e9:.3f}"
        )
    print(
        f"Maximum mass residual: {report['maximum_mass_relative_error']:.3e}; "
        "net charge is exactly conserved in every box."
    )
    print(
        "One fixed-seed realization: rankings and interval-rate fluctuations "
        "are descriptive, not significance tests."
    )


def ion_figures(report, labels):
    """Separate free ions, attachment, recombination and source events."""
    history = report["history"]
    time = np.array([r["time"] for r in history]) * 1e3
    fig, axes = _facets(labels, "Free-ion populations during water growth")
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        for key, label in [
            ("free_positive", "+ ions"),
            ("free_negative", "− ions"),
        ]:
            ax.plot(time, [r[key][i] for r in history], label=label)
        ax.set(xlabel="Time [ms]", ylabel="Free ions")
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=2,
    )
    fig, axes = _facets(
        labels, "Cumulative ion events · balances checked against material mass"
    )
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        for key, label in [
            ("attachment_positive", "+ attachment"),
            ("attachment_negative", "− attachment"),
            ("recombination", "Recombination"),
            ("injected_pairs", "Injected pairs"),
        ]:
            ax.plot(time, [r[key][i] for r in history], label=label)
        ax.set(xlabel="Time [ms]", ylabel="Events")
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=4,
    )
    last = history[-1]
    print(
        "Ion events, pooled: "
        + "; ".join(
            f"{key}: {sum(last[key]):,.0f}"
            for key in (
                "attachment_positive",
                "attachment_negative",
                "recombination",
                "injected_pairs",
            )
        )
    )
    print(
        "Aerosol number falls only through aerosol-aerosol mergers. "
        "Attachment transfers ion charge and mass without removing an aerosol; "
        "recombination removes two free ions and creates one "
        "inert neutral product."
    )
