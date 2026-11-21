"""Summarize paired modal tests and show stochastic size distributions."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student


def case(rows, **attrs):
    """Select one documented combination."""
    (row,) = [r for r in rows if all(r[k] == v for k, v in attrs.items())]
    return row


def effects(data):
    """Contrast complete attraction with hard spheres on matched inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, reg in zip(axes, ("polluted", "marine"), strict=True):
        groups = [
            ("host", False),
            ("host", True),
            ("field", False),
            ("field", True),
        ]
        for i, (width, coarse) in enumerate(groups):
            keys = dict(regime=reg, widths=width, coarse=coarse)
            for name, res, delta, color, marker in [
                ("modal", 96, -0.16, "#a14920", "s"),
                ("sectional", 32, 0, "#24576b", "o"),
            ]:
                rows = [
                    case(
                        data[name],
                        **keys,
                        law=law,
                        resolution=res,
                        hamaker=data["nominal_hamaker_J"],
                    )
                    for law in ("bare", "complete")
                ]
                v = 100 * (rows[1]["uf_cm3"][-1] / rows[0]["uf_cm3"][-1] - 1)
                ax.plot(
                    v,
                    i + delta,
                    marker,
                    color=color,
                    label=name.capitalize() if i == 0 else None,
                )
            pair = [
                case(data["monte_carlo"], **keys, law=law, resolution=48)
                for law in ("bare", "complete")
            ]
            # Seeds are paired across process choices, including initial sizes.
            raw = [np.array(r["endpoints"])[:, -1, 0] for r in pair]
            change = 100 * (raw[1] / raw[0] - 1)
            ci = (
                student.ppf(0.975, len(change) - 1)
                * change.std(ddof=1)
                / np.sqrt(len(change))
            )
            ax.errorbar(
                change.mean(),
                i + 0.16,
                xerr=ci,
                fmt="^",
                capsize=3,
                color="#567c3c",
                label="Particle MC (95% CI)" if i == 0 else None,
            )
        ax.set_title(
            "Polluted air" if reg == "polluted" else "Clean marine air"
        )
        ax.set_yticks(
            range(4),
            ["MAM fine", "MAM + coarse", "Field fine", "Field + coarse"],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.supxlabel("Change in number below 100 nm at 24 h (%)")
    return fig


def particles(data):
    """Plot exact initial bin counts and the distribution of final outcomes."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(9.5, 7.5),
        layout="constrained",
        sharex=True,
        height_ratios=[1, 1.4, 0.8],
    )
    edges = np.array(data["edges_nm"])
    width = np.diff(np.log10(edges))
    for col, reg in enumerate(("polluted", "marine")):
        initial, final, spread = axes[:, col]
        keys = dict(regime=reg, widths="host", coarse=True, resolution=48)
        for law, lab, color in [
            ("bare", "Hard sphere", "#247185"),
            ("complete", "Attraction", "#ad5224"),
        ]:
            row = case(data["monte_carlo"], **keys, law=law)
            mean = np.array(row["mean"])[-1, 1:-1]
            lo, hi = (np.array(row[k])[-1, 1:-1] for k in ("p16", "p84"))
            if law == "bare":
                initial.stairs(
                    np.array(row["mean"])[0, 1:-1] / width,
                    edges,
                    color="#52575c",
                )
            modal = case(
                data["modal"],
                regime=reg,
                widths="host",
                coarse=True,
                resolution=96,
                hamaker=data["nominal_hamaker_J"],
                law=law,
            )
            final.stairs(
                np.array(modal["spectrum_cm3"])[-1, 1:-1] / width,
                edges,
                color=color,
                ls="--",
                lw=1,
                label=lab + " · modal",
            )
            final.stairs(
                mean / width, edges, color=color, label=lab + " · particles"
            )
            final.stairs(
                hi / width,
                edges,
                baseline=lo / width,
                fill=True,
                color=color,
                alpha=0.35,
            )
            sampled = mean * row["box_cm3"] >= 10

            def deviation(bound, mean=mean, sampled=sampled):
                return 100 * (
                    np.divide(
                        bound,
                        mean,
                        out=np.full_like(mean, np.nan),
                        where=sampled,
                    )
                    - 1
                )

            spread.stairs(
                deviation(hi),
                edges,
                baseline=deviation(lo),
                fill=True,
                color=color,
                alpha=0.4,
            )
        initial.set_title(
            "Polluted air" if reg == "polluted" else "Clean marine air"
        )
        for ax in (initial, final, spread):
            ax.axvline(100, color="#777", ls=":", lw=1)
            ax.set(xscale="log", xlim=(2, 6000))
        spread.axhline(0, color="#555", lw=0.6)
        spread.set_xlabel("Dry diameter (nm)")
    axes[0, 0].set_ylabel(
        "Initial spectrum\n" + r"d$N$/dlog$_{10}D$ (cm$^{-3}$)"
    )
    axes[1, 0].set_ylabel("24 h spectrum\n" + r"d$N$/dlog$_{10}D$ (cm$^{-3}$)")
    axes[2, 0].set_ylabel("24 h spread about\nthe bin mean (%)")
    axes[1, 1].legend(fontsize=8)
    return fig
