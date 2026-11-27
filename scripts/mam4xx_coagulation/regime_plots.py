"""Effects on 24-hour ultrafine number retention in three regimes."""

import matplotlib.pyplot as plt
import numpy as np
from scripts.mam4xx_coagulation.plots import number_per_decade


def plot(report):
    """Show the range across paired environmental and resolution controls."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10, 4.6), sharey=True, layout="constrained"
    )
    labels = [
        "mode width",
        "dispersion",
        "dielectric + dispersion",
        "coarse",
        "charge",
        "ion history",
        "settling",
        "shear",
    ]
    for ax, reg in zip(
        axes, ["marine", "continental", "polluted"], strict=True
    ):
        rows = {
            x["process"]: x for x in report["contrasts"] if x["regime"] == reg
        }
        for y, lab in enumerate(labels):
            x = rows[lab]
            ax.plot([x["low"], x["high"]], [y, y], color="#457c87", lw=3)
            ax.plot(x["effect"], y, "o", color="#173a49")
        ax.axvline(0, color="#888", linestyle=":")
        ax.set(
            title=reg.capitalize(),
            yticks=np.arange(len(labels)),
            yticklabels=labels,
        )
    axes[0].invert_yaxis()
    fig.supxlabel("Change in 24 h sub-100 nm retention (%)")
    return fig


def spectra(report):
    """Show field inputs, native Brownian seed spread and paired outputs."""
    from scipy.stats import t as student

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12, 10.3),
        layout="constrained",
        sharex=True,
        height_ratios=[1, 1, 1.15, 0.7],
    )
    diameter = np.geomspace(1, 10000, 420)
    names = ("Clean marine", "Continental background", "Polluted air")
    for col, (name, (reg, modes)) in enumerate(
        zip(
            names,
            (
                (k, report["input_modes"][k])
                for k in ("marine", "continental", "polluted")
            ),
            strict=True,
        )
    ):
        top, mid, lower, difference = axes[:, col]
        total = np.zeros_like(diameter)
        for (dg, sigma, number), label, color in zip(
            modes,
            ("Aitken", "Accumulation", "Coarse"),
            ("#4b74aa", "#ad6435", "#608343"),
            strict=True,
        ):
            width = np.log10(sigma)
            values = (
                number
                / (np.sqrt(2 * np.pi) * width)
                * np.exp(-0.5 * (np.log10(diameter / (1000 * dg)) / width) ** 2)
            )
            top.plot(diameter, values, label=label, color=color)
            total += values
        top.plot(diameter, total, label="Total", color="#252525", ls="--")
        top.set(title=name, yscale="log", ylim=(0.05, total.max() * 2))
        record = next(
            r["refinement"]
            for r in report["atmospheric_reference"]
            if r["regime"] == reg
        )
        edges = np.array(record["interior_edges_nm"])
        mean = number_per_decade(
            np.array(record["spectrum"]["mean"])[-1, 1:-1], edges
        )
        n = len(record["seeds"])
        sd = (
            number_per_decade(
                np.array(record["spectrum"]["ci95"])[-1, 1:-1], edges
            )
            * np.sqrt(n)
            / student.ppf(0.975, n - 1)
        )
        mid.stairs(
            mean + sd,
            edges,
            baseline=np.maximum(0, mean - sd),
            fill=True,
            color="#42968e",
            alpha=0.35,
        )
        mid.stairs(mean, edges, color="#216b65", label="Particle mean ± 1 SD")
        for law, label, color in [
            ("fine", "Fine sectional", "#424242"),
            ("neutral", "Hard sphere", "#267892"),
            ("dispersion", "Attraction", "#af5525"),
        ]:
            r = next(
                r
                for r in report["cases"]
                if r["regime"] == reg
                and r["mechanism"] == law
                and r["control"] == "central"
            )
            d = np.array(r["diameter_nm"])
            spacing = np.diff(np.log10(d))[0]
            grid_edges = np.r_[
                d * 10 ** (-spacing / 2), d[-1] * 10 ** (spacing / 2)
            ]
            ax = mid if law == "fine" else lower
            ax.stairs(
                number_per_decade(np.array(r["spectrum_cm3"])[-1], grid_edges),
                grid_edges,
                label=label,
                color=color,
                linestyle="--" if law == "fine" else "-",
            )
            if law != "fine":
                # The reporting edges coincide with the original sectional grid.
                np.testing.assert_allclose(grid_edges, edges)
                mc = report["ranking_particles"][reg][law]
                reference = np.array(r["spectrum_cm3"])[-1]
                mean, lo, hi = (
                    np.array(mc[k])[-1, 1:-1] for k in ("mean", "p16", "p84")
                )
                lower.stairs(
                    number_per_decade(hi, edges),
                    edges,
                    baseline=number_per_decade(lo, edges),
                    fill=True,
                    color=color,
                    alpha=0.32,
                )
                lower.stairs(
                    number_per_decade(mean, edges),
                    edges,
                    color=color,
                    ls="--",
                    lw=0.75,
                )
                mask = (mean * mc["box_cm3"] >= 10) & (reference > 0)
                bottom, upper = [
                    100
                    * (
                        np.divide(
                            v,
                            reference,
                            out=np.full_like(mean, np.nan),
                            where=mask,
                        )
                        - 1
                    )
                    for v in (lo, hi)
                ]
                difference.stairs(
                    upper,
                    edges,
                    baseline=bottom,
                    fill=True,
                    color=color,
                    alpha=0.38,
                )
        difference.axhline(0, color="#777", lw=0.65)
        for ax in (top, mid, lower, difference):
            ax.axvline(100, color="#777", lw=0.8, ls=":")
            ax.set(xscale="log", xlim=(1, 10000))
        difference.set_xlabel("Dry diameter (nm)")
    axes[0, 0].set_ylabel(
        "Initial three modes\n" + r"d$N$/dlog$_{10}D$ (cm$^{-3}$)"
    )
    axes[1, 0].set_ylabel(
        "24 h · two fine modes\n" + r"d$N$/dlog$_{10}D$ (cm$^{-3}$)"
    )
    axes[2, 0].set_ylabel(
        "24 h · three modes\n" + r"d$N$/dlog$_{10}D$ (cm$^{-3}$)"
    )
    axes[3, 0].set_ylabel("Particle departure from\nits sectional bin (%)")
    for ax in axes[:3, 2]:
        ax.legend(fontsize=8, loc="upper right")
    return fig


def coefficients(report):
    """Show number and donor-volume factors on each archived state."""
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.3), layout="constrained", sharey=True
    )
    for ax, reg, title in zip(
        axes,
        ("polluted", "marine"),
        ("Polluted air", "Clean marine air"),
        strict=True,
    ):
        for i, j, label, color in [
            (0, 1, "Aitken → accumulation", "#285e70"),
            (0, 2, "Aitken → coarse", "#a64a22"),
            (1, 2, "Accumulation → coarse", "#618047"),
        ]:
            rows = [
                r
                for r in report["host_coefficients"]
                if r["regime"] == reg
                and r["coarse"] == "True"
                and (int(r["donor_index"]), int(r["receiver_index"])) == (i, j)
            ]
            hours = [int(r["hour"]) for r in rows]
            for key, style in [("number_factor", "-"), ("volume_factor", "--")]:
                ax.plot(
                    hours,
                    [float(r[key]) for r in rows],
                    style,
                    marker=".",
                    color=color,
                    label=label if key == "number_factor" else None,
                )
        ax.axhline(1, color="#777", lw=0.75)
        ax.set(title=title, xlabel="Elapsed hours", xticks=[0, 1, 6, 24])
    axes[0].set_ylabel("Attraction / hard-sphere coefficient")
    axes[1].legend(fontsize=8)
    return fig
