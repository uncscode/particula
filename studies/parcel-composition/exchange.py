"""Six-hour conservative exchange between matched synthetic aerosol parcels."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from particula.particles import get_kelvin_radius, get_mixing_state_index
from scipy.special import ndtri

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SS = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0])
SALT_LEAN, SALT_RICH = 0.1, 0.9


def population(blocks):
    """Pair equal numbers and equal dry volumes of rich and lean particles."""
    scale = np.exp(np.log(1.12) * ndtri((np.arange(blocks) + 0.5) / blocks))
    middle = np.cbrt((60.0**3 + 120.0**3) / 2)
    diameter_nm = (scale[:, None] * [60.0, middle, middle, 120.0]).ravel()
    rich_a = np.tile([1.0, 0.0, 0.0, 1.0], blocks)
    return diameter_nm, rich_a


def activated(diameter_nm, salt):
    """Equilibrium activated fraction by dry size and water supersaturation."""
    kelvin_nm = 2e9 * get_kelvin_radius(0.072, 997.0, 0.01801528, 298.15)
    kappa = 0.05 + (0.60 - 0.05) * salt
    log_s = np.sqrt(4 * kelvin_nm**3 / (27 * kappa * diameter_nm**3))
    return 100 * np.expm1(log_s[:, None]) <= SS


def metrics(blocks, bins):
    """Score three reconstructions against particle-resolved activation."""
    diameter, initial = population(blocks)
    # Both dry species have the same assumed density, so volume and mass
    # fractions coincide and dry mass weights are proportional to diameter^3.
    mass = diameter**3
    rich = activated(diameter, SALT_RICH)
    lean = activated(diameter, SALT_LEAN)
    # Log bins over 40–180 nm, including open tails in the outer bins.
    edges = np.geomspace(40, 180, bins + 1)[1:-1]
    group = np.searchsorted(edges, diameter)
    count = np.bincount(group, minlength=bins)
    initial_profile = np.divide(
        np.bincount(group, weights=initial, minlength=bins),
        count,
        out=np.zeros(bins),
        where=count > 0,
    )
    np.testing.assert_allclose(np.average(initial, weights=mass), 0.5)
    np.testing.assert_allclose(initial.mean(), 0.5)
    bulk = (rich.mean(axis=0) + lean.mean(axis=0)) / 2
    rows = []
    for hour in np.arange(0, 6.01, 0.5):
        # Each equal-volume box sends 0.20 of its inventory per hour.
        contrast = np.exp(-2 * 0.20 * hour)
        prob_a = 0.5 + contrast * (initial - 0.5)
        profile = 0.5 + contrast * (initial_profile - 0.5)
        responses = []
        for parcel, prob, p_bin in [
            ("A", prob_a, profile),
            ("B", 1 - prob_a, 1 - profile),
        ]:
            np.testing.assert_allclose(prob.mean(), 0.5, atol=1e-13)
            bulk_salt = np.average(SALT_LEAN + 0.8 * prob, weights=mass)
            np.testing.assert_allclose(bulk_salt, 0.5, atol=1e-13)
            # Class totals give the same χ as listing every represented
            # particle, since all particles within a class have one composition.
            rich_mass = np.average(prob, weights=mass)
            classes = np.array([[SALT_RICH, SALT_LEAN], [SALT_LEAN, SALT_RICH]])
            chi = get_mixing_state_index(
                classes * np.array([rich_mass, 1 - rich_mass])[:, None]
            )
            exact = (prob[:, None] * rich + (1 - prob[:, None]) * lean).mean(0)
            local = p_bin[group, None]
            histogram = (local * rich + (1 - local) * lean).mean(0)
            mean_only = activated(diameter, SALT_LEAN + 0.8 * local[:, 0]).mean(
                0
            )
            responses.append(exact)
            for i, ss in enumerate(SS):
                rows.append(
                    dict(
                        hour=hour,
                        parcel=parcel,
                        supersaturation_pct=ss,
                        chi=chi,
                        activation=exact[i],
                        size_independent=bulk[i],
                        conditional_mean=mean_only[i],
                        conditional_distribution=histogram[i],
                        diameter_bins=bins,
                        # W1 between two binary distributions in salt fraction
                        # equals their rich-probability difference times 0.8.
                        between_parcels_w1=np.average(
                            0.8 * abs(2 * profile - 1), weights=count
                        ),
                    )
                )
        np.testing.assert_allclose(np.mean(responses, axis=0), bulk, atol=2e-12)
    return rows


def size_profile(blocks):
    """Return the plotted salt-rich fraction in each occupied dry-size bin."""
    diameter, initial = population(blocks)
    edges = np.geomspace(40, 180, 13)
    group = np.searchsorted(edges[1:-1], diameter)
    count = np.bincount(group, minlength=12)
    mask = count > 0
    centers = np.sqrt(edges[:-1] * edges[1:])[mask]
    fraction = (
        np.bincount(group, weights=initial, minlength=12)[mask] / count[mask]
    )
    return centers, fraction


def figure(rows, size_profile, output):
    """Plot size-specific composition, activation spectra and closure errors."""
    subset = [r for r in rows if r["diameter_bins"] == 12]
    fig, (state_ax, *axes) = plt.subplots(
        1, 3, figsize=(13, 3.7), layout="constrained"
    )
    centers, fraction = size_profile
    for p, color, label in [
        (fraction, "#065e77", "A"),
        (1 - fraction, "#a6441d", "B"),
    ]:
        state_ax.plot(centers, 100 * p, "o-", color=color, label=label)
        state_ax.plot(
            centers, 100 * (0.5 + (p - 0.5) * np.exp(-2.4)), "--", color=color
        )
    state_ax.set(
        xlabel="Dry diameter (nm)",
        ylabel="Salt-rich particles by number (%)",
        title="Which particles carry the salt?",
    )
    state_ax.legend(frameon=False, title="Parcel")
    for parcel, color in [("A", "#065e77"), ("B", "#a6441d")]:
        selected = [
            r for r in subset if r["parcel"] == parcel and r["hour"] == 0
        ]
        axes[0].plot(
            SS,
            [100 * r["activation"] for r in selected],
            "o-",
            color=color,
            label=f"Parcel {parcel}",
        )
        axes[0].plot(
            SS,
            [100 * r["conditional_distribution"] for r in selected],
            "x",
            color=color,
        )
    axes[0].set(
        xlabel="Water supersaturation (%)",
        ylabel="Activated number (%)",
        title="Equal χ, different activation at hour 0",
    )
    axes[0].legend(frameon=False)
    for key, label, color in [
        ("size_independent", "Composition histogram alone", "#555555"),
        ("conditional_mean", "Mean salt fraction by size", "#a6441d"),
        (
            "conditional_distribution",
            "Composition histogram by size",
            "#065e77",
        ),
    ]:
        hours = sorted({r["hour"] for r in subset})
        error = [
            100
            * np.sqrt(
                np.mean(
                    [
                        (r[key] - r["activation"]) ** 2
                        for r in subset
                        if r["hour"] == h
                    ]
                )
            )
            for h in hours
        ]
        axes[1].plot(hours, error, color=color, label=label)
    axes[1].set(
        xlabel="Parcel exchange time (h)",
        ylabel="Activation RMSE (percentage points)",
        title="What survives compression?",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    """Run the exchange and save its numerical table and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks", type=int, default=25000, help="quartets per parcel"
    )
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("blocks must be positive")
    rows = [
        row for bins in (3, 6, 12, 24) for row in metrics(args.blocks, bins)
    ]
    local = Path(__file__).resolve().parent
    with (local / "exchange.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    figure(
        rows,
        size_profile(args.blocks),
        ROOT / "docs/images/parcel-composition-exchange.png",
    )


if __name__ == "__main__":
    main()
