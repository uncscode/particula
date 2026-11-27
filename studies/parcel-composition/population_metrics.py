"""Compare particle-number and dry-mass metrics during parcel exchange."""

import csv
from pathlib import Path

import numpy as np
from exchange import SALT_LEAN, SALT_RICH, population
from scipy.special import entr, rel_entr


def metrics_over_time(blocks=25000, bins=12):
    """Compare dependence within parcels and separation between parcels."""
    diameter, initial = population(blocks)
    edges = np.geomspace(40, 180, bins + 1)[1:-1]
    group = np.searchsorted(edges, diameter)
    sizes = np.bincount(group, minlength=bins) / len(group)
    rich = np.bincount(group, weights=initial, minlength=bins) / len(group)
    start = np.column_stack([rich, sizes - rich])
    independent = np.outer(sizes, start.sum(axis=0))
    dry_mass = diameter**3 / np.sum(diameter**3)
    bin_mass = np.bincount(group, weights=dry_mass, minlength=bins)
    salt = SALT_LEAN + (SALT_RICH - SALT_LEAN) * initial
    salt_mass = np.bincount(group, weights=dry_mass * salt, minlength=bins)
    initial_inventory = np.column_stack([salt_mass, bin_mass - salt_mass])
    within_particle = entr([SALT_RICH, SALT_LEAN]).sum() / np.log(2)
    records = []
    for hour in np.arange(0, 6.01, 0.5):
        retained = (1 + np.exp(-0.4 * hour)) / 2
        parcel_a = retained * start + (1 - retained) * start[:, ::-1]
        parcel_b = parcel_a[:, ::-1]
        midpoint = (parcel_a + parcel_b) / 2
        info_a = rel_entr(parcel_a, independent).sum() / np.log(2)
        info_b = rel_entr(parcel_b, independent).sum() / np.log(2)
        js_bits = (
            rel_entr(parcel_a, midpoint).sum()
            + rel_entr(parcel_b, midpoint).sum()
        ) / (2 * np.log(2))
        inventory = (
            retained * initial_inventory
            + (1 - retained) * initial_inventory[:, ::-1]
        )
        bulk = inventory.sum(axis=0)
        information_total = entr(bulk).sum() / np.log(2) - within_particle
        between_sizes = rel_entr(
            inventory, np.outer(bin_mass, bulk)
        ).sum() / np.log(2)
        active = bin_mass > 0
        conditional = inventory[active] / bin_mass[active, None]
        within_sizes = (
            bin_mass[active] @ entr(conditional).sum(axis=1) / np.log(2)
            - within_particle
        )
        np.testing.assert_allclose(
            between_sizes + within_sizes, information_total, atol=1e-13
        )
        total_variation = abs(parcel_a - parcel_b).sum() / 2
        # For the two fixed salt fractions the conditional transport cost
        # is their separation times excess rich probability in each size.
        w1 = (SALT_RICH - SALT_LEAN) * abs(
            parcel_a[:, 0] - parcel_b[:, 0]
        ).sum()
        np.testing.assert_allclose([info_a, info_b], js_bits, atol=1e-13)
        np.testing.assert_allclose(w1, 0.8 * total_variation, atol=1e-13)
        records.append(
            dict(
                hour=hour,
                mi_a_bits=info_a,
                mi_b_bits=info_b,
                joint_js_bits=js_bits,
                joint_tv=total_variation,
                mass_species_information_bits=information_total,
                mass_between_sizes_bits=between_sizes,
                mass_within_sizes_bits=within_sizes,
                size_conditional_w1=w1,
            )
        )
    return records


if __name__ == "__main__":
    rows = metrics_over_time()
    with Path(__file__).with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
