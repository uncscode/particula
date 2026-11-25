"""Compute flux-weighted edge contrasts and their numerical resolution."""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results/edge-attribution"
CONTROLS = ("no_targets", "acid", "finite", "fixed70")
PLACEMENTS = ("patterned", "opposed")


def read(level):
    """Read endpoints indexed by matched exposure, treatment and inlet."""
    with (DATA / f"{level}.csv").open() as src:
        return {
            (r["placement"], r["control"], int(r["inlet"])): r
            for r in csv.DictReader(src)
        }


def write(name, rows):
    """Save a derived flux-weighted table."""
    with (DATA / name).open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def flux_summary(data, level, indices):
    """Average endpoint concentrations with inlet vertical-flux weights."""
    out = []
    for placement in PLACEMENTS:
        for control in CONTROLS:
            rows = [data[placement, control, i] for i in indices]
            weights = np.array([float(r["base_speed_ms"]) for r in rows])
            for i, r in zip(indices, rows, strict=True):
                original = data[placement, "finite", i]
                if not (
                    all(
                        r[k] == original[k]
                        for k in (
                            "base_speed_ms",
                            "transit_s",
                            "entrained_fraction",
                            "ambient_rh",
                        )
                    )
                ):
                    raise ValueError(
                        "Matched treatments follow different trajectories"
                    )
            out.append(
                dict(
                    level=level,
                    inlets=len(indices),
                    placement=placement,
                    control=control,
                    **{
                        k: float(
                            np.average(
                                [float(r[k]) for r in rows], weights=weights
                            )
                        )
                        for k in (
                            "total_cm3",
                            "target_cm3",
                            "resident_cm3",
                            "incoming_cm3",
                            "cutoff_width_cm3",
                        )
                    },
                )
            )
    return out


def contrasts(means):
    """Separate activation on targets from their net addition to the cloud."""
    table = {(r["placement"], r["control"]): r for r in means}
    out = []
    for placement in PLACEMENTS:
        row = table[placement, "finite"]
        out.append(
            dict(
                level=row["level"],
                inlets=row["inlets"],
                placement=placement,
                tagged_cm3=row["target_cm3"],
                net_seed_addition_cm3=row["total_cm3"]
                - table[placement, "no_targets"]["total_cm3"],
                recipe_minus_acid_cm3=row["total_cm3"]
                - table[placement, "acid"]["total_cm3"],
                conservative_width_cm3=row["cutoff_width_cm3"]
                + table[placement, "acid"]["cutoff_width_cm3"],
                fixed_minus_finite_total_cm3=table[placement, "fixed70"][
                    "total_cm3"
                ]
                - row["total_cm3"],
                fixed_minus_finite_tagged_cm3=table[placement, "fixed70"][
                    "target_cm3"
                ]
                - row["target_cm3"],
            )
        )
    return out


def shift_response(data, label, inlets):
    """Paired field effects and their change under chemistry substitution."""
    selected = range(0, 128, 128 // inlets)
    means = flux_summary(data, label, selected)
    lookup = {(r["placement"], r["control"]): r for r in means}
    effect = {}
    for control in CONTROLS:
        effect[control] = (
            lookup["opposed", control]["total_cm3"]
            - lookup["patterned", control]["total_cm3"]
        )
    ratios = []
    weights = []
    for i in selected:
        pairs = []
        for placement in PLACEMENTS:
            pairs.append(
                float(data[placement, "finite", i]["total_cm3"])
                - float(data[placement, "acid", i]["total_cm3"])
            )
        ratios.append(pairs[1] - pairs[0])
        weights.append(float(data["patterned", "acid", i]["base_speed_ms"]))
    interaction = effect["finite"] - effect["acid"]
    if not (
        np.isclose(interaction, np.average(ratios, weights=weights), atol=1e-10)
    ):
        raise ValueError("Direct and weighted interaction contrasts disagree")
    return dict(
        level=label,
        inlets=inlets,
        **{c + "_shift_cm3": effect[c] for c in CONTROLS},
        recipe_by_placement_cm3=interaction,
        conservative_cutoff_width_cm3=sum(
            lookup[p, c]["cutoff_width_cm3"]
            for p in PLACEMENTS
            for c in ("acid", "finite")
        ),
        local_interaction_min_cm3=min(ratios),
        local_interaction_max_cm3=max(ratios),
    )


def main():
    """Save the placement and chemistry contrasts on each spatial grid."""
    primary = read("main")
    fine = read("fine")
    means = flux_summary(primary, "main", range(128))
    means16 = flux_summary(primary, "main_matched", range(0, 128, 8))
    fine16 = flux_summary(fine, "fine", range(0, 128, 8))
    effects = contrasts(means)
    write("flux_means.csv", means + means16 + fine16)
    write(
        "paired_effects.csv", effects + contrasts(means16) + contrasts(fine16)
    )
    write(
        "shift_contrasts.csv",
        [shift_response(primary, "main", n) for n in (8, 16, 32, 64, 128)]
        + [shift_response(fine, "fine", 16)],
    )


if __name__ == "__main__":
    main()
