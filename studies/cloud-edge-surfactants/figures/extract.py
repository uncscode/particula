"""Extract displayed model values from a regenerated cloud-edge study."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
RECIPES = ("octyl", "dodecyl")
PLACEMENTS = ("patterned", "opposed")


def records(folder, name):
    """Read a named replay output from disk."""
    with (folder / name).open() as stream:
        return list(csv.DictReader(stream))


def save(folder, name, rows):
    """Write figure coordinates without duplicating full simulation outputs."""
    with (folder / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def equilibrium(source, dest):
    """Reduce the size scans to displayed central curves and envelope edges."""
    bounds = records(source, "budget.csv")
    inventory = []
    for recipe in RECIPES:
        by_scenario = {
            r["scenario"]: float(r["minimum_dry_um"])
            for r in bounds
            if r["recipe"] == recipe
        }
        edges = [by_scenario[k] for k in ("low", "central", "high")]
        inventory.append(
            dict(
                recipe=recipe,
                central_um=by_scenario["central"],
                low_um=min(edges),
                high_um=max(edges),
            )
        )
    save(dest, "inventory.csv", inventory)
    sizes = defaultdict(dict)
    for row in records(source, "equilibrium_scans.csv"):
        size = float(row["dry_diameter_nm"])
        sizes[size][row["model"], row["scenario"], row["recipe"]] = row
    contrast, activation = [], []
    for size, rows in sorted(sizes.items()):
        scenarios = {s for m, s, r in rows if m == "finite" and r == "acid"}

        def difference(scenario, rows=rows):
            octyl, dodecyl = [
                float(rows["finite", scenario, recipe]["sigma70_mN_m"])
                for recipe in RECIPES
            ]
            return dodecyl - octyl

        delta = [difference(s) for s in scenarios]
        contrast.append(
            dict(
                dry_nm=size,
                central_mN_m=difference("central"),
                low_mN_m=min(delta),
                high_mN_m=max(delta),
            )
        )
        if size > 1000:
            continue
        for recipe in RECIPES:

            def relative(model, scenario, rows=rows, recipe=recipe):
                field = "critical_supersaturation_pct"
                acid = float(rows[model, scenario, "acid"][field])
                value = float(rows[model, scenario, recipe][field])
                return 100 * (value - acid) / acid

            finite = [relative("finite", s) for s in scenarios]
            activation.append(
                dict(
                    dry_nm=size,
                    recipe=recipe,
                    finite_pct=relative("finite", "central"),
                    direct_pct=relative("bulk", "central"),
                    low_pct=min(finite),
                    high_pct=max(finite),
                )
            )
    save(dest, "tension-contrast.csv", contrast)
    save(dest, "activation-shift.csv", activation)


def parcels(source, dest):
    """Extract the three seed classes and the two reciprocal edge profiles."""
    cloud = source / "cloud/results"
    pools = defaultdict(dict)
    for r in records(cloud, "gpu_identity_populations.csv"):
        if (
            r["regime"] in ("urban", "marine")
            and r["scenario"] in ("central", "direct", "fixed70")
            and r["mix"] == "mixed"
            and (float(r["w"]), float(r["entr"]), float(r["gas"]))
            == (0.2, 0.05, 1.0)
        ):
            values = pools[r["regime"], r["scenario"]]
            if r["mode"] in values:
                raise ValueError("Ambiguous identity case")
            values[r["mode"]] = float(r["activated_cm3"])
    identity = [
        dict(
            regime=regime,
            model=model,
            octyl_cm3=counts["octyl"],
            dodecyl_cm3=counts["dodecyl"],
            background_cm3=sum(
                v for k, v in counts.items() if k not in RECIPES
            ),
        )
        for (regime, model), counts in sorted(pools.items())
    ]
    if len(identity) != 6:
        raise ValueError("Need both backgrounds under all three tension models")
    save(dest, "identity.csv", identity)
    profiles = []
    for r in records(cloud, "milestones.csv"):
        if float(r["updraft_ms"]) == 1 and float(r["entrained_fraction"]) in (
            0,
            0.15,
        ):
            profiles.append(
                dict(
                    resident=r["resident"],
                    exchange=float(r["entrained_fraction"]),
                    height_m=float(r["height_m"]),
                    activated_cm3=float(r["activated_cm3"]),
                )
            )
    if (
        len(
            {
                (r["resident"], r["exchange"], round(r["height_m"]))
                for r in profiles
            }
        )
        != 24
        or len(profiles) != 24
    ):
        raise ValueError("Expected six heights for each of four profiles")
    save(dest, "cloud-edge.csv", profiles)


def attribution(source, dest):
    """Pair paths and calculate flux means and control contrasts."""
    data = records(source / "cloud/results/edge-attribution", "main.csv")
    lookup = {(r["placement"], r["control"], int(r["inlet"])): r for r in data}
    if len(lookup) != len(data) or len(data) != 2 * 4 * 128:
        raise ValueError(
            "Expected 128 matched inlets, two placements and four controls"
        )
    controls = ("acid", "finite", "no_targets", "fixed70")
    means = {}
    for place in PLACEMENTS:
        for control in controls:
            rows = [lookup[place, control, i] for i in range(128)]
            weights = [float(r["base_speed_ms"]) for r in rows]
            for i, row in enumerate(rows):
                if any(
                    row[k] != lookup[place, "finite", i][k]
                    for k in (
                        "base_speed_ms",
                        "transit_s",
                        "entrained_fraction",
                        "ambient_rh",
                    )
                ):
                    raise ValueError(
                        "Unequal forcing in paired attribution controls"
                    )
            means[place, control] = {
                k: float(
                    np.average([float(r[k]) for r in rows], weights=weights)
                )
                for k in ("total_cm3", "target_cm3")
            }
    toc = [
        dict(
            control=c,
            shift_gain_cm3=means["opposed", c]["total_cm3"]
            - means["patterned", c]["total_cm3"],
        )
        for c in ("acid", "finite", "fixed70")
    ]
    save(dest, "toc.csv", toc)
    effects = []
    for p in PLACEMENTS:
        finite = means[p, "finite"]
        effects.append(
            dict(
                placement=p,
                tagged_cm3=finite["target_cm3"],
                net_seed_addition_cm3=finite["total_cm3"]
                - means[p, "no_targets"]["total_cm3"],
                recipe_minus_acid_cm3=finite["total_cm3"]
                - means[p, "acid"]["total_cm3"],
            )
        )
    save(dest, "edge-attribution-effects.csv", effects)
    paths = []
    for i in range(128):
        # Inlet sampling in the periodic eddy spans one length from L/64.
        position = 240 * (1 / 64 + i / 128)
        row = dict(inlet_m=position)
        for c in ("finite", "acid"):
            row[c + "_shift_cm3"] = float(
                lookup["opposed", c, i]["total_cm3"]
            ) - float(lookup["patterned", c, i]["total_cm3"])
        paths.append(row)
    save(dest, "edge-attribution-paths.csv", paths)


def main():
    """Compile graph-only model coordinates from the study's run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=STUDY,
        help="Replay root containing the equilibrium CSVs and cloud/results",
    )
    parser.add_argument("--output", type=Path, default=STUDY / "plot_data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    equilibrium(args.source, args.output)
    parcels(args.source, args.output)
    attribution(args.source, args.output)


if __name__ == "__main__":
    main()
