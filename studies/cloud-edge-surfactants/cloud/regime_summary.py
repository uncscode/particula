"""Compare added target droplets, displaced droplets and matched acid."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REGIONS = ("marine", "smoke", "urban", "soa")


def read(stem):
    """Load a generated parcel table by its stem."""
    with (RESULTS / f"{stem}.csv").open() as src:
        return list(csv.DictReader(src))


def subset(rows, **conditions):
    """Select rows matching the specified inputs."""

    def matches(r):
        return all(
            (float(r[k]) == v if isinstance(v, (float, int)) else r[k] == v)
            for k, v in conditions.items()
        )

    return [r for r in rows if matches(r)]


def assemble():
    """Match the added particles, replacement and dilution treatments."""
    mix = read("gpu_factorial")
    pop = read("gpu_factorial_populations")
    control = read("gpu_attribution")
    tension = read("gpu_tension")
    rows = []
    for reg in REGIONS:
        for w in (0.2, 1.0):
            selector = dict(regime=reg, w=w, entr=0.05, gas=1.0)
            (run,) = subset(mix, **selector)
            members = subset(pop, **selector)
            cm = {r["mode"]: float(r["activated_cm3"]) for r in members}
            total = float(run["activated_cm3"])
            tag = cm["octyl"] + cm["dodecyl"]
            end = {
                r["mix"]: float(r["activated_cm3"])
                for r in subset(control, **selector)
            }
            addition = total - end["none"]
            spectrum = {r["scenario"]: r for r in subset(tension, **selector)}
            (gas_off,) = subset(mix, **{**selector, "gas": 0.0})
            (entr_off,) = subset(mix, **{**selector, "entr": 0.0})
            rows.append(
                dict(
                    regime=reg,
                    updraft_ms=w,
                    octyl_cm3=cm["octyl"],
                    dodecyl_cm3=cm["dodecyl"],
                    background_cm3=total - tag,
                    total_cm3=total,
                    target_activated_cm3=tag,
                    net_added_cm3=addition,
                    background_displaced_cm3=tag - addition,
                    mixed_minus_pure_acid_cm3=total - end["acid"],
                    all_dodecyl_minus_all_octyl_cm3=end["dodecyl"]
                    - end["octyl"],
                    mixed_dodecyl_minus_octyl_cm3=cm["dodecyl"] - cm["octyl"],
                    organic_on_minus_off_cm3=total
                    - float(gas_off["activated_cm3"]),
                    entrainment_on_minus_off_cm3=total
                    - float(entr_off["activated_cm3"]),
                    fixed70_total_cm3=float(
                        spectrum["fixed70"]["activated_cm3"]
                    ),
                    fixed70_dodecyl_minus_octyl_cm3=float(
                        spectrum["fixed70"]["dodecyl_minus_octyl_cm3"]
                    ),
                    adsorption_low_dodecyl_minus_octyl_cm3=float(
                        spectrum["low"]["dodecyl_minus_octyl_cm3"]
                    ),
                    adsorption_high_dodecyl_minus_octyl_cm3=float(
                        spectrum["high"]["dodecyl_minus_octyl_cm3"]
                    ),
                )
            )
    return rows


if __name__ == "__main__":
    rows = assemble()
    with (RESULTS / "regime_comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
