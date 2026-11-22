"""Publish pool identities, rate units and reaction balance checks."""

import csv
import json

from baseline_replay import FAMILIES, HERE
from growth import BANK, CAP, initial

PARENT = ("A", "C", "I")
PRIMARY = [f"{pool}_{tag}" for pool in ("G", "V", "P") for tag in PARENT]
SECONDARY = [
    f"{pool}_{family}" for pool in ("D", "W", "L", "U") for family in FAMILIES
]
NAMES = PRIMARY + ["S", "F"] + SECONDARY


def phase(index):
    """Map physical storage to the gas, particle or terminal ledger."""
    if index < 6 or index >= 26:
        return "gas"
    return "terminal" if index == 10 else "particle"


def main():
    """Ensure the bank has physical source phases and unique kinetic rows."""
    _, incumbent = initial()
    known = set()
    rows = []
    for j, reaction in enumerate(BANK):
        a, b = reaction.a, reaction.b
        inputs = [a, b] if b >= 0 else [a]
        signature = (
            tuple(sorted(inputs)),
            tuple(sorted([reaction.dest, reaction.split])),
            reaction.area_scaled,
        )
        if signature in known or (b >= 0 and phase(a) != phase(b)):
            raise ValueError(
                f"Duplicate or cross-phase reactants in {reaction.key}"
            )
        known.add(signature)
        source = phase(a)
        destinations = [reaction.dest]
        if reaction.split >= 0:
            destinations.append(reaction.split)
        units = "m3 air per umol per hour" if b >= 0 else "per hour"
        if reaction.area_scaled:
            units += " at 0.001 m2 per m3 air"
        balance = "effective organic mass"
        if reaction.key.startswith("accrete_"):
            # These new molecularity proposals must preserve both incoming tags.
            parents = sorted([PARENT[i % 3] for i in inputs])
            if "".join(parents) != NAMES[reaction.dest].split("_")[1]:
                raise ValueError(
                    f"Parent-equivalent imbalance in {reaction.key}"
                )
            balance += "; explicit A/C/I source equivalents"
        rows.append(
            dict(
                index=j,
                key=reaction.key,
                incumbent=j in incumbent,
                source_pools=" + ".join(NAMES[i] for i in inputs),
                destination_pools=" + ".join(NAMES[i] for i in destinations),
                source_phase=source,
                product_phase=" + ".join(phase(i) for i in destinations),
                rate_units=units,
                lower=0,
                upper=CAP[j],
                balance=balance,
                product_mass_fractions="0.5 + 0.5"
                if len(destinations) > 1
                else "1",
            )
        )
    saved = json.loads((HERE / "bank.json").read_text())
    if [r["key"] for r in saved] != [r["key"] for r in rows]:
        raise ValueError("Serialized bank and catalog differ")
    with (HERE / "channel_catalog.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
