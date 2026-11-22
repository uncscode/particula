"""Count distinct channels and attribute changes in the joint score."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    """Load a saved result without importing the CUDA runtime."""
    return json.loads((HERE / name).read_text())


def motifs(records):
    """Identify an addition by its unordered set of new channels."""
    return {tuple(sorted(row["extra"])) for row in records}


def write_rates(bank, lineage):
    """Export rates in physical units for every saved verified network."""
    with (HERE / "channel_catalog.csv").open() as source:
        units = {
            row["key"]: row["rate_units"] for row in csv.DictReader(source)
        }
    with (HERE / "lineage_rates.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["stage", "id", "key", "equation", "rate", "units"])
        archive = read("size_inputs.json")["cases"]["incumbent"]
        for channel, rate in zip(
            archive["channels"], archive["x"][16:], strict=True
        ):
            key = channel["key"]
            idx = next(row["id"] for row in bank if row["key"] == key)
            writer.writerow(
                ["archive", idx, key, channel["equation"], rate, units[key]]
            )
        for record in lineage:
            for idx in record["ids"]:
                channel = bank[idx]
                key = channel["key"]
                writer.writerow(
                    [
                        f"round {record['round']}",
                        idx,
                        key,
                        channel["equation"],
                        record["x"][16 + idx],
                        units[key],
                    ]
                )


def main():
    """Summarize the final verified lineage and any exploratory screens."""
    bank = read("bank.json")
    lineage = read("lineage.json")
    write_rates(bank, lineage)
    rounds = []
    initial = set(read("selection_01.json")["archive"]["ids"])
    incumbent = initial
    screened_ids = set()
    for item in sorted(HERE.glob("screen_[0-9][0-9].json")):
        n = int(item.stem.split("_")[-1])
        suffix = "" if n == 1 else f"_{n:02d}"
        direct = read(item.name)
        warm = read(f"activation_screen{suffix}.json")
        tests = motifs(direct) | motifs(warm)
        considered = {h for group in tests for h in group}
        screened_ids.update(considered)
        round_summary = dict(
            round=n,
            tested_channels=len(considered),
            direct_motifs=len(motifs(direct)),
            positive_direct_motifs=sum(r["gross"] > 1e-9 for r in direct),
            warmed_routes=len(warm),
            distinct_motifs_screened=len(tests),
        )
        choice = HERE / f"selection_{n:02d}.json"
        if choice.exists() and n <= len(lineage):
            decision = read(choice.name)
            selected = decision["selected"]
            winner = set(selected["ids"])
            additions = winner - incumbent
            previous = next(
                (
                    r
                    for r in decision["compared"]
                    if r["name"] == "previous_verified"
                ),
                decision["archive"],
            )

            def percent(row):
                return 100 * max(
                    row.get("qualification_scores", row["scores"][-2:])
                )

            parent = decision["best_parent"]
            round_summary.update(
                verified=True,
                distinct_motifs_refitted=len(motifs(decision["decisions"])),
                admitted=[bank[i]["key"] for i in sorted(additions)],
                retained_channel_count=len(winner),
                validation_step_seconds=decision.get(
                    "qualification_step_seconds", 1.875
                ),
                previous_percent=percent(previous),
                refitted_parent_percent=percent(parent),
                selected_percent=percent(selected),
                parent_refit_gain_percentage_points=percent(previous)
                - percent(parent),
                growth_gain_percentage_points=percent(parent)
                - percent(selected),
                selected_fit=selected["name"],
            )
            incumbent = winner
        else:
            round_summary.update(verified=False)
            if choice.exists():
                diagnostics = read(choice.name)
                round_summary["distinct_motifs_refitted"] = len(
                    motifs(diagnostics["decisions"])
                )
                round_summary["context"] = (
                    "Sensitivity screen descending from a superseded fit."
                )

        rounds.append(round_summary)
    result = dict(
        channel_cap=1000,
        bank_channels=len(bank),
        inherited=len(initial),
        distinct_additional_channels_screened=len(screened_ids - initial),
        rounds=rounds,
        final_channels=len(incumbent),
        appreciable=read("diagnostics.json")["selected"]["appreciable"],
        appreciable_definition="Gross transfer >= 1 ug/m3 in any run.",
    )
    (HERE / "search_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
