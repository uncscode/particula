"""Check tagged CUDA counts against an independent stiff CPU parcel solver."""

import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def read(name):
    """Load the independent numerical endpoint records."""
    with (RESULTS / name).open() as source:
        return list(csv.DictReader(source))


def verify_resolution():
    """Match tagged GPU counts to the CPU reference at both ascent speeds."""
    reference = read("cpu_populations.csv")
    compare = []
    for row in read("gpu_reference_populations.csv"):
        if row["mix"] != "mixed" or row["mode"] not in ("octyl", "dodecyl"):
            continue
        (ref,) = [
            r
            for r in reference
            if int(r["bins_per_population"]) == 128
            and r["scenario"] == "central"
            and (int(r["background_initial_cm3"]) == 1000)
            == (row["regime"] == "benchmark")
            and float(r["updraft_ms"]) == float(row["w"])
            and r["target"] == "mixed"
            and r["population"] == row["mode"]
        ]
        compare.append(
            float(row["activated_cm3"]) - float(ref["activated_cm3"])
        )
    if not (len(compare) == 8 and max(map(abs, compare)) < 0.4):
        raise ValueError("Tagged CUDA counts exceeded CPU reference tolerance")
    return dict(cpu_reference_max_target_difference_cm3=max(map(abs, compare)))


if __name__ == "__main__":
    print(verify_resolution())
