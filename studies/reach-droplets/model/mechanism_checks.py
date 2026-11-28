"""Repeat the fan-off transport and merger controls discussed in the study."""

import os

from chamber import run_case
from resolved_check import simulate
from workspace import WORK


def merger_pulse(time):
    return 1.0, 15000.0 if 3 <= time < 5 else 1.0


if __name__ == "__main__":
    root = WORK / "mechanisms"
    os.environ["OPTICAL_OUTPUT"] = str(root / "sectional")
    run_case("unforced", "4608", "jets", "standard", 0.025, 2, "cuda:0", scales=(-1.5, 0.0))
    for name, scales, schedule in (
        ("transport", (-1.5, 0.0), None),
        ("mergers", (1.0, 1.0), merger_pulse),
    ):
        simulate(
            "unforced",
            "jets",
            "standard",
            mesh="1944",
            n=262144,
            scales=scales,
            pair_schedule=schedule,
            output=root / name,
        )
