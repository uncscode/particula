"""Check lookup amortization, fitted weights and vapor clock refinement."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from condensation_options import (
    DT,
    GRID,
    HI,
    LO,
    SPECIES,
    STEPS,
    TrialBox,
    train_head,
)
from process_cost import forecast
from process_scale import score, write_csv
from stage2_conductance import examples


def timed(function):
    """Time one synchronized device operation in milliseconds."""
    torch.cuda.synchronize()
    start = perf_counter()
    out = function()
    torch.cuda.synchronize()
    return out, 1000 * (perf_counter() - start)


def prepare_heads(folder, weight):
    """Recreate the fits and save their weights for replay."""
    inputs = examples(1703, 24000, "organic")
    models = {"step": weight}
    weights = {"step_weight": np.array(weight)}
    for width in (8, 24):
        head = train_head(width, inputs)
        weights.update(
            {
                f"w{width}_{key}": val.numpy()
                for key, val in head.state_dict().items()
            }
        )
        models[width] = head.cuda().eval()
    np.savez(folder / "weights.npz", **weights)
    return models


def coefficients(folder, n, models):
    """Compare direct and table evaluation of the very same fitted head."""
    box = TrialBox(
        "physical",
        70101,
        None,
        None,
        n,
        SPECIES,
        method="physical",
        heads=models,
    )
    radius, _ = box.geometry()
    j = slice(0, 4)
    exact = box.conductance(radius, j)
    grid = torch.linspace(
        LO, HI, GRID, device=box.device, dtype=box.dtype
    ).exp()
    lanes = torch.arange(4, device=box.device)

    def lookup(table):
        t = (radius.log() - LO) * ((GRID - 1) / (HI - LO))
        if bool(((t < 0) | (t > GRID - 1)).any()):
            raise ArithmeticError("Radius outside table")
        i = t.long().clamp_max(GRID - 2)
        return torch.lerp(
            table[i[:, None], lanes],
            table[i[:, None] + 1, lanes],
            (t - i)[:, None],
        ).exp()

    def make_table():
        return box.network(grid, slice(None), 24).log()

    table, _ = timed(make_table)
    setup_times = [timed(make_table)[1] for _ in range(9)]
    methods = {
        "physical": lambda: box.conductance(radius, j),
        "direct8": lambda: box.network(radius, j, 8),
        "direct24": lambda: box.network(radius, j, 24),
        "table24": lambda: lookup(table),
    }
    records = []
    for method, fn in methods.items():
        out, _ = timed(fn)
        err = (out / exact - 1).abs().max().item() * 100
        direct = box.network(radius, j, 24)
        vs_direct = (out / direct - 1).abs().max().item() * 100
        samples = [timed(fn)[1] for _ in range(9)]
        records.append(
            dict(
                method=method,
                hosts=n,
                vapors=4,
                repeats=9,
                median_ms=median(samples),
                max_coefficient_error_pct=err,
                max_vs_direct24_pct=vs_direct,
                table_build_ms=median(setup_times)
                if method == "table24"
                else 0,
            )
        )
    write_csv(folder / "coefficient-probe.csv", records)


@torch.inference_mode()
def main():
    """Audit coefficient timing and a finer clock on one seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    settings = json.loads((folder / "settings.json").read_text())
    n = settings["hosts"]
    models = prepare_heads(folder, settings["learned_step_weight"])
    coefficients(folder, n, models)
    seed = settings["seeds"][0]
    observations = {}
    records = []
    for method in (
        "learned-step",
        "richardson",
        "fused-8",
        "fused-16",
        "fused-32",
        "fused-64",
    ):
        factory = partial(TrialBox, method=method, heads=models)
        opts = dict(hosts=n, species=SPECIES, box_type=factory, dt=DT)
        forecast("physical", 70100, (None, None), steps=1, **opts)
        timing, obs = forecast(
            "physical", seed, (None, None), steps=STEPS, **opts
        )
        observations[method] = obs[:, :-1]
        records.append(
            dict(seed=seed, method=method, total_ms=timing["total_ms"])
        )
        print(
            "reference check",
            method,
            round(timing["total_ms"] / 1000, 2),
            flush=True,
        )
    for row in records:
        row.update(
            score(observations[row["method"]], observations["fused-64"], DT)
        )
    write_csv(folder / "clock-audit.csv", records)


if __name__ == "__main__":
    main()
