"""Replay saved neural node corrections beside integration-matched controls."""

import argparse
import copy
import csv
from pathlib import Path

import torch
from node_correction import NodeCorrection
from node_fit import evaluate, load
from reduced_exchange import rollout


class Physical:
    """Expose the same forecasting interface for the bare physical control."""

    def __init__(self, scheme):
        """Select integration order for the control."""
        self.extrapolate = scheme == "extrapolated"

    def rollout(self, mass, number, gas, force, substeps):
        """Advance the control on an identical forcing history."""
        return rollout(mass, number, gas, force, substeps, self.extrapolate)


def checkpoints(files, ablate):
    """Load the declared expert architectures and their interventions."""
    models = []
    controls = set()
    for filename in files:
        ckpt = torch.load(filename, map_location="cpu", weights_only=True)
        c = ckpt["config"]
        model = NodeCorrection(
            c["hidden"],
            c["memory"],
            c["scheme"],
            c.get("mixture", False),
            c.get("physical_clock", False),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        solver = c["nodes"], c["scheme"], c["substeps"]
        controls.add(solver)
        models.append((filename.stem, *solver, ckpt["step"], model))
        if model.mixture and ablate:
            for routing in ("uniform", "instant", "recurrent"):
                variant = copy.deepcopy(model)
                variant.routing = routing
                models.append(
                    (
                        filename.stem + "-" + routing,
                        *solver,
                        ckpt["step"],
                        variant,
                    )
                )
    return models, controls


def compare(root, output, files, repeats, control_nodes, ablate, refinement):
    """Use fixed checkpoints and independently refined water histories."""
    torch.set_num_threads(1)
    models, controls = checkpoints(files, ablate)
    for n in control_nodes:
        for scheme in ("implicit", "extrapolated"):
            for substeps in (1, 2):
                controls.add((n, scheme, substeps))
    for n, scheme, substeps in sorted(controls):
        models.append(
            (
                f"physical-n{n}-{scheme}-s{substeps}",
                n,
                scheme,
                substeps,
                0,
                Physical(scheme),
            )
        )
    cohorts = [(root, k, k) for k in ("multi", "2ms", "20ms")] + [
        (root / "repeated" / f"b{b}", "multi", f"independent-{4 * b}plateaus")
        for b in repeats
    ]
    rows = []
    for folder, key, domain in cohorts:
        data = {}
        for label, n, scheme, substeps, step, model in models:
            if n not in data:
                data[n] = load(folder, key, "validation", n)
            scores = evaluate(
                model, data[n], substeps * refinement, schedules=True
            )
            for pattern, metrics in scores.items():
                rows.append(
                    dict(
                        model=label,
                        domain=domain,
                        reset_times_ms=pattern,
                        size_nodes=n,
                        scheme=scheme,
                        substeps=substeps * refinement,
                        update=step,
                        **metrics,
                    )
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(output)
    print(output, len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--files", nargs="*", type=Path, default=[])
    p.add_argument("--repeats", nargs="*", type=int, default=[1])
    p.add_argument("--control-nodes", nargs="*", type=int, default=[4])
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--refinement", type=int, default=1)
    compare(**vars(p.parse_args()))
