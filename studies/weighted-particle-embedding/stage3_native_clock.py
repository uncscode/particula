"""Test the native vapor clock with the collision operator disabled."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_evidence import write
from stage3_native_off_score import score


def run(root, until):
    """Replay the same one-second box at progressively finer vapor clocks."""
    import warp as wp
    from stage3_native_scale import measure

    folder = root / "native-clock"
    folder.mkdir(parents=True, exist_ok=True)
    wp.init()
    for n in (4096, 65536):
        for steps in (128, 256, 512):
            if time() >= until:
                return
            path = folder / f"{n}-{steps}.json"
            if path.exists():
                continue
            row = measure(
                n, 32, 1e12, 307, 311001, 1.0, 1, steps, 1, 1, collisions=False
            )[0]
            if any(h["n"] != n for h in row["history"][0]):
                raise ArithmeticError("Number changed in a vapor-only forecast")
            row["warp_pool_high_water_bytes"] = int(
                wp.get_mempool_used_mem_high("cuda:0")
            )
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(row) + "\n")
            tmp.replace(path)
            print(n, steps, row["advance_and_host_diagnostics_s"], flush=True)


def report(root, results):
    """Compare the coupled and exchange-only errors over the clock ladder."""
    rows = score(root, results, "native-clock")
    write(results / "native-vapor-clock-refinement.csv", rows)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained")
    targets = ("Full physical dynamics", "Vapor integration alone")
    for ax, target in zip(axes, targets, strict=True):
        for n in sorted({r["n"] for r in rows}):
            data = [r for r in rows if r["n"] == n and r["control"] == target]
            ax.plot(
                [r["vapor_steps_per_half"] for r in data],
                [r["phase_upper_pct"] for r in data],
                "o-",
                label=f"{n:,}",
            )
        ax.axhline(0.1, color="#666", ls="--", lw=1)
        ax.set(
            title=target,
            xlabel="Vapor substeps per half interval",
            ylabel="Upper phase mass error (%)",
            xticks=(128, 256, 512),
        )
        ax.legend(title="Explicit particles", frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(results / "native-vapor-clock-refinement.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument(
        "--until", type=lambda x: datetime.fromisoformat(x).timestamp()
    )
    operation.add_argument("--score", type=Path)
    args = parser.parse_args()
    if args.score:
        report(args.root, args.score)
    else:
        run(args.root, args.until)
