"""Stream larger native forecasts so each completed repetition is auditable."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import time

import warp as wp
from stage3_native_scale import measure


def run(folder, until, million=False):
    """Retain native 32-vapor advances with the same full physical solver."""
    folder.mkdir(parents=True, exist_ok=True)
    wp.init()
    jobs = [
        (262144, 535001, 3),
        (131072, 535001, 4),
        (262144, 535004, 12),
    ]
    if million:
        jobs = [(1000000, 546001, 3)]
    for n, first, repeats in jobs:
        if time() >= until:
            break
        seeds = [
            k
            for k in range(first, first + repeats)
            if not (folder / f"native-{n}-{k}.json").exists()
        ]
        if not seeds:
            continue

        def save(row, size=n):
            path = folder / f"native-{size}-{row['first_seed']}.json"
            row["completed_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                row["warp_pool_high_water_bytes"] = int(
                    wp.get_mempool_used_mem_high("cuda:0")
                )
            except RuntimeError as error:
                row["allocator_measurement_error"] = str(error)
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps([row]) + "\n")
            tmp.replace(path)
            print(
                path.name,
                row["setup_s"],
                row["advance_and_host_diagnostics_s"],
                flush=True,
            )
            return time() + row["advance_and_host_diagnostics_s"] * 1.12 < until

        measure(
            n,
            32,
            1e12,
            307,
            seeds[0],
            1.0,
            1,
            128,
            1,
            len(seeds),
            on_result=save,
            first_seeds=seeds,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--million", action="store_true")
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    a = p.parse_args()
    run(a.folder, a.until, a.million)
