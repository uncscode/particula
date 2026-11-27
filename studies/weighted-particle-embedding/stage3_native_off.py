"""Remove only coagulation from the original native CUDA forecast."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import time

from stage3_native_scale import measure


def run(folder, until):
    """Measure the same native vapor graph on the registered size ladder."""
    import warp as wp

    folder.mkdir(parents=True, exist_ok=True)
    wp.init()
    # Establish a small matched control, then prioritize the user's largest box.
    for rep in range(3):
        for n in (4096, 262144, 1000000, 131072):
            if time() >= until:
                return
            seed = 546001 + rep
            path = folder / f"native-off-{n}-{seed}.json"
            if path.exists():
                continue
            result = measure(
                n, 32, 1e12, 307, seed, 1.0, 1, 128, 1, 1, collisions=False
            )[0]
            if any(h["n"] != n for h in result["history"][0]):
                raise ArithmeticError(
                    "Aerosol count changed without collisions"
                )
            result["warp_pool_high_water_bytes"] = int(
                wp.get_mempool_used_mem_high("cuda:0")
            )
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(result) + "\n")
            tmp.replace(path)
            print(
                n,
                seed,
                result["setup_s"],
                result["advance_and_host_diagnostics_s"],
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    a = p.parse_args()
    run(a.folder, a.until)
