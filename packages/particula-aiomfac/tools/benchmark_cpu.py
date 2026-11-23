"""Report warmed direct CPU latency and traced peak allocation, including outputs."""

import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from particula_aiomfac import AIOMFACModel, MixtureSpec

ref = Path(__file__).resolve().parents[1] / "reference/cases"
case = json.loads((ref / "mixed.json").read_text())
registry = json.loads((ref / "components.json").read_text())
model = AIOMFACModel.prepare(
    MixtureSpec(
        {
            k: {int(i): v for i, v in registry[k].items()}
            for k in case["components"]
        }
    )
)
for size in (1, 128, 4096, 65536):
    masses = np.tile(case["fractions"], (size, 1))
    masses[:, 0] *= np.linspace(0.9, 1.1, size)
    model.evaluate(masses, 298)
    times = []
    for _ in range(7):
        start = time.perf_counter()
        model.evaluate(masses, 298)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    model.evaluate(masses, 298)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(
        f"{size:6} states  {1000 * statistics.median(times):8.3f} ms  {peak / 1024**2:6.3f} MiB"
    )
