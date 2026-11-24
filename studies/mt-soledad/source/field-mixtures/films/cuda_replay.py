"""Replay each individual particle at three exposures and five film scenarios."""

import json
import sys
import time

import numpy as np
import pandas as pd
import warp as wp
from film import EDGES, make_data
from models import average
from outlet import HERE, transport


@wp.kernel
def grow(
    cdf: wp.array2d(dtype=wp.float64),
    seed_cdf: wp.array2d(dtype=wp.float64),
    edges: wp.array(dtype=wp.float64),
    maps: wp.array4d(dtype=wp.float64),
    coated_fraction: wp.float64,
    n: int,
    seed: int,
    counts: wp.array4d(dtype=wp.int32),
):
    win, i = wp.tid()
    rng = wp.rand_init(seed, win * n + i)
    u = wp.float64(wp.randf(rng))
    j = int(0)
    while j < 19 and u > cdf[win, j]:
        j += 1
    wet = edges[j] + wp.float64(wp.randf(rng)) * (edges[j + 1] - edges[j])
    v = wp.float64(wp.randf(rng))
    jseed = int(0)
    while jseed < seed_cdf.shape[1] - 1 and v > seed_cdf[win, jseed]:
        jseed += 1
    selected = wp.float64(wp.randf(rng)) < coated_fraction
    for scenario in range(5):
        state = scenario
        if scenario >= 3 and not selected:
            state = 0
        for t in range(3):
            b = int(0)
            while b < 21 and wet >= maps[win, jseed, state * 3 + t, b]:
                b += 1
            wp.atomic_add(counts, win, scenario, t, b, 1)


def run(n):
    start = time.monotonic()
    cls = pd.read_csv(HERE / "inputs.csv")
    a = (
        cls.query("filter_id=='EPP349' and recipe=='high_C16_max2'")
        .sort_values("utc")
        .iloc[0]
    )
    c = make_data()["test"]
    maps = []
    truth = []
    for i in range(len(c["utc"])):
        rows, _, lut = transport(
            c,
            i,
            a.organic_mass_fraction,
            a.c16_organic_fraction,
            a.kappa08,
            a.density,
            513,
            True,
        )
        maps.append(lut)
        truth.append(np.array([x[4] for x in rows]).reshape(5, 3))
    maps = np.array(maps)
    maps = maps.reshape(*maps.shape[:2], 15, 21)
    d = "cuda:0"
    wc = wp.zeros((len(maps), 5, 3, 22), dtype=wp.int32, device=d)
    wp.launch(
        grow,
        dim=(len(maps), n),
        inputs=[
            wp.array(np.cumsum(c["b"], axis=1), dtype=wp.float64, device=d),
            wp.array(np.cumsum(c["weights"], axis=1), dtype=wp.float64, device=d),
            wp.array(EDGES, dtype=wp.float64, device=d),
            wp.array(maps, dtype=wp.float64, device=d),
            max(0.210347173986785, a.c16_organic_fraction),
            n,
            72194,
            wc,
        ],
        device=d,
    )
    v = wc.numpy()
    assert np.all(v.sum(axis=3) == n)
    ratio = (n - v[:, :, :, 0]) / (n - v[:, :1, :, 0])
    pred = average(np.array(truth), c["day"])
    mc = average(ratio, c["day"])
    report = dict(
        device=wp.get_device(d).name,
        warp=wp.__version__,
        particles_per_window=n,
        windows=len(maps),
        paired_particles=n * len(maps),
        scenario_exposures=15,
        filter_id=a.filter_id,
        recipe=a.recipe,
        utc=int(a.utc),
        max_count_difference=float(abs(pred - mc).max()),
        exact=pred.tolist(),
        cuda=mc.tolist(),
        inventory_error=0,
        seconds=time.monotonic() - start,
    )
    (HERE / "final" / f"cuda-{n}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    run(int(sys.argv[1]))
