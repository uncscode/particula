"""Generate pair-kernel targets during vapor transfer and binary mergers."""

import argparse
from pathlib import Path

import numpy as np
from particula.dynamics import (
    get_brownian_kernel_via_system_state,
    get_coulomb_kernel_gopalakrishnan2012_via_system_state,
    get_hard_sphere_kernel_via_system_state,
)
from stage2_population import wet_radius
from vapor_properties import materials


def run(source, out, kind, windows, seed, wide, charge_max=2):
    """Label particle pairs from vapor and perturbed merger histories."""
    rng = np.random.default_rng(seed)
    spec = materials(2)
    with np.load(source) as a:
        m, q, ids, mix = a["m"], a["q"], a["ids"], a["mix"]
    b, steps, n, s = q.shape
    i, j = np.triu_indices(n, 1)
    rows = []
    for box in range(b):
        dry = m[box, :, -2:].sum(-1)
        for sample in range(windows):
            start = int(rng.integers(steps - 3))
            wet = q[box, start : start + 4] * dry[None, :, None]
            mass = np.concatenate(
                (wet, np.broadcast_to(m[box, :, -2:], (*wet.shape[:2], 2))),
                axis=-1,
            )
            z = (
                np.zeros(n, dtype=int)
                if kind != "charged"
                else rng.integers(-charge_max, charge_max + 1, n)
            )
            zt = np.broadcast_to(z, (4, n)).copy()
            active = np.ones((4, n), dtype=bool)
            if wide:
                # Stress the native pair kernel outside the initial dry
                # spectrum.
                mass *= np.exp(rng.uniform(-1.5, 1.7, (1, n, 1)) * 3)
                if sample % 3 == 0:
                    mass[:, 0] *= 8.0
            if sample % 2:
                # Merge histories are state augmentation for pointwise rate
                # fitting. Recompute native labels on every changed state.
                losers = rng.permutation(n)[: 2 * max(1, n // 8)].reshape(-1, 2)
                for v, (left, right) in enumerate(losers):
                    f = 1 + v % 3
                    mass[f:, left] += mass[f:, right]
                    zt[f:, left] += zt[f:, right]
                    active[f:, right] = False
            r = wet_radius(mass, spec)
            total = mass.sum(-1)
            rates = []
            for frame in range(4):
                args = (r[frame], total[frame])
                if kind == "brownian":
                    target = get_brownian_kernel_via_system_state(
                        *args, 293.0, 101325.0
                    )
                elif kind == "hard-sphere":
                    target = get_hard_sphere_kernel_via_system_state(
                        *args, np.zeros(n), 293.0, 101325.0
                    )
                else:
                    target = (
                        get_coulomb_kernel_gopalakrishnan2012_via_system_state(
                            *args, zt[frame], 293.0, 101325.0
                        )
                    )
                rates.append(target[i, j])
            rows.append(
                (r, total, zt, active, np.stack(rates), ids[box], mix[box])
            )
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out,
        **dict(
            zip(
                ("r", "mass", "charge", "active", "K", "family", "mix"),
                map(np.stack, zip(*rows, strict=True)),
                strict=True,
            )
        ),
        i=i,
        j=j,
        kind=kind,
        seed=seed,
        wide=wide,
        charge_max=charge_max,
    )
    print(out, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "out"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--kind",
        default="brownian",
        choices=["brownian", "hard-sphere", "charged"],
    )
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seed", type=int, default=9893)
    p.add_argument("--charge-max", type=int, default=2)
    p.add_argument("--wide", action="store_true")
    run(**vars(p.parse_args()))
