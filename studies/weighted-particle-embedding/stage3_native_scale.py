"""Native Warp baseline for the same neutral explicit-particle scaling boxes."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import warp as wp
from particula.gas import get_molecule_mean_free_path
from stage2_native_split import NativeSplit
from stage3_charge_modes import initial_charge
from stage3_spectra import dry_radius
from vapor_properties import Materials


def box(n, ns, density, family, end, charge="neutral"):
    # Use the same SI box as the Torch study, and invert native mean-free-path
    # scaling to realize the synthetic lane-specific lengths through molar mass.
    """Construct the scaling box with native material and transport data."""
    i = np.arange(n)
    u = ((i + 1) * 0.618033988749895 + family * 0.0331) % 1
    radius = dry_radius(u, family, np.where)
    j = np.arange(ns)
    diff = 5e-6 + 1.2e-5 * (1 + j) / (1 + ns)
    alpha = 0.08 + 0.91 * (1 + (j * 7) % ns) / (1 + ns)
    path = 3 * diff / (150 + 150 * (j + 1) / ns)
    l0 = get_molecule_mean_free_path(
        molar_mass=1.0, temperature=293.0, pressure=101325.0
    )
    molar = (l0 / path) ** 2
    spec = Materials(
        tuple(f"vapor_{i}" for i in range(ns)),
        np.r_[molar, 0.1],
        np.full(ns + 1, 1400.0),
        np.full(ns, 1e-30),
        diff,
        alpha,
    )
    mass = np.zeros((n, ns + 1))
    mass[:, -1] = 4 * np.pi / 3 * 1400 * radius**3
    gas = (1 + j / ns) * 1e-8 / ns
    charges = initial_charge(i, u, charge, np.where).astype(float)
    return SimpleNamespace(
        spec=spec,
        mass=mass,
        charge=charges,
        volume=n / density,
        gas=gas,
        t=np.linspace(0, end, 5),
        levels=gas[None],
        masks=np.ones((1, ns), bool),
    )


def measure(
    n,
    ns,
    density,
    family,
    seed,
    h,
    steps,
    micro,
    boxes,
    repeats,
    charge="neutral",
    *,
    on_result=None,
    collisions=True,
    sampled_factory=None,
):
    """Time native coupled simulations and verify their inventories."""
    from particula.gpu.kernels._coagulation_config import (
        CoagulationMechanismConfig,
    )

    b = box(n, ns, density, family, h, charge)
    mech = (
        None
        if charge == "neutral"
        else CoagulationMechanismConfig(
            mechanisms=("charged_hard_sphere",),
            charged_model="gopalakrishnan2012",
        )
    )
    inputs = [b] * boxes
    wp.synchronize_device("cuda:0")
    t = perf_counter()
    sampled = sampled_factory(b, charge) if sampled_factory else None
    solver = NativeSplit(
        inputs,
        steps,
        micro,
        mechanism_config=mech,
        collisions=collisions,
        sampled_collision=sampled,
    )
    wp.synchronize_device("cuda:0")
    setup = perf_counter() - t
    out = []
    for rep in range(repeats):
        start_utc = datetime.now(timezone.utc).isoformat()
        t = perf_counter()
        obs = solver.run(
            inputs, range(seed + rep * boxes, seed + (rep + 1) * boxes)
        )
        wp.synchronize_device("cuda:0")
        elapsed = perf_counter() - t
        end_utc = datetime.now(timezone.utc).isoformat()
        hist = []
        for i in range(boxes):
            count = obs[i, :, 0] * b.volume
            pm = obs[i, :, 1 : ns + 2]
            gas = obs[i, :, ns + 2 : 2 * ns + 2]
            total = pm[:, :ns] + gas
            drift = float(
                np.max(np.abs((total - total[0]) / np.maximum(total[0], 1e-30)))
            )
            signed = obs[i, :, 2 * ns + 3] * b.volume
            if np.max(abs(signed - b.charge.sum())) > 1e-7:
                raise ArithmeticError("Native signed charge changed")
            if drift > 1e-8 or np.any(np.diff(count) > 1e-6) or np.any(gas < 0):
                raise ArithmeticError(f"Native inventory failed {drift}")
            hist.append(
                [
                    dict(
                        n=round(number),
                        gas=g.tolist(),
                        particle=np.r_[p[-1], p[:-1]].tolist(),
                        ledger=drift,
                    )
                    for number, g, p in zip(count, gas, pm, strict=True)
                ]
            )
        out.append(
            dict(
                n=n,
                species=ns,
                density=density,
                charge=charge,
                kernel="native-warp" if collisions else "native-warp-off",
                family=family,
                first_seed=seed + rep * boxes,
                boxes=boxes,
                horizon=h,
                steps=steps,
                vapor_micro=micro,
                setup_s=setup,
                advance_and_host_diagnostics_s=elapsed,
                advance_start_utc=start_utc,
                advance_end_utc=end_utc,
                max_ledger=max(x["ledger"] for series in hist for x in series),
                history=hist,
            )
        )
        if sampled is not None:
            out[-1].update(sampled.diagnostics())
        if on_result is not None and not on_result(out[-1]):
            break
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    for k, v in [
        ("n", [16, 256, 1024]),
        ("s", [2, 32]),
        ("steps", [1, 4]),
        ("micro", [1, 4]),
    ]:
        p.add_argument("--" + k, nargs="+", type=int, default=v)
    p.add_argument("--density", nargs="+", type=float, default=[1e12])
    p.add_argument("--charge", nargs="+", default=["neutral"])
    p.add_argument("--family", type=int, default=101)
    p.add_argument("--seed", type=int, default=232001)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--boxes", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    wp.init()
    a.folder.mkdir(exist_ok=True, parents=True)
    from itertools import product

    for n, s, d, k, m, z in product(
        a.n, a.s, a.density, a.steps, a.micro, a.charge
    ):
        f = a.folder / (
            f"native-{z}-{n}-{s}-{d:g}-{a.family}-{a.seed}-"
            f"{a.horizon:g}-{k}-{m}-{a.boxes}-{a.repeats}.json"
        )
        if f.exists():
            continue
        try:
            rows = measure(
                n,
                s,
                d,
                a.family,
                a.seed,
                a.horizon,
                k,
                m,
                a.boxes,
                a.repeats,
                z,
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            rows = [
                dict(
                    n=n,
                    species=s,
                    density=d,
                    charge=z,
                    kernel="native-warp",
                    family=a.family,
                    first_seed=a.seed,
                    boxes=a.boxes,
                    horizon=a.horizon,
                    steps=k,
                    vapor_micro=m,
                    status="failed",
                    why=str(error),
                )
            ]

        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(rows) + "\n")
        tmp.replace(f)
        print(
            f.name,
            rows[-1].get("advance_and_host_diagnostics_s", rows[-1].get("why")),
            flush=True,
        )
