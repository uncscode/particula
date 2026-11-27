"""Resolve water, organic vapor exchange and coagulation on separate clocks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from material_bank import mixture
from stage2_population import fixture, observable_names
from stage2_reference import reference
from stage3_multirate import Multirate, run


class SpeciesClocks(Multirate):
    """Split transfer while retaining every species in the activities."""

    def vapor(self, state, dt, steps, order="joint", net=None):
        """Subcycle the prescribed water and organic operators."""
        mode = self.options.mode
        if mode == "joint":
            return super().vapor(state, dt, steps, order, net)
        if self.options.block:
            from stage3_block_exchange import BlockExchange

            if not hasattr(self, "blocks"):
                self.blocks = BlockExchange(
                    self.spec,
                    self.device,
                    self.options.transfer,
                    self.options.organic_groups,
                    getattr(self.options, "water_partners", 0),
                )
            return self.blocks.advance(
                state,
                self.transfer,
                dt,
                steps,
                mode,
                self.options.water_steps,
                self.options.organic_steps,
                self.work,
            )
        model = self.transfer
        water = torch.arange(self.spec.species, device=self.device) == 0
        # The times sum to one full elapsed interval for each process.
        w = self.options.water_steps
        o = self.options.organic_steps
        if mode == "strang":
            schedule = [(water, 0.5, w), (~water, 1.0, o), (water, 0.5, w)]
        elif mode == "organic-outer":
            schedule = [(~water, 0.5, o), (water, 1.0, w), (~water, 0.5, o)]
        else:
            schedule = [(water, 1.0, w), (~water, 1.0, o)]
        try:
            for _ in range(steps):
                for lanes, frac, sub in schedule:
                    model.active_lanes = lanes
                    super().vapor(state, dt * frac / steps, sub, order="joint")
                    key = "water_substeps" if lanes[0] else "organic_substeps"
                    self.work[key] = self.work.get(key, 0) + sub
        finally:
            model.active_lanes = None


def record(job):
    """Generate physical event-time references for one chemistry."""
    a, f, m, d, seed, tol = job
    name = (
        f"f{f}-m{m}-n{a.slots}-d{d}-{a.reference_process}-"
        f"{a.charge}-r{seed}-full-t{tol:.0e}-LSODA-f8-l4.npz"
    )
    dest = a.folder / name
    if dest.exists():
        return
    spec = mixture(a.species, a.chemistry)
    b = fixture(f, m, a.slots, d, a.process, a.charge, 8, 4, chemistry=spec)
    tic = perf_counter()
    history, stats = reference(
        b, seed, tol, True, "brownian" if a.charge == "zero" else "charged"
    )
    tmp = dest.with_suffix(".part.npz")
    np.savez_compressed(
        tmp,
        obs=history,
        time=b.t,
        names=observable_names(spec),
        seconds=perf_counter() - tic,
        **stats,
    )
    tmp.replace(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["reference", "forecast"])
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--mode",
        choices=["joint", "strang", "organic-outer", "water-first"],
        default="strang",
    )
    for k, v in [
        ("species", 2),
        ("chemistry", 9177),
        ("water-steps", 2),
        ("organic-steps", 1),
        ("slots", 16),
        ("seeds", 16),
        ("start", 331001),
        ("batch", 8),
        ("workers", 2),
    ]:
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--families", nargs="+", type=int, default=[85001, 85002])
    p.add_argument("--mix", nargs="+", type=int, default=[0, 1])
    p.add_argument("--densities", nargs="+", type=int, default=[1, 8])
    p.add_argument("--tol", nargs="+", type=float, default=[1e-7, 1e-9])
    p.add_argument("--block", action="store_true")
    p.add_argument("--organic-groups", type=int, default=1)
    p.add_argument("--water-partners", type=int, default=0)
    p.add_argument("--transfer", default="radau-3-4")
    p.add_argument("--kernel", default="brownian")
    p.add_argument("--charge", default="zero")
    p.add_argument("--device", default="cpu")
    p.add_argument("--vtol", type=float, default=0.003)
    p.add_argument("--ctol", type=float, default=0.02)
    a = p.parse_args()
    if a.water_partners and (not a.block or a.mode == "joint"):
        p.error("Water partners require an active-block split")
    a.plateau = 0
    a.event_aware = True
    a.process = "coupled" if a.charge == "zero" else "charge"
    a.reference_process = f"reversible-s{a.species}-c{a.chemistry}"
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if a.action == "reference":
        jobs = (
            (a, *x)
            for x in product(
                a.families,
                a.mix,
                a.densities,
                range(a.start, a.start + a.seeds),
                a.tol,
            )
        )
        with ProcessPoolExecutor(a.workers) as pool:
            for i, _ in enumerate(pool.map(record, jobs), 1):
                if i % 8 == 0:
                    print(i, flush=True)
    else:
        if not a.block and a.mode != "joint" and a.transfer != "radau-3-4":
            raise ValueError(
                "Learned conductance uses joint or active-block solves"
            )
        tag = f"{a.reference_process}-{a.mode}"
        tag += f"-w{a.water_steps}-o{a.organic_steps}"
        if a.block:
            tag += f"-block{a.organic_groups}"
            if a.water_partners:
                tag += f"-partners{a.water_partners}"
        if a.transfer != "radau-3-4":
            tag += "-" + Path(a.transfer).stem
        run(
            a,
            SpeciesClocks,
            mixture(a.species, a.chemistry),
            tag,
        )
