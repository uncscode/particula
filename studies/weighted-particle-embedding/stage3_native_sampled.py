"""Run sampled collisions between native vapor half steps."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch
from stage3_one_box import Box


class ResidentPairs:
    """Update occupied native particle slots on the device."""

    def __init__(
        self, species, volume, charge, kernel, device="cuda:0", parts=1
    ):
        """Load the chosen pair law and its random generator."""
        # The two-slot fixture initializes the existing pair law and its RNG.
        # Each call binds the actual native arrays before computing any rates.
        self.box = Box(2, species, 2 / volume, charge, device, 307, 1, kernel)
        self.kernel = kernel
        self.parts = parts
        self.reset([1])

    def reset(self, seeds):
        """Start a reproducible single-box event trajectory."""
        if len(seeds) != 1:
            raise ValueError("The resident collision study requires one box")
        self.box.generator.manual_seed(int(seeds[0]))
        self.box.max_probability = 0.0
        self.steps = self.pairs = self.mergers = 0
        self.seconds = 0.0

    @torch.inference_mode()
    def update(self, mass, weights, charge, dt):
        """Merge all species and signed charge in place for accepted pairs."""
        live = torch.nonzero(weights == 1, as_tuple=True)[0]
        self.steps += 1
        if len(live) < 2:
            return
        self.box.mass = mass
        self.box.z = charge
        i, j = self.box.merger_pairs(dt, True, live)
        self.pairs += len(live) // 2
        self.mergers += len(i)
        if len(i):
            mass[i] += mass[j]
            charge[i] += charge[j]
            mass[j] = 0
            charge[j] = 0
            weights[j] = 0

    def __call__(self, particles, dt):
        """Synchronize the native and tensor streams around the collision."""
        import warp as wp

        wp.synchronize_device("cuda:0")
        tic = perf_counter()
        mass = wp.to_torch(particles.masses)[0]
        weights = wp.to_torch(particles.population)[0]
        charge = wp.to_torch(particles.charge)[0]
        for _ in range(self.parts):
            self.update(mass, weights, charge, dt / self.parts)
        torch.cuda.synchronize()
        self.seconds += perf_counter() - tic

    def diagnostics(self):
        """Return the separate collision cost and number ledger."""
        return dict(
            kernel="native-vapor-sampled-" + self.kernel,
            collision_steps=self.steps,
            substeps_per_collision_interval=self.parts,
            pairs_tested=self.pairs,
            accepted_mergers=self.mergers,
            max_probability=self.box.max_probability,
            separate_collision_s=self.seconds,
            torch_peak_bytes=torch.cuda.max_memory_allocated(),
        )


def jobs(sizes, first, repeats, ablation):
    """Order the fixed-vapor controls, then add independent repetitions."""
    for rep in range(repeats):
        seed = first + rep
        if ablation and rep < 2:
            for n, method in (
                (4096, "learned"),
                (4096, "physical"),
                (4096, "off"),
                (262144, "off"),
                (1000000, "off"),
                (1000000, "learned"),
                (1000000, "physical"),
                (262144, "physical"),
                (262144, "learned"),
                (131072, "off"),
                (131072, "physical"),
                (131072, "learned"),
            ):
                yield n, method, seed, "neutral", 1
            if rep == 0:
                for n, z in [
                    (4096, "bipolar"),
                    (65536, "bipolar"),
                    (1000000, "bipolar16"),
                ]:
                    for parts in (1, 4):
                        for method in ("physical", "learned"):
                            yield n, method, seed, z, parts
            continue
        for n in [4096] if ablation else sizes:
            for k in np.random.default_rng(seed + n).permutation(
                ["learned", "physical"]
            ):
                yield n, str(k), seed, "neutral", 1


def time_available(folder, n, until):
    """Use measured native vapor cost to avoid an unfinished long run."""
    controls = sorted((folder / "native-off").glob(f"native-off-{n}-*.json"))
    if not controls:
        return True
    base = json.loads(controls[0].read_text())
    typical = base["advance_and_host_diagnostics_s"] + base["setup_s"]
    return time() + 1.08 * typical + 3 < until


def run(folder, until, sizes, repeats, first, ablation):
    """Measure both pair laws and native uptake at fixed resolution."""
    import warp as wp
    from stage3_native_scale import measure

    wp.init()
    torch.set_num_threads(1)
    for n, kernel, seed, z, parts in jobs(sizes, first, repeats, ablation):
        if time() >= until:
            return
        off = kernel == "off"
        cache = (
            folder / ("native-off" if off else "native-sampled")
            if ablation
            else folder
        )
        cache.mkdir(parents=True, exist_ok=True)
        filename = (
            f"native-off-{n}-{seed}.json"
            if off
            else f"native-sampled-{n}-{kernel}-{seed}.json"
        )
        if z != "neutral":
            filename = f"native-sampled-{n}-{z}-{parts}-{kernel}-{seed}.json"
        path = cache / filename
        if path.exists():
            continue
        if ablation and not time_available(folder, n, until):
            continue

        def factory(box, charge, k=kernel, substeps=parts):
            return ResidentPairs(
                box.spec.species, box.volume, charge, k, parts=substeps
            )

        torch.cuda.reset_peak_memory_stats()
        try:
            row = measure(
                n,
                32,
                1e12,
                307,
                seed,
                1.0,
                1,
                128,
                1,
                1,
                charge=z,
                collisions=False,
                sampled_factory=None if off else factory,
            )[0]
            if row["history"][0][-1]["n"] != n - row.get("accepted_mergers", 0):
                raise ArithmeticError("Native and sampled count ledgers differ")
        except (ArithmeticError, RuntimeError, ValueError) as err:
            failed = folder / "native-sampled-failures"
            failed.mkdir(parents=True, exist_ok=True)
            (failed / filename).write_text(
                json.dumps(dict(n=n, kernel=kernel, seed=seed, reason=str(err)))
                + "\n"
            )
            print("Failed", n, kernel, seed, str(err), flush=True)
            continue
        row["warp_pool_high_water_bytes"] = int(
            wp.get_mempool_used_mem_high("cuda:0")
        )
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(path)
        print(
            n, kernel, seed, row["advance_and_host_diagnostics_s"], flush=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--sizes", nargs="+", type=int, default=[4096, 262144, 1000000]
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--first", type=int, default=311001)
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--warp-site", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    a = p.parse_args()
    if a.warp_site:
        sys.path.append(str(a.warp_site.resolve()))
    run(a.folder, a.until, a.sizes, a.repeats, a.first, a.ablation)
