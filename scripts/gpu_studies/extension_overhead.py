"""Measure prepared extension costs on CUDA with an explicit source revision.

Run native mode against the pre-feature and final checkouts using the same
interpreter, device, dimensions and step counts. Run resident and transfer modes
against the final checkout. Timings include one explicit synchronization per
sample, exclude setup/compilation, and are host wall time per completed step.
This bounded microbenchmark does not establish general scientific throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


def run(args):
    """Measure fully synchronized batches without competing GPU workloads."""
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import warp as wp
    from particula import Aerosol
    from particula.dynamics import Dilution, DilutionStrategy
    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    if not wp.is_cuda_available():
        raise RuntimeError("real CUDA is required; there is no CPU fallback")
    if args.capture and args.mode == "transfer":
        raise ValueError("CPU transfers cannot be captured")
    initial = Aerosol(
        particles=ParticleData(
            masses=np.full((args.boxes, args.slots, 1), 1e-18),
            concentration=np.ones((args.boxes, args.slots)),
            charge=np.zeros((args.boxes, args.slots)),
            density=np.array([1000.0]),
            volume=np.full(args.boxes, 1e-6),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["inert"],
            np.array([0.029]),
            np.full((args.boxes, 1), 1e-6),
            np.array([False]),
        ),
        environment=EnvironmentData(
            np.full(args.boxes, 298.15),
            np.full(args.boxes, 101325.0),
            np.zeros((args.boxes, 1)),
        ),
    )
    pipeline = Dilution(DilutionStrategy(0.01))
    kwargs = {}
    if args.mode != "native":
        # Load the published implementation, keeping the benchmark's scientific
        # work identical between the resident and explicit transfer paths.
        import runpy

        from particula import CPUTransfer, FieldSpec, ProcessRegistry

        example = runpy.run_path(
            str(
                args.source_root
                / ("docs/examples/gpu/prepared/custom_process_backends.py")
            )
        )
        process_type = example["PrescribedWarming"]
        process = process_type()
        initial.fields.add(FieldSpec("demo.warming", ("box",), "K", minimum=0))
        if args.mode == "transfer":
            pipeline = CPUTransfer(process) | pipeline
        else:
            registry = ProcessRegistry()
            registry.register(process_type, example["WarmingAdapter"]())
            kwargs["process_registry"] = registry
            pipeline = process | pipeline
    dt, warmup = 0.001, 20
    samples = []
    with pipeline.prepare(
        initial, device="cuda:0", time_step=dt, **kwargs
    ) as sim:
        if args.capture:
            sim.capture()
        sim.run(warmup)
        sim.synchronize()
        for _ in range(args.repeats):
            started = time.perf_counter()
            sim.run(args.steps)
            sim.synchronize()
            samples.append((time.perf_counter() - started) / args.steps)
        final = sim.snapshot()
        total_steps = warmup + args.steps * args.repeats
        if sim.completed_steps != total_steps:
            raise RuntimeError("benchmark did not complete the requested steps")
        expected = initial.gas.concentration * np.exp(-0.01 * dt * total_steps)
        np.testing.assert_allclose(
            final.gas.concentration, expected, rtol=1e-11
        )
        if args.mode != "native":
            change = 0.01 * (total_steps * dt) ** 2
            np.testing.assert_allclose(
                final.fields["demo.warming"], change, rtol=1e-10, atol=1e-12
            )
    source_hash = hashlib.sha256()
    for path in sorted((args.source_root / "particula").rglob("*.py")):
        source_hash.update(str(path.relative_to(args.source_root)).encode())
        source_hash.update(path.read_bytes())
    result = dict(
        revision=args.revision,
        package_source_sha256=source_hash.hexdigest(),
        benchmark_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        numpy=np.__version__,
        python=sys.version.split()[0],
        mode=args.mode,
        capture=args.capture,
        boxes=args.boxes,
        slots=args.slots,
        steps_per_sample=args.steps,
        warmup_steps=warmup,
        seconds_per_step=samples,
        median_seconds_per_step=statistics.median(samples),
        minimum_seconds_per_step=min(samples),
        maximum_seconds_per_step=max(samples),
        measurement="host wall time; synchronized batches; setup excluded",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    """Select source, execution path and bounded sample sizes explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--mode", choices=("native", "resident", "transfer"), default="native"
    )
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--boxes", type=int, default=8)
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.boxes, args.slots, args.steps, args.repeats) < 1:
        parser.error("dimensions and sample counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
