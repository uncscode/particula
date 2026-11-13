"""One process-isolated timing case with restored state and explicit CUDA.

Run with ``python -m scripts.gpu_studies.worker --case JSON --mode MODE
--output PATH``. The campaign controller owns wall-clock limits. Published
samples never include correctness checks, state restoration or input creation
inside resident timing. GPU timings include waiting and status observation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from particula._cpu_batch import detached
from scripts.gpu_studies.cases import StudyCase, build_case, radius_histogram
from scripts.gpu_studies.validation import (
    native_cpu_steps,
    require,
    state_hash,
    validate_evolution,
)

ROOT = Path(__file__).resolve().parents[2]
WARMUPS = 3
SAMPLES = 7
MODES = ("numpy", "direct", "prepared", "captured", "legacy", "evolution")


def executable_hash():
    """Hash current library and study sources, including dirty files."""
    digest = hashlib.sha256()
    for folder in (ROOT / "particula", Path(__file__).parent):
        for path in sorted(folder.rglob("*.py")):
            if "tests" in path.parts or path.name == "conftest.py":
                continue
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def summary(samples):
    """Retain every timing plus median, p95 and range in seconds."""
    return {
        "samples_s": samples,
        "median_s": float(np.median(samples)),
        "p95_s": float(np.percentile(samples, 95)),
        "minimum_s": min(samples),
        "maximum_s": max(samples),
    }


class Driver:
    """Use equivalent recorded kernel calls, owner steps, or CUDA replay."""

    def __init__(self, owner, case, mode):
        """Bind a fixed execution path before any timing samples."""
        self.owner, self.case, self.mode = owner, case, mode
        self.operations = owner._operations
        if mode == "legacy":
            if case.family != "brownian":
                raise ValueError("legacy comparison is Brownian-only")
            from particula.gpu.kernels.coagulation import (
                _enqueue_prepared_coagulation_call,
            )

            old = replace(
                owner._bindings["coagulation"], brownian_parallel=None
            )
            self.operations = (lambda: _enqueue_prepared_coagulation_call(old),)
        if mode == "captured":
            owner.capture()

    def run(self):
        """Run the workload, including the explicit source-off boundary."""
        owner = self.owner
        for step in range(self.case.steps):
            if self.mode in ("direct", "legacy"):
                with owner._scope():
                    for operation in self.operations:
                        operation()
            else:
                owner.step()
            if self.case.family == "full" and step == 0:
                gas = owner.diagnostics().gas_concentration
                gas[:, 1] = 0.0
                owner.update_gas(gas)
        owner.synchronize()


def reset_buffers(owner):
    """Retain device copies for identical repeated physics and RNG."""
    wp = owner._wp
    buffers = []
    for carrier, names in (
        (
            owner._session.particles,
            ("masses", "concentration", "charge", "density", "volume"),
        ),
        (
            owner._session.gas,
            ("molar_mass", "concentration", "vapor_pressure", "partitioning"),
        ),
        (
            owner._session.environment,
            ("temperature", "pressure", "saturation_ratio"),
        ),
    ):
        buffers.extend(getattr(carrier, name) for name in names)
    buffers.extend(
        value for _, _, value, _ in owner._registry._enumerate_resources()
    )
    buffers.extend(
        value
        for _, _, value in owner._registry._enumerate_published_rng_streams()
    )
    buffers.extend(owner._auxiliary.values())
    buffers.append(owner._device_failure)
    with owner._scope():
        backups = tuple(wp.clone(value) for value in buffers)
        owner._wait()

    def restore():
        with owner._scope():
            for target, source in zip(buffers, backups, strict=True):
                wp.copy(target, source)
            owner._wait()

    return restore


def _observe(report, case, initial, result):
    """Check physics and exact repeated results outside the timer."""
    science = validate_evolution(case, initial, result)
    digest = state_hash(result)
    if "result_hash" in report:
        require(
            report["result_hash"] == digest,
            "state/RNG restoration changed a repeated trajectory",
        )
    report["result_hash"] = digest
    report["science"] = science
    edges, histogram = radius_histogram(result)
    report["radius_edges_m"] = edges.tolist()
    # Aggregate and a bounded selection preserve shape without huge JSON files.
    indices = np.unique(
        np.linspace(0, case.boxes - 1, min(case.boxes, 16), dtype=int)
    )
    report["histogram_box_indices"] = indices.tolist()
    report["final_histogram_number_m3"] = histogram[indices].tolist()
    report["final_mean_histogram_number_m3"] = histogram.mean(axis=0).tolist()


def _numpy(case, initial, processes, report):
    setup, resident, overall = [], [], []
    for sample in range(WARMUPS + SAMPLES):
        start = time.perf_counter()
        current = detached(initial)
        ready = time.perf_counter()
        result = native_cpu_steps(case, current, processes)
        end = time.perf_counter()
        print(f"CPU sample {sample + 1}: {end - ready:.6g} s", flush=True)
        _observe(report, case, initial, result)
        if sample >= WARMUPS:
            setup.append(ready - start)
            resident.append(end - ready)
            overall.append(end - start)
    report["timings"] = {
        "setup": summary(setup),
        "resident": summary(resident),
        "download": summary([0.0] * SAMPLES),
        "end_to_end": summary(overall),
    }


def _check_memory(case, wp):
    device = wp.get_device("cuda:0")
    require(
        device.is_mempool_enabled,
        "VRAM qualification requires Warp mempool accounting",
    )
    limit = min(6 * 1024**3, int(device.free_memory * 0.75))
    # Includes primaries, retained process scratch, device restore copies and
    # headroom for graph/checkpoint work. This is admission, not a measurement.
    estimate = (
        case.boxes * case.slots * (case.materials * 8 * 30 + 256)
        + 128 * 1024**2
    )
    require(
        estimate < limit,
        f"estimated {estimate} bytes exceeds {limit} byte VRAM budget",
    )
    return {
        "limit_bytes": limit,
        "estimated_bytes": estimate,
        "method": (
            "Warp CUDA mempool high-water mark; "
            "excludes CUDA context/driver allocations"
        ),
    }


def _gpu(case, initial, process, mode, report):
    import warp as wp

    wp.init()
    require(
        wp.is_cuda_available(), "study requires CUDA; CPU fallback is forbidden"
    )
    report["memory"] = _check_memory(case, wp)
    report.update(
        gpu_name=wp.get_device("cuda:0").name,
        gpu_arch=wp.get_device("cuda:0").arch,
        device="cuda:0",
        warp=importlib.metadata.version("warp-lang"),
    )
    resident, download = [], []
    print(f"Preparing {case.name} on cuda:0", flush=True)
    with process.prepare(
        initial, device="cuda:0", time_step=case.time_step, root_seed=case.seed
    ) as owner:
        print("Preparation complete", flush=True)
        driver = Driver(owner, case, mode)
        restore = reset_buffers(owner)
        for sample in range(WARMUPS + SAMPLES):
            restore()
            start = time.perf_counter()
            driver.run()
            end = time.perf_counter()
            print(f"GPU sample {sample + 1}: {end - start:.6g} s", flush=True)
            result = owner.snapshot()
            observed = time.perf_counter()
            _observe(report, case, initial, result)
            if sample >= WARMUPS:
                resident.append(end - start)
                download.append(observed - end)
        report["memory"]["mempool_peak_bytes"] = wp.get_mempool_used_mem_high(
            "cuda:0"
        )
        require(
            report["memory"]["mempool_peak_bytes"]
            <= report["memory"]["limit_bytes"],
            "measured allocation exceeded VRAM budget",
        )
        del restore, driver
    report["timings"] = {
        "resident": summary(resident),
        "download": summary(download),
    }
    if mode != "legacy":
        _fresh_runs(case, initial, process, mode, report)
    report["memory"]["mempool_peak_bytes"] = wp.get_mempool_used_mem_high(
        "cuda:0"
    )
    require(
        report["memory"]["mempool_peak_bytes"]
        <= report["memory"]["limit_bytes"],
        "fresh-run allocation exceeded VRAM budget",
    )


def _fresh_runs(case, initial, process, mode, report):
    """Measure fresh end-to-end runs, including graph recording."""
    setup, capture, overall = [], [], []
    for sample in range(SAMPLES):
        start = time.perf_counter()
        with process.prepare(
            initial,
            device="cuda:0",
            time_step=case.time_step,
            root_seed=case.seed,
        ) as owner:
            owner.synchronize()
            ready = time.perf_counter()
            driver = Driver(owner, case, mode)
            recorded = time.perf_counter()
            driver.run()
            result = owner.snapshot()
            del driver
        finished = time.perf_counter()
        print(f"Fresh run {sample + 1}: {finished - start:.6g} s", flush=True)
        _observe(report, case, initial, result)
        setup.append(ready - start)
        capture.append(recorded - ready)
        overall.append(finished - start)
    report["timings"].update(
        setup=summary(setup),
        capture=summary(capture),
        end_to_end=summary(overall),
    )


def run(case, mode):
    """Return validated measurements or raise without claiming success."""
    initial, process, processes = build_case(case)
    report = {
        "schema_version": 1,
        "case": asdict(case),
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- trusted PATH
            cwd=ROOT,
            text=True,
        ).strip(),
        "executable_sha256": executable_hash(),
        "input_sha256": state_hash(initial),
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "qualified": False,
        "resident_scope": (
            "workload plus synchronization/status; "
            "restoration and analysis excluded"
        ),
        "direct_scope": (
            "retained low-level kernel sequence; "
            "no repeated public-wrapper preflight"
        ),
        "legacy_scope": (
            "previous Brownian selector, retained preparation; "
            "resident timing only"
        ),
    }
    edges, histogram = radius_histogram(initial)
    report["radius_edges_m"] = edges.tolist()
    report["initial_mean_histogram_number_m3"] = histogram.mean(axis=0).tolist()
    if mode == "evolution":
        import warp as wp
        from scripts.gpu_studies.trajectories import run_trajectory

        wp.init()
        require(wp.is_cuda_available(), "scientific study requires CUDA")
        report["memory"] = _check_memory(case, wp)
        started = time.perf_counter()
        report["experiment"] = run_trajectory(case, initial, process, processes)
        report["experiment_wall_seconds"] = time.perf_counter() - started
        report["warmups"] = report["samples"] = None
        report["memory"]["mempool_peak_bytes"] = wp.get_mempool_used_mem_high(
            "cuda:0"
        )
        require(
            report["memory"]["mempool_peak_bytes"]
            <= report["memory"]["limit_bytes"],
            "experiment exceeded VRAM budget",
        )
    elif mode == "numpy":
        _numpy(case, initial, processes, report)
    else:
        _gpu(case, initial, process, mode, report)
    report["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report["working_tree_dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],  # noqa: S607 -- trusted PATH
            cwd=ROOT,
            text=True,
        )
    )
    if mode != "numpy":
        import warp as wp

        report.update(
            device="cuda:0",
            gpu_name=wp.get_device("cuda:0").name,
            warp=importlib.metadata.version("warp-lang"),
        )
        runtime = wp.get_device("cuda:0").runtime
        report["cuda_toolkit"] = runtime.toolkit_version
        report["cuda_driver"] = runtime.driver_version
        report["nvidia_smi"] = subprocess.check_output(
            [  # noqa: S607 -- trusted local driver utility
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    report["qualified"] = True
    return report


def main():
    """Run one case/mode and atomically publish successful output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(StudyCase(**json.loads(args.case)), args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    duration = (
        report["experiment_wall_seconds"]
        if args.mode == "evolution"
        else report["timings"]["resident"]["median_s"]
    )
    print(f"PASS {report['case']['name']} {args.mode}: {duration:.6g} s")


if __name__ == "__main__":
    main()
