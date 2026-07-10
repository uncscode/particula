# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GPU Benchmark Results: Condensation & Coagulation
#
# This notebook loads benchmark results produced by
# `particula/gpu/tests/benchmark_test.py` and visualizes GPU vs CPU
# scaling for the two main aerosol-dynamics kernels:
#
# * **Condensation** -- per-particle mass transfer (embarrassingly
#   parallel, O(n) work).
# * **Coagulation** -- stochastic particle-resolved collisions
#   (sequential pair selection per box, but parallelized across boxes).
#
# Run the benchmarks first:
#
# ```bash
# pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s
# ```
#
# The benchmark helper records the source-of-record artifact at
# `.artifacts/benchmarks/gpu_benchmark_results.json`, and this notebook looks
# for that controlled path first. A local `gpu_benchmark_results.json` export is
# still accepted as a fallback, but the artifact path remains the traceable
# record for the original command and runtime metadata.

# %%
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# %% [markdown]
# ## Load benchmark data

# %%
BENCHMARK_FILENAME = "gpu_benchmark_results.json"
BENCHMARK_ARTIFACT_PATH = Path(".artifacts/benchmarks") / BENCHMARK_FILENAME


def _load_benchmark_payload(path: Path) -> dict[str, Any]:
    """Load a benchmark JSON payload from disk."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("benchmark payload must be a JSON object")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise ValueError("benchmark payload must include a dict benchmarks map")
    return payload


def _validate_benchmark_candidate(path: Path) -> None:
    """Validate that a candidate file matches the controlled benchmark schema."""
    payload = _load_benchmark_payload(path)
    metadata = payload.get("benchmark_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("benchmark payload must include benchmark_metadata")
    artifact_path = metadata.get("artifact_path")
    if artifact_path != str(BENCHMARK_ARTIFACT_PATH):
        raise ValueError(
            "benchmark payload artifact_path does not match the controlled path"
        )


def _find_benchmark_results() -> Path:
    """Find benchmark JSON, preferring the controlled artifact path."""
    current = Path.cwd().resolve()
    rejected_candidates: list[str] = []

    for parent in [current, *current.parents]:
        artifact_candidate = parent / BENCHMARK_ARTIFACT_PATH
        if artifact_candidate.exists():
            try:
                _validate_benchmark_candidate(artifact_candidate)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                rejected_candidates.append(
                    f"{artifact_candidate}: {type(exc).__name__}: {exc}"
                )
            else:
                return artifact_candidate

        if (parent / "pyproject.toml").exists():
            break

    for parent in [current, *current.parents]:
        export_candidate = parent / BENCHMARK_FILENAME
        if export_candidate.exists():
            try:
                _validate_benchmark_candidate(export_candidate)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                rejected_candidates.append(
                    f"{export_candidate}: {type(exc).__name__}: {exc}"
                )
            else:
                return export_candidate

        # Stop at the repo root (contains pyproject.toml)
        if (parent / "pyproject.toml").exists():
            break

    msg = (
        "Could not find benchmark results at either "
        f"{BENCHMARK_ARTIFACT_PATH} or {BENCHMARK_FILENAME}. "
        "Run the benchmarks first:\n"
        "  pytest particula/gpu/tests/benchmark_test.py "
        "--benchmark -v -s\n"
        "or copy/export the recorded artifact into the working directory."
    )
    if rejected_candidates:
        msg += "\nRejected candidates:\n- " + "\n- ".join(rejected_candidates)
    raise FileNotFoundError(msg)


results_path = _find_benchmark_results()

with open(results_path) as f:
    data = json.load(f)

benchmarks = data["benchmarks"]
print(f"Loaded {len(benchmarks)} benchmark entries from {results_path}")
print(f"Run started: {data.get('started_at', 'unknown')}")

# %% [markdown]
# ## Condensation scaling

# %%
# Parse condensation entries
cond_entries = {
    k: v for k, v in benchmarks.items() if k.startswith("condensation_")
}

# Build table data
cond_rows = []
for key, entry in sorted(
    cond_entries.items(),
    key=lambda x: x[1]["n_boxes"] * x[1]["n_particles"],
):
    label = key.replace("condensation_", "")
    n_total = entry["n_boxes"] * entry["n_particles"]
    gpu_s = entry["gpu_time_s"]
    steps = entry["steps"]
    gpu_per_step = gpu_s / steps
    cpu_s = entry.get("cpu_time_s")
    speedup = entry.get("speedup")
    cond_rows.append(
        {
            "label": label,
            "n_boxes": entry["n_boxes"],
            "n_particles": entry["n_particles"],
            "total": n_total,
            "gpu_s": gpu_s,
            "gpu_per_step": gpu_per_step,
            "cpu_s": cpu_s,
            "speedup": speedup,
        }
    )

# Print table
print(
    f"{'Config':<12} {'Boxes':>5} {'Particles':>10} "
    f"{'Total':>10} {'GPU (s)':>9} {'CPU (s)':>9} {'Speedup':>8}"
)
print("-" * 75)
for r in cond_rows:
    cpu_str = f"{r['cpu_s']:.4f}" if r["cpu_s"] else "--"
    spd_str = f"{r['speedup']:.1f}x" if r["speedup"] else "--"
    print(
        f"{r['label']:<12} {r['n_boxes']:>5} {r['n_particles']:>10,} "
        f"{r['total']:>10,} {r['gpu_s']:>9.4f} {cpu_str:>9} {spd_str:>8}"
    )

# %% [markdown]
# ### Condensation GPU scaling plot

# %%
cond_plot_rows = [
    r for r in cond_rows if r["total"] > 0 and r["gpu_per_step"] > 0
]
if not cond_plot_rows:
    print("No condensation benchmark entries found; skipping plot.")
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: GPU time vs total particles
    totals = [r["total"] for r in cond_plot_rows]
    gpu_times = [r["gpu_per_step"] * 1000 for r in cond_plot_rows]  # ms
    ax1.scatter(totals, gpu_times, s=50, zorder=5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Total particles (boxes x particles)")
    ax1.set_ylabel("GPU time per step (ms)")
    ax1.set_title("Condensation: GPU scaling")
    ax1.grid(True, alpha=0.3)

    # Right: Speedup (where positive CPU comparison data exists)
    cpu_rows = [
        r
        for r in cond_plot_rows
        if r["speedup"] is not None and r["speedup"] > 0
    ]
    if cpu_rows:
        labels = [r["label"] for r in cpu_rows]
        speedups = [r["speedup"] for r in cpu_rows]
        bars = ax2.barh(labels, speedups, color="steelblue")
        ax2.set_xlabel("GPU speedup vs CPU (x)")
        ax2.set_title("Condensation: GPU vs CPU speedup")
        ax2.set_xscale("log")
        for bar, spd in zip(bars, speedups, strict=True):
            ax2.text(
                bar.get_width() * 1.1,
                bar.get_y() + bar.get_height() / 2,
                f"{spd:.0f}x",
                va="center",
                fontsize=9,
            )
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.axis("off")
        ax2.set_title("Condensation: no CPU speedup data")

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Coagulation scaling

# %%
# Parse coagulation entries
coag_entries = {
    k: v for k, v in benchmarks.items() if k.startswith("coagulation_")
}

coag_rows = []
for key, entry in sorted(
    coag_entries.items(),
    key=lambda x: x[1]["n_boxes"] * x[1]["n_particles"],
):
    label = key.replace("coagulation_", "")
    n_total = entry["n_boxes"] * entry["n_particles"]
    gpu_s = entry["gpu_time_s"]
    steps = entry["steps"]
    gpu_per_step = gpu_s / steps
    cpu_s = entry.get("cpu_time_s")
    speedup = entry.get("speedup")
    coag_rows.append(
        {
            "label": label,
            "n_boxes": entry["n_boxes"],
            "n_particles": entry["n_particles"],
            "total": n_total,
            "gpu_s": gpu_s,
            "gpu_per_step": gpu_per_step,
            "cpu_s": cpu_s,
            "speedup": speedup,
        }
    )

print(
    f"{'Config':<12} {'Boxes':>5} {'Particles':>10} "
    f"{'Total':>10} {'GPU (s)':>9} {'CPU (s)':>9} {'Speedup':>8}"
)
print("-" * 75)
for r in coag_rows:
    cpu_str = f"{r['cpu_s']:.4f}" if r["cpu_s"] else "--"
    spd_str = f"{r['speedup']:.1f}x" if r["speedup"] else "--"
    print(
        f"{r['label']:<12} {r['n_boxes']:>5} {r['n_particles']:>10,} "
        f"{r['total']:>10,} {r['gpu_s']:>9.4f} {cpu_str:>9} {spd_str:>8}"
    )

# %% [markdown]
# ### Coagulation GPU scaling plot

# %%
coag_plot_rows = [
    r for r in coag_rows if r["n_particles"] > 0 and r["gpu_per_step"] > 0
]
if not coag_plot_rows:
    print("No coagulation benchmark entries found; skipping plot.")
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Separate single-box vs multi-box for color coding
    single_box = [r for r in coag_plot_rows if r["n_boxes"] == 1]
    multi_box = [r for r in coag_plot_rows if r["n_boxes"] > 1]

    # Left: GPU time vs particles per box (colored by n_boxes)
    for rows, color, label in [
        (single_box, "tab:blue", "1 box"),
        (multi_box, "tab:orange", "multi-box"),
    ]:
        if rows:
            x = [r["n_particles"] for r in rows]
            y = [r["gpu_per_step"] * 1000 for r in rows]
            ax1.scatter(x, y, s=50, color=color, label=label, zorder=5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Particles per box")
    ax1.set_ylabel("GPU time per step (ms)")
    ax1.set_title("Coagulation: GPU scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Speedup (where CPU data exists)
    cpu_rows = [r for r in coag_plot_rows if r["speedup"] is not None]
    if cpu_rows:
        labels = [r["label"] for r in cpu_rows]
        speedups = [r["speedup"] for r in cpu_rows]
        colors = [
            "tab:blue" if r["n_boxes"] == 1 else "tab:orange"
            for r in cpu_rows
        ]
        bars = ax2.barh(labels, speedups, color=colors)
        ax2.set_xlabel("GPU speedup vs CPU (x)")
        ax2.set_title("Coagulation: GPU vs CPU speedup")
        for bar, spd in zip(bars, speedups, strict=True):
            ax2.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{spd:.1f}x",
                va="center",
                fontsize=9,
            )
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.axis("off")
        ax2.set_title("Coagulation: no CPU speedup data")

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Key observations
#
# ### Condensation
#
# * The GPU condensation kernel is **embarrassingly parallel** -- each
#   particle's mass transfer is independent. This gives near-constant GPU
#   time up to ~100k particles, with linear scaling beyond that.
# * Speedups range from **~13x** at 1k particles to **~1,400x** at 100k
#   particles per box, demonstrating excellent GPU utilization.
# * At 1M particles the GPU still completes a step in under 1 ms.
#
# ### Coagulation
#
# * The GPU coagulation kernel uses a **single thread per box** for the
#   sequential acceptance-rejection pair selection loop. This means
#   single-box performance is limited by the per-particle work within
#   that one thread.
# * **Multi-box configurations** remain the effective region for the current
#   kernel because the GPU parallelizes across independent boxes. The recorded
#   many-box cases (`10x500`, `10x1k`, `50x1k`, `10x5k`, `50x5k`, `100x1k`,
#   `10x10k`) stay near the smaller timings seen at lower box counts on this
#   benchmark machine.
# * Single-box speedups are modest (**1.5--3x**) because the GPU thread does
#   essentially the same sequential work as the CPU, just with faster
#   per-operation throughput.
# * Treat the recorded `1x10k` to `1x50k` range as a **single-box caution
#   band** for the shipped one-thread-per-box path: the measured GPU step time
#   rises from **0.6809 s** to **3.3984 s**, so large single-box workloads are
#   no longer a strong GPU recommendation in this benchmark capture.
# * The `k_max` computation uses an O(n) min/max radius bound (matching
#   the CPU's bin-pair strategy) rather than the original O(n^2)
#   all-pairs scan. This reduced single-box GPU time from **29s to
#   0.28s** at 2k particles.
# * These conclusions are **machine-bounded** to the 2026-07-10 capture on the
#   recorded Warp/device context, not universal guarantees for every GPU or
#   future kernel variant.
#
# ### When to use GPU acceleration
#
# | Scenario | Recommendation |
# |---|---|
# | Condensation, any size | Always use GPU -- massive speedup |
# | Coagulation, many boxes | Use GPU for the recorded multi-box cases on this machine |
# | Coagulation, single box, <5k particles | Either -- modest GPU advantage |
# | Coagulation, single box, 10k--50k particles | Caution band -- not a strong GPU recommendation in this machine-bounded capture |
