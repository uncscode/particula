"""Particle-resolved activation at prescribed vapor forcing on NVIDIA CUDA."""

# The assertions qualify numerical results before publication.
# ruff: noqa: S101
import csv
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par
import warp as wp
from dynamic import (
    AIR_DENSITY,
    CP,
    DV,
    ES,
    KA,
    LV,
    MA,
    MW,
    OUT,
    RHO,
    A,
    R,
    T,
    arrival,
    bisect_vector,
    critical,
    nodes,
    revision,
    save_csv,
    seq,
    shortcut,
)
from matplotlib import style as mplstyle
from particula import plotting
from particula.gpu.properties.particle_properties import kelvin_term_wp

COMPONENTS = np.array([0.03, 0.12, 0.61, 0.0])
MODES = (
    (0.45, 35.0, 1.45, (0.55, 0.10, 0.05, 0.30)),
    (0.40, 95.0, 1.50, (0.18, 0.45, 0.29, 0.08)),
    (0.15, 170.0, 1.40, (0.06, 0.32, 0.59, 0.03)),
)
SEEDS = (34119, 34120, 34121, 34122)
SS = np.unique(np.r_[np.geomspace(0.05, 1.4, 49), 0.2, 0.4, 0.65, 1.0])
TIMES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
VARIANTS = ("heterogeneous", "size_homogenized", "shortcut")
# The resistance in dynamic.py is affine in 1/wet diameter at fixed T and
# alpha=1.
MASS = RHO * R * T / (ES * DV * MW)
HEAT = LV * RHO / (KA * T) * (LV * MW / (R * T) - 1)
R0 = wp.constant(wp.float64(MASS + HEAT))
R1 = wp.constant(
    wp.float64(
        MASS * 2 * DV * np.sqrt(2 * np.pi * MW / (R * T))
        + HEAT * 2 * KA * np.sqrt(2 * np.pi * MA / (R * T)) / (AIR_DENSITY * CP)
    )
)
KELVIN = wp.constant(wp.float64(A / 2))
ONE = wp.constant(wp.float64(1.0))
HALF = wp.constant(wp.float64(0.5))
NEVER = wp.constant(wp.float64(1e30))


@wp.func
def saturation(wet: wp.float64, dry: wp.float64, k: wp.float64):
    """Evaluate the κ-Köhler saturation ratio for one particle."""
    x = wet / dry
    water = x * x * x - ONE
    return water / (water + k) * kelvin_term_wp(wet * HALF, KELVIN)


@wp.func
def interval(
    left: wp.float64,
    right: wp.float64,
    d: wp.float64,
    k: wp.float64,
    s: wp.float64,
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
):
    """Integrate growth time across one wet-diameter interval."""
    total = wp.float64(0.0)
    if right > left:
        for i in range(z.shape[0]):
            wet = left + (right - left) * z[i]
            total += w[i] * (wet * R0 + R1) / (s - saturation(wet, d, k))
    return wp.float64(0.25) * (right - left) * total


@wp.kernel
def optical_times(
    d: wp.array(dtype=wp.float64),
    k: wp.array(dtype=wp.float64),
    initial: wp.array(dtype=wp.float64),
    peak: wp.array(dtype=wp.float64),
    forcing: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
):
    """Compute time to detection for each particle and forcing."""
    j, i = wp.tid()
    start = initial[i]
    target = wp.float64(1e-6)
    barrier = wp.min(peak[i], target)
    result = NEVER
    if start >= target:
        result = wp.float64(0.0)
    elif forcing[j] > saturation(barrier, d[i], k[i]):
        result = interval(
            start, barrier, d[i], k[i], forcing[j], z, w
        ) + interval(barrier, target, d[i], k[i], forcing[j], z, w)
    out[j, i] = result


def particles(seed, n):
    """Draw dry sizes and component volumes using one realization seed."""
    rng = np.random.default_rng(seed)
    mode = rng.choice(3, n, p=[v[0] for v in MODES])
    dry = np.empty(n)
    fraction = np.empty((n, 4))
    for m, (_, median, gsd, mean) in enumerate(MODES):
        pick = mode == m
        dry[pick] = (
            median * 1e-9 * np.exp(rng.normal(size=pick.sum()) * np.log(gsd))
        )
        fraction[pick] = rng.dirichlet(np.array(mean) * 8, size=pick.sum())
    # Within each 0.05-wide ln(D) group, preserve individual sizes and each
    # constituent's total volume. Number and dry material are conserved.
    _, group = np.unique(
        np.floor(np.log(dry / 1e-9) / 0.05), return_inverse=True
    )
    volume = dry**3
    totals = np.bincount(group, weights=volume)
    uniform = np.column_stack(
        [np.bincount(group, weights=volume * f) / totals for f in fraction.T]
    )[group]
    assert np.allclose(volume @ uniform, volume @ fraction, rtol=1e-12, atol=0)
    assert np.allclose(fraction.sum(axis=1), 1.0) and np.allclose(
        uniform.sum(axis=1), 1.0
    )
    return dry, mode, fraction, uniform


def prepare(d, k):
    """Compute humidified and critical diameters for the sampled particles."""
    peak, sc = critical(d, k)
    initial = bisect_vector(
        lambda wet, d=d, k=k: seq(wet, d, k) - 0.95, d, peak
    )
    return peak, sc, initial


def cuda_times(d, k, peak, initial, ss, order, device):
    """Evaluate optical arrival times on the requested CUDA device."""

    def on_device(x):
        return wp.array(
            np.ascontiguousarray(x), dtype=wp.float64, device=device
        )

    args = [
        on_device(x) for x in (d, k, initial, peak, 1 + ss / 100, *nodes(order))
    ]
    out = wp.empty((len(ss), len(d)), dtype=wp.float64, device=device)
    wp.synchronize_device(device)
    start = time.perf_counter()
    wp.launch(optical_times, dim=out.shape, inputs=[*args, out], device=device)
    wp.synchronize_device(device)
    seconds = time.perf_counter() - start
    result = out.numpy()
    assert np.all(np.isfinite(result)) and np.all(result >= 0)
    return result, seconds


def solve():  # noqa: C901 -- scan and verification sequence
    """Run paired ensembles and save counts and convergence checks."""
    start = time.perf_counter()
    assert Path(par.__file__).resolve().is_relative_to(Path.cwd() / "particula")
    device = wp.get_device("cuda:0")
    assert device.is_cuda
    rows = []
    max_parity = 0.0
    max_refine = 0.0
    max_count = 0.0
    max_seed_se = 0.0
    kernel_time = 0.0
    warm = particles(SEEDS[0], 64)
    dry = warm[0]
    k = warm[2] @ COMPONENTS
    peak, _, initial = prepare(dry, k)
    cuda_times(dry, k, peak, initial, np.array([0.4]), 32, device)
    for seed in SEEDS:
        dry, mode, f, homogeneous = particles(seed, 32768)
        k_input = f @ COMPONENTS
        _, sc, _ = prepare(dry, k_input)
        np.savez_compressed(
            OUT / f"ambient-particles-{seed}.npz",
            dry_m=dry,
            mode=mode,
            dry_volume_fractions=f,
            seed=seed,
        )
        for variant, k in zip(
            VARIANTS,
            (k_input, homogeneous @ COMPONENTS, shortcut(dry, sc)),
            strict=True,
        ):
            peak, _, initial = prepare(dry, k)
            result, dt = cuda_times(dry, k, peak, initial, SS, 64, device)
            kernel_time += dt
            coarse, _ = cuda_times(dry, k, peak, initial, SS, 32, device)
            for t in TIMES:
                fractions = (result <= t).mean(axis=1)
                if np.any(np.diff(fractions) < 0):
                    raise ValueError("Nonmonotone counts with forcing")
                max_refine = max(
                    max_refine,
                    float(
                        100
                        * np.max(abs(fractions - (coarse <= t).mean(axis=1)))
                    ),
                )
                for ss, fraction in zip(SS, fractions, strict=True):
                    rows.append(
                        dict(
                            seed=seed,
                            model=variant,
                            seconds=t,
                            ss_pct=ss,
                            optical_pct=100 * fraction,
                        )
                    )
            subset = np.arange(0, len(k), 127)
            for ss in (0.2, 0.4, 0.65, 1.0):
                cpu = arrival(
                    1e-6,
                    initial[subset],
                    peak[subset],
                    dry[subset],
                    k[subset],
                    1 + ss / 100,
                    1.0,
                    96,
                )
                gpu = result[np.where(SS == ss)[0][0], subset]
                assert np.array_equal(np.isfinite(cpu), gpu < 1e29)
                check = (cpu > 0) & (cpu <= 20.0)
                if np.any(check):
                    max_parity = max(
                        max_parity,
                        float(100 * np.max(abs(gpu[check] / cpu[check] - 1))),
                    )
                for t in TIMES:
                    max_count = max(
                        max_count,
                        100 * abs(np.mean(cpu <= t) - np.mean(gpu <= t)),
                    )
    assert max(max_refine, max_count) < 0.5 and max_parity < 0.1
    save_csv("ambient-cuda-scans.csv", rows)
    summary = []
    for variant in VARIANTS:
        for t in TIMES:
            group = [
                r for r in rows if r["model"] == variant and r["seconds"] == t
            ]
            values = np.array([r["optical_pct"] for r in group]).reshape(
                len(SEEDS), len(SS)
            )
            for ss, mean, se in zip(
                SS,
                values.mean(axis=0),
                values.std(axis=0, ddof=1) / 2,
                strict=True,
            ):
                max_seed_se = max(max_seed_se, se)
                summary.append(
                    dict(
                        model=variant,
                        seconds=t,
                        ss_pct=ss,
                        mean_optical_pct=mean,
                        seed_se_pp=se,
                    )
                )
    save_csv("ambient-cuda-summary.csv", summary)
    metadata = dict(
        device=device.name,
        alias=str(device),
        warp=wp.__version__,
        particula=par.__version__,
        revision=revision(),
        gpu_kernel_seconds=kernel_time,
        wall_seconds=time.perf_counter() - start,
        max_relative_event_time_error_pct=max_parity,
        max_cpu_gpu_count_difference_pp=max_count,
        max_quadrature_refinement_pp=max_refine,
        max_across_seed_standard_error_pp=max_seed_se,
        realizations=len(SEEDS),
        particles_per_realization=32768,
        plateaus=len(SS),
        exposures_s=TIMES,
    )
    (OUT / "ambient-cuda-checks.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(metadata)


@mplstyle.context("particula.default")
def plot():
    """Plot the particle-resolved counts and mixing-state differences."""
    with (OUT / "ambient-cuda-summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    fig, axes = plotting.subplots(1, 2)
    for variant, color, label in zip(
        VARIANTS,
        ["#00757c", "#c16a25", "#563d7c"],
        [
            "Individual compositions",
            "Same composition within size band",
            "Approximate κ reused",
        ],
        strict=True,
    ):
        subset = [
            r
            for r in rows
            if r["model"] == variant and float(r["seconds"]) == 5.0
        ]
        y = np.array([float(r["mean_optical_pct"]) for r in subset])
        se = np.array([float(r["seed_se_pp"]) for r in subset])
        axes[0].plot(SS, y, color=color, label=label)
        axes[0].fill_between(SS, y - se, y + se, color=color, alpha=0.25)
        if variant == "heterogeneous":
            reference = y
        else:
            axes[1].plot(SS, y - reference, color=color, label=label)
    for ax in axes:
        ax.set(xscale="log", xlabel="Imposed supersaturation (%)")
        ax.set_xticks(
            [0.05, 0.1, 0.2, 0.4, 0.7, 1.4],
            labels=["0.05", "0.1", "0.2", "0.4", "0.7", "1.4"],
        )
        ax.grid(alpha=0.2)
        plotting.legend(ax)
    axes[0].set_ylabel("Particles reaching 1 µm (%)")
    axes[1].set_ylabel("Difference from individual compositions (pp)")
    axes[1].axhline(0, color="gray", lw=0.8)
    fig.suptitle("Particle-resolved ambient aerosol   ·   5 s exposure")
    plotting.fit(fig)
    fig.savefig(OUT / "ambient-cuda-reference.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = ArgumentParser()
    p.add_argument("--replot", action="store_true")
    if not p.parse_args().replot:
        solve()
    plot()
