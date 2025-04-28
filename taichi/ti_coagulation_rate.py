# %%
"""
Taichi implementation of the coagulation gain rate calculation.


Add max time to run, and min iterations to run.
Adapt for fast a slow running code. Increase calls_per_repeat until ms>10ms
Use min, max and median and std. Run call calculation base on min time.
Use psutil to cpu info on type and system info. save to csv or json file.

"""

import matplotlib.pyplot as plt
import time
import timeit
import platform
import taichi as ti
import numpy as np
import psutil
import math
import statistics
import gc
from typing import Callable, Optional, Any, Dict
import json
from tqdm import tqdm  # type: ignore[import]
import particula as par

ti.init(arch=ti.cpu)  # or ti.cpu


@ti.kernel
def get_coagulation_gain_rate_continuous_taichi(
    radius: ti.types.ndarray(),
    concentration: ti.types.ndarray(),
    kernel: ti.types.ndarray(),
    gain_rate: ti.types.ndarray(),
):
    """
    Calculate the coagulation gain rate via discrete trapezoidal integration.

    gain_rate[i] = ∑_j K[i,j] · c[i] · c[j] · Δr_j

    Args:
        radius: Particle radius array.
        concentration: Particle concentration array.
        kernel: Coagulation kernel matrix.
        gain_rate: Output array to write gain rates into.
    """
    n = radius.shape[0]
    half = ti.cast(0.5, ti.f64)  # 0.5 for trapezoidal rule
    for i in range(n):
        acc = ti.cast(0, ti.f64)  # Accumulator for gain rate
        for j in range(n - 1):
            dr = radius[j + 1] - radius[j]
            # trapezoid: ½·[f(j) + f(j+1)]·Δr
            acc += (
                half
                * (
                    kernel[i, j] * concentration[i] * concentration[j]
                    + kernel[i, j + 1]
                    * concentration[i]
                    * concentration[j + 1]
                )
                * dr
            )
        gain_rate[i] = acc


def collect_system_info():
    """
    Gather CPU and OS information using psutil and platform.

    Returns:
        dict: mapping of info keys to values.
    """
    info = {}

    # CPU counts
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["total_cores"] = psutil.cpu_count(logical=True)

    # CPU frequencies
    freq = psutil.cpu_freq()
    info["max_frequency_mhz"] = freq.max
    info["min_frequency_mhz"] = freq.min
    info["current_frequency_mhz"] = freq.current

    # CPU usage
    info["cpu_usage_per_core_%"] = psutil.cpu_percent(percpu=True, interval=1)
    info["total_cpu_usage_%"] = psutil.cpu_percent()

    # OS / machine info
    uname = platform.uname()
    info["system"] = uname.system
    info["release"] = uname.release
    info["version"] = uname.version
    info["machine"] = uname.machine
    info["processor"] = uname.processor

    # Python runtime
    info["python_version"] = platform.python_version()
    info["python_build"] = platform.python_build()
    info["python_compiler"] = platform.python_compiler()

    return info


def benchmark_timer(
    func: Callable[[], Any],
    ops_per_call: float,
    max_run_time_s: float = 2.0,
    min_iterations: int = 5,
    repeats: Optional[int] = None,
) -> Dict[str, float]:
    """
    Benchmark a zero-arg `func()` using perf_counter_ns, with GC disabled.

    Args:
        func:             A no-arg callable (e.g. `lambda: work(x,y)`).
        ops_per_call:     Estimated FLOPs in each call to func().
        max_run_time_s:   If `repeats`=None, run until this many seconds elapse.
        min_iterations:   If `repeats`=None, do at least this many calls.
        repeats:          If set, run exactly this many calls.

    Returns:
        A dict with:
          - min_time_s
          - max_time_s
          - mean_time_s
          - mode_time_s
          - median_time_s
          - std_time_s
          - throughput_calls_per_s    (1 / min_time_s)
          - cycles_per_call           (min_time_s × CPU_Hz)
          - flops_per_call            (== ops_per_call)
          - flops_per_cycle           (ops_per_call / cycles_per_call)
          - report                    (human-readable summary)
    """
    # disable GC for cleaner timing
    gc.disable()

    timings: list[float] = []
    try:
        if repeats is None:
            start_global = time.perf_counter_ns()
            while (len(timings) < min_iterations) or (
                (time.perf_counter_ns() - start_global) / 1e9 < max_run_time_s
            ):
                t0 = time.perf_counter_ns()
                func()
                t1 = time.perf_counter_ns()
                timings.append((t1 - t0) / 1e9)
        else:
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                func()
                t1 = time.perf_counter_ns()
                timings.append((t1 - t0) / 1e9)
        cpu_hz = psutil.cpu_freq().current * 1e6  # MHz → Hz
    finally:
        gc.enable()

    runs = len(timings)
    # core stats
    min_time = min(timings)
    max_time = max(timings)
    mean_time = statistics.mean(timings)
    try:
        mode_time = statistics.mode(timings)
    except statistics.StatisticsError:
        mode_time = float("nan")
    median_time = statistics.median(timings)
    std_time = statistics.stdev(timings) if runs > 1 else 0.0

    # throughput & cycle/flop estimates on best-case (min_time)
    throughput = 1.0 / min_time
    cycles_per_call = min_time * cpu_hz
    flops_per_call = ops_per_call
    flops_per_cycle = (
        ops_per_call / cycles_per_call if cycles_per_call else float("nan")
    )

    # build report
    labels = [
        ("Throughput      (calls/s)", f"{throughput:,.0f}"),
        ("CPU cycles      (cycles/call)", f"{cycles_per_call:,.0f}"),
        ("Est. Flops      (flops/call)", f"{flops_per_call:,.0f}"),
        ("Efficiency      (flops/cycle)", f"{flops_per_cycle:.4f}"),
        ("Min time        (ms/call)", f"{min_time*1e3:.3f}"),
        ("STDV time       (ms/call)", f"±{std_time*1e3:.3f}"),
    ]
    header = f"Benchmark: {runs} function calls."
    lines = [header] + [f"  {label:<30}{value}" for label, value in labels]
    report = "\n".join(lines)

    array_stats = [
        runs,
        min_time,
        max_time,
        mean_time,
        mode_time,
        median_time,
        std_time,
        throughput,
        cycles_per_call,
        flops_per_call,
        flops_per_cycle,
    ]
    array_headers = [
        "function_calls",
        "min_time_s",
        "max_time_s",
        "mean_time_s",
        "mode_time_s",
        "median_time_s",
        "std_time_s",
        "throughput_calls_per_s",
        "cycles_per_call",
        "flops_per_call",
        "flops_per_cycle",
    ]

    return {
        "min_time_s": min_time,
        "max_time_s": max_time,
        "mean_time_s": mean_time,
        "mode_time_s": mode_time,
        "median_time_s": median_time,
        "std_time_s": std_time,
        "throughput_calls_per_s": throughput,
        "cycles_per_call": cycles_per_call,
        "flops_per_call": flops_per_call,
        "flops_per_cycle": flops_per_cycle,
        "function_calls": runs,
        "report": report,
        "array_stats": array_stats,
        "array_headers": array_headers,
    }


# %%
