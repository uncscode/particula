"""
Taichi implementation of the coagulation gain rate calculation.
"""
import timeit
import taichi as ti
import numpy as np
import psutil
import particula as par

ti.init(arch=ti.cpu)  # or ti.cpu


@ti.kernel
def get_coagulation_gain_rate_continuous_taichi(
    radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kernel: ti.types.ndarray(dtype=ti.f64, ndim=2),
    gain_rate: ti.types.ndarray(dtype=ti.f64, ndim=1),
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


def benchmark_timer(
    timer: timeit.Timer,
    ops_per_call: float,
    repeats: int = 2000,
    calls_per_repeat: int = 50,
) -> dict:
    """
    Run a Timer benchmark and return performance metrics plus a formatted report.
    Key names now include unit suffixes.

    Args:
        timer: A timeit.Timer instance to measure.
        ops_per_call: Estimated floating-point operations per call.
        repeats: How many times to repeat the measurement loop.
        calls_per_repeat: Number of calls in each timing.

    Returns:
        A dict containing:
          throughput_calls_per_s: Calls per second (float).
          cycles_per_call_cycles_per_call: Cycles per call (float).
          flops_per_call_flops_per_call: Flops per call (float).
          flops_per_cycle_flops_per_cycle: Flops per cycle (float).
          median_time_s: Median time per call in seconds (float).
          std_time_s: Sample standard deviation of time per call in seconds (float).
          report: A multi-line string summarizing the results.
    """
    # 1) Collect raw timing results (total seconds per batch)
    results = timer.repeat(repeat=repeats, number=calls_per_repeat)

    # 2) Compute per-call times
    per_call = np.array(results) / calls_per_repeat
    t_med = np.median(per_call)
    t_min = np.min(per_call)
    t_max = np.max(per_call)
    t_std = per_call.std(ddof=1)

    # 3) Sample current CPU frequency (Hz)
    f_hz = psutil.cpu_freq().current * 1e6

    # 4) Compute derived metrics
    cycles_per_call = t_med * f_hz
    throughput = 1.0 / t_med
    flops_per_call = ops_per_call
    flops_per_cycle = ops_per_call / cycles_per_call

    # 5) Build formatted report (no direct prints)
    # convert to ms for reporting
    t_med_ms = t_med * 1e3
    t_min_ms = t_min * 1e3
    t_std_ms = t_std * 1e3

    labels = [
        ("Throughput      (calls/s)", f"{throughput:,.0f}"),
        ("Cycles/call     (cycles)", f"{cycles_per_call:,.0f}"),
        ("Est. Math Flops  (flops/call)", f"{flops_per_call:,.0f}"),
        ("Efficiency (flops/cycle)", f"{flops_per_cycle:.3f}"),
        ("Median time     (ms/call)", f"{t_med_ms:.3f} [±{t_std_ms:.3f}]"),
        ("Min time        (ms/call)", f"{t_min_ms:.3f}"),
    ]
    header = f"Benchmark: {repeats} runs × {calls_per_repeat} calls each"
    lines = [header] + [f"  {label:<30}{value}" for label, value in labels]
    report = "\n".join(lines)

    return {
        "throughput_calls_per_s": throughput,
        "cycles_per_call": cycles_per_call,
        "flops_per_call": flops_per_call,
        "flops_per_cycle": flops_per_cycle,
        "median_time_s": t_med,
        "min_time_s": t_min,
        "max_time_s": t_max,
        "std_time_s": t_std,
        "report": report,
    }


if __name__ == "__main__":

    # --- example usage ---
    bins_total = 200  # Number of bins for the particle size distribution
    # Create fine scale radius bins on a logarithmic scale from 1 nm to 10 μm
    radius_bins = np.logspace(start=-9, stop=-4, num=bins_total)  # m (1 nm to 10 μm)

    # Calculate the mass for each particle size bin assuming a density of 1 g/cm^3 (1000 kg/m^3)
    mass_bins = 4 / 3 * np.pi * (radius_bins) ** 3 * 1e3  # kg

    # Generate a lognormal particle size distribution
    # This distribution is characterized by a mode (most probable size) of 100 nm,
    # a geometric standard deviation of 1.4, and a total number concentration of 10000 particles per cm^3.
    concentration_lognormal_0 = par.particles.get_lognormal_pmf_distribution(
        x_values=radius_bins,
        mode=np.array(100e-9),  # Mode of the distribution (100 nm)
        geometric_standard_deviation=np.array(1.4),  # Geometric standard deviation
        number_of_particles=np.array(
            1e6 * 1e6  # Total concentration (10000 cm^-3 converted to m^-3)
        ),
    )

    # Calculate the coagulation kernel
    kernel = par.dynamics.get_brownian_kernel_via_system_state(
        particle_radius=radius_bins,
        particle_mass=mass_bins,
        temperature=293.15,  # Temperature in Kelvin
        pressure=101325,  # Pressure in Pascals (1 atm)
        alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
    )

    out = np.zeros_like(concentration_lognormal_0, dtype=np.float64)

    # to taichi
    radius_bins = np.asarray(radius_bins, dtype=np.float64)
    concentration_lognormal_0 = np.asarray(
        concentration_lognormal_0, dtype=np.float64
    )
    kernel = np.asarray(kernel, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    # 2) Set up the Timer
    timer = timeit.Timer(
        stmt=lambda: get_coagulation_gain_rate_continuous_taichi(
            radius_bins,
            concentration_lognormal_0,
            kernel,
            out,
        )
    )

    ops_per_call = 9 * bins_total * (bins_total - 1)
    # benchmark the Taichi kernel
    taichi_results = benchmark_timer(
        timer=timer,
        ops_per_call=ops_per_call,
        repeats=2000,
        calls_per_repeat=50,
    )
    print("Taichi kernel")
    print(taichi_results["report"])

    # Same for standard call
    timer_python = timeit.Timer(
        stmt=lambda: par.dynamics.get_coagulation_gain_rate_continuous(
            radius_bins,
            concentration_lognormal_0,
            kernel,
        )
    )
    # benchmark the standard call
    python_results = benchmark_timer(
        timer=timer_python,
        ops_per_call=ops_per_call,
        repeats=100,
        calls_per_repeat=25,
    )
    print("Python kernel")
    print(python_results["report"])
