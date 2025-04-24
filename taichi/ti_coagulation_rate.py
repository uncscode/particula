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

    ops_per_call = 17 * bins_total * (bins_total - 1)  # 9 Number of operations
    print(f"Number of operations: {ops_per_call}")

    out = np.zeros_like(concentration_lognormal_0, dtype=np.float64)

    # to taichi
    radius_bins = np.asarray(radius_bins, dtype=np.float64)
    concentration_lognormal_0 = np.asarray(
        concentration_lognormal_0, dtype=np.float64
    )
    kernel = np.asarray(kernel, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    get_coagulation_gain_rate_continuous_taichi(
        radius_bins,
        concentration_lognormal_0,
        kernel,
        out)
    # 2) Set up the Timer
    timer = timeit.Timer(
        stmt=lambda: get_coagulation_gain_rate_continuous_taichi(
            radius_bins,
            concentration_lognormal_0,
            kernel,
            out,
        )
    )

    freq = psutil.cpu_freq()
    print(f"Current Frequency: {freq.current:.2f} MHz")
    print(f"Min Frequency:     {freq.min:.2f} MHz")
    print(f"Max Frequency:     {freq.max:.2f} MHz")

    f_hz = freq.current * 1e6   # convert MHz → Hz

    # 3) Repeat the measurement
    repeats = 2000
    calls_per_repeat = 50
    results = timer.repeat(repeat=repeats, number=calls_per_repeat)
    # results is a list of `repeats` total times (in seconds)
    freq = psutil.cpu_freq()
    print(f"Current Frequency: {freq.current:.2f} MHz")
    print(f"Min Frequency:     {freq.min:.2f} MHz")
    print(f"Max Frequency:     {freq.max:.2f} MHz")

    # 4) Convert to per-call times
    per_call_times = np.array(results) / calls_per_repeat

    # 5) Compute average and standard deviation
    avg_time = per_call_times.min()
    std_time = per_call_times.std(ddof=1)  # sample std

    # calculate opperations per cpu cycle
    # 3) compute cycles per call
    cycles_per_call = avg_time * f_hz

    # 5) finally ops per cycle
    ops_per_cycle = ops_per_call / cycles_per_call

    print(f"Taichi version:")
    print(f"Per-call time over {repeats} runs of {calls_per_repeat} calls:")
    print(f"  mean = {avg_time*1e3:.3f} ms")
    print(f"  std  = {std_time*1e3:.3f} ms")
    print(f"Cycles per call: {cycles_per_call:.1f}")
    print(f"Operations per cycle: {ops_per_cycle:.3f}")

    # # Same for standard call
    # timer_python = timeit.Timer(
    #     stmt=lambda: par.dynamics.get_coagulation_gain_rate_continuous(
    #         radius_bins,
    #         concentration_lognormal_0,
    #         kernel,
    #     )
    # )
    # results_python = timer_python.repeat(repeat=repeats, number=calls_per_repeat)
    # per_call_times_python = np.array(results_python) / calls_per_repeat
    # avg_time_python = per_call_times_python.mean()
    # std_time_python = per_call_times_python.std(ddof=1)  # sample std

    # print(f"Python version:")
    # print(f"Per-call time over {repeats} runs of {calls_per_repeat} calls:")
    # print(f"  mean = {avg_time_python*1e3:.3f} ms")
    # print(f"  std  = {std_time_python*1e3:.3f} ms")

    # speed_up = avg_time_python / avg_time
    # print(f"Speedup: {avg_time_python/avg_time:.2f}x")
    # percent_std = (std_time_python / avg_time_python)**2 + (
    #     std_time / avg_time
    # ) ** 2
    # print(f"Relative error: {np.sqrt(percent_std):.2%}")
