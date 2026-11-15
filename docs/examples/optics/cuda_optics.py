# %% [markdown]
# # CUDA optics: particle batches and a cloud spectrum
#
# Requires native CUDA, Warp and Particula's development optics API. No CPU
# fallback is used. CPU calculations below are explicit comparison oracles.
# Timings separate module compilation/cache loading, upload, preparation,
# synchronized execution and download. They describe this machine and workload,
# not universal speedups. CPU timings use the NumPy implementation, not Warp CPU.
#
# The cloud's fixed refractive index is illustrative. Its angular matrices
# assume independent particles and one common scattering-plane reference frame.
# See the [optics guide](../../guide/optics/index.md) for scope and units.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from particula.optics import (
    get_cross_sections,
    get_differential_mueller_matrix,
    get_mie_coefficients,
    get_mie_efficiencies,
    get_mueller_matrix,
    get_population_mueller_matrix,
    get_population_optics,
    get_scattering_amplitudes,
)
from particula.optics.gpu import prepare_mie, prepare_population_mie


# %%
def benchmark_case(device, count, maximum_x):
    """Compare every efficiency and separately time explicit pipeline stages."""
    x = np.geomspace(0.1, maximum_x, count)
    index = np.tile([1.5, 0.1], (count, 1))
    wp.synchronize_device(device)
    start = time.perf_counter()
    m_device = wp.array(index, dtype=wp.vec2d, device=device)
    x_device = wp.array(x, dtype=wp.float64, device=device)
    wp.synchronize_device(device)
    upload = time.perf_counter() - start
    start = time.perf_counter()
    work = prepare_mie(
        m_device,
        x_device,
        case_tile=256,
        coefficient_capacity=int(
            np.ceil(maximum_x + 4.05 * np.cbrt(maximum_x) + 2)
        ),
    )
    wp.synchronize_device(device)
    preparation = time.perf_counter() - start
    work.enqueue()
    work.check()  # Untimed warm-up and explicit status observation.
    pointer = work.efficiencies.ptr
    gpu_seconds = []
    for _ in range(5):
        start = time.perf_counter()
        work.enqueue()
        wp.synchronize_device(device)
        gpu_seconds.append(time.perf_counter() - start)
    work.check()
    assert work.efficiencies.ptr == pointer
    start = time.perf_counter()
    actual = work.efficiencies.numpy()
    download = time.perf_counter() - start
    cpu_seconds = []
    for _ in range(3):
        start = time.perf_counter()
        expected = get_mie_efficiencies(x / (2 * np.pi), 1.0, 1.5 + 0.1j)
        cpu_seconds.append(time.perf_counter() - start)
    expected_array = np.stack(expected, axis=-1)
    np.testing.assert_allclose(actual, expected_array, rtol=1e-8, atol=1e-12)
    return {
        "cases": count,
        "maximum_x": maximum_x,
        "upload_seconds": upload,
        "preparation_seconds": preparation,
        "download_seconds": download,
        "gpu_seconds": gpu_seconds,
        "cpu_seconds": cpu_seconds,
        "maximum_absolute_efficiency_error": float(
            np.max(abs(actual - expected_array))
        ),
    }


def cloud_spectrum(device):
    """Stream four cloud spectra and verify all scalar and Mueller outputs."""
    radius = np.geomspace(0.5e-6, 20e-6, 512)
    wavelength = np.array([450e-9, 550e-9, 700e-9, 900e-9])
    angles = np.linspace(0, np.pi, 61)
    shape = (len(wavelength), len(radius))
    radii = np.broadcast_to(radius, shape).copy()
    x = 2 * np.pi * radii / wavelength[:, None]
    distribution = np.exp(-0.5 * (np.log(radius / 6e-6) / 0.4) ** 2)
    weights = np.broadcast_to(
        1e8 * distribution / distribution.sum(), shape
    ).copy()
    indices = np.empty(shape + (2,))
    indices[..., 0], indices[..., 1] = 1.33, 1e-8
    work = prepare_population_mie(
        wp.array(indices, dtype=wp.vec2d, device=device),
        wp.array(x, dtype=wp.float64, device=device),
        wp.array(radii, dtype=wp.float64, device=device),
        wp.array(weights, dtype=wp.float64, device=device),
        angles=wp.array(angles, dtype=wp.float64, device=device),
        particle_tile=128,
        angle_tile=32,
        coefficient_capacity=320,
    )
    work.enqueue()
    work.check()
    bulk = work.optics.numpy()
    matrix = work.mueller.numpy()
    q = get_mie_efficiencies(radii, wavelength[:, None], 1.33 + 1e-8j)
    expected = get_population_optics(
        get_cross_sections(q, radii), q.asymmetry, weights
    )
    np.testing.assert_allclose(
        bulk, np.stack(expected, -1), rtol=1e-8, atol=1e-12
    )
    amplitudes = get_scattering_amplitudes(
        get_mie_coefficients(1.33 + 1e-8j, x), angles
    )
    differential = get_differential_mueller_matrix(
        get_mueller_matrix(*amplitudes), wavelength[:, None]
    )
    expected_matrix = get_population_mueller_matrix(differential, weights)
    scale = np.max(abs(expected_matrix), axis=(1, 2, 3))[:, None, None, None]
    np.testing.assert_allclose(
        matrix / scale, expected_matrix / scale, rtol=0, atol=1e-8
    )
    return {
        "wavelength_nm": (wavelength * 1e9).tolist(),
        "extinction_m_inv": bulk[:, 0].tolist(),
        "absorption_m_inv": bulk[:, 2].tolist(),
        "asymmetry": bulk[:, 6].tolist(),
        "angle_degrees": np.rad2deg(angles).tolist(),
        "differential_scattering_m_inv_sr": matrix[:, :, 0, 0].tolist(),
        "maximum_scaled_mueller_error": float(
            np.max(abs(matrix - expected_matrix) / scale)
        ),
    }


def simulate():
    """Execute the full native-CUDA study with explicit CPU qualification."""
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires native CUDA")
    device = wp.get_device("cuda:0")
    module_times = {}
    for name in ("mie", "angular", "_bulk"):
        start = time.perf_counter()
        wp.load_module(module=f"particula.optics.gpu.{name}", device=device)
        wp.synchronize_device(device)
        module_times[name] = time.perf_counter() - start
    return {
        "device": device.name,
        "warp_version": wp.__version__,
        "module_compile_or_cache_load_seconds": module_times,
        "benchmarks": [
            benchmark_case(device, n, x)
            for n, x in [(64, 10.0), (512, 100.0), (2048, 1000.0)]
        ],
        "cloud": cloud_spectrum(device),
    }


def plot_results(report):
    """Plot compact saved timings and cloud optics without CUDA execution."""
    cases = report["benchmarks"]
    labels = [f"{v['cases']} cases\nx ≤ {v['maximum_x']:g}" for v in cases]
    position = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for offset, key, label in [
        (-0.18, "cpu_seconds", "NumPy CPU"),
        (0.18, "gpu_seconds", "CUDA, warm"),
    ]:
        median = [np.median(v[key]) for v in cases]
        ax.bar(position + offset, median, width=0.36, label=label)
    ax.set(
        xticks=position,
        xticklabels=labels,
        yscale="log",
        ylabel="Median execution time [s]",
        title=report["device"],
    )
    ax.legend()
    cloud = report["cloud"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(cloud["wavelength_nm"], cloud["extinction_m_inv"], marker="o")
    axes[0].set(
        xlabel="Vacuum wavelength [nm]",
        ylabel="Extinction [m⁻¹]",
        title="Illustrative cloud · 100 particles/cm³",
    )
    for wave, values in zip(
        cloud["wavelength_nm"],
        cloud["differential_scattering_m_inv_sr"],
        strict=True,
    ):
        axes[1].semilogy(cloud["angle_degrees"], values, label=f"{wave:g} nm")
    axes[1].set(xlabel="Scattering angle [degrees]", ylabel="Bulk S11 [m⁻¹/sr]")
    axes[1].legend()
    print("Device:", report["device"], "Warp", report["warp_version"])
    print(
        "Module compile/cache load [s]:",
        report["module_compile_or_cache_load_seconds"],
    )
    for row in cases:
        print(
            f"{row['cases']} cases, x≤{row['maximum_x']:g}: "
            f"upload={row['upload_seconds']:.6g}s, "
            f"prepare={row['preparation_seconds']:.6g}s, "
            f"download={row['download_seconds']:.6g}s, "
            f"max Q error={row['maximum_absolute_efficiency_error']:.3g}"
        )
    print(
        "All particle efficiencies and full bulk Mueller matrices pass "
        "the declared CPU/CUDA comparisons."
    )
    return report


def run_example():
    """Execute the complete study and render its observations."""
    return plot_results(simulate())


# %%
if __name__ == "__main__":
    result = run_example()
    plt.show()
