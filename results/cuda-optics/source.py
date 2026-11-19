# %% [markdown]
# # Qualify CUDA optics before interpreting a cloud spectrum
#
# When does a batch of optical calculations benefit from CUDA, and can the device
# calculation preserve the CPU optical quantities? This example pairs a timing
# study with explicit numerical comparisons, then computes extinction and angular
# scattering for an illustrative droplet population at four wavelengths.
#
# Requires native CUDA, Warp, and Particula's development optics API. CPU runs
# are explicit comparison oracles; there is no CPU fallback. Start with the
# [sphere-to-cloud walkthrough](sphere_optics.py) for the physical quantities and
# the [optics guide](../../guide/optics/index.md) for units and reference frames.
#
# The physics assumes independent homogeneous spheres with a fixed illustrative
# refractive index. Population Mueller matrices share one scattering-plane frame.
# This is an optical calculation, not a particle-growth or multiple-scattering
# radiative-transfer simulation.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
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

# %% [markdown]
# ## Separate numerical agreement from timing
#
# The benchmark increases both batch size (64, 512, 2048) and maximum size
# parameter x (10, 100, 1000). Because both change, this is not a pure batch-size
# scaling experiment. Index 1.5 + 0.1i is held fixed. Upload and workspace
# preparation happen explicitly; warm runs reuse the output storage.
#
# All efficiency outputs are compared with NumPy at rtol=10⁻⁸ and atol=10⁻¹².
# The pointer assertion checks storage reuse. Synchronization closes each timed
# CUDA run, while five device repetitions and three NumPy repetitions supply
# execution medians. Module loading, transfers, and preparation are recorded
# separately and excluded from the warm bars.
#
# The cloud calculation uses 512 radius bins from 0.5–20 µm, four wavelengths
# from 450–900 nm, and 61 scattering angles. Weights sum to 10⁸ particles/m³
# at each wavelength. Tiled processing retains bulk optics and differential
# Mueller matrices, rather than a permanent full population of angular outputs.
# Scalar population outputs use the same relative/absolute tolerance as particle
# efficiencies. Matrix errors are normalized by each wavelength's largest
# reference element, then bounded absolutely by 10⁻⁸; that is not a relative
# error guarantee for every small matrix element.


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


@mplstyle.context("particula.default")
def plot_results(report):
    """Plot compact saved timings and cloud optics without CUDA execution."""
    cases = report["benchmarks"]
    labels = [f"{v['cases']} cases\nx ≤ {v['maximum_x']:g}" for v in cases]
    position = np.arange(len(cases))
    fig, ax = plotting.subplots(size="wide")
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
    plotting.legend(ax)
    cloud = report["cloud"]
    plotting.fit(fig)
    fig, axes = plotting.subplots(1, 2)
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
    plotting.legend(axes[1])
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
    plotting.fit(fig)
    return report


def run_example():
    """Execute the complete study and render its observations."""
    return plot_results(simulate())


# %% [markdown]
# ## Read the saved timing and optical evidence
#
# The timing bars compare warm synchronized CUDA execution with the NumPy CPU
# implementation on the recorded machine. They do not include full one-shot cost
# or establish universal speedups. Use the separately printed upload, preparation,
# and download times when judging a workflow that cannot reuse device inputs.
#
# In the cloud panels, extinction is a bulk coefficient in m⁻¹, while S11 is
# angle-dependent scattering in m⁻¹/sr. The strong forward peak shows why a
# single extinction coefficient cannot describe the directional signal. The
# 61-angle grid samples that structure; finer angular convergence would be
# needed to resolve a narrow peak for a particular instrument. Fixed refractive
# index makes the wavelength comparison a size-parameter effect in this model.
#
# The checks qualify CPU/CUDA agreement for these cases. They do not test the
# homogeneous-sphere assumption against measured cloud scattering.


# %%
if __name__ == "__main__":
    result = run_example()
    plt.show()


# %% [markdown]
# ## Choose the next comparison deliberately
#
# CUDA can reuse prepared storage while reproducing the tested CPU optical
# quantities. Whether that helps a research workflow depends on batch size,
# size parameter, repeated use, transfers, and the requested angular outputs.
#
# For a controlled performance study, fix the size-parameter range and vary only
# batch size, reporting both warm and full-pipeline times. For a scientific study,
# vary the droplet size distribution at fixed number density and compare spectral
# extinction and angle-resolved S11 with observations. Use measured refractive
# indices and convergence checks; device agreement alone cannot validate those
# physical inputs or an instrument's viewing geometry.
