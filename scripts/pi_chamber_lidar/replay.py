"""Replay the matched four-condition Π Chamber calculation on CUDA."""

import csv
import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp
from particula.gas import WaterBuckStrategy
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "docs/data/pi-chamber-lidar"
DT_S = 4
BLOCK_S = 600
BLOCKS = 36
# The two layers each contain four exchanging air regions.
LAYERS, PATCHES = 2, 4
PAR_NAMES = (
    "water_mg_m3_s",
    "vertical_exchange_s",
    "haze_deposition_s",
    "lower_water_fraction",
    "patch_exchange_s",
    "epoch_s",
    "intermittent_fraction",
    "phase_seed",
)


def read_rows(name):
    """Read a frozen input or comparison table from the companion download."""
    with (DATA / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def device_array(x):
    """Copy an array with the scalar type expected by the CUDA kernel."""
    return wp.array(
        np.ascontiguousarray(x),
        device="cuda:0",
        dtype=wp.int32 if x.dtype == np.int32 else wp.float64,
    )


def build_kernel(directory, arch):
    """Compile for the visible GPU and check each CUDA launch from Python."""
    nvcc = os.getenv("NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("Install the CUDA toolkit and put nvcc on PATH.")
    libpath = Path(directory) / "chamber.so"
    subprocess.run(  # noqa: S603 -- compile the bundled kernel without a shell
        [
            nvcc,
            "-O3",
            f"-arch=sm_{arch}",
            "-shared",
            "-Xcompiler=-fPIC",
            str(HERE / "chamber.cu"),
            "-o",
            str(libpath),
        ],
        check=True,
    )
    kernel = ct.CDLL(str(libpath)).launch
    kernel.argtypes = (
        [ct.c_int] * 10
        + [ct.c_double, ct.c_double, ct.c_int, ct.c_double]
        + [ct.c_void_p] * 12
    )
    kernel.restype = ct.c_int
    return kernel


def stage(kernel, gpu_tables, initial, params, seed, mass, *, fork, dry):
    """Advance six hours and sample the last hour in the upper layer."""
    count, gas = map(device_array, initial)
    budget = np.zeros((*initial[1].shape, 14))
    budget[..., -2] = 2.0
    ledger = device_array(budget)
    integral = wp.zeros_like(count)
    arrays = [*gpu_tables, params, count, gas, ledger, integral]
    n = count.shape[-1]
    inlet = 0.2 * WaterBuckStrategy().pure_vapor_pressure(303.15) / 2338.3
    window = np.zeros((2, n))
    for b in range(BLOCKS):
        integral.zero_()
        wp.synchronize()
        status = kernel(
            2,
            n,
            gpu_tables[0].shape[0],
            LAYERS,
            PATCHES,
            0,
            1,
            int(dry),
            BLOCK_S // DT_S,
            (b + BLOCKS * int(fork)) * BLOCK_S // DT_S,
            DT_S,
            float(inlet),
            *seed,
            *[a.ptr for a in arrays],
        )
        if status:
            raise RuntimeError(f"Chamber CUDA kernel returned {status}")
        wp.synchronize()
        if b >= BLOCKS - 6:
            window += integral.numpy()[:, PATCHES:].mean(axis=1) / (BLOCK_S * 6)
    final = (count.numpy(), gas.numpy())
    if not all(np.isfinite(a).all() and a.min() >= 0 for a in (*final, window)):
        raise ArithmeticError("Invalid chamber state")
    book = ledger.numpy().mean(axis=1)
    change = final[0] - initial[0]
    cores = change.sum(-1).mean(-1) - book[:, 0] + book[:, 1:4].sum(1)
    water = (change @ mass + final[1] - initial[1]).mean(-1)
    water = water - book[:, 4] + book[:, 5:8].sum(1)
    if max(abs(cores).max(), abs(water).max()) > 2e-5:
        raise ArithmeticError("Open-chamber number or water budget failed")
    return final, window, [float(abs(cores).max()), float(abs(water).max())]


def simulate():
    """Run the selected localized-inlet case and compare the upper counter."""
    wp.init()
    device = wp.get_device("cuda:0")
    wp.set_device(device)
    observed = {
        r["case"]: r for r in read_rows("observations/microphysics.csv")
    }
    case = next(
        r
        for r in read_rows("maintained/matched-example.csv")
        if r["label"] == "Local dry inlet"
    )
    params = np.tile([float(case[k]) for k in PAR_NAMES], (2, 1))
    feed = np.array([25000.0, 200000.0]) * 0.002 / (60 * np.pi)
    parameter_array = device_array(np.column_stack([params, feed]))
    with np.load(DATA / "four-second-kinetics.npz") as tables:
        diam, mass = tables["diam"], tables["mass"]
        speed = get_particle_settling_velocity_via_system_state(
            diam * 0.5e-6, 998.2, 293.15, 101325.0
        )
        arrays = [
            tables[k] for k in ("rh", "endpoint", "dest", "weight", "mass")
        ]
        gpu_tables = [device_array(a) for a in [*arrays, speed, diam]]
        seed = (int(tables["seed_bin"]), float(tables["seed_weight"]))
        start = (
            np.zeros((2, LAYERS * PATCHES, len(diam))),
            np.full((2, LAYERS * PATCHES), float(tables["saturation"])),
        )
    with TemporaryDirectory() as directory:
        kernel = build_kernel(directory, device.arch)
        spun, _, budget = stage(
            kernel,
            gpu_tables,
            start,
            parameter_array,
            seed,
            mass,
            fork=False,
            dry=False,
        )
        rows = []
        checks = [budget]
        visible = (diam >= 0.583) & (diam <= 40.679)
        cloud = visible & (diam >= 2)
        for flag, label in [(False, "B"), (True, "E")]:
            _, spectrum, residual = stage(
                kernel,
                gpu_tables,
                spun,
                parameter_array,
                seed,
                mass,
                fork=True,
                dry=flag,
            )
            checks.append(residual)
            for i, loading in enumerate(("clean", "polluted")):
                key = f"{loading}-{label}"
                predicted = float(spectrum[i, cloud].sum())
                water = float(spectrum[i, visible] @ mass[visible])
                np.testing.assert_allclose(
                    [predicted, water],
                    [float(case[key + "_cloud"]), float(case[key + "_liquid"])],
                    rtol=2e-5,
                    atol=1e-4,
                )
                rows.append(
                    dict(
                        case=key,
                        model_cloud=predicted,
                        measured_cloud=float(observed[key]["cloud_number_cm3"]),
                        model_water=water,
                        measured_water=float(
                            observed[key]["liquid_water_mg_m3"]
                        ),
                    )
                )
    return dict(
        states=rows,
        device=device.name,
        max_number_error_cm3=max(c[0] for c in checks),
        max_water_error_mg_m3=max(c[1] for c in checks),
    )
