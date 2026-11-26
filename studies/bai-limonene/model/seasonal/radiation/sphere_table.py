"""Water-sphere extinction and Legendre-weighted scattering from Particula/CUDA."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import warp as wp
from particula.optics import get_mie_efficiencies
from particula.optics.gpu.workspace import prepare_mie
from scipy.special import roots_legendre

HERE = Path(__file__).resolve().parent
TMP = Path(os.environ["SEASONAL_TMP"]) / "radiation"


def solar_bands(bands):
    """Equal-energy bands on a direct/diffuse mixture; use each spectrum's own weights."""
    source = np.genfromtxt(
        HERE / "astmg173.csv", names=True, skip_header=1, delimiter=","
    )
    wavelength = source["wavelength"]
    direct = source["direct"]
    diffuse = source["global"] - direct * np.cos(np.deg2rad(48.19 - 37))
    assert np.min(diffuse) >= 0
    # Convert the trapezoidal integral to positive weights on native wavelengths.
    dx = np.gradient(wavelength)
    dx[[0, -1]] *= 0.5
    direct = direct * dx
    diffuse = diffuse * dx
    primary = (direct / direct.sum() + diffuse / diffuse.sum()) / 2
    cdf = np.r_[0, np.cumsum(primary)]
    edges = np.linspace(0, 1, bands + 1)
    wave, direct_w, diffuse_w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mass = np.maximum(np.minimum(hi, cdf[1:]) - np.maximum(lo, cdf[:-1]), 0)
        frac = np.divide(
            mass, primary, out=np.zeros_like(primary), where=primary > 0
        )
        wave.append(np.dot(mass, wavelength) / mass.sum())
        direct_w.append(np.dot(direct, frac) / direct.sum())
        diffuse_w.append(np.dot(diffuse, frac) / diffuse.sum())
    assert np.allclose([sum(direct_w), sum(diffuse_w)], 1)
    return wave, direct_w, diffuse_w


@wp.kernel
def moments(
    mueller: wp.array4d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    x: wp.array(dtype=wp.float64),
    result: wp.array2d(dtype=wp.float64),
):
    """Integrate the Mueller intensity element against Legendre quadrature."""
    i, ell = wp.tid()
    total = wp.float64(0.0)
    for a in range(weights.shape[0]):
        total += mueller[i, a, 0, 0] * weights[a, ell]
    result[i, ell] = total * wp.float64(4.0) / (x[i] * x[i])


def load_water(wave):
    """Interpolate published liquid-water n and k at wavelengths in nm."""
    table = np.array(
        [
            line.split()
            for line in (HERE / "water_hale.yml").read_text().splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ],
        dtype=float,
    )
    return np.interp(wave, table[:, 0] * 1000, table[:, 1]) + 1j * np.exp(
        np.interp(wave, table[:, 0] * 1000, np.log(table[:, 2]))
    )


def run(sizes, step, max_um, degree, tag):
    """Tabulate extinction and scattering moments and verify independent oracles."""
    TMP.mkdir(parents=True, exist_ok=True)
    wp.init()
    start = time.monotonic()
    wave = [450.0, 550.0, 700.0]
    spec = []
    for n in sizes:
        frequencies, beam, sky = solar_bands(n)
        spec.extend(
            dict(n=n, j=j, nm=nm, beam=b, sky=s)
            for j, (nm, b, s) in enumerate(zip(frequencies, beam, sky))
        )
        wave.extend(frequencies)
    radius = (
        np.unique(
            np.r_[
                np.geomspace(1, 200, 81),
                np.arange(200, max_um * 1000 + step, step),
            ]
        )
        * 1e-9
    )
    wave = np.array(wave)
    index = load_water(wave)
    # r-normalized efficiencies: extinction, then scattering times each normalized
    # phase moment l=0..degree. Users multiply by the actual per-bin geometric area.
    eff = np.empty((len(radius), len(wave), degree + 2), dtype=np.float32)
    oracle = []
    phase_error = np.zeros(2)
    angular_cases = 0
    for j, (nm, ri) in enumerate(zip(wave, index)):
        x = 2 * np.pi * radius / (nm * 1e-9)
        nang = int(np.ceil((x.max() + 5 * np.cbrt(x.max()) + 30) / 32)) * 32
        nodes, wg = roots_legendre(nang)
        weights = (
            wg[:, None] * 0.5 * np.polynomial.legendre.legvander(nodes, degree)
        )
        devweights = wp.array(weights, dtype=wp.float64, device="cuda:0")
        devangles = wp.array(
            np.arccos(nodes), dtype=wp.float64, device="cuda:0"
        )
        for left in range(0, len(radius), 512):
            chosen = slice(left, left + 512)
            these = x[chosen]
            devx = wp.array(these, dtype=wp.float64, device="cuda:0")
            nri = np.tile([ri.real, ri.imag], (len(these), 1))
            ws = prepare_mie(
                wp.array(nri, dtype=wp.vec2d, device="cuda:0"),
                devx,
                angles=devangles,
                coefficient_capacity=1024,
                case_tile=512,
                angle_tile=64,
            )
            ws.enqueue()
            ws.check()
            calculated = wp.empty(
                (len(these), degree + 1), dtype=wp.float64, device="cuda:0"
            )
            wp.launch(
                moments,
                dim=(len(these), degree + 1),
                inputs=[ws.mueller, devweights, devx, calculated],
                device="cuda:0",
            )
            mom = calculated.numpy()
            q = ws.efficiencies.numpy()
            angular_cases += len(these) * nang
            err = np.array(
                [
                    np.max(np.abs(mom[:, 0] / q[:, 1] - 1)),
                    np.max(np.abs(mom[:, 1] / mom[:, 0] - q[:, 3])),
                ]
            )
            phase_error = np.maximum(phase_error, err)
            if err.max() > 2e-5 or np.min(q[:, 2]) < -1e-10:
                raise ValueError((nm, err, q[:, 2].min()))
            eff[chosen, j, 0] = q[:, 0]
            eff[chosen, j, 1:] = mom
        if j in (0, 1, 2, 6, len(wave) // 2, len(wave) - 1):
            pick = np.round(np.linspace(0, len(radius) - 1, 19)).astype(int)
            cpu = get_mie_efficiencies(radius[pick], nm * 1e-9, ri)
            oracle.append(
                float(np.max(np.abs(cpu.extinction - eff[pick, j, 0])))
            )
        if j % 12 == 0:
            print(
                f"{j}/{len(wave)} wavelength nodes in {time.monotonic() - start:.1f}s",
                flush=True,
            )
    suffix = "_" + tag if tag else ""
    filename = TMP / f"sphere_table{suffix}.npz"
    np.savez(filename, wave=wave, radius=radius, eff=eff, index=index)
    import pandas as pd

    pd.DataFrame(spec).to_csv(
        HERE / f"spectral_bands{suffix}.csv", index=False, float_format="%.12g"
    )
    with filename.open("rb") as handle:
        table_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    checks = dict(
        table_sha256=table_hash,
        device=wp.get_device("cuda:0").name,
        step_nm=step,
        radii=len(radius),
        wavelengths=len(wave),
        degree=degree,
        angular_evaluations=angular_cases,
        mie_sizes=len(radius) * len(wave),
        phase_closure_relative_q=float(phase_error[0]),
        phase_closure_absolute_g=float(phase_error[1]),
        cpu_cuda_absolute_q_after_float32_storage=max(oracle),
        elapsed_seconds=time.monotonic() - start,
        inputs={
            key: hashlib.sha256((HERE / key).read_bytes()).hexdigest()
            for key in ("water_hale.yml", "astmg173.csv")
        },
    )
    (HERE / f"sphere_checks{suffix}.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(checks, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bands", nargs="+", type=int, default=[24, 48, 96])
    p.add_argument("--step", type=float, default=6.0)
    p.add_argument("--max-um", type=float, default=17.0)
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    run(a.bands, a.step, a.max_um, a.degree, a.tag)
