"""Independent angular normalization, CPU image quadrature and tabulated effect checks."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from particula.optics import get_mie_efficiencies, get_scattering_amplitudes
from instruments import WATER, annular_signal, geometry, mie, profiles, sobol
from simulate import device_areas

ROOT = Path(__file__).resolve().parent


def cpu_area(profile, d, pitch, offset, dr, q):
    half = int(np.ceil((d / 2 + 6) / pitch)) + 1
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    dark = np.zeros_like(x, dtype=float)
    sub = (np.arange(q) + 0.5) / q - 0.5
    for dy in sub:
        for dx in sub:
            radius = (
                np.hypot(x + dx + 0.5 - offset[0], y + dy + 0.5 - offset[1])
                * pitch
            )
            dark += np.interp(
                radius, dr * np.arange(len(profile)), profile, right=0
            )
    mask = dark >= 0.4 * q * q
    adjacent = np.any(mask[1:] & mask[:-1]) or np.any(
        mask[:, 1:] & mask[:, :-1]
    )
    return 2 * pitch * np.sqrt(mask.sum() / np.pi) if adjacent else 0.0


def run(out):
    diam = np.array([2.0, 4.9, 6.0, 9.18, 13.0, 25.6, 39.0])
    max_rel, max_pixel = 0.0, 0.0
    draws = sobol(32, seed=17)
    z, p, ix = geometry(5, draws)
    dr = 0.04
    for nm in [633, 660]:
        mu, weights = np.polynomial.legendre.leggauss(320)
        s1, s2 = get_scattering_amplitudes(mie(diam, nm, WATER), np.arccos(mu))
        cross = (
            (abs(s1) ** 2 + abs(s2) ** 2)
            @ weights
            * np.pi
            / (2 * np.pi / (nm / 1000)) ** 2
        )
        ref = (
            get_mie_efficiencies(diam * 0.5e-6, nm * 1e-9, WATER).scattering
            * np.pi
            * (diam / 2) ** 2
        )
        max_rel = max(max_rel, np.max(abs(cross / ref - 1)))
        np.testing.assert_allclose(cross, ref, rtol=2e-8)
        annulus = annular_signal(diam, nm, 48)
        assert np.all((annulus > 0) & (annulus < cross))
        profile = profiles(diam, nm, WATER, z, np.arange(0, 52, dr), 80, "mie")
        gpu = device_areas(profile, diam, p, draws, ix, dr, 0.4, 3)
        cpu = np.array(
            [
                [
                    cpu_area(profile[i, iz], d, p[iz], xy, dr, 3)
                    for xy, iz in zip(draws, ix)
                ]
                for i, d in enumerate(diam)
            ]
        )
        max_pixel = max(max_pixel, np.max(abs(gpu - cpu)))
        np.testing.assert_allclose(gpu, cpu, atol=1e-5, rtol=0)
    a = pd.read_csv(out / "attribution.csv")
    np.testing.assert_allclose(
        a.f633 - a.i660, a.wavelength + a.method, atol=1e-10
    )
    np.testing.assert_allclose(
        a.native_gap,
        a.true_band_detection_gap
        + a.continuous_size_migration
        + a.channel_reporting,
        atol=1e-10,
    )
    keys = ["stage", "case", "day", "source", "low_um", "high_um"]
    c = a.pivot(index=keys, columns="calibration", values="native_gap")
    np.testing.assert_allclose(c.frozen, c.retuned)
    d = pd.read_csv(out / "refined-curves.csv.gz")
    assert d[["detect", "band13", "band1307"]].ge(0).all().all()
    assert (
        (d.band13 <= d.band1307 + 1e-9).all()
        and (d.band1307 <= d.detect + 1e-9).all()
        and (d.detect <= 1).all()
    )
    assert (d.detect <= d.trigger + 1e-9).all()
    np.testing.assert_allclose(
        a.true_band_trigger_gap + a.native_range_censor_gap,
        a.true_band_detection_gap,
        atol=1e-10,
    )
    result = dict(
        full_sphere_max_relative_error=max_rel,
        tested_cuda_images=2 * len(diam) * len(draws),
        cpu_gpu_max_diameter_error_um=max_pixel,
        passed_attribution_rows=len(a),
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
