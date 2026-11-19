"""Compare neutral kernels, tracer survival and chamber loss with Cai et al."""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import particula as par
import scipy
from scipy.integrate import simpson
from scipy.optimize import curve_fit

GIT = shutil.which("git") or "git"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get(
        "PARTICULA_SCAVENGING_OUTPUT", ROOT / ".artifacts/nacl-scavenging"
    )
)
TEMP, PRESSURE, DENSITY = 293.15, 101325.0, 2160.0
SOURCES = [
    "Particula hard sphere",
    "Published hard sphere",
    "Published enhanced",
    "Measured",
]


def revision():
    """Identify the checkout that supplied the loaded study code."""
    return subprocess.check_output(  # noqa: S603 -- fixed Git arguments
        [GIT, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def archive_table(member):
    """Verify the source archive digest and read one tab-delimited figure."""
    archive = ROOT / "docs/data/nacl-scavenging/source-data.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != (
        "99af0abd636ae4a4031d99cd922619796f24c1ba164ae1d69babee2128663593"
    ):
        raise ValueError("Unexpected digest for the Cai et al. source archive")
    with zipfile.ZipFile(archive) as data:
        return np.loadtxt(
            io.BytesIO(data.read("data/" + member + ".txt")),
            skiprows=1,
            delimiter="\t",
        )


def brownian(d, t=TEMP, p=PRESSURE):
    """Return the neutral pair rate for a 100 nm collector in cm³/s."""
    radius = np.array([d, 100.0]) * 0.5e-9
    mass = (4 * np.pi / 3) * DENSITY * radius**3
    kernel = par.dynamics.get_brownian_kernel_via_system_state(
        radius, mass, t, p
    )
    np.testing.assert_allclose(kernel, kernel.T, rtol=1e-13, atol=0)
    np.testing.assert_equal(
        np.all(np.isfinite(kernel)) and np.all(kernel > 0), True
    )
    return kernel[0, 1] * 1e6


def transition_bridge(d):
    """Evaluate supplement S4–S6 with the same gas and particle properties."""
    diameter = np.array([d, 100.0]) * 1e-9
    mass = np.pi / 6 * DENSITY * diameter**3
    diffusion = par.particles.get_diffusion_coefficient_via_system_state(
        diameter / 2, TEMP, PRESSURE
    )
    speed = par.particles.get_mean_thermal_speed(mass, TEMP)
    kn = 6 * diffusion.sum() / (diameter.sum() * np.linalg.norm(speed))
    continuum = 2 * np.pi * diameter.sum() * diffusion.sum()
    return 1e6 * continuum * (1 + kn) / (1 + 0.377 * kn + 4 / 3 * kn * (1 + kn))


def write(name, rows):
    """Save a named baseline table."""
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read(name):
    """Read rows from a previously generated baseline table."""
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def run():
    """Compare measured rates, hard-sphere predictions and particle survival."""
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not Path(par.__file__).resolve().is_relative_to(ROOT / "particula"):
        raise RuntimeError("Set PYTHONPATH to the study repository root")
    tab = archive_table("Fig6_CoagCoef")
    obs = tab[np.isfinite(tab[:, 0]), :2]
    np.testing.assert_equal(list(obs[:, 0]) == [3.6, 3.7, 4.0, 5.0, 10.0], True)
    d_obs, b_obs = obs.T
    sizes = tab[:, 6]

    def curves(x):
        return [
            np.array([brownian(d) for d in x]),
            np.interp(x, sizes, tab[:, 7]),
            np.interp(x, sizes, tab[:, 9]),
            np.exp(np.interp(np.log(x), np.log(d_obs), np.log(b_obs))),
        ]

    base, published, enhanced, _ = curves(d_obs)
    data = []
    for i, (d, b) in enumerate(obs):
        tests = {
            label: 100 * (val / base[i] - 1)
            for label, val in [
                ("size_minus5", brownian(d * 0.95)),
                ("size_plus5", brownian(d * 1.05)),
                ("temp_minus3", brownian(d, TEMP - 3)),
                ("temp_plus3", brownian(d, TEMP + 3)),
                ("pressure_minus5", brownian(d, TEMP, PRESSURE * 0.95)),
                ("pressure_plus5", brownian(d, TEMP, PRESSURE * 1.05)),
            ]
        }
        data.append(
            dict(
                d_nm=d,
                measured=b,
                particula=base[i],
                authors_hard_sphere=published[i],
                matched_transport_bridge=transition_bridge(d),
                authors_enhanced=enhanced[i],
                measured_over_particula=b / base[i],
                particula_error_pct=100 * (base[i] / b - 1),
                published_enhanced_error_pct=100 * (enhanced[i] / b - 1),
                **tests,
            )
        )
    write("coefficients.csv", data)
    # Fig. 4 columns 3 and 7 are time (h) and normalized concentration at 4 nm.
    decay = archive_table("Fig4_OverallLifetime")
    t, y = decay[:, 2], decay[:, 6]
    keep = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[keep], y[keep]

    def concentration(t, amplitude, rate):
        return amplitude * np.exp(-rate * t)

    (a, rate), _ = curve_fit(concentration, t, y, p0=[1, 7], bounds=(0, np.inf))
    lograte = -np.polyfit(t, np.log(y), 1)[0]
    beta4 = np.array([c[2] for c in [base, published, enhanced, b_obs]])
    chamber = [
        dict(
            source=k,
            beta_cm3_s=b,
            background_s=rate / 3600,
            crossover_cm3=(rate / 3600) / b,
            capture_half_life_min=np.log(2) / (b * 1e4 * 60),
        )
        for k, b in zip(SOURCES, beta4, strict=True)
    ]
    write("chamber.csv", chamber)

    # Integration grids include observation and theory knots.
    def hazards(n):
        x = np.linspace(3.6, 10, 64 * n + 1)
        rates = curves(x)
        return np.array([simpson(c, x=x) for c in rates]) * 3600 / 3

    grid = sizes[(sizes >= d_obs[0]) & (sizes <= d_obs[-1])]
    series = curves(grid)
    write(
        "model-curves.csv",
        [
            dict(d_nm=d, particula=h, authors_hard_sphere=c, authors_enhanced=e)
            for d, h, c, e in zip(grid, *series[:3], strict=True)
        ],
    )
    h0, h = hazards(4), hazards(8)
    x = np.linspace(3.6, 10, 513)
    lin = simpson(np.interp(x, d_obs, b_obs), x=x) * 3600 / 3
    np.testing.assert_equal(max(abs(h / h0 - 1)) < 1e-06, True)
    np.testing.assert_allclose(
        simpson(np.full_like(x, 1e-7), x=x) * 1200,
        1e-7 * 6.4 * 1200,
        rtol=1e-14,
    )
    cohorts = []
    for k, hazard in zip(SOURCES, h, strict=True):
        for n in [0, 100.0, 500.0, 1000.0, 3000.0, 10000.0]:
            cohorts.append(
                dict(
                    source=k,
                    n100_cm3=n,
                    survival=np.exp(-hazard * n),
                    half_survival_n100_cm3=np.log(2) / hazard,
                )
            )
        np.testing.assert_equal(
            np.all(
                np.diff([r["survival"] for r in cohorts if r["source"] == k])
                < 0
            ),
            True,
        )
    write("survival.csv", cohorts)
    meta = dict(
        beta_units="cm3/s",
        diameter_units="nm",
        pressure_Pa=PRESSURE,
        temperature_K=TEMP,
        density_kg_m3=DENSITY,
        sticking=1.0,
        decay_fit_amplitude=a,
        decay_points=len(t),
        decay_fit_h=rate,
        decay_logfit_h=lograte,
        decay_fit_rmse=np.sqrt(np.mean((concentration(t, a, rate) - y) ** 2)),
        dilution_h=0.6,
        quadrature_max_relative=max(abs(h / h0 - 1)),
        linear_interpolated_half_n_cm3=np.log(2) / lin,
        loglog_interpolated_half_n_cm3=np.log(2) / h[-1],
        python=platform.python_version(),
        particula=par.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        revision=revision(),
        backend="CPU",
        seconds=time.perf_counter() - start,
    )
    (OUT / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def plot():
    """Plot the baseline rate and survival comparisons from saved tables."""
    rows = read("coefficients.csv")
    survival = read("survival.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    grid = read("model-curves.csv")
    colors = ["#386787", "#888888", "#bb5b18", "#167462"]
    for k, key, color in zip(
        SOURCES,
        ["particula", "authors_hard_sphere", "authors_enhanced", "measured"],
        colors,
        strict=True,
    ):
        points = rows if key == "measured" else grid
        a.plot(
            [float(r["d_nm"]) for r in points],
            [float(r[key]) * 1e7 for r in points],
            "o"
            if key == "measured"
            else ("--" if key == "authors_hard_sphere" else "-"),
            color=color,
            label=k,
        )
        half = next(
            float(r["half_survival_n100_cm3"])
            for r in survival
            if r["source"] == k
        )
        n = np.geomspace(100, 10000, 200)
        b.plot(
            n,
            100 * np.exp(-np.log(2) * n / half),
            color=color,
            linestyle="--" if key == "authors_hard_sphere" else "-",
            label=k,
        )
    a.set(
        xlabel="Small-particle diameter (nm)",
        ylabel="Coagulation coefficient (10⁻⁷ cm³/s)",
        title="NaCl captured by 100 nm NaCl",
    )
    b.axhline(50, color=".65", linewidth=0.8)
    b.set(
        xscale="log",
        xlabel="100 nm particles (cm⁻³)",
        ylabel="Survival from 3.6 to 10 nm (%)",
        title="Predicted survival at 3 nm/h growth",
    )
    a.legend(frameon=False, fontsize=9)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(OUT / "scavenging.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    if not ap.parse_args().replot:
        run()
    plot()
