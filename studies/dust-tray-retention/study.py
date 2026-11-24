# %% Inputs and physical properties
"""Replay the Jia et al. early-to-late deposition comparison (CPU)."""

import argparse
import csv
import hashlib
import json
from importlib import metadata
from itertools import permutations
import platform
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import particula as par
import scipy

ROOT = Path(__file__).resolve().parent
BOOK = ROOT / "Supporting_data_v2.xlsx"
WINDS = (0, 2, 4, 6)
PUBLISHED = [(13.4, 10.8), (7.1, 9.8), (6.3, 7.5), (29.3, 17.9)]
CUTS = np.array([2.5, 10, 20, 30, 40, 63])
TEMP, PRESSURE = 293.15, 101325.0
DENSITY, HEIGHT = 2650.0, 2.0


def read_workbook():
    """Read nonempty cells in the two documented OOXML sheets, including cached numbers."""
    assert (
        hashlib.md5(BOOK.read_bytes()).hexdigest()
        == "07464b6f980a24a0b19598d503b7cd99"
    )
    with ZipFile(BOOK) as book:
        strings = [
            "".join(v.itertext())
            for v in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]

        def rows(sheet):
            result = {}
            for row in ET.fromstring(
                book.read(f"xl/worksheets/sheet{sheet}.xml")
            ).findall(".//{*}row"):
                cells = {}
                for cell in row:
                    val = cell.findtext("{*}v")
                    if val is not None:
                        cells[re.sub(r"\d+", "", cell.get("r"))] = (
                            strings[int(val)]
                            if cell.get("t") == "s"
                            else float(val)
                        )
                result[int(row.get("r"))] = cells
            return result

        spectra, original, kinetic = rows(2), rows(3), rows(1)
    mass = {}
    for row in original.values():
        if isinstance(row.get("D"), float):
            label = (int(row["A"]), row["C"], row["B"])
            mass.setdefault(label, []).append(1000 * row["D"])
    return spectra, mass, kinetic


def velocities(diameter_um, density):
    radius = np.asarray(diameter_um) * 0.5e-6
    eta = par.gas.get_dynamic_viscosity(TEMP)
    mean_path = par.gas.get_molecule_mean_free_path(
        temperature=TEMP, pressure=PRESSURE, dynamic_viscosity=eta
    )
    slip = par.particles.get_cunningham_slip_correction(
        par.particles.get_knudsen_number(mean_path, radius)
    )
    return par.particles.get_particle_settling_velocity_with_drag(
        radius, density, PRESSURE / (287.05 * TEMP), eta, slip
    )


def factors(d, rho=DENSITY, height=HEIGHT, closure="mixed"):
    rate = velocities(d, rho) / height

    def deposited(t):
        if closure == "unmixed":
            return np.minimum(rate * t, 1.0)
        return -np.expm1(-rate * t)

    early, late = deposited(1800), deposited(10800)
    assert np.all((early > 0) & (late >= early) & (late <= 1))
    transfer = late / early
    assert np.all(np.diff(transfer) <= 1e-10) and np.all(transfer <= 6 + 1e-10)
    return transfer


def normalize(x):
    assert np.isfinite(x).all() and np.min(x) >= 0 and x.sum() > 0
    return x / x.sum()


def median(d, p):
    return np.exp(np.interp(0.5, np.cumsum(normalize(p)), np.log(d)))


def tv(a, b):
    return 50 * np.abs(a - b).sum()


def save(name, entries):
    with (ROOT / name).open("w") as out:
        writer = csv.DictWriter(
            out, fieldnames=list(entries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(entries)


def particula_revision() -> str | None:
    """Return the Git revision of the imported Particula when verifiable."""
    module = Path(par.__file__).resolve()
    # A source import can shadow a different installed distribution. Check
    # the project beside the imported package, never an enclosing repository.
    root = module.parent.parent
    git_dir = root / ".git"
    project = root / "pyproject.toml"
    if git_dir.exists() and project.is_file():
        with project.open("rb") as stream:
            if (
                tomllib.load(stream).get("project", {}).get("name")
                != "particula"
            ):
                return None
        try:
            return subprocess.check_output(
                [
                    "git",
                    f"--git-dir={git_dir}",
                    "rev-parse",
                    "--verify",
                    "HEAD",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    try:
        dist = metadata.distribution("particula")
    except metadata.PackageNotFoundError:
        return None
    if dist.locate_file("particula/__init__.py").resolve() != module:
        return None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        vcs = json.loads(direct_url).get("vcs_info", {})
        if vcs.get("vcs") == "git":
            return vcs.get("commit_id")
    return None


# %% Reference and transfer scores
def run(winds):
    sheets, mass, kinetic = read_workbook()
    scores, spectra, bands, sensitivity, masses, totals, refinement = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    consistency, band_growth, spread = [], [], []
    for w in winds:
        row = 4 + 2 * w
        header = sheets[row]
        cols = [k for k in header if k not in ("A", "B", "C")]
        assert (header | sheets[row + 1])["A"] == f"{w}m/s"
        assert [sheets[r]["B"] for r in (row + 1, row + 2)] == ["0.5h", "3h"]
        d, raw_early, raw_late = [
            np.array([sheets[r][c] for c in cols]) for r in range(row, row + 3)
        ]
        assert len(d) == 41 and np.all(np.diff(d) > 0)
        p, obs = normalize(raw_early), normalize(raw_late)
        transfer = factors(d)
        model = normalize(p * transfer)
        d0, d1, dp = [median(d, s) for s in (p, obs, model)]
        scores.append(
            dict(
                wind_ms=w,
                early_d50_um=d0,
                measured_d50_um=d1,
                settling_d50_um=dp,
                observed_shift_um=d1 - d0,
                settling_shift_um=dp - d0,
                settling_error_um=dp - d1,
                copy_error_um=d0 - d1,
                settling_tv_pp=tv(model, obs),
                copy_tv_pp=tv(p, obs),
                early_mode_um=d[p.argmax()],
                observed_mode_um=d[obs.argmax()],
                predicted_mode_um=d[model.argmax()],
                raw_early_sum_pct=raw_early.sum(),
                raw_late_sum_pct=raw_late.sum(),
                published_early_um=PUBLISHED[w // 2][0],
                published_late_um=PUBLISHED[w // 2][1],
            )
        )
        for i, size in enumerate(d):
            spectra.append(
                dict(
                    wind_ms=w,
                    channel_um=size,
                    early_pct=100 * p[i],
                    measured_pct=100 * obs[i],
                    settling_pct=100 * model[i],
                    transfer=transfer[i],
                )
            )
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, np.inf)):
            keep = (d > lo) & (d <= hi)
            shares = [100 * s[keep].sum() for s in (p, obs, model)]
            bands.append(
                dict(
                    wind_ms=w,
                    band=f"{lo}-{hi}",
                    early_pct=shares[0],
                    observed_pct=shares[1],
                    settling_pct=shares[2],
                    observed_change_pp=shares[1] - shares[0],
                    settling_change_pp=shares[2] - shares[0],
                )
            )
        for rho in (1500, 2000, 2650, 3000):
            for height in (0.5, 1, 2, 4):
                for closure in ("mixed", "unmixed"):
                    pred = normalize(p * factors(d, rho, height, closure))
                    sensitivity.append(
                        dict(
                            wind_ms=w,
                            density_kgm3=rho,
                            height_m=height,
                            closure=closure,
                            d50_um=median(d, pred),
                            tv_pp=tv(pred, obs),
                        )
                    )
        # Alternative interpretation of channel locations: uniformly distributed early
        # volume per log diameter between adjacent reported cumulative coordinates.
        lower = np.r_[d[0] ** 2 / d[1], d[:-1]]
        for count in (4, 8, 16, 32):
            positions = np.exp(
                np.log(lower)[:, None]
                + np.log(d / lower)[:, None]
                * ((np.arange(count) + 0.5) / count)
            )
            fine_factors = factors(positions.ravel()).reshape(len(d), count)
            pred = normalize(p * fine_factors.mean(axis=1))
            refinement.append(
                dict(
                    wind_ms=w,
                    subchannels=count,
                    d50_um=median(d, pred),
                    tv_pp=tv(pred, obs),
                )
            )
        total0, total1 = [np.array(mass[w, t, 63]) for t in (0.5, 3.0)]
        assert len(total0) == 5 and len(total1) == (4 if w == 4 else 5)
        # Every nonzero input and endpoint channel lies within the 63 µm cut.
        assert np.all((p + obs)[d > 63] == 0)
        total_prediction = total0.mean() * np.dot(p, transfer)
        totals.append(
            dict(
                wind_ms=w,
                early_n=len(total0),
                late_n=len(total1),
                early_mg=total0.mean(),
                early_sem_mg=total0.std(ddof=1) / np.sqrt(len(total0)),
                measured_mg=total1.mean(),
                measured_sem_mg=total1.std(ddof=1) / np.sqrt(len(total1)),
                predicted_mg=total_prediction,
                observed_ratio=total1.mean() / total0.mean(),
                predicted_ratio=total_prediction / total0.mean(),
            )
        )
        # Use differences of means; no assumption about tray IDs or cross-threshold covariance.
        early_means, late_means = [
            [np.mean(mass[w, t, c]) for c in CUTS] for t in (0.5, 3.0)
        ]
        for time, measured in ((0.5, p), (3.0, obs)):
            denom = np.array(mass[w, time, 63])
            for cut in CUTS:
                numer = np.array(mass[w, time, cut])
                consistency.append(
                    dict(
                        wind_ms=w,
                        time_h=time,
                        cut_um=cut,
                        n_mass=len(numer),
                        n_total=len(denom),
                        cumulative_mass_mg=numer.mean(),
                        laser_cdf_pct=100
                        * np.interp(
                            np.log(cut), np.log(d), np.cumsum(measured)
                        ),
                        threshold_mass_share_pct=100
                        * numer.mean()
                        / denom.mean(),
                        max_any_pair_pct=100 * numer.max() / denom.min(),
                    )
                )
        for i, cut in enumerate(CUTS):
            prev = 0 if i == 0 else CUTS[i - 1]
            early_mg = np.diff(early_means, prepend=0)[i]
            late_mg = np.diff(late_means, prepend=0)[i]
            group_sizes = [
                len(mass[w, 3.0, c]) for c in ([cut] if i == 0 else [prev, cut])
            ]
            status = (
                "unequal group sizes; difference of means only"
                if len(set(group_sizes)) > 1
                else "difference of available means"
            )
            masses.append(
                dict(
                    wind_ms=w,
                    lower_um=prev,
                    upper_um=cut,
                    early_mg=early_mg,
                    late_mg=late_mg,
                    increment_mg=late_mg - early_mg,
                    late_interpretation=status,
                )
            )
        # Direct 10–20 µm mass test independent of the 41-channel representation.
        # F is the archive's gram mean; avoid the ambiguous area conversion in G.
        gram_means = {
            (r["E"], r["D"]): r["F"]
            for r in kinetic.values()
            if r.get("C") == w and "F" in r and r.get("D") in (10, 20)
        }
        initial_band = early_means[2] - early_means[1]
        # Unlabeled endpoint trays: bound the standard error over all physically
        # feasible pairings of the two nested thresholds within each endpoint.
        errors = []
        for t in (0.5, 3.0):
            upper_mass = np.array(mass[w, t, 20])
            possible = [
                upper_mass - low for low in permutations(mass[w, t, 10])
            ]
            errors.append(
                [
                    x.std(ddof=1) / np.sqrt(len(x))
                    for x in possible
                    if min(x) >= 0
                ]
            )
        spread.append(
            dict(
                wind_ms=w,
                feasible_early=len(errors[0]),
                feasible_late=len(errors[1]),
                min_change_sem_mg=np.hypot(min(errors[0]), min(errors[1])),
                max_change_sem_mg=np.hypot(max(errors[0]), max(errors[1])),
            )
        )
        for t in (0.5, 1, 1.5, 2, 3):
            measured_band = 1000 * (gram_means[t, 20] - gram_means[t, 10])
            bounds = []
            for density, height in ((DENSITY, HEIGHT), (1500.0, 4.0)):
                slowest = float(velocities(10.0, density)) / height
                upper = np.expm1(-slowest * t * 3600) / np.expm1(
                    -slowest * 1800
                )
                bounds.append(initial_band * upper)
            band_growth.append(
                dict(
                    wind_ms=w,
                    time_h=t,
                    archived_mean_mg=measured_band,
                    model_upper_mg=bounds[0],
                    slow_settling_upper_mg=bounds[1],
                    observed_to_initial=measured_band / initial_band,
                )
            )
        assert np.isclose(
            band_growth[-1]["archived_mean_mg"],
            late_means[2] - late_means[1],
            atol=0.005,
        )
    if winds == (0,):
        save("still-air.csv", scores)
        print(scores[0])
        return
    revision = particula_revision()
    for filename, data in (
        ("scores", scores),
        ("channels", spectra),
        ("bands", bands),
        ("sensitivity", sensitivity),
        ("threshold-masses", masses),
        ("total-mass", totals),
        ("refinement", refinement),
        ("consistency", consistency),
        ("middle-band-growth", band_growth),
        ("middle-band-spread", spread),
    ):
        save(filename + ".csv", data)
    meta = dict(
        python=platform.python_version(),
        particula=par.__version__,
        revision=revision,
        numpy=np.__version__,
        scipy=scipy.__version__,
        model="exact integrated per-channel depletion, CPU float64",
        wind_ms=winds,
        height_m=HEIGHT,
        density_kg_m3=DENSITY,
        temperature_K=TEMP,
        pressure_Pa=PRESSURE,
        speeds_at_2_and_10_um_mm_s=(
            1000 * velocities(np.array([2.0, 10.0]), DENSITY)
        ).tolist(),
        input_md5=hashlib.md5(BOOK.read_bytes()).hexdigest(),
        paper_doi="10.5194/acp-26-14245-2026",
        data="https://zenodo.org/records/22962214",
    )
    (ROOT / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--still", action="store_true")
    run((0,) if parser.parse_args().still else WINDS)
