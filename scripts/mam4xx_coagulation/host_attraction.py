"""Replay archived MAM fine-mode updates and a proposed coarse collection."""

import csv
import ctypes
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtr
from scripts.mam4xx_coagulation.attraction_campaign import laws
from scripts.mam4xx_coagulation.build_reference import HEADER_SHA256
from scripts.mam4xx_coagulation.modal_attraction import (
    HAMAKER,
    HOURS,
    distribution,
    fractions,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def compile_host(header, directory):
    """Compile checked-in host scaffolding around the checksum-pinned source."""
    raw = header.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
        raise ValueError("Unexpected archived MAM4xx header")
    source = raw.decode()
    a = source.index("namespace coagulation {")
    b = source.rindex(
        "KOKKOS_INLINE_FUNCTION",
        a,
        source.index("void coagulation_rates_1box("),
    )
    (directory / "modal_core.inc").write_text(source[a:b] + "}\n")
    (directory / "regime_core.inc").write_text(
        (HERE / "reference_driver.cpp").read_text().split("int main()")[0]
    )
    compiler = shutil.which("c++")
    if not compiler:
        raise RuntimeError("A C++17 compiler is required for the archive")
    lib = directory / "fine.so"
    subprocess.run(  # noqa: S603 - fixed driver and verified archive
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-I",
            str(directory),
            str(HERE / "host_attraction.cpp"),
            "-o",
            str(lib),
        ],
        check=True,
    )
    step = ctypes.CDLL(str(lib)).advance_fine
    vector = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    )
    step.argtypes = [ctypes.c_double, vector, vector, vector]
    step.restype = None
    return step


def coefficients(n, v, logs, edges, kernel):
    """Return number- and donor-volume-weighted K at the same modal state."""
    median = 1e3 * np.cbrt(6 * v / (np.pi * n * np.exp(4.5 * logs**2)))
    wn = fractions(edges, median, logs)
    wv = fractions(edges, median * np.exp(3 * logs**2), logs)
    return wn @ kernel @ wn.T, wv @ kernel @ wn.T


def collect(n, v, kn, kv, dt):
    """Integrate the coarse-only subsystem at fixed pair coefficients."""
    rate = 0.5 * kn[2, 2]
    exposure = np.log1p(rate * n[2] * dt) / rate
    n[2] /= 1 + rate * n[2] * dt
    n[:2] *= np.exp(-kn[:2, 2] * exposure)
    moved = -v[:2] * np.expm1(-kv[:2, 2] * exposure)
    v[:2] -= moved
    v[2] += moved.sum()


def replay(step, reg, coarse, law, res, dt, interval):
    """Update modal multipliers periodically along the archived trajectory."""
    dg, sigma, n = distribution(reg, "host", coarse).T.copy()
    n = n.copy()
    logs = np.log(sigma)
    v = np.pi / 6 * dg**3 * np.exp(4.5 * logs**2) * n
    start_v = v.sum()
    edges, _, kernels, _ = laws(res, HAMAKER)
    snapshots = []
    pair_rows = []
    for sec in range(0, 86401, dt):
        if sec % interval == 0 or sec in HOURS * 3600:
            bn, bv = coefficients(n, v, logs, edges, kernels["bare"])
            kn, kv = coefficients(n, v, logs, edges, kernels[law])
            factor = np.array(
                [
                    kn[0, 0] / bn[0, 0],
                    kn[1, 1] / bn[1, 1],
                    kn[0, 1] / bn[0, 1],
                    kv[0, 1] / bv[0, 1],
                ]
            )
        if sec in HOURS * 3600:
            cut = (
                np.log(
                    100
                    / (
                        1e3
                        * np.cbrt(6 * v / (np.pi * n * np.exp(4.5 * logs**2)))
                    )
                )
                / logs
            )
            snapshots.append(
                [
                    sec // 3600,
                    n @ ndtr(cut),
                    n @ ndtr(-cut),
                    v @ ndtr(3 * logs - cut),
                    n[0],
                    n[1],
                    n[2] if coarse else 0,
                    v[0],
                    v[1],
                    v[2] if coarse else 0,
                ]
            )
            for i, j in zip(*np.triu_indices(len(n)), strict=True):
                pair_rows.append(
                    [
                        sec // 3600,
                        i,
                        j,
                        bn[i, j],
                        kn[i, j],
                        kn[i, j] / bn[i, j],
                        bv[i, j] if i != j else None,
                        kv[i, j] if i != j else None,
                        kv[i, j] / bv[i, j] if i != j else None,
                    ]
                )
        if sec == 86400:
            break
        if coarse:
            collect(n, v, kn, kv, dt / 2)
        step(dt, factor, n, v)
        if coarse:
            collect(n, v, kn, kv, dt / 2)
    if np.any(n <= 0) or np.any(v <= 0):
        raise ValueError("Archived modal update left the physical domain")
    np.testing.assert_allclose(v.sum(), start_v, rtol=2e-10, atol=0)
    return snapshots, pair_rows


def write_tables(header, output):
    """Export replay controls and same-state number and volume coefficients."""
    archive = np.genfromtxt(
        ROOT / "docs/data/mam4xx-coagulation/regional-modal.csv",
        delimiter=",",
        names=True,
    )
    case_cols = [
        "regime",
        "coarse",
        "law",
        "bins_per_decade",
        "host_dt_s",
        "update_s",
    ]
    targets = (
        output / "archived-attraction.csv",
        output / "mode-coefficients.csv",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory() as folder,
        targets[0].open("w", newline="") as f,
        targets[1].open("w", newline="") as g,
    ):
        step = compile_host(header, Path(folder))
        results, pairs = (
            csv.writer(stream, lineterminator="\n") for stream in (f, g)
        )
        results.writerow(
            case_cols
            + [
                "hour",
                "below_100_cm3",
                "above_100_cm3",
                "volume_above_100_um3_cm3",
                "aitken_cm3",
                "accumulation_cm3",
                "coarse_cm3",
                "aitken_um3_cm3",
                "accumulation_um3_cm3",
                "coarse_um3_cm3",
            ]
        )
        pairs.writerow(
            case_cols
            + [
                "hour",
                "donor_index",
                "receiver_index",
                "number_bare_cm3_s",
                "number_test_cm3_s",
                "number_factor",
                "volume_bare_cm3_s",
                "volume_test_cm3_s",
                "volume_factor",
            ]
        )
        for reg, idx in [("polluted", 2), ("marine", 0)]:
            for coarse in (False, True):
                for law in ("bare", "complete"):
                    for res, dt, interval in [
                        (24, 30, 300),
                        (48, 10, 60),
                        (96, 10, 30),
                    ]:
                        config = [reg, coarse, law, res, dt, interval]
                        rows, coeffs = replay(
                            step, reg, coarse, law, res, dt, interval
                        )
                        if not coarse and law == "bare":
                            ref = archive[
                                (archive["regime"] == idx)
                                & (archive["dt_s"] == dt)
                            ]
                            expected = np.column_stack(
                                [
                                    ref[k]
                                    for k in (
                                        "n_ait_cm3",
                                        "n_acc_cm3",
                                        "v_ait_um3_cm3",
                                        "v_acc_um3_cm3",
                                    )
                                ]
                            )
                            np.testing.assert_allclose(
                                np.asarray(rows)[:, [4, 5, 7, 8]],
                                expected,
                                rtol=1e-10,
                                atol=0,
                            )
                        results.writerows(config + r for r in rows)
                        if (res, law) == (96, "complete"):
                            pairs.writerows(config + r for r in coeffs)


def generate(header, output):
    """Publish the two finished tables after the archive replay passes."""
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as folder:
        staged = Path(folder)
        write_tables(header, staged)
        for path in staged.iterdir():
            path.replace(output / path.name)


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
