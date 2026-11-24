"""Synthetic laser-diffraction and transit-sizing tests for the Jia tray spectra."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import nnls, linprog
from particula.optics import (
    get_mie_coefficients,
    get_scattering_amplitudes,
)

ROOT = Path(__file__).resolve().parent
WATER = 1.333
LAMBDA = 0.520  # vacuum wavelength in micrometres
NOMINAL = 1.52 + 0.002j
WINDS = (0, 2, 4, 6)
ANGLES = np.geomspace(0.08, 130, 49)


def read(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (ROOT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


CHANNELS = read("channels.csv")
DIAM = np.array(
    [float(r["channel_um"]) for r in CHANNELS if r["wind_ms"] == "0"]
)
LOG = np.log(DIAM)
I = np.eye(len(DIAM))
D2 = np.diff(I, n=2, axis=0)


def spectrum(w, field):
    f = np.array([float(r[field]) for r in CHANNELS if int(r["wind_ms"]) == w])
    return f / f.sum()


def frac(q, cut):
    return np.interp(np.log(cut), LOG, np.cumsum(q))


def med(q):
    return np.exp(np.interp(0.5, np.cumsum(q) / q.sum(), LOG))


def cutrow(cut):
    return np.array([frac(e, cut) for e in I])


@lru_cache(maxsize=48)
def scattering(n, k, quadrature=36, diameter_order=3):
    """Annular scattering cross section per unit particle volume (µm⁻¹)."""
    edges = np.deg2rad(ANGLES)
    a, b = edges[:-1], edges[1:]
    nodes, weights = leggauss(quadrature)
    theta = (a + b)[:, None] / 2 + (b - a)[:, None] * nodes / 2
    weights = (b - a)[:, None] / 2 * weights * np.sin(theta) * 2 * np.pi
    ka = 2 * np.pi * WATER / LAMBDA
    output = np.zeros((len(a), len(DIAM)))
    # Uniform volume per log diameter within each nominal channel.
    nodes, vw = leggauss(diameter_order)
    for x, weight in zip(nodes, vw / 2, strict=True):
        ds = DIAM * np.exp(x * np.median(np.diff(LOG)) / 2)
        coef = get_mie_coefficients(complex(n, k) / WATER, ka * ds / 2)
        s1, s2 = get_scattering_amplitudes(coef, theta.ravel())
        power = (abs(s1) ** 2 + abs(s2) ** 2) / 2
        annulus = (power.reshape(len(ds), len(a), -1) * weights).sum(-1) / ka**2
        output += weight * (annulus / (np.pi / 6 * ds[:, None] ** 3)).T
    return output


def transit(aspect, offset=0.0, orientation="random", order=16):
    """Map true volume to apparent volume for accepted single-grain scan chords."""
    nodes, weights = leggauss(order)
    unit = (nodes + 1) / 2
    weights = weights / 2
    cos_axis, phi, off = np.meshgrid(
        unit, unit * np.pi / 2, unit * offset, indexing="ij"
    )
    # Short semi-axis is D/(2 A^(1/3)). Project onto the plane normal to sight.
    stretch = np.sqrt(1 + (aspect**2 - 1) * (1 - cos_axis**2))
    ratio = aspect ** (-1 / 3) / np.sqrt(
        np.cos(phi) ** 2 / stretch**2 + np.sin(phi) ** 2
    )
    if orientation == "long":
        ratio[:] = aspect ** (2 / 3)
    if orientation == "short":
        ratio[:] = aspect ** (-1 / 3)
    ratio *= np.sqrt(1 - off**2)
    ratio = ratio.ravel()
    p = np.einsum("i,j,k->ijk", weights, weights, weights).ravel()
    kernel = np.zeros_like(I)
    for j, d in enumerate(DIAM):
        log_chord = np.log(d * ratio)
        idx = np.searchsorted(LOG, log_chord, side="right").clip(
            1, len(DIAM) - 1
        )
        f = ((log_chord - LOG[idx - 1]) / (LOG[idx] - LOG[idx - 1])).clip(0, 1)
        # All accepted events count once. Converting apparent counts to volume weights by chord³.
        volume = p * ratio**3
        kernel[:, j] = np.bincount(
            idx - 1, weights=volume * (1 - f), minlength=len(DIAM)
        ) + np.bincount(idx, weights=volume * f, minlength=len(DIAM))
    return kernel


def retrieve(A, b, smooth, weight):
    # Unknown multiplicative concentration / gain; normalize only after solving.
    M = A / weight[:, None]
    y = b / weight
    M = np.vstack([M / np.sqrt(len(b)), smooth * D2])
    y = np.r_[y / np.sqrt(len(b)), np.zeros(len(D2))]
    q, _ = nnls(M, y, maxiter=1000)
    err = (A @ q - b) / weight
    return q / q.sum(), np.sqrt(np.mean(err**2)), max(abs(err))


def tolerance_bounds(A, b, weight, eps):
    """Normalized-bin LP bounds; one positive gain scales the synthetic observation."""
    aa = A / weight[:, None]
    yy = b / weight
    C = np.r_[np.c_[aa, -yy - eps], np.c_[-aa, yy - eps]]
    E = np.r_[np.ones(len(DIAM)), 0][None, :]
    opts = dict(
        A_ub=C,
        b_ub=np.zeros(len(C)),
        A_eq=E,
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
    )

    def extrem(c, sign):
        lp = linprog(np.r_[sign * c, 0], **opts)
        if not lp.success:
            if lp.status == 2:
                return None
            raise RuntimeError(lp.message)
        return float(sign * lp.fun)

    f20 = cutrow(20.0)
    middle = f20 - cutrow(10.0)
    if extrem(f20, 1) is None:
        return None

    # CDF is interpolated on log diameter; locate extremal median by bisection.
    def extreme_median(sign):
        low, high = LOG[0], LOG[-1]
        for _ in range(20):
            mid = (low + high) / 2
            if extrem(cutrow(np.exp(mid)), sign) >= 0.5:
                high = mid
            else:
                low = mid
        return np.exp((low + high) / 2)

    result = dict(
        median_lo=extreme_median(-1),
        median_hi=extreme_median(1),
        under20_lo=100 * extrem(f20, 1),
        under20_hi=100 * extrem(f20, -1),
        middle_lo=100 * extrem(middle, 1),
        middle_hi=100 * extrem(middle, -1),
    )
    opts["A_eq"] = np.r_[cutrow(63.0), 0.0][None, :]
    result.update(
        under20_over63_lo=100 * extrem(f20, 1),
        under20_over63_hi=100 * extrem(f20, -1),
    )
    return result


def run():
    matrices = {}
    reference = scattering(NOMINAL.real, NOMINAL.imag)
    for n in (1.48, 1.52, 1.55):
        for k in (0.0006, 0.002, 0.0048):
            matrices[f"n{n}_k{k}"] = scattering(n, k)
    rows = []
    bins = []
    bounds = []
    signal = []
    noise = []
    tests = []

    def record(w, h, family, case, reference_f, q, rms, peak, subgroup):
        rows.append(
            dict(
                wind=w,
                hours=h,
                family=family,
                case=case,
                group=subgroup,
                d50_um=med(q),
                under20_pct=100 * frac(q, 20),
                middle_pct=100 * (frac(q, 20) - frac(q, 10)),
                tv_from_archive_pp=50 * sum(abs(q - reference_f)),
                rms_pct=100 * rms,
                max_pct=100 * peak,
            )
        )
        bins.append(q)

    angular_masks = {
        "full": np.ones(len(ANGLES) - 1, bool),
        "limited": (ANGLES[:-1] >= 1) & (ANGLES[1:] <= 65),
    }
    geometries = {
        f"AR{ar}_{ori}_offset{off}": transit(ar, off, ori)
        for ar, ori, off in [
            (1.0, "random", 0.0),
            (1.5, "random", 0.0),
            (2.0, "random", 0.0),
            (1.5, "random", 0.35),
            (2.0, "random", 0.35),
            (1.5, "long", 0.0),
            (1.5, "short", 0.0),
            (2.0, "long", 0.0),
            (2.0, "short", 0.0),
        ]
    }
    for w in WINDS:
        for h, field in [(0.5, "early_pct"), (3.0, "measured_pct")]:
            f = spectrum(w, field)
            b0 = reference @ f
            for j, rr in enumerate(b0):
                signal.append(
                    dict(
                        wind=w,
                        hours=h,
                        inner_deg=ANGLES[j],
                        outer_deg=ANGLES[j + 1],
                        synthetic_per_um=rr,
                    )
                )
            for geometry, K in geometries.items():
                # Invert the archive's normalized apparent-volume bins directly.
                weight = np.full_like(f, max(f))
                q, rms, peak = retrieve(K, f, 0.01, weight)
                record(
                    w,
                    h,
                    "transit",
                    geometry,
                    f,
                    q,
                    rms,
                    peak,
                    "random" if "random" in geometry else "aligned",
                )
            for cut in (5.0, 10.0, 20.0):
                for side in ("fine", "coarse"):
                    # Independent particle detection with a smooth efficiency
                    # transition; the asymptotic efficiency ratio is two.
                    fine = 1 / (1 + (DIAM / cut) ** 4)
                    efficiency = 1 - 0.5 * (
                        fine if side == "fine" else 1 - fine
                    )
                    q = f / efficiency
                    q /= q.sum()
                    record(
                        w,
                        h,
                        "transit",
                        f"half_{side}_at{cut}",
                        f,
                        q,
                        0.0,
                        0.0,
                        "acceptance",
                    )
            for name, K in matrices.items():
                for coverage, mask in angular_masks.items():
                    b = b0[mask]
                    weight = np.maximum(b, 0.01 * b.max())
                    A = K[mask]
                    for smooth in (0.0, 0.1, 1.0):
                        q, rms, peak = retrieve(A, b, smooth, weight)
                        record(
                            w,
                            h,
                            "diffraction",
                            f"{name}_{coverage}_s{smooth}",
                            f,
                            q,
                            rms,
                            peak,
                            coverage,
                        )
                    # Independent-of-regularizer range at fixed per-ring error incl amplitude floor.
                    if (
                        w in (2, 6)
                        and (w == 2 or h == 3)
                        and (name == "n1.52_k0.002" or coverage == "full")
                    ):
                        for eps in (
                            [0.01, 0.05] if name == "n1.52_k0.002" else [0.05]
                        ):
                            ran = tolerance_bounds(A, b, weight, eps)
                            bounds.append(
                                dict(
                                    wind=w,
                                    hours=h,
                                    case=name,
                                    coverage=coverage,
                                    tolerance_pct=eps * 100,
                                    feasible=ran is not None,
                                    **(
                                        ran
                                        or dict.fromkeys(
                                            [
                                                "median_lo",
                                                "median_hi",
                                                "under20_lo",
                                                "under20_hi",
                                                "middle_lo",
                                                "middle_hi",
                                                "under20_over63_lo",
                                                "under20_over63_hi",
                                            ],
                                            "",
                                        )
                                    ),
                                )
                            )
            # Synthetic independent detector errors. Keep seed independent between endpoints.
            rng = np.random.default_rng(9205 + 100 * w + int(h * 10))
            weight = np.maximum(b0, 0.01 * b0.max())
            for i in range(64):
                noisy = np.maximum(
                    0, b0 + rng.normal(size=len(b0)) * 0.01 * weight
                )
                q, rms, peak = retrieve(reference, noisy, 0.1, weight)
                noise.append(
                    dict(
                        wind=w,
                        hours=h,
                        draw=i,
                        d50_um=med(q),
                        under20_pct=100 * frac(q, 20),
                        middle_pct=100 * (frac(q, 20) - frac(q, 10)),
                    )
                )
    fine = scattering(NOMINAL.real, NOMINAL.imag, 72)
    for w in WINDS:
        for f in ("early_pct", "measured_pct"):
            b0 = reference @ spectrum(w, f)
            bf = fine @ spectrum(w, f)
            tests.append(max(abs(bf - b0) / np.maximum(b0, 0.01 * b0.max())))
    if max(tests) > 0.005:
        raise ValueError(("Angular quadrature", max(tests)))
    if np.max(abs(geometries["AR1.0_random_offset0.0"] - I)) > 1e-10:
        raise ValueError("Sphere transit")
    save("optical-cases.csv", rows)
    save("optical-feasible.csv", bounds)
    save("optical-signals.csv", signal)
    save("optical-noise.csv", noise)
    np.savez_compressed(
        ROOT / "optical-bins.npz", spectra=np.array(bins), diameter=DIAM
    )
    return max(tests)


if __name__ == "__main__":
    print("Maximum angular refinement difference", run(), flush=True)
