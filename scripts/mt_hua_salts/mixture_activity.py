"""Apply archived AIOMFAC-web water activities to the Mt. Hua extracts."""

import csv
import gzip
import json
import re

import numpy as np
import warp as wp
from particula.gpu.properties.particle_properties import kelvin_term_wp
from scipy.optimize import brentq
from scripts.mt_hua_salts.study import (
    A_RADIUS,
    ANION_EQ,
    CATION_EQ,
    DENSITY,
    DRY_RADIUS,
    KAPPA,
    RH,
    WATER,
    cpu_growth,
    pairings,
)
from scripts.mt_hua_salts.web_inputs import RATIOS, ROOT, compositions


def archived():
    """Read and check official water fractions, temperature and flags."""
    results = {}
    for path in sorted((ROOT / "aiomfac").glob("batch-*.json.gz")):
        data = json.loads(gzip.decompress(path.read_bytes()))
        output = data["output"]
        if "AIOMFAC-web, version 3.15" not in output:
            raise ValueError("Unexpected AIOMFAC version")
        content = output.split("Component's name      : 'Water'")[1].split(
            "Mixture's"
        )[0]
        rows = np.array(
            [
                list(map(float, line.split()))
                for line in content.splitlines()
                if re.match(r"^\s*\d{3}\s+298", line)
            ]
        )
        groups = rows.reshape(len(data["mixture_ids"]), len(RATIOS), 9)
        for key, arr in zip(data["mixture_ids"], groups, strict=True):
            np.testing.assert_allclose(arr[:, 1], 298.15)
            np.testing.assert_allclose(
                arr[:, 3], RATIOS / (1 + RATIOS), atol=6e-7
            )
            np.testing.assert_allclose(
                arr[:, 7], arr[:, 2] / 100, atol=0.000051
            )
            if key in results:
                raise ValueError("Duplicate mixture ID")
            results[key] = arr
    return results


@wp.kernel
def root_kernel(
    limits: wp.array2d(dtype=wp.float64),
    rho: wp.array(dtype=wp.float64),
    water: wp.float64,
    dry: wp.float64,
    kelvin: wp.float64,
    rh: wp.float64,
    out: wp.array(dtype=wp.float64),
):
    """Solve the Kelvin balance with log-linear activity interpolation."""
    i = wp.tid()
    left = limits[i, 0]
    right = limits[i, 1]
    for _ in range(50):
        mid = (left + right) * wp.float64(0.5)
        aw = limits[i, 2] + (mid - limits[i, 0]) / (
            limits[i, 1] - limits[i, 0]
        ) * (limits[i, 3] - limits[i, 2])
        g = wp.pow(
            wp.float64(1.0) + wp.exp(mid) * rho[i] / water,
            wp.float64(1.0) / wp.float64(3.0),
        )
        if aw * kelvin_term_wp(dry * g, kelvin) > rh:
            right = mid
        else:
            left = mid
    out[i] = wp.pow(
        wp.float64(1.0)
        + wp.exp((left + right) * wp.float64(0.5)) * rho[i] / water,
        wp.float64(1.0) / wp.float64(3.0),
    )


def bracket(table, rho, stride):
    """Identify a unique stable crossing among adjacent unflagged points."""
    r = RATIOS[::stride]
    aw = table[::stride, 7]
    flag = table[::stride, 8]
    g = np.cbrt(1 + r * rho / WATER)
    sat = aw * np.exp(A_RADIUS / (DRY_RADIUS * g))
    picks = np.flatnonzero(
        (sat[:-1] < RH) & (sat[1:] >= RH) & (flag[:-1] == 0) & (flag[1:] == 0)
    )
    if len(picks) != 1:
        return None
    i = picks.item()
    return [np.log(r[i]), np.log(r[i + 1]), aw[i], aw[i + 1]]


def pure_reference(rows):
    """Apply Equation 5 to the independently computed pure web endpoints."""
    pure = {r["key"]: r["growth"] for r in rows if r["key"].startswith("pure")}
    for r in rows:
        if "observed" in r:
            terms = [
                (w, pure.get(f"pure_{i}"))
                for i, w in enumerate(r["dry_volume_fraction"])
                if w > 0
            ]
            if all(g is not None for _, g in terms):
                r["zsr_web_pure"] = np.cbrt(sum(w * g**3 for w, g in terms))
                r["mixing_effect"] = r["growth"] - r["zsr_web_pure"]


def simulate():
    """Pair tabular liquid growth with the original neutral-salt envelope."""
    tables = archived()
    with (ROOT / "matched.csv").open() as f:
        samples = list(csv.DictReader(f))
    control = (ROOT / "aiomfac/nacl-control.txt").read_text()
    if "8.91330E-01" not in control:
        raise ValueError("Unexpected NaCl control")
    predicted = np.interp(
        np.log(85 / 15), np.log(RATIOS), tables["pure_5"][:, 7]
    )
    np.testing.assert_allclose(predicted, 0.891330, atol=0.001)
    bounds = []
    densities = []
    rows = []
    skipped = []
    pure_g = np.array([cpu_growth(k) for k in KAPPA.ravel()])
    for key, mass in compositions():
        rho = 1 / (mass / DENSITY.ravel()).sum()
        table = tables[key]
        b = bracket(table, rho, 1)
        training = bracket(table, rho, 2)
        if b is None or training is None:
            skipped.append(key)
            continue

        def saturation(logr, b=b, rho=rho):
            aw = np.interp(logr, b[:2], b[2:])
            return (
                aw
                * np.exp(
                    A_RADIUS
                    / (DRY_RADIUS * np.cbrt(1 + np.exp(logr) * rho / WATER))
                )
                - RH
            )

        cpu = np.cbrt(1 + np.exp(brentq(saturation, *b[:2])) * rho / WATER)
        v = mass / DENSITY.ravel()
        v /= v.sum()
        row = dict(
            key=key,
            cpu=cpu,
            zsr=float(np.cbrt(v @ pure_g**3)),
            density=rho,
            dry_volume_fraction=v.tolist(),
            flags=np.unique(table[:, 8]).tolist(),
            aw=table[:, 7].tolist(),
        )
        if not key.startswith("pure"):
            sample = samples[int(key)]
            c = (
                np.array(
                    [float(sample[k]) for k in ("nh4", "na", "k", "ca", "mg")]
                )
                / CATION_EQ
            )
            a = (
                np.array([float(sample[k]) for k in ("so4", "no3", "cl")])
                / ANION_EQ
            )
            fraction = pairings(c, a)[-2]
            row.update(
                site=sample["site"],
                date=sample["sample_start_local"],
                observed=float(sample["growth_inorganic_90"]),
                minimum_zsr_kappa=cpu_growth(np.sum(fraction * KAPPA)),
            )
        rows.append(row)
        bounds.extend([b, training])
        densities.extend([rho, rho])

    def device(a):
        return wp.array(np.asarray(a), dtype=wp.float64, device="cuda:0")

    out = wp.zeros(len(bounds), dtype=wp.float64, device="cuda:0")
    wp.launch(
        root_kernel,
        len(bounds),
        [
            device(bounds),
            device(densities),
            WATER,
            DRY_RADIUS,
            A_RADIUS,
            RH,
            out,
        ],
        device="cuda:0",
    )
    for r, (fine, coarse) in zip(rows, out.numpy().reshape(-1, 2), strict=True):
        np.testing.assert_allclose(fine, r["cpu"], rtol=1e-10)
        r.update(growth=fine, sparse_grid=coarse)
        if "observed" in r:
            r["season"] = (
                "summer" if r["date"][5:7] in ("07", "08") else "winter"
            )
            r["zsr_residual"] = r["zsr"] - r["observed"]
            r["activity_residual"] = fine - r["observed"]
    pure_reference(rows)
    return dict(rows=rows, skipped=skipped, water_dry=RATIOS.tolist())
