"""Tabulate ternary dry-particle activation under an aqueous/pure-oil reservoir approximation."""

import json
import time

import numpy as np
import warp as wp
from observations import SET, P
from particula_aiomfac import AIOMFACModel, MixtureSpec
from scipy.constants import R

KELVIN_NM = 4e9 * 0.072 * 0.01801528 / (997 * R * 298.15)
DENS = np.array([1.506, 0.8054])
MOLAR = np.array([182.172, 422.82])
CARBON = DENS * np.array([6, 30]) * 12.011 / MOLAR
GROUPS = {
    "water": {16: 1},
    "sorbitol": {150: 2, 151: 4, 153: 6},
    "squalane": {1: 8, 2: 16, 3: 6},
    "ammonium_sulfate": {204: 2, 261: 1},
    "sodium_chloride": {202: 1, 242: 1},
}


def prepare(backend, groups=GROUPS):
    return AIOMFACModel.prepare(
        MixtureSpec(groups),
        execution=backend,
        **({"device": "cuda:0"} if backend == "cuda" else {}),
    )


def grid(n):
    return np.array(
        [
            (i / n, j / n, (n - i - j) / n)
            for i in range(n + 1)
            for j in range(n + 1 - i)
        ]
    )


def thresholds(model, comps, r, nw, dens=DENS):
    w = np.geomspace(0.5, 4e5, nw)
    shape = (len(comps), len(w))
    mass = np.zeros((*shape, 5))
    mass[:, :, 0] = 0.997 * w
    salt, sugar, oil = comps.T
    mass[:, :, 1] = (sugar * dens[0])[:, None]
    mass[:, :, 3] = (salt * (1 - r) * 1.77)[:, None]
    mass[:, :, 4] = (salt * r * 2.165)[:, None]
    total = (oil * dens[1])[:, None]
    # Equilibrate aqueous oil activity against a pure oil reservoir (unit activity).
    mass[:, :, 2] = mass[:, :, 0] * 1e-12

    def evaluate():
        a = model.evaluate(mass.reshape(-1, 5), 298.15)
        return a.neutral_activities.reshape(*shape, -1), a.neutral_ids

    a, ids = evaluate()
    oi = ids.index("squalane")
    wi = ids.index("water")
    for _ in range(3):
        mass[:, :, 2] = np.minimum(
            total, mass[:, :, 2] / np.maximum(a[:, :, oi], 1e-200)
        )
        a, _ = evaluate()
    aw = a[:, :, wi]
    assert np.isfinite(aw).all() and np.all(aw > 0)
    # Dry oil volume remains in the total wet diameter. Pure oil uses a wettable insoluble limit.
    den = np.log1p(SET / 100) - np.log(aw[:, :, None])
    eligible = den > 0
    curve = np.divide(
        KELVIN_NM,
        np.cbrt(1 + w)[None, :, None] * den,
        out=np.zeros_like(den),
        where=eligible,
    )
    pure = oil == 1
    ix = np.argmax(curve, axis=1)
    assert np.all(eligible[~pure]) and np.all(
        (ix[~pure] > 0) & (ix[~pure] < nw - 1)
    )
    diam = curve.max(axis=1)
    diam[pure] = KELVIN_NM / np.log1p(SET / 100)
    row = np.arange(len(comps))[:, None]
    active = (mass[row, ix, 2] < 0.999 * total) & (~pure[:, None])
    oil_err = float(np.max(abs(a[row, ix, oi][active] - 1), initial=0))
    assert oil_err < 1e-4, oil_err
    return diam, oil_err


def main():
    start = time.monotonic()
    gpu = prepare("cuda")
    cpu = prepare("cpu")
    test = grid(2)
    pilot = np.max(
        abs(
            thresholds(cpu, test, 0.3, 193)[0]
            - thresholds(gpu, test, 0.3, 193)[0]
        )
    )
    assert pilot < 1e-5
    comps = grid(8)
    rs = np.linspace(0, 1, 17)
    vals = [thresholds(gpu, comps, r, 385) for r in rs]
    extra = thresholds(gpu, comps, 0.5, 769)[0]
    refine = np.max(abs(extra - vals[8][0]))
    np.savez_compressed(
        P / "thresholds.npz",
        comps=comps,
        r=rs,
        d=np.array([v[0] for v in vals]),
    )
    info = dict(
        device=wp.get_device("cuda:0").name,
        groups=GROUPS,
        pilot_max_nm=float(pilot),
        water_refinement_nm=float(refine),
        max_oil_activity_error=max(v[1] for v in vals),
        gpu_activity_evaluations=4 * (6 * 193 + 17 * 45 * 385 + 45 * 769),
        wall_s=time.monotonic() - start,
    )
    (P / "thermodynamics.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
