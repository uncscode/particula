"""Direct activity predictions for filter-constrained molecular recipes, before observation joins."""

import argparse
import json
from math import gcd
import time
import numpy as np
import pandas as pd
import warp as wp
import particula
import particula_aiomfac
from particula_aiomfac import AIOMFACModel, MixtureSpec
from recipe import HERE, NAMES, MWA, CHARGE

NEUTRALS = {
    "water": {16: 1},
    "glycerol": {150: 2, 151: 1, 153: 3},
    "butanediol": {142: 2, 150: 2, 153: 2},
    "succinic": {2: 2, 137: 2},
    "levulinic": {18: 1, 2: 2, 137: 1},
    "hexadecanol": {145: 1, 146: 14, 150: 1, 153: 1},
    "vanillic": {9: 3, 10: 2, 17: 1, 24: 1, 137: 1},
}
CATIONS = {
    "NH4": (204, 1, 18.038),
    "Na": (202, 1, 22.99),
    "H": (205, 1, 1.008),
    "Mg": (223, 2, 24.305),
    "K": (203, 1, 39.098),
    "Ca": (221, 2, 40.078),
}
ANIONS = [245, 261, 242]
METRICS = [
    "gf85_30",
    "gf85_60",
    "gf85_90",
    "gf95_100",
    "dcrit04_nm",
    "dcrit08_nm",
    "dcrit04_nominal_nm",
    "dcrit08_nominal_nm",
    "gf76_60",
    "gf78_60",
    "gf80_60",
    "gf82_60",
    "gf84_60",
]
KEY = [
    "filter_id",
    "mask",
    "utc",
    "recipe",
    "closure",
    "model",
    "class_id",
    "number_fraction",
    "draw",
    "ss04_pct",
    "ss08_pct",
]


def ionic(closure):
    if closure == "sodium":
        return ["NH4", "Na"]
    if closure == "proton":
        return ["NH4", "H"]
    return ["NH4", "Na", "Mg", "K", "Ca", "H"]


def spec(closure):
    components = NEUTRALS.copy()
    for cat in ionic(closure):
        cid, z, m = CATIONS[cat]
        for a, za in zip(ANIONS, CHARGE):
            g = gcd(z, int(za))
            components[f"{cat}_{a}"] = {cid: int(za) // g, a: z // g}
    return MixtureSpec(components)


def salt_masses(r, closure):
    anion = r[["nitrate", "sulfate", "chloride"]].to_numpy(float) / MWA
    eq = anion @ CHARGE
    cat = {"NH4": r.ammonium.to_numpy() / 18.038}
    missing = eq - cat["NH4"]
    if np.any(missing < 0):
        raise ValueError("Unbalanced negative deficit")
    if closure in ("sodium", "proton"):
        cat["Na" if closure == "sodium" else "H"] = missing
    else:
        fraction = float(closure.split("_")[-1]) / 100
        xrf = pd.read_csv(HERE / "xrf.csv")
        for name in ("Na", "Mg", "K", "Ca"):
            cat[name] = np.zeros(len(r))
        for i, row in enumerate(r.itertuples()):
            match = xrf[(xrf.begin <= row.utc) & (xrf.end >= row.utc + 21600)]
            if len(match) != 1:
                raise ValueError("Unmatched XRF case")
            for name in ("Na", "Mg", "K", "Ca"):
                _, z, m = CATIONS[name]
                cat[name][i] = match[name].iloc[0] * row.cut_scale * fraction * z / m
        total = np.sum([cat[name] for name in ("Na", "Mg", "K", "Ca")], axis=0)
        if np.any(total > missing):
            raise ValueError("XRF oversupplies measured charge")
        cat["H"] = missing - total
    if np.max(abs(sum(cat.values()) - eq)) > 1e-13:
        raise ValueError("Charge conservation")
    salts = []
    for name in ionic(closure):
        _, z, m = CATIONS[name]
        paired = anion * (cat[name] / eq)[:, None]
        salts.extend((paired * (MWA + CHARGE * m / z)).T)
    return np.column_stack(salts)


def inventories(r, closure):
    salt = salt_masses(r, closure)
    organic = r[NAMES].to_numpy(float).copy()
    residual = r.organic_total.to_numpy() - organic.sum(1)
    if residual.min() < -1e-8:
        raise ValueError("Negative organic residual")
    if "residual_u" in r:
        soluble = np.where(
            r.recipe == "low", 0.25 * r.residual_u, 0.75 + 0.25 * r.residual_u
        )
        scale = 4 ** (2 * r.solubility_u.to_numpy() - 1)
    else:
        soluble = (r.recipe == "high").to_numpy(float)
        scale = np.ones(len(r))
    remaining = soluble * np.maximum(0, residual - r.amine_ug_m3.to_numpy())
    organic[:, 0] += remaining
    orgvol = r.organic_total.to_numpy() / 1.5
    dryvol = orgvol + salt.sum(1) / 1.76
    orgfrac = orgvol / dryvol
    norm_org = organic / dryvol[:, None]
    norm_salt = salt / dryvol[:, None]
    return np.c_[norm_org, np.zeros(len(r)), norm_salt], orgfrac, scale


def activity(model, m, q, T, limit, kind):
    size = len(m)
    inp = np.empty((size, len(q), m.shape[1] + 1))
    inp[:, :, 0] = 0.997 * q
    inp[:, :, 1:] = m[:, None, :]
    # Undissolved material stays in physical dry volume. Cap in g/g liquid water.
    inp[:, :, 3] = np.minimum(inp[:, :, 3], 0.075 * limit[:, None] * inp[:, :, 0])
    inp[:, :, 5] = np.minimum(inp[:, :, 5], 4.1e-8 * limit[:, None] * inp[:, :, 0])
    if kind == "org":
        inp[:, :, 7:] = 0
    if kind == "salt":
        inp[:, :, 1:7] = 0
    out = model.evaluate(inp, T[:, None])
    a = out.neutral_activities[:, :, 0]
    if not np.isfinite(a).all() or np.min(a) <= 0:
        raise ValueError("Invalid water activity")
    return a, out.neutral_mole_fractions[:, :, 0], inp


def inverse(q, act, u):
    # Follow the last crossing toward dilution; reject unstable/re-entrant portions in inversion.
    out = np.empty((len(act), len(u)))
    for i, row in enumerate(act):
        take = (row > 0) & (row < 1)
        row = row[take]
        qr = q[take]
        if len(np.unique(row)) < 2:
            out[i] = 0
            continue
        x = -np.log1p(-row)
        # A solubility-limited pool below its first deliquescent point contributes zero water.
        take = np.r_[x[:-1] < np.minimum.accumulate(x[::-1])[::-1][1:], True]
        out[i] = np.exp(
            np.interp(u, x[take], np.log(qr[take]), left=-np.inf, right=np.nan)
        )
        if np.any(~np.isfinite(out[i])):
            raise ValueError("Unbracketed water activity")
    return out


def predict(v, u, T, settings):
    a = -np.expm1(-u)
    gf = np.cbrt(1 + v)
    A = 4e9 * 0.072 * 0.01801528 / (8.314462618 * 997 * T)
    result = np.empty((len(v), len(METRICS)))
    for k, (d, rh) in zip(
        (0, 1, 2, 3, 8, 9, 10, 11, 12),
        (
            (30, 0.85),
            (60, 0.85),
            (90, 0.85),
            (100, 0.95),
            (60, 0.76),
            (60, 0.78),
            (60, 0.80),
            (60, 0.82),
            (60, 0.84),
        ),
    ):
        ss = np.log(a)[None, :] + A[:, None] / d / gf
        for i, row in enumerate(ss):
            ix = np.flatnonzero(row >= np.log(rh))
            if not len(ix) or ix[0] == 0:
                raise ValueError("Humidified root unbracketed")
            j = ix[0]
            res = np.interp(np.log(rh), row[j - 1 : j + 1], gf[i, j - 1 : j + 1])
            result[i, k] = res
    for k, ss in enumerate(
        (
            settings[:, 0],
            settings[:, 1],
            np.full(len(v), 0.4),
            np.full(len(v), 0.8),
        ),
        4,
    ):
        diameter = A[:, None] / (gf * (np.log1p(ss[:, None] / 100) - np.log(a)))
        ix = np.argmax(diameter, axis=1)
        if min(ix) == 0 or max(ix) == len(u) - 1:
            raise ValueError("Unbracketed critical peak")
        result[:, k] = diameter[np.arange(len(v)), ix]
    return result


def evaluate(r, model, closure, cls, grid, all_controls):
    q = np.geomspace(0.15 if closure != "sodium" else 0.005, 1e5, grid)
    qo = np.geomspace(0.005, 1e5, grid)
    u = np.linspace(-np.log(1 - 0.55), -np.log(1 - 0.99993), 2 * grid + 1)
    m, f, limit = inventories(r, closure)
    pop = np.ones(len(r))
    if cls != "uniform":
        rich = (1 + f) / 2
        frac = rich if cls == "org_rich" else f / 2
        m[:, :6] *= (frac / f)[:, None]
        m[:, 6:] *= ((1 - frac) / (1 - f))[:, None]
        pop = f if cls == "org_rich" else 1 - f
        if max(abs(f * rich + (1 - f) * f / 2 - f)) > 1e-14:
            raise ValueError("Class conservation")
    temp = r.temperature_K.to_numpy()
    settings = r[["ss04_pct", "ss08_pct"]].to_numpy()
    a, ideal, _ = activity(model, m, q, temp, limit, "full")
    cases = {"full": inverse(q, a, u)}
    if all_controls:
        cases["ideal"] = inverse(q, ideal, u)
        org, _, _ = activity(model, m, qo, temp, limit, "org")
        salts, _, _ = activity(model, m, q, temp, limit, "salt")
        cases["two_pool"] = inverse(qo, org, u) + inverse(q, salts, u)
    frame = []
    for label, curve in cases.items():
        value = predict(curve, u, temp, settings)
        frame.append(
            pd.DataFrame(
                dict(
                    filter_id=r.filter_id.to_numpy(),
                    mask=r["mask"].to_numpy(),
                    utc=r.utc.to_numpy(),
                    recipe=r.recipe.to_numpy(),
                    closure=closure,
                    model=label,
                    class_id=cls,
                    number_fraction=pop,
                    draw=r.draw.to_numpy(),
                    ss04_pct=settings[:, 0],
                    ss08_pct=settings[:, 1],
                    **dict(zip(METRICS, value.T)),
                )
            )
        )
    del cases
    return frame


def run(stage, device):
    started = time.monotonic()
    r = pd.read_csv(HERE / "recipes.csv")
    r["draw"] = 0
    if stage == "pilot":
        r = r[(r.filter_id == "EPP538")].iloc[[0, 3]].copy()
    grid = {"pilot": 97, "central": 257, "refine": 513, "uncertainty": 129}[stage]
    if stage == "uncertainty":
        draws = pd.read_csv(HERE / "draws.csv.gz")
        r = r.drop(columns=["draw", *NAMES]).merge(
            draws, on=["filter_id", "mask", "recipe"]
        )
        for n in NAMES:
            r[n] *= r.organic_total
    frames = []
    rows = 0
    for closure in (
        ["sodium", "proton", "metal_50", "metal_100"]
        if stage in ("central", "refine")
        else ["sodium"]
    ):
        print(stage, closure, flush=True)
        eligible = r.copy()
        if closure.startswith("metal"):
            eligible = eligible[eligible.start == 1686362400]
        model = AIOMFACModel.prepare(
            spec(closure),
            **({"execution": "cuda", "device": "cuda:0"} if device == "cuda" else {}),
        )
        classes = (
            ["uniform", "org_rich", "salt_rich"]
            if closure == "sodium" and stage in ("central", "refine")
            else ["uniform"]
        )
        for cls in classes:
            for i in range(0, len(eligible), 24):
                batch = eligible.iloc[i : i + 24]
                controls = cls == "uniform"
                frames.extend(evaluate(batch, model, closure, cls, grid, controls))
                rows += len(batch) * grid * (3 if controls else 1)
    final = pd.concat(frames, ignore_index=True)
    name = "predictions" if stage == "central" else stage + "-" + device
    if stage == "uncertainty":
        final = final.drop(
            columns=[
                "gf76_60",
                "gf78_60",
                "gf80_60",
                "gf82_60",
                "gf84_60",
                "dcrit04_nominal_nm",
                "dcrit08_nominal_nm",
            ]
        )
    final.to_csv(
        HERE / (name + ".csv.gz"),
        index=False,
        float_format="%.8g",
        compression={"method": "gzip", "mtime": 0},
    )
    record = dict(
        stage=stage,
        device=wp.get_device("cuda:0").name if device == "cuda" else "CPU",
        gpu=device == "cuda",
        rows=rows,
        n_predictions=len(final),
        grid=grid,
        elapsed_s=time.monotonic() - started,
        versions=[
            particula.__version__,
            particula_aiomfac.__version__,
            wp.__version__,
        ],
        model_revision=particula_aiomfac.UPSTREAM_REVISION,
        estimated_pairs=AIOMFACModel.prepare(spec("sodium")).estimated_pairs,
    )
    (HERE / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(record)
    print(final.head())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pilot", "central", "refine", "uncertainty"])
    p.add_argument("device", choices=["cpu", "cuda"])
    args = p.parse_args()
    run(args.stage, args.device)
