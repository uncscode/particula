"""Shared-parameter fits and blocked transfer tests for the Jia tray study."""

from functools import lru_cache
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.constants import epsilon_0, elementary_charge
from scipy.optimize import minimize
from scipy.stats import qmc

# A replay may point to an existing compatible PyTorch installation.
if os.environ.get("JIA_TORCH_PATH"):
    sys.path.append(os.environ["JIA_TORCH_PATH"])
import torch
from rebound import (
    AREA,
    WORKBOOK_AREA,
    inputs,
    read_workbook,
    table,
    trace,
    velocities,
    DENSITY,
)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
T = torch.as_tensor
ROOT = Path(__file__).parent
WINDS = [0, 2, 4, 6]
HOURS = np.array([0.5, 1, 1.5, 2, 3])
CUT = [2.5, 10, 20, 30, 40, 63]
# (low, high, use logarithmic coordinate, start). Allowed ranges are scenarios.
PAR = {
    "local_ratio": (0.02, 15.0, True, 1.0),
    "normal_sd_over_ustar": (0.05, 3.0, True, 0.7),
    "capture10_ms": (0.005, 1.0, True, 0.08),
    "late_threshold_ratio": (1.0, 12.0, True, 2.0),
    "adhesion_time_min": (2.0, 180.0, True, 30.0),
    "delivery_lag_min": (0.2, 180.0, True, 12.0),
    "settling_depletion": (0.0, 1.0, False, 0.03),
    "removal6_per_h": (0.0, 4.0, False, 0.3),
    "air_size_exponent": (0.0, 1.0, False, 0.5),
    "air_median_um": (1.0, 60.0, True, 14.0),
    "air_log_sd": (0.25, 2.0, True, 0.9),
    "coarse_median_um": (20.0, 60.0, True, 34.0),
    "coarse_log_sd": (0.15, 1.0, True, 0.35),
    "mix_intercept": (-8.0, 8.0, False, 0.0),
    "mix_linear": (-10.0, 10.0, False, 0.0),
    "mix_curvature": (-16.0, 16.0, False, 4.0),
    "charge_uC_m2": (0.0, 10.0, False, 2.0),
    "power_gain": (0.0, 3.0, False, 0.6),
    "power_time": (0.1, 3.0, True, 1.0),
}
COMMON = list(PAR)[:8]
NAMES = {
    "blind": COMMON + ["air_median_um", "air_log_sd"],
    "blind_mixture": COMMON
    + [
        "air_median_um",
        "air_log_sd",
        "coarse_median_um",
        "coarse_log_sd",
        "mix_intercept",
        "mix_linear",
        "mix_curvature",
    ],
    "gravity": [],
    "conditional": COMMON + ["air_size_exponent"],
    "hybrid": COMMON + ["air_size_exponent", "power_gain", "power_time"],
    "power": ["power_gain", "power_time"],
    "persist": [],
}


def write(name, rows):
    with (ROOT / name).open("w") as out:
        w = csv.DictWriter(out, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def data(minutes):
    diam = np.array(
        [
            float(r["channel_um"])
            for r in table("channels.csv")
            if int(r["wind_ms"]) == 0
        ]
    )
    # Each row maps a channel mass vector to one exact cumulative threshold.
    cutmap = np.array(
        [
            [
                np.interp(
                    np.log(c), np.log(diam), np.cumsum(np.eye(len(diam))[i])
                )
                for i in range(len(diam))
            ]
            for c in CUT
        ]
    )
    kinetic, masses = read_workbook()[2], read_workbook()[1]
    specs = []
    mass = []
    sem = []
    air = []
    friction = []
    c0 = []
    for w in WINDS:
        _, early, late, u, *_ = inputs(w)
        specs.append([early[: len(diam)], late[: len(diam)]])
        ke = {
            (r["D"], r["E"]): r
            for r in kinetic.values()
            if r.get("C") == w and "F" in r
        }
        mm = []
        ss = []
        for h in HOURS:
            if h in (0.5, 3):
                a = [np.array(masses[w, h, c]) for c in CUT]
                mm.append([x.mean() for x in a])
                ss.append([x.std(ddof=1) / np.sqrt(len(x)) for x in a])
            else:
                mm.append([1000 * ke[c, h]["F"] for c in CUT])
                ss.append([WORKBOOK_AREA * ke[c, h]["H"] for c in CUT])
        mass.append(mm)
        sem.append(ss)
        friction.append(u)
        sec, conc = trace(w)
        edges = np.arange(181) * 60
        series = np.interp(edges, sec, conc)
        air.append(
            ((series[:-1] + series[1:]) / 2).reshape(-1, minutes).mean(axis=1)
        )
        c0.append(series[0])
    dic = {
        k: T(np.asarray(v, dtype=float))
        for k, v in dict(
            d=diam,
            vs=velocities(diam, DENSITY),
            cutmap=cutmap,
            shape=specs,
            mass=mass,
            sem=sem,
            air=air,
            u=friction,
            c0=c0,
        ).items()
    }
    # Zhang et al. (2014), Appendix B, smooth wood at a 15 mm reference height.
    zdiam = np.array([7.5, 12.5, 17.5, 22.5, 27.5, 40.0])
    low = np.array([3.52, 13.86, 24.13, 32.79, 53.35, 108.70]) * 0.001
    high = np.array([32.35, 33.24, 39.00, 74.90, 63.52, 163.72]) * 0.001
    low_sd = np.array([3.89, 3.96, 2.11, 2.01, 0.30, 0.24]) * 0.001
    high_sd = np.array([11.45, 6.04, 5.37, 14.32, 5.42, 49.74]) * 0.001
    mapped = np.interp(
        WINDS,
        [0, 2.30, 3.48, 5.49, 7.31, 8.08],
        [0, 0.12, 0.14, 0.28, 0.34, 0.57],
    )
    for name, delta in [("zhang", 0), ("zhang_low", -1), ("zhang_high", 1)]:
        lo = np.maximum(low + delta * low_sd, 0.00001)
        hi = np.maximum(high + delta * high_sd, 0.00001)
        # Extrapolate between gravity at zero wind and first measured condition.
        ratio = np.array(
            [
                np.interp(x, [0, 0.12, 0.40], [1, l / v, h / v])
                for x in mapped
                for l, h, v in zip(lo, hi, velocities(zdiam, 2200), strict=True)
            ]
        ).reshape(4, -1)
        dic[name] = (
            T(
                np.array(
                    [np.interp(np.log(diam), np.log(zdiam), r) for r in ratio]
                )
            )
            * dic["vs"]
        )
    dic["minutes"] = minutes
    return dic


def decode(x, names):
    ans = {}
    for i, n in enumerate(names):
        lo, hi, log, _ = PAR[n]
        ans[n] = (
            lo * torch.exp(x[i] * np.log(hi / lo))
            if log
            else lo + (hi - lo) * x[i]
        )
    return ans


@lru_cache
def image_speed_squared(pool):
    """Image-energy contribution per squared unit of contact charge scale."""
    from tribocharging import source

    result = []
    for w in WINDS:
        d, m, _, _, z, _ = source(w, 12, 1.0, pool)
        radius = d[:41] * 0.5e-6
        work = (
            0.5
            * (z[:41] * elementary_charge) ** 2
            / (16 * np.pi * epsilon_0 * radius)
        )
        result.append(2 * work / m[:41] * (1 / 0.5**2 - 1))
    return T(np.array(result))


def predict(x, names, model, z, settings):
    """Return dry mg per laser channel at every tray retrieval."""
    par = decode(x, names)
    par.update({k: T(v) for k, v in settings.get("fixed", {}).items()})
    d, v, air, u = (z[n] for n in ["d", "vs", "air", "u"])
    obs0 = z["shape"][:, 0, :] * z["mass"][:, 0, -1, None]
    if model == "gravity":
        deposited = -torch.expm1(-T(HOURS)[:, None] * 3600 * v / 2)
        return obs0[:, None, :] * (deposited / deposited[0])[None, :, :]
    if model in ("persist", "power"):
        factor = (
            1
            + par["power_gain"] * (T((HOURS - 0.5) / 2.5) ** par["power_time"])
            if model == "power"
            else torch.ones(5)
        )
        return obs0[:, None, :] * factor[None, :, None]
    if settings.get("external_shear"):
        u = T(
            np.interp(
                WINDS,
                [0, 2.30, 3.48, 5.49, 7.31, 8.08],
                [0, 0.12, 0.14, 0.28, 0.34, 0.57],
            )
        )
    dt = z["minutes"]
    n = air.shape[1]
    tm = (torch.arange(n) + 0.5) * dt
    late = (tm - 30).clamp(min=0)
    if settings.get("instant_air"):
        local = air
    else:
        q = dt / par["delivery_lag_min"]
        ix = torch.arange(n)
        dist = ix[:, None] - ix[None, :] - 1
        kernel = (
            torch.exp(-q * dist.clamp(min=0)) * (dist >= 0) * (-torch.expm1(-q))
        )
        start = z["c0"][:, None] * torch.exp(-q * ix)[None, :] + air @ kernel.T
        local = air + (start - air) * (-torch.expm1(-q)) / q
    if model.startswith("blind"):
        raw = (
            -0.5
            * (torch.log(d / par["air_median_um"]) / par["air_log_sd"]) ** 2
        )
        base = raw.expand(len(WINDS), -1)
        if model == "blind_mixture":
            ws = T(np.array(WINDS) / 3.0 - 1)
            eta = (
                par["mix_intercept"]
                + par["mix_linear"] * ws
                + par["mix_curvature"] * ws**2
            )
            coarse = (
                -0.5
                * (
                    torch.log(d / par["coarse_median_um"])
                    / par["coarse_log_sd"]
                )
                ** 2
            )
            base = torch.logaddexp(
                torch.log_softmax(raw, 0)[None, :]
                + torch.nn.functional.logsigmoid(-eta)[:, None],
                torch.log_softmax(coarse, 0)[None, :]
                + torch.nn.functional.logsigmoid(eta)[:, None],
            )
    else:
        base = torch.log(z["shape"][:, 0, :].clamp(min=1e-100)) - par[
            "air_size_exponent"
        ] * torch.log(v[None, :] + 0.001)
    # Settling-driven size sorting is an effective relative depletion of coarse grains.
    age = tm if model.startswith("blind") else late
    shape = torch.softmax(
        base[:, None, :]
        - par["settling_depletion"]
        * age[None, :, None]
        * 60
        * v[None, None, :]
        / 2,
        dim=-1,
    )
    vd = (
        par["capture10_ms"]
        * (
            1
            + (par["late_threshold_ratio"] - 1)
            * (-torch.expm1(-late / par["adhesion_time_min"]))
        )[:, None]
        * (d / 10) ** (-settings.get("capture_exp", 0.5))
    )
    if settings.get("charge_pool"):
        vd = torch.sqrt(
            vd[None, :, :] ** 2
            + par["charge_uC_m2"] ** 2
            * image_speed_squared(settings["charge_pool"])[:, None, :]
        )
    # Rayleigh-distributed critical speeds imply sticking probability exp(-w²/(2 vc²)).
    sd = (par["normal_sd_over_ustar"] * u)[:, None, None]
    s2 = sd**2 + vd**2
    mu = v * vd**2 / s2
    sig = sd * vd / torch.sqrt(s2)
    nd = mu / sig.clamp(min=1e-12)
    flux = (
        vd
        / torch.sqrt(s2)
        * torch.exp(-(v**2) / (2 * s2))
        * (
            mu * torch.special.ndtr(nd)
            + sig * torch.exp(-0.5 * nd**2) / np.sqrt(2 * np.pi)
        )
    )
    if settings.get("zhang"):
        normal = v[None, None, :] / sd.clamp(min=1e-12)
        gross = v * torch.special.ndtr(normal) + sd * torch.exp(
            -0.5 * normal**2
        ) / np.sqrt(2 * np.pi)
        in_range = (d >= 7.5) & (d <= 40.0)
        flux = torch.where(
            in_range, flux / gross * z[settings["zhang"]][:, None, :], flux
        )
    dep = AREA * par["local_ratio"] * local[:, :, None] * shape * flux * dt * 60
    # Source only after initialization for the early-conditioned variant.
    if model in ("conditional", "hybrid"):
        dep = dep * (tm >= 30)[None, :, None]
    # A steep shared wind response approximates onset near the mobilization screen.
    mobilization = (
        T(np.array(WINDS) / 6.0) ** 6
        if not settings.get("hard_mobilization")
        else T([0.0, 0.0, 0.0, 1.0])
    )
    rate = par["removal6_per_h"] * mobilization[:, None] * (d / 10)[None, :]
    step = rate * dt / 60
    survival = -torch.expm1(-step.clamp(min=1e-12)) / step.clamp(min=1e-12)
    end = (tm + dt / 2) / 60
    elapsed = T(HOURS)[:, None] - end[None, :]
    integ = (
        torch.exp(
            -rate[:, None, None, :] * elapsed.clamp(min=0)[None, :, :, None]
        )
        * (elapsed >= -1e-10)[None, :, :, None]
    )
    result = torch.sum(
        dep[:, None, :, :] * survival[:, None, None, :] * integ, dim=2
    )
    if model in ("conditional", "hybrid"):
        result = result + obs0[:, None, :] * torch.exp(
            -rate[:, None, :] * T(HOURS - 0.5)[None, :, None]
        )
    if model == "hybrid":
        # Empirical total inventory, with normalized size composition from the
        # mechanistic model. Absolute arrival rates lose their interpretation.
        total = z["mass"][:, 0, -1, None] * (
            1 + par["power_gain"] * T((HOURS - 0.5) / 2.5) ** par["power_time"]
        )
        result = result * (total / (result @ z["cutmap"][-1]))[:, :, None]
    return result


def masks(model, fold):
    eligible = np.ones((4, 5), bool)
    if not model.startswith("blind"):
        eligible[:, 0] = False
    train = eligible.copy()
    if fold.startswith("corner"):
        idx = WINDS.index(int(fold[6:]))
        train[idx] = False
        train[:, HOURS > 1.5] = False
        test = np.zeros_like(train)
        test[idx, HOURS > 1.5] = True
        return train, test
    if fold.startswith("wind"):
        train[WINDS.index(int(fold[4:]))] = False
    elif fold.startswith("time"):
        train[:, HOURS == float(fold[4:])] = False
    elif fold == "forward":
        train[:, HOURS > 1.5] = False
    return train, eligible & ~train


def residual(y, z, mask, settings):
    pred = y @ z["cutmap"].T
    massmask = mask.copy()
    if settings.get("omit6mass"):
        massmask[3, 4] = False
    err = (
        (pred - z["mass"])
        / torch.clamp(z["sem"], min=settings.get("sem_floor", 0.75))
    )[massmask]
    out = [err.flatten() / np.sqrt(err.numel())]
    specmask = mask[:, [0, 4]]
    if settings.get("omit6shape"):
        specmask[3, 1] = False
    if specmask.any() and settings.get("shape_weight", 1) > 0:
        s = y[:, [0, 4], :]
        s = s / s.sum(dim=-1, keepdim=True)
        h = (torch.sqrt(s + 1e-12) - torch.sqrt(z["shape"] + 1e-12))[specmask]
        out.append(
            h.flatten()
            * np.sqrt(
                settings.get("shape_weight", 1) / (2 * sum(specmask.flatten()))
            )
            / 0.10
        )
    return torch.cat(out)


def scores(y, z, mask):
    pp = y @ z["cutmap"].numpy().T
    obs = z["mass"].numpy()
    r = pp - obs
    ans = dict(
        cells=int(mask.sum()),
        mass_mae_mg=np.mean(abs(r[mask])),
        total_mae_mg=np.mean(abs(r[:, :, 5][mask])),
        middle_mae_mg=np.mean(abs((r[:, :, 2] - r[:, :, 1])[mask])),
    )
    tv = []
    dm = []
    for w, t in zip(*np.where(mask), strict=True):
        if t in (0, 4):
            d = z["d"].numpy()
            sp = y[w, t] / sum(y[w, t])
            measured = z["shape"][w, int(t == 4)].numpy()
            med = lambda f: np.exp(
                np.interp(0.5, np.cumsum(f) / sum(f), np.log(d))
            )
            tv.append(50 * sum(abs(sp - measured)))
            dm.append(abs(med(sp) - med(measured)))
    ans.update(
        tv_pp=np.mean(tv) if tv else "", median_mae_um=np.mean(dm) if dm else ""
    )
    return ans


def start_point(names):
    vals = []
    for n in names:
        lo, hi, log, x = PAR[n]
        vals.append(
            np.log(x / lo) / np.log(hi / lo) if log else (x - lo) / (hi - lo)
        )
    return np.array(vals)


def alternate_data(z, settings):
    if not settings.get("kinetic"):
        return z
    z = dict(z)
    z["mass"] = z["mass"].clone()
    z["sem"] = z["sem"].clone()
    for r in read_workbook()[2].values():
        if "F" in r and r["E"] in (0.5, 3.0):
            idx = (
                WINDS.index(r["C"]),
                int(np.where(HOURS == r["E"])[0][0]),
                CUT.index(r["D"]),
            )
            z["mass"][idx] = 1000 * r["F"]
            z["sem"][idx] = WORKBOOK_AREA * r["H"]
    return z


def fit(job):
    key, model, fold, settings, seeds = job
    z = alternate_data(DATA, settings)
    tr, te = masks(model, fold)
    names = [n for n in NAMES[model] if n not in settings.get("fixed", {})]

    if settings.get("charge_pool") and "charge_uC_m2" not in settings.get(
        "fixed", {}
    ):
        names.append("charge_uC_m2")

    def evaluate(x):
        v = T(x).requires_grad_(True)
        r = residual(predict(v, names, model, z, settings), z, tr, settings)
        loss = r @ r
        gradient = torch.autograd.grad(loss, v)[0]
        return loss.item(), gradient.detach().numpy()

    starts = [start_point(names)]
    if names and seeds > 1:
        starts += list(
            qmc.LatinHypercube(len(names), seed=547).random(seeds - 1)
        )
    if "charge_uC_m2" in names:
        # The neutral model is nested. Seed from its optimum for this same
        # training split, including a nonzero seed away from the q² boundary.
        neutral = next(
            r
            for r in json.loads((ROOT / "global-core.json").read_text())[
                "results"
            ]
            if r["case"] == "conditional" and r["fold"] == fold
        )
        for q0 in (0.0, 1.0, 5.0):
            physical = neutral["params"] | {"charge_uC_m2": q0}
            initial = []
            for n in names:
                lo, hi, log, _ = PAR[n]
                val = physical[n]
                initial.append(
                    np.log(val / lo) / np.log(hi / lo)
                    if log
                    else (val - lo) / (hi - lo)
                )
            starts.append(np.clip(initial, 0, 1))
    screen_size = 0
    if len(names) > 2:
        # Sample the entire scaled box before local AD minimizations. Keep the
        # best separated seeds as well as the original space-filling starts.
        pool = qmc.Sobol(len(names), scramble=True, seed=173).random_base2(
            8 if fold == "all" else 6
        )
        screen_size = len(pool)
        with torch.no_grad():

            def loss_at(point):
                r = residual(
                    predict(T(point), names, model, z, settings),
                    z,
                    tr,
                    settings,
                )
                return float(r @ r)

            order = np.argsort([loss_at(point) for point in pool])
        ranked = []
        for j in order:
            if all(np.linalg.norm(pool[j] - prev) > 0.5 for prev in ranked):
                ranked.append(pool[j])
            if len(ranked) == (4 if fold == "all" else 2):
                break
        starts += ranked
    trials = []
    for i, x in enumerate(starts):
        if names:
            rr = minimize(
                evaluate,
                x,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * len(x),
                options=dict(maxiter=160, ftol=1e-9, gtol=3e-5, maxls=30),
            )
            objective, point, status = rr.fun, rr.x, rr.message
        else:
            point = np.array([])
            r = residual(
                predict(T(point), names, model, z, settings), z, tr, settings
            )
            objective = float(r @ r)
            status = "fixed"
        trials.append((objective, point, i, str(status)))
    # Continue each competitive capped solution so start selection cannot prefer
    # an unfinished iterate just because other starts finished sooner.
    best_so_far = min(a for a, *_ in trials)
    for j, (a, b, i, status_text) in enumerate(trials):
        if "LIMIT" in status_text and a <= 1.05 * best_so_far:
            rr = minimize(
                evaluate,
                b,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * len(names),
                options=dict(maxiter=900, ftol=1e-11, gtol=1e-5, maxls=40),
            )
            if rr.fun <= a:
                trials[j] = (rr.fun, rr.x, i, str(rr.message))
    objective, x, seed, status = min(trials, key=lambda a: a[0])
    y = predict(T(x), names, model, z, settings).detach().numpy()
    if not (np.isfinite(y).all() and y.min() >= 0):
        raise ValueError(key)
    params = {
        k: float(v) for k, v in decode(T(x), names).items()
    } | settings.get("fixed", {})
    row = dict(
        case=key,
        model=model,
        fold=fold,
        objective=objective,
        seed=seed,
        status=status,
        params=params,
        screen_size=screen_size,
        starts=[
            dict(
                seed=i,
                objective=a,
                status=s,
                held=scores(
                    predict(T(b), names, model, z, settings).detach().numpy(),
                    z,
                    te,
                )
                if te.any()
                else {},
                params={k: float(v) for k, v in decode(T(b), names).items()},
            )
            for a, b, i, s in trials
        ],
        settings=settings,
        training=scores(y, z, tr),
        held=scores(y, z, te) if te.any() else {},
        prediction=y.tolist(),
    )
    if fold == "all" and key in (
        "conditional",
        "blind",
        "blind_mixture",
        "hybrid",
    ):
        jac = torch.func.jacfwd(
            lambda v: residual(
                predict(v, names, model, z, settings), z, tr, settings
            )
        )(T(x)).numpy()
        _, sv, vt = np.linalg.svd(jac, full_matrices=False)
        # Gradient validation at an interior point; finite differences only for verification.
        point = start_point(names)
        direction = np.cos(np.arange(len(names)))
        h = 1e-5
        fd = (
            evaluate(point + h * direction)[0]
            - evaluate(point - h * direction)[0]
        ) / (2 * h)
        ad = evaluate(point)[1] @ direction
        row["sensitivity"] = dict(
            names=names,
            singular_values=sv.tolist(),
            weak_direction=vt[-1].tolist(),
            column_norm=np.linalg.norm(jac, axis=0).tolist(),
            gradient_relative_error=abs(fd - ad) / max(abs(ad), 1e-10),
        )
    print(key, fold, round(objective, 4), flush=True)
    return row


def jobs():
    runs = []
    for name in (
        "blind",
        "conditional",
        "power",
        "persist",
        "gravity",
        "blind_mixture",
    ):
        for fold in (
            ["all"]
            + [f"wind{w}" for w in WINDS]
            + [f"time{t:g}" for t in HOURS[1:]]
            + ["forward"]
        ):
            runs.append(
                (
                    name,
                    name,
                    fold,
                    {},
                    20
                    if name == "blind_mixture" and fold == "all"
                    else 8
                    if fold == "all" or name == "blind_mixture"
                    else 5,
                )
            )
    alternatives = {
        "fixed_adhesion": {
            "fixed": {"late_threshold_ratio": 1.0, "adhesion_time_min": 30.0}
        },
        "no_lag": {"fixed": {"delivery_lag_min": 0.2}, "instant_air": True},
        "zhang_wood": {"external_shear": True, "zhang": "zhang"},
        "zhang_lower": {"external_shear": True, "zhang": "zhang_low"},
        "zhang_upper": {"external_shear": True, "zhang": "zhang_high"},
        "hard_mobilization": {"hard_mobilization": True},
        "no_removal": {"fixed": {"removal6_per_h": 0.0}},
        "no_sort": {"fixed": {"settling_depletion": 0.0}},
        "external_shear": {"external_shear": True},
        "mass_only": {"shape_weight": 0.0},
        "shape_x4": {"shape_weight": 4.0},
        "equal_mass_scale": {"sem_floor": 100.0, "shape_weight": 0.0001},
    }
    for key, s in alternatives.items():
        for fold in ["all"] + [f"wind{w}" for w in WINDS]:
            settings = dict(s)
            if key == "hard_mobilization" and fold == "wind6":
                settings["fixed"] = {"removal6_per_h": 0.0}
            runs.append(
                (key, "conditional", fold, settings, 8 if fold == "all" else 5)
            )
    return runs


def extra_jobs():
    scenarios = {
        "kinetic_endpoints": {"kinetic": True},
        "lag_2min": {"fixed": {"delivery_lag_min": 2.0}},
        "lag_30min": {"fixed": {"delivery_lag_min": 30.0}},
        "lag_90min": {"fixed": {"delivery_lag_min": 90.0}},
        "lag_360min": {"fixed": {"delivery_lag_min": 360.0}},
        "vc_02": {"fixed": {"capture10_ms": 0.02}},
        "vc_05": {"fixed": {"capture10_ms": 0.05}},
        "vc_20": {"fixed": {"capture10_ms": 0.2}},
        "vc_50": {"fixed": {"capture10_ms": 0.5}},
        "ratio_3": {"fixed": {"late_threshold_ratio": 3.0}},
        "ratio_30": {"fixed": {"late_threshold_ratio": 30.0}},
    }
    return [
        (key, "conditional", fold, s, 6 if fold == "all" else 4)
        for key, s in scenarios.items()
        for fold in (
            ["all"] + [f"wind{w}" for w in WINDS]
            if key in ("kinetic_endpoints", "lag_30min", "vc_05", "ratio_3")
            else ["all"]
        )
    ]


def combined_jobs():
    usual = (
        ["all"]
        + [f"wind{w}" for w in WINDS]
        + [f"time{t:g}" for t in HOURS[1:]]
        + ["forward"]
    )
    corners = [f"corner{w}" for w in WINDS]
    reduced = {
        "instant_air": True,
        "fixed": {
            "late_threshold_ratio": 1.0,
            "adhesion_time_min": 30.0,
            "delivery_lag_min": 0.2,
            "settling_depletion": 0.0,
        },
    }
    return [
        (key, model, fold, settings, 10 if fold == "all" else 6)
        for key, model, settings in [
            ("hybrid", "hybrid", {}),
            ("hybrid_reduced", "hybrid", reduced),
            ("conditional", "conditional", {}),
            ("power", "power", {}),
            ("persist", "persist", {}),
        ]
        for fold in ((usual + corners) if model == "hybrid" else corners)
    ]


def measurement_jobs():
    return [
        (key, model, "all", settings, 10)
        for model in ("conditional", "hybrid")
        for key, settings in [
            (model + "_drop6mass", {"omit6mass": True}),
            (model + "_drop6laser", {"omit6shape": True}),
        ]
    ]


def charge_jobs():
    return [
        (key, "conditional", fold, settings, 8 if fold == "all" else 5)
        for key, settings in [
            ("charge_inferred", {"charge_pool": "inferred"}),
            ("charge_bed", {"charge_pool": "bed10"}),
            (
                "charge_strong",
                {"charge_pool": "inferred", "fixed": {"charge_uC_m2": 10.0}},
            ),
        ]
        for fold in ["all", "wind0", "wind2", "wind4", "wind6", "forward"]
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument(
        "--set",
        choices=[
            "core",
            "stress",
            "extra",
            "combined",
            "measurement",
            "charge",
        ],
        default="core",
    )
    args = ap.parse_args()
    DATA = data(args.minutes)
    jr = jobs()
    jr = (
        jr[:60]
        if args.set == "core"
        else jr[60:]
        if args.set == "stress"
        else extra_jobs()
        if args.set == "extra"
        else combined_jobs()
        if args.set == "combined"
        else measurement_jobs()
        if args.set == "measurement"
        else charge_jobs()
    )
    if not jr:
        raise RuntimeError("No jobs")
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(fit, jr))
    out = dict(
        torch=torch.__version__,
        numpy=np.__version__,
        minutes=args.minutes,
        seconds=time.time() - started,
        workbook_sha256=hashlib.sha256(
            (ROOT / "Supporting_data_v2.xlsx").read_bytes()
        ).hexdigest(),
        results=sorted(results, key=lambda r: (r["case"], r["fold"])),
    )
    (ROOT / f"global-{args.set}.json").write_text(
        json.dumps(out, separators=(",", ":"), allow_nan=False)
    )
