# %% Run 1833 forcing, spectra and conservative sectional chamber replays.
"""Run from the verified Particula worktree with PYTHONPATH set to its root."""

import argparse
import ctypes
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import particula as par

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["PARTICULA_ROOT"]).resolve()
OUT = Path(os.environ["STUDY_PACKET"]).resolve()
SOURCE = Path(os.environ.get("CLOUD_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(ROOT / "studies/dunne-cloud/companion"))
from extract_run1833 import curves, median_samples, spectrum

sys.path.insert(0, str(ROOT / "studies/dunne-cloud/analysis"))
from attraction import acid_pair

STAGES = np.array(
    [
        -6.50,
        -5.99,
        -4.83,
        -3.58,
        -2.77,
        -1.77,
        -0.72,
        -0.14,
        4.29,
        5.87,
        7.81,
        9.82,
        11.14,
        11.68,
        13.03,
        13.53,
        13.82,
        15.22,
        16.20,
        16.76,
        19.87,
        21.27,
        22.77,
        24.27,
        25.02,
        26.54,
    ]
)
T, P = 297.8, 98700.0
START = time.monotonic()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs():
    path = SOURCE / "donahue2026-figure-data.zip"
    with path.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == "5f6f48d653a989a57b633472908ebbddffad10a014db73032dc9f940912d4c85"
        )
    with ZipFile(path) as z:
        t, d, e, y = spectrum(z)
        (lin,) = curves(z, "11a")
        assert (
            np.max(
                np.abs(lin[:, 1] - y[:, np.argmin(abs(t + 1.483056))])
                / np.maximum(lin[:, 1], 1)
            )
            < 1.00001
        )
        np.testing.assert_allclose(
            lin[lin[:, 1] > 1, 1],
            y[lin[:, 1] > 1, np.argmin(abs(t + 1.483056))],
            rtol=1e-6,
        )
        uhel, ufra = [median_samples(a, t) for a in curves(z, "02b")]
        oh, author, _, _, corrected = curves(z, "12c")
        cpc, total = curves(z, "06b")
        measured = lambda panel: median_samples(curves(z, panel)[0], t)
        so, o3 = measured("04b"), measured("04a")
        lamps = [median_samples(a, t) for a in curves(z, "02a")]
        loss = curves(z, "12d")[-1]
        kso = curves(z, "12b")[5]
        soff = curves(z, "13b")[0]
        # R71-R73 chain using published OH, measured SO2 and original first-order sinks.
        raw = oh[:, 0]
        air = P / (1.380649e-23 * T) * 1e-6
        measured_so = (
            np.interp(raw, t[np.isfinite(so)], so[np.isfinite(so)]) * 1e-9 * air
        )
        prod = (
            oh[:, 1] * kso[:, 1] * measured_so / soff[:, 1]
            + curves(z, "07")[9][0, 1]
        )
        h = np.diff(raw) * 3600
        l = (loss[1:, 1] + loss[:-1, 1]) / 2
        decay = np.exp(-l * h)
        source = -np.expm1(-l * h) / l * (prod[1:] + prod[:-1]) / 2
        reduced = np.empty(len(raw))
        reduced[0] = author[0, 1]
        for i in range(len(h)):
            reduced[i + 1] = decay[i] * reduced[i] + source[i]
        # Detailed acid-budget check against Figure 7 rates on its UVH interval.
        rates = curves(z, "07")
        model8 = curves(z, "08a")[0]
        rtime = model8[:, 0]
        lh = rates[8][:, 1] / model8[:, 1]
        width = np.diff(rtime) * 3600
        mid = (lh[1:] + lh[:-1]) / 2
        yu = [model8[0, 1]]
        for a, b in zip(
            np.exp(-mid * width),
            -np.expm1(-mid * width)
            / mid
            * (rates[-1][1:, 1] + rates[-1][:-1, 1])
            / 2,
        ):
            yu.append(a * yu[-1] + b)
        budget_ref = float(
            np.sqrt(np.mean((np.array(yu) - model8[:, 1]) ** 2))
            / np.max(model8[:, 1])
        )
    matched = np.isfinite(ufra)
    full = np.interp(t, t[matched], ufra[matched])
    filled_uhel = full.copy()
    usable = (~matched) & np.isfinite(uhel)
    filled_uhel[usable] = uhel[usable]
    stage = np.searchsorted(STAGES, t, side="right")
    gas = np.stack(
        [
            1.15 * full,
            np.interp(t, author[:, 0], author[:, 1]),
            np.interp(t, raw, reduced),
            1.15 * filled_uhel,
            full,
        ]
    )
    missing = np.all(y[:7] == 0, axis=0) & (t >= 15.8) & (t <= 18.2)
    assert missing.sum() > 50
    assert np.min(gas) >= 0
    observed = pd.DataFrame(
        dict(
            archive_h=t,
            stage=stage,
            ufra_raw=ufra,
            uhel_raw=uhel,
            measured_cal=gas[0],
            author_chem=gas[1],
            integrated_chem=gas[2],
            uhel_gapfill=gas[3],
            ozone_ppbv=o3,
            so2_ppbv=so,
            led=lamps[0],
            excimer=lamps[1],
            hglamp=lamps[2],
            cpc=cpc[:, 1],
            small_missing=missing,
        )
    )
    observed.to_csv(OUT / "forcing.csv", index=False)
    pd.DataFrame(
        dict(archive_h=rtime, figure8_acid=model8[:, 1], figure7_balance=yu)
    ).to_csv(OUT / "gas-reference.csv", index=False)
    write_json(
        "reference.json",
        dict(
            uvh_normalized_balance_rmse=budget_ref,
            shape=list(y.shape),
            calibrated_points=len(corrected),
            missing_small_spectra=int(missing.sum()),
        ),
    )
    return t, d, e, y.T, gas, stage, missing


def overlap(low, high, edges):
    return np.maximum(
        0,
        np.minimum(high[:, None], edges[1:])
        - np.maximum(low[:, None], edges[:-1]),
    )


def grid(n, d, edges):
    ed = np.linspace(np.log10(2.9), edges[-1], n + 1)
    dx = ed[1] - ed[0]
    centers = 10 ** ((ed[:-1] + ed[1:]) / 2)
    dia = 10**ed
    radius = dia * 0.5e-9
    diff = 8e-6 * (T / 293.15) ** 1.75
    speed = np.sqrt(8 * 8.314462618 * T / (np.pi * 0.098079))
    vapor = par.dynamics.get_first_order_mass_transport_k(
        radius,
        par.particles.get_vapor_transition_correction(
            3 * diff / (speed * radius), 1.0
        ),
        diff,
    )
    ref, pair = acid_pair(T, P, 6.4e-20, 1.0, 1500.0, 19.5, 151)
    # Stay inside the qualified capture table. Smoothly return to vapor
    # transfer between 9.7 and 19.5 nm, identical large-size law thereafter.
    fraction = np.clip(np.log(19.5 / dia) / np.log(19.5 / 9.7), 0, 1)
    enhanced = vapor + fraction * (np.interp(dia, ref, pair) - vapor)
    hard_d, hard_k = acid_pair(T, P, 0.0, 1.0, 1500.0, 19.5, 151)
    finite = vapor + fraction * (np.interp(dia, hard_d, hard_k) - vapor)
    # Hydrated ammonium sulfate inventory in acid formula units. Analytic dN/dD.
    kelvin = 4 * 0.079 * 0.01801528 / (1000 * 8.314462618 * T) * 1.0e9
    x = 0.58 * np.exp(-kelvin / dia)
    unit = 98.079 + 2 * 17.03052 + 18.01528 * 3 * x / (1 - x)
    derivative_unit = 18.01528 * 3 * x * kelvin / (dia**2 * (1 - x) ** 2)
    units = 1350 * np.pi / 6 * (dia * 1e-9) ** 3 / (unit * 1.66053906660e-27)
    der = units * (3 / dia - derivative_unit / unit)
    vel = np.stack([vapor, enhanced, finite]) * 1e6 / der / (np.log(10) * dia)
    wall = 0.002 * np.sqrt(
        par.particles.get_diffusion_coefficient_via_system_state(
            centers * 0.5e-9, T, P
        )
        / diff
    )
    vol = centers**3
    K = (
        par.dynamics.get_brownian_kernel_via_system_state(
            centers * 0.5e-9, np.pi / 6 * 1350 * (centers * 1e-9) ** 3, T, P
        )
        * 1e6
    )
    i, j = np.triu_indices(n)
    v = vol[i] + vol[j]
    hi = np.searchsorted(vol, v)
    valid = hi < n
    i, j, v, hi = (a[valid] for a in (i, j, v, hi))
    f = (v - vol[hi - 1]) / (vol[hi] - vol[hi - 1])
    k = K[i, j] * np.where(i == j, 0.5, 1.0)
    target = np.r_[hi, hi - 1]
    order = np.argsort(target)
    offsets = np.r_[0, np.bincount(target, minlength=n).cumsum()].astype("i4")
    pairs = [
        offsets,
        np.tile(i, 2)[order].astype("i4"),
        np.tile(j, 2)[order].astype("i4"),
        np.r_[k * f, k * (1 - f)][order],
    ]
    overflow = np.searchsorted(vol, vol[-1] - vol, side="right").astype("i4")
    map_in = overlap(ed[:-1], ed[1:], edges)
    map_out = overlap(edges[:-1], edges[1:], ed) / dx
    # Independently audit fixed-pivot coagulation in both number and volume.
    probe = 100 * np.exp(-(((np.log(centers / 12)) / 0.7) ** 2)) + 0.1
    coeff = K * probe[:, None] * probe[None, :]
    above = vol[:, None] + vol[None, :] > vol[-1]
    gain = np.bincount(
        target[order],
        weights=pairs[3] * probe[pairs[1]] * probe[pairs[2]],
        minlength=n,
    )
    change = gain - coeff.sum(1)
    assert (
        abs(change.sum() + 0.5 * (coeff.sum() + coeff[above].sum()))
        < 1e-11 * coeff.sum()
    )
    volume_out = 0.5 * np.sum(coeff * above * (vol[:, None] + vol[None, :]))
    assert abs(change @ vol + volume_out) < 1e-11 * (coeff.sum(1) @ vol)
    return dx, ed, vel, wall, K, pairs, overflow, map_in, map_out


def simulate(
    data,
    take,
    configs,
    n,
    dt,
    clearing=False,
    reference=False,
    common_entry=None,
):
    import warp as wp

    t, d, e, y, gas, stage, missing = data
    t, y, gas, stage = t[take], y[take], gas[:, take], stage[take]
    dx, ed, v, wall, K, pairs, over, mi, mo = grid(n, d, e)
    flux_ref = (
        np.full(len(t), -1.0)
        if common_entry is None
        else common_entry * gas[0] * v[1, 0]
    )
    configs = np.array(configs, dtype=float)
    gas = gas[configs[:, 0].astype(int)]
    initial = mi @ y[0]
    inlet = y[:, 3].copy()
    clear = np.zeros(len(t))
    if clearing:
        clear[1:] = (
            0.15
            * np.isin(stage[1:], [2, 5, 11, 14])
            * (stage[1:] != stage[:-1])
        )
    if reference:
        # Independent NumPy SSP-RK2 pilot reference (growth and losses only).
        state = initial.copy()
        history = [state.copy()]
        gain, ws, law, coal = configs[0, 1:]
        assert coal == 0
        for j, dur in enumerate(np.diff(t) * 3600):
            ns = max(
                1,
                int(np.ceil(dur / dt)),
                int(
                    np.ceil(
                        dur
                        * np.max(gas[0, j : j + 2])
                        * gain
                        * np.max(v[int(law)])
                        / (0.42 * dx)
                    )
                ),
            )
            for s in range(ns):
                f = (s + 0.5) / ns
                a = ((1 - f) * gas[0, j] + f * gas[0, j + 1]) * gain
                bd = inlet[j] * (1 - f) + inlet[j + 1] * f

                def rhs(state):
                    l, r = np.diff(state)[:-1], np.diff(state)[1:]
                    slope = np.r_[
                        0,
                        np.where(
                            l * r > 0,
                            np.sign(l) * np.minimum(abs(l), abs(r)),
                            0,
                        ),
                        0,
                    ]
                    return (
                        -np.diff(
                            a
                            * v[int(law)]
                            * np.r_[bd, (state + 0.5 * slope) / dx]
                        )
                        - (ws * wall + 0.0001754037) * state
                    )

                pred = state + dur / ns * rhs(state)
                state = 0.5 * (state + pred + dur / ns * rhs(pred))
            state *= 1 - clear[j + 1]
            history.append(state.copy())
        return np.asarray(history) @ mo.T / np.diff(e)
    lib = ctypes.CDLL(os.environ["CLOUD_SOLVER"])
    lib.replay.argtypes = (
        [ctypes.c_int] * 3
        + [ctypes.c_double] * 2
        + [ctypes.POINTER(ctypes.c_void_p)]
    )
    items = [
        t,
        gas,
        inlet,
        initial,
        np.r_[v.ravel(), v.max(axis=1)],
        wall,
        configs[:, 1:],
        K,
        *pairs,
        over,
        clear,
        flux_ref,
    ]
    dev = "cuda:0"
    arrays = [wp.array(np.ascontiguousarray(x), device=dev) for x in items]
    out = wp.empty((len(configs), len(t), n), dtype=wp.float64, device=dev)
    ledger = wp.empty((len(configs), n, 6), dtype=wp.float64, device=dev)
    wp.synchronize_device(dev)
    ptrs = (ctypes.c_void_p * 17)(*[a.ptr for a in arrays + [out, ledger]])
    rc = lib.replay(len(configs), n, len(t), dx, dt, ptrs)
    if rc:
        raise RuntimeError(f"CUDA error {rc}")
    rows = out.numpy()
    led = ledger.numpy().sum(1)
    end = rows[:, -1].sum(1)
    error = end - (initial.sum() + led[:, 0] - led[:, 1:].sum(1))
    relative = abs(error) / np.maximum(1, led[:, 0] + initial.sum())
    assert (
        np.isfinite(rows).all()
        and rows.min() >= -1e-10
        and relative.max() < 1e-9
    ), (rows.min(), relative.max())
    return rows @ mo.T / np.diff(e), dict(
        bins=n,
        step_cap_s=dt,
        cases=len(configs),
        spectra=len(t),
        min_population=float(rows.min()),
        max_relative_number_residual=float(relative.max()),
        ledgers=np.column_stack(
            [np.full(len(end), initial.sum()), led, end, error]
        ).tolist(),
    )


def score(y, obs, e, floor=1.0, detailed=False):
    # Ignore all centers at/below the measured entry. Each retained bin is whole.
    widths = np.diff(e)[4:]
    y = y[..., 4:]
    obs = obs[..., 4:]
    val = (
        np.abs(np.log10(floor + y) - np.log10(floor + obs)) @ widths
    ) / widths.sum()
    return val if detailed else val.mean(axis=-1)


def run(mode):
    import warp as wp

    wp.init()
    assert wp.get_cuda_device_count() > 0
    data = inputs()
    t, d, e, y, gas, stage, missing = data
    train_idx = np.flatnonzero(stage <= 7)
    test_idx = np.flatnonzero((stage >= 9) & (stage <= 18) & (~missing))
    ledgers = []
    if mode == "controls":
        from controls import controls

        controls(data, train_idx, test_idx)
        return
    if mode == "pilot":
        idx = train_idx[:30]
        cp = simulate(data, idx, [(0, 1, 1, 0, 0)], 64, 30, reference=True)
        gp, log = simulate(
            data, idx, [(0, 1, 1, 0, 0), (1, 1, 1, 1, 1)], 64, 30
        )
        delta = np.max(abs(cp - gp[0]))
        assert delta < 1e-7
        write_json(
            "pilot.json",
            dict(
                elapsed_s=time.monotonic() - START,
                device=wp.get_device("cuda:0").name,
                max_cpu_gpu_density_delta=float(delta),
                **log,
            ),
        )
        print("pilot", delta, time.monotonic() - START)
        return
    # Fit the same wall and growth multipliers for both forcings, from measured
    # acid and Hg-Xe only. Select separately for each uptake law for a sensitivity.
    config = [
        (0, g, w, law, 1)
        for law, g, w in itertools.product(
            [0, 1], [0.8, 0.9, 1.0, 1.1, 1.2], [0.5, 0.75, 1.0, 1.25, 1.5]
        )
    ]
    pop, log = simulate(data, train_idx, config, 96, 24)
    ledgers.append(dict(label="calibration", **log))
    serr = score(pop[:, 1:], y[train_idx[1:]], e)
    table = pd.DataFrame(
        config, columns=["forcing", "growth", "wall", "law", "coag"]
    )
    table["train_score"] = serr
    table.to_csv(OUT / "training.csv", index=False)
    # Main setup uses the enhanced-law reference's training choice for BOTH laws and forcings.
    best = table.loc[table[table.law == 1].train_score.idxmin()]
    other = table.loc[table[table.law == 0].train_score.idxmin()]
    selected = [
        (f, best.growth, best.wall, law, 1)
        for f, law in itertools.product([0, 1], [0, 1])
    ]
    variants = selected + [
        (f, best.growth, best.wall, law, 0)
        for f, law in itertools.product([0, 1], [0, 1])
    ]
    variants += [(f, best.growth, best.wall, 1, 1) for f in (2, 3, 4)]
    variants += [(f, other.growth, other.wall, 0, 1) for f in (0, 1)]
    # Retain all calibration solutions within 0.05 dex of the enhanced-law training optimum.
    near = table[
        (table.law == 1) & (table.train_score <= best.train_score + 0.05)
    ]
    variants += [
        (f, row.growth, row.wall, 1, 1)
        for row in near.itertuples()
        for f in (0, 1)
    ]
    variants = list(dict.fromkeys(variants))
    selected = list(dict.fromkeys(selected))
    scores = []
    stages = []
    store = {}
    selected_meta = {}
    for tag, idx, cs, n, dt, clear in [
        ("train", train_idx, selected, 192, 12, False),
        ("test", test_idx, variants, 192, 12, False),
        ("coarse", test_idx, selected, 96, 24, False),
        ("fine", test_idx, selected, 288, 6, False),
        ("timestep", test_idx, selected, 192, 6, False),
        ("field", test_idx, selected, 192, 12, True),
    ]:
        p, log = simulate(data, idx, cs, n, dt, clear)
        ledgers.append(dict(label=tag, **log))
        use = np.arange(len(idx)) > 0
        work = use & (stage[idx] < 18) if tag != "train" else use
        for i, (conf, pred) in enumerate(zip(cs, p)):
            f, g, w, law, coal = conf
            meta = dict(
                run=tag, case=i, forcing=f, growth=g, wall=w, law=law, coag=coal
            )
            scores.append(
                dict(
                    **meta,
                    score=float(score(pred[work], y[idx[work]], e)),
                    floor10=float(score(pred[work], y[idx[work]], e, 10)),
                )
            )
            centers = d[4:]
            widths = np.diff(e)[4:]
            thresh = 10 ** e[4]
            for s in np.unique(stage[idx]):
                mask = (stage[idx] == s) & use
                if not mask.any():
                    continue
                yp, yo = pred[mask, 4:], y[idx[mask], 4:]
                npred = yp @ widths
                nobs = yo @ widths

                def shape(arr):
                    weights = arr * widths
                    mean = (weights @ np.log(centers)) / weights.sum(1)
                    return np.exp(
                        np.sqrt(
                            np.sum(
                                weights
                                * (np.log(centers) - mean[:, None]) ** 2,
                                axis=1,
                            )
                            / weights.sum(1)
                        )
                    )

                stages.append(
                    dict(
                        **meta,
                        stage=int(s),
                        spectra=int(mask.sum()),
                        score=float(score(pred[mask], y[idx[mask]], e)),
                        count_pred=npred.mean(),
                        count_obs=nobs.mean(),
                        count_mae=abs(npred - nobs).mean(),
                        mode_pred_nm=np.median(centers[yp.argmax(1)]),
                        mode_obs_nm=np.median(centers[yo.argmax(1)]),
                        width_pred=np.median(shape(yp)),
                        width_obs=np.median(shape(yo)),
                    )
                )
        if tag in ("train", "test"):
            store[tag + "_indices"] = idx
            store[tag + "_selected"] = p[: len(selected)]
            selected_meta[tag] = cs[: len(selected)]
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(stages).to_csv(OUT / "stages.csv", index=False)
    np.savez_compressed(
        OUT / "spectra.npz",
        time=t,
        diameter=d,
        edges=e,
        observed=y,
        stage=stage,
        **store,
    )
    write_json(
        "campaign.json",
        dict(
            device=wp.get_device("cuda:0").name,
            revision=subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            python=sys.version,
            particula_source=par.__file__,
            source_version=par.__version__,
            warp=wp.__version__,
            runs=ledgers,
            ledger_columns=[
                "initial",
                "entry",
                "upper_growth_outflow",
                "wall",
                "dilution",
                "coag_mergers_and_overflow",
                "field",
                "final",
                "closure_residual",
            ],
            selected=selected_meta,
            common_growth=best.growth,
            common_wall=best.wall,
            alternative_vapor_fit=[other.growth, other.wall],
            near_count=len(near),
            elapsed_s=time.monotonic() - START,
            training_cases=len(config),
            cuda_cases=sum(l["cases"] for l in ledgers),
        ),
    )
    print(
        "selected",
        best.to_dict(),
        "production seconds",
        time.monotonic() - START,
    )
    print(pd.DataFrame(scores).query("run=='fine'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "production", "controls"])
    run(parser.parse_args().mode)
