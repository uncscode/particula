# %% Measured lamp and precursor replay with a resolved condensation sink
"""Reduced gas kinetics. Only the trained radical loss is fitted here."""

import json
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar
from study import (
    OUT,
    SOURCE,
    P,
    T,
    acid_pair,
    curves,
    inputs,
    par,
    score,
    simulate,
    write_json,
)


def filled(x):
    return np.interp(
        np.arange(len(x)), np.flatnonzero(np.isfinite(x)), x[np.isfinite(x)]
    )


def acid_balance(t, production, loss, initial):
    h = np.diff(t) * 3600
    mid = (loss[1:] + loss[:-1]) / 2
    survival = np.exp(-mid * h)
    feed = -np.expm1(-mid * h) / mid * (production[1:] + production[:-1]) / 2
    a = np.empty(len(t))
    a[0] = initial
    for i in range(len(h)):
        a[i + 1] = survival[i] * a[i] + feed[i]
    return a


def gas_production(t, stage, frame):
    # Compare a reduced independent light-to-acid replay with the archived model,
    # allowing the authored relative lamp actinometry as a declared input.
    air = P / (1.380649e-23 * T) * 1e-6
    # Buck water saturation pressure in Pa at Celsius temperature.
    tc = T - 273.15
    es = 611.21 * np.exp((18.678 - tc / 234.5) * tc / (257.14 + tc))
    water = air * 0.58 * es / P
    o3 = filled(frame.ozone_ppbv.to_numpy()) * air * 1e-9
    so2 = filled(frame.so2_ppbv.to_numpy()) * air * 1e-9
    wet = 1.63e-10 * np.exp(60 / T) * water
    branch = wet / (
        wet
        + (air - water)
        * (0.79 * 1.8e-11 * np.exp(107 / T) + 0.21 * 3.2e-11 * np.exp(67 / T))
    )
    with ZipFile(SOURCE / "donahue2026-figure-data.zip") as z:
        productions = curves(z, "12a")
        original = curves(z, "12c")
        sulfloss = curves(z, "12d")
        oh_rates = curves(z, "12b")
        source = curves(z, "07")[9][0, 1]
    interp = lambda trace: np.interp(t, trace[:, 0], trace[:, 1])
    prim = np.zeros(len(t))
    coeff = {}
    for name, j, mask in [
        ("hglamp", 1, stage <= 6),
        ("excimer", 2, (stage >= 9) & (stage <= 17)),
        ("led", 0, (stage >= 19)),
    ]:
        light = np.maximum(
            0, median_filter(filled(frame[name].to_numpy()), size=3)
        )
        threshold = light > 0.1 * light[mask].max()
        valid = mask & threshold & (interp(productions[j]) > 0)
        sensitivity = np.median(
            interp(productions[j])[valid] / (light[valid] * o3[valid])
        )
        coeff[name] = sensitivity
        prim += light * o3 * sensitivity
    # SR22 falloff ratio is evaluated from the authors' archived rate coefficient,
    # separating rate-convention differences from the measured SO2 forcing.
    with ZipFile(SOURCE / "donahue2026-figure-data.zip") as z:
        kso = np.median(oh_rates[5][:, 1] / curves(z, "13b")[0][:, 1])
    carbon = 1.3e-13 * (1 + 0.6 * air / 2.5e19) * 100e-9 * air
    fwd = carbon + 1.7e-12 * np.exp(-940 / T) * o3 + kso * so2
    recycle = 1e-14 * np.exp(-490 / T) * o3
    rc = 4.8e-11 * np.exp(250 / T)
    self_rate = (3e-13 * np.exp(460 / T) + 2.1e-33 * np.exp(920 / T) * air) * (
        1 + 2.8e-18 * water
    )
    src = 2 * prim * branch
    oth = np.full(len(t), 0.0001754037)

    # H2O2 and H2 reactions and tracer chemistry are small in the two scored
    # lamp campaigns; their omitted recycling is part of this closure error.
    def radicals(loss, stop):
        ho2 = np.empty(stop)
        ho2[0] = original[2][0, 1]
        for i in range(stop - 1):
            q = ho2[i]
            h = (t[i + 1] - t[i]) * 3600 / 12
            for j in range(12):
                f = (j + 0.5) / 12
                forward = (1 - f) * fwd[i] + f * fwd[i + 1]
                rec = (1 - f) * recycle[i] + f * recycle[i + 1]
                feed = (1 - f) * src[i] + f * src[i + 1]

                def rhs(val):
                    oh = (feed + rec * val) / (forward + oth[i] + rc * val)
                    return (
                        forward * oh
                        - (rec + loss + rc * oh + 2 * self_rate * val) * val
                    )

                mid = q + 0.5 * h * rhs(q)
                q += h * rhs(mid)
            ho2[i + 1] = q
        oh = (src[:stop] + recycle[:stop] * ho2) / (
            fwd[:stop] + oth[:stop] + rc * ho2
        )
        return oh, ho2

    stop = np.flatnonzero(stage <= 7)[-1] + 1
    target = interp(original[2])[:stop]

    def cost(k):
        _, ho = radicals(k, stop)
        return np.mean((ho - target) ** 2) / np.mean(target**2)

    fit = minimize_scalar(
        cost, bounds=(0.0, 0.025), method="bounded", options={"xatol": 1e-7}
    )
    oh, ho = radicals(fit.x, len(t))
    assert np.min(ho) >= 0
    return (
        oh * kso * so2 + source,
        interp(sulfloss[-1]),
        interp(sulfloss[6]),
        dict(
            OH=oh,
            HO2=ho,
            OH_author=interp(original[0]),
            HO2_author=interp(original[2]),
        ),
        dict(
            oxygen1D_water_branch=branch,
            kSO2=kso,
            constant_acid_inflow=source,
            lamp_coefficients=coeff,
            radical_loss_s=fit.x,
            uvh_relative_ho2_rmse=float(np.sqrt(fit.fun)),
        ),
    )


def capture(d, g):
    diff = 8e-6 * (T / 293.15) ** 1.75
    radius = 0.5e-9 * d
    velocity = np.sqrt(8 * 8.314462618 * T / (np.pi * 0.098079))
    vapor = par.dynamics.get_first_order_mass_transport_k(
        radius,
        par.particles.get_vapor_transition_correction(
            3 * diff / (velocity * radius), 1.0
        ),
        diff,
    )
    ref, k = acid_pair(T, P, 6.4e-20, 1.0, 1500.0, 19.5, 151)
    f = np.clip(np.log(19.5 / d) / np.log(19.5 / 9.7), 0, 1)
    return g * 1e6 * (vapor + f * (np.interp(d, ref, k) - vapor))


def main():
    start = time.monotonic()
    data = inputs()
    t, d, e, y, gas, stage, missing = data
    frame = pd.read_csv(OUT / "forcing.csv")
    prod, loss, old_sink, radical, meta = gas_production(t, stage, frame)
    summary = json.loads((OUT / "campaign.json").read_text())
    g, w = summary["common_growth"], summary["common_wall"]
    caps = capture(d, g)
    observed_sink = y @ (caps * np.diff(e))
    assert np.min(loss - old_sink) > 0
    open_acid = acid_balance(t, prod, loss, gas[1, 0])
    frame["reduced_lamp_acid"] = open_acid
    frame["production"] = prod
    for key, value in radical.items():
        frame[key] = value
    frame["original_sink"] = old_sink
    frame["observed_spectrum_sink"] = observed_sink
    idx = np.flatnonzero((stage >= 9) & (stage <= 18) & ~missing)
    # Initial acid is the reduced model carried through the preceding stages.
    # Below 2.9 nm retain the observed contribution, bounded within each log bin.
    small_width = np.maximum(0, np.minimum(e[1:], np.log10(2.9)) - e[:-1])
    small = y @ (caps * small_width)
    cases = []
    fields = {}
    metrics = []
    errors = []
    mask = stage[idx] < 18
    mask[0] = False

    def iterate(label, acid):
        mix = gas.copy()
        mix[2, idx] = acid
        pred, log = simulate(
            (*data[:4], mix, *data[5:]), idx, [(2, g, w, 1, 1)], 192, 12
        )
        cases.append(dict(label=label, **log))
        return pred[0]

    def keep(label, acid, pred, sink):
        fields[label] = pred
        metrics.append(
            dict(
                case=label,
                score=float(score(pred[mask], y[idx[mask]], e)),
                mean_acid=float(acid[mask].mean()),
                mean_observed_acid=float(gas[0, idx][mask].mean()),
                count_pred=float(np.mean(pred[mask, 4:] @ np.diff(e)[4:])),
                count_obs=float(np.mean(y[idx[mask], 4:] @ np.diff(e)[4:])),
            )
        )
        frame.loc[idx, label + "_acid"] = acid
        frame.loc[idx, label + "_sink"] = sink

    p = iterate("lamp_published_sink", open_acid[idx])
    keep("lamp_published_sink", open_acid[idx], p, old_sink[idx])
    fixed = loss[idx] - old_sink[idx]
    ac = acid_balance(
        t[idx], prod[idx], fixed + observed_sink[idx], open_acid[idx[0]]
    )
    p = iterate("lamp_native_observed_sink", ac)
    keep("lamp_native_observed_sink", ac, p, observed_sink[idx])
    for k in range(24):
        # Mapped prediction includes partial measured bins down to the inlet.
        cs = small[idx] + p @ (caps * np.diff(e))
        next_ac = acid_balance(t[idx], prod[idx], fixed + cs, open_acid[idx[0]])
        err = np.max(abs(next_ac - ac)) / np.max(ac)
        errors.append(float(err))
        ac = next_ac
        p = iterate(f"coupling_{k + 1}", ac)
        if err < 1e-5:
            break
    assert errors[-1] < 1e-5
    keep("lamp_coupled", ac, p, cs)
    # Same observed and self-consistent sink tests at higher grid resolution can be
    # rerun by changing the two explicit resolution arguments in iterate().
    frame.to_csv(OUT / "lamp_budget.csv", index=False)
    np.savez_compressed(OUT / "lamp_spectra.npz", **fields)
    pd.DataFrame(metrics).to_csv(OUT / "lamp_scores.csv", index=False)
    write_json(
        "lamp.json",
        dict(
            **meta,
            iterations=errors,
            runs=cases,
            seconds=time.monotonic() - start,
        ),
    )
    print(pd.DataFrame(metrics).to_string(index=False), errors)


if __name__ == "__main__":
    main()
