"""Private two-pulse sulfur return sensitivity using first-pulse fits."""

import csv
import numpy as np
from scipy.optimize import nnls
from warm import HERE, measurements, write
from replay import ZipFile, ARCHIVE, curves, median_samples
from sulfur_inputs import inputs


def gas_response(survival, gain, mid_source):
    state = np.zeros(len(mid_source) + 1)
    for i in range(len(mid_source)):
        state[i + 1] = survival[i] * state[i] + gain[i] * mid_source[i]
    return state


def main():
    t, *_, measured, of, pf, obs, hp, valid, so = measurements()
    h = t - t[0]
    _, fixed, _, _ = inputs()
    with ZipFile(ARCHIVE) as z:
        led = np.interp(t, *curves(z, "02a")[0].T)
        ozone = median_samples(curves(z, "04a")[0], t)
        rate = np.interp(t, *curves(z, "12d")[-1].T)
    source = led / led.max() * so / np.median(so) * ozone / np.median(ozone)
    driver = (source[1:] + source[:-1]) * 0.5
    dt = np.diff(t) * 3600
    rate = 0.5 * (rate[:-1] + rate[1:])
    train = (h >= 21.9) & (h < 26.7) & valid
    hold = (h >= 31.55) & valid
    blocks = np.floor((h[train] - 21.9) / 0.6).astype(int)
    data = list(csv.DictReader((HERE / "warm-parameters.csv").open()))
    rng = np.random.default_rng(183309)
    scores = []
    for p in data:
        sink_mult = rng.uniform(0.9, 1.1)
        k = rate * sink_mult
        surv = np.exp(-k * dt)
        gain = -np.expm1(-k * dt) / k
        # Sources are in arbitrary units; solve their nonnegative amplitudes on LED1.
        # Keep the OH reconstruction fixed; vary only response of the extra gases.
        lifetime = float(p["led_lifetime_s"])
        half = np.exp(-dt * 0.5 / lifetime)
        storage = gas_response(half**2, 1 - half**2, driver)
        delayed = driver + (storage[:-1] - driver) * half
        gas_bg = gas_response(surv, gain, np.ones_like(driver))
        gas_return = gas_response(surv, gain, delayed)
        q = float(p["acid"]) * measured
        unit = rng.exponential(size=blocks.max() + 1)
        weights = np.sqrt(unit[blocks] / np.mean(unit[blocks]))
        for model, cols in {
            "retained": [fixed],
            "background": [fixed, gas_bg],
            "delayed_return": [fixed, gas_return],
            "both": [fixed, gas_bg, gas_return],
        }.items():
            X = np.array(cols).T
            train_X = X[train] * weights[:, None]
            scale = np.linalg.norm(train_X, axis=0)
            coef = nnls(train_X / scale, q[train] * weights)[0] / scale
            error = np.mean((X[hold] @ coef - q[hold]) ** 2) ** 0.5
            scores.append(
                dict(
                    draw=p["draw"],
                    family=model,
                    scale=p["acid"],
                    extra_sink_multiplier=sink_mult,
                    lifetime_s=lifetime,
                    second_LED_n=hold.sum(),
                    second_LED_rmse_molecules_cm3=error,
                )
            )
    write("warm-led.csv", scores)


if __name__ == "__main__":
    main()
