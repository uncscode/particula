"""Test whether LED-correlated chamber-wall sulfur returns to measured gas-phase acid."""

import numpy as np
from scipy.optimize import nnls
from balance import write
from replay import measurements, ZipFile, ARCHIVE, curves, median_samples
from sulfur_inputs import inputs


def main():
    t, _, _, _, measured, _, _, _, _, valid, so = measurements()
    h = t - t[0]
    _, baseline, _, _ = inputs()
    with ZipFile(ARCHIVE) as z:
        led = np.interp(t, *curves(z, "02a")[0].T)
        ozone = median_samples(curves(z, "04a")[0], t)
        rate = np.interp(t, *curves(z, "12d")[-1].T)
    source = led / led.max() * so / np.median(so) * ozone / np.median(ozone)

    def response(lifetime, driver):
        reservoir = 0.0
        gas = 0.0
        out = [gas]
        for j, dt in enumerate(np.diff(t) * 3600):
            x = (driver[j] + driver[j + 1]) * 0.5
            k = (rate[j] + rate[j + 1]) * 0.5
            flux = x
            if lifetime:
                half = np.exp(-dt * 0.5 / lifetime)
                flux = x + (reservoir / lifetime - x) * half
                reservoir = x * lifetime + (reservoir - x * lifetime) * half**2
            gas = gas * np.exp(-k * dt) + flux * (-np.expm1(-k * dt)) / k
            out.append(gas)
        return np.array(out)

    train = (h >= 21.9) & (h < 26.7) & valid
    lamp2 = (h >= 31.55) & valid
    scores = []
    for scale in [1.05, 1.15, 1.25]:
        y = scale * measured
        for label, lifetime in [
            ("wall_retains_sulfur", None),
            ("constant_background", None),
            ("vapor_return", 0),
            ("vapor_return", 600),
            ("vapor_return", 3600),
            ("background_and_return", 0),
        ]:
            col = [baseline]
            if label in ("constant_background", "background_and_return"):
                col.append(response(0, np.ones_like(source)))
            if lifetime is not None:
                col.append(response(lifetime, source))
            X = np.array(col).T
            # Shared OH-production amplitude absorbs gas calibration. Wall return is one-sided.
            norm = np.linalg.norm(X[train], axis=0)
            p = nnls(X[train] / norm, y[train])[0] / norm
            for name, m in [
                ("first_LED_fit", train),
                ("second_LED_test", lamp2),
            ]:
                scores.append(
                    dict(
                        acid_scale=scale,
                        scenario=label,
                        reservoir_seconds=lifetime,
                        period=name,
                        n=m.sum(),
                        gas_amplitude=p[0],
                        return_amplitude_cm3_s=p[-1]
                        if lifetime is not None
                        else 0.0,
                        rmse_cm3=np.mean(((X @ p - y)[m]) ** 2) ** 0.5,
                        mean_return_cm3=(X[m, -1] * p[-1]).mean()
                        if lifetime is not None
                        else 0.0,
                    )
                )
    write("LED_sulfur.csv", scores)


if __name__ == "__main__":
    main()
