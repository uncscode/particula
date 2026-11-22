"""Paired uncertainty for the King Sejong activation spectrum."""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scripts.antarctic_ccn.model import BANDS, DATA, make_thresholds, responses
from scripts.antarctic_ccn.recovery import detection_rates

SEED = 12520430
N = 1000
FIT = [0, 2, 3, 4]


def date_weights(dates):
    """Return date multiplicities for the original split and 1,000 resamples."""
    days, idx = np.unique(dates.astype("datetime64[D]"), return_inverse=True)
    rng = np.random.default_rng(SEED)
    result = []
    for use in [np.arange(len(days)) % 4 != 0, np.arange(len(days)) % 4 == 0]:
        n = use.sum()
        counts = np.zeros((N + 1, len(days)))
        counts[0, use] = 1
        counts[1:, use] = rng.multinomial(n, np.full(n, 1 / n), size=N)
        result.append(counts[:, idx])
    return result


def summary(values):
    """Summarize the nominal estimate and its percentile interval."""
    low, high = np.quantile(values[1:], [0.025, 0.975])
    return dict(mean=values[0], lo95=low, hi95=high)


def fitted_fraction(salt, delta, observed, train):
    """Fit one bounded coating fraction on the four training channels."""
    x = delta[:, FIT]
    return np.clip(
        (train @ ((observed[:, FIT] - salt[:, FIT]) * x).sum(1))
        / (train @ np.square(x).sum(1)),
        0,
        1,
    )


def _band_contrasts(bands, w, salt_bias):
    sizes = []
    for (lo, hi), band in zip(BANDS, bands, strict=True):
        change = np.diff(w @ band)
        for j, k in enumerate(["rise24", "rise46"]):
            base = np.diff(salt_bias)[:, j]
            for key, v in [
                ("change", change[:, j]),
                (
                    "change_abs_bias",
                    abs(base + change[:, j]) - abs(base),
                ),
            ]:
                sizes.append(
                    dict(
                        lo=lo,
                        hi=hi,
                        metric=k,
                        contrast=key,
                        **summary(v),
                    )
                )
    return sizes


def calculate():
    """Simulate the measured calendar and refit each training-date draw."""
    with np.load(DATA / "inputs.npz") as src:
        data = dict(src)
    observed = data["measured"]
    xu, distributed, salt, bands, _, _ = responses(
        data, make_thresholds(data, data["sc"])
    )
    train, test = date_weights(data["dates"])
    delta = bands.sum(0)
    fraction = fitted_fraction(salt, delta, observed, train)
    signals = {
        "measured": observed,
        "Xu": xu,
        "distributed": distributed,
        "salt": salt,
        "maximum": salt + delta,
        "film_common": salt + fraction[0] * delta,
    }
    spectra, contrasts = [], []
    for sample, counts in [("all", train + test), ("held-out", test)]:
        w = counts / counts.sum(1, keepdims=True)
        hours = counts[0] > 0
        salt_bias = w @ (salt - observed)
        for name, signal in signals.items():
            if name == "film_common":
                pred = w @ salt + fraction[:, None] * (w @ delta)
            else:
                pred = w @ signal
            bias = pred - w @ observed
            quantities = np.column_stack([pred, np.diff(pred)[:, :2]])
            hourly = np.column_stack(
                [signal[hours], np.diff(signal[hours])[:, :2]]
            )
            bias_metrics = np.column_stack([bias, np.diff(bias)[:, :2]])
            ref = np.column_stack([salt_bias, np.diff(salt_bias)[:, :2]])
            for j, metric in enumerate(
                ["C2", "C4", "C6", "C8", "C10", "rise24", "rise46"]
            ):
                p10, p90 = np.quantile(hourly[:, j], [0.1, 0.9])
                spectra.append(
                    dict(
                        sample=sample,
                        model=name,
                        metric=metric,
                        hourly_p10=p10,
                        hourly_p90=p90,
                        **summary(quantities[:, j]),
                    )
                )
                if sample == "held-out":
                    for key, metric_values in [
                        ("bias", bias_metrics[:, j]),
                        (
                            "change_abs_bias",
                            abs(bias_metrics[:, j]) - abs(ref[:, j]),
                        ),
                    ]:
                        contrasts.append(
                            dict(
                                model=name,
                                metric=metric,
                                contrast=key,
                                **summary(metric_values),
                            )
                        )
    sizes = _band_contrasts(
        bands,
        test / test.sum(1, keepdims=True),
        (test / test.sum(1, keepdims=True)) @ (salt - observed),
    )
    # The four disjoint fractions are fitted independently of 0.4%.
    x = bands[:, :, FIT].transpose(1, 2, 0)
    y = (observed - salt)[:, FIT]
    count_zero = 0
    for weights in train[1:]:
        use = weights > 0
        mult = np.sqrt(weights[use, None])
        solution = lsq_linear(
            (x[use] * mult[:, :, None]).reshape(-1, 4),
            (y[use] * mult).ravel(),
            bounds=(0, 1),
            tol=1e-12,
        )
        count_zero += solution.x[2] < 1e-6
    weights = test / test.sum(1, keepdims=True)
    full = (weights @ delta)[:, 1]
    missing = (weights @ (observed - salt))[:, 1]
    effects = pd.read_csv(DATA / "attribution-ranges.csv")
    effects_by_model = {
        ("salt", "unexplained"): missing,
        ("film_common", "uplift"): fraction * full,
        ("maximum", "uplift"): full,
        ("film_common", "unexplained"): missing - fraction * full,
    }
    for i, r in effects.iterrows():
        stats = summary(effects_by_model[(r.model, r.outcome)])
        effects.loc[i, ["point", "sampling_lo", "sampling_hi"]] = list(
            stats.values()
        )
    return dict(
        spectra=spectra,
        bands=sizes,
        contrasts=contrasts,
        zero_band_refits=int(count_zero),
        seed=SEED,
        envelope=pd.read_csv(DATA / "calibration-ranges.csv").to_dict(
            "records"
        ),
        effects=effects.to_dict("records"),
        recovery=detection_rates(),
    )


def export(report):
    """Save the compact plotted intervals for readers and the TOC."""
    dest = DATA.parents[2] / "website/public/study-data/antarctic-ccn"
    dest.mkdir(exist_ok=True, parents=True)
    for key, file in [
        ("spectra", "plot-spreads"),
        ("bands", "size-spreads"),
        ("contrasts", "paired-bias"),
    ]:
        pd.DataFrame(report[key]).to_csv(
            dest / f"{file}.csv", index=False, float_format="%.10g"
        )


if __name__ == "__main__":
    export(calculate())
