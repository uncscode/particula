"""Test exchange with a time-varying incoming population at the Tangaroa."""

import json
import platform
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import particula
import warp as wp
from scipy.optimize import lsq_linear, nnls
from scipy.signal import fftconvolve
from scipy.special import expit
from scripts.sea2cloud_events.process_inverse import (
    DX,
    OUTPUT,
    ROOT,
    START_SIZES,
    exposure_table,
    physics,
    read,
)
from scripts.sea2cloud_events.toc import graphical_abstract

COLORS = {
    "constant": "#58656b",
    "replay": "#14778a",
    "ship proxy": "#98813c",
    "pulse": "#a8452b",
}


@wp.kernel
def propagate(
    rate: wp.array(dtype=wp.float64),
    response: wp.array2d(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
):
    """Integrate midpoint cohort contributions to all three receptor bands."""
    i, b = wp.tid()
    s = wp.float64(0)
    for j in range(i):
        s += rate[j] * response[i - j - 1, b]
    out[i, b] = s


def convolve(rate, kernel):
    """Causal convolution at receptor edges, with one initial empty interval."""
    return np.vstack(
        [np.zeros(3), fftconvolve(rate[:, None], kernel, axes=0)[: len(rate)]]
    )


def gpu_counts(rate, kernel):
    """Independent CUDA summation of the age contributions."""
    arr = [
        wp.array(x, dtype=wp.float64, device="cuda:0") for x in (rate, kernel)
    ]
    out = wp.empty((len(rate) + 1, 3), dtype=wp.float64, device="cuda:0")
    wp.launch(propagate, out.shape, [*arr, out], device="cuda:0")
    return out.numpy()


def band_weights(size):
    """Smooth dry diameter cuts used in the preceding counter comparison."""

    def below(c):
        return expit((c - size) / (0.05 * c))

    return np.stack([below(3), below(10), below(25) - below(12)], axis=-1)


def proxy(series, times, start, onset, replay):
    """Use past spectra with a pre-event median after a two-hour gap."""
    pre = series[(series[:, 0] >= start) & (series[:, 0] + 0.25 <= onset)]
    low, mid, high = np.quantile(pre[:, 1], [0.1, 0.5, 0.9])
    look = np.where(
        (times >= onset) & replay, onset - 12 + (times - onset) % 12, times
    )
    idx = np.searchsorted(series[:, 0] + 0.25, look, side="right") - 1
    selected = series[np.maximum(idx, 0)]
    valid = (idx >= 0) & (look - selected[:, 0] - 0.25 <= 2)
    level = np.where(valid, selected[:, 1], mid)
    return np.clip((level - low) / max(high - low, 1e-9), 0, 1), valid


def summed_spectrum(lo, hi):
    """Integrate Figure 3b per-bin number by bin-center selection."""
    bg = read("background")
    cut = bg[(bg[:, 1] >= lo) & (bg[:, 1] < hi)]
    return (
        pd.DataFrame(cut[:, [0, 2]])
        .groupby(0)[1]
        .sum()
        .reset_index()
        .to_numpy()
    )


@dataclass
class Box:
    """Linear cohort responses to replacement, local acid births and growth."""

    time: np.ndarray
    mid: np.ndarray
    source_kernel: np.ndarray
    initial: np.ndarray
    local: np.ndarray
    baseline: float
    tail_kernel: np.ndarray


def make_box(start, finish, dt, tau, vapor, tables, baseline):
    """Construct age-dependent responses at exact output times and midpoints."""
    t = np.linspace(start, finish, round((finish - start) / dt) + 1)
    a = t - start
    mid = (t[:-1] + t[1:]) / 2
    size, h = tables
    x = np.arange(size.shape[1]) * DX

    def kernel(ages, indices):
        responses = []
        for k in indices:
            d = np.interp(vapor * ages, x, size[k])
            loss = np.exp(-ages * (1 / tau + np.interp(vapor * ages, x, h[k])))
            responses.append(loss[:, None] * band_weights(d))
        return np.mean(responses, axis=0)

    acid = read("hourly-acid")
    forcing = np.interp(mid, acid[:, 0], acid[:, 1], left=0)
    rate = 3600 * 3.5e-37 * forcing**4.807
    local = convolve(rate, dt * kernel(mid - start, [0]))
    return Box(
        t,
        mid,
        dt / tau * kernel(mid - start, [2, 3, 4, 5]),
        kernel(a, [2, 3, 4, 5]),
        local,
        baseline,
        dt / tau * kernel(mid - start, [6]),
    )


def sample(t, pred, when, band=1):
    """Evaluate a modeled stock at observation-bin centers."""
    return np.interp(when, t, pred[:, band])


def fit_coefficients(bases, offset, time, obs, target):
    """Fit nonnegative reservoir and initial concentrations in cm-3."""
    columns = np.column_stack([sample(time, b, obs[:, 0]) for b in bases])
    y = target - sample(time, offset, obs[:, 0])
    result = nnls(columns, y)[0]
    if np.any(result > 6000):
        result = lsq_linear(columns, y, bounds=(0, 6000), method="bvls").x
    return result


def compare(box, train, indicators):
    """Fit pre-event coefficients once, then evaluate alternative boundaries."""
    one = np.ones_like(box.mid)
    constant = convolve(one, box.source_kernel)
    y = train[:, 1:].mean(axis=1)
    initial, level = fit_coefficients(
        [box.initial, constant], box.local, box.time, train, y
    )
    fits = {"constant": (initial, level * one)}
    training_proxy = convolve(indicators["ship proxy"], box.source_kernel)
    init, poor, rich = fit_coefficients(
        [box.initial, constant - training_proxy, training_proxy],
        box.local,
        box.time,
        train,
        y,
    )
    for name, indicator in indicators.items():
        fits[name] = (init, poor * (1 - indicator) + rich * indicator)
    return fits


def pulse_fit(box, obs, onset, end, initial, omit_peak=False):
    """Profile one event-conditioned Gaussian pulse and a steady background."""
    use = obs[(obs[:, 0] >= onset) & (obs[:, 0] <= end + 6)]
    y = use[:, 1:].mean(axis=1)
    event = use[:, 0] < end
    peak = np.flatnonzero(event)[np.argmax(use[event, 2])]
    if omit_peak:
        use, y = np.delete(use, peak, axis=0), np.delete(y, peak)
    else:
        y[peak] = use[peak, 2]
    # Hold the independently fitted initial inventory fixed across histories.
    offset = box.local + initial * box.initial
    constant = convolve(np.ones_like(box.mid), box.source_kernel)
    best = None
    for center in np.arange(onset - 3, end + 0.01, 0.25):
        for width in (0.5, 1, 2, 4):
            shape = np.exp(-4 * np.log(2) * ((box.mid - center) / width) ** 2)
            extra = convolve(shape, box.source_kernel)
            coeff = fit_coefficients(
                [constant, extra], offset, box.time, use, y
            )
            residual = (
                sample(
                    box.time,
                    offset + coeff[0] * constant + coeff[1] * extra,
                    use[:, 0],
                )
                - y
            )
            loss = np.mean(residual**2)
            if best is None or loss < best[0]:
                best = (loss, center, width, coeff, shape)
    _, center, width, (base, amplitude), shape = best
    return initial, base + amplitude * shape, (center, width, base, amplitude)


def history(box, init, air):
    """Predict the 1–3, 1–10 and 12–25 nm number stocks."""
    pred = box.local + init * box.initial + convolve(air, box.source_kernel)
    pred[:, 2] += box.baseline
    return pred


def score(box, pred, obs, spec, onset, end):
    """Compare the event peak, decline, bin spans and independent size stock."""
    ts = box.time
    peak_window = (ts >= onset) & (ts <= end)
    evt = obs[(obs[:, 0] >= onset) & (obs[:, 0] < end)]
    post = obs[(obs[:, 0] >= end) & (obs[:, 0] <= end + 6)]

    def err(data):
        p = sample(ts, pred, data[:, 0])
        return np.mean(
            np.maximum.reduce(
                [data[:, 1] - p, p - data[:, 2], np.zeros(len(p))]
            )
        )

    i = np.flatnonzero(peak_window)[np.argmax(pred[peak_window, 1])]
    p = sample(ts, pred, spec[:, 0], 2)
    cross = np.flatnonzero((pred[1:, 1] > 500) & (pred[:-1, 1] <= 500)) + 1
    nearby = cross[(ts[cross] >= onset - 1) & (ts[cross] <= end)]
    return dict(
        peak_cm3=pred[i, 1],
        peak_hour=ts[i],
        at_peak_bin=sample(ts, pred, evt[np.argmax(evt[:, 2]), 0]),
        crossing_hour=ts[nearby[0]] if len(nearby) else np.nan,
        event_span_mae=err(evt),
        post_span_mae=err(post),
        larger_mae=np.mean(abs(p - spec[:, 1])),
        larger_min=p.min(),
        larger_max=p.max(),
    )


def inputs(start, onset):
    """Reuse pre-event Particula rates and the corresponding background."""
    bg = read("background")
    pre = bg[
        (bg[:, 0] >= start)
        & (bg[:, 0] < onset)
        & (bg[:, 1] < 25)
        & (bg[:, 1] >= 12)
    ]
    baseline = pd.DataFrame(pre).groupby(1)[2].median().sum()
    return exposure_table(
        *physics(start, onset, "humid Brownian"),
        initial_sizes=np.r_[START_SIZES, 16.0],
    ), baseline


def indicators(box, onset, series):
    """Return replay and contemporaneous ship-proxy inputs."""
    return {
        name: proxy(series, box.mid, box.time[0], onset, replay)[0]
        for name, replay in (("replay", True), ("ship proxy", False))
    }


def leading_tail(box, params, init, air, spec):
    """Anchor a 16 nm pulse to the first spectrum and test all later stocks."""
    center, width, _, amplitude = params
    small = history(box, init, air)
    rows = []
    for lead in (0, 0.5, 1, 2):
        shape = np.exp(
            -4 * np.log(2) * ((box.mid - center + lead) / width) ** 2
        )
        unit = convolve(amplitude * shape, box.tail_kernel)
        src = sample(box.time, unit, spec[:, 0], 2)
        bg = sample(box.time, small, spec[:, 0], 2)
        ratio = max(0.0, (spec[0, 1] - bg[0]) / src[0])
        for (hour, measured), pred in zip(spec, bg + ratio * src, strict=True):
            rows.append([lead, ratio, hour, measured, pred, hour == spec[0, 0]])
    return rows


def events():
    """Sweep exchange times and vapor levels for both events."""
    obs = read("observations")
    tracer = summed_spectrum(25, 100)
    large = summed_spectrum(12, 25)
    scores, curves, tests, lead_rows = [], [], [], []
    for day, start, onset, end in ((19, -3, 9, 16), (22, 54, 78, 79)):
        train = obs[(obs[:, 0] >= start) & (obs[:, 0] < onset)]
        spec = large[(large[:, 0] >= onset) & (large[:, 0] <= end + 6)]
        tables, baseline = inputs(start, onset)
        for tau in (0.5, 2, 6):
            for vapor in (0, 1, 4):
                box = make_box(
                    start, end + 6, 0.05, tau, vapor, tables, baseline
                )
                fits = compare(box, train, indicators(box, onset, tracer))
                init, air, pulse = pulse_fit(
                    box, obs, onset, end, fits["constant"][0]
                )
                if day == 22:
                    lead_rows.extend(
                        [tau, vapor, *r]
                        for r in leading_tail(box, pulse, init, air, spec)
                    )
                fits["pulse"] = init, air
                hidden_init, hidden_air, hidden = pulse_fit(
                    box, obs, onset, end, init, omit_peak=True
                )
                fits["peak withheld"] = hidden_init, hidden_air
                settings = {"pulse": pulse, "peak withheld": hidden}
                tail = convolve(air - pulse[2], box.tail_kernel)
                fits["pulse + 5% tail"] = init, air
                settings["pulse + 5% tail"] = pulse
                for name, (init, air) in fits.items():
                    pred = history(box, init, air)
                    extra = 0.05 * tail if name == "pulse + 5% tail" else 0.0
                    pred += extra
                    shape = settings.get(name, [np.nan] * 4)
                    scores.append(
                        dict(
                            day=day,
                            tau_h=tau,
                            vapor_million=vapor,
                            history=name,
                            initial=init,
                            incoming_max=air.max(),
                            pulse_center=shape[0],
                            pulse_fwhm=shape[1],
                            base_cm3=shape[2],
                            pulse_amplitude_cm3=shape[3],
                            **score(box, pred, obs, spec, onset, end),
                        )
                    )
                    if tau == 2 and vapor == 1:
                        curves.append(
                            pd.DataFrame(
                                dict(
                                    day=day,
                                    history=name,
                                    hour=box.time,
                                    incoming_cm3=np.r_[air, air[-1]],
                                    n1_3=pred[:, 0],
                                    n1_10=pred[:, 1],
                                    n12_25=pred[:, 2],
                                )
                            )
                        )
                        direct = gpu_counts(air, box.source_kernel)
                        np.testing.assert_allclose(
                            direct, convolve(air, box.source_kernel), atol=1e-8
                        )
                        fine = make_box(
                            start,
                            end + 6,
                            0.025,
                            tau,
                            vapor,
                            tables,
                            baseline,
                        )
                        fine_extra = 0.0
                        if name == "pulse + 5% tail":
                            fine_extra = 0.05 * convolve(
                                np.repeat(air - pulse[2], 2), fine.tail_kernel
                            )
                        # Keep the same piecewise constant input.
                        error = np.max(
                            abs(
                                (
                                    history(fine, init, np.repeat(air, 2))
                                    + fine_extra
                                )[::2]
                                - pred
                            )
                        )
                        np.testing.assert_array_less(error, 0.5)
                        tests.append(
                            [
                                day,
                                name,
                                float(error),
                                float(
                                    np.max(
                                        abs(
                                            direct
                                            - convolve(air, box.source_kernel)
                                        )
                                    )
                                ),
                            ]
                        )
    lead_data = pd.DataFrame(
        lead_rows,
        columns=[
            "tau_h",
            "vapor_million",
            "lead_h",
            "ratio_16nm",
            "hour",
            "observed",
            "predicted",
            "calibration_bin",
        ],
    )
    return pd.DataFrame(scores), pd.concat(curves), tests, lead_data


def rolling():
    """Out-of-sample scores on precisely the earlier 56 held-out bins."""
    obs = read("observations")
    tracer = summed_spectrum(25, 100)
    rows = []
    for onset in range(12, 84, 12):
        start = onset - 12
        train = obs[(obs[:, 0] >= start) & (obs[:, 0] < onset)]
        test = obs[(obs[:, 0] >= onset) & (obs[:, 0] < onset + 12)]
        tables, base = inputs(start, onset)
        box = make_box(start, onset + 12, 0.05, 2, 1, tables, base)
        for name, (init, air) in compare(
            box, train, indicators(box, onset, tracer)
        ).items():
            pred = sample(box.time, history(box, init, air), test[:, 0])
            for (t, low, high), p in zip(test, pred, strict=True):
                rows.append([onset, name, t, low, high, p])
    return pd.DataFrame(
        rows, columns="origin history hour low high prediction".split()
    )


def plot(curves, lead_data):
    """Show incoming histories and the two receptor size bands."""
    fig, axs = plt.subplots(
        3, 2, figsize=(10, 8), layout="constrained", sharex="col"
    )
    obs = read("observations")
    for j, (day, onset, end) in enumerate(((19, 9, 16), (22, 78, 79))):
        lo, hi = onset - 3, end + 6
        for name, color in COLORS.items():
            frame = curves[
                (curves.day == day)
                & (curves.history == name)
                & curves.hour.between(lo, hi)
            ]
            for ax, key in zip(
                axs[:, j], ["incoming_cm3", "n1_10", "n12_25"], strict=True
            ):
                ax.plot(
                    frame.hour - onset,
                    frame[key],
                    color=color,
                    lw=2 if name == "pulse" else 1.5,
                    ls="--" if name == "ship proxy" else "-",
                    label=name,
                )
        selected = obs[(obs[:, 0] >= lo) & (obs[:, 0] <= hi)]
        ax = axs[1, j]
        ax.vlines(
            selected[:, 0] - onset,
            selected[:, 1],
            selected[:, 2],
            color="#17272b",
            lw=2,
        )
        ax.scatter(
            selected[:, 0] - onset,
            selected[:, 1:].mean(axis=1),
            s=8,
            c="#17272b",
            zorder=5,
        )
        sp = summed_spectrum(12, 25)
        sp = sp[(sp[:, 0] >= lo) & (sp[:, 0] <= hi)]
        axs[2, j].scatter(
            sp[:, 0] - onset,
            sp[:, 1],
            s=20,
            c="#17272b",
            zorder=5,
            label="Figure 3b",
        )
        tail = curves[
            (curves.day == day)
            & (curves.history == "pulse + 5% tail")
            & curves.hour.between(lo, hi)
        ]
        axs[2, j].plot(
            tail.hour - onset,
            tail.n12_25,
            ls=":",
            color=COLORS["pulse"],
            lw=2,
            label="Pulse + 5% at 16 nm",
        )
        if j == 1:
            lead = lead_data.query(
                "tau_h==2 and vapor_million==1 and lead_h==1"
            )
            axs[2, j].plot(
                lead.hour - onset,
                lead.predicted,
                "x--",
                ms=4,
                lw=1,
                color="#3b6972",
                label="16 nm one hour earlier",
            )
            handles, labels = axs[2, j].get_legend_handles_labels()
            axs[2, j].legend(
                handles[-3:], labels[-3:], fontsize=7, loc="upper right"
            )
        axs[2, j].set_yscale("log")
        for ax in axs[:, j]:
            ax.axvspan(0, end - onset, color="#d9e5ea", alpha=0.4, zorder=-2)
        axs[0, j].set_title(f"March {day}")
        axs[2, j].set_xlabel("Hours from reported onset")
    for ax, label in zip(
        axs[:, 0],
        ["Incoming number", "Ship 1–10 nm", "Ship 12–25 nm"],
        strict=True,
    ):
        ax.set_ylabel(label + " (cm⁻³)")
    axs[0, 1].legend(fontsize=8)
    fig.savefig(
        ROOT / "docs/images/sea2cloud-events/incoming-population.png", dpi=165
    )
    plt.close(fig)


def plot_leads(lead_data):
    """Show held-out stock errors and their dependence on air exchange."""
    central = lead_data.query("tau_h==2 and vapor_million==1")
    fig, (curve, grid) = plt.subplots(
        1, 2, figsize=(9, 3.7), layout="constrained"
    )
    for lead, color, style in ((0, "#a8452b", "-"), (1, "#14778a", "--")):
        q = central[central.lead_h == lead]
        curve.plot(
            q.hour - 78,
            q.predicted,
            style,
            color=color,
            lw=1.8,
            label="Simultaneous" if lead == 0 else "One hour earlier",
        )
    q = central[central.lead_h == 0]
    curve.scatter(
        q.hour[1:] - 78,
        q.observed[1:],
        s=22,
        color="#17272b",
        zorder=4,
        label="Withheld spectra",
    )
    curve.scatter(
        0,
        q.observed.iloc[0],
        s=48,
        facecolor="white",
        edgecolor="#17272b",
        zorder=4,
        label="Amplitude fit",
    )
    curve.set(
        yscale="log",
        title="March 22 at the ship",
        xlabel="Hours since 06:00 UTC",
        ylabel="12–25 nm number (cm⁻³)",
    )
    curve.legend(fontsize=8)
    test = lead_data.query("vapor_million==1 and not calibration_bin").copy()
    test["error"] = abs(test.predicted - test.observed)
    errors = test.groupby(["tau_h", "lead_h"]).error.mean().unstack()
    grid.imshow(np.log10(errors), cmap="Blues", aspect="auto", vmin=0, vmax=2.1)
    for (i, j), value in np.ndenumerate(errors.to_numpy()):
        grid.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            color="white" if value > 20 else "#142e35",
        )
    grid.set_xticks(
        range(len(errors.columns)), [f"{v:g}" for v in errors.columns]
    )
    grid.set_yticks(range(len(errors)), [f"{v:g}" for v in errors.index])
    grid.set(
        title="Later-stock mean absolute error (cm⁻³)",
        xlabel="16 nm arrival lead (h)",
        ylabel="Exchange time (h)",
    )
    fig.savefig(ROOT / "docs/images/sea2cloud-events/size-lead.png", dpi=165)
    plt.close(fig)


def run():
    """Save the completed cases with numerical and inventory checks."""
    scores, curves, checks, lead_data = events()
    rolls = rolling()
    np.testing.assert_array_equal(rolls.groupby("history").size(), [56] * 3)
    # Exercise the full exchange operator at fixed size and zero scavenging.
    ds = np.repeat(np.r_[START_SIZES, 16.0][:, None], 2, axis=1)
    audit = make_box(0, 24, 0.05, 0.5, 0, (ds, np.zeros_like(ds)), 0)
    receptor = history(audit, 1, np.ones_like(audit.mid)) - audit.local
    equilibrium = band_weights(START_SIZES[2:]).mean(axis=0)
    balance_error = float(np.max(abs(receptor - equilibrium)))
    np.testing.assert_array_less(balance_error, 0.00042)
    for name, df in (
        ("scores", scores),
        ("curves", curves),
        ("rolling", rolls),
        ("size-lead", lead_data),
    ):
        df.to_csv(
            OUTPUT / f"incoming-{name}.csv", index=False, float_format="%.9g"
        )
    (OUTPUT / "incoming-checks.json").write_text(
        json.dumps(
            {
                "device": wp.get_device("cuda:0").name,
                "warp": wp.__version__,
                "python": platform.python_version(),
                "particula_source_version": particula.__version__,
                "distribution_metadata": {
                    k: version(k) for k in ("particula", "numpy", "scipy")
                },
                "sha256": {
                    p.name: sha256(p.read_bytes()).hexdigest()
                    for p in [
                        ROOT / "scripts/sea2cloud_events/incoming.py",
                        ROOT / "scripts/sea2cloud_events/process_inverse.py",
                        *(
                            ROOT / "docs/data/sea2cloud-events" / name
                            for name in (
                                "hourly-acid.csv",
                                "background.csv",
                                "observations.csv",
                            )
                        ),
                    ]
                },
                "lossless_unit_exchange_max_absolute": balance_error,
                "columns": [
                    "march_day",
                    "history",
                    "half_step_max_cm3",
                    "gpu_cpu_max_cm3",
                ],
                "cases": checks,
            },
            indent=2,
        )
        + "\n"
    )
    plot(curves, lead_data)
    graphical_abstract(lead_data)
    plot_leads(lead_data)


if __name__ == "__main__":
    run()
