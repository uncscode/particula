"""Replay ship-track acid histories through a pre-event Brownian sink."""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scripts.sea2cloud_events.figure_inputs import ROOT
from scripts.sea2cloud_events.study import CAL_LIMIT, EXPONENT, rate_table


def read(name):
    """Read a published-figure reconstruction."""
    return np.loadtxt(ROOT / (name + ".csv"), delimiter=",", skiprows=1)


def trajectory(time, acid, diameter, hazard, growth, initial):
    """Propagate local births and a separate incoming size distribution."""
    dt = time[1] - time[0]
    age = time - time[0]
    births = 3600 * dt * 3.5e-37 * ((acid[1:] + acid[:-1]) / 2) ** EXPONENT
    cumulative_birth = np.r_[0, births.cumsum()]

    def cohort(age_h, start):
        size = start + growth * age_h
        integral = cumulative_trapezoid(
            np.interp(size, diameter, hazard), age_h * 3600, initial=0
        )
        return size, np.exp(-integral)

    def imported(lo, hi):
        total = np.zeros_like(age)
        for start in (1.5, 3.0, 6.0, 8.0):
            size, surv = cohort(age, start)
            total += 0.25 * surv * (size >= lo) * (size < hi)
        return total

    # A birth is represented at the midpoint of its integration interval.
    ages = np.r_[0, age[1:] - dt / 2]
    size, surv = cohort(ages, 1.0)

    def local(lo, hi):
        weights = (surv * (size >= lo) * (size < hi))[1:]
        return np.r_[0, np.convolve(births, weights)[: len(age) - 1]]

    small = local(1, 10)
    inc = imported(1, 10)
    resident = local(12, 25) + initial * imported(12, 25)
    # Crossing the 10-nm boundary is a cumulative flux, whereas observations
    # above the cutoff are a stock. Save each measure separately.
    lifetime = 9 / growth
    passed = np.interp(
        age - lifetime, age, cumulative_birth, left=0
    ) * np.interp(lifetime, ages, surv)
    for start in (1.5, 3.0, 6.0, 8.0):
        _, s = cohort(age, start)
        leave = (10 - start) / growth
        passed += 0.25 * initial * (age >= leave) * np.interp(leave, age, s)
    if np.any(small > cumulative_birth + 1e-7) or np.any(
        passed > initial + cumulative_birth + 1e-7
    ):
        raise ValueError("Invalid population accounting")
    return small, inc, resident, passed


def evaluate(start, end, opt, acid_table, obs, spectra, d, rate):
    """Score pre-specified controls on the published event and its aftermath."""
    growth, bg_factor, initial, missing, dt = opt
    t = np.arange(start - 12, end + 6 + dt / 2, dt)
    acid = np.interp(t, acid_table[:, 0], acid_table[:, 1])
    acid[t < 0] = missing
    pre = (spectra[:, 0] >= start - 12) & (spectra[:, 0] < start)
    bins = np.unique(spectra[:, 1])
    n = np.median(spectra[pre, 2].reshape(-1, len(bins)), axis=0)
    hazard = bg_factor * n @ rate * 1e6
    local, incoming, resident, passed = trajectory(
        t, acid, d, hazard, growth, initial
    )
    event = (t >= start - 1e-7) & (t <= end + 1e-7)
    o = obs[(obs[:, 0] >= start) & (obs[:, 0] <= end)]
    peak_ix = np.argmax(o[:, 2])
    target_t, peak = o[peak_ix, 0], o[peak_ix, 2]
    results = {}
    for name, series in (
        ("local", local),
        ("incoming", initial * incoming),
        ("combined", local + initial * incoming),
    ):
        crossings = (
            np.flatnonzero((series[1:] > 500) & (series[:-1] <= 500)) + 1
        )
        # Score the specified event with a two-hour onset margin.
        near = crossings[(t[crossings] >= start - 2) & (t[crossings] <= end)]
        peak_index = np.flatnonzero(event)[np.argmax(series[event])]
        post = np.flatnonzero((t > t[peak_index]) & (series < 500))
        results[name] = dict(
            peak=float(series[peak_index]),
            peak_hour=float(t[peak_index]),
            onset=float(t[near[0]]) if len(near) else None,
            below_after_peak=float(t[post[0]])
            if len(post) and series[peak_index] > 500
            else None,
        )
    lf = np.interp(target_t, t, local)
    survival = np.interp(target_t, t, incoming)
    maxt = np.max(acid[t <= target_t])
    multiplier = (peak / lf) ** (1 / EXPONENT) if lf > 0 else None
    selected = np.arange(0, len(t), round(0.25 / dt))
    intensity = (spectra[:, 1] >= 12) & (spectra[:, 1] < 25)
    close = spectra[
        (spectra[:, 0] >= start - 12) & (spectra[:, 0] <= end + 6) & intensity
    ]
    times = np.unique(close[:, 0])
    measured = np.array([close[close[:, 0] == v, 2].sum() for v in times])
    baseline = n[(bins >= 12) & (bins < 25)].sum()
    # Add the independently fixed pre-event background to the cohort excess.
    pred = np.interp(times, t, resident) + baseline
    after = times >= start
    measured_bins = obs[(obs[:, 0] >= start - 12) & (obs[:, 0] <= end + 6)]
    local_at_bins = np.interp(measured_bins[:, 0], t, local)
    residual = np.maximum(0, measured_bins[:, 1:] - local_at_bins[:, None])
    return dict(
        incoming_required_at_ship=np.column_stack(
            (measured_bins[:, 0], residual)
        ).tolist(),
        growth=growth,
        bg_factor=bg_factor,
        initial=initial,
        missing=missing,
        dt=dt,
        background_cm3=n.sum(),
        background_bins=n.tolist(),
        diameter_nm=bins.tolist(),
        results=results,
        peak_input_cm3=max(0, peak - lf) / survival if survival > 0 else None,
        acid_factor=multiplier,
        scaled_within_calibration=bool(
            multiplier and maxt * multiplier <= CAL_LIMIT
        ),
        t=t[selected].tolist(),
        local=local[selected].tolist(),
        incoming=(initial * incoming[selected]).tolist(),
        residence_12_25=resident[selected].tolist(),
        transfer_10=passed[selected].tolist(),
        transit_at_end=float(np.interp(end, t, passed)),
        measured_12_25=list(
            zip(times.tolist(), measured.tolist(), strict=True)
        ),
        background_12_25=baseline,
        smps_rmse=float(np.sqrt(np.mean((pred[after] - measured[after]) ** 2)))
        if after.any()
        else None,
    )


def simulate():
    """Freeze background before detection and compare competing histories."""
    spectra = read("background")
    obs = read("observations")
    acid = read("hourly-acid")
    d = np.linspace(1, 25, 481)
    rate = np.array([rate_table(b, d) for b in np.unique(spectra[:, 1])])
    events = []
    for start, end in ((9.0, 16.0), (78.0, 79.0)):
        base = (1.0, 1.0, 350.0, 0.0, 1 / 60)
        options = [base, (1.0, 1.0, 350.0, 0.0, 1 / 120)]
        for growth in (0.25, 1.0, 4.0):
            for factor in (0.5, 2.0):
                for initial in (40.0, 1306.0):
                    options.append((growth, factor, initial, 0.0, 1 / 60))
        if start == 9:
            options.append((1.0, 1.0, 350.0, CAL_LIMIT, 1 / 60))
        if start == 9:
            options.append((0.25, 0.5, 1306.0, CAL_LIMIT, 1 / 60))
        else:
            # Last resolved Figure 5a bin strictly before initialization.
            previous = obs[obs[:, 0] < start - 12][-1]
            for count in previous[1:]:
                options.append((0.25, 1.0, float(count), 0.0, 1 / 60))
        rows = [
            evaluate(start, end, opt, acid, obs, spectra, d, rate)
            for opt in options
        ]
        for offset in (-5e5, 5e5):
            perturbed = acid.copy()
            perturbed[:, 1] = np.maximum(0.0, acid[:, 1] + offset)
            variant = evaluate(
                start, end, base, perturbed, obs, spectra, d, rate
            )
            variant["acid_offset"] = offset
            rows.append(variant)
        rows.append(
            evaluate(
                start,
                end,
                (1.0, 15.0, 350.0, 0.0, 1 / 60),
                acid,
                obs,
                spectra,
                d,
                rate,
            )
        )
        for k in ("peak", "peak_hour"):
            np.testing.assert_allclose(
                rows[0]["results"]["local"][k],
                rows[1]["results"]["local"][k],
                rtol=0.015,
                atol=0.02,
            )
        track = read("track")
        subset = track[(track[:, 0] >= start - 12) & (track[:, 0] <= end)]
        events.append(
            dict(start=start, end=end, cases=rows, track=subset.tolist())
        )
    return dict(events=events, obs=obs.tolist(), acid=acid.tolist())
