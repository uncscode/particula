"""Compare held-out 12-hour windows and summarize the full Tangaroa record."""

from zipfile import ZipFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.sea2cloud_events.process_inverse import (
    OUTPUT,
    ROOT,
    Model,
    exposure_table,
    interval_error,
    optimize,
    physics,
    read,
)

ORIGIN = pd.Timestamp("2020-03-19", tz="UTC")


def archive_series():
    """Reduce each clock separately and retain sample counts."""
    hourly, daily = [], []
    with ZipFile(OUTPUT / "tangaroa-counts.zip") as archive:
        for anchor in ("cpc-shift", "psm-shift"):
            with archive.open(anchor + "-pairs.csv") as stream:
                q = pd.read_csv(stream)
            q["utc"] = pd.to_datetime(q.utc, format="mixed", utc=True)
            q["period"] = q.utc.dt.floor("30min")
            q["hour"] = (q.utc - ORIGIN).dt.total_seconds() / 3600
            for date, g in q.groupby(q.utc.dt.date):
                candidate = g[
                    g.same_tracer_triplet & g.inlet_evidence.eq("VOC candidate")
                ]
                clear = candidate[candidate.flag.eq("0x0")]
                eligible = clear.difference_cm3.between(
                    0, np.inf
                ) & clear.step_difference_cm3.between(0, clear.difference_cm3)
                daily.append(
                    dict(
                        anchor=anchor,
                        date=date,
                        plateaus=len(g),
                        candidate=len(candidate),
                        cpc_matched=candidate.difference_cm3.count(),
                        clear=len(clear),
                        nested=eligible.sum(),
                        negative_1_10=(clear.difference_cm3 < 0).sum(),
                        above500=(clear.difference_cm3 > 500).sum(),
                        above500_nested=(
                            (clear.difference_cm3 > 500) & eligible
                        ).sum(),
                        median_1_3=candidate.step_difference_cm3.median(),
                        median_clear_1_10=clear.difference_cm3.median(),
                    )
                )
            for (period, tag), group in q[q.same_tracer_triplet].groupby(
                ["period", "inlet_evidence"]
            ):
                for band, column, rows in (
                    ("1-3", "step_difference_cm3", group),
                    (
                        "1-10 clear",
                        "difference_cm3",
                        group[group.flag.eq("0x0")],
                    ),
                    (
                        "1-10 flagged",
                        "difference_cm3",
                        group[group.flag.notna() & group.flag.ne("0x0")],
                    ),
                ):
                    values = rows[column].dropna()
                    if len(values):
                        hourly.append(
                            dict(
                                anchor=anchor,
                                period_utc=period,
                                stream=tag,
                                band=band,
                                median_hour=rows.loc[
                                    values.index, "hour"
                                ].median(),
                                count=len(values),
                                minimum=values.min(),
                                median=values.median(),
                                maximum=values.max(),
                            )
                        )
    return pd.DataFrame(hourly), pd.DataFrame(daily)


def rolling(half):
    """Fit the preceding twelve hours and evaluate the following twelve."""
    obs = read("observations")
    acid = read("hourly-acid")
    end = acid[-1, 0]
    target = half[
        half.stream.eq("VOC candidate") & half.band.isin(["1-3", "1-10 clear"])
    ]
    rows, instrument, parameters = [], [], []
    for onset in np.arange(12, end, 12):
        stop, start = onset + 12, onset - 12
        training = obs[(obs[:, 0] >= start) & (obs[:, 0] < onset)]
        test = obs[(obs[:, 0] >= onset) & (obs[:, 0] < stop)]
        raw = target[
            target.median_hour.between(onset, stop, inclusive="left")
        ].copy()
        times = np.r_[test[:, 0], raw.median_hour]
        cuts = np.r_[
            np.full(len(test), 10), np.where(raw.band.eq("1-3"), 3, 10)
        ]
        tables = exposure_table(*physics(start, onset, "humid Brownian"))
        train = Model(
            start, training[:, 0], np.full(len(training), 10), tables, 0.05
        )
        model = Model(start, times, cuts, tables, 0.05)
        y = training[:, 1:].mean(axis=1)
        for form in (
            "acid only",
            "no uptake",
            "1 nm supply",
            "5 nm supply",
            "last midpoint",
        ):
            if form == "last midpoint":
                pred = np.full(len(test), y[-1])
            else:
                p = optimize(train, y, form)
                parameters.append([onset, form, len(training), len(test), *p])
                pred = model.evaluate(p)
                q = raw.copy()
                q["model"], q["start_hour"] = form, onset
                q["pred_cm3"] = pred[len(test) :]
                instrument.append(q)
            for t, val in zip(test, pred[: len(test)], strict=True):
                rows.append([onset, form, *t, val])
    ts = pd.DataFrame(
        rows, columns="start_hour model hour low high pred_cm3".split()
    )
    ts["mid_error"] = ts.pred_cm3 - (ts.low + ts.high) / 2
    ts["span_error"] = interval_error(
        ts.pred_cm3.to_numpy(), ts[["low", "high"]].to_numpy()
    )
    fits = pd.DataFrame(
        parameters,
        columns=(
            "start_hour model train_bins test_bins vapor_million "
            "initial_thousand "
            "source1_hundreds_per_h source5_hundreds_per_h"
        ).split(),
    )
    return ts, pd.concat(instrument, ignore_index=True), fits


def figure(half, obs, matched):
    """Plot held-out windows and the two cruise counter bands."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    ax = axes[0]
    data = obs.drop_duplicates("hour")
    ax.vlines(
        ORIGIN + pd.to_timedelta(data.hour, unit="h"),
        data.low,
        data.high,
        color=".35",
        lw=2,
        label="Figure 5a span",
    )
    ax.scatter(
        ORIGIN + pd.to_timedelta(data.hour, unit="h"),
        (data.low + data.high) / 2,
        s=8,
        color=".35",
    )
    for form, color in [
        ("1 nm supply", "#0072b2"),
        ("last midpoint", "#cc7900"),
    ]:
        for j, (_, g) in enumerate(
            obs[obs.model.eq(form)].groupby("start_hour")
        ):
            ax.plot(
                ORIGIN + pd.to_timedelta(g.hour, unit="h"),
                g.pred_cm3,
                ".-",
                color=color,
                lw=1,
                ms=3,
                label=form if j == 0 else None,
            )
    ax.set_title("Preceding 12 h fit · following 12 h withheld")
    ax.legend(fontsize=8)
    for ax, band in zip(axes[1:], ["1-3", "1-10 clear"], strict=True):
        q = half[
            half.stream.eq("VOC candidate")
            & half.band.eq(band)
            & half.anchor.eq("cpc-shift")
        ]
        t = ORIGIN + pd.to_timedelta(q.median_hour, unit="h")
        ax.scatter(
            t,
            q["median"],
            s=8,
            c="#454545",
            label="Candidate inlet median per sampled half hour",
        )
        model = matched[
            matched.anchor.eq("cpc-shift")
            & matched.band.eq(band)
            & matched.model.eq("1 nm supply")
        ]
        ax.scatter(
            ORIGIN + pd.to_timedelta(model.median_hour, unit="h"),
            model.pred_cm3,
            s=11,
            marker="x",
            c="#0072b2",
            label="Particula at matched times",
        )
        ax.axhline(0, color=".6", lw=0.7)
        ax.set_yscale("symlog", linthresh=50)
        limits = ax.get_ylim()
        ax.set_yticks(
            [
                v
                for v in (-1000, -100, 0, 100, 1000)
                if limits[0] < v < limits[1]
            ]
        )
        ax.set_xlim(ORIGIN, ORIGIN + pd.Timedelta(days=8, hours=8))
        ax.set_title(
            "Signed PSM steps · 1–3 nm"
            if band == "1-3"
            else "Signed PSM minus CPC · 1–10 nm, clear CPC flag"
        )
        ax.legend(fontsize=8)
    for ax in axes:
        ax.set_ylabel("Number (cm⁻³)")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("2020 UTC on the PSM clock")
    fig.savefig(
        ROOT / "docs/images/sea2cloud-events/cruise-series.png", dpi=150
    )
    plt.close(fig)


def run():
    """Save cruise summaries and check held-out coverage."""
    half, daily = archive_series()
    obs, matched, fits = rolling(half)
    np.testing.assert_array_equal(
        daily.groupby("anchor").plateaus.sum(), [5934, 5934]
    )
    np.testing.assert_array_less(obs.hour, obs.start_hour + 12)
    np.testing.assert_equal((obs.hour >= obs.start_hour).all(), True)
    if not np.isfinite(matched.pred_cm3).all() or (matched.pred_cm3 < 0).any():
        raise ValueError("Invalid size-band prediction")
    for key, df in {
        "daily": daily,
        "half-hour": half,
        "predictions": obs,
        "matched": matched,
        "fits": fits,
    }.items():
        df.to_csv(
            OUTPUT / f"cruise-{key}.csv", index=False, float_format="%.8g"
        )
    figure(half, obs, matched)


if __name__ == "__main__":
    run()
