"""Reduce the March 2020 Tangaroa Level 0 counts on their archive UTC clock.

Run from the checkout with Python, NumPy, pandas and matplotlib:
python scripts/sea2cloud_events/level_zero.py RAW_DIRECTORY OUTPUT_DIRECTORY
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from defusedxml import ElementTree

ROOT = Path(__file__).resolve().parent.parent.parent
EPOCH = pd.Timestamp("2020-03-19", tz="UTC")
CPC_BYTES = [118768, 119156, 118914, 118622, 118622, 91316, 119140, 87827]

PACKET_NOTE = """Tangaroa aerosol counter reconstruction, March 2020

PSM: Sellegri, Peltola and Chamba, doi:10.6096/4009. Parallel CPC: Peltola
and Sellegri, doi:10.6096/4011. Tracer timing: ASIT VOC tank products
doi:10.6096/4010 and native VOC product doi:10.25326/y3vz-sp14. Campaign
observations are distributed under CC BY 4.0. Published size-band series and
acid forcing are from Salignat et al., doi:10.5194/egusphere-2026-6891.
Input-audit.json identifies the source files used and their SHA256 digests.

Extract this package. Put the corresponding raw files into one directory.
With Python 3.12, pandas, matplotlib, NumPy, SciPy and defusedxml installed:
python level_zero.py RAW OUT --history ship-history.json

UTC in archive tables follows timestamp labels. In cpc-shift products the
CPC clock is moved to the PSM clock, in psm-shift products the reverse.
The applied offset (minutes) is the maximum non-event rank correlation in
the one-minute clock scan. Preserve both time conventions in subsequent work.
psm-plateaus.csv gives the retained parts of both flow steps. In the pairs
tables, first/last/utc delimit and center the high-flow plateau. Neighboring
low-flow measurements bracket the estimate. early/late select CPC records
one minute earlier/later than the central join. A missing field is unpaired.
cm3 suffix means particles per cubic centimetre. Signed differences retain
negatives. CPC flags 0x40 and 0x80 mark liquid-level and concentration
errors; 0xc0 includes both. The VOC candidate tag is a tracer window. The
four-port log, inlet efficiency corrections and exhaust mask are unverified.
The summary windows are left-closed, right-open. Figure 5a markers and model
curves retain their published time base in every panel.
"""


def psm_steps(path):
    """Summarize contiguous, settled, status-clean portions of flow plateaus."""
    d = pd.read_csv(path, skiprows=1, header=None)
    if d.shape[1] != 23 or not d[20].eq("Not Airmodus CPC").all():
        raise ValueError(f"Unrecognized A11 format in {path.name}")
    t = pd.to_datetime(d[0], format="%d.%m.%Y %H:%M:%S", utc=True)
    if not t.is_monotonic_increasing or not d[2].eq(0).all():
        raise ValueError("Inspect clock order and cutoff calibration")
    duplicates = t.duplicated(keep=False)
    dt = t.diff().dt.total_seconds()
    valid = (
        d[14].eq(1)
        & d[15].eq(1)
        & d[18].eq(1)
        & d[21].eq("0x0000")
        & d[22].eq("0x0000")
        & np.isfinite(d[1])
        & d[1].ge(0)
        & ~duplicates
    )
    mode = pd.Series(
        np.select(
            [d[3].between(0.95, 1.05), d[3].between(0.08, 0.12)], [1, 3], 0
        )
    )
    group = (mode.ne(mode.shift()) | dt.ne(1)).cumsum()
    valid &= mode.ne(0) & (
        t - t.groupby(group).transform("min")
    ).dt.total_seconds().ge(10)
    points = pd.DataFrame(dict(utc=t, signal=d[1], cutoff=mode, group=group))
    steps = (
        points[valid]
        .groupby("group")
        .agg(
            utc=("utc", "median"),
            first=("utc", "min"),
            last=("utc", "max"),
            nominal_nm=("cutoff", "first"),
            seconds=("signal", "count"),
            psm_cm3=("signal", "mean"),
        )
    )
    steps = steps[steps.seconds.ge(30)].copy()
    qc = dict(
        rows=len(d),
        repeated_rows=int(t.duplicated().sum()),
        absent_seconds=int(dt.sub(1).clip(lower=0).sum()),
        usable_seconds=int(steps.seconds.sum()),
        steps=len(steps),
    )
    return steps, qc


def read_cpc(path):
    """Retain the original status word with every minute reading."""
    d = pd.read_csv(path, comment="#", header=None, skipinitialspace=True)
    t = pd.to_datetime(
        d.iloc[:, :6].set_axis(
            ["year", "month", "day", "hour", "minute", "second"], axis=1
        ),
        utc=True,
    )
    if t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError(f"Ambiguous CPC timestamps in {path}")
    if not (np.isfinite(d[6]) & d[6].ge(0)).all():
        raise ValueError(f"Invalid CPC concentration in {path}")
    return pd.DataFrame(dict(cpc_utc=t, cpc_cm3=d[6], flag=d[7]))


def tracer_windows(path):
    """Use only the interiors of contiguous groups of populated VOC scans."""
    with ZipFile(path) as archive:
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    times = []
    for row in sheet.findall(".//{*}row")[1:]:
        cells = {c.get("r").rstrip("0123456789"): c for c in row}
        if not any(
            col not in {"A", "B", "C", "D", "E", "F"}
            and cell.get("t", "n") == "n"
            and cell.findtext("{*}v") is not None
            for col, cell in cells.items()
        ):
            continue
        values = [float(cells[col].findtext("{*}v")) for col in "ABCDEF"]
        times.append(
            pd.Timestamp(*map(int, values[:5]), tz="UTC")
            + pd.Timedelta(seconds=values[5])
        )
    stamps = pd.Series(sorted(times))
    groups = stamps.diff().dt.total_seconds().gt(210).cumsum()
    windows = stamps.groupby(groups).agg(["min", "max", "count"])
    return windows[windows["count"] >= 2]


def assign_tracers(steps, raw):
    """Tag plateaus contained between consecutive scans on one tracer stream."""
    steps["inlet_evidence"] = "unresolved"
    for pattern, tag in [
        ("*EXPCONTROLE*.xlsx", "control tank"),
        ("*EXPO3*.xlsx", "ozone tank"),
        ("Sea2Cloud_PTRMS-VOC_Raw*.xlsx", "VOC candidate"),
    ]:
        (path,) = raw.glob(pattern)
        for start, end, _ in tracer_windows(path).itertuples(index=False):
            mask = steps["first"].ge(start) & steps["last"].le(end)
            if (mask & steps.inlet_evidence.ne("unresolved")).any():
                raise ValueError("Conflicting PTR-MS inlet labels")
            steps.loc[mask, "inlet_evidence"] = tag


def pair_channels(steps, cpc):
    """Keep signed step differences and bounded nearest-minute CPC matches."""
    high = steps[steps.nominal_nm.eq(1)].copy()
    low = steps[steps.nominal_nm.eq(3)][["utc", "psm_cm3", "inlet_evidence"]]
    for direction in ("backward", "forward"):
        neighbor = low.rename(columns={k: f"{direction}_{k}" for k in low})
        high = pd.merge_asof(
            high,
            neighbor,
            left_on="utc",
            right_on=f"{direction}_utc",
            direction=direction,
            tolerance=pd.Timedelta(seconds=75),
        )
    span = high.forward_utc - high.backward_utc
    weight = (high.utc - high.backward_utc) / span
    interp = high.backward_psm_cm3 + weight * (
        high.forward_psm_cm3 - high.backward_psm_cm3
    )
    high["step_difference_cm3"] = high.psm_cm3 - interp
    high["same_tracer_triplet"] = (
        high.inlet_evidence.ne("unresolved")
        & high.inlet_evidence.eq(high.backward_inlet_evidence)
        & high.inlet_evidence.eq(high.forward_inlet_evidence)
    )
    for shift, suffix in [(0, ""), (-60, "_early"), (60, "_late")]:
        high["join_time"] = high.utc + pd.Timedelta(seconds=shift)
        right = cpc.rename(columns={k: k + suffix for k in cpc})
        high = pd.merge_asof(
            high,
            right,
            left_on="join_time",
            right_on="cpc_utc" + suffix,
            direction="nearest",
            tolerance=pd.Timedelta(seconds=30),
        )
        high["difference_cm3" + suffix] = (
            high.psm_cm3 - high["cpc_cm3" + suffix]
        )
    high["nominal_3_10_cm3"] = interp - high.cpc_cm3
    np.testing.assert_allclose(
        high.step_difference_cm3 + high.nominal_3_10_cm3,
        high.difference_cm3.where(interp.notna()),
        rtol=1e-10,
        atol=1e-9,
        equal_nan=True,
    )
    return high.drop(columns="join_time")


def _clock_scan(steps, cpc, out):
    """Estimate relative clock lag on non-event days with clear CPC flags."""
    s = steps[steps.nominal_nm.eq(3)].set_index("utc").psm_cm3
    c = cpc[cpc.flag.eq("0x0")].set_index("cpc_utc").cpc_cm3
    rows = []

    def score(resolution, offsets):
        psm = s.resample(f"{resolution}min").median().rename("psm")
        counter = c.resample(f"{resolution}min").median().rename("cpc")
        for minutes in offsets:
            shifted = counter.shift(minutes, freq="min")
            paired = pd.concat([psm, shifted], axis=1).dropna()
            day = paired.index.day
            for name, mask in [
                ("non-event", day.isin([20, 21, 23, 24, 25, 26])),
                ("event days", day.isin([19, 22])),
            ]:
                matched = paired[mask]
                rows.append(
                    dict(
                        minutes_added_to_cpc=minutes,
                        subset=name,
                        resolution_min=resolution,
                        n=len(matched),
                        rho=matched.psm.corr(matched.cpc, method="spearman"),
                    )
                )

    score(5, range(-1440, 1441, 5))
    coarse = pd.DataFrame(rows).query("subset == 'non-event'")
    best = int(coarse.loc[coarse.rho.idxmax(), "minutes_added_to_cpc"])
    score(1, range(best - 5, best + 6))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "clock-scan.csv", index=False, float_format="%.6f")
    fine = scores.query("subset == 'non-event' and resolution_min == 1")
    lag = int(fine.loc[fine.rho.idxmax(), "minutes_added_to_cpc"])
    # Compare unshifted and aligned clocks on exactly the same PSM samples.
    p = s.resample("5min").median().rename("psm")
    baseline = c.resample("5min").median().rename("baseline")
    shifted = (
        c.shift(lag, freq="min").resample("5min").median().rename("aligned")
    )
    common = pd.concat([p, baseline, shifted], axis=1).dropna()
    controls = []
    for day, group in common.groupby(common.index.day):
        if len(group) >= 10:
            controls.append(
                dict(
                    day=day,
                    n=len(group),
                    raw_rho=group.psm.corr(group.baseline, method="spearman"),
                    aligned_rho=group.psm.corr(
                        group.aligned, method="spearman"
                    ),
                )
            )
    pd.DataFrame(controls).to_csv(
        out / "clock-common-support.csv", index=False, float_format="%.6f"
    )
    return pd.Timedelta(minutes=lag)


def report(raw, out, history):
    """Write matched observations, input provenance and the event comparison."""
    out.mkdir(parents=True, exist_ok=True)
    metadata, plates, counters = {}, [], []
    inputs = sorted(raw.glob("Sea2Cloud_ASIT-PSM_L0_*.dat"))
    if len(inputs) != 9:
        raise ValueError("Expected the nine March 19–27 PSM files")
    for path in inputs:
        block, metadata[path.name] = psm_steps(path)
        plates.append(block)
    for day, size in enumerate(CPC_BYTES, 19):
        (path,) = raw.glob(f"Sea2Cloud_ASIT-CPC_L0_2020-03-{day}T*H*M.csv")
        if path.stat().st_size != size:
            raise ValueError(f"Incomplete CPC file {path}")
        counter = read_cpc(path)
        metadata[path.name] = dict(
            rows=len(counter), flags=counter.flag.value_counts().to_dict()
        )
        counters.append(counter)
        inputs.append(path)
    steps = pd.concat(plates, ignore_index=True)
    assign_tracers(steps, raw)
    cpc = pd.concat(counters, ignore_index=True)
    high = pair_channels(steps, cpc)
    lag = _clock_scan(steps, cpc, out)
    shifted_cpc = cpc.copy()
    shifted_cpc["cpc_utc"] += lag
    aligned = pair_channels(steps, shifted_cpc)
    shifted_steps = steps.copy()
    for col in ["utc", "first", "last"]:
        shifted_steps[col] -= lag
    assign_tracers(shifted_steps, raw)
    other_anchor = pair_channels(shifted_steps, cpc)
    for label, data in [("cpc-shift", aligned), ("psm-shift", other_anchor)]:
        data.to_csv(
            out / f"{label}-pairs.csv", index=False, float_format="%.6f"
        )
    for path in inputs + sorted(raw.glob("*VOC*.xlsx")):
        record = metadata.setdefault(path.name, {})
        record.update(
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    steps.to_csv(out / "psm-plateaus.csv", index=False, float_format="%.6f")
    high.to_csv(out / "psm-cpc-pairs.csv", index=False, float_format="%.6f")
    (out / "input-audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for label, data in [
        ("archive", high),
        ("cpc-shift", aligned),
        ("psm-shift", other_anchor),
    ]:
        event_report(data, out, history, label)

    _bundle(out, history)


def _bundle(out, history):
    """Package the observation tables with the precise model curves used."""
    source = json.loads(history.read_text())
    replay = dict(
        obs=source["obs"],
        acid=source["acid"],
        events=[
            dict(
                start=e["start"],
                end=e["end"],
                cases=[{k: e["cases"][0][k] for k in ("t", "local")}],
            )
            for e in source["events"]
        ],
    )
    with ZipFile(out / "tangaroa-counts.zip", "w", ZIP_DEFLATED) as bundle:
        names = [
            "psm-plateaus.csv",
            "input-audit.json",
            "clock-scan.csv",
            "clock-common-support.csv",
            "psm-cpc-pairs.csv",
            "cpc-shift-pairs.csv",
            "psm-shift-pairs.csv",
            "local-1-3-march-19.csv",
            "local-1-3-march-22.csv",
        ]
        for anchor in ["archive", "cpc-shift", "psm-shift"]:
            names.extend([f"{anchor}-events.png", f"{anchor}-comparison.csv"])
        for name in names:
            bundle.write(out / name, name)
        bundle.write(Path(__file__), "level_zero.py")
        bundle.writestr("ship-history.json", json.dumps(replay))
        bundle.writestr("READ-ME.txt", PACKET_NOTE)


def _ceiling(event_t, acid, n):
    """Integrate two hours of births in the absence of losses."""
    ages = (np.arange(n) + 0.5) * 2 / n
    times = event_t[:, None] - ages
    if times.min() < acid[0, 0] or times.max() > acid[-1, 0]:
        raise ValueError("Acid outside observed support")
    return (7200 / n) * (
        3.5e-37 * np.interp(times, acid[:, 0], acid[:, 1]) ** 4.807
    ).sum(axis=1)


def event_report(high, out, history, label_clock):
    """Compare the signed observations with published bins and model curves."""
    result = json.loads(history.read_text())
    obs = np.array(result["obs"])
    hours = (high.utc - EPOCH).dt.total_seconds() / 3600
    rows = []
    fig, panels = plt.subplots(
        2, 2, figsize=(11, 7.5), sharex="col", layout="constrained"
    )
    for i, event in enumerate(result["events"]):
        start, end = event["start"], event["end"]
        day = 19 + int(start // 24)
        view = hours.between(start - 12, end + 6)
        valid = high.flag.eq("0x0")
        candidate = high.inlet_evidence.eq("VOC candidate")
        ax, detail = panels[:, i]
        for mask, color, label in [
            (valid, "#9aabb2", "All inlets, CPC clear"),
            (
                candidate & high.flag.eq("0x40"),
                "#d47a1b",
                "VOC window, liquid flag",
            ),
            (candidate & valid, "#137468", "VOC window, CPC clear"),
        ]:
            selected = view & mask
            ax.scatter(
                hours[selected] - start,
                high.difference_cm3[selected],
                s=11,
                color=color,
                label=label,
                alpha=0.8,
            )
        o = obs[(obs[:, 0] >= start - 12) & (obs[:, 0] <= end + 6)]
        ax.vlines(o[:, 0] - start, o[:, 1], o[:, 2], color="#632393", lw=2)
        ax.scatter(
            o[:, 0] - start,
            o[:, 2],
            s=9,
            c="#632393",
            label="Figure 5a, bin maximum",
        )
        case = event["cases"][0]
        # Loss-free local births still inside the ideal 1–3 nm window at
        # 1 nm/h. Use five-second midpoint quadrature on the hourly acid field.
        # Keep the known model initialization and exclude unknown acid hours.
        event_t = np.arange(start, end + 1 / 120, 1 / 60)
        acid = np.array(result["acid"])

        bound = _ceiling(event_t, acid, 1440)
        np.testing.assert_allclose(
            bound, _ceiling(event_t, acid, 2880), rtol=1e-4
        )
        detail.plot(
            event_t - start,
            bound,
            color="#163d71",
            lw=2,
            label="Local formation ceiling",
        )
        pd.DataFrame(
            dict(hours_since_onset=event_t - start, local_1_3_ceiling_cm3=bound)
        ).to_csv(
            out / f"local-1-3-march-{day}.csv", index=False, float_format="%.8g"
        )
        ax.plot(
            np.array(case["t"]) - start,
            case["local"],
            color="#163d71",
            label="Local formation, 1–10 nm",
        )
        ax.axhline(500, color="#444", ls="--", lw=0.8)
        for a in (detail, ax):
            a.axvspan(0, end - start, color="#efd8aa", alpha=0.30, zorder=0)
            a.axhline(0, color="#777", lw=0.5)
        sub = view & high.same_tracer_triplet & candidate
        detail.scatter(
            hours[sub] - start, high.step_difference_cm3[sub], s=14, c="#137468"
        )
        ax.set_yscale("symlog", linthresh=10)
        detail.set_yscale("symlog", linthresh=5)
        ax.set_title(f"March {day}  ·  onset {int(start % 24):02d}:00 UTC")
        detail.set_xlabel("Hours from published onset")
        for period, a, b in [
            ("pre", start - 12, start),
            ("event", start, end),
            ("post", end, end + 6),
        ]:
            window = hours.ge(a) & hours.lt(b)
            for label, m in [
                ("all", high.utc.notna()),
                ("VOC candidate", candidate),
                ("control tank", high.inlet_evidence.eq("control tank")),
                ("ozone tank", high.inlet_evidence.eq("ozone tank")),
            ]:
                for flag in ["0x0", "0x40", "0x80", "0xc0"]:
                    mask = window & m & high.flag.eq(flag)
                    data = high[mask]
                    values = data.difference_cm3
                    robust = values.gt(500)
                    for suffix in ["_early", "_late"]:
                        robust &= data["flag" + suffix].eq(flag) & data[
                            "difference_cm3" + suffix
                        ].gt(500)
                    rows.append(
                        dict(
                            day=day,
                            period=period,
                            inlet=label,
                            cpc_flag=flag,
                            n=len(data),
                            above_500=int(values.gt(500).sum()),
                            timing_robust_above=int(robust.sum()),
                            median_cm3=values.median(),
                            max_cm3=values.max(),
                            max_utc=data.loc[values.idxmax(), "utc"]
                            if len(data)
                            else None,
                        )
                    )
    panels[0, 0].set_ylabel("1–10 nm difference (cm⁻³)")
    panels[1, 0].set_ylabel("1–3 nm step difference (cm⁻³)")
    handles, labels = panels[0, 1].get_legend_handles_labels()
    handles.append(panels[1, 1].get_legend_handles_labels()[0][0])
    labels.append("Local ceiling, 1–3 nm")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    for a in panels.ravel():
        a.set_yticks(
            [
                v
                for v in [-1000, -100, -10, 0, 10, 100, 1000, 5000]
                if a.get_ylim()[0] <= v <= a.get_ylim()[1]
            ]
        )
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    panels[1, 0].set_title("Matched VOC windows only")
    panels[1, 1].set_title("Matched VOC windows only")
    fig.suptitle(
        {
            "archive": "Archive clock labels",
            "cpc-shift": "CPC aligned to the PSM clock",
            "psm-shift": "PSM aligned to the CPC clock",
        }[label_clock]
    )
    fig.savefig(out / f"{label_clock}-events.png", dpi=175)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        out / f"{label_clock}-comparison.csv", index=False, float_format="%.6f"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "website/results/marine-history/data.json",
    )
    args = parser.parse_args()
    report(args.raw, args.out, args.history)
