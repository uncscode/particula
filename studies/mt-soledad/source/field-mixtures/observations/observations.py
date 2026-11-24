"""Build equal-time distributions from archived 12-minute Mt. Soledad growth scans."""

from pathlib import Path
import os
import argparse
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

BLOCK = 21600
SCAN = 720
# May 23 log states 11:20–3:00; the daily event is taken as 11:20–15:00.
EVENTS = [
    ("2023-04-26", "11:20", "14:20", "NaCl"),
    ("2023-05-08", "11:45", "14:40", "AS"),
    ("2023-05-16", "11:40", "16:00", "NaCl"),
    ("2023-05-23", "11:20", "15:00", "AS"),
    ("2023-05-30", "10:20", "14:20", "NaCl"),
    ("2023-06-14", "13:00", "16:00", "AS"),
]
CLOCKS = {"raw_UTC": 0, "raw_minus7": -7, "raw_PDT": 7}
GROWTH_HREF = "https://library.ucsd.edu/dc/object/bb35956742/_7_1.zip/download"


def unix(time_string):
    return int(pd.Timestamp(time_string, tz="UTC").timestamp())


def load_scans(path):
    scans = []
    curves = []
    with zipfile.ZipFile(path) as archive:
        for filename in sorted(archive.namelist()):
            size = int(filename.split("_")[-2])
            with archive.open(filename) as source:
                frame = pd.read_csv(source)
            grid = frame.iloc[0, 3:].to_numpy(float)
            np.testing.assert_allclose(grid, np.linspace(0.5, 2.495, 400), atol=1e-14)
            values = frame.iloc[1:, 3:].to_numpy(float)
            raw = frame.iloc[1:, 0].to_numpy(np.int64) - 2082844800
            assert np.all(np.diff(raw) > 0)
            area = np.trapezoid(values, grid)
            pdf = values / np.where(area > 0, area, np.nan)[:, None]
            mean = np.trapezoid(pdf * grid, grid)
            sd = np.sqrt(np.trapezoid(pdf * (grid - mean[:, None]) ** 2, grid))
            valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
            cdf = cumulative_trapezoid(np.nan_to_num(pdf), grid, initial=0, axis=1)
            medians = np.array(
                [
                    np.interp(0.5, f, grid) if complete else np.nan
                    for f, complete in zip(cdf, valid)
                ]
            )
            modes = np.where(valid, grid[np.argmax(np.nan_to_num(pdf), axis=1)], np.nan)
            flags = pd.DataFrame(
                {
                    "size_nm": size,
                    "raw_time": raw,
                    "author_mean": frame.iloc[1:, 1].to_numpy(),
                    "author_spread": frame.iloc[1:, 2].to_numpy(),
                    "area": area,
                    "scan_gf_mean": mean,
                    "scan_gf_sd": sd,
                    "scan_gf_median": medians,
                    "scan_gf_mode": modes,
                    "pdf_complete": valid,
                }
            )
            scans.append(flags)
            curves.append(pdf)
    return pd.concat(scans, ignore_index=True), grid, np.vstack(curves)


def flag_calibrations(obs, out):
    log_events = []
    obs["calibration_day"] = ""
    obs["calibration_core"] = False
    valid = obs.pdf_complete & obs.area.between(0.98, 1.02)
    for date, start, stop, salt in EVENTS:
        t0, t1 = [unix(f"{date} {h}") for h in (start, stop)]
        for off in range(-2, 11):
            lo, hi = t0 + off * 3600, t1 + off * 3600
            core = obs.raw_time.between(lo + SCAN, hi - SCAN)
            for size, sub in obs[valid & core].groupby("size_nm"):
                log_events.append(
                    {
                        "date": date,
                        "salt": salt,
                        "offset_from_written_h": off,
                        "size_nm": size,
                        "scans": len(sub),
                        "median_gf": sub.scan_gf_mean.median(),
                        "median_pdf_sd": sub.scan_gf_sd.median(),
                        "n_author_mean": sub.author_mean.count(),
                    }
                )
        start_time, stop_time = t0 + 7 * 3600, t1 + 7 * 3600
        obs.loc[
            obs.raw_time.between(start_time - SCAN, stop_time + SCAN), "calibration_day"
        ] = date
        obs.loc[
            obs.raw_time.between(start_time + SCAN, stop_time - SCAN),
            "calibration_core",
        ] = True
    pd.DataFrame(log_events).to_csv(out / "calibration-clock.csv", index=False)
    cores = obs[valid & obs.calibration_core].copy()
    cores["salt"] = cores.calibration_day.map({x[0]: x[-1] for x in EVENTS})
    stats = cores.groupby(["salt", "size_nm"]).agg(
        scans=("size_nm", "size"),
        events=("calibration_day", "nunique"),
        median_gf=("scan_gf_mean", "median"),
        median_scan_median=("scan_gf_median", "median"),
        median_scan_mode=("scan_gf_mode", "median"),
        mean_scan_mean=("scan_gf_mean", "mean"),
        median_pdf_sd=("scan_gf_sd", "median"),
        sd_p10=("scan_gf_sd", lambda x: x.quantile(0.1)),
        sd_p90=("scan_gf_sd", lambda x: x.quantile(0.9)),
        gf_p10=("scan_gf_mean", lambda x: x.quantile(0.1)),
        gf_p90=("scan_gf_mean", lambda x: x.quantile(0.9)),
    )
    stats.to_csv(out / "standards.csv")


def summarize_windows(obs, grid, pdf, windows, out):
    results = []
    archive = {size: [] for size in (30, 60, 90)}
    for clock, offset in CLOCKS.items():
        work = obs.loc[obs.good].copy()
        work["time"] = work.raw_time + offset * 3600
        work["utc"] = (work.time // BLOCK) * BLOCK
        for (utc, size), sub in work[work.utc.isin(windows)].groupby(
            ["utc", "size_nm"]
        ):
            dens = pdf[sub.index].mean(axis=0)
            assert abs(np.trapezoid(dens, grid) - 1) < 2e-13
            mean = np.trapezoid(dens * grid, grid)
            sd = np.sqrt(np.trapezoid(dens * (grid - mean) ** 2, grid))
            within = np.sqrt(np.mean(sub.scan_gf_sd**2))
            between = np.std(sub.scan_gf_mean)
            np.testing.assert_allclose(sd**2, within**2 + between**2, rtol=1e-11)
            f = cumulative_trapezoid(dens, grid, initial=0)
            q25, q50, q75 = np.interp([0.25, 0.5, 0.75], f, grid)
            archive[size].append((clock, utc, dens))
            results.append(
                dict(
                    clock=clock,
                    utc=utc,
                    size_nm=size,
                    n_scans=len(sub),
                    n_size_hours=sub.raw_time.floordiv(3600).nunique(),
                    gf_mean=mean,
                    gf_sd=sd,
                    gf_iqr=q75 - q25,
                    gf_p25=q25,
                    gf_median=q50,
                    gf_p75=q75,
                    gf_within_sd=within,
                    gf_between_sd=between,
                    author_gf_mean=sub.author_mean.mean(),
                    author_spread_mean=sub.author_spread.mean(),
                    time_first=sub.time.min(),
                    time_last=sub.time.max(),
                    scan_centroid_utc=sub.time.mean(),
                    gf_below_one=np.trapezoid(dens[grid <= 1], grid[grid <= 1]),
                )
            )
    for size, rows in archive.items():
        np.savez_compressed(
            out / f"pdf{size}.npz",
            clock=np.array([r[0] for r in rows]),
            utc=np.array([r[1] for r in rows]),
            gf=grid,
            density=np.stack([r[2] for r in rows]),
        )
    result = pd.DataFrame(results)
    byblock = result.groupby(["clock", "utc"])
    result["n_sizes"] = byblock.size_nm.transform("nunique")
    result["cross_size_span_min"] = byblock.scan_centroid_utc.transform(
        lambda x: (x.max() - x.min()) / 60
    )
    result.loc[result.n_sizes < 3, "cross_size_span_min"] = np.nan
    result.to_csv(out / "windows.csv", index=False)
    return result


def match_filters(filters, windows, out):
    src = pd.read_csv(filters)
    src = src[src.FilterMask.isin([1, 2, 3])].copy()

    def parse(times):
        return (
            pd.to_datetime(times, format="%m/%d/%Y %H:%M", utc=True)
            .astype("int64")
            .floordiv(10**9)
        )

    src["start_utc"], src["stop_utc"] = parse(src.RealStart), parse(src.RealStop)
    src["cut_nm"] = src.FilterMask.map({1: 1000, 2: 180, 3: 500})
    filters_all = []
    for win in (
        windows[windows.n_sizes == 3].drop_duplicates(["clock", "utc"]).itertuples()
    ):
        selection = src[(src.start_utc <= win.utc) & (src.stop_utc >= win.utc + BLOCK)]
        for filt in selection.itertuples():
            filters_all.append(
                dict(
                    clock=win.clock,
                    utc=win.utc,
                    filter_id=filt.FilterID,
                    cut_nm=filt.cut_nm,
                    start_utc=filt.start_utc,
                    stop_utc=filt.stop_utc,
                    maintenance=filt.MaintenanceFlag,
                    x3OFG=filt.x3OFG,
                    eligible=filt.MaintenanceFlag == 0 and filt.x3OFG == 1,
                )
            )
    covered = pd.DataFrame(filters_all)
    covered.to_csv(out / "filter-all.csv", index=False)
    covered[covered.eligible].to_csv(out / "filter-windows.csv", index=False)


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument(
        "--growth",
        type=Path,
        default=Path(os.environ.get("SOLEDAD_INPUT", "inputs")) / "growth.zip",
    )
    arg.add_argument(
        "--filters",
        type=Path,
        default=Path(os.environ.get("SOLEDAD_INPUT", "inputs")) / "ftir.csv",
    )
    arg.add_argument(
        "--chemistry",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "organic-films/bulk/composition.csv",
    )
    arg.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = arg.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs, grid, pdf = load_scans(a.growth)
    flag_calibrations(obs, a.out)
    obs["good"] = (
        obs.pdf_complete
        & obs.area.between(0.98, 1.02)
        & obs.author_mean.between(0.5, 3)
        & np.isfinite(obs.author_spread)
        & (obs.author_spread >= 0)
        & obs.calibration_day.eq("")
    )
    obs.to_csv(
        a.out / "scan-qc.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.10g",
    )
    windows = summarize_windows(obs, grid, pdf, pd.read_csv(a.chemistry).utc, a.out)
    match_filters(a.filters, windows, a.out)
    checks = {
        "input_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [a.growth, a.filters, a.chemistry]
        },
        "source_url": GROWTH_HREF,
        "published_rows": len(obs),
        "median_abs_difference_author_mean_pdf_mean_good": float(
            (obs.author_mean - obs.scan_gf_mean).abs()[obs.good].median()
        ),
        "good": int(obs.good.sum()),
        "flags_by_size": obs.groupby("size_nm")[
            ["pdf_complete", "calibration_core", "good"]
        ]
        .sum()
        .to_dict("index"),
    }
    (a.out / "qc.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
