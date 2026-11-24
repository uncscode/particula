"""Match published and Particula spectra to CLOUD diameter bins and sample times."""

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PACKET = Path(os.environ.get("CLOUD_PACKET", PACKAGE)).resolve()
RESULTS = Path(os.environ.get("CLOUD_RESULTS", HERE)).resolve()
FIGURES = Path(
    os.environ.get(
        "CLOUD_FIGURES", PACKAGE.parents[1] / "docs/images/cloud-1833"
    )
).resolve()
MODELS = ["Vapor transfer", "Hard sphere", "Attraction"]
PERIODS = {"training": (1, 7), "excimer_led": (9, 19), "mixed": (21, 25)}
with np.load(PACKET / "spectra.npz") as archive:
    time, diameter, edges, observed, stage = [
        archive[k] for k in ("time", "diameter", "edges", "observed", "stage")
    ]
widths = np.diff(edges)
tail = np.maximum(0, edges[1:] - np.maximum(edges[:-1], np.log10(9.7)))
missing = pd.read_csv(PACKET / "forcing.csv").small_missing.to_numpy()


def author_bins(z, panel, target):
    with z.open(f"archive/{panel}.dat") as f:
        lines = (
            line for line in f if line.strip() and not line.startswith(b"%")
        )
        times = np.fromstring(next(lines), sep=" ")
        centers = np.fromstring(next(lines), sep=" ")
        mids = (centers[:-1] + centers[1:]) / 2
        bins = np.r_[2 * centers[0] - mids[0], mids, 2 * centers[-1] - mids[-1]]
        overlap = np.maximum(
            0,
            np.minimum(edges[1:, None], bins[1:])
            - np.maximum(edges[:-1, None], bins[:-1]),
        )
        np.testing.assert_allclose(overlap.sum(axis=1), widths)
        wtail = np.maximum(
            0,
            np.minimum(bins[1:], edges[-1])
            - np.maximum(bins[:-1], np.log10(9.7)),
        )
        densities = []
        for raw in lines:
            logs = np.fromstring(raw, sep=" ")
            assert np.isfinite(logs).all() and logs.size == len(times)
            densities.append(
                np.interp(
                    target,
                    times,
                    np.where(logs > 0, 10**logs, 0.0),
                    left=np.nan,
                    right=np.nan,
                )
            )
        return (overlap @ densities / widths[:, None]).T, wtail @ densities


def diagnostics(pred, upper, idx):
    truth = observed[idx]
    a, b = pred[:, 4:] * widths[4:], truth[:, 4:] * widths[4:]
    positive = (a.sum(axis=1) > 0) & (b.sum(axis=1) > 0)
    shape = np.full(len(idx), np.nan)
    shape[positive] = 0.5 * np.abs(
        a[positive] / a[positive].sum(1)[:, None]
        - b[positive] / b[positive].sum(1)[:, None]
    ).sum(1)
    vals = {"shape": shape}
    for floor in (1, 10):
        vals[f"logerr{floor}"] = (
            np.abs(np.log10((floor + pred[:, 4:]) / (floor + truth[:, 4:])))
            @ widths[4:]
            / widths[4:].sum()
        )
    for val in vals.values():
        val[missing[idx]] = np.nan
    obs_tail = truth @ tail
    return pd.DataFrame(
        dict(
            archive_h=time[idx],
            paper_h=time[idx] + 6.5,
            stage=stage[idx],
            observed_tail=obs_tail,
            model_tail=upper,
            tail_abs=np.abs(upper - obs_tail),
            **vals,
        )
    )


def main():
    records = []
    fine = {}
    archive = (
        Path(os.environ.get("CLOUD_INPUTS", PACKAGE / "inputs"))
        / "donahue2026-figure-data.zip"
    )
    with archive.open("rb") as handle:
        if (
            hashlib.file_digest(handle, "sha256").hexdigest()
            != "5f6f48d653a989a57b633472908ebbddffad10a014db73032dc9f940912d4c85"
        ):
            raise ValueError("CLOUD figure archive checksum mismatch")
    with ZipFile(archive) as z:
        for panel, segment in [
            ("fig09a", "training"),
            ("fig10b", "excimer_led"),
        ]:
            model, upper = author_bins(z, panel, time)
            fine[panel] = model
            first, last = PERIODS[segment]
            idx = np.flatnonzero((stage >= first) & (stage <= last))
            records.append(
                diagnostics(model[idx], upper[idx], idx).assign(
                    model="Donahue fit",
                    cells=0,
                    segment=segment,
                    init=np.arange(idx.size) == 0,
                )
            )
            if panel == "fig10b":
                # Match the independent Fig. 10a measurement panel to Fig. 6a.
                _, measured_tail = author_bins(z, "fig10a", time)
                overlap = np.isfinite(measured_tail)
                assert np.count_nonzero(overlap) > 300
                np.testing.assert_allclose(
                    measured_tail[overlap], observed[overlap] @ tail, rtol=1e-6
                )
    for n in [192, 288]:
        with np.load(RESULTS / f"predictions-{n}.npz") as a:
            for seg, array in a.items():
                first, last = PERIODS[seg.removesuffix("_hold")]
                idx = np.flatnonzero((stage >= first) & (stage <= last))
                for pid, name in enumerate(MODELS):
                    records.append(
                        diagnostics(array[pid], array[pid] @ tail, idx).assign(
                            model=name,
                            cells=n,
                            segment=seg,
                            init=np.arange(len(idx)) == 0,
                        )
                    )
    rows = pd.concat(records, ignore_index=True)
    rows.to_csv(RESULTS / "matched-series.csv.gz", index=False)
    np.savez_compressed(RESULTS / "published-spectra.npz", **fine)
    scored = rows.loc[~rows.init].copy()
    summaries = []
    for (model, cells, segment), group in scored.groupby(
        ["model", "cells", "segment"]
    ):
        for label, grp in list(group.groupby("stage")) + [
            (
                999,
                group[
                    group.stage.between(9, 17)
                    if segment.startswith("excimer")
                    else np.full(len(group), True)
                ],
            )
        ]:
            if grp.empty:
                continue
            summaries.append(
                dict(
                    model=model,
                    cells=cells,
                    segment=segment,
                    stage=label,
                    tail_n=grp.tail_abs.count(),
                    spectra_n=grp.logerr1.count(),
                    shape_n=grp["shape"].count(),
                    shape_pct=100 * grp["shape"].mean(),
                    factor=10 ** grp.logerr1.mean(),
                    factor_floor10=10 ** grp.logerr10.mean(),
                    mae_cm3=grp.tail_abs.mean(),
                    observed_cm3=grp.loc[
                        grp.model_tail.notna(), "observed_tail"
                    ].mean(),
                    predicted_cm3=grp.model_tail.mean(),
                )
            )
    pd.DataFrame(summaries).to_csv(RESULTS / "stage-scores.csv", index=False)
    rng = np.random.default_rng(1833)
    blocks = []
    for seg in ["excimer_led", "mixed"]:
        group = scored[(scored.cells == 288) & (scored.segment == seg)]
        if seg == "excimer_led":
            group = group[group.stage <= 17]
        paired = group.pivot(
            index=["stage", "archive_h"], columns="model", values="logerr1"
        ).dropna()
        diff = paired["Vapor transfer"] - paired["Attraction"]
        agg = diff.groupby(level="stage").agg(["sum", "count"])
        draws = rng.integers(0, len(agg), size=(20000, len(agg)))
        ratios = 10 ** (
            agg["sum"].to_numpy()[draws].sum(1)
            / agg["count"].to_numpy()[draws].sum(1)
        )
        blocks.append(
            dict(
                segment=seg,
                vapor_error_factor_over_attraction=10 ** diff.mean(),
                low=np.quantile(ratios, 0.025),
                high=np.quantile(ratios, 0.975),
            )
        )
    (RESULTS / "stage-resampling.json").write_text(
        json.dumps(blocks, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
