"""Compare frozen replays with Figure 5 and fit the water response."""

import hashlib
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MIX = {14: "CI", 15: "CI", 16: "AC", 18: "AC", 19: "AI", 21: "AI", 24: "ACI"}


def yield_comparison(replay):
    """Match on total VOC consumption inferred from each all-VOC point."""
    points = pd.read_csv(DATA / "figure5.csv")
    points["mass"] = 10 ** (
        1
        + (points.marker_x_px - points.x10_px)
        / (points.x100_px - points.x10_px)
    )
    points["observed"] = (
        0.4
        * (points.y0_px - points.marker_y_px)
        / (points.y0_px - points.y04_px)
    )
    points["consumed"] = points.mass / points.observed
    curves = pd.read_csv(DATA / "single_curves.csv")
    rows = []
    for (fit, run), sim in replay[replay.run.isin(MIX)].groupby(["fit", "run"]):
        consumed = sim[["alpha", "cresol", "isoprene"]].sum(axis=1).to_numpy()
        # Choose the first time a consumption value is reached during a plateau.
        unique, ix = np.unique(consumed, return_index=True)
        selected = points[points.mixture == MIX[run]]
        for p in selected.itertuples():
            row = dict(
                fit=fit,
                run=run,
                mixture=MIX[run],
                consumed=p.consumed,
                observed=p.observed,
                observed_mass=p.mass,
            )
            if unique[0] <= p.consumed <= unique[-1]:
                t = np.interp(p.consumed, unique, sim.seconds.to_numpy()[ix])
                row["seconds"] = t
                corrected = sim.soa + sim.wall
                independent = sim.filter(like="_only_soa_corrected").sum(axis=1)
                row["model"] = np.interp(t, sim.seconds, corrected) / p.consumed
                row["separate_parents"] = (
                    np.interp(t, sim.seconds, independent) / p.consumed
                )
                # Same simulated point using Manchester's alternate denominator.
                denom = np.interp(t, sim.seconds, sim.alpha + sim.cresol)
                row["alpha_consumed"] = np.interp(t, sim.seconds, sim.alpha)
                row["cresol_consumed"] = np.interp(t, sim.seconds, sim.cresol)
                row["isoprene_consumed"] = np.interp(
                    t, sim.seconds, sim.isoprene
                )
                additive = 0.0
                for parent, curve in curves.groupby("parent"):
                    amount = row[f"{parent}_consumed"]
                    if amount > 0:
                        additive += np.interp(
                            amount,
                            curve.consumed,
                            curve.mass,
                            left=np.nan,
                            right=np.nan,
                        )
                row["paper_single_curve_yield"] = additive / p.consumed
                row["model_excluding_isoprene"] = (
                    row["model"] * p.consumed / denom if denom > 0 else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def water_records():
    """Read dimensionless kappa and the instrument conditions."""
    records = []
    for entry in pd.read_csv(DATA / "sources.csv").itertuples():
        raw = (DATA / entry.file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError(f"Source checksum mismatch: {entry.file}")
        channel = entry.file.split("_UMAN_")[1].split(".")[0]
        ccn = channel == "CCN"
        columns = (
            ["seconds", "supersaturation_pct", "observed", "activation_nm"]
            if ccn
            else ["seconds", "observed", "growth_factor"]
        )
        rows = pd.read_csv(
            StringIO(raw.decode().split("&&&&&&&&&&")[1].strip()),
            sep="\t",
            header=None,
            names=columns,
        )
        rows = rows[rows.observed.between(0, 1) & rows.seconds.le(18000)].copy()
        rows["run"], rows["channel"] = entry.run, channel
        records.append(rows)
    return pd.concat(records, ignore_index=True)


def predict(data, k_org, enrichment):
    """Predict water uptake using weighted dry component volumes."""
    organic = data.organic_volume * enrichment
    return (data.seed_k * data.seed_volume + k_org * organic) / (
        data.seed_volume + organic
    )


def calibrate(data, free_enrichment):
    """Fit a shared organic kappa and optional volume multiplier."""
    # Equal weight per experiment so its sampling density cannot select the fit.
    weights = 1 / np.sqrt(data.groupby("run").seconds.transform("count"))

    def residual(x):
        return weights * (
            predict(data, x[0], x[1] if free_enrichment else 1) - data.observed
        )

    if free_enrichment:
        return least_squares(
            residual, [0.05, 1.0], bounds=([0, 0.25], [0.30, 4])
        ).x
    return (least_squares(residual, [0.05], bounds=(0, 0.30)).x[0], 1.0)


def water_comparison(replay):
    """Fit both water probes and test transfer between the two runs."""
    water = water_records()
    rows = []
    estimates = []
    for fit, simulations in replay.groupby("fit"):
        matched = []
        for (run, channel), series in water.groupby(["run", "channel"]):
            anchor_channel = (
                "HTDMA_100nm" if channel == "HTDMA_150nm" else channel
            )
            anchors = water[
                (water.run == run)
                & (water.channel == anchor_channel)
                & (water.seconds < 0)
            ]
            if anchors.empty:
                raise ValueError(
                    f"No pre-light seed measurement for {run}, {channel}"
                )
            subset = series[series.seconds >= 0].copy()
            subset["seed_k"] = anchors.sort_values("seconds").observed.iloc[-1]
            sim = simulations[simulations.run == run]
            subset["seed_volume"] = (
                np.interp(subset.seconds, sim.seconds, sim.seed) / 1.77
            )
            subset["organic_volume"] = (
                np.interp(subset.seconds, sim.seconds, sim.soa) / 1.4
            )
            matched.append(subset)
        matched = pd.concat(matched)
        for channel in ("HTDMA_100nm", "CCN"):
            eligible = matched[matched.channel == channel]
            for train in ("6", "15", "pooled"):
                training = (
                    eligible
                    if train == "pooled"
                    else eligible[eligible.run == int(train)]
                )
                for mode in ("zero_organic_kappa", "volume_mixing", "enriched"):
                    k_org, scale = (
                        (0, 1)
                        if mode == "zero_organic_kappa"
                        else calibrate(training, mode == "enriched")
                    )
                    estimates.append(
                        dict(
                            fit=fit,
                            channel=channel,
                            train=train,
                            mode=mode,
                            k_org=k_org,
                            enrichment=scale,
                        )
                    )
                    evaluation = (
                        eligible
                        if channel == "CCN"
                        else matched[matched.channel.str.startswith("HTDMA")]
                    )
                    rows.append(
                        evaluation.assign(
                            fit=fit,
                            trained_on=train,
                            mode=mode,
                            predicted=predict(evaluation, k_org, scale),
                        )
                    )
    result = pd.concat(rows, ignore_index=True)
    result["squared_error"] = (result.predicted - result.observed) ** 2
    summary = (
        result.groupby(["fit", "trained_on", "mode", "run", "channel"])
        .squared_error.agg(["count", "mean"])
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary.pop("mean"))
    pd.DataFrame(estimates).to_csv(HERE / "water_parameters.csv", index=False)
    result.to_csv(
        HERE / "water_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return summary


if __name__ == "__main__":
    history = pd.read_csv(HERE / "replay.csv.gz")
    yield_comparison(history).to_csv(HERE / "yield_points.csv", index=False)
    water_comparison(history).to_csv(HERE / "water_scores.csv", index=False)
