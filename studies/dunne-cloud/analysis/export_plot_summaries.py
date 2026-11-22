"""Export only the numerical series and references drawn in the eight figures."""

from pathlib import Path
import argparse
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "monte-carlo"
LINES = [366, 332, 334, 344, 336, 338]
QS = ["p05", "median", "p95"]


def read(name):
    return pd.read_csv(MC / (name + ".csv"), float_precision="round_trip")


def export(out):
    def save(name, frame):
        frame.to_csv(out / (name + ".csv"), index=False)

    def stack(name, frames):
        save(name, pd.concat(frames, ignore_index=True))

    def warm(name):
        return read(name).query('selection=="all" and band=="≥9.7 nm"')

    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "neutral_33_43rh.csv").set_index("source_csv_line")

    def condition(line):
        row = raw.loc[line]
        label = f"{row.temperature_K:g} K · {row['H2SO4_cm-3'] / 1e6:g}"
        if line == 366:
            label += " · lower acid"
        if line == 332:
            label += " · higher acid"
        return label

    src = read("source-bounds")
    src["condition"] = src.line.map(condition)
    src["observed"] = src.line.map(raw["J_1.7_cm-3_s-1"])
    save(
        "formation-overview", src.loc[src.line == 336].assign(ammonia_pptv=0.38)
    )
    hold = read("holdout-bounds").query(
        'sample=="neutral" and shape=="linear" and mode=="neutral"'
    )
    stack(
        "formation-bounds",
        [
            src.assign(quantity="formation_rate_cm-3_s-1"),
            hold[["split", "topology", *QS]].assign(
                quantity="holdout_error_factor"
            ),
        ],
    )
    clusters = []
    for name, metric in [
        ("cluster_first_passage", "pentamer_probability"),
        ("cluster_dimer_summary", "model_over_observation"),
    ]:
        f = read(name)
        f = f.loc[(f.quantity == metric) & (f.hamaker_J > 0)].copy()
        if metric == "pentamer_probability":
            f = f[f.case.isin(LINES)].copy()
            f["condition"] = f.case.map(condition)
        else:
            f = f[
                f.case.isin(
                    ["CLOUD5_1", "CLOUD5_2", "CLOUD5_3", "flowtube_265K"]
                )
            ].copy()
            f["condition"] = f.case.map(
                dict(
                    CLOUD5_1="7.3 pptv",
                    CLOUD5_2="4.4 pptv",
                    CLOUD5_3="2.3 pptv",
                    flowtube_265K="265 K",
                )
            )
        clusters.append(
            f[
                [
                    "case",
                    "condition",
                    "chemical",
                    "hamaker_J",
                    "quantity",
                    "q05",
                    "q50",
                    "q95",
                ]
            ]
        )
    stack("cluster-bounds", clusters)
    keys = ["line", "law"]
    survival = read("transport-survival-summary")[
        keys + ["survival10_" + q for q in QS]
    ]
    flux = read("transport-flux-summary").query('source=="full"')[
        keys + ["J10_" + q for q in QS]
    ]
    pop = read("transport-population-summary").query('source=="full"')[
        keys + ["zero10"]
    ]
    runs = read("transport-population").query(
        'source=="full" and dt_s==30 and window_end_h==18'
    )
    volumes = (
        runs.groupby(keys)
        .sample_volume_cm3.agg(volume_min_cm3="min", volume_max_cm3="max")
        .reset_index()
    )
    transport = (
        survival.merge(flux, on=keys)
        .merge(pop, on=keys)
        .merge(volumes, on=keys)
    )
    save(
        "transport-bounds",
        transport.assign(condition=transport.line.map(condition)),
    )
    scan = (
        read("scan-bounds")
        .query(
            'shape=="linear" and split in ["complete","223_low"] and scenario in ["new_common","new_differential"]'
        )
        .copy()
    )
    scan["observed"] = (
        raw.loc[332, "J_1.7_cm-3_s-1"] / raw.loc[366, "J_1.7_cm-3_s-1"]
    )
    archive = read("archive-scores").query(
        'law=="S1_precise" and subset in ["figure45","all","primary","neutral","ionized"]'
    )
    r2 = (
        archive.groupby("subset")
        .log_r2.quantile([0.05, 0.5, 0.95])
        .unstack()
        .set_axis(QS, axis=1)
        .reset_index()
    )
    nominal = json.loads((ROOT / "results/reference_metrics.json").read_text())
    nom = {
        k: nominal["S1_precise_" + s]["log_r2"]
        for k, s in [
            ("all", "all"),
            ("primary", "rh"),
            ("neutral", "neutral"),
            ("ionized", "ionized"),
        ]
    }
    r2["nominal"] = r2.subset.map(dict(nom, figure45=0.8953))
    stack(
        "design-and-reference",
        [
            scan.drop(columns="shape").assign(quantity="acid_step_rate_ratio"),
            r2.assign(quantity="log_rate_R2"),
        ],
    )
    curves = read("warm-bands").query(
        'band=="≥9.7 nm" and family in ["gas_matched","measured"] and (hour<=6.0001 or hour>=6.36)'
    )
    # Both models share the measured two-minute chamber series.
    obs = curves.pivot(index="hour", columns="family", values="observed")
    assert obs.gas_matched.equals(obs.measured)
    curve = curves.pivot(index="hour", columns="family", values=QS)
    curve.columns = [f"{fam}_{q}" for q, fam in curve.columns]
    save("warm-ensemble", curve.assign(observed=obs.measured).reset_index())
    errors = warm("warm-paired").query(
        'period in ["peak","later"] and family in ["gas_matched","gas_absolute","ozone","peroxide","both","bulk","dimer","measured_clear","gas_matched_clear"]'
    )
    save("warm-paired-errors", errors[["period", "family", "comparator", *QS]])
    scores = warm("warm-scores").query(
        'family in ["gas_transfer","finite_hard","vdw_fixed","empirical","measured","extended_attraction","ammoniation"]'
    )
    peak = scores.query('period=="peak"')[
        ["family", "observed_max", *["max_" + q for q in QS]]
    ].rename(columns={"max_" + q: q for q in QS})
    later = scores.query('period=="later"')[["family", *QS]]
    stack(
        "warm-growth-controls",
        [
            peak.assign(quantity="peak_count_cm-3"),
            later.assign(quantity="later_RMSE_cm-3"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    export(parser.parse_args().out)
