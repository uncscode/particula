"""Reduce a completed replay to the numerical tables underlying the publication."""

import argparse
import gzip
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import pandas as pd

from replot import eddy, cloud, baseline_rows
from check_selected import selected_rows


def export(run, data):
    data.mkdir(parents=True, exist_ok=True)
    campaign = run / "cloud-sources/results"
    old = run / "cloud-eddy/results"
    ledger = campaign / "runs.jsonl.gz"
    if not ledger.exists():
        ledger = campaign / "runs.jsonl"
    cases = json.loads(
        (Path(__file__).parent / "cloud-sources/replay-cases.json").read_text()
    )
    opener = gzip.open if ledger.suffix == ".gz" else open
    with opener(ledger, "rt") as stream:
        selected_rows(map(json.loads, stream), cases).to_csv(
            data / "selected-endpoints.csv", index=False
        )
    # Plate summaries and paired ensemble statistics behind the sensitivity text.
    # Preserve the published values without storing individual fits or GPU checkpoints.
    by_stage = {
        run: [
            "intervals.csv",
            "duration_intervals.csv",
            "landmark_intervals.csv",
            "reference_weighting.csv",
            "blocked_permutation.csv",
            "rank_quartiles.csv",
            "boxes_retained_pulse.csv",
            "renewal_observations.csv",
            "source_exchange.csv",
        ],
        old: ["contrasts.csv", "oracle.json"],
        campaign: [
            "paired.csv",
            "resident-balance.csv",
            "interactions.csv",
            "evidence.csv",
            "air-volume.csv",
        ],
    }
    with ZipFile(data / "supporting-tables.zip", "w") as archive:
        for source, names in by_stage.items():
            for name in names:
                archive.writestr(
                    ZipInfo(name),
                    (source / name).read_bytes(),
                    compress_type=ZIP_DEFLATED,
                )
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)

        def save(name, table):
            table.to_csv(stage / name, index=False)

        def read(name):
            return pd.read_csv(run / name)

        for name in (
            "intervals.csv",
            "paper_reference.csv",
            "renewal_observations.csv",
            "source_exchange.csv",
        ):
            (stage / name).write_bytes((run / name).read_bytes())
        wells = read("well_forecasts.csv.gz")
        fields = [
            f"{law}_{t}"
            for law in ("persistent", "population")
            for t in (0, 0.5, 2, 4, 6, 8, 10)
        ]
        groups = wells.groupby("sample")
        save(
            "well-means.csv",
            groups[fields].mean().assign(entry=groups.size()).reset_index(),
        )
        scores = read("plate_scores.csv").query(
            "variant=='main' and subset=='all' and model=='persistent'"
        )
        save(
            "plate_scores.csv",
            scores[["sample", "variant", "subset", "model", "late"]],
        )
        boxes = read("boxes.csv").query("reset==True")
        save(
            "boxes.csv",
            boxes.groupby(
                ["reset", "sample", "model", "fraction"], as_index=False
            ).late_ice_per_l.mean(),
        )
        history = pd.concat(
            [pd.read_csv(old / f"{stage}_history.csv") for stage in eddy.STAGES]
        )
        save(
            "eddy-history.csv",
            history[history.case.isin(["fresh", "aged", "closed_matched"])][
                ["case", "t_h", "primary", "LWP"]
            ],
        )
        save(
            "contrasts.csv",
            pd.read_csv(old / "contrasts.csv").query("metric=='primary'"),
        )
        fields = pd.read_csv(old / "fields.csv").query("t_s>=14400")
        save(
            "fields.csv",
            fields.groupby("cell", as_index=False)
            .liquid_g_m3.mean()
            .assign(t_s=14400),
        )
        paths = pd.read_csv(old / "tagged_paths.csv.gz")
        ids = [
            sorted(g.id.unique())[g.id.nunique() // 2]
            for _, g in paths.groupby(paths.birth_s < 0)
        ]
        nodes = paths[paths.id.isin(ids) & paths.fate.eq(1)]
        nodes[
            ["birth_s", "id", "fate", "time_s", "z_m", "phase", "first_ice_s"]
        ].to_csv(
            stage / "tagged_paths.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        events = pd.read_csv(old / "freezing_ids.csv")
        save(
            "freezing_ids.csv",
            events[events.id.isin(ids) | events.first_freeze_s.ge(14400)][
                ["first_freeze_s", "source", "x_m", "z_m", "id"]
            ],
        )
        paired = pd.read_csv(campaign / "paired.csv")
        used = paired.grid.eq(10) & (
            (
                paired.metric.eq("primary")
                & paired.comparison.isin(["closed", "old_dust"])
            )
            | (
                paired.metric.eq("fresh_primary")
                & paired.sky.eq("reference")
                & paired.comparison.eq("closed")
                & paired.abundance.eq("air_volume")
            )
        )
        save("paired.csv", paired[used])
        _, _, _, baselines = baseline_rows(paired)
        baselines.to_csv(data / "entrainment-baselines.csv", index=False)
        save("liquid-history.csv", cloud.mean_liquid_history(ledger))
        save(
            "mineral-means.csv",
            cloud.mineral_means(
                cloud.load(ledger).drop_duplicates(cloud.EXPERIMENT)
            ),
        )
        profile = pd.read_csv(campaign / "profiles.csv")
        save(
            "profiles.csv",
            profile[profile.model_z_m.between(0, 600, inclusive="left")][
                ["profile", "model_z_m", "liquid_g_m3"]
            ],
        )
        (stage / "air-volume.csv").write_bytes(
            (campaign / "air-volume.csv").read_bytes()
        )
        with ZipFile(data / "figure-tables.zip", "w") as archive:
            for path in sorted(stage.iterdir()):
                meta = ZipInfo(path.name)
                archive.writestr(
                    meta, path.read_bytes(), compress_type=ZIP_DEFLATED
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        type=Path,
        help="replay directory containing assay tables and both cloud stages",
    )
    parser.add_argument(
        "data", type=Path, help="output directory for compact tables"
    )
    args = parser.parse_args()
    export(args.run, args.data)
