"""Package the numerical inputs to the published figures or replot them."""

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOREAL = HERE / "boreal"
ARCHIVE = ROOT / "website/public/study-data/boreal/figure-data.zip"
sys.path.extend([str(HERE / "aging"), str(BOREAL)])


def plot_tables():
    """Select plotted cases, columns and summaries from completed campaigns."""
    import exchange
    import overview_figures
    from burst_count_results import METHODS
    from burst_counts import FINE, FRESH
    from redraw_design import TEST

    def read(name):
        return pd.read_csv(BOREAL / name)

    def aging(name):
        return pd.read_csv(HERE / "aging" / f"{name}.csv")

    centers, rich = exchange.size_profile(25000)
    yield (
        "salt-by-size",
        pd.DataFrame({"diameter_nm": centers, "rich_fraction": rich}),
    )
    yield (
        "exchange",
        pd.read_csv(HERE / "exchange.csv").query("diameter_bins == 12"),
    )
    yield "aging-budget", aging("budget").query("split == 'held-out'")
    scores = aging("shape_scores").query("split == 'held-out'")
    yield (
        "aging-shape",
        scores[["seed", "hour", "parcel", "split", "method", "chi_rmse"]],
    )
    worst = (
        scores.query("method == 'mean + SD'").nlargest(1, "chi_rmse").iloc[0]
    )
    profiles = aging("profiles")
    selected = (
        (profiles.seed == worst.seed)
        & (profiles.hour == worst.hour)
        & (profiles.parcel == worst.parcel)
    )
    history = (
        (profiles.seed == 1199)
        & (profiles.parcel == "B")
        & profiles.hour.isin([0, 3, 6])
    )
    yield "aging-profiles", profiles[selected | history]
    curves = aging("curves").query("split == 'held-out'")
    worst = curves.loc[
        (curves.reference - curves.representative).abs().idxmax()
    ]
    selected = (
        (curves.seed == worst.seed)
        & (curves.hour == worst.hour)
        & (curves.parcel == worst.parcel)
    )
    channel = curves.ss_pct.iloc[(curves.ss_pct - 0.2).abs().argmin()]
    yield (
        "aging-curves",
        curves[selected | ((curves.seed == 1199) & (curves.ss_pct == channel))],
    )

    yield "overview-channel", overview_figures.channel_summary()
    for name in [
        "cloud-error",
        "cloud-history",
        "redraw-independent-events",
        "coating-peaks",
        "adaptive-qualification",
        "adaptive-channel-summary",
    ]:
        yield name, read(f"{name}.csv")
    yield (
        "stage-two",
        read("stage-two-histories.csv").query(
            "dt==75 and n==25000 and stream==0"
        )["seed method phase dt n stream ks_pp w1_nm drops_3h".split()],
    )
    yield (
        "coating",
        read("coating-histories.csv").query(
            "method=='local' and budget in [4,8] and stream==0"
            " and dt==75 and n==25000"
        )[
            (
                "seed acid exchange lift method budget stream dt n future_pct"
            ).split()
        ],
    )
    yield (
        "response",
        read("response-qualification.csv").query(
            "method=='redraw' and window=='two short windows'"
        )["seed method window family exceedances status".split()],
    )
    yield (
        "adaptive-edges",
        read("adaptive-edges.csv.gz").query(
            "seed==10111 and start==3600 and parcel==1"
        ),
    )
    yield (
        "redraw-independent",
        read("redraw-independent.csv").query("horizon=='path'"),
    )
    yield (
        "burst-replication",
        read("redraw-burst-replication.csv").query("horizon=='path'"),
    )
    yield (
        "burst-example",
        read("redraw-burst-leads.csv.gz").query(
            "seed==10723 and n==50000 and lead==75"
            " and recipe in ['full','10-16-1']"
        )["seed n lead recipe draw births ref_births".split()],
    )
    memory = read("redraw-memory.csv")
    for stage in ("encode", "decode"):
        memory[stage + "_extra"] = (
            memory[stage + "_peak_rss_bytes"]
            - memory[stage + "_base_rss_bytes"]
        )
    yield (
        "redraw-memory",
        memory[
            (
                "n recipe start stored_array_bytes archive_bytes "
                "encode_extra decode_extra"
            ).split()
        ],
    )
    sample = read("redraw-trajectories.csv.gz")
    sample = sample[
        (sample.seed == TEST[0])
        & (sample.n == 10000)
        & sample.start.isin([1800, 9000])
        & sample.recipe.isin(["full", "10-32-1"])
    ]
    metrics = ["abs_a", "drops_cm3", "births", "organic_transfer"]
    yield (
        "redraw-course",
        sample[
            [
                "seed",
                "n",
                "start",
                "recipe",
                "draw",
                "lead",
                *metrics,
                *["ref_" + x for x in metrics],
            ]
        ],
    )
    yield "burst-histories", read("burst-count-histories.csv").query("dt==75")
    totals = read("burst-count-draws.csv.gz").query("dt==75 and seed==10723")
    yield (
        "burst-counts",
        totals.groupby(["seed", "dt", "draw"], as_index=False)[
            ["original", *METHODS, "full_mean"]
        ].sum(),
    )
    sub = read("burst-substeps.csv.gz")
    sub = sub[(sub.method == "full") & sub.seed.isin([FINE[0], FRESH[1]])]
    yield (
        "burst-steps",
        sub.groupby(
            ["method", "seed", "dt", "draw"], as_index=False
        ).expected.sum(),
    )
    conditional = read("burst-conditional-pairs.csv.gz")
    yield (
        "burst-conditional",
        conditional["seed full_a full_b redraw_a redraw_b".split()],
    )


def pack():
    """Build a deterministic bundle of ordinary CSVs with labeled columns."""
    with ZipFile(ARCHIVE, "w") as archive:
        for name, table in plot_tables():
            info = ZipInfo(name + ".csv")
            archive.writestr(
                info,
                table.to_csv(index=False, float_format="%.15g"),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )


def replot(output):
    """Redraw the study figures without running the simulation."""
    import adaptive_results
    import analyze
    import burst_count_results
    import burst_qualification
    import coating_results
    import exchange
    import figure as cloud
    import overview_figures
    import redraw_results
    import redraw_scale_results as scaling
    import response_results
    import stage_two_results

    output.mkdir(parents=True, exist_ok=True)

    def image(stem):
        return output / f"parcel-{stem}.png"

    with ZipFile(ARCHIVE) as archive:

        def table(name):
            with archive.open(name + ".csv") as stream:
                return pd.read_csv(stream, float_precision="round_trip")

        overview_figures.IMAGES = output
        overview_figures.toc(table("response"))
        overview_figures.row_figure()
        overview_figures.plot_channel(table("overview-channel"))
        profile = table("salt-by-size")
        exchange.figure(
            table("exchange").to_dict("records"),
            (profile.diameter_nm.to_numpy(), profile.rich_fraction.to_numpy()),
            image("composition-exchange"),
        )
        analyze.IMAGE = image("composition-aging")
        profiles, curves = table("aging-profiles"), table("aging-curves")
        analyze.plot(
            16, table("aging-budget"), profiles, table("aging-shape"), curves
        )
        analyze.plot_history(profiles, curves)
        cloud.plot(
            table("cloud-error"),
            table("cloud-history"),
            image("boreal-cloud-error"),
        )
        stage_two_results.plot(table("stage-two"), image("stage-two"))
        coating_results.figure(
            table("coating-peaks"), table("coating"), image("coating-restarts")
        )
        redraw_results.render(
            table("redraw-course"), "10-32-1", image("redraw-course")
        )
        scaling.reliability_figure(
            table("redraw-independent"),
            table("redraw-independent-events"),
            output,
        )
        scaling.failure_figure(
            pd.read_csv(BOREAL / "redraw-failure-matrix.csv"), output
        )
        scaling.burst_figure(
            table("burst-replication"), table("burst-example"), output
        )
        scaling.scaling_figure(table("redraw-memory"), output)
        burst_count_results.plot(
            table("burst-counts"),
            table("burst-histories"),
            table("burst-steps"),
            image("burst-counts"),
        )
        burst_qualification.figure(
            table("burst-conditional"), image("burst-conditional")
        )
        response_results.figure(
            table("response"), image("response-qualification")
        )
        adaptive_results.plot(
            table("adaptive-edges"),
            table("adaptive-qualification"),
            table("adaptive-channel-summary"),
            image("adaptive-grid"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Refresh the CSV bundle from regenerated simulation outputs",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/images")
    args = parser.parse_args()
    if args.pack:
        pack()
    replot(args.output)
