"""Render the Manchester channel diagram from paired fits and search rounds."""

import json
from pathlib import Path

import pandas as pd
from scripts.render_literature_tocs import IMAGES, render_svg

HERE = Path(__file__).resolve().parent


def render():
    """Split the worst-error reduction and count improved runs."""
    search = json.loads(
        (HERE.parent / "milestone2/search_summary.json").read_text()
    )
    pairs = pd.read_csv(HERE / "paired_changes.csv")
    joint = pairs.query("step == '0.9375s' and metric == 'joint'")
    refit = sum(
        r["parent_refit_gain_percentage_points"] for r in search["rounds"]
    )
    growth = sum(r["growth_gain_percentage_points"] for r in search["rounds"])
    fraction = refit / (refit + growth)
    width = 270 * fraction
    render_svg(
        HERE / "toc.svg.in",
        IMAGES / "manchester-soa/campaign-performance-toc.svg",
        inherited=search["inherited"],
        final=search["final_channels"],
        gain=f"{refit + growth:.5f}",
        refit_pct=f"{100 * fraction:.0f}",
        channel_pct=f"{100 * (1 - fraction):.0f}",
        refit_width=f"{width:.1f}",
        channel_width=f"{270 - width:.1f}",
        split=f"{15 + width:.1f}",
        improved=(joint.change_pp < 0).sum(),
        runs=len(joint),
    )


if __name__ == "__main__":
    render()
