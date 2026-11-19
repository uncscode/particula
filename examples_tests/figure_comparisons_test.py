"""Keep five-series legends distinguishable on the published GPU figures."""

import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
from scripts.gpu_studies.population_figures import population_figures

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "website/results"


def assert_five_styles(lines):
    """Every labeled series has a unique color, stroke and marker combination."""
    assert len(lines) == 5
    assert (
        len(
            {
                (line.get_color(), line.get_linestyle(), line.get_marker())
                for line in lines
            }
        )
        == 5
    )


def test_survival_cohort_styles():
    """The interval estimates reuse their cohort's curve color and marker."""
    source = ROOT / "docs/examples/gpu/prepared/gpu_lucky_few.py"
    data = json.loads((RESULTS / "gpu-lucky-few/data.json").read_text())
    fig = runpy.run_path(str(source))["plot_results"](data)
    try:
        lines = fig.axes[0].lines
        assert_five_styles(lines)
        for line, interval in zip(lines, fig.axes[1].containers, strict=True):
            point = interval.lines[0]
            assert point.get_color() == line.get_color()
            assert point.get_marker() == line.get_marker()
    finally:
        plt.close(fig)


def test_charge_class_styles():
    """The charge legend and each condition panel use the same five keys."""
    report = json.loads(
        (RESULTS / "gpu-environment-sweep/data.json").read_text()
    )
    labels = [str(i) for i in range(len(report["history"][0]["number"]))]
    before = set(plt.get_fignums())
    try:
        population_figures(report, labels)
        figs = [plt.figure(i) for i in set(plt.get_fignums()) - before]
        (charge,) = [
            fig
            for fig in figs
            if fig.get_suptitle().startswith("Charge population")
        ]
        assert_five_styles(charge.legends[0].legend_handles)
        for ax in charge.axes:
            if ax.get_visible():
                assert_five_styles(ax.lines)
    finally:
        for i in set(plt.get_fignums()) - before:
            plt.close(i)
