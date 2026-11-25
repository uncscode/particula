"""Draw the published figures from compact numerical tables, without running CUDA."""

from pathlib import Path
import argparse
import zipfile
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "docs/data/lund-freezing"
IMAGES = ROOT.parents[1] / "docs/images/lund-freezing"
sys.path.extend([str(ROOT / "cloud-eddy"), str(ROOT / "cloud-sources")])
import analyze as eddy
import figures as assay
import plots as cloud


def baseline_rows(paired):
    """Show the absolute levels alongside both paired treatment contrasts."""
    names = {
        "reference": "Reference eddy",
        "isdac": "ISDAC profile",
        "mpace": "M-PACE profile",
    }
    data = paired.query(
        "source == 'MineralDust' and grid == 10 and abundance == 'equal' and metric == 'primary'"
    )
    ref = data[data.comparison == "closed"].set_index("sky").loc[list(names)]
    old = data[data.comparison == "old_dust"].set_index("sky").loc[list(names)]
    np.testing.assert_array_equal(ref.n, 8)
    np.testing.assert_array_equal(old.n, 8)
    np.testing.assert_allclose(ref.value, old.value)
    for control in (old, ref):
        np.testing.assert_allclose(
            control.value - control.control, control.delta
        )
    df = pd.DataFrame(
        {
            "cloud": names.values(),
            "fresh": ref.value.to_numpy(),
            "open_survivor": old.control.to_numpy(),
            "closed": ref.control.to_numpy(),
        }
    )
    for label, control in [
        ("fresh_minus_open_survivor", old),
        ("fresh_minus_closed", ref),
    ]:
        for stat in ("delta", "lo", "hi"):
            df[f"{label}_{stat}"] = control[stat].to_numpy()
    return names, ref, old, df


def baselines(folder, paired):
    names, ref, old, _ = baseline_rows(paired)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    teal, rust = colors[:2]
    ink = plt.rcParams["text.color"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 3.65), layout="constrained", sharey=True
    )
    y = np.arange(len(names))
    for x, label, col, mark, shift in [
        (ref.control, "Closed top", rust, "s", 0.17),
        (old.control, "Open, survivor-like dust", teal, "^", 0),
        (ref.value, "Open, fresh dust", ink, "o", -0.17),
    ]:
        left.plot(
            x,
            y + shift,
            linestyle="none",
            marker=mark,
            color=col,
            ms=7,
            label=label,
        )
    for v, lab, col, mark, shift in [
        (ref, "Fresh minus closed top", rust, "s", 0.09),
        (old, "Fresh minus survivor-like inflow", teal, "^", -0.09),
    ]:
        right.errorbar(
            v.delta,
            y + shift,
            xerr=[v.delta - v.lo, v.hi - v.delta],
            fmt=mark,
            color=col,
            ms=7,
            capsize=3,
            label=lab,
        )
    left.set(
        yticks=y,
        yticklabels=list(names.values()),
        ylim=(2.45, -0.45),
        xlim=(0, 0.25),
        title="Freezing over six hours",
        xlabel="Primary freezing events per litre",
    )
    right.axvline(0, color=ink, lw=1)
    right.set(
        xlim=(-0.045, 0.085),
        title="Effect of admitting fresh dust",
        xlabel="Change in events per litre",
    )
    for ax in (left, right):
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.29), fontsize=9, ncol=1
        )
    fig.savefig(folder / "entrainment-baselines.png", dpi=200)
    plt.close(fig)


def render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        with zipfile.ZipFile(data / "figure-tables.zip") as archive:
            for name in archive.namelist():
                if Path(name).name != name:
                    raise ValueError(
                        f"Expected a flat table archive, got {name}"
                    )
                (work / name).write_bytes(archive.read(name))
        paired = pd.read_csv(work / "paired.csv")
        with plt.style.context("particula.default"):
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
            assay.render(work)
        with plt.style.context("particula.default"):
            eddy.plot(work, pd.read_csv(work / "eddy-history.csv"))
        with plt.style.context("particula.default"):
            cloud.supply_figure(work, paired)
            cloud.source_figure(work, paired.query("abundance == 'equal'"))
            cloud.water_figure(work, paired, None, None)
            cloud.mineral_figure(work, None)
            baselines(work, paired)
        cloud.toc(work, paired)
        for image in sorted(work.glob("*.png")):
            if image.name != "toc.png":
                shutil.copyfile(image, output / image.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=IMAGES)
    args = parser.parse_args()
    render(args.data, args.output)
