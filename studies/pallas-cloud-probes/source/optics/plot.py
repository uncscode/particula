"""Plot simulated acquisition and band assignment from the verified response table."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from particula import plotting


def render(curves, image):
    with plt.style.context("particula.default"):
        fig, axes = plotting.subplots(
            1, 3, size="square", sharex=True, sharey=True
        )
        for ax, field, title in zip(
            axes,
            ["trigger", "detect", "band13"],
            ["Triggered", "Registered", "Assigned to 6–13 µm"],
            strict=True,
        ):
            ax.axvspan(6, 13, color="0.92", zorder=0, linewidth=0)
            for probe, color, native_nm in [
                ("F", plotting.COLORS["teal"], 633),
                ("I", plotting.COLORS["rust"], 660),
            ]:
                for nm in (633, 660):
                    curve = curves[(curves.probe == probe) & (curves.nm == nm)]
                    label = "FSSP" if probe == "F" else "ICEMET"
                    ax.plot(
                        curve.true_um,
                        curve[field],
                        color=color,
                        linestyle="-" if nm == native_nm else "--",
                        label=f"{label} {nm} nm",
                    )
            ax.set_title(title)
            ax.set_xticks([2, 6, 10, 13, 18])
            ax.set_xlim(1, 18)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_ylim(-0.045, 1.065)
            ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        axes[0].set_ylabel("Fraction of encounters")
        axes[1].set_xlabel("True droplet diameter (µm)")
        plotting.legend(
            axes[0], placement="inside", loc="lower right", fontsize=9
        )
        plotting.fit(fig)
        fig.savefig(image, dpi=300)
        plt.close(fig)


def run(out):
    frame = pd.read_csv(out / "refined-curves.csv.gz")
    fields = ["probe", "nm", "true_um", "trigger", "detect", "band13"]
    curves = frame.query(
        "case == 'nominal' and calibration == 'frozen' and true_um <= 18"
    )
    csv_file = out / "detection-efficiency.csv"
    curves[fields].to_csv(csv_file, index=False)
    render(pd.read_csv(csv_file), out / "detection-efficiency.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?")
    parser.add_argument(
        "--csv", type=Path, help="Render an existing response table"
    )
    parser.add_argument("--output", type=Path, help="PNG destination for --csv")
    args = parser.parse_args()
    if args.csv and args.output and args.results is None:
        render(pd.read_csv(args.csv), args.output)
    elif args.results and not (args.csv or args.output):
        run(args.results)
    else:
        parser.error(
            "Pass a replay optics directory, or --csv and --output together."
        )
