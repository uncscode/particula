"""Draw the reviewed 3-inch summer parcel comparison from the plotted values."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from particula.plotting.toc import add_toc_logo, save_toc_svg

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "website/public/study-data/bai-limonene"
IMAGES = ROOT / "docs/images/bai-limonene"


def render(data=DATA, output=IMAGES / "seasonal-toc.svg"):
    """Compare the dark and sunlit summer parcels in the same cloud regime."""
    case = pd.read_csv(data / "summer-pair.csv").iloc[0]
    ink, teal, rust = "#243e43", "#135963", "#a44d21"
    with plt.style.context("particula.toc"):
        fig = plt.figure(figsize=(3, 3), dpi=300)
        ax = fig.add_axes(
            [0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1), aspect="equal"
        )
        ax.axis("off")

        def label(x, y, value, size, **kwargs):
            ax.text(
                x,
                y,
                value,
                fontsize=size,
                color=kwargs.pop("color", ink),
                ha=kwargs.pop("ha", "center"),
                va="center",
                **kwargs,
            )

        label(
            0.055, 0.925, "A clean summer cloud", 12.8, weight="bold", ha="left"
        )
        label(
            0.055, 0.854, "72 h aerosol age  ·  0.25 m/s ascent", 7.5, ha="left"
        )
        ax.plot([0.055, 0.945], [0.807, 0.807], color="#a9b2a6", lw=0.7)

        for x, name, color, number, diameter, flux in [
            (
                0.265,
                "Dark-aged",
                teal,
                case.dark_activated_cm3,
                case.dark_median_um,
                case.dark_up_site_wm2,
            ),
            (
                0.735,
                "Sun-aged",
                rust,
                case.activated_cm3,
                case.median_um,
                case.dark_up_site_wm2 + case.delta_up_site_wm2,
            ),
        ]:
            label(x, 0.744, name, 10, color=color, weight="bold")
            # The outlined cloud and rays indicate reflection. The large circle
            # shows one median droplet, with diameter proportional to wet D50.
            outline = [
                (-0.15, -0.024),
                (-0.177, -0.02),
                (-0.175, 0.047),
                (-0.122, 0.046),
                (-0.116, 0.10),
                (-0.049, 0.116),
                (-0.012, 0.071),
                (0.040, 0.107),
                (0.091, 0.079),
                (0.100, 0.047),
                (0.167, 0.057),
                (0.177, -0.010),
                (0.143, -0.024),
            ]
            verts = [(x + dx, 0.549 + dy) for dx, dy in outline]
            path = MplPath(
                verts + [verts[0]],
                [MplPath.MOVETO] + [MplPath.CURVE4] * 12 + [MplPath.CLOSEPOLY],
            )
            ax.add_patch(
                PathPatch(path, facecolor="#e3e9df", edgecolor=ink, lw=0.9)
            )
            for a, b in [
                ((x - 0.10, 0.683), (x - 0.057, 0.629)),
                ((x + 0.052, 0.626), (x + 0.092, 0.684)),
            ]:
                ax.annotate(
                    "",
                    xy=b,
                    xytext=a,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=1.2,
                        color=color,
                        mutation_scale=6,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )
            ax.add_patch(Circle((x, 0.47), diameter * 0.00275, facecolor=color))
            label(x, 0.47, f"{diameter:.1f}", 8.6, color="white", weight="bold")
            label(x, 0.352, f"{number:.1f} droplets/cm³", 7.9)
            label(x, 0.161, f"{flux:.2f}", 14, color=color, weight="bold")
        label(0.5, 0.267, "Reflected sunlight (W/m²)", 9, weight="bold")
        label(0.5, 0.065, "Circles show median wet diameter (µm).", 7)
        add_toc_logo(fig, corner="top right")
        save_toc_svg(
            fig,
            output,
            title="A clean summer cloud after dark or sunlit aerosol aging",
            description=(
                "After 72 hours of aerosol aging, the modeled parcel rises at "
                "0.25 m/s. From dark to sunlit aging, droplet number falls "
                f"from {case.dark_activated_cm3:.1f} to "
                f"{case.activated_cm3:.1f} per cubic centimetre, "
                "median wet diameter rises from "
                f"{case.dark_median_um:.1f} to {case.median_um:.1f} µm, and "
                f"reflected sunlight falls from {case.dark_up_site_wm2:.2f} to "
                f"{case.dark_up_site_wm2 + case.delta_up_site_wm2:.2f} W/m²."
            ),
        )
        plt.close(fig)


if __name__ == "__main__":
    render()
