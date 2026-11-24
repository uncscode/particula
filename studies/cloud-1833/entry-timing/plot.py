"""Plot the timing comparison from retained numerical output."""

from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from particula import plotting

plt.style.use("particula.default")

out = Path(sys.argv[1])
data = pd.read_csv(out / "tail.csv")
window = data.archive_h.between(7.5, 11.2)
fig, ax = plotting.subplots(size="wide")
for name, label, color, style in [
    ("observed", "Observed", plt.rcParams["text.color"], "-"),
    (
        "empirical | measured",
        "Entry estimated from measurements",
        plotting.COLORS["teal"],
        "-",
    ),
    (
        "predicted timing, empirical total | measured",
        "Modeled timing, same total entry",
        plotting.COLORS["rust"],
        "--",
    ),
    (
        "empirical timing, predicted total | measured",
        "Estimated timing, modeled total",
        plotting.COLORS["indigo"],
        "-.",
    ),
    (
        "predicted | measured",
        "Modeled timing and total entry",
        plotting.COLORS["berry"],
        ":",
    ),
]:
    ax.plot(
        data.archive_h[window] + 6.5,
        data.loc[window, name],
        style,
        color=color,
        label=label,
        linewidth=2,
    )
for level in [1000, 5000]:
    ax.axhline(level, color="0.65", linewidth=0.8, zorder=0)
ax.set(
    xlabel="Time from run start (hours)",
    ylabel="Particles from 9.7 to 64.7 nm (cm⁻³)",
    ylim=(0, None),
)
plotting.legend(ax, placement="right")
plotting.fit(fig)
destination = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else out / "entry-timing.png"
)
fig.savefig(destination)
plt.close(fig)
