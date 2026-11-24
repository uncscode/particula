"""Plot the timing comparison from retained numerical output."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path(sys.argv[1])
data = pd.read_csv(out / "tail.csv")
window = data.archive_h.between(7.5, 11.2)
fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
for name, label, color, style in [
    ("observed", "Observed", "#222222", "-"),
    ("empirical | measured", "Entry estimated from measurements", "#197863", "-"),
    ("predicted timing, empirical total | measured", "Modeled timing, same total entry", "#bc5821", "--"),
    ("empirical timing, predicted total | measured", "Estimated timing, modeled total", "#3d6c9c", "-."),
    ("predicted | measured", "Modeled timing and total entry", "#764b99", ":"),
]:
    ax.plot(data.archive_h[window]+6.5, data.loc[window, name], style, color=color, label=label, linewidth=2)
for level in [1000, 5000]:
    ax.axhline(level, color="0.65", linewidth=.8, zorder=0)
ax.set(xlabel="Time from run start (hours)", ylabel="Particles from 9.7 to 64.7 nm (cm⁻³)", ylim=(0,None))
ax.legend(frameon=False, fontsize=9)
destination = Path(sys.argv[2]) if len(sys.argv) > 2 else out / "entry-timing.png"
fig.savefig(destination, dpi=150)
