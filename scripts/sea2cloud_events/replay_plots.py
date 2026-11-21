"""Show ship-track forcing and the predicted particle response."""

import matplotlib.pyplot as plt
import numpy as np


def plot(report):
    """Show pre-event and event concentrations for both dates."""
    fig, axes = plt.subplots(
        2, 2, figsize=(8.4, 5.9), layout="constrained", sharex="col"
    )
    obs = np.array(report["obs"])
    acid = np.array(report["acid"])
    for j, event in enumerate(report["events"]):
        row = event["cases"][0]
        start = event["start"]
        t = np.array(row["t"]) - start
        a, b = axes[:, j]
        a.plot(acid[:, 0] - start, acid[:, 1] / 1e6, color="#b45c27")
        a.set_title("March " + ("19" if j == 0 else "22"))
        a.set_ylabel("Ship-track HNO₃\n(10⁶ molecules cm⁻³)")
        local = np.array(row["local"])
        inc = np.array(row["incoming"])
        b.plot(t, local, label="Local", color="#008677")
        b.plot(t, inc, label="Incoming", color="#9358a1")
        b.plot(t, inc + local, label="Combined", color="#184663", ls="--")
        x = obs[:, 0] - start
        b.vlines(
            x,
            obs[:, 1],
            np.maximum(obs[:, 1] + 12, obs[:, 2]),
            color="#222",
            label="Figure 5a range",
        )
        b.axhline(500, color=".5", ls=":", lw=1)
        for ax in (a, b):
            ax.axvspan(0, event["end"] - start, color="gray", alpha=0.12)
            ax.set_xlim(-12, event["end"] - start + 6)
        b.set(
            xlabel="Hours from reported onset",
            ylabel="1–10 nm particles (cm⁻³)",
            ylim=(-50, 2550 if j == 0 else 1000),
        )
    axes[1, 0].legend(fontsize=7, ncol=2)
    return fig
