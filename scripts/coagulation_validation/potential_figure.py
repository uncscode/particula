"""Replot dielectric pair energies and encounter rates for the theory guide."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    get_van_der_waals_kernel_via_system_state,
)
from particula.util.constants import BOLTZMANN_CONSTANT
from scripts.coagulation_validation.profiles import potential_profile


def plot():
    """Compute energy and rate at matching temperatures, radii and charges."""
    t = 298.0
    r = np.array([45.0, 75.0]) * 1e-9
    mass = 4 * np.pi / 3 * 2160 * r**3
    hamaker = 8.9e-20
    hard = get_van_der_waals_kernel_via_system_state(
        r, mass, [0, 0], t, 101325.0, 0.0, 1.0
    )[0, 1]
    gap = np.geomspace(0.35, 500, 350)
    x = r.sum() / (r.sum() + gap * 1e-9)
    fig, (energy, rates) = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.8, 1]},
    )
    for k, (z, color) in enumerate(
        [
            ((0, 0), "#43505c"),
            ((1, 0), "#008078"),
            ((2, 2), "#b45423"),
            ((1, -1), "#844985"),
        ]
    ):
        _, u = potential_profile(
            r[0] / r.sum(),
            z,
            hamaker / (t * BOLTZMANN_CONSTANT),
            COULOMB_LENGTH_T / (t * r.sum()),
            5.9,
        )
        value = (
            get_van_der_waals_kernel_via_system_state(
                r, mass, z, t, 101325.0, hamaker, 5.9
            )[0, 1]
            / hard
        )
        label = " / ".join(f"{i:+d}" if i else "0" for i in z)
        energy.plot(gap, u(x), label=label, color=color)
        rates.barh(k, value, color=color)
        rates.text(value + 0.04, k, f"{value:.2f}", va="center", fontsize=9)
    energy.axhline(0, color=".6", lw=0.8)
    energy.set(
        xscale="log",
        ylim=(-3, 1.5),
        xlabel="Surface gap (nm)",
        ylabel="Pair energy / kT",
    )
    energy.legend(title="Charges (e)", ncol=2, fontsize=8)
    rates.set(
        yticks=range(4),
        yticklabels=["0 / 0", "+1 / 0", "+2 / +2", "+1 / −1"],
        xlabel="K / hard-sphere K",
    )
    rates.set_xlim(0, rates.get_xlim()[1] * 1.15)
    rates.axvline(1, color=".4", lw=1, ls=":")
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/images/coagulation/van-der-waals-potentials.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    plot()
