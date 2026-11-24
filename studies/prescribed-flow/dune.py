"""Run the synthetic dune experiment from the worktree (PYTHONPATH=.)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.coupling.dispersion import gaussian_plume, puff_concentration
from particula.coupling.prescribed import DuneFlow, Sounding
from particula.dynamics.parcel import ParcelInitialState
from particula.dynamics.parcel_replay import replay
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_with_drag,
)


def main():
    """Track air, ejected sand and fine airborne dust over a dune."""
    profile = Sounding.from_csv(
        Path(__file__).with_name("synthetic_sounding.csv")
    )
    flow = DuneFlow(profile, height=8, width=18, lid=100)
    times = np.linspace(0, 24, 97)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    x = np.linspace(-62, 160, 500)
    axes[0].fill_between(x, 0, flow.ground(x), color="tan", label="Sand bed")
    template = ParcelInitialState(
        np.array([60e-9]), np.array([1e7]), 0.5, 100000, 290, 0.008
    )
    for base in (2.0, 6.0):
        track, trace = flow.parcel_trace(
            f"air-{base:g}",
            [-60, 0, base],
            times,
            template,
            provenance="synthetic sounding and Gaussian dune",
        )
        wet = replay(trace, max_step=0.025).wet_radius
        axes[0].plot(
            track.xyz[:, 0],
            track.xyz[:, 2],
            label=f"Air at {base:g} m initially",
        )
        print(
            f"{trace.parcel_id}: peak height {trace.height.max():.2f} m, "
            f"wet radius range {wet.min() * 1e9:.3f}–{wet.max() * 1e9:.3f} nm"
        )

    # The finite patch contains 1 g of each size. Each prescribed launch pulse
    # can entrain 0.1 g per size at 1 m AGL, above an illustrative threshold.
    release_x = -15.0
    launch = (release_x, 0, flow.ground(release_x) + 1)
    initial_bed = 2e-3
    bed, airborne, deposited = initial_bed, 0.0, 0.0
    for radius, label in [(5e-6, "Fine dust"), (75e-6, "Sand")]:
        lifted = flow.pickup(
            release_x,
            reference_height=2,
            drag_coefficient=0.004,
            threshold=0.25,
            available_mass=1e-3,
            pulse_mass=1e-4,
        )
        if lifted == 0:
            continue
        bed -= lifted
        # Spherical quartz and representative near-surface air; continuum drag.
        speed = get_particle_settling_velocity_with_drag(
            radius, 2650, 1.2, 1.8e-5, 1.0
        )
        track = flow.track(launch, times, settling=float(speed))
        axes[0].plot(track.xyz[:, 0], track.xyz[:, 2], "--", label=label)
        if track.deposition_time is None:
            airborne += lifted
        else:
            deposited += lifted
            print(f"{label} deposited at {track.deposition_time:.3f} s")
    np.testing.assert_allclose(
        bed + airborne + deposited, initial_bed, rtol=1e-14, atol=0
    )
    print(
        f"Bed / airborne / redeposited: {bed * 1e3:.4f} / "
        f"{airborne * 1e3:.4f} / {deposited * 1e3:.4f} g"
    )
    axes[0].set(
        xlabel="East (m)",
        ylabel="Height above datum (m)",
        title="Wind over an 8 m dune",
    )
    axes[0].legend(fontsize=8)

    y = np.linspace(-100, 100, 401)
    for stability in ("A", "D", "F"):
        c = gaussian_plume(
            300, y, 5, rate=1e-6, wind=5, height=5, stability=stability
        )
        axes[1].plot(y, c * 1e9, label=f"Class {stability}")
    axes[1].set(
        xlabel="Crosswind (m)",
        ylabel="Dust (µg/m³)",
        title="Steady plume at 300 m",
    )
    axes[1].legend()
    x = np.linspace(-30, 175, 401)
    for age in (5, 15, 25):
        c = puff_concentration(
            x,
            0,
            5,
            mass=1e-5,
            age=age,
            wind=5,
            height=5,
            diffusivity=[3, 1, 0.5],
        )
        axes[2].plot(x, c * 1e9, label=f"{age} s")
    top_hat = puff_concentration(
        x,
        0,
        5,
        mass=1e-5,
        age=15,
        wind=5,
        height=5,
        diffusivity=[3, 1, 0.5],
        shape="top_hat",
    )
    axes[2].plot(x, top_hat * 1e9, "--", color="black", label="15 s top-hat")
    axes[2].set(
        xlabel="Downwind (m)",
        ylabel="Dust (µg/m³)",
        title="Single puff along its centerline",
    )
    axes[2].legend()
    output = Path(".artifacts/prescribed-flow/dune.png")
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
