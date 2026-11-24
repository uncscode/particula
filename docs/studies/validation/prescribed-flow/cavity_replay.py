# %% [markdown]
# # Can Particula reproduce the cavity’s tracer circulation?
#
# A moving lid pulls fluid to the right across the top of a sealed cavity.
# The return flow carries tracer downward and back under the release. We use
# OpenFOAM to spin up a 10 cm square cavity for 0.5 s at Reynolds number 100,
# then freeze the computed velocity and pressure-corrected face fluxes.
# OpenFOAM and Particula independently advance the same square tracer patch
# for another 80 ms. Cell-centre velocity sets the scene. The exported face
# fluxes determine how much volume crosses each cell face.
# [Finite-volume transport and replay](../../../theory/dynamics/prescribed-transport.md).
#
# The download includes the native-order cell, face and concentration arrays
# and their provenance. Run from the repository or extracted download root.
# The independent solver is needed only to regenerate those inputs.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula import plotting
from particula.coupling.cavity_replay import read_cavity
from particula.coupling.grid_inventory import InventoryLayout, Uptake
from particula.coupling.grid_run import GridRun

INPUT = Path("docs/data/prescribed-flow-cavity")


# %% [markdown]
# ## Compare two updates to the same finite-volume balance
#
# The host writes concentration in kg/m³. Multiplying by the physical
# volume of each cell gives Particula its initial gas inventory in kg.
# The cavity has a 1 cm depth even though the flow varies in only two
# dimensions. Both solvers use upwind advection and constant diffusivity
# of 10⁻⁵ m²/s. OpenFOAM treats the update implicitly. Particula uses
# explicit donor-cell transport. We expect some field difference even when
# both conserve mass. Kinematic pressure from the flow calculation stays
# in its native m²/s². The prescribed aerosol background is 293.15 K
# at 101325 Pa with 0.005 kg vapor per kg dry air.
#
# Before running the replay, the exporter measured the change in the host
# solution at half the timestep and on a 32 × 32 mesh, restricted by physical
# volume to 16 × 16 cells. The field error is the integral of the absolute
# concentration difference divided by initial tracer mass. Centroid error
# is Euclidean distance in metres. For each metric, the frozen envelope
# uses the larger of the mesh change and five times the timestep change.
# The leading explicit and implicit Euler errors have opposite signs;
# four times the coarse-to-half-step difference estimates their separation
# in the first-order limit. The factor of five allows higher-order effects.
# This case-specific envelope measures numerical resolution.


# %%
def simulate(device="cpu"):
    """Run passive replay and local uptake at all 41 common records."""
    case = read_cavity(INPUT)
    flow = case.forcing
    grid = flow.grid
    volumes = grid.volume
    reference = case.concentration
    initial = reference[0] * volumes
    total = initial.sum()

    passive = GridRun(
        flow, InventoryLayout(("tracer",)), initial[:, None], device=device
    )
    masses = [initial]
    for _ in flow.frames:
        passive.advance()
        masses.append(passive.snapshot()["state"][:, 0])
    masses = np.asarray(masses)
    host = reference * volumes
    replay = masses / volumes
    field = np.sum(abs(masses - host), axis=1) / total

    def centroid(mass):
        return (mass @ grid.centers) / mass.sum(axis=1)[:, None]

    centroid_host = centroid(host)
    centroid_replay = centroid(masses)
    distance = np.linalg.norm(centroid_replay - centroid_host, axis=1)
    host_budget = host.sum(axis=1) / total - 1
    replay_budget = masses.sum(axis=1) / total - 1
    limits = case.metadata["limits"]
    if (
        field.max() > limits["relative_l1"]
        or distance.max() > limits["centroid_m"]
        or max(abs(host_budget).max(), abs(replay_budget).max())
        > limits["relative_mass"]
    ):
        raise ValueError(
            "passive replay exceeds the frozen host-resolution or mass limit"
        )

    # All cells contain seed aerosol, so uptake occurs everywhere tracer goes.
    rate = 8.0  # inverse seconds
    inventory = np.column_stack(
        (initial, 1e8 * volumes, 1e-7 * volumes, 0 * volumes)
    )
    active = GridRun(
        flow,
        InventoryLayout(("tracer",), ("seed",), ("dry", "condensed")),
        inventory,
        uptake=Uptake("tracer", "seed", "condensed", rate),
        device=device,
    )
    uptake = [[1.0, 0.0]]
    for _ in flow.frames:
        active.advance()
        budget = active.budget()["state"]
        uptake.append((budget[[0, 3]] / total).tolist())
    uptake = np.array(uptake)
    elapsed = flow.times - flow.times[0]
    np.testing.assert_allclose(
        uptake[:, 0], np.exp(-rate * elapsed), atol=1e-12
    )
    np.testing.assert_allclose(uptake.sum(axis=1), 1, atol=1e-12, rtol=0)

    def relative_l1(a, b):
        return (abs(a - b) * volumes).sum(axis=1) / total

    return {
        "solver": case.metadata["build"],
        "device": str(active.device),
        "precision": "float64",
        "image": case.metadata["image"],
        "fixture_sha256": case.metadata["sha256"],
        "shape": list(grid.shape),
        "elapsed_s": elapsed.tolist(),
        "mass_kg": float(total),
        "diffusivity_m2_s": case.metadata["diffusivity_m2_s"],
        "host_step_s": float(flow.times[1] - flow.times[0]),
        "host_scheme": "implicit Euler, upwind",
        "replay_scheme": "explicit Euler, upwind",
        "limits": limits,
        "refinement": case.metadata["refinement"],
        "half_step_l1": relative_l1(reference, case.half_step).tolist(),
        "fine_mesh_l1": relative_l1(
            case.half_step, case.fine_restricted
        ).tolist(),
        "field_l1": field.tolist(),
        "centroid_error_m": distance.tolist(),
        "centroid_host_m": centroid_host[:, :2].tolist(),
        "centroid_replay_m": centroid_replay[:, :2].tolist(),
        "host_relative_mass_change": host_budget.tolist(),
        "replay_relative_mass_change": replay_budget.tolist(),
        "extent_cm": (100 * np.asarray(grid.bounds[:2])).ravel().tolist(),
        "initial_kg_m3": reference[0].reshape(grid.shape)[:, :, 0].T.tolist(),
        "host_kg_m3": reference[-1].reshape(grid.shape)[:, :, 0].T.tolist(),
        "replay_kg_m3": replay[-1].reshape(grid.shape)[:, :, 0].T.tolist(),
        "uptake_s1": rate,
        "uptake_mass_fractions": uptake.tolist(),
    }


# %% [markdown]
# ## Look for a displaced plume and a conserved inventory
#
# Compare the final fields on the same concentration scale. The residual
# uses its own symmetric scale so displaced tracer appears as adjacent
# positive and negative cells. The centroid path gives another view of
# the displacement. Compare the field error to the two independent
# refinements and the host and replay mass changes to their initial masses.


# %%
def plot_results(report):
    """Draw passive tracer fields, numerical errors, budgets and active uptake."""
    fig, axes = plotting.subplots(2, 3)
    time = 1000 * np.asarray(report["elapsed_s"])
    host = 1e6 * np.asarray(report["host_kg_m3"])
    replay = 1e6 * np.asarray(report["replay_kg_m3"])
    residual = replay - host
    for ax, array, title in zip(
        axes[0],
        [host, replay, residual],
        ["OpenFOAM", "Particula", "Particula − OpenFOAM"],
        strict=True,
    ):
        difference = ax is axes[0, 2]
        limit = (
            abs(residual).max() if difference else max(host.max(), replay.max())
        )
        im = ax.imshow(
            array,
            origin="lower",
            extent=report["extent_cm"],
            aspect="equal",
            cmap=plotting.colormap("difference" if difference else "magnitude"),
            vmin=-limit if difference else 0,
            vmax=limit,
            interpolation="nearest",
        )
        if not difference:
            path = np.array(
                report[
                    "centroid_host_m"
                    if ax is axes[0, 0]
                    else "centroid_replay_m"
                ]
            )
            ax.plot(100 * path[:, 0], 100 * path[:, 1], color="white", lw=1.3)
        fig.colorbar(im, ax=ax, shrink=0.72, label="mg/m³")
        ax.set(xlabel="x (cm)", ylabel="y (cm)", title=title)
    ax = axes[1, 0]
    ax.plot(time, report["field_l1"], label="Replay")
    ax.plot(time, report["half_step_l1"], label="Host timestep")
    ax.plot(time, report["fine_mesh_l1"], label="Host mesh")
    ax.set(xlabel="Elapsed time (ms)", ylabel="Relative field L1")
    plotting.legend(ax)
    ax = axes[1, 1]
    ax.plot(
        time,
        np.array(report["host_relative_mass_change"]) * 1e12,
        label="OpenFOAM",
    )
    ax.plot(
        time,
        np.array(report["replay_relative_mass_change"]) * 1e12,
        label="Particula",
    )
    ax.set(xlabel="Elapsed time (ms)", ylabel="Relative mass change × 10¹²")
    plotting.legend(ax)
    fractions = np.array(report["uptake_mass_fractions"])
    ax = axes[1, 2]
    ax.plot(time, fractions[:, 0], label="Gas")
    ax.plot(time, fractions[:, 1], label="Condensed")
    ax.plot(time, fractions.sum(axis=1), ls=":", label="Combined")
    ax.set(xlabel="Elapsed time (ms)", ylabel="Fraction of initial tracer")
    plotting.legend(ax)
    plotting.fit(fig)
    return fig


# %% [markdown]
# ## Add local aerosol uptake
#
# Once passive transport passes its independent checks, give every cell a
# populated seed bin and transfer tracer to that bin at 8 s⁻¹. The closed
# cavity has an analytic gas fraction of exp(−8t). Its lost gas mass must
# reappear in condensed material. This second experiment checks the local
# process budget. OpenFOAM supplies the comparison for passive transport.
#
# Change the uptake rate to move the gas-to-particle crossover without
# changing the passive tracer map. To study transport resolution, regenerate
# the host export at a different mesh or timestep and establish its refinement
# envelope before examining the replay residual.

# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", action="store_true", help="run on the NVIDIA GPU"
    )
    device = "cuda:0" if parser.parse_args().cuda else "cpu"
    report = simulate(device)
    print(
        f"Maximum field L1 {max(report['field_l1']):.4f}; "
        f"maximum centroid error {1000 * max(report['centroid_error_m']):.3f} mm."
    )
    plot_results(report)
    plt.show()


# %% [markdown]
# At the reference resolution, the largest relative field L1 was 0.0391.
# The largest displacement between the two centers was 0.351 mm. The host
# mesh refinement alone moved the centroid by as much as 0.895 mm. Inspect
# the residual around sharp concentration gradients when increasing the
# mesh resolution. The tracer mass budget tests the conservative update
# independently of how sharply either scheme represents those gradients.
