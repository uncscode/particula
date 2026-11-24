"""Compare a moving parcel, vertical section and 3D dune grid on CPU or CUDA."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from particula.coupling.flow_factory import FlowFactory
from particula.coupling.flow_replay import GridReplay
from particula.coupling.flows import DirectionalMixing
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun, Release
from particula.dynamics.parcel import ParcelInitialState
from particula.dynamics.parcel_motion import Particle
from particula.dynamics.parcel_replay import replay
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_with_drag,
)


def setup_grid(flow, ny):
    """Give both experiments the same crosswind extent, bed and lid."""
    shape = (60, ny, 24)
    bounds = ((-60, 120), (-30, 30), (0, 100))
    edges = np.linspace(*bounds[0], shape[0] + 1)
    return FlowGrid(
        shape,
        bounds,
        ("open", "open", "periodic", "periodic", "deposit", "closed"),
        tuple(flow.dune.ground(edges)),
    )


def initial_cloud(grid):
    """Initialize one neutral fine-dust class in a finite Gaussian cloud."""
    x, y, eta = grid.coordinates.T
    weights = (
        np.exp(
            -0.5 * (((x + 30) / 6) ** 2 + (y / 6) ** 2 + ((eta - 6) / 3) ** 2)
        )
        * grid.volume
    )
    # A single bin contains 1 mg of fine mineral dust across either grid.
    mass = 1e-6 * weights / weights.sum()
    particle_mass = 4 * np.pi / 3 * (5e-6) ** 3 * 2650
    return np.column_stack((mass / particle_mass, mass))


def ejected_grains(flow, grid):
    """At 2 s, release one finite pulse from each of two bed reservoirs."""
    x = -15.0
    ref_height = 1.0
    position = (x, 0.0, flow.dune.ground(x) + ref_height)
    eta = (position[2] - grid.ground(x)) / grid.jacobian(x)
    ix = tuple(
        np.searchsorted(edges, p, side="right") - 1
        for edges, p in zip(grid.edges, (x, 0.0, eta), strict=True)
    )
    cell = str(np.ravel_multi_index(ix, grid.shape))
    releases = []
    for radius, name, index in [(5e-6, "dust", 0), (75e-6, "sand", 1)]:
        mass = 4 * np.pi / 3 * radius**3 * 2650
        lifted = flow.dune.pickup(
            x,
            reference_height=2,
            drag_coefficient=0.004,
            threshold=0.25,
            available_mass=0.001,
            pulse_mass=0.0001,
        )
        if lifted == 0:
            continue
        masses = [0.0, 0.0]
        masses[index] = mass
        grain = Particle(
            name, cell, position, radius, 2650, lifted / mass, tuple(masses), 0
        )
        slip = float(
            get_particle_settling_velocity_with_drag(
                radius, 2650, 1.2, 1.8e-5, 1.0
            )
        )
        releases.append(Release(grain, slip, 1))
    return releases


def grid_experiment(folder, flow, ny, device):
    """Run the prescribed forcing and repeat from its saved grid archive."""
    grid = setup_grid(flow, ny)
    times = np.arange(0, 26, 2, dtype=float)
    forcing = GridReplay.record(
        grid,
        flow,
        times,
        DirectionalMixing((0.4, 1.0, 0.15)),
        device=device,
        provenance="Synthetic SI sounding, attached flow over an 8 m ridge",
    )
    stem = "section" if ny == 1 else "volume"
    path = folder / f"{stem}-forcing.npz"
    forcing.save(path)
    loaded = FlowFactory().get_strategy("grid_replay", {"path": path})
    final = None
    releases = ejected_grains(flow, grid)
    for provider in (forcing, loaded):
        run = GridRun(
            provider,
            InventoryLayout((), ("fine",), ("mineral",)),
            initial_cloud(grid),
            releases=releases,
            bed=[0.001, 0.001],
            device=device,
        )
        tracks = []
        for frame in provider.frames:
            run.advance()
            observed=run.snapshot()
            tracks.append(observed["world"])
            if frame.end==8:
                mass=observed["state"][:,1].reshape(grid.shape)
        state = run.snapshot()
        if final is not None:
            for field in state:
                np.testing.assert_array_equal(state[field], final[field])
        final = state
    budget = run.budget()
    np.testing.assert_allclose(
        budget["state"][1] + budget["exports"][1], 1e-6, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        budget["bed_mass"]
        + budget["deposited"][2:]
        + budget["airborne"][2:]
        + budget["escaped"][2:],
        [0.001, 0.001],
        rtol=1e-12,
        atol=0,
    )
    sand = state["ids"].tolist().index("sand")
    ref = flow.dune.track(
        releases[sand].particle.position,
        [2.0, 24.0],
        settling=releases[sand].slip,
    )
    peak_speed = forcing.frames[0].air[:, 0].max()
    metrics = dict(
        peak_east_wind_m_s=float(peak_speed),
        binned_airborne_kg=budget["state"][1],
        binned_export_kg=budget["exports"][1],
        bed_kg=budget["bed_mass"].tolist(),
        resolved_airborne_kg=budget["airborne"][2:].tolist(),
        deposited_kg=budget["deposited"][2:].tolist(),
        escaped_kg=budget["escaped"][2:].tolist(),
        sand_impact_s=state["event_time"][sand],
        continuous_impact_s=ref.deposition_time,
        sand_impact_x_m=state["world"][sand, 0],
        continuous_impact_x_m=ref.xyz[-1, 0],
        replay_identical=True,
    )
    return grid, mass, np.array(tracks), metrics


def main(device="cpu", output=Path(".artifacts/prescribed-flow/grid")):
    """Save numerical forcing, comparison figure and measured mass ledgers."""
    output.mkdir(exist_ok=True, parents=True)
    flow = FlowFactory().get_strategy(
        "dune",
        dict(
            sounding=Path(__file__).with_name("synthetic_sounding.csv"),
            height=8,
            width=18,
            lid=100,
        ),
    )
    aerosol = ParcelInitialState(
        np.array([60e-9]), np.array([1e7]), 0.5, 1e5, 290, 0.008
    )
    track, trace = flow.parcel_trace(
        "air",
        [-45, 0, 4],
        np.linspace(0, 24, 97),
        aerosol,
        provenance="synthetic sounding over Gaussian dune",
    )
    path = output / "air.csv"
    trace.save(path)
    loaded = (
        FlowFactory()
        .get_strategy("trajectory_replay", {"path": path})
        .trajectory("air")
    )
    wet = replay(loaded, max_step=0.025, device=device).wet_radius
    np.testing.assert_array_equal(
        wet, replay(trace, max_step=0.025, device=device).wet_radius
    )
    g2, m2, tracks, section = grid_experiment(output, flow, 1, device)
    g3, m3, _, volume = grid_experiment(output, flow, 8, device)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    x = np.linspace(-60, 120, 301)
    axes[0].fill_between(x, 0, flow.dune.ground(x), color="#d4b27d")
    axes[0].plot(
        track.xyz[:, 0], track.xyz[:, 2], label="Air parcel", color="#034078"
    )
    for j, name in enumerate(("Fine dust", "Sand")):
        axes[0].plot(
            tracks[:, j, 0], tracks[:, j, 2], "o-", markersize=3, label=name
        )
    axes[0].set(
        ylabel="Height above datum (m)",
        title="Moving parcel and ejected grains",
        ylim=(0, 16),
    )
    axes[0].legend()
    x, z = np.meshgrid(g2.edges[0], g2.edges[2], indexing="ij")
    z = g2.ground(x) + g2.jacobian(x) * z
    concentrations = m2[:, 0, :] / g2.volume.reshape(g2.shape)[:, 0, :] * 1e9
    im = axes[1].pcolormesh(
        x, z, concentrations, cmap="viridis", shading="flat"
    )
    axes[1].fill_between(g2.edges[0], 0, g2.bed, color="#d4b27d")
    axes[1].set(
        ylabel="Height above datum (m)",
        ylim=(0, 35),
        title="Vertical section at 8 s",
    )
    fig.colorbar(im, ax=axes[1], label="Fine dust (µg/m³)")
    im = axes[2].pcolormesh(
        g3.edges[0],
        g3.edges[1],
        m3.sum(axis=2).T * 1e9 / np.prod(g3.spacing[:2]),
        cmap="viridis",
    )
    axes[2].set(ylabel="North (m)", title="3D column inventory at 8 s")
    fig.colorbar(im, ax=axes[2], label="Fine dust (µg/m²)")
    for ax in axes:
        ax.set_xlabel("East (m)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(
        device=device,
        parcel_peak_height_m=float(trace.height.max()),
        wet_radius_nm=(wet.min() * 1e9, wet.max() * 1e9),
        section=section,
        volume=volume,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/prescribed-flow/grid")
    )
    args = parser.parse_args()
    main(args.device, args.output)
