# %% [markdown]
# # Where does a dust puff go when the wind turns?
#
# Three synthetic soundings turn an initially eastward wind through north
# to west in 20 s. An air parcel and a puff of suspended dust start at the
# same point, 5 m above level ground. Their horizontal displacements have
# an exact reference because each wind component varies linearly between
# soundings. The air also moistens and dries. A soluble coating represented
# by a single hygroscopic class responds to that humidity along the track.
#
# The input table gives wind in m/s, height above the level bed in metres,
# temperature in K, pressure in Pa and water vapor in kg per kg of dry air.
# The heights sampled by each sounding may differ. The wind and humidity
# are uniform with height here so we can isolate their variation in time.
# For measured soundings, convert to the same vertical datum and time
# origin, resolve wind into east and north components, and restrict the
# calculation to the height range covered by every profile. Load the measured
# table with `SoundingSequence.from_csv`, supply its source and time origin,
# and pass it to `simulate(sequence=profiles, height=...)`. The downloaded
# example includes the same CSV in its repository-relative location.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.coupling.flow_factory import FlowFactory
from particula.coupling.flow_replay import GridReplay
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun
from particula.coupling.sounding_sequence import SoundingSequence
from particula.dynamics.parcel import ParcelInitialState
from particula.dynamics.parcel_replay import replay

TABLE = Path("docs/data/prescribed-flow/turning-soundings.csv")


# %% [markdown]
# ## Follow the same wind in two ways
#
# Sample the air parcel at one-second intervals and replay its water uptake
# with growth steps of at most 0.01 s. The grid represents the same release
# by dry material and particle number in 2 m horizontal cells in a 10 m
# deep layer. Give the fixed grid the mass in a single cell and calculate
# its centroid after each accepted interval. No settling, emission or eddy
# diffusion acts in this experiment. The first-order donor scheme spreads
# the field during advection. Its first moment can still follow the parcel.
#
# Along either ten-second leg, integrating the linear wind gives the mean
# of the two endpoint vectors times the elapsed time. The expected positions
# are (10, 10) m at the turn and (0, 20) m at the end. The trajectory uses
# CPU quadrature of the prescribed wind. Field sampling, grid transport
# and wet growth use the selected device.


# %%
def simulate(device="cuda:0", sequence=None, height=5.0):
    """Return the parcel, grid and humidity histories on a shared schedule."""
    seq = sequence or SoundingSequence.from_csv(
        TABLE,
        time_origin="release at 0 s",
        provenance="synthetic turning wind, common level-bed datum",
    )
    flow = FlowFactory().get_strategy("sounding_sequence", {"sequence": seq})
    times = np.union1d(seq.times, np.linspace(seq.times[0], seq.times[-1], 21))
    shared = seq.profiles[0].height
    half_depth = min(5.0, height - shared[0], shared[-1] - height)
    if half_depth <= 0:
        raise ValueError("choose a release height inside the shared profile")

    aerosol = ParcelInitialState(
        np.array([70e-9]), np.array([1e7]), 0.5, 1e5, 290, 0.006
    )
    track, trace = flow.parcel_trace(
        "air", (0, 0, height), times, aerosol, provenance="turning wind control"
    )
    growth = replay(trace, max_step=0.01, device=device)
    extent = max(61.0, 3 * np.max(np.abs(track.xyz[:, :2])))
    grid = FlowGrid(
        (61, 61, 1),
        (
            (-extent, extent),
            (-extent, extent),
            (height - half_depth, height + half_depth),
        ),
    )

    forcing = GridReplay.record(
        grid, flow, times, device=device, provenance=trace.provenance
    )
    layout = InventoryLayout((), ("fine",), ("dry",))
    state = np.zeros((len(grid.volume), 2))
    mass_per_particle = 4 * np.pi / 3 * 70e-9**3 * 2600
    state[len(state) // 2] = [1e7, 1e7 * mass_per_particle]
    grid_run = GridRun(forcing, layout, state, device=device)
    centroid = [[0.0, 0.0]]
    initial_mass = state[:, 1].sum()
    deviations = []
    for frame in forcing.frames:
        grid_run.advance()
        if frame.end not in times:
            continue
        mass = grid_run.snapshot()["state"][:, 1]
        centroid.append((mass @ grid.centers[:, :2] / mass.sum()).tolist())
        deviations.append(abs(mass.sum() / initial_mass - 1))

    if sequence is None:
        # Closed-form displacement for the supplied piecewise linear winds.
        x = 2 * times - 0.1 * times**2
        y = np.where(times <= 10, 0.1 * times**2, 20 - 0.1 * (20 - times) ** 2)
        exact = np.column_stack((x, y))
        np.testing.assert_allclose(track.xyz[:, :2], exact, atol=1e-11)
        np.testing.assert_allclose(centroid, exact, atol=1e-10)
    return dict(
        time_s=times.tolist(),
        xy_m=track.xyz[:, :2].tolist(),
        centroid_m=centroid,
        wind_m_s=[
            flow.sample(point[None, :], t, device=device)[0, :2].tolist()
            for point, t in zip(track.xyz, times, strict=True)
        ],
        wet_radius_nm=(1e9 * growth.wet_radius.ravel()).tolist(),
        vapor_g_kg=(1e3 * trace.vapor).tolist(),
        max_relative_mass_error=max(deviations),
        max_growth_step_s=0.01,
        dry_radius_nm=70.0,
        dry_particle_density_kg_m3=2600.0,
        hygroscopicity=0.5,
        dx_m=grid.spacing[0],
        layer_depth_m=2 * half_depth,
        height_m=height,
        precision="float64",
        source=str(TABLE) if sequence is None else seq.provenance,
        provenance=trace.provenance,
        device=wp.get_device(device).name,
    )


# %% [markdown]
# The arrows show wind vectors at the initial position, the turn and the
# endpoint. The grid centroids are plotted every two seconds so the parcel
# track remains visible. Compare the vapor history with the wet-radius panel
# to see how quickly the hygroscopic class adjusts to the external water supply.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot wind and displacement in plan view beside the wet-radius history."""
    fig, (path, vapor, radius) = plotting.subplots(1, 3)
    xy = np.asarray(report["xy_m"])
    centroid = np.asarray(report["centroid_m"])
    t = np.asarray(report["time_s"])
    path.plot(*xy.T, label="Parcel")
    path.plot(*centroid[::2].T, "o", fillstyle="none", label="Grid centroid")
    idx = [0, len(t) // 2, len(t) - 1]
    u, v = np.asarray(report["wind_m_s"])[idx].T
    duration = (t[-1] - t[0]) / 10
    path.quiver(
        *xy[idx].T,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1 / duration,
        width=0.009,
        color="0.25",
    )
    points = np.vstack((xy, xy[idx] + duration * np.column_stack((u, v))))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    half_width = max(1.0, 0.60 * np.ptp(points, axis=0).max())
    path.set(
        xlabel="East (m)",
        ylabel="North (m)",
        aspect="equal",
        xlim=(center[0] - half_width, center[0] + half_width),
        ylim=(center[1] - half_width, center[1] + half_width),
    )
    plotting.legend(path)
    vapor.plot(t, report["vapor_g_kg"], "--", color="C1")
    vapor.set(xlabel="Time (s)", ylabel="Vapor (g/kg dry air)")
    radius.plot(t, report["wet_radius_nm"], color="C0")
    radius.set(xlabel="Time (s)", ylabel="Wet radius (nm)")
    plotting.fit(fig)
    return fig


# %% [markdown]
# With zero initial offset both representations should finish 20 m north
# of the release. Dry mass closes even as the fixed-grid puff broadens.
# Wet radius responds to the imposed humidity maximum near the wind turn.
# Run the downloaded script from the extracted directory. Use `--cuda` to
# reproduce the published GPU run or omit the flag for CPU.


# %%
def run_example(device="cuda:0"):
    """Run and display the displacement and wet-growth comparison."""
    result = simulate(device)
    print(f"Final grid centroid (m) {np.round(result['centroid_m'][-1], 6)}")
    print(
        f"Maximum relative dry-mass error {result['max_relative_mass_error']:.2e}"
    )
    plot_results(result)
    plt.show()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", action="store_true", help="run on the NVIDIA GPU"
    )
    device = "cuda:0" if parser.parse_args().cuda else "cpu"
    run_example(device)

# %% [markdown]
# Keeping just the initial and final soundings would remove the northward
# wind entirely. Moving the middle humidity peak while retaining its wind
# tests a different question. The grid centroid should keep its track,
# while the peak wet radius moves in time. For shear with height, repeat at
# several release levels and compare their centroids. The prescribed-air
# model keeps an external water budget during uptake.
# [Parcel forcing and its water budget](../../../theory/thermodynamics/parcel-dynamics.md)
