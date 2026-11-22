# %% [markdown]
# # Where does a particle group travel in wind and updraft?
#
# A wind can carry particles out of a fixed group of parcels or carry the
# whole group along with them. A falling particle can even gain height above
# the ground while it loses height inside a rising group. Compare the same
# population in a vertical stack, a periodic horizontal row, and a translating
# block. Each arrangement is **one connected domain** with two named groups,
# A and B. A group is a pair of neighboring cells whose membership stays fixed.
# Its label lets us count transfers across the group boundary and track where
# particles released in each group end up.
#
# The air moves at 0.70 m/s east and 0.022 m/s up in all three experiments.
# Both components are prescribed, with no buoyancy feedback. The moving mesh
# follows the air as a rigid body. The particles remain at their release size
# in uniform air at 293.15 K and 101325 Pa. A single CPU suffices for this run.

# %%
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from matplotlib.patches import Rectangle
from particula.dynamics.parcel import ParcelInitialState, run_parcel
from particula.dynamics.parcel_cluster import ParcelCluster
from particula.dynamics.parcel_motion import (
    MotionState,
    Particle,
    advance_motion,
)
from particula.dynamics.parcel_replay import ParcelTrace
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)

AIR = np.array([0.70, 0.0, 0.022])  # m/s in the fixed world frame
DURATION = 14.0  # s
TEMPERATURE = 293.15  # K
PRESSURE = 101325.0  # Pa
DENSITY = 1000.0  # kg/m³
RADII = np.array([1e-6, 10e-6])  # m
COUNTS = (4, 1)  # represented particles per cell for the two radii


# %% [markdown]
# ## Group four cells and release the same population in each layout
#
# The fixed stack has four horizontal layers, each 7 cm deep. A contains
# the bottom pair. The A–B interface at 14 cm is internal. The stack’s top
# is open at 28 cm. The periodic row has
# four half-meter cells, with A on the west and B on the east. Crossing its
# eastern boundary returns a particle to A. In the translating block A is
# the lower pair of cells and B the upper pair. The two release sizes start
# together in each cell. A count of four small particles for every large one
# makes a number-weighted ensemble mean different from a mean over records.
#
# A uniform stationary thermodynamic trace supplies each cell's declared
# state. The motion experiment imposes the same velocity independently of
# this trace. The [air viscosity](../../../../theory/properties/air-transport.md#air-viscosity)
# and slip-corrected Stokes speeds depend on temperature and pressure.


# %%
def setup(case, base_trace):
    """Return one grouped domain, its particle releases, and its face rules."""
    shapes = {"stack": (1, 1, 4), "row": (4, 1, 1), "block": (2, 1, 2)}
    widths = {
        "stack": (20, 1, 0.07),
        "row": (0.5, 1, 1),
        "block": (0.5, 1, 0.5),
    }
    shape, width = shapes[case], widths[case]
    traces, groups = {}, {}
    for x, _, z in np.ndindex(*shape):
        if case == "row":
            group, member = "AB"[x // 2], x % 2 + 1
        elif case == "stack":
            group, member = "AB"[z // 2], z % 2 + 1
        else:
            group, member = "AB"[z], x + 1
        cell = f"{group}{member}"
        traces[cell] = replace(base_trace, parcel_id=cell)
        groups[cell] = group
    cluster = ParcelCluster.regular(traces, shape=shape, widths=width)
    particles = []
    for cell, (lo, hi) in cluster.bounds.items():
        xyz = (np.asarray(lo) + hi) / 2
        if case == "stack":
            xyz[0] = 5.0  # m, enough downwind room before any east-face exit
        elif case == "block" and groups[cell] == "B":
            xyz[2] = lo[2] + 0.05  # m, 5 cm above the group interface
        for radius, count in zip(RADII, COUNTS, strict=True):
            size = round(radius * 1e6)
            particles.append(
                Particle(
                    id=f"{cell}-{size}",
                    cell=cell,
                    position=tuple(xyz),
                    radius=radius,
                    density=DENSITY,
                    count=count,
                    masses=(float(4 * np.pi * radius**3 * DENSITY / 3),),
                    charge=0,
                )
            )
    policies = {}
    if case == "row":
        policies = {("A1", 0, -1): "periodic", ("B2", 0, 1): "periodic"}
    elif case == "stack":
        policies = {("B2", 2, 1): "open"}
    mesh = AIR if case == "block" else np.zeros(3)
    return cluster, groups, particles, policies, mesh


# %% [markdown]
# ## Measure displacement through crossings and departures
#
# Compare the trajectories against constant-velocity motion. In the world
# frame a particle travels east at the wind speed and rises at $w-v_s$,
# where $v_s$ is its terminal speed. Subtract mesh displacement to obtain
# its motion relative to the cells. For the periodic row, add one domain
# length per wrap to recover distance traveled. A particle that leaves the
# open stack contributes to the displacement average up to its exit, after
# which its recorded displacement stays fixed. This average accounts for the
# entire released population and has a finite observation boundary.
#
# Save the cell and group of origin, signed displacements and destination
# for every particle record. Check position to 1 nanometer and exit time
# to 1 microsecond, well below the cell widths and output spacing. Count
# crossings by represented number so events from the two sizes carry the
# same weights as the displacement means.


# %%
def number_weighted_means(records):
    """Average every release over represented number on the common clock."""
    means = {}
    for label, size in (("all", None), ("small", 1), ("large", 10)):
        rows = [
            p
            for p in records.values()
            if size is None or p["radius_um"] == size
        ]
        means[label] = {
            field: np.average(
                [p[field] for p in rows],
                axis=0,
                weights=[p["count"] for p in rows],
            ).tolist()
            for field in ("dx_world_m", "dz_world_m", "dz_mesh_m")
        }
    return means


def crossing_counts(case, events, records):
    """Count represented boundary passages and verify each expected circuit."""
    group_ledger = {}
    for source, destination in (("A", "B"), ("B", "A"), ("B", "outside")):
        group_ledger[f"{source} to {destination}"] = sum(
            e["count"]
            for e in events
            if (e["source"], e["destination"]) == (source, destination)
        )
    intergroup = group_ledger["A to B"] + group_ledger["B to A"]
    exits_count = group_ledger["B to outside"]
    if intergroup <= 0 or (case == "stack" and exits_count == 0):
        raise AssertionError(
            "The release must exercise the intended group boundary"
        )
    if case == "row":
        for item in records.values():
            expected = int((item["initial_m"][0] + AIR[0] * DURATION) // 2.0)
            if item["wraps"] != expected or expected < 2:
                raise AssertionError(
                    "Every particle must complete multiple periodic circuits"
                )
    return group_ledger, intergroup, exits_count


def trajectories(case, trace):
    """Advance motion, close each release cohort, and check its analytic path."""
    cluster, groups, releases, boundaries, mesh = setup(case, trace)
    time = np.linspace(0.0, DURATION, 29)
    state = MotionState(time=0.0, particles=tuple(releases))
    speeds = get_particle_settling_velocity_via_system_state(
        RADII, DENSITY, TEMPERATURE, PRESSURE
    )
    vertical = {r: AIR[2] - v for r, v in zip(RADII, speeds, strict=True)}
    records = {
        p.id: dict(
            cell=p.cell,
            group=groups[p.cell],
            radius_um=round(p.radius * 1e6),
            count=p.count,
            initial_m=list(p.position),
            dx_world_m=[0.0],
            dz_world_m=[0.0],
            dz_mesh_m=[0.0],
            x_wrapped_m=[p.position[0]],
        )
        for p in releases
    }
    max_error = 0.0
    for dt in np.diff(time):
        state = advance_motion(
            cluster,
            state,
            duration=float(dt),
            air_velocity=AIR,
            mesh_velocity=mesh,
            capacity=len(releases),
            boundaries=boundaries,
            temperature=TEMPERATURE,
            pressure=PRESSURE,
            device="cpu",
        )
        live = {p.id: p for p in state.particles}
        exits = {e.particle: e for e in state.events if e.kind == "open"}
        if set(exits) & set(live) or len(exits) + len(live) != len(releases):
            raise AssertionError(
                "Each release must remain live or have one recorded exit"
            )
        for p in releases:
            series = records[p.id]
            if p.id in exits:
                particle, clock = exits[p.id].record, exits[p.id].time
            else:
                particle, clock = live[p.id], state.time
            wraps = sum(
                e.kind == "periodic" and e.particle == p.id
                for e in state.events
            )
            dx = particle.position[0] - p.position[0] + 2.0 * wraps
            dz = particle.position[2] - p.position[2]
            relative_z = dz - mesh[2] * clock
            if case == "stack":
                exit_time = (0.28 - p.position[2]) / vertical[p.radius]
                expected_clock = min(state.time, exit_time)
            else:
                expected_clock = state.time
            np.testing.assert_allclose(clock, expected_clock, atol=1e-6, rtol=0)
            error = np.abs(
                np.array([dx, dz, relative_z])
                - expected_clock
                * np.array(
                    [
                        AIR[0],
                        vertical[p.radius],
                        vertical[p.radius] - mesh[2],
                    ]
                )
            )
            max_error = max(max_error, float(max(error)))
            np.testing.assert_allclose(error, 0.0, atol=1e-9)
            series["dx_world_m"].append(float(dx))
            series["dz_world_m"].append(float(dz))
            series["dz_mesh_m"].append(float(relative_z))
            series["x_wrapped_m"].append(float(particle.position[0]))
            series["destination"] = (
                "outside" if p.id in exits else groups[particle.cell]
            )
            series["wraps"] = wraps
    events = [
        dict(
            time_s=e.time,
            kind=e.kind,
            particle=e.particle,
            origin=records[e.particle]["group"],
            source=groups[e.source],
            destination=groups.get(e.destination, "outside"),
            count=e.record.count,
        )
        for e in state.events
    ]
    group_ledger, intergroup, exits_count = crossing_counts(
        case, events, records
    )
    return dict(
        group_ledger=group_ledger,
        time_s=time.tolist(),
        records=records,
        means=number_weighted_means(records),
        events=events,
        cells={k: np.asarray(b).tolist() for k, b in cluster.bounds.items()},
        groups=groups,
        group_crossings=intergroup,
        open_exits=exits_count,
        max_position_error_m=max_error,
        mesh_velocity_m_s=mesh.tolist(),
    )


def simulate():
    """Run the three prescribed-flow experiments using Warp CPU."""
    initial = ParcelInitialState(
        dry_radius=np.array([70e-9]),
        number=np.array([1e6]),
        kappa=0.5,
        pressure=PRESSURE,
        temperature=TEMPERATURE,
        vapor=0.01,
    )
    stationary = run_parcel(initial, updraft=0.0, timestep=DURATION, steps=1)
    trace = ParcelTrace.record(
        stationary, parcel_id="base", provenance="Particula stationary parcel"
    )
    return {
        name: trajectories(name, trace) for name in ("stack", "row", "block")
    }


# %% [markdown]
# ## Follow a release in the world and in its parcel group
#
# The upper panels show the initial cell footprints in an east–height slice.
# Blue denotes A and orange denotes B. Each black dot represents the two sizes
# released at the same point. The three geometries have different axis scales.
# The lower panels show the number-weighted means, with the size-resolved
# means beside them for the vertical cases. Wrapped coordinates in the center
# panel are for one release in A and one in B. Their common unwrapped travel
# measures the full downwind distance.


# %%
def plot_results(report):
    """Draw group boundaries and observed displacements in all three frames."""
    with mplstyle.context("particula.default"):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained")
        colors = {"A": "#d6eaf5", "B": "#ffdfae"}
        for col, (key, title) in enumerate(
            (
                ("stack", "Fixed vertical stack"),
                ("row", "Periodic horizontal row"),
                ("block", "Translating block"),
            )
        ):
            ax, history = axes[0, col], axes[1, col]
            data = report[key]
            for cell, (lo, hi) in data["cells"].items():
                ax.add_patch(
                    Rectangle(
                        (lo[0], lo[2]),
                        hi[0] - lo[0],
                        hi[2] - lo[2],
                        facecolor=colors[data["groups"][cell]],
                        edgecolor="0.3",
                    )
                )
                ax.text(
                    lo[0] + 0.08 * (hi[0] - lo[0]),
                    hi[2] - 0.12 * (hi[2] - lo[2]),
                    cell,
                    va="top",
                    fontsize=9,
                )
            rec = list(data["records"].values())
            for p in rec[::2]:
                ax.plot(
                    p["initial_m"][0], p["initial_m"][2], "o", color="0.1", ms=4
                )
            ax.autoscale_view()
            ax.set(title=title, xlabel="East (m)", ylabel="Height (m)")
            time = data["time_s"]
            if key == "row":
                for cell, style in (("A1", "-"), ("B1", "--")):
                    particle_id = f"{cell}-1"
                    x = data["records"][particle_id]["x_wrapped_m"]
                    t_path, x_path = [0.0], [x[0]]
                    for event in data["events"]:
                        if (
                            event["kind"] == "periodic"
                            and event["particle"] == particle_id
                        ):
                            t_path.extend(
                                [event["time_s"], np.nan, event["time_s"]]
                            )
                            x_path.extend([2.0, np.nan, 0.0])
                    t_path.append(time[-1])
                    x_path.append(x[-1])
                    (line,) = history.plot(
                        t_path, x_path, style, label=f"From {cell}, wrapped"
                    )
                    history.plot(time, x, ".", color=line.get_color(), ms=2)
                history.plot(
                    time,
                    data["means"]["all"]["dx_world_m"],
                    ":",
                    color="0.15",
                    label="Mean distance",
                )
                history.set_ylabel("East position or distance (m)")
            else:
                for label, size, style in (
                    ("small", 1, "-"),
                    ("large", 10, "--"),
                ):
                    history.plot(
                        time,
                        100 * np.asarray(data["means"][label]["dz_world_m"]),
                        style,
                        label=f"{size} µm radius",
                    )
                history.plot(
                    time,
                    100 * np.asarray(data["means"]["all"]["dz_world_m"]),
                    ":",
                    color="0.15",
                    label="World mean",
                )
                if key == "block":
                    history.plot(
                        time,
                        100 * np.asarray(data["means"]["all"]["dz_mesh_m"]),
                        "-.",
                        color="#845188",
                        label="Moving-mesh mean",
                    )
                history.set_ylabel("Vertical displacement (cm)")
            history.set_xlabel("Time (s)")
            history.legend(loc="best", fontsize=8, frameon=True)
        return fig


# %% [markdown]
# Read the final displacement alongside the number of passages between
# groups. A single particle can contribute several passages on the periodic
# row. The stack's exit count is the represented number discharged through its
# top; an interior crossing preserves the number in the combined domain.


# %%
def run_example():
    """Report crossing counts and compare size-specific and mean displacements."""
    report = simulate()
    for key, data in report.items():
        print(
            f"{key}: {data['group_crossings']:g} represented group-boundary passages, {data['open_exits']:g} top exits"
        )
        for group, label in (
            ("small", "1 µm radius"),
            ("large", "10 µm radius"),
            ("all", "all represented particles"),
        ):
            fields = [
                data["means"][group][field][-1]
                for field in ("dx_world_m", "dz_world_m", "dz_mesh_m")
            ]
            print(
                f"  {label}: east {fields[0]:.3f} m, world height change {fields[1]:+.3f} m, mesh height change {fields[2]:+.3f} m"
            )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# After 14 s the translating block has risen 30.8 cm. The population has
# traveled 9.8 m east and risen 27.3 cm in the world frame. Relative to its
# cells the population has fallen 3.54 cm. Both 10 µm-radius releases in B
# enter A, even as they rise above their original world heights. The smaller
# particles settle more slowly and remain in their release cells.
#
# The same wind carries each release through five periodic wraps in the row.
# All 20 represented particles remain in the domain. In the fixed stack, 18
# leave through its open top. The flattening stack curves reflect that finite
# observation window. Increasing the stack height would let us observe more
# of their rise before they leave.
