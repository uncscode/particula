# %% [markdown]
# # Where does a particle group travel in wind and updraft?
#
# Follow 1,024 individual particles through a vertical stack, a periodic row,
# and a block of cells moving with the wind. Each layout contains four cells in
# one connected domain. Group A and group B each contain two adjacent cells.
# Each particle has its own ID, position and three component masses. The same
# IDs carry the same radii and recipes through all three experiments.
#
# The prescribed air velocity is 0.70 m/s east and 0.022 m/s up. There is no
# buoyancy feedback. The moving block translates rigidly with the air. Radii
# and composition stay fixed in uniform air at 293.15 K and 101325 Pa. This
# CPU example asks which particles change groups and how much of each species
# they carry through a face.

# %%
from collections import Counter
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from matplotlib.patches import Rectangle
from particula.dynamics.parcel import RD, ParcelInitialState, run_parcel
from particula.dynamics.parcel_cluster import ParcelCluster
from particula.dynamics.parcel_motion import (
    MotionState,
    Particle,
    advance_motion,
)
from particula.dynamics.parcel_replay import ParcelTrace
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)

AIR = np.array([0.70, 0.0, 0.022])  # m/s, fixed world frame
DURATION = 14.0  # s
TEMPERATURE, PRESSURE = 293.15, 101325.0  # K, Pa
RADII_UM = np.array([1.0, 3.0, 6.0, 10.0])
SPECIES = ("Ammonium sulfate", "Sodium chloride", "Sucrose")
DENSITIES = np.array([1770.0, 2165.0, 1587.0])  # assumed kg/m³
RECIPES = (*SPECIES, "Ternary mixture")
VOLUME_FRACTIONS = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.40, 0.35, 0.25]]
)
SEED = 1447


# %% [markdown]
# ## Release the same population into three layouts
#
# Set a particle's species masses from its spherical volume: component mass
# is total volume times that component's volume fraction and material density.
# The sum of component masses divided by total volume gives the density used
# for settling. Each radius includes pure ammonium sulfate, sodium chloride,
# sucrose and an internal mixture of all three. These are specified compact
# dry particles. The ternary volume fractions are 40% ammonium sulfate,
# 35% sodium chloride and 25% sucrose. Densities of 1770 and 1587 kg/m³
# follow the [chamber example](../../coupled-populations/chambers/cloud_chamber_single_cycle.py)
# for ammonium sulfate and sucrose. We assign sodium chloride 2165 kg/m³,
# close to the [reported solid density](https://www.sigmaaldrich.com/US/en/product/sial/s7653)
# of 2.16 g/cm³. Water uptake, shape changes and chemistry are held fixed.
#
# The stack's four layers are 7 cm deep. A contains the bottom pair. The A–B
# interface is at 14 cm and the open top is at 28 cm. The periodic row has
# four half-meter cells, A to the west and B to the east. The moving block
# has two columns, with A below B. In each cell, every size–recipe pair is
# released at two east/height locations with eight individual particles per
# location. A seeded stratified draw gives each particle a different north
# coordinate. The four radii and recipes are equally abundant. With uniform
# air and no collisions, particles released together in east and height have
# matching trajectories if their size and density also match.
#
# `replicas_per_stratum` controls population size and `seed` controls the
# transverse release coordinates. The default run advances 4 × 4 × 4 × 2 × 8
# particle records. Each record counts as one physical particle.


# %%
def layout(case, base_trace):
    """Build the four-cell domain; A and B are stable groups of two IDs."""
    shapes = {"stack": (1, 1, 4), "row": (4, 1, 1), "block": (2, 1, 2)}
    widths = {
        "stack": (20, 1, 0.07),
        "row": (0.5, 1, 1),
        "block": (0.5, 1, 0.5),
    }
    traces, groups = {}, {}
    for x, _, z in np.ndindex(*shapes[case]):
        if case == "row":
            group, member = "AB"[x // 2], x % 2 + 1
        elif case == "stack":
            group, member = "AB"[z // 2], z % 2 + 1
        else:
            group, member = "AB"[z], x + 1
        cell = f"{group}{member}"
        traces[cell] = replace(base_trace, parcel_id=cell)
        groups[cell] = group
    cluster = ParcelCluster.regular(
        traces, shape=shapes[case], widths=widths[case]
    )
    boundaries = {
        "stack": {("B2", 2, 1): "open"},
        "row": {("A1", 0, -1): "periodic", ("B2", 0, 1): "periodic"},
        "block": {},
    }
    return (
        cluster,
        groups,
        boundaries[case],
        AIR if case == "block" else np.zeros(3),
    )


def release_points(case, bounds, group):
    """Two x/z strata per cell, kept away from unrelated exterior faces."""
    lo, hi = np.asarray(bounds)
    fractions = np.array([0.35, 0.65])
    x = lo[0] + fractions * (hi[0] - lo[0])
    z = lo[2] + fractions * (hi[2] - lo[2])
    if case == "stack":
        x = np.array([4.8, 5.2])
        if lo[2] == 0:
            z = np.array([0.064, 0.067])
    elif case == "block":
        # Lower releases need room to settle; upper releases start just above the face.
        z = lo[2] + (
            np.array([0.40, 0.45]) if group == "A" else np.array([0.04, 0.08])
        )
    elif case == "row":
        z = np.array([0.40, 0.55])
    return np.column_stack([x, z])


def release(case, cluster, groups, replicas, seed):
    """Seed distinct y positions, fixed species masses, and stable physical IDs."""
    rng = np.random.default_rng(seed)
    n_cell = len(RADII_UM) * len(RECIPES) * 2 * replicas
    batch, recipes, sizes, points, tracked = [], [], [], {}, {}
    for cell in sorted(cluster.bounds):
        y_positions = (rng.permutation(n_cell) + rng.random(n_cell)) / n_cell
        points[cell] = release_points(
            case, cluster.bounds[cell], groups[cell]
        ).tolist()
        for i in range(n_cell):
            size, recipe, stratum, replica = np.unravel_index(
                i, (len(RADII_UM), len(RECIPES), 2, replicas)
            )
            x, z = points[cell][stratum]
            radius = float(RADII_UM[size] * 1e-6)
            volume = 4 * np.pi * radius**3 / 3
            masses = volume * VOLUME_FRACTIONS[recipe] * DENSITIES
            particle = Particle(
                id=f"{cell}-{size}-{recipe}-{stratum}-{replica}",
                cell=cell,
                position=(x, float(y_positions[i]), z),
                radius=radius,
                density=float(masses.sum() / volume),
                count=1,
                masses=tuple(masses),
                charge=0,
            )
            batch.append(particle)
            recipes.append(recipe)
            sizes.append(size)
            if (recipe, stratum, replica) == (3, 0, 0) and size in (
                0,
                len(RADII_UM) - 1,
            ):
                tracked[particle.id] = dict(
                    cell=cell,
                    group=groups[cell],
                    radius_um=RADII_UM[size],
                    recipe=RECIPES[recipe],
                    initial_m=list(particle.position),
                    wrapped_x_m=[x],
                    wrap_times_s=[],
                )
    if len({p.id for p in batch}) != len(batch) or len(
        {p.position for p in batch}
    ) != len(batch):
        raise AssertionError(
            "Physical particles require unique IDs and release positions"
        )
    return tuple(batch), np.array(recipes), np.array(sizes), points, tracked


# %% [markdown]
# ## Compare transport against constant velocity and close each species budget
#
# A particle of terminal speed $v_s$ travels east at the wind speed and rises
# at $w-v_s$. Subtract mesh displacement to express its movement relative to
# the local cells. The kinematic reference and the mover use the same
# Stokes–Cunningham terminal-speed law and [air viscosity](../../../../theory/properties/air-transport.md#air-viscosity).
# Check that each particle Reynolds number remains below the mover's creeping-
# flow limit of 0.1. Check every physical ID at each observation against its
# expected world and mesh displacement within 1 nm. A periodic wrap adds one
# domain length to eastward travel. After a particle exits the fixed stack,
# keep its last displacement in the ensemble mean and its mass in outflow.
#
# Use two independent budgets. Sum the active records in A and B and the
# records at the open boundary. Separately accumulate the signed transfers
# from the initial group inventories. Both methods must give the same
# particle count and mass of **each** species in each group at every sample.


# %%
def settling_reference(sizes, recipes):
    """Independent straight-line kinematics use the same prescribed drag law."""
    density = VOLUME_FRACTIONS[recipes] @ DENSITIES
    radii = RADII_UM[sizes] * 1e-6
    speed = get_particle_settling_velocity_via_system_state(
        radii, density, TEMPERATURE, PRESSURE
    )
    reynolds = (
        2
        * radii
        * speed
        * PRESSURE
        / (RD * TEMPERATURE * get_dynamic_viscosity(TEMPERATURE))
    )
    if np.max(reynolds) > 0.1:
        raise ValueError(
            "Particle Reynolds number exceeds the selected settling regime"
        )
    return speed, reynolds


def group_inventories(group_index, inventory):
    """Sum count and each species mass independently over A, B and outflow."""
    return np.array([inventory[group_index == k].sum(axis=0) for k in range(3)])


def accumulate_crossings(
    events, groups, ledger, exits, wraps, passages, tracked
):
    """Book each actual face crossing once into count and component mass."""
    for e in events:
        particle = e.record
        source = 0 if groups[e.source] == "A" else 1
        destination = (
            2 if e.destination is None else int(groups[e.destination] == "B")
        )
        if e.kind == "periodic":
            wraps[e.particle] += 1
            if e.particle in tracked:
                tracked[e.particle]["wrap_times_s"].append(e.time)
        if e.kind == "open":
            if e.particle in exits:
                raise AssertionError("Particle left the domain twice")
            exits[e.particle] = e
        if source != destination:
            moved = np.array((particle.count, *particle.masses))
            ledger[source] -= moved
            ledger[destination] += moved
            passages[source, destination] += moved


def observe(state, ids, groups, exits):
    """Collect world positions and unaltered inventories by stable ID."""
    live = {p.id: p for p in state.particles}
    if (
        len(live) != len(state.particles)
        or live.keys() & exits.keys()
        or (live.keys() | exits.keys()) != set(ids)
    ):
        raise AssertionError(
            "Each release must be live or have a single recorded exit"
        )
    records = [live[k] if k in live else exits[k].record for k in ids]
    clock = np.array([state.time if k in live else exits[k].time for k in ids])
    index = np.array(
        [int(groups[live[k].cell] == "B") if k in live else 2 for k in ids]
    )
    inventory = np.array([(p.count, *p.masses) for p in records])
    return (
        np.array([p.position for p in records]),
        clock,
        group_inventories(index, inventory),
        inventory,
    )


# %% [markdown]
# ## Match each ID to its prediction and fate
#
# Sample all cells once per second, include exits in the inventory at their
# recorded times, and check the complete ordered mass vector for each ID.
# Retain final height and cell for all releases, representative paths for
# the layout plot, and mean displacements for the size–composition plot.


# %%
def trajectories(case, trace, replicas, seed, device):
    """Advance a common physical cohort and retain compact summaries by recipe."""
    cluster, groups, boundaries, mesh = layout(case, trace)
    releases, recipes, sizes, points, tracked = release(
        case, cluster, groups, replicas, seed
    )
    ids = [p.id for p in releases]
    tracked_rows = {k: ids.index(k) for k in tracked}
    initial_xyz = np.array([p.position for p in releases])
    initial_inventory = np.array([(p.count, *p.masses) for p in releases])
    initial_groups = np.array([int(groups[p.cell] == "B") for p in releases])
    ledger = group_inventories(initial_groups, initial_inventory)
    totals = ledger.sum(axis=0)
    speed, reynolds = settling_reference(sizes, recipes)
    world_velocity = np.broadcast_to(AIR, initial_xyz.shape).copy()
    world_velocity[:, 2] -= speed
    max_clock = np.full(len(releases), DURATION)
    if case == "stack":
        rising = world_velocity[:, 2] > 0
        max_clock[rising] = np.minimum(
            DURATION,
            (0.28 - initial_xyz[rising, 2]) / world_velocity[rising, 2],
        )
    state = MotionState(time=0.0, particles=releases)
    time = np.linspace(0.0, DURATION, 15)
    frames = [np.zeros_like(initial_xyz)]
    observations = [ledger.copy()]
    maxima = np.zeros(4)  # count and three species
    max_position_error = max_clock_error = 0.0
    exits, wraps = {}, Counter()
    passages = np.zeros((3, 3, 4))  # source × destination × inventory lane
    for dt in np.diff(time):
        cursor = len(state.events)
        state = advance_motion(
            cluster,
            state,
            duration=float(dt),
            air_velocity=AIR,
            mesh_velocity=mesh,
            capacity=len(ids),
            boundaries=boundaries,
            temperature=TEMPERATURE,
            pressure=PRESSURE,
            device=device,
        )
        accumulate_crossings(
            state.events[cursor:],
            groups,
            ledger,
            exits,
            wraps,
            passages,
            tracked,
        )
        xyz, clock, budget, inventory = observe(state, ids, groups, exits)
        np.testing.assert_array_equal(inventory, initial_inventory)
        error = np.max(np.abs(budget - ledger), axis=0)
        maxima = np.maximum(maxima, error)
        # Count is exact. Accumulating signed face fluxes can cancel down to
        # zero; use each component's own initial budget for its roundoff scale.
        np.testing.assert_array_equal(budget[:, 0], ledger[:, 0])
        np.testing.assert_allclose(
            (budget - ledger)[:, 1:] / totals[1:], 0, atol=2e-12
        )
        np.testing.assert_allclose(budget.sum(axis=0) / totals, 1, rtol=2e-13)
        observations.append(budget)
        delta = xyz - initial_xyz
        delta[:, 0] += 2 * np.array([wraps[k] for k in ids])
        expected_clock = np.minimum(state.time, max_clock)
        max_clock_error = max(
            max_clock_error, float(np.max(np.abs(clock - expected_clock)))
        )
        world_error = np.max(
            np.abs(delta - world_velocity * expected_clock[:, None])
        )
        max_position_error = max(max_position_error, float(world_error))
        mesh_z = delta[:, 2] - mesh[2] * clock
        np.testing.assert_allclose(
            mesh_z,
            (world_velocity[:, 2] - mesh[2]) * expected_clock,
            atol=1e-9,
            rtol=0,
        )
        frames.append(np.column_stack([delta[:, 0], delta[:, 2], mesh_z]))
        for k, i in tracked_rows.items():
            tracked[k]["wrapped_x_m"].append(float(xyz[i, 0]))
    if max_position_error > 1e-9 or max_clock_error > 1e-6:
        raise AssertionError(
            "A particle trajectory missed the prescribed kinematic control"
        )
    circuits = np.array([wraps[k] for k in ids])
    if case == "row":
        expected_wraps = np.floor(
            (initial_xyz[:, 0] + AIR[0] * DURATION) / 2
        ).astype(int)
        np.testing.assert_array_equal(circuits, expected_wraps)
        assert min(circuits) >= 2
    assert passages[0, 1, 0] + passages[1, 0, 0] > 0
    if case == "stack":
        assert len(exits) > 0
    final_cells = {p.id: p.cell for p in state.particles}
    return dict(
        case=case,
        time_s=time.tolist(),
        particle_count=len(sizes),
        device=device,
        replicas_per_stratum=replicas,
        seed=seed,
        radii_um=RADII_UM.tolist(),
        recipes=list(RECIPES),
        species=list(SPECIES),
        material_densities_kg_m3=DENSITIES.tolist(),
        volume_fractions=VOLUME_FRACTIONS.tolist(),
        release_catalog={
            "id": ids,
            "initial_cell": [p.cell for p in releases],
            "radius_index": sizes.tolist(),
            "recipe_index": recipes.tolist(),
        },
        final={
            "cell": ["outside" if k in exits else final_cells[k] for k in ids],
            "exited": [k in exits for k in ids],
            "height_m": xyz[:, 2].tolist(),
            "height_change_m": delta[:, 2].tolist(),
            "residence_time_s": clock.tolist(),
            "wraps": circuits.tolist(),
        },
        radius_recipe_counts=np.bincount(sizes * len(RECIPES) + recipes)
        .reshape(len(RADII_UM), len(RECIPES))
        .tolist(),
        min_max_reynolds=[float(min(reynolds)), float(max(reynolds))],
        groups=groups,
        cells={k: np.array(v).tolist() for k, v in cluster.bounds.items()},
        release_points_xz_m=points,
        tracked_paths=tracked,
        **displacement_summary(frames, sizes, recipes),
        budget_groups=["A", "B", "outflow"],
        budget_lanes=["count", *[s + " kg" for s in SPECIES]],
        budget_history=np.array(observations).tolist(),
        max_budget_discrepancies=maxima.tolist(),
        max_position_error_m=max_position_error,
        max_clock_error_s=max_clock_error,
        min_max_wraps=[int(min(circuits)), int(max(circuits))],
        gross_transfers={
            "A to B": passages[0, 1].tolist(),
            "B to A": passages[1, 0].tolist(),
            "B to outside": passages[1, 2].tolist(),
        },
        group_ledger={
            "A to B": int(passages[0, 1, 0]),
            "B to A": int(passages[1, 0, 0]),
            "B to outside": int(passages[1, 2, 0]),
        },
        group_crossings=int(passages[0, 1, 0] + passages[1, 0, 0]),
        open_exits=int(passages[1, 2, 0]),
    )


def displacement_summary(frames, sizes, recipes):
    """Average physical displacements over the tracked size and recipe bins."""
    values = np.array(frames)
    means = {}
    for tag, mask in (
        ("all", np.ones(len(sizes), dtype=bool)),
        ("small", sizes == 0),
        ("large", sizes == len(RADII_UM) - 1),
    ):
        paths = values[:, mask].mean(axis=1)
        means[tag] = {
            k: path.tolist()
            for k, path in zip(
                ("dx_world_m", "dz_world_m", "dz_mesh_m"), paths.T, strict=True
            )
        }
    by_recipe = [
        [
            values[-1, (sizes == i) & (recipes == j)].mean(axis=0).tolist()
            for j in range(len(RECIPES))
        ]
        for i in range(len(RADII_UM))
    ]
    return {
        "means": means,
        "displacement_fields": [
            "east_world_m",
            "height_world_m",
            "height_mesh_m",
        ],
        "final_displacement_by_radius_recipe_m": by_recipe,
    }


def simulate(replicas_per_stratum=8, seed=SEED, *, device="cpu"):
    """Run three layouts with exactly 128 × replicas_per_stratum physical IDs."""
    if type(replicas_per_stratum) is not int or replicas_per_stratum < 1:
        raise ValueError("replicas_per_stratum must be a positive integer")
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
        case: trajectories(case, trace, replicas_per_stratum, seed, device)
        for case in ("stack", "row", "block")
    }


# %% [markdown]
# ## Follow a release in the world and in its parcel group
#
# The upper panels show the initial cell footprints in an east–height slice.
# Blue denotes A and orange denotes B. Each black dot marks one release
# stratum containing all sizes and recipes spread across the north coordinate. The three geometries have different axis scales.
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
            for xz in data["release_points_xz_m"].values():
                ax.plot(*np.asarray(xz).T, "o", color="0.1", ms=4)
            ax.autoscale_view()
            ax.set(title=title, xlabel="East (m)", ylabel="Height (m)")
            time = data["time_s"]
            if key == "row":
                for cell, style in (("A1", "-"), ("B1", "--")):
                    particle_id = f"{cell}-0-3-0-0"
                    path = data["tracked_paths"][particle_id]
                    x = path["wrapped_x_m"]
                    t_path, x_path = [0.0], [x[0]]
                    for wrap in path["wrap_times_s"]:
                        t_path.extend([wrap, np.nan, wrap])
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
# ## At the same radius, which material rises?
#
# Compare the mean final *world* height change for each radius and recipe in
# the translating block. Each colored square averages 64 individual particles
# from the same east–height strata for that size and recipe. Each row
# isolates the role of density at fixed radius. Then compare how much of the initial number and mass ends
# up in each part of the fixed stack. Each mass bar includes that species in
# both the pure particles and the ternary mixture.


# %%
def plot_composition(report):
    """Compare signed lofting by size and recipe and stack export by component."""
    block, stack = report["block"], report["stack"]
    with mplstyle.context("particula.default"):
        fig, (height, budget) = plt.subplots(
            1, 2, figsize=(11, 4), layout="constrained"
        )
        dz = (
            100
            * np.asarray(block["final_displacement_by_radius_recipe_m"])[
                :, :, 1
            ]
        )
        limit = np.max(np.abs(dz))
        image = height.imshow(
            dz, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
        )
        height.set(
            xticks=range(len(RECIPES)),
            xticklabels=[
                "Ammonium\nsulfate",
                "Sodium\nchloride",
                "Sucrose",
                "Ternary\nmixture",
            ],
            yticks=range(len(RADII_UM)),
            yticklabels=[f"{r:g}" for r in block["radii_um"]],
            ylabel="Particle radius (µm)",
            title="Height change in moving block",
        )
        height.grid(False)
        for (i, j), val in np.ndenumerate(dz):
            height.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                color="white" if abs(val) > limit * 0.6 else "black",
            )
        fig.colorbar(image, ax=height, label="World height change (cm)")
        history = np.asarray(stack["budget_history"])
        fraction = history[-1] / history[0].sum(axis=0)
        bottom = np.zeros(4)
        for share, name, color in zip(
            fraction,
            ("A", "B", "Outflow"),
            ("#72ABCC", "#ECAF60", "#7B718E"),
            strict=True,
        ):
            budget.bar(
                range(4), 100 * share, bottom=bottom, color=color, label=name
            )
            bottom += 100 * share
        budget.set(
            xticks=range(4),
            xticklabels=[
                "Particle\nnumber",
                "Ammonium\nsulfate",
                "Sodium\nchloride",
                "Sucrose",
            ],
            ylabel="Share of initial inventory (%)",
            title="Final inventory in fixed stack",
            ylim=(0, 106),
        )
        budget.legend(loc="lower center", ncols=3, fontsize=8, frameon=True)
        return fig


# %% [markdown]
# The final catalog links each release ID to its ending cell or open exit,
# residence time, height change and number of wraps. Use the same row across
# layouts to follow one individual. Here two 10 µm-radius particles released
# from identical east/height coordinates in B1 illustrate the range of outcomes.


# %%
def run_example():
    """Summarize physical fate and show the recipe dependence at fixed radius."""
    report = simulate()
    for name, data in report.items():
        print(
            f"{name}: {data['particle_count']} physical particles, "
            f"{data['group_crossings']} group passages, {data['open_exits']} top exits."
        )
        mean = data["means"]["all"]
        print(
            f"  Mean travel east {mean['dx_world_m'][-1]:.3f} m; "
            f"world and mesh height changes {100 * mean['dz_world_m'][-1]:+.2f} and {100 * mean['dz_mesh_m'][-1]:+.2f} cm."
        )
    row = report["row"]
    mass = np.asarray(row["gross_transfers"]["A to B"])[1:] * 1e9
    print(
        f"Periodic A to B cumulative mass (µg), sulfate salt / NaCl / sucrose: {mass.round(3)}"
    )
    block = report["block"]
    for recipe in (1, 2):
        particle_id = f"B1-3-{recipe}-0-0"
        i = block["release_catalog"]["id"].index(particle_id)
        final = block["final"]
        print(
            f"{particle_id} ({RECIPES[recipe]}) ends in {final['cell'][i]} with world height change {100 * final['height_change_m'][i]:+.2f} cm."
        )
    plot_results(report)
    plot_composition(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# At 14 s the block has moved 9.8 m east and risen 30.8 cm. Its particles
# rise 19.34 cm on average while falling 11.46 cm relative to the cells. The
# sign depends on composition at the upper end of the size range. At 10 µm
# radius pure sucrose rises 3.88 cm in the world frame, while denser sodium
# chloride falls 5.92 cm. A total of 256 particles move from B down into A.
#
# In the periodic row all 1,024 particles remain in the domain after four or
# five wraps. The cumulative sodium chloride mass passing from A to B is
# 4.752 µg, matched by 4.752 µg passing from B back to A. In the fixed stack
# 704 particles exit through the top, carrying 15.3% of the initial ammonium sulfate mass, 13.7% of sodium chloride and
# 23.3% of sucrose. Many small particles can leave while the more massive
# large particles stay. These fractions depend on the specified size and
# recipe abundances and release heights. A different particle population
# would need new count and component-mass budgets.
