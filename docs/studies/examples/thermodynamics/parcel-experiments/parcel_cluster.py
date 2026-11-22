# %% [markdown]
# # What crosses a shared parcel face?
#
# A gas follows the prescribed air exchange between two parcels. A droplet can
# move in the opposite direction if its settling speed exceeds the updraft.
# Use two experiments to track these mechanisms separately. First move an
# organic vapor across a face and ask how receiver particulate mass responds
# to the flow rate. Then release two particles of different sizes above a
# horizontal face and predict which crosses first.
#
# These are prescribed-flow experiments. The parcel histories come from
# Particula, and the settling column has uniform air velocity. The smooth
# sensitivity and the particle crossing time each have an independent check.
# The full example requires an NVIDIA GPU for the reverse-mode calculation.

# %%
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.dynamics.parcel import RD, RV, ParcelInitialState, run_parcel
from particula.dynamics.parcel_cluster import (
    Connection,
    ParcelCluster,
    run_cluster,
)
from particula.dynamics.parcel_motion import (
    MotionState,
    Particle,
    advance_motion,
)
from particula.dynamics.parcel_replay import OrganicForcing, ParcelTrace
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)

FLOW = 0.02  # m³/s
GAS_MASS = 1e-8  # kg of test organic vapor initially in the donor
DURATION = 4.0  # s
POSITION_ATOL = 1e-12  # m, roundoff in a meter-scale constant-speed path
TIME_ATOL = 1e-9  # s, position roundoff divided by a millimeter/s velocity
MASS_ATOL = 1e-20  # kg, one part in 10¹² of the initial gas inventory


# %% [markdown]
# ## Give each parcel its own forcing record
#
# The donor contains one kilogram of dry air, and the receiver contains two.
# At equal pressure and temperature their unequal volumes make concentration
# an unsuitable quantity to sum across the face. Generate separate histories
# using different updrafts and recording cadences so matching the same record
# number would also give the wrong exchange time. Track inventories in kg
# and compare all parcel diagnostics at common physical times in seconds.


# %%
def make_traces(*, stationary=False):
    """Record two short Particula trajectories on distinct time meshes."""
    traces = {}
    for name, dry_mass, speed, spacing in (
        ("donor", 1.0, 0.5, 0.25),
        ("receiver", 2.0, 1.5, 0.5),
    ):
        initial = ParcelInitialState(
            dry_radius=np.array([70e-9]),
            number=np.array([5e7]) * dry_mass,
            kappa=0.5,
            pressure=90000.0,
            temperature=288.15,
            vapor=0.0118,
            dry_air_mass=dry_mass,
        )
        trajectory = run_parcel(
            initial,
            updraft=0.0 if stationary else speed,
            timestep=spacing,
            steps=round(DURATION / spacing),
        )
        traces[name] = ParcelTrace.record(
            trajectory, parcel_id=name, provenance="Particula CPU parcel ascent"
        )
    return traces


# %% [markdown]
# ## Diagnose organic mass and its sensitivity to flow
#
# Each transfer uses the donor's current gas concentration. An irreversible
# first-order uptake of the dilute organic vapor represents transfer to the
# particle phase. Follow gas and particulate mass together to close each
# parcel's budget. Terminal receiver particle mass measures how much of
# the transported vapor has condensed by the observation time.
#
# Perturb the flow on both sides of its specified value and repeat the whole
# experiment. Centered differences check the CUDA reverse pass on the same
# time mesh. Halving the exchange step in a CPU calculation tests sensitivity
# to the duration over which donor concentration is held fixed.


# %%
def exchange(traces, flow, step, device, *, observation=None, uptake=0.25):
    """Carry test vapor between different parcel volumes and into particles."""
    configured = {
        name: replace(
            trace,
            organic=OrganicForcing(
                name="SOA",
                initial_gas=GAS_MASS if name == "donor" else 0.0,
                uptake_rate=uptake if name == "receiver" else 0.0,
            ),
        )
        for name, trace in traces.items()
    }
    cluster = ParcelCluster.custom(
        configured,
        centers={"donor": (0.5, 0.5, 0.5), "receiver": (0.5, 0.5, 1.5)},
        connections=[Connection("donor", "receiver", flow)],
    )
    return run_cluster(
        cluster,
        max_step=step,
        device=device,
        gradient=("donor", "receiver", "receiver", observation)
        if observation is not None
        else None,
    )


def centered_flow_difference(traces, step, device, delta):
    """Return kg particulate organic per (m³/s) for fixed elapsed time."""
    values = [
        exchange(traces, FLOW + sign * delta, step, device)
        .parcels["receiver"]
        .organic_particle[-1]
        for sign in (-1, 1)
    ]
    return float((values[1] - values[0]) / (2 * delta))


def exchange_report():
    """Check signed budgets, numerical sensitivity and step refinement."""
    traces = make_traces()
    step = 1 / 32
    run = exchange(traces, FLOW, step, "cuda:0", observation="particle")
    donor, receiver = (run.parcels[name] for name in ("donor", "receiver"))
    donor_total = donor.organic_gas + donor.organic_particle
    receiver_total = receiver.organic_gas + receiver.organic_particle
    face = run.transfers["donor", "receiver"]
    donor_residual = donor_total - GAS_MASS + face
    receiver_residual = receiver_total - face
    np.testing.assert_allclose(donor_residual, 0, atol=MASS_ATOL)
    np.testing.assert_allclose(receiver_residual, 0, atol=MASS_ATOL)
    perturbations = np.array([1e-2, 1e-3, 1e-4]) * FLOW
    centered = [
        centered_flow_difference(traces, step, "cuda:0", delta)
        for delta in perturbations
    ]
    gradient = float(run.flow_gradient)
    # The middle and smallest perturbations lie in the O(delta²) regime.
    if gradient <= 0:
        raise AssertionError("Receiver uptake should increase with inflow")
    np.testing.assert_allclose(centered[1:], gradient, rtol=1e-5, atol=0)
    cpu_steps = [step, step / 2, step / 4]
    cpu_fd = [
        centered_flow_difference(traces, size, "cpu", perturbations[1])
        for size in cpu_steps
    ]
    np.testing.assert_allclose(cpu_fd[0], gradient, rtol=1e-5, atol=0)
    # First-order exchange with donor fraction <= 0.001 per step permits a
    # percent-level step bias. Retain all values so the reader can inspect it.
    np.testing.assert_allclose(cpu_fd[-1], gradient, rtol=0.02, atol=0)
    # With a stationary donor each update leaves a fraction 1 − QΔt/V.
    # Differentiate its n-step product analytically for a gas-only control.
    passive_step = 0.25
    n_steps = round(DURATION / passive_step)
    volume = (RD + 0.0118 * RV) * 288.15 / 90000.0
    fraction = FLOW * passive_step / volume
    expected = (
        GAS_MASS
        * n_steps
        * passive_step
        / volume
        * (1 - fraction) ** (n_steps - 1)
    )
    passive = exchange(
        make_traces(stationary=True),
        FLOW,
        passive_step,
        "cuda:0",
        uptake=0,
        observation="gas",
    )
    np.testing.assert_allclose(
        passive.flow_gradient, expected, rtol=1e-10, atol=0
    )
    return {
        "passive_gas_gradient_kg_s_m3": float(passive.flow_gradient),
        "passive_analytical_gradient_kg_s_m3": float(expected),
        "device": wp.get_device("cuda:0").name,
        "precision": "float64",
        "time_s": run.time.tolist(),
        "donor_total_kg": donor_total.tolist(),
        "receiver_total_kg": receiver_total.tolist(),
        "receiver_particle_kg": receiver.organic_particle.tolist(),
        "cumulative_face_kg": face.tolist(),
        "max_donor_budget_error_kg": float(np.max(np.abs(donor_residual))),
        "max_receiver_budget_error_kg": float(
            np.max(np.abs(receiver_residual))
        ),
        "flow_m3_s": FLOW,
        "flow_gradient_kg_s_m3": gradient,
        "flow_deltas_m3_s": perturbations.tolist(),
        "cuda_fd_kg_s_m3": centered,
        "cpu_steps_s": cpu_steps,
        "cpu_fd_kg_s_m3": cpu_fd,
    }


# %% [markdown]
# ## Release two particles above a horizontal face
#
# In a separate fixed column, place both particles one centimeter above the
# interface at $z=1$ m. Use radii of 1 and 10 µm, material density of
# 1000 kg/m³, and room-temperature air. Compute the terminal speeds with
# the [Stokes–Cunningham settling law](https://github.com/uncscode/particula/blob/f25f230e6e/particula/particles/properties/settling_velocity.py#L39-L60)
# and set the updraft halfway between the speeds. The smaller particle rises
# and the larger one sinks. With constant properties the exact crossing time
# is the initial distance to the face divided by the large particle's net
# downward speed. Check the event time and each particle's altitude on both
# CPU and CUDA. Both paths use the same terminal speeds, so this comparison
# tests the event timing and particle motion conditional on that drag law.


# %%
def settling_report():
    """Compare face crossing and signed vertical displacement on CPU and CUDA."""
    temperature, pressure = 293.15, 101325.0  # K, Pa
    radii = np.array([1.0, 10.0]) * 1e-6
    density = 1000.0
    speeds = get_particle_settling_velocity_via_system_state(
        radii, density, temperature, pressure
    )
    updraft = float(np.mean(speeds))
    relative = updraft - speeds
    face_height, start_height = 1.0, 1.01
    expected_time = float((face_height - start_height) / relative[1])
    # Sample before and after the crossing. A uniform schedule avoids giving
    # the event solver the analytically predicted event as a time boundary.
    times = np.linspace(0, 2.07 * expected_time, 17)
    expected_z = start_height + times[:, None] * relative
    cluster = ParcelCluster.regular(
        make_traces(), shape=(1, 1, 2), widths=(1.0, 1.0, 1.0)
    )
    particles = tuple(
        Particle(
            id=name,
            cell="receiver",
            position=(0.5, 0.5, start_height),
            radius=float(radius),
            density=density,
            count=count,
            masses=(float(4 * np.pi * radius**3 * density / 3),),
            charge=charge,
        )
        for name, radius, count, charge in zip(
            ("small", "large"), radii, (3.0, 2.0), (1.0, -1.0), strict=True
        )
    )
    report = {}
    for device in ("cpu", "cuda:0"):
        state = MotionState(time=0.0, particles=particles)
        heights = [[start_height, start_height]]
        for dt in np.diff(times):
            state = advance_motion(
                cluster,
                state,
                duration=float(dt),
                air_velocity=(0.0, 0.0, updraft),
                device=device,
                capacity={"donor": 2, "receiver": 2},
                temperature=temperature,
                pressure=pressure,
            )
            by_id = {p.id: p for p in state.particles}
            if set(by_id) != {"large", "small"} or len(state.particles) != 2:
                raise AssertionError(
                    "Both represented particle IDs must survive"
                )
            heights.append([by_id[p.id].position[2] for p in particles])
        for p in particles:
            final = by_id[p.id]
            if (final.count, final.masses, final.charge) != (
                p.count,
                p.masses,
                p.charge,
            ):
                raise AssertionError("Crossing changed the particle inventory")
        internal = [
            event
            for event in state.events
            if event.source == "receiver" and event.destination == "donor"
        ]
        if len(internal) != 1 or internal[0].particle != "large":
            raise AssertionError("Exactly the large particle must cross once")
        actual_time = float(internal[0].time)
        np.testing.assert_allclose(
            actual_time, expected_time, atol=TIME_ATOL, rtol=0
        )
        np.testing.assert_allclose(
            heights, expected_z, atol=POSITION_ATOL, rtol=0
        )
        report[device] = {
            "height_m": heights,
            "crossing_time_s": actual_time,
            "crossing_time_error_s": actual_time - expected_time,
            "max_position_error_m": float(np.max(np.abs(heights - expected_z))),
        }
    return {
        "time_s": times.tolist(),
        "reference_height_m": expected_z.tolist(),
        "reference_crossing_s": expected_time,
        "air_velocity_m_s": updraft,
        "settling_speeds_m_s": speeds.tolist(),
        "radii_um": (radii * 1e6).tolist(),
        "face_height_m": face_height,
        "devices": report,
    }


# %% [markdown]
# ## Read flux, response and crossing against the controls
#
# Follow the donor loss and receiver gain beside the cumulative face transfer.
# Only part of the transferred mass has condensed by the end. Compare the
# gradient with centered differences as the perturbation decreases. In the
# settling panel the crosses mark numerical positions and the lines give the
# constant-velocity solution. The vertical marker gives the predicted time at
# which the larger particle enters the lower cell.


# %%
def simulate():
    """Run the two independent cluster experiments."""
    return {"exchange": exchange_report(), "settling": settling_report()}


@mplstyle.context("particula.default")
def plot_results(report):
    """Display extensive budgets, flow sensitivity and differential motion."""
    exchange_data, motion = report["exchange"], report["settling"]
    fig, axes = plotting.subplots(2, 2)
    budgets, moved, sensitivity, settling = axes.flat
    time = exchange_data["time_s"]
    for field, label, style in (
        ("donor_total_kg", "Donor", "-"),
        ("receiver_total_kg", "Receiver", "--"),
    ):
        budgets.plot(
            time, 1e9 * np.array(exchange_data[field]), style, label=label
        )
    budgets.set(xlabel="Time (s)", ylabel="Total organic mass (µg)")
    plotting.legend(budgets)
    for field, label, style in (
        ("cumulative_face_kg", "Through face", "-"),
        ("receiver_particle_kg", "Condensed", "--"),
    ):
        moved.plot(
            time, 1e9 * np.array(exchange_data[field]), style, label=label
        )
    moved.set(xlabel="Time (s)", ylabel="Organic mass (µg)")
    plotting.legend(moved)
    relative_error = np.abs(
        np.asarray(exchange_data["cuda_fd_kg_s_m3"])
        / exchange_data["flow_gradient_kg_s_m3"]
        - 1
    )
    sensitivity.loglog(exchange_data["flow_deltas_m3_s"], relative_error, "o-")
    sensitivity.set(
        xlabel="Flow perturbation (m³/s)", ylabel="Relative gradient error"
    )
    for j, radius in enumerate(motion["radii_um"]):
        (line,) = settling.plot(
            motion["time_s"],
            np.array(motion["reference_height_m"])[:, j],
            label=f"{radius:g} µm radius",
        )
        settling.plot(
            motion["time_s"],
            np.array(motion["devices"]["cuda:0"]["height_m"])[:, j],
            "x",
            color=line.get_color(),
        )
    settling.axhline(motion["face_height_m"], linestyle=":", color="0.35")
    settling.axvline(
        motion["reference_crossing_s"], linestyle="--", color="0.35"
    )
    settling.set(xlabel="Time (s)", ylabel="Particle height (m)")
    plotting.legend(settling)
    plotting.fit(fig)
    return fig


# %% [markdown]
# Print the signed sensitivity, parcel budget residuals, and crossing-time
# errors. These quantities have different units and answer separate questions
# about mass accounting, differentiability and displacement.


# %%
def run_example():
    """Print the numerical checks and display the two experiments."""
    report = simulate()
    exchange_data, settling = report["exchange"], report["settling"]
    for name in ("donor", "receiver"):
        error = exchange_data[f"max_{name}_budget_error_kg"]
        print(
            f"{name.capitalize()} maximum organic budget error = {error:.3g} kg"
        )
    print(
        f"Receiver particle-mass sensitivity = {exchange_data['flow_gradient_kg_s_m3']:+.6g} kg s/m³"
    )
    for device, values in settling["devices"].items():
        print(
            f"{device} crossing = {values['crossing_time_s']:.8g} s, error = {values['crossing_time_error_s']:+.3g} s"
        )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# After 4 s the receiver has gained 0.819 µg of organic material, of which
# 0.307 µg has condensed. The local derivative predicts that increasing the
# flow by 0.0002 m³/s adds about 0.0030 µg to the particles at 4 s. Reducing
# the CPU step from 1/32 to 1/128 s changes the sensitivity by 0.53%. This
# timestep effect exceeds the centered-difference error at the original step.
#
# The two particles start 1 cm above the face. In the 0.00612 m/s updraft,
# the particle of 10 µm radius reaches the lower cell after 1.669 s. Its
# companion of 1 µm radius
# has risen 1 cm by the same time. The fixed geometry and constant settling
# speeds make crossing time predictable. To examine a translating column,
# prescribe its face velocity and compare each particle's motion relative to
# that face.
