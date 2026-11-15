# %% [markdown]
# # Extend a process on CPU and CUDA
#
# This Particula 0.3 development example implements the same prescribed warming
# law in NumPy and Warp: dT/dt = a t, with a in K s⁻² and absolute time t in s.
# A registered box field records cumulative warming in K. This is an imposed
# boundary condition, not an energy-conserving atmospheric heating model.
# Native dilution is composed after the custom operation.
#
# We compare pure CPU, resident CUDA, captured resident CUDA and explicit
# `CPUTransfer` execution in two boxes. We check the analytical trajectory,
# detached snapshots and exact checkpoint continuation. CUDA is required;
# there is no CPU fallback and this small example makes no speedup claim.
# Download the ZIP to include the Warp kernel helper. Install Particula,
# NumPy, Matplotlib and Warp, then run `python example.py` from the extracted
# directory. All extension code uses public interfaces.

# %%
"""Equivalent public NumPy/Warp extensions with explicit transfer and restart."""

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from particula import (
    Aerosol,
    BoxProcess,
    CPUTransfer,
    FieldSpec,
    GPUProcessPlan,
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.dynamics import Dilution, DilutionStrategy
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scripts.gpu_studies.extension_kernels import prescribed_warming


class PrescribedWarming(BoxProcess):
    """Integrate a time-dependent external temperature forcing."""

    def __init__(self, rate=0.02):
        """Declare configuration and permitted native/registered destinations."""
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        self.rate = float(rate)
        super().__init__(
            name="warming",
            extension_id="example.prescribed_warming",
            reads=("environment.temperature", "demo.warming"),
            writes=("environment.temperature", "demo.warming"),
            parameters=(self.rate,),
        )

    def validate(self, aerosol):
        """Require a per-box registered warming field with Kelvin units."""
        spec = aerosol.fields.specs.get("demo.warming")
        if spec is None or spec.axes != ("box",) or spec.units != "K":
            raise ValueError("demo.warming must be a box field in K")

    def advance(self, aerosol, *, time, time_step, context):
        """Advance ordinary NumPy arrays in place over the exact interval."""
        increment = self.rate * (time * time_step + 0.5 * time_step**2)
        aerosol.environment.temperature[:] += increment
        aerosol.fields["demo.warming"][:] += increment


class WarmingAdapter:
    """Explicit resident implementation; registration matches the exact class."""

    def plan(self, process, aerosol):
        """Validate detached host state and freeze immutable configuration."""
        process.validate(aerosol)
        return GPUProcessPlan(
            name=process.name,
            extension_id=process.extension_id,
            version=process.version,
            reads=process.reads,
            writes=process.writes,
            parameters=(("rate", process.rate),),
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
        )

    def prepare(self, plan, context):
        """Retain fixed buffers once; later calls only enqueue the kernel."""
        temperature = context.arrays["environment.temperature"]
        args = (
            temperature,
            context.arrays["demo.warming"],
            context.clock,
            context.failure,
            wp.float64(plan.values["rate"]),
            wp.float64(context.time_step),
        )

        def enqueue():
            wp.launch(
                prescribed_warming,
                dim=len(context.logical_box_ids),
                inputs=args,
                device=context.device,
            )

        return PreparedOperation(plan.name, enqueue, args, plan.capabilities)


# %% [markdown]
# ## Choose execution explicitly
#
# `ProcessRegistry` connects the exact CPU process class to its GPU adapter.
# A resident kernel checks the shared failure flag, writes only declared
# buffers, and reads interval time from the device clock. It must not allocate,
# download or synchronize during a timestep. Python launch closures are trusted
# implementations, not a security sandbox.
#
# `CPUTransfer(process)` opts into a host round trip instead. Particula downloads
# state, runs the ordinary transactional CPU callback and uploads permitted
# writes into existing device buffers. This accepts arbitrary Python libraries
# but cannot be captured. It can still checkpoint and resume. A snapshot is
# detached analysis data; a checkpoint preserves exact continuation state.


# %%
def build_aerosol():
    """Construct two independent boxes and an owned warming field."""
    state = Aerosol(
        particles=ParticleData(
            masses=np.full((2, 4, 1), 1e-18),
            concentration=np.ones((2, 4)),
            charge=np.zeros((2, 4)),
            density=np.array([1000.0]),
            volume=np.array([1e-6, 2e-6]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            name=["inert"],
            molar_mass=np.array([0.029]),
            concentration=np.full((2, 1), 1e-6),
            partitioning=np.array([False]),
        ),
        environment=EnvironmentData(
            temperature=np.array([280.0, 300.0]),
            pressure=np.full(2, 101325.0),
            saturation_ratio=np.zeros((2, 1)),
        ),
    )
    state.fields.add(FieldSpec("demo.warming", ("box",), "K", minimum=0))
    return state


def compare_checkpoint(left, right):
    """Compare canonical physical/field bytes and absolute time continuation."""
    assert left.resident.payloads == right.resident.payloads
    assert left.field_payloads == right.field_payloads
    assert left.extension_state == right.extension_state
    assert left.host_rng == right.host_rng
    assert left.clock == right.clock
    assert left.resident.completed_steps == right.resident.completed_steps
    assert left.resident.simulated_time == right.resident.simulated_time


def gpu_trajectory(mode, times):
    """Run one explicit GPU mode and verify restart from its midpoint."""
    initial = build_aerosol()
    registry = ProcessRegistry()
    registry.register(PrescribedWarming, WarmingAdapter())
    warming = PrescribedWarming()
    custom = CPUTransfer(warming) if mode == "transfer" else warming
    processes = custom | Dilution(DilutionStrategy(0.01))
    history = [initial.environment.temperature.copy()]
    with processes.prepare(
        initial,
        device="cuda:0",
        time_step=0.5,
        time=float(times[0]),
        process_registry=registry,
    ) as sim:
        before = sim.checkpoint()
        if mode == "captured":
            sim.capture()
            compare_checkpoint(sim.checkpoint(), before)
        elif mode == "transfer":
            try:
                sim.capture()
            except ValueError:
                pass
            else:
                raise AssertionError("CPUTransfer unexpectedly captured")
            compare_checkpoint(sim.checkpoint(), before)
        for step in range(1, len(times)):
            sim.step()
            snapshot = sim.snapshot()
            history.append(snapshot.environment.temperature.copy())
            assert sim.current_time == times[step]
            if step == 20:
                saved = sim.checkpoint()
                # Changing detached analysis data cannot alter device state.
                snapshot.environment.temperature[:] = -1
                snapshot.fields["demo.warming"][:] = -1
                compare_checkpoint(sim.checkpoint(), saved)
        final = sim.checkpoint()
        final_state = sim.snapshot()
    with processes.resume(saved, process_registry=registry) as resumed:
        if mode == "captured":
            resumed.capture()
        resumed.run(20)
        compare_checkpoint(resumed.checkpoint(), final)
    np.testing.assert_allclose(
        final_state.fields["demo.warming"],
        0.01 * (times[-1] ** 2 - times[0] ** 2),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        final_state.gas.concentration,
        initial.gas.concentration * np.exp(-0.01 * 20),
        rtol=1e-13,
    )
    np.testing.assert_array_equal(initial.fields["demo.warming"], 0)
    return np.asarray(history)


def simulate():
    """Compare all four execution paths with an independent exact solution."""
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires real CUDA")
    times = np.arange(41) * 0.5 + 5.0
    initial = build_aerosol()
    cpu = build_aerosol()
    processes = PrescribedWarming() | Dilution(DilutionStrategy(0.01))
    history = [cpu.environment.temperature.copy()]
    for time in times[:-1]:
        processes.execute(cpu, 0.5, time=float(time))
        history.append(cpu.environment.temperature.copy())
    trajectories = {"cpu": np.asarray(history)}
    for mode in ("resident", "captured", "transfer"):
        trajectories[mode] = gpu_trajectory(mode, times)
    expected = initial.environment.temperature + 0.01 * (
        times[:, None] ** 2 - times[0] ** 2
    )
    for values in trajectories.values():
        np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-12)
    np.testing.assert_array_equal(
        trajectories["captured"], trajectories["resident"]
    )
    return dict(
        time=times.tolist(),
        reference=expected.tolist(),
        trajectories={
            key: values.tolist() for key, values in trajectories.items()
        },
        device=wp.get_device("cuda:0").name,
    )


# %%
def plot_results(report):
    """Compare trajectories and numerical residuals from compact saved data."""
    reference = np.asarray(report["reference"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {
        "cpu": "black",
        "resident": "tab:blue",
        "captured": "tab:green",
        "transfer": "tab:orange",
    }
    for mode, history in report["trajectories"].items():
        values = np.asarray(history)
        axes[1].plot(
            report["time"],
            np.max(np.abs(values - reference), axis=1),
            label=mode,
            color=colors[mode],
        )
    resident = np.asarray(report["trajectories"]["resident"])
    for box in range(2):
        axes[0].plot(report["time"], resident[:, box], label=f"Box {box + 1}")
        axes[0].plot(report["time"][::4], reference[::4, box], "k.")
    axes[0].set(
        ylabel="Temperature (K)", title="Resident CUDA · dots: exact solution"
    )
    axes[1].set(
        ylabel="Maximum absolute error (K)", title="All four execution paths"
    )
    for ax in axes:
        ax.set_xlabel("Absolute simulation time (s)")
        ax.legend()
        ax.grid(alpha=0.2)
    plt.show()


def run_example():
    """Execute the complete CUDA comparison and render its verified outputs."""
    report = simulate()
    print("CUDA device:", report["device"])
    print("NumPy/Warp parity, detached snapshots, capture and restart passed.")
    plot_results(report)
    return report


if __name__ == "__main__":
    run_example()
