# %% [markdown]
# # What does the second gust lift from a partly depleted dune?
#
# Two identical triangular gusts pass over a dry patch near a dune crest.
# Each gust exceeds both prescribed pickup thresholds. The first gust can
# exhaust the fine dust, so a second gust with the same wind need not produce
# the same airborne size distribution. Track the erodible bed separately from
# grains that settle back onto it.
#
# The example runs on CPU or CUDA and uses synthetic SI soundings, two dry classes and
# a finite patch of 1 m² slope area. The coefficients and settling speeds are
# experiment inputs. The [surface-emission equations](../../../theory/dynamics/surface-emission.md)
# describe how the model converts friction velocity to represented number
# and dry mass.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import plotting
from particula.coupling import FlowFactory, FlowGrid
from particula.coupling.flow_replay import GridReplay
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun
from particula.coupling.prescribed import Sounding
from particula.coupling.sounding_sequence import SoundingSequence
from particula.coupling.surface_emission import (
    BedPatch,
    DustClass,
    SurfaceEmission,
)

# %% [markdown]
# ## Two gusts over a finite bed
#
# Each gust ramps from 0 to 8 m/s in 4 s and decays over the next 4 s.
# A 4 m-high Gaussian ridge lies in a 20 m-deep layer. The 2-D vertical slice
# represents 1 m of crosswind width. A patch immediately downwind of the
# crest supplies 50 mg of 4 µm grains and 1000 mg of 40 µm grains.
# After the second gust, 2 s of calm lets the last coarse grains settle.
# The source samples each accepted interval and each wind knot at 1 m above
# the local bed. Integrate emission across each wind segment and inject its
# mass at the start of each 0.25 s transport interval. A smaller interval
# refines release timing relative to settling. Compare the peak friction
# velocity with the two thresholds.


# %%
def simulate(device="cpu"):
    """Compute cumulative fate and gust-integrated emission for both classes."""
    times = (0.0, 4.0, 8.0, 12.0, 16.0, 18.0)
    winds = (0.0, 8.0, 0.0, 8.0, 0.0, 0.0)
    profiles = tuple(
        Sounding(
            [0.0, 20.0],
            [290.0, 290.0],
            [100000.0, 99765.0],
            [0.007, 0.007],
            [u, u],
            [0.0, 0.0],
        )
        for u in winds
    )
    seq = SoundingSequence(
        times,
        profiles,
        "experiment start",
        "synthetic profiles, z in m above flat datum",
    )
    flow = FlowFactory().get_strategy(
        "dune", dict(sounding=seq, height=4.0, width=8.0, lid=20.0)
    )
    x = np.linspace(-24.0, 24.0, 25)
    bed = 4.0 * np.exp(-0.5 * (x / 8.0) ** 2)
    grid = FlowGrid(
        (24, 1, 4),
        ((-24.0, 24.0), (0.0, 1.0), (0.0, 20.0)),
        ("open", "open", "closed", "closed", "deposit", "closed"),
        tuple(bed),
    )
    forcing = GridReplay.record(
        grid,
        flow,
        np.arange(0.0, 18.01, 0.25),
        provenance=seq.provenance,
        device=device,
    )
    classes = (
        DustClass("fine", 0.20, 1.0e-5, 4.0e-6, 2500.0, (1.0,), 0.0012),
        DustClass("coarse", 0.35, 1.0e-5, 40.0e-6, 2500.0, (1.0,), 0.12),
    )
    patch = BedPatch(
        "crest",
        int(np.ravel_multi_index((12, 0, 0), grid.shape)),
        1.0,
        (50.0e-6, 1.0e-3),
    )
    source = SurfaceEmission(
        forcing,
        flow,
        (patch,),
        classes,
        ("mineral",),
        reference_height=1.0,
        ejection_height=0.15,
        drag_coefficient=0.01,
        representation="resolved",
        device=device,
    )
    layout = InventoryLayout(("passive gas",))
    run = GridRun(
        forcing,
        layout,
        np.zeros((len(grid.volume), layout.width)),
        surface=source,
        device=device,
    )
    fate = []
    interval, patch_index, size_class = source.records.T
    kg = source.emitted[interval, patch_index, size_class]
    for i in range(len(forcing.times)):
        if i:
            run.advance()
        state = run.snapshot()["status"]
        ledgers = [source.remaining[i, 0]]
        for code in (1, 2, 3):
            ledgers.append(
                np.bincount(
                    size_class,
                    weights=kg * (state == code),
                    minlength=len(classes),
                )
            )
        fate.append(np.stack(ledgers))
    fate = np.stack(fate)
    np.testing.assert_allclose(
        fate.sum(axis=1),
        np.broadcast_to(patch.available, (len(fate), 2)),
        rtol=5.0e-14,
    )
    second = forcing.times[:-1] >= 8.0
    gust = np.stack(
        [source.emitted[mask, 0, :].sum(axis=0) for mask in (~second, second)]
    )
    return dict(
        time_s=forcing.times.tolist(),
        wind_time_s=source.wind_times.tolist(),
        friction_m_s=source.friction_velocity[:, 0].tolist(),
        gust_kg=gust.tolist(),
        fate_kg=fate.tolist(),
        threshold_m_s=[d.threshold for d in classes],
        diameter_um=[1.0e6 * d.diameter for d in classes],
        device=device,
        precision="float64",
        dt_s=0.25,
        provenance=seq.provenance,
        wind_knot_s=list(times),
        upstream_east_m_s=list(winds),
    )


# %% [markdown]
# ## Read the emitted sizes and their fates
#
# The bars give total kg emitted in each size class per gust. The two lower
# panels follow those same classes from bed into air, deposition and escape
# through an open boundary. Fine and coarse panels have independent mass
# scales. Summing their four curves at each time recovers the respective
# initial class inventory.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Draw friction velocity, emitted size distribution and per-class fates."""
    fig, axes = plotting.subplots(2, 2, size="wide")
    wind, size, fine, coarse = axes.flat
    wind.plot(report["wind_time_s"], report["friction_m_s"], label="Crest")
    for t, d, style in zip(
        report["threshold_m_s"], report["diameter_um"], ("--", ":"), strict=True
    ):
        wind.axhline(t, linestyle=style, label=f"{d:.0f} µm onset")
    wind.set(xlabel="Time (s)", ylabel="Friction velocity (m/s)")
    wind.set_ylim(-0.05, 1.2 * max(report["friction_m_s"]))
    plotting.legend(wind, placement="inside", loc="upper center", ncols=3)
    kg = np.asarray(report["gust_kg"])
    for i in range(2):
        size.bar(
            np.arange(2) + 0.36 * (i - 0.5),
            kg[i],
            width=0.36,
            label=f"Gust {i + 1}",
        )
    size.set_xticks([0, 1], [f"{d:.0f} µm" for d in report["diameter_um"]])
    size.set(xlabel="Emitted diameter", ylabel="Emitted mass (kg)")
    plotting.legend(size)
    fate = np.asarray(report["fate_kg"])
    for j, ax in enumerate((fine, coarse)):
        for i, (name, style) in enumerate(
            zip(
                ("Bed", "Airborne", "Deposited", "Exported"),
                ("-", "--", "-.", ":"),
                strict=True,
            )
        ):
            ax.plot(
                report["time_s"], fate[:, i, j], linestyle=style, label=name
            )
        ax.set(
            xlabel="Time (s)",
            ylabel=f"{report['diameter_um'][j]:g} µm class mass (kg)",
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plotting.legend(ax)
    plotting.fit(fig)
    return fig


# %% [markdown]
# Print the two gust totals in milligrams for easier comparison across classes.
# The saved figure data retain kilograms throughout.


# %%
def run_example(device="cpu"):
    """Report full-resolution dry mass totals for the two gusts."""
    report = simulate(device)
    for i, (fine, coarse) in enumerate(np.asarray(report["gust_kg"]) * 1e6, 1):
        print(
            f"Gust {i} emitted {fine:.2f} mg of 4 µm grains and {coarse:.2f} mg of 40 µm grains."
        )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", action="store_true", help="run on the NVIDIA GPU"
    )
    device = "cuda:0" if parser.parse_args().cuda else "cpu"
    run_example(device)

# %% [markdown]
# ## What changes between gusts?
#
# The wind exceeds the fine threshold twice, but the second gust encounters
# an empty fine reservoir. The first gust lifts all 50 mg of fine grains.
# Each gust lifts 154.76 mg of coarse grains, which settle into the deposit
# reservoir. Increasing the fine supply alone changes the second
# gust's size distribution. Changing the settling speed instead shifts the
# balance between airborne, deposited and exported mass after emission.
# A measured application needs independent size-resolved vertical flux and
# bed-supply information to constrain those two effects separately.
